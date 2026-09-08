"""主窗口: 数据源 (孪生 / UDP / 回放), 蜂窝热图 + 观测视图 + 诊断 + 场景面板, 录制.
用法: python -m honeycomb_host.gui.main [--source twin|udp|replay] [--file rec.npz] [--scene point_press]
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QDockWidget, QToolBar, QComboBox, QPushButton, QLabel,
                             QFileDialog, QWidget, QStatusBar)
import pyqtgraph as pg
from .. import geometry as G
from ..twin import Twin, Scenes, Environment, NoiseModel
from ..fastmodel import FastModel, ModelConfig
from ..pipeline import Pipeline
from ..sources import TwinSource, UdpSource, ReplaySource, PipelineWorker
from ..recorder import Recorder
from .. import protocol as P
from .hexmap import HexMap
from .obsview import ObsView
from .diag import DiagView
from .scene_panel import ScenePanel
from .view3d import View3D

pg.setConfigOptions(antialias=False, background='w', foreground='k')

class MainWindow(QMainWindow):
    def __init__(self, source='twin', file=None, scene='point_press', noise='hardware', udp_device=None):
        super().__init__()
        self.setWindowTitle('方案C 19 单元蜂窝触觉 — 主机 (数字孪生 / UDP / 回放)')
        self.resize(1500, 900)
        self.env = Environment(noise=NoiseModel(preset=noise))
        self.model = FastModel(cfg=ModelConfig(f0=self.env.f0, gap=self.env.gap))
        self.pipeline = Pipeline(self.env, model=self.model)
        self.worker = PipelineWorker(self.pipeline)
        self.worker.result_ready.connect(self._on_result)
        self.source = None; self.latest = None; self.recorder = Recorder()
        self.udp_device = udp_device or (P.FPGA_IP, P.FPGA_PORT)
        # 部件
        self.hexmap = HexMap(); self.setCentralWidget(self.hexmap)
        self.obs = ObsView(); self.diag = DiagView(); self.scene_panel = ScenePanel(lambda: self.source)
        self.view3d = View3D(); self.view3d.gap = self.env.gap
        docks = {}
        for name, w, area in (('观测', self.obs, Qt.DockWidgetArea.RightDockWidgetArea),
                              ('3D 场形变', self.view3d, Qt.DockWidgetArea.RightDockWidgetArea),
                              ('诊断', self.diag, Qt.DockWidgetArea.BottomDockWidgetArea),
                              ('孪生场景', self.scene_panel, Qt.DockWidgetArea.LeftDockWidgetArea)):
            d = QDockWidget(name); d.setWidget(w); self.addDockWidget(area, d); docks[name] = d
        self.tabifyDockWidget(docks['观测'], docks['3D 场形变'])     # 右侧标签页: 观测 / 3D
        docks['观测'].raise_()
        self.hexmap.unit_clicked.connect(self.obs.set_unit)
        # 工具栏
        tb = QToolBar(); self.addToolBar(tb)
        tb.addWidget(QLabel(' 数据源: ')); self.src_combo = QComboBox(); self.src_combo.addItems(['twin', 'udp', 'replay']); tb.addWidget(self.src_combo)
        self.btn_start = QPushButton('启动'); self.btn_stop = QPushButton('停止'); tb.addWidget(self.btn_start); tb.addWidget(self.btn_stop)
        self.btn_rec = QPushButton('录制'); self.btn_rec.setCheckable(True); tb.addWidget(self.btn_rec)
        self.btn_save = QPushButton('保存录制…'); tb.addWidget(self.btn_save)
        self.btn_reset = QPushButton('复位跟踪'); tb.addWidget(self.btn_reset)
        self.btn_start.clicked.connect(lambda: self.start_source(self.src_combo.currentText()))
        self.btn_stop.clicked.connect(self.stop_source)
        self.btn_rec.toggled.connect(self._toggle_rec); self.btn_save.clicked.connect(self._save_rec)
        self.btn_reset.clicked.connect(lambda: self.pipeline.tracker.reset())
        self.setStatusBar(QStatusBar())
        # 刷新
        self.timer = QTimer(self); self.timer.timeout.connect(self._redraw); self.timer.start(33)
        self.worker.start()
        self.replay_file = file; self.scene_name = scene
        self.src_combo.setCurrentText(source)
        self.start_source(source)

    def start_source(self, kind):
        self.stop_source()
        if kind == 'twin':
            tw = Twin(env=self.env, model=self.model, scene=getattr(Scenes, self.scene_name)())
            self.source = TwinSource(tw)
        elif kind == 'udp':
            self.source = UdpSource(device=self.udp_device)
        else:
            path = self.replay_file or QFileDialog.getOpenFileName(self, '选择录制', '', '*.npz')[0]
            if not path:
                return
            self.source = ReplaySource(path)
        self.source.frame_ready.connect(self.worker.push)
        self.source.status.connect(lambda s: self.statusBar().showMessage(s, 5000))
        self.source.start()
        self.statusBar().showMessage(f'数据源: {kind}', 3000)
    def stop_source(self):
        if self.source is not None:
            self.source.stop(); self.source = None
    def _toggle_rec(self, on):
        self.worker.recorder = self.recorder if on else None
        if on:
            self.recorder.clear()
    def _save_rec(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存录制', 'rec.npz', '*.npz')
        if path:
            self.recorder.save(path); self.statusBar().showMessage(f'已保存 {len(self.recorder.frames)} 帧 → {path}', 5000)
    def _on_result(self, res):
        self.latest = res
    def _redraw(self):
        res = self.latest
        if res is None:
            return
        self.latest = None
        self.hexmap.update_result(res); self.obs.update_result(res)
        if self.view3d.isVisible():
            self.view3d.update_result(res)
        self.diag.update_result(res, f'队列丢弃 {self.worker.n_dropped}  录制 {len(self.recorder.frames)} 帧')
    def closeEvent(self, ev):
        self.stop_source(); self.worker.stop(); super().closeEvent(ev)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='twin', choices=['twin', 'udp', 'replay'])
    ap.add_argument('--file', default=None); ap.add_argument('--scene', default='point_press', choices=Scenes.ALL)
    ap.add_argument('--noise', default='hardware', choices=['hardware', 'sig2_matched', 'off'])
    ap.add_argument('--device', default=None, help='udp: ip:port (默认 192.168.2.128:5000; sim_device 用 127.0.0.1:5000)')
    ap.add_argument('--screenshot', default=None, help='无头冒烟: 跑 N 秒后截图退出 (path)')
    ap.add_argument('--seconds', type=float, default=3.0)
    a = ap.parse_args(argv)
    dev = None
    if a.device:
        ip, port = a.device.split(':'); dev = (ip, int(port))
    app = QApplication(sys.argv[:1])
    win = MainWindow(a.source, a.file, a.scene, a.noise, dev)
    win.show()
    if a.screenshot:
        def shot():
            win.grab().save(a.screenshot)
            if win.view3d.gl_ok:
                win.view3d.view.grabFramebuffer().save(a.screenshot.replace('.png', '_3d.png'))
            print('screenshot', a.screenshot, 'frames', win.pipeline._last_seq, 'gl', win.view3d.gl_ok); app.quit()
        QTimer.singleShot(int(a.seconds * 1000), shot)
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
