"""回板测试面板: 状态 / 单驻留 / 基线 / 单元扫描 / 垫片标定 / 噪声.

直接消费数据源的原始帧 (任意驻留表), 不经 Pipeline; 对孪生数据源额外提供"放置垫片/无环"控制,
使同一套操作可以先在数字孪生上演练 (见 docs/回板测试手册_数字孪生.md)。"""
from __future__ import annotations
import time
import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QFileDialog,
                             QLineEdit, QProgressBar, QGroupBox)
import pyqtgraph as pg
from .. import geometry as G
from .. import protocol as P
from ..twin import default_dwell_table, dwell_word, Scenes, word_fields
from ..bringup import frame_to_Z, obs61_from_frame, Baseline, CalibLog, RunningStats, model_shim, single_dwell_table

REGS_SHOW = ['DEVICE_ID', 'FW_ID', 'STATUS', 'LINK_STATUS', 'ERR_CNT', 'FRAME_CTRL', 'FRAME_ID', 'N_DWELL', 'DWELL_NSAMP',
             'BLANK_NSAMP', 'DRIVE_AMP', 'RF_EN']


def _dspin(lo, hi, val, step, dec=2, suffix=''):
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step); s.setDecimals(dec); s.setSuffix(suffix)
    return s


class BringupPanel(QWidget):
    def __init__(self, source_getter, pipeline, parent=None):
        super().__init__(parent)
        self.get_source = source_getter
        self.pipeline = pipeline
        self.env = pipeline.env
        self.model = pipeline.model
        self.w = 2 * np.pi * self.env.f0
        self.baseline: Baseline | None = None
        self.calib = CalibLog()
        self.stats = RunningStats(200)
        self._collect = None          # (n_wanted, frames, callback)
        self.last_frame = None
        self.hist_V = []; self.hist_ph = []
        self.scan_ema = None
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2)
        self.tabs = QTabWidget(); lay.addWidget(self.tabs)
        self._build_status(); self._build_single(); self._build_baseline(); self._build_scan(); self._build_calib(); self._build_noise()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(500)

    # ---------- 数据入口 ----------
    @pyqtSlot(object)
    def on_frame(self, fr):
        self.last_frame = fr
        self.stats.push(fr, self.env)
        if self._collect is not None:
            n, frames, cb = self._collect
            frames.append(fr)
            self.progress.setValue(len(frames))
            if len(frames) >= n:
                self._collect = None
                cb(frames)
        Z, meta = frame_to_Z(fr, self.env)
        if len(fr.dwells) == 1:
            self._update_single(Z, meta)
        elif len(fr.dwells) == 63:
            self._update_scan(fr)

    def _send(self, name, value=None):
        src = self.get_source()
        if src is None:
            return False
        return src.send_command(P.Command(name, value))

    def _collect_frames(self, n, cb):
        self.progress.setMaximum(n); self.progress.setValue(0)
        self._collect = (n, [], cb)

    # ---------- 1 状态 ----------
    def _build_status(self):
        w = QWidget(); f = QGridLayout(w)
        self.reg_labels = {}
        for r, name in enumerate(REGS_SHOW):
            f.addWidget(QLabel(name), r, 0); lab = QLabel('—'); self.reg_labels[name] = lab; f.addWidget(lab, r, 1)
        row = len(REGS_SHOW)
        self.btn_refresh = QPushButton('刷新寄存器'); self.btn_refresh.clicked.connect(self.refresh_regs); f.addWidget(self.btn_refresh, row, 0)
        self.auto_refresh = QCheckBox('每秒自动刷新'); f.addWidget(self.auto_refresh, row, 1)
        self.amp = QSpinBox(); self.amp.setRange(0, 8191); self.amp.setValue(4096); f.addWidget(QLabel('DRIVE_AMP'), row + 1, 0); f.addWidget(self.amp, row + 1, 1)
        b = QPushButton('写 DRIVE_AMP'); b.clicked.connect(lambda: self._send('DRIVE_AMP', self.amp.value())); f.addWidget(b, row + 1, 2)
        self.rf = QCheckBox('RF_EN (PA_RUN)'); self.rf.toggled.connect(lambda on: self._send('RF_EN', int(on))); f.addWidget(self.rf, row + 2, 0)
        b2 = QPushButton('START'); b2.clicked.connect(lambda: self._send('start')); f.addWidget(b2, row + 2, 1)
        b3 = QPushButton('STOP'); b3.clicked.connect(lambda: self._send('stop')); f.addWidget(b3, row + 2, 2)
        self.status_line = QLabel(''); f.addWidget(self.status_line, row + 3, 0, 1, 3)
        f.setRowStretch(row + 4, 1)
        self.tabs.addTab(w, '状态')

    def refresh_regs(self):
        src = self.get_source()
        if src is None:
            self.status_line.setText('无数据源'); return
        for name in REGS_SHOW:
            v = src.read_reg(name)
            if v is None:
                self.reg_labels[name].setText('无应答'); continue
            txt = f'{v:#010x}' if name in ('DEVICE_ID', 'FW_ID') else str(v)
            if name == 'STATUS':
                txt += f'  running={v & 1} adc_locked={(v >> 1) & 1} phy={(v >> 2) & 1} link_ready={(v >> 3) & 1} fault={(v >> 4) & 1}'
            if name == 'LINK_STATUS':
                txt += f'  ready={v & 1} fault={(v >> 1) & 1} timeouts={(v >> 8) & 255} faults={(v >> 16) & 255}'
            if name == 'ERR_CNT':
                txt += f'  tx_drop={v & 255} sat_dwells={(v >> 8) & 255}'
            self.reg_labels[name].setText(txt)
        self.status_line.setText(f'刷新 {time.strftime("%H:%M:%S")}  帧率 {self.stats.fps():.1f} Hz  丢帧 {self.stats.n_gap}')

    # ---------- 2 单驻留 ----------
    def _build_single(self):
        w = QWidget(); v = QVBoxLayout(w); f = QFormLayout()
        self.s_drv = QSpinBox(); self.s_drv.setRange(0, 31); self.s_drv.setValue(9); f.addRow('drv (0-18, 31=无)', self.s_drv)
        self.s_sns = QSpinBox(); self.s_sns.setRange(0, 31); self.s_sns.setValue(9); f.addRow('sns (0-18, 31=无)', self.s_sns)
        self.s_pga = QComboBox(); self.s_pga.addItems(['0: ×1', '1: ×10', '2: ×100']); f.addRow('PGA', self.s_pga)
        self.s_ref = QCheckBox('ref (F÷100→S)'); self.s_vna = QCheckBox('vna (电流标定)'); hh = QHBoxLayout(); hh.addWidget(self.s_ref); hh.addWidget(self.s_vna); f.addRow(hh)
        v.addLayout(f)
        hb = QHBoxLayout()
        b1 = QPushButton('只测这一条'); b1.clicked.connect(self.apply_single); hb.addWidget(b1)
        b2 = QPushButton('恢复默认 63 表'); b2.clicked.connect(lambda: self._send('dwell_table', default_dwell_table())); hb.addWidget(b2)
        v.addLayout(hb)
        self.single_text = QLabel('—'); self.single_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); v.addWidget(self.single_text)
        self.pw_single = pg.PlotWidget(title='|V| (mV, 左)  ∠V (°, 右)'); self.pw_single.setBackground('w'); self.pw_single.addLegend()
        self.c_absV = self.pw_single.plot(pen=pg.mkPen((30, 90, 200), width=2), name='|V| mV')
        self.pw_single.showAxis('right'); self.vb2 = pg.ViewBox(); self.pw_single.scene().addItem(self.vb2)
        self.pw_single.getAxis('right').linkToView(self.vb2); self.vb2.setXLink(self.pw_single)
        self.c_ph = pg.PlotCurveItem(pen=pg.mkPen((200, 60, 60), width=1)); self.vb2.addItem(self.c_ph)
        self.pw_single.getViewBox().sigResized.connect(lambda: self.vb2.setGeometry(self.pw_single.getViewBox().sceneBoundingRect()))
        v.addWidget(self.pw_single)
        self.tabs.addTab(w, '单驻留')

    def apply_single(self):
        pga = self.s_pga.currentIndex()
        tbl = single_dwell_table(self.s_drv.value(), self.s_sns.value(), pga, int(self.s_ref.isChecked()), int(self.s_vna.isChecked()))
        self.hist_V.clear(); self.hist_ph.clear(); self.stats = RunningStats(self.n_win.value())
        self._send('dwell_table', tbl)

    def _update_single(self, Z, meta):
        V, I = meta['V'][0], meta['I'][0]
        self.hist_V.append(abs(V) * 1e3); self.hist_ph.append(np.degrees(np.angle(V)))
        self.hist_V = self.hist_V[-400:]; self.hist_ph = self.hist_ph[-400:]
        z = Z[0]
        f = word_fields(int(meta['words'][0]))
        txt = (f'word 0x{int(meta["words"][0]):04x} drv={f["drv"]} sns={f["sns"]} pga=×{meta["pga"][0]:g} ref={f["ref"]} vna={f["vna"]}   '
               f'|V|={abs(V)*1e3:.3f} mV ∠{np.degrees(np.angle(V)):.2f}°   |I|={abs(I)*1e3:.2f} mV (→{abs(I)/self.env.link.isense_V_per_A*1e3:.2f} mA)')
        if np.isfinite(z):
            txt += f'\nZ = {z.real:.4f} + j{z.imag:.4f} Ω   L = {z.imag / self.w * 1e9:.3f} nH   R = {z.real:.4f} Ω'
            n = meta['obs'][0]
            if n is not None:
                car = self.model.carrier_L()[n]
                base = self.baseline.L61[n] if self.baseline is not None else None
                txt += f'   模型载波 {car:.3f} nH  ΔL(模型) {z.imag / self.w * 1e9 - car:+.3f}'
                if base is not None:
                    txt += f'  ΔL(基线) {z.imag / self.w * 1e9 - base:+.3f}'
        txt += f'   sat={int(meta["sat"][0])} timeout={int(meta["link_timeout"][0])} fault={int(meta["link_fault"][0])}  seq={meta["seq"]}'
        self.single_text.setText(txt)
        self.c_absV.setData(self.hist_V); self.c_ph.setData(np.arange(len(self.hist_ph)), np.array(self.hist_ph))

    # ---------- 3 基线 ----------
    def _build_baseline(self):
        w = QWidget(); v = QVBoxLayout(w); hb = QHBoxLayout()
        self.bl_n = QSpinBox(); self.bl_n.setRange(1, 5000); self.bl_n.setValue(100); hb.addWidget(QLabel('帧数')); hb.addWidget(self.bl_n)
        b = QPushButton('采集基线 (无环)'); b.clicked.connect(self.collect_baseline); hb.addWidget(b)
        b2 = QPushButton('保存 JSON…'); b2.clicked.connect(self.save_baseline); hb.addWidget(b2)
        b3 = QPushButton('载入 JSON…'); b3.clicked.connect(self.load_baseline); hb.addWidget(b3)
        b4 = QPushButton('应用为载波'); b4.clicked.connect(self.apply_baseline); hb.addWidget(b4)
        b5 = QPushButton('恢复模型载波'); b5.clicked.connect(lambda: (self.pipeline.set_baseline(None), self.bl_info.setText('已恢复模型载波'))); hb.addWidget(b5)
        v.addLayout(hb)
        self.progress = QProgressBar(); v.addWidget(self.progress)
        self.bl_info = QLabel('未采集'); v.addWidget(self.bl_info)
        self.bl_table = QTableWidget(G.NOBS, 6); self.bl_table.setHorizontalHeaderLabels(['观测', '类型', 'L 实测 nH', 'L 模型 nH', 'ΔL nH', 'σ_L nH / R Ω'])
        v.addWidget(self.bl_table)
        self.tabs.addTab(w, '基线')

    def collect_baseline(self):
        self._send('dwell_table', default_dwell_table())
        self._collect_frames(self.bl_n.value(), self._baseline_done)
        self.bl_info.setText('采集中…（确认板上没有铜环）')

    def _baseline_done(self, frames):
        try:
            self.baseline = Baseline.from_frames(frames, self.env, note='GUI')
        except ValueError as e:
            self.bl_info.setText(f'失败: {e}'); return
        car = self.model.carrier_L()
        bl = self.baseline
        for n, (kind, i, j) in enumerate(G.OBS):
            vals = [f'{n}', f'{kind} {i}' + (f'-{j}' if kind == 'edge' else ''), f'{bl.L61[n]:.3f}', f'{car[n]:.3f}', f'{bl.L61[n] - car[n]:+.3f}',
                    f'{(bl.sigma_L61[n] if bl.sigma_L61 is not None else 0):.4f} / {bl.R61[n]:.3f}']
            for c, t in enumerate(vals):
                self.bl_table.setItem(n, c, QTableWidgetItem(t))
        d = bl.L61 - car
        self.bl_info.setText(f'{bl.n_frames} 帧 @ {bl.when}: 自观测 ΔL(实测−模型) 中位 {np.median(d[:19]):+.3f} nH, 边 {np.median(d[19:]):+.3f} nH; '
                             f'自观测 σ_L 中位 {np.median(bl.sigma_L61[:19]) if bl.sigma_L61 is not None else 0:.4f} nH')

    def save_baseline(self):
        if self.baseline is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存基线', 'baseline.json', '*.json')
        if path:
            self.baseline.save(path); self.bl_info.setText(f'已保存 {path}')

    def load_baseline(self):
        path, _ = QFileDialog.getOpenFileName(self, '载入基线', '', '*.json')
        if path:
            self.baseline = Baseline.load(path); self._baseline_done_from_loaded()

    def _baseline_done_from_loaded(self):
        bl = self.baseline; car = self.model.carrier_L()
        for n, (kind, i, j) in enumerate(G.OBS):
            for c, t in enumerate([f'{n}', f'{kind} {i}' + (f'-{j}' if kind == 'edge' else ''), f'{bl.L61[n]:.3f}', f'{car[n]:.3f}', f'{bl.L61[n] - car[n]:+.3f}', f'{bl.R61[n]:.3f}']):
                self.bl_table.setItem(n, c, QTableWidgetItem(t))
        self.bl_info.setText(f'已载入 {bl.n_frames} 帧基线 @ {bl.when} {bl.note}')

    def apply_baseline(self):
        if self.baseline is None:
            self.bl_info.setText('先采集或载入基线'); return
        self.pipeline.set_baseline(self.baseline.L61)
        self.bl_info.setText('已把实测基线应用为跟踪器载波 (refl = L − 基线)')

    # ---------- 4 单元扫描 ----------
    def _build_scan(self):
        w = QWidget(); v = QVBoxLayout(w); hb = QHBoxLayout()
        hb.addWidget(QLabel('参考:')); self.scan_ref = QComboBox(); self.scan_ref.addItems(['基线 (若有) 否则模型', '模型载波']); hb.addWidget(self.scan_ref)
        hb.addWidget(QLabel('EMA α')); self.scan_alpha = _dspin(0.01, 1.0, 0.2, 0.05, 2); hb.addWidget(self.scan_alpha)
        hb.addWidget(QLabel('选中单元')); self.scan_unit = QSpinBox(); self.scan_unit.setRange(0, 18); self.scan_unit.setValue(9); hb.addWidget(self.scan_unit)
        b = QPushButton('清零 EMA'); b.clicked.connect(lambda: setattr(self, 'scan_ema', None)); hb.addWidget(b); hb.addStretch()
        v.addLayout(hb)
        self.pw_scan = pg.PlotWidget(title='ΔL 自观测 (nH) 相对参考, 按单元'); self.pw_scan.setBackground('w')
        self.bar_scan = pg.BarGraphItem(x=np.arange(G.NU), height=np.zeros(G.NU), width=0.7, brush=(60, 120, 200)); self.pw_scan.addItem(self.bar_scan)
        v.addWidget(self.pw_scan)
        self.scan_text = QLabel('—'); v.addWidget(self.scan_text)
        self.tabs.addTab(w, '单元扫描')

    def _ref_L(self):
        if self.scan_ref.currentIndex() == 0 and self.baseline is not None:
            return self.baseline.L61, '基线'
        return self.model.carrier_L(), '模型'

    def _update_scan(self, fr):
        z61, meta = obs61_from_frame(fr, self.env)
        if z61 is None:
            return
        L = np.imag(z61) / self.w * 1e9
        a = self.scan_alpha.value()
        self.scan_ema = L if self.scan_ema is None else (1 - a) * self.scan_ema + a * L
        if self.tabs.currentIndex() != 3:
            return
        ref, name = self._ref_L()
        dL = self.scan_ema - ref
        self.bar_scan.setOpts(height=dL[:G.NU])
        u = self.scan_unit.value()
        edges = ' '.join(f'{(j if i == u else i)}:{dL[n]:+.3f}' for n, (k, i, j) in enumerate(G.OBS) if k == 'edge' and (i == u or j == u))
        self.scan_text.setText(f'参考={name}  单元 {u}: ΔL_self {dL[u]:+.3f} nH  邻边 {edges}  (sat {int(meta["sat"].sum())})')

    # ---------- 5 垫片标定 ----------
    def _build_calib(self):
        w = QWidget(); v = QVBoxLayout(w); f = QFormLayout()
        self.c_unit = QSpinBox(); self.c_unit.setRange(0, 18); self.c_unit.setValue(9); f.addRow('单元', self.c_unit)
        self.c_gap = _dspin(0.8, 3.5, 1.75, 0.05, 3, ' mm'); f.addRow('间隙 (环质心→L1)', self.c_gap)
        self.c_tilt = _dspin(0, 15, 0, 0.5, 1, ' °'); f.addRow('倾角', self.c_tilt)
        self.c_dir = _dspin(0, 360, 0, 15, 0, ' °'); f.addRow('倾斜方位', self.c_dir)
        self.c_dx = _dspin(-1.5, 1.5, 0, 0.1, 2, ' mm'); f.addRow('偏移 x', self.c_dx)
        self.c_dy = _dspin(-1.5, 1.5, 0, 0.1, 2, ' mm'); f.addRow('偏移 y', self.c_dy)
        self.c_n = QSpinBox(); self.c_n.setRange(1, 5000); self.c_n.setValue(50); f.addRow('帧数', self.c_n)
        self.c_note = QLineEdit(); f.addRow('备注', self.c_note)
        v.addLayout(f)
        hb = QHBoxLayout()
        b = QPushButton('记录此工况'); b.clicked.connect(self.record_calib); hb.addWidget(b)
        b2 = QPushButton('保存 CSV…'); b2.clicked.connect(self.save_calib); hb.addWidget(b2)
        b3 = QPushButton('模型预测'); b3.clicked.connect(self.predict_calib); hb.addWidget(b3)
        v.addLayout(hb)
        g = QGroupBox('数字孪生 (仅 twin 数据源): 把上面的工况放进孪生'); gh = QHBoxLayout(g)
        b4 = QPushButton('放置垫片+环 (其余无环)'); b4.clicked.connect(lambda: self.twin_scene('shim')); gh.addWidget(b4)
        b5 = QPushButton('全部无环'); b5.clicked.connect(lambda: self.twin_scene('no_rings')); gh.addWidget(b5)
        b6 = QPushButton('全部静息'); b6.clicked.connect(lambda: self.twin_scene('rest')); gh.addWidget(b6)
        v.addWidget(g)
        self.calib_info = QLabel('—'); v.addWidget(self.calib_info)
        self.calib_table = QTableWidget(0, 9); self.calib_table.setHorizontalHeaderLabels(['时间', '单元', '间隙', '倾角/方位', 'dx,dy', 'ΔL_self 实测', 'σ', 'ΔL_self 模型', '邻边 实测/模型'])
        v.addWidget(self.calib_table)
        self.tabs.addTab(w, '垫片标定')

    def _calib_args(self):
        return dict(unit=self.c_unit.value(), gap_mm=self.c_gap.value(), tilt_deg=self.c_tilt.value(), tilt_dir_deg=self.c_dir.value(),
                    dx_mm=self.c_dx.value(), dy_mm=self.c_dy.value())

    def twin_scene(self, kind):
        src = self.get_source()
        if src is None or not hasattr(src, 'set_scene'):
            self.calib_info.setText('当前数据源不是孪生'); return
        a = self._calib_args()
        sc = Scenes.shim(a['unit'], a['gap_mm'], a['tilt_deg'], a['tilt_dir_deg'], a['dx_mm'], a['dy_mm'], model_gap=self.env.gap) if kind == 'shim' \
            else Scenes.no_rings() if kind == 'no_rings' else Scenes.rest()
        src.set_scene(sc); self.scan_ema = None; self.stats = RunningStats(self.n_win.value())
        self.calib_info.setText(f'孪生场景 → {sc.name}')

    def predict_calib(self):
        a = self._calib_args()
        _, ms, me = model_shim(self.model, **a)
        self.calib_info.setText(f'模型: ΔL_self {ms:+.3f} nH; 邻边 ' + ' '.join(f'{k}:{v:+.3f}' for k, v in sorted(me.items())))

    def record_calib(self):
        if self.baseline is None:
            self.calib_info.setText('先采集/载入无环基线'); return
        self._send('dwell_table', default_dwell_table())
        a = self._calib_args(); note = self.c_note.text()
        def done(frames):
            row = self.calib.add_from_frames(frames, self.env, self.baseline.L61, self.model, note=note, **a)
            r = self.calib_table.rowCount(); self.calib_table.insertRow(r)
            e = ' '.join(f'{k}:{v:+.3f}/{row.model_edges_nH.get(k, float("nan")):+.3f}' for k, v in sorted(row.dL_edges_nH.items()))
            for c, t in enumerate([row.when[-8:], str(row.unit), f'{row.gap_mm:.3f}', f'{row.tilt_deg:g}/{row.tilt_dir_deg:g}', f'{row.dx_mm:g},{row.dy_mm:g}',
                                   f'{row.dL_self_nH:+.4f}', f'{row.sigma_self_nH:.4f}', f'{row.model_self_nH:+.4f}', e]):
                self.calib_table.setItem(r, c, QTableWidgetItem(t))
            self.calib_info.setText(f'记录 {r + 1} 行: ΔL_self 实测 {row.dL_self_nH:+.4f} vs 模型 {row.model_self_nH:+.4f} nH '
                                    f'({(row.dL_self_nH / row.model_self_nH - 1) * 100 if row.model_self_nH else 0:+.1f} %)')
        self._collect_frames(self.c_n.value(), done)
        self.calib_info.setText('采集中…')

    def save_calib(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存标定记录', 'calib.csv', '*.csv')
        if path:
            self.calib.save_csv(path); self.calib.save_json(path.rsplit('.', 1)[0] + '.json'); self.calib_info.setText(f'已保存 {path}')

    # ---------- 6 噪声 ----------
    def _build_noise(self):
        w = QWidget(); v = QVBoxLayout(w); hb = QHBoxLayout()
        hb.addWidget(QLabel('窗口 (帧)')); self.n_win = QSpinBox(); self.n_win.setRange(10, 5000); self.n_win.setValue(200); hb.addWidget(self.n_win)
        b = QPushButton('重置统计'); b.clicked.connect(lambda: (setattr(self, 'stats', RunningStats(self.n_win.value())))); hb.addWidget(b); hb.addStretch()
        v.addLayout(hb)
        self.noise_text = QLabel('—'); v.addWidget(self.noise_text)
        self.pw_noise = pg.PlotWidget(title='σ_L 每观测 (nH): 0-18 自, 19-60 边'); self.pw_noise.setBackground('w'); self.pw_noise.setLogMode(y=True)
        self.bar_noise = pg.BarGraphItem(x=np.arange(G.NOBS), height=np.full(G.NOBS, 1e-4), width=0.8, brush=(120, 80, 160)); self.pw_noise.addItem(self.bar_noise)
        v.addWidget(self.pw_noise)
        self.tabs.addTab(w, '噪声/稳定性')

    def _tick(self):
        if self.auto_refresh.isChecked() and self.tabs.currentIndex() == 0:
            self.refresh_regs()
        if self.tabs.currentIndex() == 5:
            st = self.stats
            sig = st.sigma_L61()
            txt = f'帧率 {st.fps():.1f} Hz  帧数 {st.n_frames}  丢帧 {st.n_gap}  饱和驻留 {st.n_sat}  链路超时 {st.n_timeout}  FAULT {st.n_fault}'
            if sig is not None:
                nH_um = 11.0    # 自反射灵敏度量级 nH/mm → 1 nH ≈ 90 µm
                txt += f'\nσ_L 自观测中位 {np.median(sig[:19]):.4f} nH (≈{np.median(sig[:19]) / nH_um * 1e3:.1f} µm)  边中位 {np.median(sig[19:]):.5f} nH  ({len(st.L61)} 帧窗口)'
                self.bar_noise.setOpts(height=np.maximum(sig, 1e-6))
            sv = st.sigma_V()
            if sv is not None:
                txt += f'\nσ|V| 中位 {np.median(sv) * 1e6:.1f} µV'
            self.noise_text.setText(txt)
