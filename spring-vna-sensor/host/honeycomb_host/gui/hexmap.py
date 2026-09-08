"""蜂窝热图: 19 个六边形 (填色 = w / ΔT / 自反射), 箭头 = (u,v) 或剪切, 下沉单元描边; 点击发单元 id."""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt, QPointF
from PyQt6.QtGui import QPolygonF, QPen, QBrush, QColor
from PyQt6.QtWidgets import QGraphicsPolygonItem, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from .. import geometry as G

MODES = {'w (µm)': ('w', -500, 50), 'u (µm)': ('u', -200, 200), 'v (µm)': ('v', -200, 200),
         'ΔT 环 (K)': ('dT', -5, 20), '自反射 (nH)': ('refl', -120, 0), '残差 χ': ('resid', -5, 5)}

class HexMap(QWidget):
    unit_clicked = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout(); lay.addLayout(top)
        top.addWidget(QLabel('填色:')); self.mode = QComboBox(); self.mode.addItems(list(MODES)); top.addWidget(self.mode)
        top.addWidget(QLabel('箭头:')); self.arrow = QComboBox(); self.arrow.addItems(['(u,v) 位移', '无']); top.addWidget(self.arrow)
        top.addStretch()
        self.pw = pg.PlotWidget(); lay.addWidget(self.pw)
        self.pw.setAspectLocked(True); self.pw.setBackground('w'); self.pw.hideAxis('left'); self.pw.hideAxis('bottom')
        self.cmap = pg.colormap.get('viridis')
        self.items = []
        for i in range(G.NU):
            v = G.hex_vertices(*G.XY[i], G.PITCH / np.sqrt(3) * 0.97)
            it = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in v]))
            it.setPen(QPen(QColor(200, 60, 60) if G.SUNK[i] else QColor(60, 60, 60), 0.25 if G.SUNK[i] else 0.08))
            it.setBrush(QBrush(QColor(220, 220, 220)))
            it.unit = i
            self.pw.addItem(it); self.items.append(it)
            t = pg.TextItem(str(i), color=(0, 0, 0), anchor=(0.5, 0.5)); t.setPos(G.XY[i, 0], G.XY[i, 1] + 1.6)
            self.pw.addItem(t)
        self.arrows = pg.PlotDataItem(pen=pg.mkPen((10, 10, 200), width=2), connect='pairs'); self.pw.addItem(self.arrows)
        self.heads = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(10, 10, 200), pen=None); self.pw.addItem(self.heads)
        self.cbar = pg.ColorBarItem(colorMap=self.cmap, width=10, interactive=False)
        self.cbar.setLevels((-500, 50))
        self.pw.getPlotItem().layout.addItem(self.cbar, 2, 5)
        self.pw.scene().sigMouseClicked.connect(self._click)
        self.arrow_scale = 10.0    # µm → mm 显示
    def _click(self, ev):
        pos = self.pw.getPlotItem().vb.mapSceneToView(ev.scenePos())
        d = np.hypot(G.XY[:, 0] - pos.x(), G.XY[:, 1] - pos.y())
        if d.min() < G.PITCH / 2:
            self.unit_clicked.emit(int(np.argmin(d)))
    def update_result(self, res):
        key, lo, hi = MODES[self.mode.currentText()]
        tr = res.track
        if key == 'w' and tr is not None:
            vals = tr.w * 1e3
        elif key == 'u' and tr is not None:
            vals = tr.u * 1e3
        elif key == 'v' and tr is not None:
            vals = tr.v * 1e3
        elif key == 'dT':
            vals = res.dT_ring
        elif key == 'refl':
            vals = res.refl_nH[:G.NU]
        elif key == 'resid' and tr is not None:
            vals = tr.resid[:G.NU]
        else:
            vals = np.zeros(G.NU)
        self.cbar.setLevels((lo, hi))
        cols = self.cmap.map(np.clip((vals - lo) / (hi - lo), 0, 1), mode='qcolor')
        for it, c in zip(self.items, cols):
            it.setBrush(QBrush(c))
        if self.arrow.currentText().startswith('(u') and tr is not None:
            u = tr.u * 1e3 / self.arrow_scale * 0.1; v = tr.v * 1e3 / self.arrow_scale * 0.1   # 100µm → 1mm
            xs = np.empty(2 * G.NU); ys = np.empty(2 * G.NU)
            xs[0::2] = G.XY[:, 0]; ys[0::2] = G.XY[:, 1]; xs[1::2] = G.XY[:, 0] + u; ys[1::2] = G.XY[:, 1] + v
            self.arrows.setData(xs, ys); self.heads.setData(xs[1::2], ys[1::2])
        else:
            self.arrows.setData([], []); self.heads.setData([], [])
