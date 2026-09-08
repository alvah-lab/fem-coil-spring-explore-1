"""61 观测视图: 反射(扣载波)条形 + 加权残差条形; 选中单元自观测时序; 奇异值谱."""
from __future__ import annotations
from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from .. import geometry as G

class ObsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.p1 = pg.PlotWidget(title='反射 L − 载波 (nH): 0..18 自观测, 19..60 边观测'); lay.addWidget(self.p1)
        self.p2 = pg.PlotWidget(title='加权残差 (y − ŷ)/σ'); lay.addWidget(self.p2)
        self.p3 = pg.PlotWidget(title='选中单元: 自观测反射 (nH) 与 w (µm) 时序'); lay.addWidget(self.p3)
        self.p4 = pg.PlotWidget(title='Jacobian 奇异值 (57), 红=弱模式'); lay.addWidget(self.p4)
        self.p4.setLogMode(y=True)
        x = np.arange(G.NOBS)
        col = np.array(['#1f77b4'] * G.NU + ['#ff7f0e'] * G.NE)
        self.b1 = pg.BarGraphItem(x=x, height=np.zeros(G.NOBS), width=0.8, brushes=list(col)); self.p1.addItem(self.b1)
        self.b2 = pg.BarGraphItem(x=x, height=np.zeros(G.NOBS), width=0.8, brushes=list(col)); self.p2.addItem(self.b2)
        self.p2.setYRange(-5, 5)
        self.c_refl = self.p3.plot(pen=pg.mkPen('#1f77b4', width=1.5), name='refl')
        self.c_w = self.p3.plot(pen=pg.mkPen('#d62728', width=1.5), name='w')
        self.p3.addLegend()
        self.sv = self.p4.plot(pen=None, symbol='o', symbolSize=5, symbolBrush='k')
        self.sv_weak = self.p4.plot(pen=None, symbol='o', symbolSize=7, symbolBrush='r')
        self.unit = G.CI
        self.hist_refl = deque(maxlen=770); self.hist_w = deque(maxlen=770); self.hist_t = deque(maxlen=770)
    def set_unit(self, u: int):
        self.unit = u; self.hist_refl.clear(); self.hist_w.clear(); self.hist_t.clear()
        self.p3.setTitle(f'单元 {u}: 自观测反射 (nH) 与 w (µm) 时序')
    def update_result(self, res):
        self.b1.setOpts(height=res.refl_nH)
        tr = res.track
        if tr is not None:
            self.b2.setOpts(height=np.clip(tr.resid, -5, 5))
            S = tr.sv; order = np.argsort(S)
            self.sv.setData(np.arange(len(S)), np.sort(S)[::-1])
            self.sv_weak.setData(np.arange(len(S) - 3, len(S)), np.sort(S)[:3][::-1])
            self.hist_w.append(tr.w[self.unit] * 1e3)
        else:
            self.hist_w.append(0.0)
        self.hist_refl.append(res.refl_nH[self.unit]); self.hist_t.append(res.t_s)
        t = np.array(self.hist_t)
        self.c_refl.setData(t, np.array(self.hist_refl)); self.c_w.setData(t, np.array(self.hist_w))
