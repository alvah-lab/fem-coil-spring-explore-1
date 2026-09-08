"""诊断: 帧率 / seq 缺口 / 饱和 / tracker ms / χ² / 重线性化与复位 / 链路增益历史 / 真值误差 (孪生时)."""
from __future__ import annotations
from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from .. import geometry as G

class DiagView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel(); self.lbl.setStyleSheet('font-family: monospace'); lay.addWidget(self.lbl)
        self.p = pg.PlotWidget(title='链路增益 |G| (相对首帧) / tracker ms / χ²'); lay.addWidget(self.p)
        self.p.addLegend()
        self.c_g = self.p.plot(pen='b', name='|G|−1 ×1e4'); self.c_ms = self.p.plot(pen='g', name='ms'); self.c_chi = self.p.plot(pen='r', name='χ²')
        self.h = {k: deque(maxlen=500) for k in ('t', 'g', 'ms', 'chi')}
        self.n_gap = 0; self.n_sat = 0; self.n_relin = 0; self.n_reset = 0
    def update_result(self, res, extra: str = ''):
        tr = res.track
        self.n_gap += res.seq_gap; self.n_sat += int(res.sat.sum())
        if tr is not None:
            self.n_relin += int(tr.relin); self.n_reset += int(tr.reset)
        self.h['t'].append(res.t_s); self.h['g'].append((abs(res.link_gain) - 1) * 1e4)
        self.h['ms'].append(tr.ms if tr else 0); self.h['chi'].append(min(tr.chi2, 1e3) if tr else 0)
        t = np.array(self.h['t'])
        self.c_g.setData(t, np.array(self.h['g'])); self.c_ms.setData(t, np.array(self.h['ms'])); self.c_chi.setData(t, np.array(self.h['chi']))
        txt = [f'seq {res.seq:8d}  t {res.t_s:8.3f}s  fps {res.fps:5.1f}  seq缺口 {self.n_gap}  饱和驻留 {self.n_sat}',
               f'链路 |G| {abs(res.link_gain):.6f} ∠ {np.degrees(np.angle(res.link_gain)):+.4f}°']
        if tr is not None:
            txt.append(f'tracker {tr.ms:5.1f}ms  χ² {tr.chi2:8.2f}  |dq| {tr.dq_norm:.4f}  重线性化 {self.n_relin}  复位 {self.n_reset}  钳位 {tr.clamped}')
            txt.append(f'弱模式幅 {np.array2string(tr.weak_amp, precision=3)}  中心 w {tr.w[G.CI]*1e3:+7.1f}µm  均 u {tr.u.mean()*1e3:+6.1f} v {tr.v.mean()*1e3:+6.1f}µm')
            if res.truth is not None:
                e = tr.q - res.truth.q
                txt.append(f'真值误差: w max {np.abs(e[:G.NU]).max()*1e3:6.2f}µm  面内 max {np.abs(e[G.NU:]).max()*1e3:6.2f}µm  '
                           f'真值中心 w {res.truth.q[G.CI]*1e3:+7.1f}µm')
        if extra:
            txt.append(extra)
        self.lbl.setText('\n'.join(txt))
