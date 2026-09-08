"""孪生场景控制面板: 施力点/深度/σ/共模剪切/环 ΔT/噪声预设/漂移/C_off. 只在孪生源下启用."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QFormLayout, QSlider, QLabel, QComboBox, QCheckBox, QHBoxLayout, QPushButton)
from ..twin import Scenes, _gauss_w
from .. import geometry as G

class _Slider(QWidget):
    def __init__(self, lo, hi, val, scale=1.0, fmt='{:.2f}', cb=None):
        super().__init__()
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        self.s = QSlider(Qt.Orientation.Horizontal); self.s.setRange(int(lo / scale), int(hi / scale)); self.s.setValue(int(val / scale))
        self.l = QLabel(); self.scale = scale; self.fmt = fmt
        h.addWidget(self.s); h.addWidget(self.l)
        self.s.valueChanged.connect(lambda v: (self.l.setText(fmt.format(v * scale)), cb and cb()))
        self.l.setText(fmt.format(val))
    def value(self):
        return self.s.value() * self.scale

class ScenePanel(QWidget):
    def __init__(self, source_getter, parent=None):
        super().__init__(parent)
        self.get_source = source_getter
        f = QFormLayout(self)
        self.preset = QComboBox(); self.preset.addItems(list(Scenes.ALL)); f.addRow('预置场景', self.preset)
        self.x0 = _Slider(-12, 12, 0, 0.1, '{:.1f} mm', self.apply); f.addRow('施力点 x', self.x0)
        self.y0 = _Slider(-12, 12, 0, 0.1, '{:.1f} mm', self.apply); f.addRow('施力点 y', self.y0)
        self.depth = _Slider(-0.9, 0.2, -0.4, 0.01, '{:.2f} mm', self.apply); f.addRow('压深 (负=压)', self.depth)
        self.sigma = _Slider(2, 15, 6.2, 0.1, '{:.1f} mm', self.apply); f.addRow('σ', self.sigma)
        self.su = _Slider(-0.4, 0.4, 0, 0.01, '{:.2f} mm', self.apply); f.addRow('共模剪切 u', self.su)
        self.sv = _Slider(-0.4, 0.4, 0, 0.01, '{:.2f} mm', self.apply); f.addRow('共模剪切 v', self.sv)
        self.dT = _Slider(0, 40, 0, 1, '{:.0f} K', self.apply); f.addRow('接触 ΔT (环)', self.dT)
        self.noise = QComboBox(); self.noise.addItems(['hardware', 'sig2_matched', 'off']); f.addRow('噪声预设', self.noise)
        self.drift = QCheckBox('链路漂移'); f.addRow(self.drift)
        self.coff = QCheckBox('C_off 干扰'); self.coff.setChecked(True); f.addRow(self.coff)
        self.btn = QPushButton('应用预置'); f.addRow(self.btn)
        self.btn.clicked.connect(self.apply_preset)
        self.noise.currentTextChanged.connect(self.apply_env); self.drift.toggled.connect(self.apply_env); self.coff.toggled.connect(self.apply_env)
    def apply_preset(self):
        src = self.get_source()
        if src is None or not hasattr(src, 'set_scene'):
            return
        src.set_scene(getattr(Scenes, self.preset.currentText())())
    def apply(self):
        src = self.get_source()
        if src is None or not hasattr(src, 'set_scene'):
            return
        x0, y0, d, s = self.x0.value(), self.y0.value(), self.depth.value(), self.sigma.value()
        su, sv, dT = self.su.value(), self.sv.value(), self.dT.value()
        def q(t):
            return np.concatenate([_gauss_w(x0, y0, d, s), np.full(G.NU, su), np.full(G.NU, sv)])
        dT_fn = (lambda t: _gauss_w(x0, y0, dT, s)) if dT > 0 else None
        from ..twin import Scene
        src.set_scene(Scene('manual', q, dT_of_t=dT_fn))
    def apply_env(self):
        src = self.get_source()
        if src is None or not hasattr(src, 'set_param'):
            return
        preset, drift, coff = self.noise.currentText(), self.drift.isChecked(), self.coff.isChecked()
        def fn(tw):
            tw.env.noise.preset = preset; tw.env.link.drift = drift; tw.env.coff_enable = coff
        src.set_param(fn)
