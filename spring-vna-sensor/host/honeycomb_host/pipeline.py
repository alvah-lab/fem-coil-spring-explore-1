"""主机处理链: Frame → (参考驻留增益 / 电流通道) → Z=V/I → 载波标定 → L_eff → Tracker → 温度."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import geometry as G
from .fastmodel import FastModel, ModelConfig
from .twin import (Frame, decode_dwells, OBS_REF, OBS_ISENSE, N_DWELL, FLAG_SAT, LSB, Environment)
from .invert import Tracker, TrackerConfig, TrackerOutput

@dataclass
class Calibration:
    """标定状态. carrier_Z: 静置帧的 61 复阻抗 (含载波+静息反射); 反演用 model 自己的载波, 这里只做链路/电流标定."""
    link_gain: complex = 1 + 0j
    link_gain_i: complex = 1 + 0j
    ref_V0: complex | None = None      # 首帧参考驻留 (无漂移时基准)
    ref_I0: complex | None = None
    rest_Z: np.ndarray | None = None   # 静置平均 (用于显示"扣静息"的反射变化)
    n_rest: int = 0

@dataclass
class FrameResult:
    seq: int
    t_s: float
    Z: np.ndarray                  # (61,) complex 端口阻抗 (已除链路增益)
    L_nH: np.ndarray               # Im/ω
    refl_nH: np.ndarray            # L − 模型载波
    R_ohm: np.ndarray              # 实部
    dT_ring: np.ndarray            # (19,) K 由 Re/Im 估计 (相对首帧)
    sat: np.ndarray                # (63,) bool
    link_gain: complex
    track: TrackerOutput | None
    truth: object = None
    fps: float = 0.0
    seq_gap: int = 0

class Pipeline:
    def __init__(self, env: Environment, model: FastModel | None = None, tracker_cfg: TrackerConfig | None = None,
                 pga_gains=(1.0, 10.0, 100.0, 100.0), isense_V_per_A=100.0, track: bool = True):
        self.env = env
        self.model = model or FastModel(cfg=ModelConfig(f0=env.f0, gap=env.gap))
        self.w = 2 * np.pi * env.f0
        self.pga_gains = np.array(pga_gains)
        self.isense = isense_V_per_A
        self.cal = Calibration()
        self.tracker = Tracker(self.model, tracker_cfg) if track else None
        self.carrier_L = self.model.carrier_L()
        self.q_ring0 = None
        self._last_seq = None
        self._t_last = None
        self.fps = 0.0

    def calibrate_rest(self, Z: np.ndarray):
        c = self.cal
        c.rest_Z = Z.copy() if c.rest_Z is None else (c.rest_Z * c.n_rest + Z) / (c.n_rest + 1)
        c.n_rest += 1

    def process(self, fr: Frame, wall_t: float | None = None) -> FrameResult:
        d = fr.dwells
        n_eff = fr.dwell_nsamp - self.env.link.blank_nsamp
        V, I = decode_dwells(d, n_eff)
        pga = self.pga_gains[(d['dwell_word'] >> 4) & 3]
        sat = (d['flags'] & FLAG_SAT) != 0
        # 参考驻留 → 链路增益 (相对首帧)
        c = self.cal
        if c.ref_V0 is None:
            c.ref_V0 = V[OBS_REF] / pga[OBS_REF]; c.ref_I0 = I[OBS_ISENSE]
        if abs(c.ref_V0) > 0:
            c.link_gain = (V[OBS_REF] / pga[OBS_REF]) / c.ref_V0
        if abs(c.ref_I0) > 0:
            c.link_gain_i = I[OBS_ISENSE] / c.ref_I0
        # Z = V / I  (I 通道以 isense 折算电流); 除去 PGA 与链路增益
        Iamp = I[:G.NOBS] / self.isense
        Z = V[:G.NOBS] / pga[:G.NOBS] / np.where(np.abs(Iamp) > 0, Iamp, 1e-12)
        Z = Z / (c.link_gain / c.link_gain_i)
        L = np.imag(Z) / self.w * 1e9
        R = np.real(Z)
        refl = L - self.carrier_L
        # 温度: 自观测 Re/Im 比 (相对首帧), ΔT = Q0·Δ(Re/Im)/α  (环项占比小, 只做相对指示)
        ratio = R[:G.NU] / np.maximum(np.abs(L[:G.NU]) * 1e-9 * self.w, 1e-12)
        if self.q_ring0 is None:
            self.q_ring0 = ratio.copy()
        # 环反射实部 ≈ (ω M)^2 R_r/|Z_r|^2; 用模型比例: dRe_ring/dT = |refl|·ω/Q_r·α → ΔT = ΔRe/(ω|refl|·α/Q_r)
        Qr = self.w * self.model.ring.L_self / self.model.ring.R
        dRe = (ratio - self.q_ring0) * np.abs(L[:G.NU]) * 1e-9 * self.w
        denom = np.maximum(np.abs(refl[:G.NU]) * 1e-9 * self.w / Qr * self.env.alpha_cu, 1e-12)
        dT = dRe / denom
        tr = self.tracker.update(L) if self.tracker is not None else None
        gap = 0
        if self._last_seq is not None:
            gap = (fr.seq - self._last_seq - 1) & 0xFFFF      # u16 回绕
            if gap > 0x7FFF:
                gap = 0
        self._last_seq = fr.seq
        if wall_t is not None:
            if self._t_last is not None and wall_t > self._t_last:
                self.fps = 0.9 * self.fps + 0.1 / (wall_t - self._t_last)
            self._t_last = wall_t
        return FrameResult(fr.seq, fr.t_ticks / self.env.fs, Z, L, refl, R, dT, sat, c.link_gain, tr, fr.truth, self.fps, gap)
