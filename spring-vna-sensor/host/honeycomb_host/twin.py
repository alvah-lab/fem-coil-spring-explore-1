"""数字孪生: 表面位姿场 q(t) → 61 复阻抗观测 → 驻留级 I/Q 累加值 (与 FPGA 同格式).

level 0 (默认, 实时 77Hz): 闭式累加值 + 噪声模型.
level 1 (离线): 生成 62.5MSps 样本流并做与 FPGA 相同的 NCO 累加, 用于钉死符号/常数.

累加约定 (与协议文档一致):
  v[n] = Re(V·e^{jωt_n}),  I_acc = Σ v[n] cos(2π k n/N),  Q_acc = Σ v[n] sin(2π k n/N)
  → I = N_eff/2·Re(V)/LSB,  Q = −N_eff/2·Im(V)/LSB;  复幅度 V = (I − jQ)·2·LSB/N_eff
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from . import geometry as G
from .fastmodel import FastModel, ModelConfig, obs_of

DWELL_DTYPE = np.dtype([('obs_id', 'u1'), ('dwell_word', 'u2'), ('flags', 'u1'),
                        ('V_I', 'i4'), ('V_Q', 'i4'), ('I_I', 'i4'), ('I_Q', 'i4')])
FLAG_SAT = 0x01          # ADC 饱和
FLAG_REF = 0x02          # 参考驻留
FLAG_ISENSE = 0x04       # 电流标定驻留
OBS_REF, OBS_ISENSE = 61, 62
N_DWELL = 63
ADC_FS_V = 5.0
ADC_BITS = 12
LSB = 2 * ADC_FS_V / (1 << ADC_BITS)

def dwell_word(drv: int, sns: int, pga: int = 0, ref: int = 0, vna: int = 0) -> int:
    """A704 驻留字 [drv 5][sns 5][pga 2][ref 1][vna 1][spare 2] (DUAL_FPGA_IO.md)."""
    return ((drv & 31) << 11) | ((sns & 31) << 6) | ((pga & 3) << 4) | ((ref & 1) << 3) | ((vna & 1) << 2)

def default_dwell_table() -> np.ndarray:
    """61 观测 + 参考 + 电流标定: (63,) dwell_word. PGA: self→0 (×1), edge→2 (×100), ref→1."""
    words = []
    for kind, i, j in G.OBS:
        words.append(dwell_word(i, j, 0 if kind == 'self' else 2))
    words.append(dwell_word(31, 31, 1, ref=1))
    words.append(dwell_word(31, 31, 0, vna=1))
    return np.array(words, dtype=np.uint16)

@dataclass
class NoiseModel:
    adc_sigma_V: float = 0.8e-3          # 每样本 ADC 折算噪声 (≈31µV 幅度等效/20µs)
    fe_density_V_rtHz: float = 2e-9      # 前端噪声密度 (折算到线圈端口)
    quantize: bool = True
    seed: int = 0
    preset: str = 'hardware'             # 'hardware' | 'sig2_matched' | 'off'
    rel_floor: float = 1e-3              # sig2_matched: 反射 1e-3 相对
    abs_floor_nH: float = 0.005          # sig2_matched: 5pH 地板

@dataclass
class LinkModel:
    gain0: complex = 1 + 0j
    rw_amp_per_rt_s: float = 1e-4        # 链路增益随机游走 (幅)
    rw_phase_rad_per_rt_s: float = 1e-4  # (相位)
    pga_gains: tuple = (1.0, 10.0, 100.0, 100.0)
    isense_V_per_A: float = 100.0        # Rs 10Ω × 10
    blank_nsamp: int = 312               # 开关后 5µs 作废 @62.5MSps
    drift: bool = False

@dataclass
class Environment:
    f0: float = 8e6
    fs: float = 62.5e6
    dwell_nsamp: int = 12500             # = 1600 个 8MHz 周期 (k 整数)
    i_drive_A: float = 10e-3
    gap: float = G.GAP_REST              # 静息环质心→L1
    dT_ring_K: np.ndarray = field(default_factory=lambda: np.zeros(G.NU))
    dT_coil_K: np.ndarray = field(default_factory=lambda: np.zeros(G.NU))
    alpha_cu: float = 3.9e-3
    c_off_F: np.ndarray = field(default_factory=lambda: np.full(G.NU, 125e-15))
    coff_enable: bool = True
    noise: NoiseModel = field(default_factory=NoiseModel)
    link: LinkModel = field(default_factory=LinkModel)
    switch_us: float = 5.0               # 每驻留开关+稳定
    @property
    def nco_word(self) -> int:
        return int(round(self.f0 / self.fs * 2 ** 32)) & 0xFFFFFFFF
    @property
    def frame_period_s(self) -> float:
        return N_DWELL * (self.dwell_nsamp / self.fs + self.switch_us * 1e-6)

@dataclass
class TwinTruth:
    t: float
    q: np.ndarray                        # (57,) mm
    poses: np.ndarray                    # (19,5)
    clamped: np.ndarray
    z61: np.ndarray                      # 复阻抗 (无 C_off)
    z61_coff: np.ndarray                 # 含 C_off 干扰
    link_gain: complex
    L61_nH: np.ndarray                   # Im/ω 真值 (无 C_off)

@dataclass
class Frame:
    seq: int
    t_ticks: int
    dwell_nsamp: int
    nco_word: int
    dwells: np.ndarray                   # (63,) DWELL_DTYPE
    frame_id: int = 0
    flags: int = 0
    truth: TwinTruth | None = None

# ---------------- 场景 ----------------
def _gauss_w(x0, y0, depth, sigma):
    r2 = (G.XY[:, 0] - x0) ** 2 + (G.XY[:, 1] - y0) ** 2
    return depth * np.exp(-r2 / (2 * sigma ** 2))

@dataclass
class Scene:
    name: str
    q_of_t: Callable[[float], np.ndarray]   # t → q(57) mm
    duration_s: float = 1e9
    dT_of_t: Callable[[float], np.ndarray] | None = None

class Scenes:
    @staticmethod
    def rest():
        return Scene('rest', lambda t: np.zeros(3 * G.NU))
    @staticmethod
    def point_press(x0=0.0, y0=0.0, depth=-0.4, sigma=G.PITCH * 1.2, shear=(0.0, 0.0), ramp_s=0.5):
        def q(t):
            a = min(1.0, t / ramp_s) if ramp_s > 0 else 1.0
            w = _gauss_w(x0, y0, depth, sigma) * a
            return np.concatenate([w, np.full(G.NU, shear[0] * a), np.full(G.NU, shear[1] * a)])
        return Scene('point_press', q)
    @staticmethod
    def shear_cm(u=0.15, v=0.0, w0=-0.2):
        return Scene('shear_cm', lambda t: np.concatenate([np.full(G.NU, w0), np.full(G.NU, u), np.full(G.NU, v)]))
    @staticmethod
    def tilt(ax=0.02, ay=0.0, w0=-0.2):
        w = w0 + ax * G.XY[:, 1] - ay * G.XY[:, 0]
        return Scene('tilt', lambda t: np.concatenate([w, np.zeros(G.NU), np.zeros(G.NU)]))
    @staticmethod
    def sweep(depth=-0.4, sigma=G.PITCH, speed=8.0, period_s=3.0):
        def q(t):
            x0 = -10 + 20 * ((t * speed / 20.0) % 1.0)
            return np.concatenate([_gauss_w(x0, 0.0, depth, sigma), np.zeros(2 * G.NU)])
        return Scene('sweep', q)
    @staticmethod
    def impact(depth=-0.5, t0=0.5, tau=0.05, sigma=G.PITCH):
        def q(t):
            a = np.exp(-(t - t0) / tau) if t >= t0 else 0.0
            return np.concatenate([_gauss_w(0, 0, depth, sigma) * a, np.zeros(2 * G.NU)])
        return Scene('impact', q)
    @staticmethod
    def thermal_touch(dT=10.0, t0=0.5, tau=1.0, sigma=G.PITCH):
        def q(t):
            return np.concatenate([_gauss_w(0, 0, -0.1, sigma), np.zeros(2 * G.NU)])
        def dT_of_t(t):
            return _gauss_w(0, 0, dT, sigma) * (1 - np.exp(-(t - t0) / tau)) if t >= t0 else np.zeros(G.NU)
        return Scene('thermal_touch', q, dT_of_t=dT_of_t)
    ALL = ('rest', 'point_press', 'shear_cm', 'tilt', 'sweep', 'impact', 'thermal_touch')

# ---------------- 孪生 ----------------
class Twin:
    def __init__(self, env: Environment | None = None, model: FastModel | None = None,
                 scene: Scene | None = None, dwell_table: np.ndarray | None = None):
        self.env = env or Environment()
        self.model = model or FastModel(cfg=ModelConfig(f0=self.env.f0, gap=self.env.gap))
        self.scene = scene or Scenes.rest()
        self.dwell_table = default_dwell_table() if dwell_table is None else dwell_table
        self.rng = np.random.default_rng(self.env.noise.seed)
        self.seq = 0
        self.t = 0.0
        self.link_gain = complex(self.env.link.gain0)
        self.link_gain_i = complex(self.env.link.gain0)
        self.w = 2 * np.pi * self.env.f0
        self.V_ref0 = 0.3 + 0j                       # 参考驻留幅度 (V, 无 PGA)
        self.I_ref0 = self.env.i_drive_A
        self.carrier_Z = self.model.carrier_Z()
        self.sig_nH = self.sigma_model()

    # ---- 噪声 (sig2_matched 用) ----
    def sigma_model(self) -> np.ndarray:
        refl = self.model.observe_L(np.zeros((G.NU, 5))) - self.model.carrier_L()
        n = self.env.noise
        return np.maximum(n.rel_floor * np.abs(refl), n.abs_floor_nH)

    # ---- 物理 ----
    def z61(self, q: np.ndarray, dT_ring=None, dT_coil=None):
        env = self.env
        poses, clamped = G.clamp_pose(G.pose_of(q), env.gap)
        Rr = self.model.ring.R * (1 + env.alpha_cu * (env.dT_ring_K if dT_ring is None else dT_ring))
        Rc = self.model.coil.R * (1 + env.alpha_cu * (env.dT_coil_K if dT_coil is None else dT_coil))
        Z = self.model.fold(*self.model.blocks(poses), Rr, Rc)
        z = obs_of(Z)
        zc = self.coff_fold(Z) if env.coff_enable else z
        return z, zc, poses, clamped

    def coff_fold(self, Z: np.ndarray) -> np.ndarray:
        """逐驻留精确折叠: 非活动线圈以 1/(jωC_off) 端接. Z_AA,eff = Z_AA − Z_AK (Z_KK + diag(Zt))^-1 Z_KA."""
        w = self.w
        Zt = 1 / (1j * w * self.env.c_off_F)
        out = np.empty(G.NOBS, complex)
        for n, (kind, i, j) in enumerate(G.OBS):
            A = [i] if kind == 'self' else [i, j]
            K = [k for k in range(G.NU) if k not in A]
            ZKK = Z[np.ix_(K, K)] + np.diag(Zt[K])
            Zeff = Z[np.ix_(A, A)] - Z[np.ix_(A, K)] @ np.linalg.solve(ZKK, Z[np.ix_(K, A)])
            out[n] = Zeff[0, 0] if kind == 'self' else Zeff[0, 1]
        return out

    # ---- 驻留级累加 ----
    def accumulate(self, V: np.ndarray, Ich: np.ndarray, n_eff: int, sat: np.ndarray | None = None):
        """复电压 (63,) → I/Q 累加值 int32 (V 通道, I 通道)."""
        nm = self.env.noise
        N = n_eff
        def acc(x):
            I = N / 2 * np.real(x) / LSB
            Q = -N / 2 * np.imag(x) / LSB
            if nm.preset == 'hardware':
                s_v = np.hypot(nm.adc_sigma_V, nm.fe_density_V_rtHz * np.sqrt(self.env.fs / 2))
                s = s_v * np.sqrt(N / 2) / LSB
                I = I + self.rng.standard_normal(len(x)) * s
                Q = Q + self.rng.standard_normal(len(x)) * s
            return I, Q
        VI, VQ = acc(V); II, IQ = acc(Ich)
        if nm.quantize:
            VI, VQ, II, IQ = (np.rint(a) for a in (VI, VQ, II, IQ))
        lim = 2 ** 31 - 1
        return (np.clip(VI, -lim, lim).astype(np.int32), np.clip(VQ, -lim, lim).astype(np.int32),
                np.clip(II, -lim, lim).astype(np.int32), np.clip(IQ, -lim, lim).astype(np.int32))

    def step(self, dt: float | None = None) -> Frame:
        env = self.env
        dt = env.frame_period_s if dt is None else dt
        q = np.asarray(self.scene.q_of_t(self.t), float)
        dT = self.scene.dT_of_t(self.t) if self.scene.dT_of_t else None
        z, zc, poses, clamped = self.z61(q, dT)
        # 链路漂移
        if env.link.drift:
            g = self.link_gain
            g *= (1 + env.link.rw_amp_per_rt_s * np.sqrt(dt) * self.rng.standard_normal()) * \
                 np.exp(1j * env.link.rw_phase_rad_per_rt_s * np.sqrt(dt) * self.rng.standard_normal())
            self.link_gain = g
        pga_idx = (self.dwell_table >> 4) & 3
        gpga = np.array(env.link.pga_gains)[pga_idx]
        # 61 观测: V = Z·I·G·PGA ; 参考; 电流标定
        V = np.empty(N_DWELL, complex); Ich = np.empty(N_DWELL, complex)
        zsig = zc if env.coff_enable else z
        if env.noise.preset == 'sig2_matched':
            # 在阻抗域加噪声: ΔL ~ N(0, sig_nH) → ΔZ = jωΔL
            zsig = zsig + 1j * self.w * self.sig_nH * 1e-9 * self.rng.standard_normal(G.NOBS)
        V[:G.NOBS] = zsig * env.i_drive_A * self.link_gain * gpga[:G.NOBS]
        Ich[:G.NOBS] = env.i_drive_A * self.link_gain_i * env.link.isense_V_per_A
        V[OBS_REF] = self.V_ref0 * self.link_gain * gpga[OBS_REF]
        Ich[OBS_REF] = self.I_ref0 * self.link_gain_i * env.link.isense_V_per_A
        V[OBS_ISENSE] = 0.0
        Ich[OBS_ISENSE] = self.I_ref0 * self.link_gain_i * env.link.isense_V_per_A
        sat = (np.abs(V) > ADC_FS_V) | (np.abs(Ich) > ADC_FS_V)
        n_eff = env.dwell_nsamp - env.link.blank_nsamp
        VI, VQ, II, IQ = self.accumulate(V, Ich, n_eff)
        d = np.zeros(N_DWELL, DWELL_DTYPE)
        d['obs_id'] = np.arange(N_DWELL); d['dwell_word'] = self.dwell_table
        d['flags'] = sat.astype(np.uint8) * FLAG_SAT
        d['flags'][OBS_REF] |= FLAG_REF; d['flags'][OBS_ISENSE] |= FLAG_ISENSE
        d['V_I'], d['V_Q'], d['I_I'], d['I_Q'] = VI, VQ, II, IQ
        truth = TwinTruth(self.t, q, poses, clamped, z, zc, self.link_gain, np.imag(z) / self.w * 1e9)
        fr = Frame(self.seq, int(round(self.t * env.fs)), env.dwell_nsamp, env.nco_word, d, self.seq, 0, truth)
        self.seq += 1; self.t += dt
        return fr

    # ---- level 1 (离线) ----
    def render_samples(self, V: np.ndarray, n: int | None = None) -> np.ndarray:
        """复幅度 (m,) → 样本流 (m, N) int (ADC 码, 含量化与噪声)."""
        N = self.env.dwell_nsamp if n is None else n
        k = self.env.f0 / self.env.fs
        t = np.arange(N)
        v = np.real(V[:, None] * np.exp(1j * 2 * np.pi * k * t)[None, :])
        if self.env.noise.preset == 'hardware':
            v = v + self.rng.standard_normal(v.shape) * self.env.noise.adc_sigma_V
        return np.clip(np.rint(v / LSB), -2048, 2047).astype(np.int16)

    def nco_accumulate(self, samples: np.ndarray, blank: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """FPGA 等价: 消隐前 blank 个样本, 累加 Σ v cos, Σ v sin."""
        N = samples.shape[1]
        k = self.env.f0 / self.env.fs
        t = np.arange(N)
        c = np.cos(2 * np.pi * k * t); s = np.sin(2 * np.pi * k * t)
        c[:blank] = 0; s[:blank] = 0
        return samples @ c, samples @ s

def decode_dwells(d: np.ndarray, n_eff: int) -> tuple[np.ndarray, np.ndarray]:
    """累加值 → 复幅度 V, I (伏). V = (I − jQ)·2·LSB/N_eff."""
    V = (d['V_I'].astype(float) - 1j * d['V_Q'].astype(float)) * 2 * LSB / n_eff
    I = (d['I_I'].astype(float) - 1j * d['I_Q'].astype(float)) * 2 * LSB / n_eff
    return V, I
