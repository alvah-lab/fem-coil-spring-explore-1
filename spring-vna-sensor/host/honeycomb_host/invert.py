"""跟踪反演: 有状态双速截断 SVD 高斯牛顿 (源自 fasthenry_runs/honeycomb/scheme_c_e2e.py).

强子空间 (57−3 维) 每帧 1~2 步阻尼 GN; 3 个弱模式 (共模 u/v + 旋转) 慢通道小步;
Jacobian 缓存, 触发式重线性化 (双缓冲); 位姿只钳位不断言.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import threading
import time
import numpy as np
from . import geometry as G
from .fastmodel import FastModel

@dataclass
class TrackerConfig:
    n_weak: int = 3
    damping: float = 0.6
    clip: float = 0.25
    slow_damping: float = 0.5
    slow_clip: float = 0.1
    slow_every: int = 8
    iters_per_frame: int = 1
    relin_every: int = 100
    relin_dq: float = 0.15
    reset_resid_factor: float = 5.0
    rel_floor: float = 1e-3
    abs_floor_nH: float = 0.005
    box_w: float = 0.75
    box_uv: float = 0.5

@dataclass
class TrackerOutput:
    q: np.ndarray
    poses: np.ndarray
    y_pred: np.ndarray
    resid: np.ndarray
    chi2: float
    sv: np.ndarray
    weak_amp: np.ndarray
    dq_norm: float
    clamped: bool
    relin: bool
    reset: bool
    ms: float
    @property
    def w(self): return self.q[:G.NU]
    @property
    def u(self): return self.q[G.NU:2 * G.NU]
    @property
    def v(self): return self.q[2 * G.NU:]

class Tracker:
    def __init__(self, model: FastModel, cfg: TrackerConfig | None = None, sig: np.ndarray | None = None, q0=None):
        self.model = model
        self.cfg = cfg or TrackerConfig()
        self.q = np.zeros(3 * G.NU) if q0 is None else np.array(q0, float)
        self.box = np.concatenate([np.full(G.NU, self.cfg.box_w), np.full(2 * G.NU, self.cfg.box_uv)])
        if sig is None:
            refl = model.observe_L(np.zeros((G.NU, 5))) - model.carrier_L()
            sig = np.maximum(self.cfg.rel_floor * np.abs(refl), self.cfg.abs_floor_nH)
        self.sig = sig
        self.frame = 0
        self.chi2_hist: list[float] = []
        self.sat_count = 0
        self.n_reset = 0; self.n_relin = 0
        self._pending = None
        self._relin_thread = None
        self.async_relin = True          # 重线性化在后台线程 (双缓冲)
        self.linearize(self.q)

    # ---- 线性化 ----
    def linearize(self, q_lin: np.ndarray) -> dict:
        """同步重线性化 (阻塞 ~0.15s)."""
        self.lin = self._compute_lin(q_lin)
        self.n_relin += 1
        return self.lin

    def _compute_lin(self, q_lin):
        J = self.model.jacobian(G.pose_of(q_lin))
        Jc = (J / self.sig[:, None]) @ G.T_S
        U, S, Vt = np.linalg.svd(Jc, full_matrices=False)
        ss = np.sort(S)
        cut = ss[self.cfg.n_weak] * 0.5
        Sinv = np.where(S > cut, 1 / np.maximum(S, 1e-30), 0.0)
        Sinv_w = np.where((S <= cut) & (S > 1e-6), 1 / np.maximum(S, 1e-30), 0.0)
        return dict(q_lin=np.array(q_lin), J=J, S=S, cut=cut, Jp=(Vt.T * Sinv) @ U.T,
                    Jp_w=(Vt.T * Sinv_w) @ U.T, weak=Vt[np.argsort(S)[:self.cfg.n_weak]])

    def request_relinearize(self, q_lin=None):
        """后台线程重线性化 (~0.15s), 完成后在下一帧 update() 原子替换; 已在算则忽略."""
        if self._relin_thread is not None and self._relin_thread.is_alive():
            return False
        q_lin = self.q.copy() if q_lin is None else np.array(q_lin)
        def work():
            self._pending = self._compute_lin(q_lin)
        self._relin_thread = threading.Thread(target=work, daemon=True); self._relin_thread.start()
        return True
    def _take_pending(self):
        if self._pending is not None:
            self.lin = self._pending; self._pending = None; self.n_relin += 1
            return True
        return False

    def reset(self):
        self.q[:] = 0.0
        self.linearize(self.q)
        self.chi2_hist.clear(); self.sat_count = 0; self.n_reset += 1

    # ---- 每帧 ----
    def update(self, y_L_nH: np.ndarray) -> TrackerOutput:
        t0 = time.perf_counter(); cfg = self.cfg
        relin = self._take_pending()
        lin = self.lin
        did_reset = False
        y = np.asarray(y_L_nH, float)
        dq_norm = 0.0; clamped_any = False
        for _ in range(cfg.iters_per_frame):
            poses, cl = G.clamp_pose(G.pose_of(self.q), self.model.cfg.gap)
            clamped_any |= bool(cl.any())
            r = (y - self.model.observe_L(poses)) / self.sig
            dq = np.clip(lin['Jp'] @ r, -cfg.clip, cfg.clip)
            self.q = np.clip(self.q + cfg.damping * dq * G.RANGE_Q, -self.box, self.box)
            dq_norm = float(np.linalg.norm(dq))
            if np.mean(np.abs(dq) >= cfg.clip * 0.999) > 0.5:
                self.sat_count += 1
            else:
                self.sat_count = 0
        if self.frame % cfg.slow_every == cfg.slow_every - 1:
            poses, _ = G.clamp_pose(G.pose_of(self.q), self.model.cfg.gap)
            r = (y - self.model.observe_L(poses)) / self.sig
            dqw = np.clip(lin['Jp_w'] @ r, -cfg.slow_clip, cfg.slow_clip)
            self.q = np.clip(self.q + cfg.slow_damping * dqw * G.RANGE_Q, -self.box, self.box)
        poses, cl = G.clamp_pose(G.pose_of(self.q), self.model.cfg.gap)
        y_pred = y - r * self.sig          # 步前预测 (省一次正演; 与当前 q 差一步)
        resid = r
        chi2 = float(np.mean(resid ** 2))
        self.chi2_hist.append(chi2)
        if len(self.chi2_hist) > 50:
            self.chi2_hist.pop(0)
        # 发散保护
        med = np.median(self.chi2_hist[:-3]) if len(self.chi2_hist) > 6 else None
        if self.sat_count >= 3 or (med is not None and med > 0 and
                                   all(c > cfg.reset_resid_factor * med for c in self.chi2_hist[-3:]) and chi2 > 25):
            self.reset(); did_reset = True; lin = self.lin
            poses, cl = G.clamp_pose(G.pose_of(self.q), self.model.cfg.gap)
            y_pred = self.model.observe_L(poses); resid = (y - y_pred) / self.sig; chi2 = float(np.mean(resid ** 2))
        # 重线性化触发
        far = np.max(np.abs((self.q - lin['q_lin']) / G.RANGE_Q)) > cfg.relin_dq
        if not did_reset and (far or (self.frame + 1) % cfg.relin_every == 0):
            if self.async_relin:
                self.request_relinearize()
            else:
                self.linearize(self.q.copy()); relin = True
        weak_amp = lin['weak'] @ (self.q / G.RANGE_Q)
        self.frame += 1
        return TrackerOutput(self.q.copy(), poses, y_pred, resid, chi2, lin['S'], weak_amp, dq_norm,
                             clamped_any or bool(cl.any()), relin, did_reset, (time.perf_counter() - t0) * 1e3)
