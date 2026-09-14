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
    box_w: float = 1.6          # |w| 上限: 静息 2.53 → 绝对间隙 1.0..3.0 (clamp_pose 再按绝对间隙钳位)
    box_uv: float = 0.5         # |u|,|v| 上限 (柔性模式; 刚性模式用 0.6 容纳标定板 0.5 偏移)

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
        self._gen = 0                    # 线性化代数: 掩码切换后, 后台算出的旧代数结果作废
        self.async_relin = True          # 重线性化在后台线程 (双缓冲)
        self.present = np.ones(G.NU, bool)   # 环在位掩码 (回板阶段部分放环)
        self.rigid = False                   # 刚性模式: 位姿 = q 直接映射 (无柔性面先验), 刚性标定板/部分放环用
        self._frozen = np.zeros(3 * G.NU, bool)
        self.Rr = None                       # None = 模型默认; 缺席环 1e12 Ω
        self.linearize(self.q)

    # ---- 部分放环 ----
    def set_present(self, mask) -> bool:
        """更新环在位掩码: 缺席环从网络移除 (R→∞), 其位姿冻结为 0, 同步重线性化. 掩码未变返回 False."""
        mask = np.asarray(mask, bool)
        if mask.shape != (G.NU,) or np.array_equal(mask, self.present):
            return False
        self.present = mask.copy()
        self.Rr = None if mask.all() else np.where(mask, self.model.ring.R, 1e12)
        self._frozen = np.tile(~mask, 3)
        self.q[:] = 0.0                      # 换板/换取向: 全部位姿从 0 重新收敛 (沿用旧值会离线性化点太远而走过头)
        self._gen += 1; self._pending = None
        self.linearize(self.q)
        self.chi2_hist.clear(); self.sat_count = 0
        return True

    def set_rigid(self, rigid: bool) -> bool:
        """切换柔性先验 / 刚性映射; 变化时同步重线性化."""
        rigid = bool(rigid)
        if rigid == self.rigid:
            return False
        self.rigid = rigid
        self.box = np.concatenate([np.full(G.NU, self.cfg.box_w), np.full(2 * G.NU, 0.6 if rigid else self.cfg.box_uv)])
        self._gen += 1; self._pending = None
        self.linearize(self.q)
        self.chi2_hist.clear(); self.sat_count = 0
        return True

    def _pose_of(self, q):
        return G.pose_of_rigid(q) if self.rigid else G.pose_of(q)

    def _observe(self, poses):
        return self.model.observe_L(poses) if self.Rr is None else self.model.observe_L(poses, self.Rr)

    # ---- 线性化 ----
    def linearize(self, q_lin: np.ndarray) -> dict:
        """同步重线性化 (阻塞 ~0.15s)."""
        self.lin = self._compute_lin(q_lin)
        self.n_relin += 1
        return self.lin

    def _compute_lin(self, q_lin):
        J = self.model.jacobian(self._pose_of(q_lin), R_ring=self.Rr)
        Jc = (J / self.sig[:, None]) @ (G.T_S_RIGID if self.rigid else G.T_S)
        if not self.present.all():
            Jc[:, np.tile(~self.present, 3)] = 0.0     # 缺席环的 (w,u,v) 冻结
        U, S, Vt = np.linalg.svd(Jc, full_matrices=False)
        ss = np.sort(S)
        nz = ss[ss > 1e-9 * max(ss[-1], 1e-30)]        # 冻结 DOF 的零奇异值不参与弱模式判定
        cut = (nz[self.cfg.n_weak] if len(nz) > self.cfg.n_weak else 0.0) * 0.5
        eps = 1e-9 * max(S.max(), 1e-30)            # 冻结 DOF 的数值零奇异值 (~1e-13) 不能取倒数
        Sinv = np.where(S > max(cut, eps), 1 / np.maximum(S, 1e-30), 0.0)
        Sinv_w = np.where((S <= cut) & (S > max(1e-6, eps)), 1 / np.maximum(S, 1e-30), 0.0)
        return dict(q_lin=np.array(q_lin), J=J, S=S, cut=cut, Jp=(Vt.T * Sinv) @ U.T,
                    Jp_w=(Vt.T * Sinv_w) @ U.T, weak=Vt[np.argsort(S)[:self.cfg.n_weak]])

    def request_relinearize(self, q_lin=None):
        """后台线程重线性化 (~0.15s), 完成后在下一帧 update() 原子替换; 已在算则忽略."""
        if self._relin_thread is not None and self._relin_thread.is_alive():
            return False
        q_lin = self.q.copy() if q_lin is None else np.array(q_lin)
        gen = self._gen
        def work():
            lin = self._compute_lin(q_lin)
            if gen == self._gen:          # 掩码没变才可用
                self._pending = lin
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
            poses, cl = G.clamp_pose(self._pose_of(self.q), self.model.cfg.gap)
            clamped_any |= bool(cl.any())
            r = (y - self._observe(poses)) / self.sig
            chi2_cur = float(np.mean(r ** 2))
            dq = np.clip(lin['Jp'] @ r, -cfg.clip, cfg.clip)
            # 回溯: 试探步使 χ² 明显变差 (线性化点太远/过冲) 则减半, 最多 3 次
            step = cfg.damping
            n_try = 3 if chi2_cur > 50 else 1        # 接近收敛 (噪声主导) 时不回溯, 省正演
            for attempt in range(n_try):
                q_new = np.clip(self.q + step * dq * G.RANGE_Q, -self.box, self.box)
                q_new[self._frozen] = 0.0
                p_new, _ = G.clamp_pose(self._pose_of(q_new), self.model.cfg.gap)
                r_new = (y - self._observe(p_new)) / self.sig
                if float(np.mean(r_new ** 2)) <= chi2_cur * 1.05 or attempt == n_try - 1:
                    break
                step *= 0.5
            if step < cfg.damping and self.async_relin:
                self.request_relinearize()          # 线性化点已过时
            self.q = q_new
            r = r_new
            dq_norm = float(np.linalg.norm(dq)) * step / cfg.damping
            if np.mean(np.abs(dq) >= cfg.clip * 0.999) > 0.5:
                self.sat_count += 1
            else:
                self.sat_count = 0
        if self.frame % cfg.slow_every == cfg.slow_every - 1:
            poses, _ = G.clamp_pose(self._pose_of(self.q), self.model.cfg.gap)
            r = (y - self._observe(poses)) / self.sig
            dqw = np.clip(lin['Jp_w'] @ r, -cfg.slow_clip, cfg.slow_clip)
            self.q = np.clip(self.q + cfg.slow_damping * dqw * G.RANGE_Q, -self.box, self.box)
            self.q[self._frozen] = 0.0
        poses, cl = G.clamp_pose(self._pose_of(self.q), self.model.cfg.gap)
        y_pred = y - r * self.sig          # 步前预测 (省一次正演; 与当前 q 差一步)
        resid = r
        chi2 = float(np.mean(resid ** 2))
        self.chi2_hist.append(chi2)
        if len(self.chi2_hist) > 50:
            self.chi2_hist.pop(0)
        # 发散保护
        med = np.median(self.chi2_hist[:-3]) if len(self.chi2_hist) > 6 else None
        if (med is not None and med > 0 and
                                   all(c > cfg.reset_resid_factor * med for c in self.chi2_hist[-3:]) and chi2 > 25):
            self.reset(); did_reset = True; lin = self.lin
            poses, cl = G.clamp_pose(self._pose_of(self.q), self.model.cfg.gap)
            y_pred = self._observe(poses); resid = (y - y_pred) / self.sig; chi2 = float(np.mean(resid ** 2))
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
