"""毫秒级正演: 轴对称矢势表 + 线积分.

每个源 (线圈 = 一组共轴圆环; 铜环 = 3 根共轴丝) 对自身轴对称, 单位电流矢势 A_φ(ρ,z) 由
Maxwell 椭圆积分公式打表; 任意目标闭合折线的互感 M = ∮ A_φ φ̂·dl (目标点变换到源体坐标).
5-DOF 位姿精确, 只有源截面被表化. 自感为常数 (L_coil, L_ring), 不经线积分.

绝对尺度对齐 track3-kicad-1/sim/spiral_reflection_check.py (Neumann, 双方采纳: 自反射 62/22nH@1.75mm);
`neumann_direct()` 为该内核的逐字复制, 仅测试用.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import ellipk, ellipe
from . import geometry as G

MU0 = 4e-7 * np.pi

# ---------------- 规格 ----------------
@dataclass(frozen=True)
class CoilSpec:
    """双层平面螺旋 → 共轴圆环组 (r_mm, dz_mm, weight). 冻结: 11.72 匝/层, 匝距 0.178, 外圈 2.4555, 内圈 0.37."""
    r_out: float = 2.5 - 0.089 / 2
    r_in: float = 0.37
    pitch: float = 0.178
    layer_dz: float = G.COIL_DZ          # 第二层相对顶层 (-0.10)
    L_self: float = 1.11e-6              # spiral_geom --check (Neumann) 值
    R: float = 1.2                       # Ω @8MHz
    def loops(self) -> np.ndarray:
        """(n,3): r, dz, weight — 整匝权重 1, 末匝按分数匝."""
        turns = (self.r_out - self.r_in) / self.pitch
        n_full = int(np.floor(turns)); frac = turns - n_full
        rs = self.r_out - self.pitch * np.arange(n_full + 1)
        w = np.ones(n_full + 1); w[-1] = frac
        out = []
        for dz in (0.0, self.layer_dz):
            for r, ww in zip(rs, w):
                if ww > 1e-6:
                    out.append((r, dz, ww))
        return np.array(out)

@dataclass(frozen=True)
class RingSpec:
    """冲压紫铜环 Ø5/Ø3/0.2 → 3 根同心丝 (电流均分, M 取平均)."""
    filaments: tuple = (1.6, 2.0, 2.4)
    L_self: float = 5.25e-9              # GMD 公式 (spiral_reflection_check)
    R: float = 6e-3                      # Ω @8MHz
    nseg: int = 72

@dataclass(frozen=True)
class ModelConfig:
    f0: float = 8e6
    gap: float = G.GAP_NOM               # 标称环质心→L1 (pose dz 相对此值)
    table_h: float = 0.02                # 表步长 mm
    rho_max: float = 27.0
    neighbor_pitches: float = 3.2        # 只算此距离内的 环-环 / 线圈-环 对

# ---------------- 矢势表 ----------------
def a_phi_loop(a: float, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
    """半径 a 圆环单位电流的 A_φ, 长度全用 mm (最终 M=ΣA·dl 再乘 1e-3 化为 H, 与 neumann() 同约定)."""
    rho = np.maximum(rho, 1e-6)
    k2 = 4 * a * rho / ((a + rho) ** 2 + z ** 2)
    k2 = np.clip(k2, 0.0, 1.0 - 1e-12)
    k = np.sqrt(k2)
    return MU0 / np.pi * np.sqrt(a / rho) / k * ((1 - k2 / 2) * ellipk(k2) - ellipe(k2))

class ATable:
    """A_φ(ρ,z) 表 (float32), 源 = 若干共轴圆环 (r, dz, w) 的加权和."""
    def __init__(self, loops: np.ndarray, zmin: float, zmax: float, h: float = 0.02, rho_max: float = 27.0):
        self.h = h; self.zmin = zmin; self.zmax = zmax
        self.rho = np.arange(0.0, rho_max + h, h)
        self.z = np.arange(zmin, zmax + h, h)
        RHO, Z = np.meshgrid(self.rho, self.z, indexing='ij')
        A = np.zeros_like(RHO)
        for r, dz, w in loops:
            A += w * a_phi_loop(r, RHO, Z - dz)
        self.A = A.astype(np.float32)
    @classmethod
    def from_arrays(cls, A, zmin, zmax, h, rho_max):
        t = cls.__new__(cls); t.h = h; t.zmin = zmin; t.zmax = zmax; t.A = A
        t.rho = np.arange(0.0, rho_max + h, h); t.z = np.arange(zmin, zmax + h, h); return t
    def line_integral(self, pts: np.ndarray) -> np.ndarray:
        """pts (..., n+1, 3) 目标闭合折线 (源体坐标, mm) → M (...) [H]."""
        mid = 0.5 * (pts[..., :-1, :] + pts[..., 1:, :])
        dl = pts[..., 1:, :] - pts[..., :-1, :]
        rho = np.hypot(mid[..., 0], mid[..., 1])
        phidl = (-mid[..., 1] * dl[..., 0] + mid[..., 0] * dl[..., 1]) / np.maximum(rho, 1e-9)
        ci = np.stack([rho.ravel() / self.h, (mid[..., 2].ravel() - self.zmin) / self.h])
        A = map_coordinates(self.A, ci, order=1, mode='nearest').reshape(rho.shape)
        return np.sum(A * phidl, axis=-1) * 1e-3

# ---------------- 几何 ----------------
def rot_matrix(tax: float, tay: float) -> np.ndarray:
    """与 gen_ring_array.ring_nodes 一致: 先绕 x 转 tax, 再绕 y 转 tay."""
    cx, sx = np.cos(tax), np.sin(tax); cy, sy = np.cos(tay), np.sin(tay)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx

def circle_local(r: float, nseg: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, nseg + 1)
    return np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], 1)

def obs_of(Z: np.ndarray) -> np.ndarray:
    """(19,19) → 61 观测 (19 自 + 42 边, EDGES 字典序)."""
    return np.concatenate([np.diag(Z), Z[G.EDGE_IDX[:, 0], G.EDGE_IDX[:, 1]]])

def neumann(pa: np.ndarray, pb: np.ndarray) -> float:
    """两条闭合折线互感 (H), 坐标 mm (spiral_reflection_check.neumann 逐字)."""
    da = np.diff(pa, axis=0); db = np.diff(pb, axis=0)
    ma = (pa[:-1] + pa[1:]) / 2; mb = (pb[:-1] + pb[1:]) / 2
    d = np.linalg.norm(ma[:, None, :] - mb[None, :, :], axis=2)
    return MU0 / (4 * np.pi) * np.sum((da @ db.T) / d) * 1e-3

# ---------------- 模型 ----------------
class FastModel:
    ZC = (0.2, 5.5)      # 线圈表 z 范围 (目标环相对线圈顶层)
    ZR = (-2.2, 2.2)     # 环表 z 范围

    def __init__(self, coil: CoilSpec = CoilSpec(), ring: RingSpec = RingSpec(),
                 cfg: ModelConfig = ModelConfig(), cache_dir: str | None = None):
        self.coil, self.ring, self.cfg = coil, ring, cfg
        self.w = 2 * np.pi * cfg.f0
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'data')
        self._coil_loops = coil.loops()
        self._local_ring = np.stack([circle_local(r, ring.nseg) for r in ring.filaments])   # (nf, n+1, 3) 环-环
        self._local_ring_c = np.stack([circle_local(r, ring.nseg // 2) for r in ring.filaments])  # 线圈-环 (36 段够)
        d = np.linalg.norm(G.XY[:, None, :] - G.XY[None, :, :], axis=2)
        self.near = d < cfg.neighbor_pitches * G.PITCH
        self._build_tables()
        self.Mcc = self._load_or_build('Mcc', self._coil_coil_neumann)
        self.Mcc[np.diag_indices(G.NU)] = coil.L_self

    # ---- 缓存 ----
    def _key(self, tag):
        s = repr((self.coil, self.ring, self.cfg.table_h, self.cfg.rho_max, self.cfg.neighbor_pitches, tag))
        return hashlib.md5(s.encode()).hexdigest()[:10]
    def _load_or_build(self, tag, builder):
        path = os.path.join(self.cache_dir, f'fastmodel_{tag}_{self._key(tag)}.npy')
        if os.path.exists(path):
            return np.load(path)
        arr = builder()
        try:
            np.save(path, arr)
        except OSError:
            pass
        return arr
    def _build_tables(self):
        cfg = self.cfg
        path = os.path.join(self.cache_dir, f'fastmodel_tables_{self._key("tab")}.npz')
        if os.path.exists(path):
            d = np.load(path)
            self.tab_coil = ATable.from_arrays(d['c_A'], *self.ZC, cfg.table_h, cfg.rho_max)
            self.tab_ring = ATable.from_arrays(d['r_A'], *self.ZR, cfg.table_h, cfg.rho_max)
            return
        self.tab_coil = ATable(self._coil_loops, *self.ZC, cfg.table_h, cfg.rho_max)
        fil = np.array([(rr, 0.0, 1.0 / len(self.ring.filaments)) for rr in self.ring.filaments])
        self.tab_ring = ATable(fil, *self.ZR, cfg.table_h, cfg.rho_max)
        try:
            np.savez_compressed(path, c_A=self.tab_coil.A, r_A=self.tab_ring.A)
        except OSError:
            pass

    # ---- 几何 ----
    def ring_polys(self, poses: np.ndarray, local=None) -> np.ndarray:
        """poses (19,5) → 世界坐标环折线 (19, nf, n+1, 3). 环质心 z = gap + dz."""
        local = self._local_ring if local is None else local
        out = np.empty((G.NU,) + local.shape)
        for u in range(G.NU):
            dx, dy, dz, tax, tay = poses[u]
            R = rot_matrix(tax, tay)
            c = np.array([G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.cfg.gap + dz])
            out[u] = local @ R.T + c
        return out

    def _coil_polys(self, nseg):
        return {v: [(circle_local(r, nseg) + np.array([G.XY[v, 0], G.XY[v, 1], G.COIL_Z0[v] + dz]), w)
                    for r, dz, w in self._coil_loops] for v in range(G.NU)}

    def _coil_coil_neumann(self) -> np.ndarray:
        """线圈-线圈互感 (常数块, 一次性直接 Neumann, 36 段)."""
        polys = self._coil_polys(36)
        M = np.zeros((G.NU, G.NU))
        for u in range(G.NU):
            for v in range(u + 1, G.NU):
                if not self.near[u, v]:
                    continue
                s = 0.0
                for pa, wa in polys[u]:
                    for pb, wb in polys[v]:
                        s += wa * wb * neumann(pa, pb)
                M[u, v] = M[v, u] = s
        return M

    # ---- 位姿相关块 ----
    def blocks(self, poses: np.ndarray, rp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """→ (Mcr (19c,19r) H, Mrr (19,19) H 对角=L_ring). 只算邻域内的对."""
        rp = self.ring_polys(poses) if rp is None else rp
        rpc = self.ring_polys(poses, self._local_ring_c)
        Mcr = np.zeros((G.NU, G.NU)); Mrr = np.zeros((G.NU, G.NU))
        for u in range(G.NU):
            idx = np.where(self.near[u])[0]
            loc = rpc[idx] - np.array([G.XY[u, 0], G.XY[u, 1], G.COIL_Z0[u]])
            Mcr[u, idx] = self.tab_coil.line_integral(loc).mean(axis=1)
        for u in range(G.NU):
            idx = np.array([v for v in range(u + 1, G.NU) if self.near[u, v]])
            if len(idx) == 0:
                continue
            dx, dy, dz, tax, tay = poses[u]
            R = rot_matrix(tax, tay)
            c = np.array([G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.cfg.gap + dz])
            loc = (rp[idx] - c) @ R
            m = self.tab_ring.line_integral(loc).mean(axis=1)
            Mrr[u, idx] = m; Mrr[idx, u] = m
        Mrr[np.diag_indices(G.NU)] = self.ring.L_self
        return Mcr, Mrr

    def fold(self, Mcr, Mrr, R_ring=None, R_coil=None) -> np.ndarray:
        """Zeff (19,19) complex = R_c + jω L_cc + ω² M_cr (R_r + jω L_rr)^-1 M_rc."""
        w = self.w
        Rr = np.full(G.NU, self.ring.R) if R_ring is None else np.asarray(R_ring, float)
        Rc = np.full(G.NU, self.coil.R) if R_coil is None else np.asarray(R_coil, float)
        Zr = np.diag(Rr) + 1j * w * Mrr
        return np.diag(Rc) + 1j * w * self.Mcc + w * w * (Mcr @ np.linalg.solve(Zr, Mcr.T))

    def observe_Z(self, poses, R_ring=None, R_coil=None) -> np.ndarray:
        """61 复阻抗观测 (钳位后)."""
        poses, _ = G.clamp_pose(poses, self.cfg.gap)
        return obs_of(self.fold(*self.blocks(poses), R_ring, R_coil))

    def observe_L(self, poses) -> np.ndarray:
        """61 观测有效电感 Im(Z)/ω, nH (反演拟合量)."""
        return np.imag(self.observe_Z(poses)) / self.w * 1e9

    def carrier_Z(self) -> np.ndarray:
        """无环反射的固定载波观测 (61,) complex."""
        return obs_of(np.diag(np.full(G.NU, self.coil.R)) + 1j * self.w * self.Mcc)

    def carrier_L(self) -> np.ndarray:
        return np.imag(self.carrier_Z()) / self.w * 1e9

    def update_ring(self, Mcr, Mrr, u: int, poses, rp_u=None):
        """环 u 位姿变化后原地更新 Mcr[:,u] 与 Mrr[u,:] (其余不变). 返回 (Mcr, Mrr) 新副本."""
        Mcr = Mcr.copy(); Mrr = Mrr.copy()
        if rp_u is None:
            dx, dy, dz, tax, tay = poses[u]
            R = rot_matrix(tax, tay)
            c = np.array([G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.cfg.gap + dz])
            rp_u = self._local_ring @ R.T + c
        else:
            dx, dy, dz, tax, tay = poses[u]; R = rot_matrix(tax, tay)
            c = np.array([G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.cfg.gap + dz])
        idx = np.where(self.near[u])[0]
        # 线圈 v (源) → 环 u (目标, 36 段)
        rpc_u = self._local_ring_c @ R.T + c
        loc = rpc_u[None] - np.stack([G.XY[idx, 0], G.XY[idx, 1], G.COIL_Z0[idx]], 1)[:, None, None, :]
        Mcr[idx, u] = self.tab_coil.line_integral(loc).mean(axis=1)
        # 环 u (源) → 环 v (目标)
        idx2 = idx[idx != u]
        rp_v = self.ring_polys_subset(poses, idx2)
        loc = (rp_v - c) @ R
        m = self.tab_ring.line_integral(loc).mean(axis=1)
        Mrr[u, idx2] = m; Mrr[idx2, u] = m
        return Mcr, Mrr

    def ring_polys_subset(self, poses, idx):
        out = np.empty((len(idx),) + self._local_ring.shape)
        for k, u in enumerate(idx):
            dx, dy, dz, tax, tay = poses[u]
            R = rot_matrix(tax, tay)
            c = np.array([G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.cfg.gap + dz])
            out[k] = self._local_ring @ R.T + c
        return out

    def jacobian(self, poses=None, delta: dict = G.DELTA) -> np.ndarray:
        """(61,95) nH / 满量程, 中心差分, 逐环行更新 (~0.15s)."""
        poses = np.zeros((G.NU, 5)) if poses is None else np.array(poses, float)
        Mcr0, Mrr0 = self.blocks(poses)
        J = np.zeros((G.NOBS, G.NU * 5))
        for u in range(G.NU):
            for d, dof in enumerate(G.DOFS):
                dd = delta[dof]
                pp = poses.copy(); pp[u, d] += dd
                pm = poses.copy(); pm[u, d] -= dd
                yp = np.imag(obs_of(self.fold(*self.update_ring(Mcr0, Mrr0, u, pp))))
                ym = np.imag(obs_of(self.fold(*self.update_ring(Mcr0, Mrr0, u, pm))))
                J[:, u * 5 + d] = (yp - ym) / self.w * 1e9 / (2 * dd) * G.RANGE[dof]
        return J

    # ---- 参考内核 (测试用, 秒级) ----
    def neumann_direct(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rp = self.ring_polys(poses)
        cp = self._coil_polys(72)
        nf = rp.shape[1]
        Mcr = np.zeros((G.NU, G.NU)); Mrr = np.zeros((G.NU, G.NU))
        for u in range(G.NU):
            for v in range(G.NU):
                if self.near[u, v]:
                    Mcr[u, v] = np.mean([sum(w * neumann(pa, rp[v, f]) for pa, w in cp[u]) for f in range(nf)])
        for u in range(G.NU):
            for v in range(u + 1, G.NU):
                if self.near[u, v]:
                    m = np.mean([neumann(rp[u, f], rp[v, g]) for f in range(nf) for g in range(nf)])
                    Mrr[u, v] = Mrr[v, u] = m
        Mrr[np.diag_indices(G.NU)] = self.ring.L_self
        return Mcr, Mrr
