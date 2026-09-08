"""几何与观测定义 —— 单一来源.

常量复制自 (不 import, 因为那些脚本导入即跑 FastHenry):
  fasthenry_runs/honeycomb/pcb_verdict.py   L21-37  (lattice / ADJ / EDGES / CI / OBS / RANGE / DELTA)
  fasthenry_runs/honeycomb/series_scheme.py L118-144 (柔性先验 T_phys / T_s / RANGE_q)
  track3-kicad-1/sim/spiral_reflection_check.py (线圈层高约定: 环质心离 L1 1.75mm)
坐标: mm; 角度: rad. 单元 id 顺序 = lattice 枚举顺序 = layout.json 的 id.
"""
from __future__ import annotations
import json
import os
import numpy as np

PITCH = 5.2
DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))   # 满量程 (归一化用)
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))   # 中心差分步长

# ---- lattice (pcb_verdict.py L21-28) ----
_a1 = np.array([PITCH, 0.0]); _a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS: list[tuple[float, float, int]] = []
for _i in range(-3, 4):
    for _j in range(-3, 4):
        _p = _i * _a1 + _j * _a2
        if np.linalg.norm(_p) < PITCH * 2.3:
            UNITS.append((float(_p[0]), float(_p[1]), (_i - _j) % 3))
NU = len(UNITS)                                   # 19
XY = np.array([[u[0], u[1]] for u in UNITS])      # (19,2)
COLOR = np.array([u[2] for u in UNITS])           # 三色格
SUNK = COLOR == 0                                 # 下沉单元 (L3+L4): [1,3,6,9,12,15,17]
ADJ = {i: [j for j in range(NU) if i != j and
           np.hypot(XY[i, 0] - XY[j, 0], XY[i, 1] - XY[j, 1]) < PITCH * 1.05] for i in range(NU)}
EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})   # 42, 字典序
NE = len(EDGES)
CI = int(min(range(NU), key=lambda u: np.hypot(XY[u, 0], XY[u, 1])))         # 中心单元 = 9
OBS = [('self', i, i) for i in range(NU)] + [('edge', i, j) for i, j in EDGES]  # 61
NOBS = len(OBS)
EDGE_IDX = np.array(EDGES)                        # (42,2)

# ---- 层高 (相对 L1 顶层铜, 向下为负). 环质心在 +GAP ----
# 叠层 L1 0.1 L2 0.6 L3 0.1 L4 → 标称 L1/L2 = 0/-0.10, 下沉 L3/L4 = -0.73/-0.83 (冻结层深差 0.73)
LAYER_Z_NOMINAL = (0.0, -0.10)
LAYER_Z_SUNK = (-0.73, -0.83)
COIL_Z0 = np.where(SUNK, LAYER_Z_SUNK[0], LAYER_Z_NOMINAL[0])   # 每单元顶层铜 z
COIL_DZ = -0.10                                                  # 第二层相对顶层
GAP_NOM = 1.75      # 判决点 (环质心→L1)
GAP_REST = 2.53     # BF-1000 2.38mm 静息
GAP_MIN = 1.35      # 45% 压缩

# ---- 位姿 / q 向量 ----
# pose (19,5) = [dx, dy, dz, tax, tay] 相对标称 (dz 相对 GAP); q (57,) = [w(19), u(19), v(19)] 物理 mm
POSE_BOX = np.array([1.0, 1.0, 0.85, 0.35, 0.35])   # 钳位盒 (|x,y|<1, dz 下界另限, |tilt|<0.35 rad)
DZ_MIN, DZ_MAX = GAP_MIN - GAP_REST - 0.1, 0.35    # 相对 GAP_REST 使用时的 dz 范围; 相对 GAP_NOM 时由 ModelConfig 给

def clamp_pose(pose: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """钳位: 绝对间隙 gap+dz ∈ [1.0, 3.0], |x,y|<1.0, |tilt|<0.35. 只钳不断言. 返回 (钳后, 被钳 mask)."""
    p = np.array(pose, dtype=float).reshape(NU, 5)
    lo = np.array([-1.0, -1.0, 1.0 - gap, -0.35, -0.35])
    hi = np.array([1.0, 1.0, 3.0 - gap, 0.35, 0.35])
    q = np.clip(p, lo, hi)
    return q, (q != p)

# ---- 柔性先验 T (series_scheme.py L118-144) ----
def _build_T():
    Gx = np.zeros((NU, NU)); Gy = np.zeros((NU, NU))
    for u in range(NU):
        nb = [u] + ADJ[u]
        A = np.array([[1.0, XY[k, 0] - XY[u, 0], XY[k, 1] - XY[u, 1]] for k in nb])
        P = np.linalg.pinv(A)
        for m, k in enumerate(nb):
            Gx[u, k] = P[1, m]; Gy[u, k] = P[2, m]
    T = np.zeros((NU * 5, 3 * NU))
    for u in range(NU):
        T[u * 5 + 0, NU + u] = 1.0
        T[u * 5 + 1, 2 * NU + u] = 1.0
        T[u * 5 + 2, u] = 1.0
        T[u * 5 + 3, :NU] = +Gy[u]
        T[u * 5 + 4, :NU] = -Gx[u]
    return Gx, Gy, T

GX, GY, T_PHYS = _build_T()                       # T_PHYS (95,57)
RANGE_P = np.array([RANGE[d] for d in DOFS] * NU)  # (95,)
RANGE_Q = np.concatenate([np.full(NU, RANGE['z']), np.full(NU, RANGE['x']), np.full(NU, RANGE['y'])])  # (57,)
T_S = (T_PHYS / RANGE_P[:, None]) * RANGE_Q[None, :]   # 缩放域映射

def pose_of(q: np.ndarray) -> np.ndarray:
    """q(57) 物理 → pose (19,5) 物理."""
    return (T_PHYS @ np.asarray(q, float)).reshape(NU, 5)

def split_q(q):
    q = np.asarray(q); return q[:NU], q[NU:2 * NU], q[2 * NU:]

# ---- layout.json ----
def layout_records() -> list[dict]:
    return [dict(id=i, x_mm=round(float(XY[i, 0]), 3), y_mm=round(float(XY[i, 1]), 3),
                 layer='L3+L4(下沉)' if SUNK[i] else 'L1+L2(标称)', sunk=bool(SUNK[i])) for i in range(NU)]

def write_layout(path: str) -> None:
    json.dump(layout_records(), open(path, 'w'), ensure_ascii=False, indent=1)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def hex_vertices(cx: float, cy: float, r: float | None = None) -> np.ndarray:
    """单元六边形 (pointy-top? 三角格 a1 沿 x → 六边形顶点在 30°+k60°), 外接圆半径 r=pitch/√3."""
    r = PITCH / np.sqrt(3) if r is None else r
    a = np.deg2rad(30 + 60 * np.arange(6))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
