#!/usr/bin/env python3
"""方案B: 串联半端口 (T+B 串联, 互感增强) 可解性评估
观测 61 = 19 端口自感 + 42 邻边打包互感; 从存档 J_stag (145x95) 线性折叠
步骤: 1 折叠+抽查  2 无先验 61x95  3 柔性先验 61x57 (tilt=梯度)  4 端到端
"""
import numpy as np
import json, time
import gen_ring_array as G

PITCH, GAP_NOM, STAG = 5.2, 1.75, 0.9
DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'

# ---- 19 单元几何/OBS (与 compute_spectrum19.py 完全一致的确定性构造) ----
a1 = np.array([PITCH, 0.0]); a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            UNITS.append((p[0], p[1], (i - j) % 3))
NU = len(UNITS)
BOT_Z = [-STAG if c == 0 else 0.0 for _, _, c in UNITS]
ADJ = {i: [j for j in range(NU) if i != j and
           np.hypot(UNITS[i][0] - UNITS[j][0],
                    UNITS[i][1] - UNITS[j][1]) < PITCH * 1.05]
       for i in range(NU)}
OBS_A = []
for i in range(NU):
    OBS_A.append(('TB', i, i))
    for j in ADJ[i]:
        OBS_A.append(('TB', i, j))
EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})
for i, j in EDGES:
    OBS_A.append(('TT', i, j))
IDX_A = {o: k for k, o in enumerate(OBS_A)}
CI = min(range(NU), key=lambda u: np.hypot(UNITS[u][0], UNITS[u][1]))

# ---- 载入存档 Jacobian ----
npz = np.load(REP + 'honeycomb_spectrum19.npz')
J_A, M0_A = npz['J_stag'], npz['M0_stag']          # 145x95 (全量程相对), nH
assert J_A.shape == (len(OBS_A), NU * 5)

# ---- 标称全矩阵 (取 L_T/L_B 对角与 BB 常数) ----
def solve_full(poses):
    rings = []
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, GAP_NOM + dz, tax, tay)))
    for u, (cx, cy, c) in enumerate(UNITS):
        rings.append((f'B{u}', (cx, cy, BOT_Z[u], 0.0, 0.0)))
    Lm, _ = G.solve(rings, 'series_tmp')
    return Lm * 1e9                                  # nH, 顺序 T0..T18,B0..B18

Lm0 = solve_full(np.zeros((NU, 5)))
Ti = lambda u: u
Bi = lambda u: NU + u
# 一致性: npz M0 与本次标称解
chk = max(abs(Lm0[Ti(i), Bi(i)] - M0_A[IDX_A[('TB', i, i)]]) for i in range(NU))
print(f'标称解 vs npz M0 一致性: max diff {chk:.2e} nH')

# ---- 方案B 观测定义与折叠 ----
# self_i = L_Ti + L_Bi + 2*M(Ti,Bi);  edge_(i,j) = TT + TB(ij) + TB(ji) + BB
OBS_B = [('self', i, i) for i in range(NU)] + [('edge', i, j) for i, j in EDGES]
NB = len(OBS_B)
y0 = np.zeros(NB)
for k, (kind, i, j) in enumerate(OBS_B):
    if kind == 'self':
        y0[k] = Lm0[Ti(i), Ti(i)] + Lm0[Bi(i), Bi(i)] + 2 * Lm0[Ti(i), Bi(i)]
    else:
        y0[k] = (Lm0[Ti(i), Ti(j)] + Lm0[Ti(i), Bi(j)] +
                 Lm0[Ti(j), Bi(i)] + Lm0[Bi(i), Bi(j)])
print(f'方案B: {NB} 观测 (self 19 + edge {len(EDGES)}), L_port0={y0[0]:.2f}nH, '
      f'M_edge0 典型 {y0[NU]:.3f}nH')

dM_A = J_A * M0_A[:, None]                 # 反归一化: nH / 全量程
J_B = np.zeros((NB, NU * 5))
for k, (kind, i, j) in enumerate(OBS_B):
    if kind == 'self':
        J_B[k] = 2 * dM_A[IDX_A[('TB', i, i)]]
    else:
        J_B[k] = (dM_A[IDX_A[('TT', i, j)]] + dM_A[IDX_A[('TB', i, j)]] +
                  dM_A[IDX_A[('TB', j, i)]])
J_B /= y0[:, None]                          # 相对变化归一

# ---- 折叠抽查: 3 个 DOF 列直接有限差分 ----
def observe_B(poses):
    Lm = solve_full(poses)
    y = np.zeros(NB)
    for k, (kind, i, j) in enumerate(OBS_B):
        if kind == 'self':
            y[k] = Lm[Ti(i), Ti(i)] + Lm[Bi(i), Bi(i)] + 2 * Lm[Ti(i), Bi(i)]
        else:
            y[k] = (Lm[Ti(i), Ti(j)] + Lm[Ti(i), Bi(j)] +
                    Lm[Ti(j), Bi(i)] + Lm[Bi(i), Bi(j)])
    return y

DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))
print('折叠抽查 (中心单元 x/z/tay):')
for dof in ['x', 'z', 'tay']:
    d = DELTA[dof]; c = DOFS.index(dof)
    pp = np.zeros((NU, 5)); pp[CI, c] = +d
    pm = np.zeros((NU, 5)); pm[CI, c] = -d
    col_fd = (observe_B(pp) - observe_B(pm)) / (2 * d) * RANGE[dof] / y0
    col_fold = J_B[:, CI * 5 + c]
    ref = np.max(np.abs(col_fd))
    err = np.max(np.abs(col_fd - col_fold)) / ref
    print(f'  {dof}: 最大相对偏差 {err*100:.2f}% (幅度 {ref:.4f})')

# ---- 步骤2: 无先验 61x95 ----
sv_B = np.linalg.svd(J_B, compute_uv=False)
rank_B = int(np.sum(sv_B > 1e-8))
print(f'\n无先验: rank {rank_B}/95, 盲空间 {NU*5 - rank_B} 维 (计数上限 61)')
print('sv 范围:', f'{sv_B[0]:.3f} .. {sv_B[rank_B-1]:.5f}')

# ---- 步骤3: 柔性先验 T (95x57) ----
# q = [w(19), u(19), v(19)] (物理 mm); tilt = w 的 LS 平面梯度
Gx = np.zeros((NU, NU)); Gy = np.zeros((NU, NU))
for u in range(NU):
    nb = [u] + ADJ[u]
    A = np.array([[1.0, UNITS[k][0] - UNITS[u][0], UNITS[k][1] - UNITS[u][1]]
                  for k in nb])
    P = np.linalg.pinv(A)          # 3 x len(nb); 行1=∂w/∂x, 行2=∂w/∂y
    for m, k in enumerate(nb):
        Gx[u, k] = P[1, m]
        Gy[u, k] = P[2, m]
# 自检: 平面场 w=a x + b y
a_, b_ = 0.3, -0.7
wp = np.array([a_ * UNITS[k][0] + b_ * UNITS[k][1] for k in range(NU)])
assert np.allclose(Gx @ wp, a_, atol=1e-9) and np.allclose(Gy @ wp, b_, atol=1e-9)
print('T 自检: 平面场梯度精确 OK')

T_phys = np.zeros((NU * 5, 3 * NU))        # pose_phys = T_phys @ q_phys
for u in range(NU):
    T_phys[u * 5 + 0, NU + u] = 1.0        # x = u
    T_phys[u * 5 + 1, 2 * NU + u] = 1.0    # y = v
    T_phys[u * 5 + 2, u] = 1.0             # z = w
    T_phys[u * 5 + 3, :NU] = +Gy[u]        # tax = +dw/dy
    T_phys[u * 5 + 4, :NU] = -Gx[u]        # tay = -dw/dx
RANGE_p = np.array([RANGE[d] for d in DOFS] * NU)
RANGE_q = np.concatenate([np.full(NU, RANGE['z']),
                          np.full(NU, RANGE['x']), np.full(NU, RANGE['y'])])
T_s = (T_phys / RANGE_p[:, None]) * RANGE_q[None, :]     # 缩放域间映射

J_Bc = J_B @ T_s                            # 61 x 57
sv_Bc = np.linalg.svd(J_Bc, compute_uv=False)
rank_Bc = int(np.sum(sv_Bc > 1e-6))
cond_Bc = sv_Bc[0] / sv_Bc[-1] if rank_Bc == 3 * NU else np.inf
print(f'\n先验约束 61x57: rank {rank_Bc}/57, min sv {sv_Bc[-1]:.5f}, '
      f'cond {cond_Bc:.1f}')
J_Ac = J_A @ T_s                            # 对照: 方案A同先验 145x57
sv_Ac = np.linalg.svd(J_Ac, compute_uv=False)
print(f'对照 方案A+先验 145x57: min sv {sv_Ac[-1]:.5f}, cond {sv_Ac[0]/sv_Ac[-1]:.1f}')

# 弱模式形态
U_, S_, Vt_ = np.linalg.svd(J_Bc)
weak = Vt_[-1]
wm = dict(w=float(np.linalg.norm(weak[:NU])),
          u=float(np.linalg.norm(weak[NU:2*NU])),
          v=float(np.linalg.norm(weak[2*NU:])))
print('最弱模式分量占比:', {k: f'{v:.2f}' for k, v in wm.items()})

# ---- 直接运行时保存基础判决 json (端到端见 series_fix.py, 用噪声地板归一+规范固定) ----
if __name__ == '__main__':
    out = dict(n_obs=NB, n_pose=NU * 5, n_q=3 * NU,
               rank_noprior=rank_B, blind_dim=NU * 5 - rank_B,
               sv_B_noprior=sv_B.tolist(),
               rank_prior=rank_Bc, cond_prior=float(cond_Bc),
               sv_B_prior=sv_Bc.tolist(), sv_A_prior=sv_Ac.tolist(),
               cond_A_prior=float(sv_Ac[0] / sv_Ac[-1]), weak_mode_parts=wm,
               L_port0_nH=float(y0[0]), M_edge0_nH=float(y0[NU]))
    with open(REP + 'honeycomb_series.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('saved honeycomb_series.json (base)')
