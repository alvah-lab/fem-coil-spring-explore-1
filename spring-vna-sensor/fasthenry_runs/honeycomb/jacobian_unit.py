#!/usr/bin/env python3
"""步骤3: 单元级 Jacobian (中心单元 T0 的 5-DOF)
观测: 绝对7个 M(T0->B0..B6) + 相对6个 M(T0->T1..T6) = 13
DOF: x, y, z, tax(绕x), tay(绕y);  全量程归一化
输出: 7x5 与 13x5 Jacobian, SVD 条件数, {x,tay} 子空间夹角, 线性区/对称性自检
"""
import numpy as np
import json
from gen_ring_array import build_patch, solve, GAP_NOM

# 全量程定义 (条件数的归一化标尺)
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DOFS = ['x', 'y', 'z', 'tax', 'tay']
DELTA = dict(x=0.1, y=0.1, z=0.1, tax=np.deg2rad(2), tay=np.deg2rad(2))

OBS_ABS = [f'B{j}' for j in range(7)]           # 绝对观测 (内层全固定)
OBS_REL = [f'T{j}' for j in range(1, 7)]        # 相对观测
OBS_ALL = OBS_ABS + OBS_REL

def observe(top0_pose):
    """T0 位姿 (dx,dy,dz,tax,tay) -> 13 观测 (单匝 nH)"""
    rings = build_patch(top_poses={0: top0_pose})
    Lm, _ = solve(rings, 'jac_tmp')
    names = [r[0] for r in rings]
    idx = {n: i for i, n in enumerate(names)}
    return np.array([Lm[idx['T0'], idx[t]] for t in OBS_ALL]) * 1e9

def pose_of(dof, val):
    p = [0.0] * 5
    p[DOFS.index(dof)] = val
    return tuple(p)

def jacobian(delta_scale=1.0):
    """中心差分 Jacobian, 列=DOF (按全量程归一), 行=观测相对变化 dM/M0"""
    M0 = observe((0, 0, 0, 0, 0))
    J = np.zeros((len(OBS_ALL), len(DOFS)))
    for c, dof in enumerate(DOFS):
        d = DELTA[dof] * delta_scale
        Mp = observe(pose_of(dof, +d))
        Mm = observe(pose_of(dof, -d))
        J[:, c] = (Mp - Mm) / (2 * d) * RANGE[dof] / M0   # 全量程相对变化
    return J, M0

def svd_report(J, label):
    s = np.linalg.svd(J, compute_uv=False)
    cond = s[0] / s[-1]
    print(f'{label}: 奇异值 {np.array2string(s, precision=4)}  cond={cond:.1f}')
    return s, cond

def col_angle(J, i, j):
    a, b = J[:, i], J[:, j]
    c = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.rad2deg(np.arccos(np.clip(c, 0, 1)))

if __name__ == '__main__':
    out = {}
    J, M0 = jacobian(1.0)
    print('M0 (nH):', np.array2string(M0, precision=4))
    print('\nJacobian (全量程相对变化, 行=B0..B6,T1..T6, 列=x,y,z,tax,tay):')
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print(J)

    print('\n--- SVD ---')
    s7, c7 = svd_report(J[:7], '绝对 7x5')
    s13, c13 = svd_report(J, '全 13x5')

    # {x, tay} / {y, tax} 剪切-倾斜可分性
    for pair in [('x', 'tay'), ('y', 'tax')]:
        i, jx = DOFS.index(pair[0]), DOFS.index(pair[1])
        a7 = col_angle(J[:7], i, jx)
        a13 = col_angle(J, i, jx)
        print(f'夹角 {pair}: 绝对7观测 {a7:.1f}°, 全13观测 {a13:.1f}°')
        out[f'angle_{pair[0]}_{pair[1]}'] = dict(abs7=a7, all13=a13)
        # 2x5 子问题条件数 (只保留这两列)
        sub7 = np.linalg.svd(J[:7][:, [i, jx]], compute_uv=False)
        sub13 = np.linalg.svd(J[:, [i, jx]], compute_uv=False)
        print(f'  2列子问题 cond: 绝对 {sub7[0]/sub7[1]:.1f}, 全 {sub13[0]/sub13[1]:.1f}')
        out[f'cond2_{pair[0]}_{pair[1]}'] = dict(abs7=float(sub7[0] / sub7[1]),
                                                 all13=float(sub13[0] / sub13[1]))

    # 线性区检查: delta 减半
    Jh, _ = jacobian(0.5)
    lin_err = np.max(np.abs(Jh - J) / (np.max(np.abs(J))))
    print(f'\n线性区检查 (delta减半 Jacobian 最大相对偏差): {lin_err*100:.2f}%')

    # 对称性检查: 沿60°方向平移 => 观测应是沿x平移的邻居下标轮换
    d = DELTA['x']
    Mx = observe((d, 0, 0, 0, 0))
    M60 = observe((d * np.cos(np.pi / 3), d * np.sin(np.pi / 3), 0, 0, 0))
    # 邻居 j 对应方位角 60*(j-1); 60°旋转 => B_j -> B_{j+1} 轮换 (B0不变)
    perm = [0] + [1 + (j % 6) for j in range(1, 7)]          # B: [B0,B2..B6,B1]?
    # 正确轮换: 旋转+60°后, 原方位角th的邻居移到th+60 => 新观测B_{j}=旧B_{j-1}
    permB = [0, 6, 1, 2, 3, 4, 5]
    permB = [0] + [1 + ((j - 1 - 1) % 6) for j in range(1, 7)]
    Mx_rot = Mx[:7][permB]
    sym_err = np.max(np.abs(M60[:7] - Mx_rot)) / np.max(np.abs(Mx[:7] - observe((0,0,0,0,0))[:7]) + 1e-12)
    print(f'六重对称检查 (60°平移 vs x平移轮换, 相对扰动幅度): {sym_err*100:.1f}%')

    out.update(M0_nH=M0.tolist(), J_fullrange_rel=J.tolist(),
               obs=OBS_ALL, dofs=DOFS, range=dict((k, float(v)) for k, v in RANGE.items()),
               sv_abs7=s7.tolist(), cond_abs7=float(c7),
               sv_all13=s13.tolist(), cond_all13=float(c13),
               linearity_err=float(lin_err))
    jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_jacobian_unit.json'
    with open(jp, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', jp)
