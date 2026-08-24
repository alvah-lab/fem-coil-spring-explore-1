#!/usr/bin/env python3
"""步骤4: 阵列级联合分析
观测(邻接限定): T_i->B_j (i,j 邻接或相同) 31个 + T_i->T_j 邻接边 12个 = 43
未知: 7单元 x 5DOF = 35
(a) 联合 Jacobian 43x35: 秩/条件数/最弱模式
(b) 联动端到端: 中心点压+邻居联动 alpha, 高斯牛顿反演恢复位姿场
"""
import numpy as np
import json
from gen_ring_array import build_patch, solve, hex_centers, GAP_NOM

DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))
NU = 7

# 邻接表 (7单元 patch: 0=中心; 1..6 环, 环上 j 与 j%6+1 相邻)
ADJ = {0: [1, 2, 3, 4, 5, 6]}
for j in range(1, 7):
    ADJ[j] = [0, j % 6 + 1, (j - 2) % 6 + 1]

# 观测列表
OBS = []                       # (kind, i, j): 'TB' T_i->B_j ; 'TT' T_i->T_j
for i in range(NU):
    OBS.append(('TB', i, i))
    for j in ADJ[i]:
        OBS.append(('TB', i, j))
TT_EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})
for i, j in TT_EDGES:
    OBS.append(('TT', i, j))
print(f'观测数: {len(OBS)} (TB={sum(1 for o in OBS if o[0]=="TB")}, '
      f'TT={len(TT_EDGES)}), 未知数: {NU*5}')

def observe_all(poses):
    """poses: (7,5) 数组 -> 观测向量 (nH)"""
    tp = {u: tuple(poses[u]) for u in range(NU)}
    rings = build_patch(top_poses=tp)
    Lm, _ = solve(rings, 'joint_tmp')
    names = [r[0] for r in rings]
    idx = {n: k for k, n in enumerate(names)}
    v = []
    for kind, i, j in OBS:
        a = idx[f'T{i}']
        b = idx[f'B{j}'] if kind == 'TB' else idx[f'T{j}']
        v.append(Lm[a, b])
    return np.array(v) * 1e9

def joint_jacobian(base_poses=None):
    if base_poses is None:
        base_poses = np.zeros((NU, 5))
    M0 = observe_all(base_poses)
    J = np.zeros((len(OBS), NU * 5))
    col = 0
    for u in range(NU):
        for d, dof in enumerate(DOFS):
            dd = DELTA[dof]
            pp = base_poses.copy(); pp[u, d] += dd
            pm = base_poses.copy(); pm[u, d] -= dd
            J[:, col] = (observe_all(pp) - observe_all(pm)) / (2 * dd) \
                        * RANGE[dof] / M0
            col += 1
    return J, M0

def linked_poses(delta0, alpha, tilt_deg):
    """中心点压联动模式: 中心 dz=-delta0, 邻居 dz=-alpha*delta0 + 向心倾斜"""
    p = np.zeros((NU, 5))
    p[0, 2] = -delta0
    cs = hex_centers()
    t = np.deg2rad(tilt_deg)
    for u in range(1, NU):
        p[u, 2] = -alpha * delta0
        phi = np.arctan2(cs[u][1], cs[u][0])
        p[u, 3] = t * np.sin(phi)     # tax
        p[u, 4] = -t * np.cos(phi)    # tay
    return p

if __name__ == '__main__':
    out = dict(n_obs=len(OBS), n_unknown=NU * 5)

    # ---- (a) 联合 Jacobian ----
    print('计算联合 Jacobian (70 次求解)...')
    J, M0 = joint_jacobian()
    s = np.linalg.svd(J, compute_uv=False)
    print(f'奇异值范围: max={s[0]:.3f} min={s[-1]:.5f}  cond={s[0]/s[-1]:.1f}')
    print('最小5个奇异值:', np.array2string(s[-5:], precision=5))
    # 最弱模式解读
    U, S, Vt = np.linalg.svd(J)
    weak = Vt[-1].reshape(NU, 5)
    print('最弱模式 (各单元x,y,z,tax,tay 分量, 归一化):')
    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    print(weak / np.max(np.abs(weak)))
    out['sv'] = s.tolist()
    out['cond'] = float(s[0] / s[-1])
    out['weak_mode'] = weak.tolist()

    # ---- (b) 联动端到端反演 ----
    print('\n端到端: 联动模式 (delta0=0.4mm, alpha=0.3, tilt=3deg)')
    truth = linked_poses(0.4, 0.3, 3.0)
    M_meas = observe_all(truth)
    # 高斯牛顿 (用标称点 Jacobian 作准牛顿, 观测归一 dM/M0, DOF 全量程归一)
    scale = np.array([RANGE[d] for d in DOFS] * NU)
    x = np.zeros(NU * 5)          # 归一化位姿变量
    for it in range(12):
        M_cur = observe_all((x * scale).reshape(NU, 5))
        r = (M_meas - M_cur) / M0
        dx, *_ = np.linalg.lstsq(J, r, rcond=None)
        x += dx
        if np.linalg.norm(dx) < 1e-6:
            break
    est = (x * scale).reshape(NU, 5)
    err = est - truth
    print(f'迭代 {it+1} 次, 残差 |r|={np.linalg.norm(r):.2e}')
    print('恢复误差 (mm / rad):')
    print(err)
    print(f'最大 z 误差 {np.max(np.abs(err[:,2]))*1000:.2f} um, '
          f'最大平移误差 {np.max(np.abs(err[:,:2]))*1000:.2f} um, '
          f'最大倾角误差 {np.rad2deg(np.max(np.abs(err[:,3:]))):.4f} deg')
    # alpha 恢复
    alpha_est = -np.mean(est[1:, 2]) / -est[0, 2] if est[0, 2] != 0 else np.nan
    print(f'恢复 alpha = {alpha_est:.4f} (真值 0.300), '
          f'中心压深 = {-est[0,2]:.4f}mm (真值 0.400)')
    out['e2e'] = dict(truth=truth.tolist(), est=est.tolist(),
                      alpha_est=float(alpha_est),
                      max_err_z_um=float(np.max(np.abs(err[:, 2])) * 1000),
                      max_err_xy_um=float(np.max(np.abs(err[:, :2])) * 1000),
                      max_err_tilt_deg=float(np.rad2deg(np.max(np.abs(err[:, 3:])))))

    # ---- 噪声鲁棒性: 1e-3 相对噪声 x 200 次蒙特卡洛 (线性传播) ----
    # 位姿误差协方差 ~ pinv(J) * diag(noise) ; noise=1e-3 (相对每个观测自身)
    Jp = np.linalg.pinv(J)
    sigma_pose_scaled = np.sqrt(np.sum(Jp ** 2, axis=1)) * 1e-3
    sp = sigma_pose_scaled.reshape(NU, 5)
    # 换算物理单位
    phys = sp * np.array([[RANGE[d] for d in DOFS]])
    print('\n1e-3 相对测量噪声下的位姿 1σ分辨率 (中心单元):')
    print(f'  x,y: {phys[0,0]*1000:.2f}, {phys[0,1]*1000:.2f} um; '
          f'z: {phys[0,2]*1000:.2f} um; '
          f'tilt: {np.rad2deg(phys[0,3])*60:.2f}, {np.rad2deg(phys[0,4])*60:.2f} arcmin')
    out['noise_1e-3_pose_sigma'] = phys.tolist()

    jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_joint.json'
    with open(jp, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', jp)
