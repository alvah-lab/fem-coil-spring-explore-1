#!/usr/bin/env python3
"""步骤4b: 弱模式解读 + 截断SVD反演修复 + 交错锚点高度设计变体
"""
import numpy as np
import json
import gen_ring_array as G
from gen_ring_array import solve, hex_centers, GAP_NOM
from joint_array import (DOFS, RANGE, DELTA, NU, OBS, TT_EDGES,
                         linked_poses)

# ---- 支持内层高度模式的 patch 构建 ----
def build_patch2(top_poses=None, gap=GAP_NOM, bot_z=None):
    cs = hex_centers()
    if bot_z is None:
        bot_z = [0.0] * NU
    rings = []
    for u, (cx, cy) in enumerate(cs):
        dx = dy = dz = tax = tay = 0.0
        if top_poses and u in top_poses:
            dx, dy, dz, tax, tay = top_poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, gap + dz, tax, tay)))
    for u, (cx, cy) in enumerate(cs):
        rings.append((f'B{u}', (cx, cy, bot_z[u], 0.0, 0.0)))
    return rings

def observe_all2(poses, bot_z=None):
    tp = {u: tuple(poses[u]) for u in range(NU)}
    rings = build_patch2(top_poses=tp, bot_z=bot_z)
    Lm, _ = solve(rings, 'jfix_tmp')
    names = [r[0] for r in rings]
    idx = {n: k for k, n in enumerate(names)}
    v = []
    for kind, i, j in OBS:
        a = idx[f'T{i}']
        b = idx[f'B{j}'] if kind == 'TB' else idx[f'T{j}']
        v.append(Lm[a, b])
    return np.array(v) * 1e9

def joint_jacobian2(bot_z=None):
    base = np.zeros((NU, 5))
    M0 = observe_all2(base, bot_z)
    J = np.zeros((len(OBS), NU * 5))
    col = 0
    for u in range(NU):
        for d, dof in enumerate(DOFS):
            dd = DELTA[dof]
            pp = base.copy(); pp[u, d] += dd
            pm = base.copy(); pm[u, d] -= dd
            J[:, col] = (observe_all2(pp, bot_z) - observe_all2(pm, bot_z)) \
                        / (2 * dd) * RANGE[dof] / M0
            col += 1
    return J, M0

def mode_overlap(v):
    """v: (35,) 归一化模式 -> 与解释基的重叠"""
    basis = {}
    for d, dof in enumerate(DOFS):
        e = np.zeros((NU, 5)); e[:, d] = 1
        basis[f'common_{dof}'] = e.flatten() / np.linalg.norm(e)
    ov = {k: float(v @ b) for k, b in basis.items()}
    return ov

if __name__ == '__main__':
    out = {}
    # ---------- (1) 弱模式解读 ----------
    print('=== 基线 (所有锚点 z=0) ===')
    J, M0 = joint_jacobian2()
    U, S, Vt = np.linalg.svd(J)
    print('最小5奇异值:', np.array2string(S[-5:], precision=5))
    for k in range(1, 4):
        ov = mode_overlap(Vt[-k])
        top = sorted(ov.items(), key=lambda x: -abs(x[1]))[:3]
        print(f'弱模式{k} (sv={S[-k]:.5f}) 重叠:',
              ', '.join(f'{n}={v:+.2f}' for n, v in top))
    out['baseline_sv_min5'] = S[-5:].tolist()

    # ---------- (2) 截断SVD反演: 联动场景 ----------
    truth = linked_poses(0.4, 0.3, 3.0)
    scale = np.array([RANGE[d] for d in DOFS] * NU)
    truth_scaled = (truth.flatten()) / scale
    # 真值在零空间的分量
    null_frac = np.linalg.norm(Vt[-3:] @ truth_scaled) / np.linalg.norm(truth_scaled)
    print(f'\n联动真值在3个弱模式上的分量占比: {null_frac*100:.2f}%')

    CUT = 0.05
    Sinv = np.where(S > CUT, 1 / np.maximum(S, 1e-30), 0.0)
    Jp = (Vt.T * Sinv) @ U.T[:len(S)]

    M_meas = observe_all2(truth)
    x = np.zeros(NU * 5)
    for it in range(15):
        M_cur = observe_all2((x * scale).reshape(NU, 5))
        r = (M_meas - M_cur) / M0
        dx = Jp @ r
        x += dx
        if np.linalg.norm(dx) < 1e-7:
            break
    est = (x * scale).reshape(NU, 5)
    err = est - truth
    alpha_est = np.mean(est[1:, 2]) / est[0, 2]
    print(f'截断SVD反演: {it+1} 次迭代, |r|={np.linalg.norm(r):.2e}')
    print(f'  z 误差 max {np.max(np.abs(err[:,2]))*1000:.2f} um, '
          f'xy 误差 max {np.max(np.abs(err[:,:2]))*1000:.2f} um, '
          f'倾角误差 max {np.rad2deg(np.max(np.abs(err[:,3:]))):.3f} deg')
    print(f'  中心压深 {-est[0,2]:.4f}mm (真值0.400), alpha={alpha_est:.4f} (真值0.300)')
    out['e2e_truncated'] = dict(
        null_frac=float(null_frac),
        max_err_z_um=float(np.max(np.abs(err[:, 2])) * 1000),
        max_err_xy_um=float(np.max(np.abs(err[:, :2])) * 1000),
        max_err_tilt_deg=float(np.rad2deg(np.max(np.abs(err[:, 3:])))),
        alpha_est=float(alpha_est), depth_est_mm=float(-est[0, 2]))

    # 加噪声版本: 1e-3 相对噪声, 20次
    rng = np.random.default_rng(7)
    errs = []
    for trial in range(20):
        Mn = M_meas * (1 + 1e-3 * rng.standard_normal(len(M_meas)))
        x = np.zeros(NU * 5)
        for it in range(10):
            M_cur = observe_all2((x * scale).reshape(NU, 5))
            r = (Mn - M_cur) / M0
            dx = Jp @ r
            x += dx
            if np.linalg.norm(dx) < 1e-7:
                break
        errs.append((x * scale).reshape(NU, 5) - truth)
    errs = np.array(errs)
    print(f'  噪声1e-3: z 1σ={np.std(errs[:,:,2])*1000:.2f}um, '
          f'xy 1σ={np.std(errs[:,:,:2])*1000:.2f}um, '
          f'tilt 1σ={np.rad2deg(np.std(errs[:,:,3:]))*60:.1f} arcmin')
    out['e2e_noise'] = dict(z_um=float(np.std(errs[:, :, 2]) * 1000),
                            xy_um=float(np.std(errs[:, :, :2]) * 1000),
                            tilt_arcmin=float(np.rad2deg(np.std(errs[:, :, 3:])) * 60))

    # ---------- (3) 交错锚点高度变体 ----------
    print('\n=== 变体: 交错内层锚点高度 (奇数环单元 z=-0.6) ===')
    bz = [0.0, -0.6, 0.0, -0.6, 0.0, -0.6, 0.0]
    J2, M02 = joint_jacobian2(bot_z=bz)
    S2 = np.linalg.svd(J2, compute_uv=False)
    print('最小5奇异值:', np.array2string(S2[-5:], precision=5))
    print(f'cond: {S2[0]/S2[-1]:.1f}  (基线 {S[0]/S[-1]:.1f})')
    U2, S2f, Vt2 = np.linalg.svd(J2)
    for k in range(1, 4):
        ov = mode_overlap(Vt2[-k])
        top = sorted(ov.items(), key=lambda x: -abs(x[1]))[:3]
        print(f'弱模式{k} (sv={S2f[-k]:.5f}) 重叠:',
              ', '.join(f'{n}={v:+.2f}' for n, v in top))
    out['staggered_sv_min5'] = S2[-5:].tolist()
    out['staggered_cond'] = float(S2[0] / S2[-1])
    out['staggered_bot_z'] = bz

    jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_joint_fix.json'
    with open(jp, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', jp)
