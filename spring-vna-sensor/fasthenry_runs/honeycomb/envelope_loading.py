#!/usr/bin/env python3
"""步骤4c/5: (1) 工作包络上单元级13x5条件数扫描
           (2) 第三线圈负载效应 (成对独立性前提) 解析量化
           (3) 量程安全性: 邻居净间隙 vs 剪切/倾斜行程
"""
import numpy as np
import json
import gen_ring_array as G
from gen_ring_array import build_patch, solve, GAP_NOM

DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))
OBS_ALL = [f'B{j}' for j in range(7)] + [f'T{j}' for j in range(1, 7)]

def observe(top0_pose, gap):
    rings = build_patch(top_poses={0: top0_pose}, gap=gap)
    Lm, _ = solve(rings, 'env_tmp')
    names = [r[0] for r in rings]
    idx = {n: i for i, n in enumerate(names)}
    return np.array([Lm[idx['T0'], idx[t]] for t in OBS_ALL]) * 1e9

def cond_at(base_pose, gap):
    """在给定工作点的 13x5 条件数"""
    M0 = observe(base_pose, gap)
    J = np.zeros((13, 5))
    for c, dof in enumerate(DOFS):
        d = DELTA[dof]
        pp = list(base_pose); pp[c] += d
        pm = list(base_pose); pm[c] -= d
        J[:, c] = (observe(tuple(pp), gap) - observe(tuple(pm), gap)) \
                  / (2 * d) * RANGE[dof] / M0
    s = np.linalg.svd(J, compute_uv=False)
    return s[0] / s[-1], s

if __name__ == '__main__':
    out = {}
    # ---------- (1) 包络扫描 ----------
    print('=== 工作包络 13x5 条件数扫描 (相对标称 gap=1.75 的 T0 位姿偏移) ===')
    cases = [
        ('标称 g=1.75', (0, 0, 0, 0, 0), 1.75),
        ('全压缩 g=1.00', (0, 0, -0.75, 0, 0), 1.75),
        ('松弛 g=2.50', (0, 0, 0.75, 0, 0), 1.75),
        ('剪切 x=0.5', (0.5, 0, 0, 0, 0), 1.75),
        ('倾斜 10deg', (0, 0, 0, 0, np.deg2rad(10)), 1.75),
        ('压缩+剪切+倾斜', (0.4, 0, -0.5, 0, np.deg2rad(8)), 1.75),
        ('松弛+剪切+倾斜', (0.4, 0, 0.6, 0, np.deg2rad(8)), 1.75),
    ]
    env = {}
    for name, pose, gap in cases:
        c, s = cond_at(pose, gap)
        env[name] = dict(cond=float(c), sv=s.tolist())
        print(f'  {name:24s} cond={c:.2f}  sv_min={s[-1]:.4f}')
    out['envelope'] = env

    # ---------- (2) 第三线圈负载效应 (解析, 束绕单位) ----------
    print('\n=== 第三线圈负载效应 (i 驱动, j 接收, k 挂负载 Zk) ===')
    N2 = 225
    rings = build_patch()
    Lm14, _ = solve(rings, 'load_tmp')
    names = [r[0] for r in rings]
    idx = {n: i for i, n in enumerate(names)}
    Lb = Lm14 * N2                      # H, 束绕
    Lself = Lb[idx['T1'], idx['T1']]
    f = 10e6; w = 2 * np.pi * f
    # 最坏三元组: i=T0驱动, j=B1接收(弱观测), k=T1 (与两者都强耦合)
    M_ik = abs(Lb[idx['T0'], idx['T1']])
    M_kj = abs(Lb[idx['T1'], idx['B1']])
    M_ij = abs(Lb[idx['T0'], idx['B1']])
    print(f'  三元组 T0->B1 经 T1: M_ij={M_ij*1e9:.1f}nH, '
          f'M_ik={M_ik*1e9:.1f}nH, M_kj={M_kj*1e9:.1f}nH, L={Lself*1e6:.2f}uH')
    res = {}
    for zk_name, Zk in [('短路', 0.0), ('50欧', 50.0),
                        ('谐振串联RLC(R=2欧)', complex(2.0, -w * Lself)),
                        ('开路(1M欧)', 1e6)]:
        # I_k/I_i = -jw M_ik/(Zk + jwL_k); dM_eff = M_ik*M_kj*w/|Zk+jwLk| 形式
        denom = abs(Zk + 1j * w * Lself)
        dM = w * M_ik * M_kj / denom / w   # = M_ik*M_kj/|Zk/w + jLk|... 简化
        dM = M_ik * M_kj * w / denom
        res[zk_name] = dict(dM_nH=dM * 1e9, vs_Mij_pct=dM / M_ij * 100)
        print(f'  k 负载={zk_name:18s}: |dM_eff|={dM*1e9:.2f}nH '
              f'= M_ij 的 {dM/M_ij*100:.1f}%')
    out['third_coil_loading'] = res
    print('  => 非活动线圈必须开路/高阻; 串联谐振负载绝对禁止')

    # ---------- (3) 机械行程安全 ----------
    print('\n=== 行程安全 ===')
    clearance = 5.2 - 5.0
    print(f'  相邻净间隙 {clearance:.1f}mm: 相向剪切 2x0.5mm 会碰撞!')
    print(f'  安全相向剪切 < {clearance/2*0.8:.2f}mm/单元 (留20%裕量)')
    tilt_drop = 2.5 * np.sin(np.deg2rad(10))
    print(f'  倾斜10°边缘下探 {tilt_drop:.2f}mm (g=1.0 时余 {1.0-tilt_drop:.2f}mm)')
    out['mechanical'] = dict(clearance_mm=clearance,
                             safe_shear_mm=clearance / 2 * 0.8,
                             tilt10_edge_drop_mm=float(tilt_drop))

    jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_envelope.json'
    with open(jp, 'w') as f2:
        json.dump(out, f2, indent=1)
    print('saved', jp)
