#!/usr/bin/env python3
"""投板前判决 (精简): 真实层深0.73mm + 8MHz + 紫铜环, 单环等效有源线圈
可分性匝数无关 -> 用单环x匝数缩放, 快速跑 61x95 判决 vs 理想仿真对比
螺旋真实自感/Q 由 pcb_validation 一次重解补充"""
import numpy as np
import json, time, os
import gen_ring_array as G
from series_scheme import T_s, RANGE_q, NU as NU_ser

F = 8e6
W = 2 * np.pi * F
GAP = 1.75           # 环到标称螺旋
STAG = 0.73          # 层深交错
NB = 12 * 2          # 双层螺旋总匝(耦合缩放 24匝? M∝N. 用有效匝数)
R_RING = 6.0e-3      # 紫铜环 Ø5/0.2 @8MHz AC电阻(下方pcb_validation实测校准)
DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))
REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'

PITCH = 5.2
a1 = np.array([PITCH, 0.0]); a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            UNITS.append((p[0], p[1], (i - j) % 3))
NU = len(UNITS)
# 层深: 标称质心+0.08, 下沉-0.65 (相对环高 GAP): 环-螺旋间距 = GAP-coilz
COILZ = [(-0.65 if c == 0 else 0.08) for _, _, c in UNITS]
ADJ = {i: [j for j in range(NU) if i != j and
           np.hypot(UNITS[i][0] - UNITS[j][0],
                    UNITS[i][1] - UNITS[j][1]) < PITCH * 1.05] for i in range(NU)}
EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})
CI = min(range(NU), key=lambda u: np.hypot(UNITS[u][0], UNITS[u][1]))
OBS = [('self', i, i) for i in range(NU)] + [('edge', i, j) for i, j in EDGES]

def solve(top_poses, tag):
    """有源单环(在COILZ) + 无源环(在GAP+dz). 返回 L矩阵 nH"""
    assert np.all(np.abs(top_poses[:, :2]) < 1.0) and \
           np.all(np.abs(top_poses[:, 2]) < 1.2) and \
           np.all(np.abs(top_poses[:, 3:]) < 0.35)
    rings = []
    for u, (cx, cy, c) in enumerate(UNITS):
        rings.append((f'C{u}', (cx, cy, COILZ[u], 0.0, 0.0), 0.2))
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = top_poses[u]
        rings.append((f'R{u}', (cx + dx, cy + dy, GAP + dz, tax, tay), 1.0))
    L = ['* pcbv', '.units mm', f'.default sigma=5.8e7 nhinc=1 nwinc=1', '']
    for name, pose, w in rings:
        p = G.ring_nodes(pose)
        for i, (x, y, z) in enumerate(p):
            L.append(f'N{name}_{i} x={x:.5f} y={y:.5f} z={z:.5f}')
        for i in range(len(p) - 1):
            L.append(f'E{name}_{i} N{name}_{i} N{name}_{i+1} w={w} h=0.2')
        L.append('')
    for name, _, _ in rings:
        L.append(f'.external N{name}_0 N{name}_{G.NSEG}')
    L.append(f'.freq fmin={F} fmax={F} ndec=1')
    L.append('.end')
    inp = os.path.join(G.WORKDIR, f'{tag}.inp')
    open(inp, 'w').write('\n'.join(L) + '\n')
    Z, _ = G.run_fasthenry(inp)
    return np.imag(Z) / W * 1e9

def fold(Lm):
    """C体制: 有源螺旋端口有效电感 (无源环反射)"""
    L = Lm * 1e-9
    s = np.array([float(NB)] * NU + [1.0] * NU)   # 有源NB匝, 无源单匝
    Lp = L * np.outer(s, s)
    Lcc = Lp[:NU, :NU]; Lcr = Lp[:NU, NU:]; Lrr = Lp[NU:, NU:]
    Zr = R_RING * np.eye(NU) + 1j * W * Lrr
    Zeff = 1j * W * Lcc + W * W * (Lcr @ np.linalg.solve(Zr, Lcr.T))
    return np.imag(Zeff) / W * 1e9

def obs(Leff):
    return np.array([Leff[i, i] for i in range(NU)] +
                    [Leff[i, j] for i, j in EDGES])

def observe(top_poses, tag='pcbv_tmp'):
    return obs(fold(solve(top_poses, tag)))

if __name__ == '__main__':
    t0 = time.time()
    y0 = observe(np.zeros((NU, 5)), 'pcbv_b')
    L_bg = obs(np.imag(1j * W * (solve(np.zeros((NU, 5)), 'pcbv_b') * 1e-9 *
               np.outer(np.array([NB]*NU+[1.]*NU), np.array([NB]*NU+[1.]*NU)))[:NU, :NU]) / W * 1e9)
    refl = y0 - L_bg
    print(f'{NU}单元 层深交错{STAG}mm F={F/1e6}MHz', flush=True)
    print(f'自观测反射 [{np.min(np.abs(refl[:NU])):.1f},{np.max(np.abs(refl[:NU])):.1f}]nH, '
          f'下沉vs标称: {abs(refl[CI]):.1f} vs '
          f'{np.median([abs(refl[i]) for i in range(NU) if COILZ[i]>0]):.1f}nH', flush=True)

    print(f'Jacobian (190解, 单解~0.5s)...', flush=True)
    dY = np.zeros((len(y0), NU * 5))
    col = 0
    for u in range(NU):
        for d, dof in enumerate(DOFS):
            dd = DELTA[dof]
            pp = np.zeros((NU, 5)); pp[u, d] = dd
            pm = np.zeros((NU, 5)); pm[u, d] = -dd
            dY[:, col] = (observe(pp) - observe(pm)) / (2 * dd) * RANGE[dof]
            col += 1
        if (u + 1) % 5 == 0:
            print(f'  {u+1}/{NU} ({time.time()-t0:.0f}s)', flush=True)

    sig2 = np.maximum(1e-3 * np.abs(refl), 0.005)
    Jc = (dY / sig2[:, None]) @ T_s
    U_, S_, Vt_ = np.linalg.svd(Jc)
    ss = np.sort(S_)
    ng = int(np.sum(S_ < S_[0] * 1e-4))
    cond = S_[0] / ss[ng] if ng < len(ss) else np.inf
    Jp = np.linalg.pinv(Jc, rcond=ss[max(ng, 3)] * 0.5 / S_[0])
    sq = np.sqrt(np.sum(Jp ** 2, axis=1)) * RANGE_q
    rank = int(np.sum(np.linalg.svd(dY, compute_uv=False) > 1e-8))
    print(f'\n=== 投板前判决 (真实PCB配置) ===')
    print(f'无先验 61x95: rank {rank}/95')
    print(f'柔性先验 61x57: 规范模式 {ng}, 去规范 cond {cond:.1f} '
          f'(理想仿真 9.3)')
    print(f'分辨率(调零): w {np.mean(sq[:NU])*1000:.2f}um, '
          f'面内差分 {np.mean(sq[NU:])*1000:.2f}um (理想 0.8/3.1)')
    print(f'总耗时 {time.time()-t0:.0f}s')
    out = dict(stagger_mm=STAG, freq_MHz=F/1e6, n_units=NU,
               refl_self_range=[float(np.min(np.abs(refl[:NU]))),
                                float(np.max(np.abs(refl[:NU])))],
               rank_noprior=rank, n_gauge=ng, cond_gauge_fixed=float(cond),
               res_w_um=float(np.mean(sq[:NU])*1000),
               res_uv_um=float(np.mean(sq[NU:])*1000))
    json.dump(out, open(REP + 'pcb_verdict.json', 'w'), indent=1)
    print('saved pcb_verdict.json')
