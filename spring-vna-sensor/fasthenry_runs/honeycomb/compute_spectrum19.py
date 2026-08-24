#!/usr/bin/env python3
"""19单元联合 Jacobian 完整奇异值谱: 无交错基线 vs 三色格下沉0.9mm
存 J 矩阵 + 全谱到 reports/honeycomb_spectrum19.npz
"""
import numpy as np
import time
import gen_ring_array as G

PITCH, GAP_NOM, STAG = 5.2, 1.75, 0.9
DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))

a1 = np.array([PITCH, 0.0]); a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            UNITS.append((p[0], p[1], (i - j) % 3))
NU = len(UNITS)
ADJ = {i: [j for j in range(NU) if i != j and
           np.hypot(UNITS[i][0] - UNITS[j][0],
                    UNITS[i][1] - UNITS[j][1]) < PITCH * 1.05]
       for i in range(NU)}
OBS = []
for i in range(NU):
    OBS.append(('TB', i, i))
    for j in ADJ[i]:
        OBS.append(('TB', i, j))
for i, j in sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]}):
    OBS.append(('TT', i, j))
NOBS, NUNK = len(OBS), NU * 5

def observe(poses, bot_z):
    rings = []
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, GAP_NOM + dz, tax, tay)))
    for u, (cx, cy, c) in enumerate(UNITS):
        rings.append((f'B{u}', (cx, cy, bot_z[u], 0.0, 0.0)))
    Lm, _ = G.solve(rings, 'spec19_tmp')
    names = [r[0] for r in rings]
    idx = {n: k for k, n in enumerate(names)}
    return np.array([Lm[idx[f'T{i}'], idx[f'B{j}' if k == 'TB' else f'T{j}']]
                     for k, i, j in OBS]) * 1e9

def jac(bot_z, tag):
    t0 = time.time()
    base = np.zeros((NU, 5))
    M0 = observe(base, bot_z)
    J = np.zeros((NOBS, NUNK))
    col = 0
    for u in range(NU):
        for d, dof in enumerate(DOFS):
            dd = DELTA[dof]
            pp = base.copy(); pp[u, d] += dd
            pm = base.copy(); pm[u, d] -= dd
            J[:, col] = (observe(pp, bot_z) - observe(pm, bot_z)) / (2 * dd) \
                        * RANGE[dof] / M0
            col += 1
        print(f'  [{tag}] unit {u+1}/{NU} ({time.time()-t0:.0f}s)', flush=True)
    return J, M0

bz_flat = [0.0] * NU
bz_stag = [-STAG if c == 0 else 0.0 for _, _, c in UNITS]
print(f'{NU} units, obs={NOBS}, unknowns={NUNK}', flush=True)
J_flat, M0_flat = jac(bz_flat, 'flat')
J_stag, M0_stag = jac(bz_stag, 'stag')
sv_flat = np.linalg.svd(J_flat, compute_uv=False)
sv_stag = np.linalg.svd(J_stag, compute_uv=False)
print('flat  min5:', np.sort(sv_flat)[:5])
print('stag  min5:', np.sort(sv_stag)[:5])
np.savez('/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_spectrum19.npz',
         sv_flat=sv_flat, sv_stag=sv_stag, J_flat=J_flat, J_stag=J_stag,
         M0_flat=M0_flat, M0_stag=M0_stag,
         units=np.array([(u[0], u[1], u[2]) for u in UNITS]))
print('saved honeycomb_spectrum19.npz')
