#!/usr/bin/env python3
"""完整验证: 19单元 (38环) 三色格交错锚点 0.9mm
(a) 联合 Jacobian (95 未知) SVD: 共模模式是否被提升, 剩余规范模式数
(b) 端到端: 联动点压 + 共模剪切 + 共模倾斜 (以前的简并方向), 截断SVD反演
(c) 1e-3 噪声蒙特卡洛
"""
import numpy as np
import json, time, sys
import gen_ring_array as G

PITCH, GAP_NOM, STAG = 5.2, 1.75, 0.9
DOFS = ['x', 'y', 'z', 'tax', 'tay']
RANGE = dict(x=0.5, y=0.5, z=0.75, tax=np.deg2rad(10), tay=np.deg2rad(10))
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))

# ---- 19单元三角格 + 三色 ----
a1 = np.array([PITCH, 0.0]); a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS = []          # (cx, cy, color)
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            UNITS.append((p[0], p[1], (i - j) % 3))
NU = len(UNITS)
BOT_Z = [-STAG if c == 0 else 0.0 for _, _, c in UNITS]
# 邻接
ADJ = {i: [] for i in range(NU)}
for i in range(NU):
    for j in range(NU):
        if i != j and np.hypot(UNITS[i][0] - UNITS[j][0],
                               UNITS[i][1] - UNITS[j][1]) < PITCH * 1.05:
            ADJ[i].append(j)
OBS = []
for i in range(NU):
    OBS.append(('TB', i, i))
    for j in ADJ[i]:
        OBS.append(('TB', i, j))
EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})
for i, j in EDGES:
    OBS.append(('TT', i, j))
NOBS, NUNK = len(OBS), NU * 5
print(f'{NU} units ({sum(1 for u in UNITS if u[2]==0)} lowered), '
      f'obs={NOBS} (TT edges={len(EDGES)}), unknowns={NUNK}', flush=True)

def build(poses):
    rings = []
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, GAP_NOM + dz, tax, tay)))
    for u, (cx, cy, c) in enumerate(UNITS):
        rings.append((f'B{u}', (cx, cy, BOT_Z[u], 0.0, 0.0)))
    return rings

def observe(poses):
    rings = build(poses)
    Lm, _ = G.solve(rings, 'full19_tmp')
    names = [r[0] for r in rings]
    idx = {n: k for k, n in enumerate(names)}
    v = [Lm[idx[f'T{i}'], idx[f'B{j}' if k == 'TB' else f'T{j}']]
         for k, i, j in OBS]
    return np.array(v) * 1e9

t_start = time.time()
base = np.zeros((NU, 5))
M0 = observe(base)
J = np.zeros((NOBS, NUNK))
col = 0
for u in range(NU):
    for d, dof in enumerate(DOFS):
        dd = DELTA[dof]
        pp = base.copy(); pp[u, d] += dd
        pm = base.copy(); pm[u, d] -= dd
        J[:, col] = (observe(pp) - observe(pm)) / (2 * dd) * RANGE[dof] / M0
        col += 1
    print(f'  unit {u+1}/{NU} done ({time.time()-t_start:.0f}s)', flush=True)

U, S, Vt = np.linalg.svd(J)
print('\n最小8个奇异值:', np.array2string(np.sort(S)[:8], precision=5), flush=True)
# 规范模式判定: sv < 0.01
n_gauge = int(np.sum(S < 0.01))
print(f'近零模式数 (sv<0.01): {n_gauge}')
# 共模基重叠检查
def common(dof):
    e = np.zeros((NU, 5)); e[:, DOFS.index(dof)] = 1
    return e.flatten() / np.linalg.norm(e)
for name in ['x', 'y', 'tax', 'tay']:
    b = common(name)
    # b 在弱空间 (sv<0.05) 的投影
    weak = Vt[S < 0.05]
    frac = np.linalg.norm(weak @ b) if len(weak) else 0.0
    print(f'  共模 {name}: 弱空间投影 {frac*100:.1f}% '
          f'(基线无交错时接近 100%)')

# ---- (b) 端到端: 联动 + 共模 ----
truth = np.zeros((NU, 5))
truth[0, 2] = -0.4                      # 中心(索引找 0,0)
ci = min(range(NU), key=lambda u: np.hypot(UNITS[u][0], UNITS[u][1]))
truth = np.zeros((NU, 5))
truth[ci, 2] = -0.4
for u in range(NU):
    if u == ci:
        continue
    r = np.hypot(UNITS[u][0] - UNITS[ci][0], UNITS[u][1] - UNITS[ci][1])
    if r < PITCH * 1.05:                # 联动邻居
        truth[u, 2] = -0.3 * 0.4
        phi = np.arctan2(UNITS[u][1], UNITS[u][0])
        t = np.deg2rad(3.0)
        truth[u, 3] = t * np.sin(phi)
        truth[u, 4] = -t * np.cos(phi)
truth[:, 0] += 0.15                     # 共模剪切 x (以前简并方向)
truth[:, 4] += np.deg2rad(2.0)          # 共模倾斜 tay (以前简并方向)

M_meas = observe(truth)
scale = np.array([RANGE[d] for d in DOFS] * NU)
CUT = 0.02
Sinv = np.where(S > CUT, 1 / np.maximum(S, 1e-30), 0.0)
Jp = (Vt.T * Sinv) @ U.T[:len(S)]
x = np.zeros(NUNK)
for it in range(12):
    r = (M_meas - observe((x * scale).reshape(NU, 5))) / M0
    dx = Jp @ r
    x += dx
    if np.linalg.norm(dx) < 1e-7:
        break
est = (x * scale).reshape(NU, 5)
err = est - truth
print(f'\n端到端 ({it+1} 迭代, |r|={np.linalg.norm(r):.2e}):')
print(f'  共模剪切恢复: mean x = {np.mean(est[:,0])*1000:.1f} um (真值 150.0)')
print(f'  共模倾斜恢复: mean tay = {np.rad2deg(np.mean(est[:,4])):.3f} deg (真值 2.000)')
print(f'  中心压深: {-est[ci,2]:.4f} mm (真值 0.400)')
print(f'  最大误差: z {np.max(np.abs(err[:,2]))*1000:.2f} um, '
      f'xy {np.max(np.abs(err[:,:2]))*1000:.2f} um, '
      f'tilt {np.rad2deg(np.max(np.abs(err[:,3:]))):.3f} deg')

# ---- (c) 噪声 MC ----
rng = np.random.default_rng(11)
errs = []
for trial in range(10):
    Mn = M_meas * (1 + 1e-3 * rng.standard_normal(NOBS))
    x = np.zeros(NUNK)
    for it2 in range(6):
        r = (Mn - observe((x * scale).reshape(NU, 5))) / M0
        x += Jp @ r
    errs.append((x * scale).reshape(NU, 5) - truth)
    print(f'  MC {trial+1}/10', flush=True)
errs = np.array(errs)
cme = np.mean(errs[:, :, 0], axis=1)         # 每次试验的共模x误差
cmt = np.rad2deg(np.mean(errs[:, :, 4], axis=1))
print(f'\n噪声1e-3 (10次): z 1σ={np.std(errs[:,:,2])*1000:.2f}um, '
      f'xy 1σ={np.std(errs[:,:,:2])*1000:.2f}um, '
      f'tilt 1σ={np.rad2deg(np.std(errs[:,:,3:]))*60:.1f} arcmin')
print(f'  共模剪切 1σ={np.std(cme)*1000:.2f}um, 共模倾斜 1σ={np.std(cmt)*60:.2f} arcmin')

out = dict(n_units=NU, n_obs=NOBS, n_unknown=NUNK,
           sv_min8=np.sort(S)[:8].tolist(), n_gauge=n_gauge,
           e2e=dict(cm_x_um=float(np.mean(est[:, 0]) * 1000),
                    cm_tay_deg=float(np.rad2deg(np.mean(est[:, 4]))),
                    depth_mm=float(-est[ci, 2]),
                    max_err_z_um=float(np.max(np.abs(err[:, 2])) * 1000),
                    max_err_xy_um=float(np.max(np.abs(err[:, :2])) * 1000),
                    max_err_tilt_deg=float(np.rad2deg(np.max(np.abs(err[:, 3:]))))),
           noise=dict(z_um=float(np.std(errs[:, :, 2]) * 1000),
                      xy_um=float(np.std(errs[:, :, :2]) * 1000),
                      tilt_arcmin=float(np.rad2deg(np.std(errs[:, :, 3:])) * 60),
                      cm_x_um=float(np.std(cme) * 1000),
                      cm_tay_arcmin=float(np.std(cmt) * 60)),
           total_time_s=time.time() - t_start)
jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_full19.json'
with open(jp, 'w') as f:
    json.dump(out, f, indent=1)
print('saved', jp, f'({out["total_time_s"]:.0f}s total)')
