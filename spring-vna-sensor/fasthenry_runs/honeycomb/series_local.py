#!/usr/bin/env python3
"""方案B 判决性实验: 真值邻域局部可反演性
(1) 真值处中心差分 J (d=0.01), 线性一致性: J@dq vs 实际 dy (5um 尺度扰动)
(2) 从 真值+扰动 出发 3 次 GN -> 是否收敛回真值
"""
import numpy as np
import json, time
from series_scheme import UNITS, NU, CI, T_phys, RANGE_q, y0, observe_B as _obs

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
sig = 1e-3 * np.maximum(np.abs(y0), 0.5)
N_GAUGE = 3
pose_of = lambda q: (T_phys @ q).reshape(NU, 5)

def observe(poses):
    assert np.all(np.abs(poses[:, 2]) < 1.2) and \
           np.all(np.abs(poses[:, 3:]) < 0.35), 'pose 超界'
    return _obs(poses)

def jac_c(q, d):
    """中心差分 q 空间 Jacobian (114 解)"""
    J = np.zeros((len(y0), 3 * NU))
    for c in range(3 * NU):
        qp = q.copy(); qp[c] += d * RANGE_q[c]
        qm = q.copy(); qm[c] -= d * RANGE_q[c]
        J[:, c] = (observe(pose_of(qp)) - observe(pose_of(qm))) / (2 * d) / sig
    return J

r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
w_true = -0.4 * np.exp(-r2 / (2 * (5.2 * 1.2) ** 2))
q_true = np.concatenate([w_true, np.full(NU, 0.15), np.zeros(NU)])
y_true = observe(pose_of(q_true))

t0 = time.time()
print('真值处中心差分 J (d=0.01, 114 解)...', flush=True)
J = jac_c(q_true, 0.01)
U, S, Vt = np.linalg.svd(J)
ss = np.sort(S)
print(f'真值处 sv 最小6: {ss[:6]}  去规范 cond={S[0]/ss[N_GAUGE]:.1f}', flush=True)

# ---- (1) 线性一致性 @ 5um/微小尺度 ----
rng = np.random.default_rng(7)
dq = rng.standard_normal(3 * NU)
dq = dq / np.linalg.norm(dq) * 0.01           # 全量程的1% ~ 7.5um w
y2 = observe(pose_of(q_true + dq * RANGE_q))
r_act = (y2 - y_true) / sig
r_lin = J @ dq
dev = np.linalg.norm(r_act - r_lin) / np.linalg.norm(r_act)
print(f'线性一致性 (1%全量程随机扰动): |act|={np.linalg.norm(r_act):.2f} '
      f'偏差={dev*100:.1f}%', flush=True)

# ---- (2) 局部 GN: 从 真值+5%全量程扰动 出发 ----
dq0 = rng.standard_normal(3 * NU)
dq0 = dq0 / np.linalg.norm(dq0) * 0.05
q = q_true + dq0 * RANGE_q
err0 = np.linalg.norm((q - q_true)) * 1000
print(f'\n起点偏离 |dq|={err0:.1f} (范数, um混合)', flush=True)
gauge = Vt[np.argsort(S)[:N_GAUGE]]
for it in range(3):
    Ji = jac_c(q, 0.01)
    Ui, Si, Vti = np.linalg.svd(Ji)
    cut = np.sort(Si)[N_GAUGE] * 0.5
    Sinv = np.where(Si > cut, 1 / np.maximum(Si, 1e-30), 0.0)
    Jp = (Vti.T * Sinv) @ Ui.T[:len(Si)]
    r = (y_true - observe(pose_of(q))) / sig
    q = q + (Jp @ r) * RANGE_q
    # 误差投影掉规范
    e_s = (q - q_true) / RANGE_q
    e_obs = (e_s - gauge.T @ (gauge @ e_s)) * RANGE_q
    print(f'GN it{it+1}: |r|={np.linalg.norm(r):8.2f}  '
          f'可观测误差范数={np.linalg.norm(e_obs)*1000:8.2f} um  '
          f'w最大误差={np.max(np.abs(e_obs[:NU]))*1000:.2f} um '
          f'({time.time()-t0:.0f}s)', flush=True)

out = json.load(open(REP + 'honeycomb_series.json'))
out['local_test'] = dict(
    sv_min6_at_truth_central=ss[:6].tolist(),
    cond_gauge_fixed=float(S[0] / ss[N_GAUGE]),
    linearity_dev_1pct=float(dev),
    final_w_max_err_um=float(np.max(np.abs(e_obs[:NU])) * 1000),
    final_err_norm_um=float(np.linalg.norm(e_obs) * 1000))
with open(REP + 'honeycomb_series.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved local_test')
