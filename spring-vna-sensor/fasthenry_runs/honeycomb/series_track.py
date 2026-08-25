#!/usr/bin/env python3
"""方案B 跟踪模式验证: 增量加载 (5 步 ramp), 每步从上一步热启动 + 2 次 GN
模拟真实使用: 传感器连续读数, 变形渐进, 求解器始终在收敛域内
"""
import numpy as np
import json, time
from series_scheme import UNITS, NU, CI, T_phys, RANGE_q, y0, observe_B as _obs

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
sig = 1e-3 * np.maximum(np.abs(y0), 0.5)
N_GAUGE = 3

def observe(poses):
    assert np.all(np.abs(poses[:, :2]) < 1.0) and \
           np.all(np.abs(poses[:, 2]) < 1.2) and \
           np.all(np.abs(poses[:, 3:]) < 0.35), 'pose 超界'
    return _obs(poses)

pose_of = lambda q: (T_phys @ q).reshape(NU, 5)

def jac_q(q, y_at=None, d=0.05):
    if y_at is None:
        y_at = observe(pose_of(q))
    J = np.zeros((len(y0), 3 * NU))
    for c in range(3 * NU):
        qp = q.copy(); qp[c] += d * RANGE_q[c]
        J[:, c] = (observe(pose_of(qp)) - y_at) / d / sig
    return J, y_at

def trunc_pinv(J):
    U, S, Vt = np.linalg.svd(J)
    cut = np.sort(S)[N_GAUGE] * 0.5
    Sinv = np.where(S > cut, 1 / np.maximum(S, 1e-30), 0.0)
    return (Vt.T * Sinv) @ U.T[:len(S)], S, Vt

r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
w_full = -0.4 * np.exp(-r2 / (2 * (5.2 * 1.2) ** 2))
u_full = np.full(NU, 0.15)
q_full = np.concatenate([w_full, u_full, np.zeros(NU)])

t0 = time.time()
q = np.zeros(3 * NU)
gauge0 = None
hist = []
for tf in [0.2, 0.4, 0.6, 0.8, 1.0]:
    q_t = tf * q_full
    y_t = observe(pose_of(q_t))
    for sub in range(2):
        J, y_cur = jac_q(q)
        Jp, S, Vt = trunc_pinv(J)
        if gauge0 is None:
            gauge0 = Vt[np.argsort(S)[:N_GAUGE]]
        r = (y_t - y_cur) / sig
        dq = np.clip(Jp @ r, -0.3, 0.3)
        q = q + 0.9 * dq * RANGE_q
    # 该步误差 (可观测子空间)
    qt_obs = (q_t / RANGE_q - gauge0.T @ (gauge0 @ (q_t / RANGE_q))) * RANGE_q
    err = q - qt_obs
    ew = np.max(np.abs(err[:NU])) * 1000
    euv = np.max(np.abs(err[NU:])) * 1000
    hist.append(dict(t=tf, r=float(np.linalg.norm(r)), w_err_um=float(ew),
                     uv_err_um=float(euv)))
    print(f'ramp t={tf}: |r|={np.linalg.norm(r):7.2f}  w误差={ew:7.2f}um  '
          f'面内误差={euv:6.2f}um  ({time.time()-t0:.0f}s)', flush=True)

print(f'\n最终: 中心压深 {-q[CI]*1000:.1f} um (真值 {-w_full[CI]*1000:.1f}), '
      f'共模剪切恢复 {np.mean(q[NU:2*NU])*1000:.1f} um (规范盲)')
out = json.load(open(REP + 'honeycomb_series.json'))
out['tracking'] = dict(ramp=hist, depth_um=float(-q[CI] * 1000),
                       depth_true_um=float(-w_full[CI] * 1000),
                       w_true=w_full.tolist(), w_est=q[:NU].tolist(),
                       total_time_s=time.time() - t0)
with open(REP + 'honeycomb_series.json', 'w') as f:
    json.dump(out, f, indent=1)
print(f'saved tracking ({time.time()-t0:.0f}s)')
