#!/usr/bin/env python3
"""方案B 端到端 v2: 线性度诊断 + 同伦渐进 + 需要时刷新 Jacobian"""
import numpy as np
import json, time
from series_scheme import (UNITS, NU, CI, DOFS, RANGE, T_phys, RANGE_q,
                           y0, observe_B, T_s, J_B, J_A)

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
FLOOR = 0.5
sig = 1e-3 * np.maximum(np.abs(y0), FLOOR)
Jw = (J_B * y0[:, None]) / sig[:, None]
Jwc = Jw @ T_s
U_, S_, Vt_ = np.linalg.svd(Jwc)
n_gauge = 3
gauge = Vt_[np.argsort(S_)[:n_gauge]]
CUT = np.sort(S_)[n_gauge] * 0.5

def make_pinv(Um, Sm, Vtm, cut):
    Sinv = np.where(Sm > cut, 1 / np.maximum(Sm, 1e-30), 0.0)
    return (Vtm.T * Sinv) @ Um.T[:len(Sm)]

Jp = make_pinv(U_, S_, Vt_, CUT)

# 真值
sw = 5.2 * 1.2
w_true = -0.4 * np.exp(-np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)])**2
                       / (2 * sw ** 2))
u_true = np.full(NU, 0.15); v_true = np.zeros(NU)
q_true = np.concatenate([w_true, u_true, v_true])
q_true_s = q_true / RANGE_q
pose_true = (T_phys @ q_true).reshape(NU, 5)
y_base = observe_B(np.zeros((NU, 5)))
y_meas = observe_B(pose_true)

# ---- 线性度诊断 ----
r_actual = (y_meas - y_base) / sig
r_linear = Jwc @ q_true_s
num = np.linalg.norm(r_actual - r_linear) / np.linalg.norm(r_actual)
print(f'线性度: |actual-linear|/|actual| = {num*100:.1f}%  '
      f'(|actual|={np.linalg.norm(r_actual):.1f})')
# 最不线性的 5 个观测
dev = np.abs(r_actual - r_linear)
worst = np.argsort(dev)[::-1][:5]
print('偏差最大观测 idx:', worst.tolist(), '偏差:', np.round(dev[worst], 1).tolist())

def gn(y_target, q0, Jp_use, iters, damp):
    qs = q0.copy()
    for it in range(iters):
        pose = (T_phys @ (qs * RANGE_q)).reshape(NU, 5)
        r = (y_target - observe_B(pose)) / sig
        dq = Jp_use @ r
        qs += damp * dq
        if np.linalg.norm(dq) < 1e-6:
            break
    return qs, np.linalg.norm(r), np.linalg.norm(dq)

# ---- 同伦: 4 步渐进到全测量 ----
t0 = time.time()
qs = np.zeros(3 * NU)
for tfrac in [0.25, 0.5, 0.75, 1.0]:
    yt = y_base + tfrac * (y_meas - y_base)
    qs, rn, dn = gn(yt, qs, Jp, 10, 0.5)
    print(f'同伦 t={tfrac}: |r|={rn:.2f} |dq|={dn:.2e}')

if dn > 1e-3:
    print('同伦未收敛 -> 在当前估计处刷新 Jacobian (114 解)...')
    def jac_q(qs_at):
        base_pose = (T_phys @ (qs_at * RANGE_q)).reshape(NU, 5)
        Jn = np.zeros((len(y0), 3 * NU))
        d = 0.04
        for c in range(3 * NU):
            qp = qs_at.copy(); qp[c] += d
            qm = qs_at.copy(); qm[c] -= d
            yp = observe_B((T_phys @ (qp * RANGE_q)).reshape(NU, 5))
            ym = observe_B((T_phys @ (qm * RANGE_q)).reshape(NU, 5))
            Jn[:, c] = (yp - ym) / (2 * d) / sig
        return Jn
    Jn = jac_q(qs)
    Un, Sn, Vtn = np.linalg.svd(Jn)
    cutn = np.sort(Sn)[n_gauge] * 0.5
    Jpn = make_pinv(Un, Sn, Vtn, cutn)
    qs, rn, dn = gn(y_meas, qs, Jpn, 15, 0.7)
    print(f'刷新后: |r|={rn:.2f} |dq|={dn:.2e}')

q_est = qs * RANGE_q
q_true_obs = (q_true_s - gauge.T @ (gauge @ q_true_s)) * RANGE_q
ew = np.abs(q_est[:NU] - q_true_obs[:NU]) * 1000
euv = np.abs(q_est[NU:] - q_true_obs[NU:]) * 1000
print(f'\n结果 ({time.time()-t0:.0f}s): w 场最大误差 {np.max(ew):.2f} um, '
      f'中心压深 {-q_est[CI]*1000:.1f} um (真值 {-w_true[CI]*1000:.1f})')
print(f'可观测面内最大误差 {np.max(euv):.2f} um; '
      f'共模剪切恢复 {np.mean(q_est[NU:2*NU])*1000:.1f} um (规范盲, 预期0)')

# 噪声 MC (从收敛解出发, 每次少量迭代)
Jp_final = Jpn if dn <= 1e-3 and 'Jpn' in dir() else Jp
rng = np.random.default_rng(41)
errs = []
for trial in range(10):
    yn = y_meas + sig * rng.standard_normal(len(y0))
    qn, _, _ = gn(yn, qs, Jp_final, 5, 0.7)
    errs.append((qn - q_true_obs / RANGE_q * RANGE_q / RANGE_q) * RANGE_q - 0)
    errs[-1] = (qn * RANGE_q) - q_est          # 相对收敛解的散布
errs = np.array(errs)
nw = np.std(errs[:, :NU]) * 1000
nuv = np.std(errs[:, NU:]) * 1000
print(f'噪声 (0.5pH地板): w 1σ={nw:.2f} um, 面内 1σ={nuv:.2f} um')

out = json.load(open(REP + 'honeycomb_series.json'))
out['e2e_v2'] = dict(linearity_dev=float(num),
                     w_max_err_um=float(np.max(ew)),
                     uv_obs_max_err_um=float(np.max(euv)),
                     depth_um=float(-q_est[CI] * 1000),
                     depth_true_um=float(-w_true[CI] * 1000),
                     w_true=w_true.tolist(), w_est=q_est[:NU].tolist(),
                     noise_w_um=float(nw), noise_uv_um=float(nuv))
with open(REP + 'honeycomb_series.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved e2e_v2')
