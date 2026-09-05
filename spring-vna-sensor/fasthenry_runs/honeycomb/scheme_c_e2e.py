#!/usr/bin/env python3
"""方案C 端到端 v2: 截掉3个弱共模方向(慢通道另解), 阻尼+步长限幅拟牛顿"""
import numpy as np
import json, time
from scheme_c_analysis import observe, pose_of, NU, CI, UNITS, RANGE_q
from series_scheme import T_s

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
npz = np.load(REP + 'scheme_c_jacobian.npz')
dY, y0, sig2 = npz['dY'], npz['y0'], npz['sig2']

Jc = (dY / sig2[:, None]) @ T_s
U_, S_, Vt_ = np.linalg.svd(Jc)
ss = np.sort(S_)
N_WEAK = 3
cut = ss[N_WEAK] * 0.5
Sinv = np.where(S_ > cut, 1 / np.maximum(S_, 1e-30), 0.0)
Jp = (Vt_.T * Sinv) @ U_.T[:len(S_)]
weak = Vt_[np.argsort(S_)[:N_WEAK]]
print(f'截断: 弱模式 sv {ss[:N_WEAK]} (含共模u/v+旋转), '
      f'强子空间 cond={S_[0]/ss[N_WEAK]:.1f}', flush=True)

r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
w_true = -0.4 * np.exp(-r2 / (2 * (5.2 * 1.2) ** 2))
q_true = np.concatenate([w_true, np.full(NU, 0.15), np.zeros(NU)])
y_meas = observe(pose_of(q_true), 'schc_e2e2')

t0 = time.time()
q = np.zeros(3 * NU)
for it in range(15):
    r = (y_meas - observe(pose_of(q))) / sig2
    dq = Jp @ r
    dq = np.clip(dq, -0.25, 0.25)            # 步长限幅 (全量程单位)
    q = q + 0.6 * dq * RANGE_q
    nrm = np.linalg.norm(dq)
    print(f'it{it+1}: |r|={np.linalg.norm(r):9.2f} |dq|={nrm:.4f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    if nrm < 5e-4:
        break

qt_s = q_true / RANGE_q
qt_obs = (qt_s - weak.T @ (weak @ qt_s)) * RANGE_q      # 可观测子空间真值
err = q - qt_obs
print(f'\n可观测子空间误差: w 最大 {np.max(np.abs(err[:NU]))*1000:.2f} um, '
      f'面内最大 {np.max(np.abs(err[NU:]))*1000:.2f} um')
print(f'中心压深 {-q[CI]*1000:.1f} um (真值 {-w_true[CI]*1000:.1f}); '
      f'共模剪切(弱方向,慢通道另解) 恢复 {np.mean(q[NU:2*NU])*1000:.1f} um')

# 弱共模慢通道: 收敛点处专解3个弱方向 (小步高阻尼)
q_slow = q.copy()
Sinv_w = np.where((S_ <= cut) & (S_ > 1e-6), 1 / np.maximum(S_, 1e-30), 0.0)
Jp_w = (Vt_.T * Sinv_w) @ U_.T[:len(S_)]
for it2 in range(6):
    r = (y_meas - observe(pose_of(q_slow))) / sig2
    dqw = np.clip(Jp_w @ r, -0.1, 0.1)
    q_slow = q_slow + 0.5 * dqw * RANGE_q
err_full = q_slow - q_true
print(f'加弱通道后: 共模剪切恢复 {np.mean(q_slow[NU:2*NU])*1000:.1f}/150.0 um, '
      f'全空间 w 最大误差 {np.max(np.abs(err_full[:NU]))*1000:.2f} um, '
      f'面内最大 {np.max(np.abs(err_full[NU:]))*1000:.2f} um', flush=True)

# 噪声 MC (强子空间)
rng = np.random.default_rng(61)
errs = []
for trial in range(8):
    yn = y_meas + sig2 * rng.standard_normal(len(y0))
    qn = q.copy()
    for it3 in range(4):
        qn = qn + 0.6 * np.clip(Jp @ ((yn - observe(pose_of(qn))) / sig2),
                                -0.25, 0.25) * RANGE_q
    errs.append(qn - q)
errs = np.array(errs)
print(f'噪声MC(S2, 强子空间): w 1σ={np.std(errs[:,:NU])*1000:.2f} um, '
      f'面内 1σ={np.std(errs[:,NU:])*1000:.2f} um')

out = json.load(open(REP + 'honeycomb_scheme_c.json'))
out['e2e_v2'] = dict(
    w_max_err_um=float(np.max(np.abs(err[:NU])) * 1000),
    uv_obs_max_err_um=float(np.max(np.abs(err[NU:])) * 1000),
    depth_um=float(-q[CI] * 1000),
    cm_u_slow_um=float(np.mean(q_slow[NU:2 * NU]) * 1000),
    full_w_max_err_um=float(np.max(np.abs(err_full[:NU])) * 1000),
    full_uv_max_err_um=float(np.max(np.abs(err_full[NU:])) * 1000),
    mc_w_um=float(np.std(errs[:, :NU]) * 1000),
    mc_uv_um=float(np.std(errs[:, NU:]) * 1000))
with open(REP + 'honeycomb_scheme_c.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved e2e_v2')
