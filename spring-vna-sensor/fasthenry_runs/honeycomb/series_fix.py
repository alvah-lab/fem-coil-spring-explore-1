#!/usr/bin/env python3
"""方案B 修复分析: (1) y0 分布诊断 (2) 噪声地板归一 (3) 弱模式形态
(4) 规范固定 + 阻尼端到端 (5) 规范固定后的分辨率
"""
import numpy as np
import json, time
from series_scheme import (UNITS, NU, ADJ, EDGES, OBS_B, NB, CI, DOFS, RANGE,
                           T_phys, RANGE_q, y0, J_B, dM_A, observe_B, T_s, J_A)

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
print('--- y0 诊断 ---')
y0_edge = y0[NU:]
print(f'edge y0: min|{np.min(np.abs(y0_edge)):.4f}| max|{np.max(np.abs(y0_edge)):.4f}| nH')
print('最小的5个 |y0_edge|:', np.sort(np.abs(y0_edge))[:5])

# ---- 噪声地板归一: sigma_k = 1e-3 * max(|y0_k|, FLOOR) ----
FLOOR = 0.5   # nH, 绝对测量地板对应 0.5nH x 1e-3 = 0.5pH
sig = 1e-3 * np.maximum(np.abs(y0), FLOOR)
# J 重新加权: 行 = dY(全量程)/sigma  (无量纲 SNR Jacobian)
dY = J_B * y0[:, None]                 # nH / 全量程
Jw = dY / sig[:, None]
Jwc = Jw @ T_s
U_, S_, Vt_ = np.linalg.svd(Jwc)
print(f'\n加权 61x57: 最小6 sv: {np.sort(S_)[:6]}')
n_gauge = int(np.sum(S_ < S_[0] * 1e-3))
print(f'规范模式数: {n_gauge}, 去规范后 cond = {S_[0]/np.sort(S_)[n_gauge]:.1f}')

# 弱模式形态 (前3弱)
for k in range(1, 4):
    m = Vt_[-k]
    w_, u_, v_ = m[:NU], m[NU:2*NU], m[2*NU:]
    # 旋转模式检测: u,v 与切向场的相关
    pos = np.array([[UNITS[i][0], UNITS[i][1]] for i in range(NU)])
    tang = np.stack([-pos[:, 1], pos[:, 0]], 1)
    tang /= np.linalg.norm(tang)
    rot_corr = abs(np.concatenate([u_, v_]) @ np.concatenate([tang[:, 0], tang[:, 1]]))
    print(f'弱模式{k} (sv={np.sort(S_)[k-1]:.2e}): |w|={np.linalg.norm(w_):.2f} '
          f'|u|={np.linalg.norm(u_):.2f} |v|={np.linalg.norm(v_):.2f} '
          f'mean_u={np.mean(u_):+.3f} mean_v={np.mean(v_):+.3f} rot_corr={rot_corr:.2f}')

# ---- 端到端 (规范固定: 截断掉3个规范模式; 阻尼0.7) ----
sig_w = 5.2 * 1.2
w_true = -0.4 * np.exp(-np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)])**2
                       / (2 * sig_w ** 2))
u_true = np.full(NU, 0.15)      # 共模剪切: 属于规范盲方向, 预期不可恢复
v_true = np.zeros(NU)
q_true = np.concatenate([w_true, u_true, v_true])
pose_true = (T_phys @ q_true).reshape(NU, 5)
y_meas = observe_B(pose_true)

CUT = np.sort(S_)[n_gauge] * 0.5          # 只截掉规范模式
Sinv = np.where(S_ > CUT, 1 / np.maximum(S_, 1e-30), 0.0)
Jp = (Vt_.T * Sinv) @ U_.T[:len(S_)]
DAMP = 0.7
qs = np.zeros(3 * NU)
t0 = time.time()
for it in range(20):
    pose = (T_phys @ (qs * RANGE_q)).reshape(NU, 5)
    r = (y_meas - observe_B(pose)) / sig
    dq = Jp @ r
    qs += DAMP * dq
    if np.linalg.norm(dq) < 1e-6:
        break
q_est = qs * RANGE_q
# 误差按可观测子空间评估: 把真值投影掉规范方向
gauge = Vt_[np.argsort(S_)[:n_gauge]]             # 规范模式行 (Vt_行k对应S_[k])
q_true_s = q_true / RANGE_q
q_true_obs = q_true_s - gauge.T @ (gauge @ q_true_s)
q_err_obs = (qs - q_true_obs) * RANGE_q
ew = np.abs(q_err_obs[:NU]) * 1000
euv = np.abs(q_err_obs[NU:]) * 1000
print(f'\n端到端 ({it+1} 迭代, {time.time()-t0:.0f}s, |dq|={np.linalg.norm(dq):.2e}):')
print(f'  w 场最大误差 {np.max(ew):.2f} um; 中心压深 {-q_est[CI]*1000:.1f} um '
      f'(真值 {-w_true[CI]*1000:.1f})')
print(f'  可观测面内分量最大误差 {np.max(euv):.2f} um')
print(f'  共模剪切 (规范盲方向) 恢复 {np.mean(q_est[NU:2*NU])*1000:.1f} um '
      f'(真值150, 预期≈0, 由边界/力学先验另行固定)')

# ---- 噪声 MC (规范固定) ----
rng = np.random.default_rng(31)
errs = []
for trial in range(10):
    yn = y_meas + sig * rng.standard_normal(NB)
    qs = np.zeros(3 * NU)
    for it2 in range(10):
        pose = (T_phys @ (qs * RANGE_q)).reshape(NU, 5)
        qs += DAMP * (Jp @ ((yn - observe_B(pose)) / sig))
    errs.append((qs - q_true_obs) * RANGE_q)
errs = np.array(errs)
nw = np.std(errs[:, :NU]) * 1000
nuv = np.std(errs[:, NU:]) * 1000
print(f'噪声 (0.5pH地板+1e-3相对): w 1σ={nw:.2f}um, 面内 1σ={nuv:.2f}um')

# 方案A同先验同噪声模型分辨率 (线性)
M0_A = np.load(REP + 'honeycomb_spectrum19.npz')['M0_stag']
sigA = 1e-3 * np.maximum(np.abs(M0_A), FLOOR)
JwA = (J_A * M0_A[:, None]) / sigA[:, None]
JwAc = JwA @ T_s
sA = np.sqrt(np.sum(np.linalg.pinv(JwAc) ** 2, axis=1)) * RANGE_q
print(f'对照 方案A+先验: w 1σ={np.mean(sA[:NU])*1000:.2f}um, '
      f'面内 1σ={np.mean(sA[NU:])*1000:.2f}um, '
      f'cond={np.linalg.cond(JwAc):.1f}')

out = json.load(open(REP + 'honeycomb_series.json'))
out['fixed'] = dict(
    floor_nH=FLOOR, n_gauge=n_gauge,
    sv_weighted=np.sort(S_).tolist(),
    cond_gauge_fixed=float(S_[0] / np.sort(S_)[n_gauge]),
    e2e=dict(w_max_err_um=float(np.max(ew)),
             uv_obs_max_err_um=float(np.max(euv)),
             depth_um=float(-q_est[CI] * 1000),
             w_true=w_true.tolist(), w_est=q_est[:NU].tolist()),
    noise=dict(w_um=float(nw), uv_um=float(nuv)),
    noise_A=dict(w_um=float(np.mean(sA[:NU]) * 1000),
                 uv_um=float(np.mean(sA[NU:]) * 1000)))
with open(REP + 'honeycomb_series.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved (fixed)')
