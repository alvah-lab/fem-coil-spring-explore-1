#!/usr/bin/env python3
"""方案B 端到端 v3: 完整高斯-牛顿 (每步刷新 q 空间 Jacobian) + 位姿钳位
背景: 0.2mm 邻缘间隙使 TT 耦合在标称点纯二次 (一阶为零) -> 拟牛顿必发散;
      工作点处对称破缺, TT 出现一阶斜率 -> 用工作点 Jacobian 评估可解性
"""
import numpy as np
import json, time
from series_scheme import (UNITS, NU, CI, T_phys, RANGE_q, y0,
                           observe_B as _observe_B)

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
FLOOR = 0.5
sig = 1e-3 * np.maximum(np.abs(y0), FLOOR)
N_GAUGE = 3

def observe_B(poses):
    # 位姿安全钳位: 防发散位姿喂给 FastHenry (曾致 50GB OOM)
    assert np.all(np.abs(poses[:, :2]) < 1.0), 'xy 超界'
    assert np.all(np.abs(poses[:, 2]) < 1.2), 'z 超界'
    assert np.all(np.abs(poses[:, 3:]) < 0.35), 'tilt 超界'
    return _observe_B(poses)

def pose_of(q_phys):
    return (T_phys @ q_phys).reshape(NU, 5)

def jac_q(q_phys, y_at=None, d=0.05):
    """前向差分 q 空间 Jacobian (58 解), 行归一 sig"""
    if y_at is None:
        y_at = observe_B(pose_of(q_phys))
    J = np.zeros((len(y0), 3 * NU))
    for c in range(3 * NU):
        qp = q_phys.copy()
        qp[c] += d * RANGE_q[c]
        J[:, c] = (observe_B(pose_of(qp)) - y_at) / (d) / sig
    return J, y_at            # 列 = 每全量程

def trunc_pinv(J, n_cut):
    U, S, Vt = np.linalg.svd(J)
    order = np.argsort(S)
    cut = S[order[n_cut]] * 0.5 if n_cut > 0 else 0.0
    Sinv = np.where(S > cut, 1 / np.maximum(S, 1e-30), 0.0)
    return (Vt.T * Sinv) @ U.T[:len(S)], S, Vt

# ---- 真值 ----
r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
w_true = -0.4 * np.exp(-r2 / (2 * (5.2 * 1.2) ** 2))
u_true = np.full(NU, 0.15)
v_true = np.zeros(NU)
q_true = np.concatenate([w_true, u_true, v_true])
y_meas = observe_B(pose_of(q_true))

# ---- 工作点 Jacobian SVD (在真值处) ----
t0 = time.time()
print('工作点 Jacobian (真值处, 58 解)...', flush=True)
J_op, _ = jac_q(q_true, y_meas)
_, S_op, Vt_op = np.linalg.svd(J_op)
s_sorted = np.sort(S_op)
print(f'工作点 最小6 sv: {s_sorted[:6]}', flush=True)
n_blind_op = int(np.sum(S_op < s_sorted[N_GAUGE] * 0.05))
print(f'工作点近零模式数: {n_blind_op}, '
      f'去规范 cond = {S_op[0]/s_sorted[N_GAUGE]:.1f}')
# 共模在弱空间的投影
def common_vec(block):
    e = np.zeros(3 * NU); e[block * NU:(block + 1) * NU] = 1
    return e / np.linalg.norm(e)
weakV = Vt_op[np.argsort(S_op)[:N_GAUGE]]
for name, b in [('共模u', 1), ('共模v', 2)]:
    fr = np.linalg.norm(weakV @ common_vec(b))
    print(f'  {name} 在3弱模式空间投影: {fr*100:.0f}%')

# ---- 完整 GN ----
q = np.zeros(3 * NU)
gauge0 = None
for it in range(6):
    J, y_cur = jac_q(q)
    Jp, S, Vt = trunc_pinv(J, N_GAUGE)
    if gauge0 is None:
        gauge0 = Vt[np.argsort(S)[:N_GAUGE]]
    r = (y_meas - y_cur) / sig
    dq = Jp @ r
    # 阻尼 + 步长限幅 (每全量程单位)
    step = np.clip(dq, -0.4, 0.4)
    q = q + 0.8 * step * RANGE_q
    print(f'GN it{it+1}: |r|={np.linalg.norm(r):.2f} |dq|={np.linalg.norm(dq):.3f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    if np.linalg.norm(dq) < 5e-3:
        break

# 评估 (可观测子空间: 真值投影掉收敛点的3个规范模式)
q_s = q / RANGE_q
qt_s = q_true / RANGE_q
qt_obs = (qt_s - gauge0.T @ (gauge0 @ qt_s)) * RANGE_q
err = (q_s * RANGE_q) - qt_obs
ew = np.abs(err[:NU]) * 1000
euv = np.abs(err[NU:]) * 1000
print(f'\n结果: w 场最大误差 {np.max(ew):.2f} um '
      f'(中心 {-q[CI]*1000:.1f} / 真值 {-w_true[CI]*1000:.1f} um)')
print(f'可观测面内最大误差 {np.max(euv):.2f} um; '
      f'共模剪切恢复 {np.mean(q[NU:2*NU])*1000:.1f} um (规范盲, 预期约0)')

# 噪声: 收敛点 Jacobian 线性传播
Jf, _ = jac_q(q)
Jpf, Sf, Vtf = trunc_pinv(Jf, N_GAUGE)
sq = np.sqrt(np.sum(Jpf ** 2, axis=1)) * RANGE_q      # 1 sigma (物理)
print(f'噪声 (0.5pH地板, 线性传播): w 1σ={np.mean(sq[:NU])*1000:.2f} um, '
      f'面内 1σ={np.mean(sq[NU:])*1000:.2f} um')

out = json.load(open(REP + 'honeycomb_series.json'))
out['e2e_v3'] = dict(
    method='full GN, fresh J each iter, forward-diff d=0.05',
    op_point_sv_min6=s_sorted[:6].tolist(),
    op_point_cond_gauge_fixed=float(S_op[0] / s_sorted[N_GAUGE]),
    w_max_err_um=float(np.max(ew)), uv_obs_max_err_um=float(np.max(euv)),
    depth_um=float(-q[CI] * 1000), depth_true_um=float(-w_true[CI] * 1000),
    w_true=w_true.tolist(), w_est=q[:NU].tolist(),
    noise_w_um=float(np.mean(sq[:NU]) * 1000),
    noise_uv_um=float(np.mean(sq[NU:]) * 1000),
    total_time_s=time.time() - t0)
with open(REP + 'honeycomb_series.json', 'w') as f:
    json.dump(out, f, indent=1)
print(f'saved e2e_v3 ({time.time()-t0:.0f}s)')
