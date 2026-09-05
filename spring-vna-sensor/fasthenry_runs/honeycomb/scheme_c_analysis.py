#!/usr/bin/env python3
"""方案C 完整 FEM 判决: 无源单匝短路铜环顶层 + 有源交错锚点底层(15股捆束)
物理: 短路环感应电流反射 -> 底层端口有效阻抗
     Z_eff = jwL_bb + w^2 L_bt (R_t + jw L_tt)^-1 L_tb   (含环-环杂化, 精确)
观测: Im(Z_eff)/w = 有效电感, 19 自 + 42 邻边 = 61 个 (温漂免疫分量)
分析: 信号预算 / 61x95 无先验 / 61x57 柔性先验 / 线性度 / 端到端 / 温漂 / 杂化
"""
import numpy as np
import json, time, os
import gen_ring_array as G
from series_scheme import (UNITS, NU, ADJ, EDGES, CI, BOT_Z, DOFS, RANGE,
                           T_phys, RANGE_q, T_s)

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
F0 = 2.5e6
W = 2 * np.pi * F0
GAP_NOM = 1.75
TOP_W = 0.44          # Ø0.5mm 圆线等效方截面 mm
BOT_W = 0.2           # 15股捆束等效
R_TOP = 4.1e-3        # 0.5mm线 2.5MHz 趋肤 AC 电阻 (欧)
NB = 15               # 底层匝数
DELTA = dict(x=0.05, y=0.05, z=0.05, tax=np.deg2rad(1), tay=np.deg2rad(1))

# ---------------- FastHenry: 每环独立截面 ----------------
def write_inp_c(path, rings):
    """rings: list of (name, pose, width)"""
    L = ['* scheme C', '.units mm', '.default sigma=5.8e7 nhinc=1 nwinc=1', '']
    nseg = G.NSEG
    for name, pose, w in rings:
        p = G.ring_nodes(pose)
        for i, (x, y, z) in enumerate(p):
            L.append(f'N{name}_{i} x={x:.6f} y={y:.6f} z={z:.6f}')
        for i in range(len(p) - 1):
            L.append(f'E{name}_{i} N{name}_{i} N{name}_{i+1} w={w} h={w}')
        L.append('')
    for name, _, _ in rings:
        L.append(f'.external N{name}_0 N{name}_{nseg}')
    L.append(f'.freq fmin={G.FREQ:.4g} fmax={G.FREQ:.4g} ndec=1')
    L.append('.end')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')

def solve_c(top_poses, tag='schc_tmp'):
    assert np.all(np.abs(top_poses[:, :2]) < 1.0) and \
           np.all(np.abs(top_poses[:, 2]) < 1.2) and \
           np.all(np.abs(top_poses[:, 3:]) < 0.35), 'pose 超界'
    rings = []
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = top_poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, GAP_NOM + dz, tax, tay), TOP_W))
    for u, (cx, cy, c) in enumerate(UNITS):
        rings.append((f'B{u}', (cx, cy, BOT_Z[u], 0.0, 0.0), BOT_W))
    inp = os.path.join(G.WORKDIR, f'{tag}.inp')
    write_inp_c(inp, rings)
    Z, _ = G.run_fasthenry(inp)
    return G.z_to_L(Z) * 1e9        # nH 单匝, 顺序 T0..18, B0..18

# ---------------- 电路折叠 ----------------
def fold(Lm_nH, r_top=R_TOP, kill_hybrid=False):
    L = Lm_nH * 1e-9
    s = np.array([1.0] * NU + [float(NB)] * NU)
    Lp = L * np.outer(s, s)
    Ltt = Lp[:NU, :NU].copy()
    if kill_hybrid:
        Ltt = np.diag(np.diag(Ltt))
    Ltb = Lp[:NU, NU:]
    Lbb = Lp[NU:, NU:]
    Zt = r_top * np.eye(NU) + 1j * W * Ltt
    A = np.linalg.solve(Zt, Ltb)
    Zeff = 1j * W * Lbb + W * W * (Ltb.T @ A)
    return Zeff, Lbb

def obs(Zeff):
    Leff = np.imag(Zeff) / W * 1e9      # nH
    y = [Leff[i, i] for i in range(NU)] + [Leff[i, j] for i, j in EDGES]
    return np.array(y)

def observe(top_poses, tag='schc_tmp'):
    Zeff, _ = fold(solve_c(top_poses, tag))
    return obs(Zeff)

pose_of = lambda q: (T_phys @ q).reshape(NU, 5)

if __name__ == '__main__':
    t0 = time.time()
    out = {}
    # ---------- 基线 & 信号预算 ----------
    Lm0 = solve_c(np.zeros((NU, 5)), 'schc_base')
    Zeff0, Lbb0 = fold(Lm0)
    y0 = obs(Zeff0)
    y_direct = np.array([Lbb0[i, i] for i in range(NU)] +
                        [Lbb0[i, j] for i, j in EDGES]) * 1e9
    refl0 = y0 - y_direct
    L_top = Lm0[CI, CI]
    Q_top = W * L_top * 1e-9 / R_TOP
    print(f'顶环: L={L_top:.2f}nH (单匝Ø0.5mm), Q={Q_top:.1f} @2.5MHz')
    print(f'自观测: 背景 {y_direct[CI]:.1f}nH, 反射 {refl0[CI]:.2f}nH '
          f'({refl0[CI]/y_direct[CI]*100:.1f}%)')
    ei = NU  # 第一条边
    print(f'边观测: 背景 {y_direct[ei]:.2f}nH, 反射 {refl0[ei]:.3f}nH')
    print(f'反射幅度: 自 [{np.min(np.abs(refl0[:NU])):.1f}, '
          f'{np.max(np.abs(refl0[:NU])):.1f}]nH, '
          f'边 [{np.min(np.abs(refl0[NU:])):.3f}, '
          f'{np.max(np.abs(refl0[NU:])):.3f}]nH', flush=True)
    out['top_L_nH'] = float(L_top); out['top_Q'] = float(Q_top)
    out['refl_self_nH'] = refl0[:NU].tolist()
    out['refl_edge_range_nH'] = [float(np.min(np.abs(refl0[NU:]))),
                                 float(np.max(np.abs(refl0[NU:])))]

    # ---------- 温漂 & 杂化 (折叠层, 零额外求解) ----------
    y_hot = obs(fold(Lm0, r_top=R_TOP * 1.004)[0])       # +1K
    tempco = np.max(np.abs(y_hot - y0) / np.maximum(np.abs(refl0), 1e-3))
    print(f'温漂: +1K 对反射分量的最大相对影响 {tempco*100:.4f}% (理论~1/Q^2)')
    y_nohyb = obs(fold(Lm0, kill_hybrid=True)[0])
    hyb = np.max(np.abs(y_nohyb - y0) / np.maximum(np.abs(refl0), 1e-3))
    print(f'环-环杂化: 对反射分量的最大相对贡献 {hyb*100:.2f}%', flush=True)
    out['tempco_per_K'] = float(tempco)
    out['hybridization_frac'] = float(hyb)

    # ---------- Jacobian (190 求解) ----------
    print('Jacobian (95 DOF 中心差分, ~190 解)...', flush=True)
    dY = np.zeros((len(y0), NU * 5))   # nH / 全量程
    col = 0
    for u in range(NU):
        for d, dof in enumerate(DOFS):
            dd = DELTA[dof]
            pp = np.zeros((NU, 5)); pp[u, d] = +dd
            pm = np.zeros((NU, 5)); pm[u, d] = -dd
            dY[:, col] = (observe(pp) - observe(pm)) / (2 * dd) * RANGE[dof]
            col += 1
        print(f'  unit {u+1}/{NU} ({time.time()-t0:.0f}s)', flush=True)

    # ---------- 两种噪声场景 ----------
    sig1 = 1e-3 * np.maximum(np.abs(y0), 0.5)            # S1: 直接测总量
    sig2 = np.maximum(1e-3 * np.abs(refl0), 0.005)       # S2: 调零后测反射
    for name, sig in [('S1直接', sig1), ('S2调零', sig2)]:
        Jw = dY / sig[:, None]
        sv = np.linalg.svd(Jw, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-9))
        print(f'\n[{name}] 无先验 61x95: rank {rank}/95, 盲维 {NU*5-rank}')
        Jc = Jw @ T_s
        U_, S_, Vt_ = np.linalg.svd(Jc)
        ss = np.sort(S_)
        n_gauge = int(np.sum(S_ < S_[0] * 1e-4))
        print(f'[{name}] 先验 61x57: 最小5 sv {ss[:5]}')
        print(f'[{name}] 规范模式数: {n_gauge}, '
              f'去规范 cond = {S_[0]/ss[n_gauge]:.1f}')
        # 共模投影
        for cname, blk in [('共模u', 1), ('共模v', 2), ('共模w', 0)]:
            e = np.zeros(3 * NU); e[blk * NU:(blk + 1) * NU] = 1
            e /= np.linalg.norm(e)
            weakV = Vt_[np.argsort(S_)[:max(n_gauge, 3)]]
            print(f'  {cname} 弱空间投影: {np.linalg.norm(weakV @ e)*100:.0f}%')
        # 分辨率
        Jp = np.linalg.pinv(Jc, rcond=ss[n_gauge] * 0.5 / S_[0] if n_gauge else 1e-8)
        sq = np.sqrt(np.sum(Jp ** 2, axis=1)) * RANGE_q
        print(f'[{name}] 噪声分辨率: w 1σ={np.mean(sq[:NU])*1000:.2f}um, '
              f'面内 1σ={np.mean(sq[NU:])*1000:.2f}um', flush=True)
        out[f'{name}'] = dict(rank_noprior=rank,
                              sv_prior_min5=ss[:5].tolist(),
                              sv_prior=np.sort(S_).tolist(),
                              n_gauge=n_gauge,
                              cond_gauge_fixed=float(S_[0] / ss[n_gauge]),
                              res_w_um=float(np.mean(sq[:NU]) * 1000),
                              res_uv_um=float(np.mean(sq[NU:]) * 1000))

    # ---------- 线性度 (联动+共模真值) ----------
    r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
    w_true = -0.4 * np.exp(-r2 / (2 * (5.2 * 1.2) ** 2))
    q_true = np.concatenate([w_true, np.full(NU, 0.15), np.zeros(NU)])
    y_meas = observe(pose_of(q_true), 'schc_e2e')
    p_s = pose_of(q_true).flatten() / np.array(
        [RANGE[d] for d in DOFS] * NU)
    dev = np.linalg.norm((y_meas - y0) - dY @ p_s) / np.linalg.norm(y_meas - y0)
    print(f'\n线性度 (全幅联动+共模真值): 偏差 {dev*100:.1f}%', flush=True)
    out['linearity_dev'] = float(dev)

    # ---------- 存档 (端到端见 scheme_c_e2e.py) ----------
    np.savez(REP + 'scheme_c_jacobian.npz', dY=dY, y0=y0, refl0=refl0,
             sig1=sig1, sig2=sig2, y_meas=y_meas, q_true=q_true)
    out['total_time_s'] = time.time() - t0
    with open(REP + 'honeycomb_scheme_c.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'saved honeycomb_scheme_c.json + jacobian npz '
          f'({time.time()-t0:.0f}s)')
    raise SystemExit(0)

    # ---------- (已废弃, 保留参考) 端到端 (S2 加权, 标称J 阻尼拟牛顿) ----------
    Jw2 = dY / sig2[:, None]
    Jc2 = Jw2 @ T_s
    U_, S_, Vt_ = np.linalg.svd(Jc2)
    ss = np.sort(S_)
    n_g = int(np.sum(S_ < S_[0] * 1e-4))
    cut = ss[n_g] * 0.5 if n_g else ss[0] * 0.5
    Sinv = np.where(S_ > cut, 1 / np.maximum(S_, 1e-30), 0.0)
    Jp = (Vt_.T * Sinv) @ U_.T[:len(S_)]
    gauge = Vt_[np.argsort(S_)[:n_g]] if n_g else np.zeros((0, 3 * NU))
    q = np.zeros(3 * NU)
    for it in range(10):
        r = (y_meas - observe(pose_of(q))) / sig2     # q 为物理量 (mm/rad)
        dq_s = Jp @ r
        q = q + 0.7 * dq_s * RANGE_q
        if np.linalg.norm(dq_s) < 1e-4:
            break
    qs = q / RANGE_q
    qt_s = q_true / RANGE_q
    if n_g:
        qt_s = qt_s - gauge.T @ (gauge @ qt_s)
    err = (qs - qt_s) * RANGE_q
    print(f'端到端 ({it+1} 迭代): w 最大误差 {np.max(np.abs(err[:NU]))*1000:.2f}um, '
          f'面内最大 {np.max(np.abs(err[NU:]))*1000:.2f}um')
    print(f'  中心压深 {-q[CI]*1000:.1f}/400.0 um, '
          f'共模剪切恢复 {np.mean(q[NU:2*NU])*1000:.1f}/150.0 um', flush=True)
    out['e2e'] = dict(iters=it + 1,
                      w_max_err_um=float(np.max(np.abs(err[:NU])) * 1000),
                      uv_max_err_um=float(np.max(np.abs(err[NU:])) * 1000),
                      depth_um=float(-q[CI] * 1000),
                      cm_u_um=float(np.mean(q[NU:2 * NU]) * 1000))

    # ---------- 噪声蒙特卡洛 (S2) ----------
    rng = np.random.default_rng(51)
    errs = []
    for trial in range(8):
        yn = y_meas + sig2 * rng.standard_normal(len(y0))
        qn = q.copy()
        for it2 in range(4):
            r = (yn - observe(pose_of(qn))) / sig2
            qn = qn + 0.7 * (Jp @ r) * RANGE_q
        errs.append(qn - q)
    errs = np.array(errs)
    print(f'噪声MC(S2): w 1σ={np.std(errs[:,:NU])*1000:.2f}um, '
          f'面内 1σ={np.std(errs[:,NU:])*1000:.2f}um', flush=True)
    out['mc_S2'] = dict(w_um=float(np.std(errs[:, :NU]) * 1000),
                        uv_um=float(np.std(errs[:, NU:]) * 1000))

    np.savez(REP + 'scheme_c_jacobian.npz', dY=dY, y0=y0, refl0=refl0,
             sig1=sig1, sig2=sig2)
    out['total_time_s'] = time.time() - t0
    with open(REP + 'honeycomb_scheme_c.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'saved honeycomb_scheme_c.json ({time.time()-t0:.0f}s)')
