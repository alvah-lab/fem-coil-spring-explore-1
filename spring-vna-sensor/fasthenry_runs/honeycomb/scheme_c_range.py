#!/usr/bin/env python3
"""方案C: 顶环反射贡献的空间衰减 —— 截断半径能取多小?"""
import numpy as np
from scheme_c_analysis import solve_c, UNITS, NU, EDGES, CI, W, R_TOP

Lm0 = solve_c(np.zeros((NU, 5)), 'dc_base')
d = np.array([np.hypot(UNITS[k][0] - UNITS[CI][0],
                       UNITS[k][1] - UNITS[CI][1]) for k in range(NU)])
print('单元到中心距离分层:', sorted(set(np.round(d, 1))), 'mm')

def obs_sub(Lm, keep):
    L = Lm * 1e-9
    s = np.array([1.0] * NU + [15.0] * NU)
    Lp = L * np.outer(s, s)
    idx = list(keep)
    if not idx:
        Lbb = Lp[NU:, NU:]
        Leff = np.imag(1j * W * Lbb) / W * 1e9
    else:
        Ltt = Lp[:NU, :NU][np.ix_(idx, idx)]
        Ltb = Lp[:NU, NU:][idx, :]
        Lbb = Lp[NU:, NU:]
        Zt = R_TOP * np.eye(len(idx)) + 1j * W * Ltt
        Zeff = 1j * W * Lbb + W * W * (Ltb.T @ np.linalg.solve(Zt, Ltb))
        Leff = np.imag(Zeff) / W * 1e9
    return np.array([Leff[i, i] for i in range(NU)] +
                    [Leff[i, j] for i, j in EDGES])

y_none = obs_sub(Lm0, [])
refl_full = obs_sub(Lm0, range(NU)) - y_none
ci_edges = [e for e, (i, j) in enumerate(EDGES) if i == CI or j == CI]

print('\n=== 中心观测反射: 按纳入顶环的最大半径收敛 ===')
print(f'  全反射基线: 中心自观测 {refl_full[CI]:.3f}nH')
for Rmax in [0.1, 5.3, 6.1, 9.1, 10.5, 99]:
    keep = [k for k in range(NU) if d[k] <= Rmax]
    y = obs_sub(Lm0, keep) - y_none
    e_self = abs(y[CI] - refl_full[CI]) / abs(refl_full[CI]) * 100
    e_edge = max(abs(y[NU + e] - refl_full[NU + e]) /
                 abs(refl_full[NU + e]) for e in ci_edges) * 100
    print(f'  半径<={Rmax:5}mm ({len(keep):2}环): '
          f'自观测误差 {e_self:6.2f}%, 中心边最大误差 {e_edge:6.2f}%')

# 单个顶环对中心自观测的贡献 vs 距离 (拆解)
print('\n=== 各顶环对中心自观测反射的单独贡献 (增量法) ===')
base = obs_sub(Lm0, [CI])[CI] - y_none[CI]      # 只中心顶环
print(f'  仅中心顶环: {base:.3f}nH ({base/refl_full[CI]*100:.1f}% of full)')
for ring_r in sorted(set(np.round(d, 1)))[1:]:
    ks = [k for k in range(NU) if abs(d[k] - ring_r) < 0.1]
    keep = [CI] + ks
    val = obs_sub(Lm0, keep)[CI] - y_none[CI]
    print(f'  +{ring_r:.1f}mm 环 ({len(ks)}个): 累计 {val:.3f}nH '
          f'({val/refl_full[CI]*100:.1f}%), 增量贡献 '
          f'{(val-base)/refl_full[CI]*100:+.2f}%')
