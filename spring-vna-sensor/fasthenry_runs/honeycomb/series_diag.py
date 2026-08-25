#!/usr/bin/env python3
"""方案B 非线性隔离诊断: 哪个变形成分破坏线性?
(a) 纯 w 场小幅 -0.1  (b) 纯 w 场 -0.4  (c) 纯共模 u 0.15  (d) w-0.4 无倾斜
每个: |actual-linear|/|actual|, 最差观测分解
"""
import numpy as np
from series_scheme import (UNITS, NU, CI, T_phys, RANGE_q, y0, observe_B,
                           T_s, J_B, OBS_B, EDGES)

FLOOR = 0.5
sig = 1e-3 * np.maximum(np.abs(y0), FLOOR)
Jw = (J_B * y0[:, None]) / sig[:, None]
Jwc = Jw @ T_s

def check(name, q_phys, poses_override=None):
    q_s = q_phys / RANGE_q
    poses = (T_phys @ q_phys).reshape(NU, 5) if poses_override is None \
        else poses_override
    assert np.all(np.abs(poses[:, :2]) < 1.0) and \
           np.all(np.abs(poses[:, 2]) < 1.2) and \
           np.all(np.abs(poses[:, 3:]) < 0.35), 'pose 超界'
    y = observe_B(poses)
    r_act = (y - y_base) / sig
    if poses_override is None:
        r_lin = Jwc @ q_s
    else:
        p_s = (poses.flatten()) / np.array([0.5, 0.5, 0.75,
                                            np.deg2rad(10), np.deg2rad(10)] * NU)
        r_lin = Jw @ p_s
    dev = np.linalg.norm(r_act - r_lin) / max(np.linalg.norm(r_act), 1e-9)
    k = int(np.argmax(np.abs(r_act - r_lin)))
    print(f'{name:28s} |act|={np.linalg.norm(r_act):8.1f} 偏差={dev*100:6.1f}% '
          f'最差obs={OBS_B[k]} (act={r_act[k]:.1f} lin={r_lin[k]:.1f})')
    return y

y_base = observe_B(np.zeros((NU, 5)))
r2 = np.array([np.hypot(*UNITS[k][:2]) for k in range(NU)]) ** 2
sw = 5.2 * 1.2

for amp in [0.1, 0.4]:
    w = -amp * np.exp(-r2 / (2 * sw ** 2))
    q = np.concatenate([w, np.zeros(NU), np.zeros(NU)])
    check(f'纯w场 幅度-{amp} (含梯度倾斜)', q)

# 共模 u
q = np.concatenate([np.zeros(NU), np.full(NU, 0.15), np.zeros(NU)])
check('纯共模u 0.15', q)

# w 场但强制零倾斜 (隔离倾斜成分)
w = -0.4 * np.exp(-r2 / (2 * sw ** 2))
poses = np.zeros((NU, 5)); poses[:, 2] = w
check('w-0.4 强制零倾斜', np.concatenate([w, np.zeros(2 * NU)]), poses)

# 纯倾斜 (真值梯度, 零位移)
q_w = np.concatenate([w, np.zeros(2 * NU)])
poses_t = (T_phys @ q_w).reshape(NU, 5)
poses_only_tilt = np.zeros((NU, 5)); poses_only_tilt[:, 3:] = poses_t[:, 3:]
p_s = poses_only_tilt.flatten() / np.array([0.5, 0.5, 0.75,
                                            np.deg2rad(10), np.deg2rad(10)] * NU)
y = observe_B(poses_only_tilt)
r_act = (y - y_base) / sig
r_lin = Jw @ p_s
dev = np.linalg.norm(r_act - r_lin) / np.linalg.norm(r_act)
k = int(np.argmax(np.abs(r_act - r_lin)))
print(f'{"纯倾斜(真值梯度场)":28s} |act|={np.linalg.norm(r_act):8.1f} '
      f'偏差={dev*100:6.1f}% 最差obs={OBS_B[k]} (act={r_act[k]:.1f} lin={r_lin[k]:.1f})')
