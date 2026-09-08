#!/usr/bin/env python3
"""方案C 尺寸缩放: 5mm -> 1mm (全部几何 x1/5), 验证 Q/信号/趋肤/温漂"""
import numpy as np
import subprocess, re, os, time
import gen_ring_array as G

WD = G.WORKDIR
MU0 = 4e-7 * np.pi
SIGMA_CU = 5.8e7

def skin_R_ring(D_mm, wire_mm, f, N=1):
    """圆环 AC 电阻 (含趋肤), 方截面 wire x wire 近似圆"""
    d = np.sqrt(4 / np.pi) * wire_mm * 1e-3        # 等效圆直径
    a = d / 2
    delta = 1 / np.sqrt(np.pi * f * MU0 * SIGMA_CU)
    # AC 有效截面 (趋肤环)
    if delta >= a:
        Aeff = np.pi * a * a                        # DC 极限
    else:
        Aeff = np.pi * (a * a - (a - delta) ** 2)
    length = N * np.pi * D_mm * 1e-3
    return length / (SIGMA_CU * Aeff), delta * 1e3  # ohm, delta_mm

def build_pair_c(scale, f, tag):
    """底层驱动环 + 顶层短路环, 共轴, 算反射.
    返回 (L_bot单匝nH, L_top单匝nH, M nH, k)"""
    D = 5.0 * scale
    gap = 1.75 * scale
    wire_bot = 0.2 * scale
    wire_top = 0.44 * scale
    r = D / 2
    rings = [('B', (0, 0, 0, 0, 0), wire_bot),
             ('T', (0, 0, gap, 0, 0), wire_top)]
    L = ['* scale', '.units mm', f'.default sigma={SIGMA_CU} nhinc=1 nwinc=1', '']
    for name, pose, w in rings:
        p = G.ring_nodes(pose, r=r)
        for i, (x, y, z) in enumerate(p):
            L.append(f'N{name}_{i} x={x:.6f} y={y:.6f} z={z:.6f}')
        for i in range(len(p) - 1):
            L.append(f'E{name}_{i} N{name}_{i} N{name}_{i+1} w={w:.4f} h={w:.4f}')
        L.append('')
    for name, _, _ in rings:
        L.append(f'.external N{name}_0 N{name}_{G.NSEG}')
    L.append(f'.freq fmin={G.FREQ} fmax={G.FREQ} ndec=1')
    L.append('.end')
    inp = os.path.join(WD, f'{tag}.inp')
    open(inp, 'w').write('\n'.join(L) + '\n')
    Z, _ = G.run_fasthenry(inp)
    Lm = G.z_to_L(Z) * 1e9
    k = Lm[0, 1] / np.sqrt(Lm[0, 0] * Lm[1, 1])
    return Lm[0, 0], Lm[1, 1], Lm[0, 1], k

f = 2.5e6
w = 2 * np.pi * f
print(f'{"scale":>6} {"D(mm)":>6} {"Lbot(nH)":>9} {"Ltop(nH)":>9} {"M(nH)":>7} '
      f'{"k":>6} {"Rtop(mΩ)":>9} {"δ(mm)":>7} {"Q_top":>6} {"反射(nH)":>9}')
for scale in [1.0, 0.6, 0.4, 0.2]:
    D = 5.0 * scale
    Lb, Lt, M, k = build_pair_c(scale, f, f'scl_{int(scale*100)}')
    Rtop, delta = skin_R_ring(D, 0.44 * scale, f)
    Q = w * Lt * 1e-9 / Rtop
    # 反射到底层自感 (单顶环, 束绕N=15 底层): 反射等效 = w M^2/(R+jwL) 的 Im/w
    # 单匝 M, 底层 15 匝 -> M_eff x15; 顶环单匝
    Mb = M * 15
    refl = -(w * w * Mb * 1e-9 * (Lt * 1e-9)) / ((Rtop) ** 2 + (w * Lt * 1e-9) ** 2) / w * 1e9
    print(f'{scale:6.1f} {D:6.2f} {Lb:9.2f} {Lt:9.3f} {M:7.3f} {k:6.3f} '
          f'{Rtop*1e3:9.2f} {delta:7.3f} {Q:6.2f} {refl:9.3f}')

print('\n温漂 (1/Q^2 免疫): scale=1.0 -> 1/Q^2 =', 1 / 28.4 ** 2,
      ', scale=0.2 -> 1/Q^2 =', '待上表 Q 值代入')
