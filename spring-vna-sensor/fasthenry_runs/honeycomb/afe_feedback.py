#!/usr/bin/env python3
"""回应 AFE 反馈: 8MHz + PCB螺旋 + 冲压铜环 下重算关键规格
1. 线圈/环 真实 R_ac, Q, DC电阻 (解析趋肤, 不用FastHenry的R)
2. 自/边观测载波与反射范围
3. 第三线圈干扰 vs C_off 扫描 -> 最大允许断态电容
4. 驱动电压/环电流/发热
5. 4MHz vs 8MHz 分辨率
"""
import numpy as np
import json
from pcb_verdict import (solve, fold, obs, UNITS, NU, EDGES, CI, COILZ, NB,
                         W as W8, F as F8, R_RING)
import gen_ring_array as G

MU0 = 4e-7 * np.pi
SIG = 5.8e7

def skin_R(length_mm, w_mm, t_mm, f):
    d = 1 / np.sqrt(np.pi * f * MU0 * SIG)      # 趋肤深度 m
    w, t, L = w_mm * 1e-3, t_mm * 1e-3, length_mm * 1e-3
    # 扁导体 AC 有效周长厚度 (双面趋肤)
    if d >= t / 2:
        Aeff = w * t
    else:
        Aeff = 2 * d * (w + t) - 4 * d * d      # 周边趋肤环
    Rdc = L / (SIG * w * t)
    Rac = L / (SIG * Aeff)
    return Rdc, Rac, d * 1e6

print('=== 1. 线圈/环 R_ac/Q (解析趋肤) ===')
# 螺旋: 24匝(12x2层), 平均直径2.85mm, 线宽0.089, 1oz=35um
avg_d = (5.0 + 0.7) / 2
Lspiral_len = 24 * np.pi * avg_d
for f in [8e6, 10e6, 4e6]:
    Rdc, Rac, dsk = skin_R(Lspiral_len, 0.089, 0.035, f)
    L = 1.5e-6
    print(f'  螺旋@{f/1e6:.0f}MHz: 线长{Lspiral_len:.0f}mm, δ={dsk:.0f}um, '
          f'Rdc={Rdc:.2f}Ω Rac={Rac:.2f}Ω Q={2*np.pi*f*L/Rac:.0f}')
# 铜环 Ø5/Ø3/0.2 紫铜: 周长 π*4mm(中径), 宽1mm 厚0.2
ring_len = np.pi * 4.0
for f in [8e6, 4e6]:
    Rdc, Rac, dsk = skin_R(ring_len, 1.0, 0.2, f)
    print(f'  铜环@{f/1e6:.0f}MHz: δ={dsk:.0f}um, Rdc={Rdc*1e3:.2f}mΩ '
          f'Rac={Rac*1e3:.2f}mΩ (代入判决用 R_RING)')

print('\n=== 2. 载波与反射 (8MHz, 冲压环 Ø5/Ø3/0.2) ===')
Lm0 = solve(np.zeros((NU, 5)), 'afe_b')
# 有效电感矩阵 (含反射)
Leff = fold(Lm0)
# 纯载波 (无反射) = L_cc 对角/邻边
s = np.array([float(NB)] * NU + [1.0] * NU)
Lcc = (Lm0 * 1e-9 * np.outer(s, s))[:NU, :NU] * 1e9   # nH
y0 = obs(Leff)
carrier = np.array([Lcc[i, i] for i in range(NU)] + [Lcc[i, j] for i, j in EDGES])
refl = y0 - carrier
print(f'  自观测: 载波 {np.median([carrier[i] for i in range(NU)]):.0f}nH, '
      f'反射 [{np.min(np.abs(refl[:NU])):.0f},{np.max(np.abs(refl[:NU])):.0f}]nH '
      f'({np.max(np.abs(refl[:NU]))/np.median(carrier[:NU])*100:.0f}%)')
print(f'  边观测: 载波 |{np.min(np.abs(carrier[NU:])):.2f}~{np.max(np.abs(carrier[NU:])):.2f}|nH, '
      f'反射 [{np.min(np.abs(refl[NU:])):.2f},{np.max(np.abs(refl[NU:])):.2f}]nH')
# 电压: 8MHz, 10mA
print(f'  边观测反射电压@10mA: [{W8*np.min(np.abs(refl[NU:]))*1e-9*10e-3*1e3:.2f},'
      f'{W8*np.max(np.abs(refl[NU:]))*1e-9*10e-3*1e3:.2f}]mV')

print('\n=== 3. 第三线圈干扰 vs C_off (8MHz) ===')
# 驱动i, 测边j, 第三有源线圈k断态C_off. 直接线圈-线圈路径
# 用最坏三元组: 相邻三线圈. M值取 L_cc 邻边(束绕NB^2已含)
Ledge = np.abs([Lcc[i, j] for i, j in EDGES])
M_ik = np.max(Ledge) * 1e-9        # 最强邻耦 (驱动->第三)
M_kj = np.max(Ledge) * 1e-9
M_ij_sig = np.min(np.abs(refl[NU:])) * 1e-9   # 最弱边观测信号
Lk = 1.5e-6
print(f'  最坏三元组: M_ik=M_kj={M_ik*1e9:.1f}nH, 信号 M_ij(反射)={M_ij_sig*1e9:.2f}nH')
print(f'  {"C_off":>8} {"|Z_off|@8M":>11} {"干扰/信号":>9}')
for Coff_fF in [50, 86, 150, 250, 500, 1000]:
    Coff = Coff_fF * 1e-15
    Zoff = abs(1 / (1j * W8 * Coff) + 1j * W8 * Lk)   # C_off串联线圈L
    ratio = W8 * M_ik * M_kj / (M_ij_sig * Zoff) * 100
    print(f'  {Coff_fF:>6}fF {Zoff/1e3:>9.0f}kΩ {ratio:>8.1f}%')
# 求解 0.1% 和 1% 对应的最大 C_off
for tgt in [0.1, 1.0]:
    Coff_max = tgt / 100 * M_ij_sig / (W8 * M_ik * M_kj) * (W8)  # 近似 1/(wCoff)主导
    Coff_max = tgt / 100 * M_ij_sig / (W8 * M_ik * M_kj) / (1/W8)
    # |Z|~1/(wC): ratio= w M_ik M_kj/(M_ij /(wC))... 解C
    Cmax = tgt / 100 * M_ij_sig / (W8 * M_ik * M_kj) * W8   # = tgt*M_ij*w/(w^2 M M)
    Cmax = (tgt/100) * M_ij_sig / (W8**2 * M_ik * M_kj / W8)  # redo below numerically
    print(f'  -> 干扰<{tgt}%: 需 |Z_off| >= '
          f'{W8*M_ik*M_kj/(M_ij_sig*tgt/100)/1e3:.0f}kΩ = C_off <= '
          f'{1/(W8*W8*M_ik*M_kj/(M_ij_sig*tgt/100))*1e15:.0f}fF (单只)')

print('\n=== 4. 驱动/发热 (8MHz, 10mA) ===')
Vdrive = W8 * 1.5e-6 * 10e-3
print(f'  端口电压 ~{Vdrive*1e3:.0f}mVp (主要是感抗), 线圈损耗 I²R={0.01**2*2:.4f}W=0.2mW')
# 环电流: I_ring = wM_cr I / |Zring|, M_cr自耦
Lcr0 = (Lm0 * 1e-9 * np.outer(s, s))[CI, NU + CI]
Iring = W8 * abs(Lcr0) * 10e-3 / abs(R_RING + 1j * W8 * 6.9e-9)
print(f'  铜环感应电流 ~{Iring*1e3:.1f}mA, 环损耗 I²R={Iring**2*R_RING*1e6:.2f}µW (可忽略)')

json.dump(dict(
    spiral_Rdc=float(skin_R(Lspiral_len,0.089,0.035,8e6)[0]),
    spiral_Rac_8M=float(skin_R(Lspiral_len,0.089,0.035,8e6)[1]),
    carrier_self_nH=float(np.median(carrier[:NU])),
    refl_self_nH=[float(np.min(np.abs(refl[:NU]))),float(np.max(np.abs(refl[:NU])))],
    carrier_edge_nH=[float(np.min(np.abs(carrier[NU:]))),float(np.max(np.abs(carrier[NU:])))],
    refl_edge_nH=[float(np.min(np.abs(refl[NU:]))),float(np.max(np.abs(refl[NU:])))],
    Coff_max_0p1pct_fF=float(1/(W8*W8*M_ik*M_kj/(M_ij_sig*0.1/100))*1e15),
    Coff_max_1pct_fF=float(1/(W8*W8*M_ik*M_kj/(M_ij_sig*1.0/100))*1e15),
    Vdrive_mVp=float(Vdrive*1e3), Iring_mA=float(Iring*1e3)),
    open('/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/afe_feedback.json','w'),indent=1)
print('\nsaved afe_feedback.json')
