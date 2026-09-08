#!/usr/bin/env python3
"""图表36: ASIC 单通道详细原理图 (电流镜副本, 5-bit 电流舵调零 DAC, 底板四相采样, 电荷累加, 共享 7-bit SAR, 时序)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(24, 14.3), dpi=100)
ax.set_xlim(0, 24); ax.set_ylim(0, 14.3); ax.axis('off'); fig.patch.set_facecolor('white')
LW = 1.5
C_SIG = '#1f4e79'; C_CLK = '#c55a11'; C_DIG = '#375623'; C_DAC = '#7030a0'; C_TX = '#9c2b2b'; C_PWR = '#7f7f7f'

def wire(pts, color='k', lw=LW, ls='-'):
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, solid_capstyle='round')
def dot(x, y, r=0.045, color='k'): ax.add_patch(Circle((x, y), r, fc=color, ec=color))
def box(x, y, w, h, text, color='k', fc='white', fs=9.5, bold=False, ls='-'):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=color, lw=1.5, ls=ls))
    if text: ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, fontweight='bold' if bold else 'normal')
def cap(x, y, horiz=False, d=0.08):
    if horiz: wire([(x - d, y - 0.2), (x - d, y + 0.2)]); wire([(x + d, y - 0.2), (x + d, y + 0.2)])
    else: wire([(x - 0.2, y - d), (x + 0.2, y - d)]); wire([(x - 0.2, y + d), (x + 0.2, y + d)])
def sw(x, y, label, color=C_CLK, length=0.45, fs=8.5, above=True):
    ax.plot([x, x + length * 0.75], [y, y + 0.2], color='k', lw=LW); dot(x, y, 0.035); dot(x + length, y, 0.035)
    ax.text(x + length / 2, y + (0.33 if above else -0.33), label, ha='center', va='center', fontsize=fs, color=color, fontweight='bold')
def swv(x, y, label, color=C_CLK, length=0.45, fs=8.5):
    ax.plot([x, x - 0.2], [y, y + length * 0.75], color='k', lw=LW); dot(x, y, 0.035); dot(x, y + length, 0.035)
    ax.text(x + 0.1, y + length / 2, label, ha='left', va='center', fontsize=fs, color=color, fontweight='bold')
def gnd(x, y, label=None):
    wire([(x, y), (x, y - 0.12)]); wire([(x - 0.16, y - 0.12), (x + 0.16, y - 0.12)]); wire([(x - 0.1, y - 0.19), (x + 0.1, y - 0.19)]); wire([(x - 0.04, y - 0.26), (x + 0.04, y - 0.26)])
    if label: ax.text(x + 0.2, y - 0.18, label, fontsize=7.5, color=C_PWR)
def opamp(x, y, size=0.45, label=''):
    ax.add_patch(Polygon([(x, y - size), (x, y + size), (x + 1.3 * size, y)], closed=True, fc='white', ec='k', lw=LW))
    ax.text(x + 0.13, y + 0.2, '−', fontsize=10, va='center'); ax.text(x + 0.13, y - 0.2, '+', fontsize=10, va='center')
    if label: ax.text(x + 0.42, y, label, fontsize=8, ha='center', va='center')
def coil(x, y, n=4, r=0.15):
    for i in range(n): ax.add_patch(Arc((x, y + (2 * i + 1) * r), 2 * r, 2 * r, theta1=-90, theta2=90, lw=LW, color='k'))
def nmos(cx, cy, label='', color='k'):
    wire([(cx - 0.45, cy), (cx - 0.2, cy)], color=color); wire([(cx - 0.2, cy - 0.25), (cx - 0.2, cy + 0.25)], color=color)
    wire([(cx - 0.12, cy - 0.3), (cx - 0.12, cy + 0.3)], color=color, lw=2.4)
    wire([(cx - 0.12, cy + 0.22), (cx, cy + 0.22), (cx, cy + 0.45)], color=color); wire([(cx - 0.12, cy - 0.22), (cx, cy - 0.22), (cx, cy - 0.45)], color=color)
    if label: ax.text(cx + 0.1, cy, label, fontsize=8, va='center', color=color)
def arrow(p0, p1, color='k', lw=1.3, ls='-', ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, linestyle=ls))

# ===================== 区域框 =====================
box(0.3, 8.5, 7.2, 5.5, '', color=C_TX, ls='--'); ax.text(0.45, 13.8, '发射域 AVDD_TX', fontsize=10.5, color=C_TX, fontweight='bold')
box(7.8, 8.5, 9.7, 5.5, '', color=C_SIG, ls='--'); ax.text(7.95, 13.8, '接收域 AVDD_RX (每通道)', fontsize=10.5, color=C_SIG, fontweight='bold')
box(17.8, 8.5, 5.9, 5.5, '', color=C_DIG, ls='--'); ax.text(17.95, 13.8, '共享: MUX + 读出 ADC', fontsize=10.5, color=C_DIG, fontweight='bold')
box(0.3, 0.3, 7.2, 7.7, '', color=C_DAC, ls='--'); ax.text(0.45, 7.75, '调零 DAC (接收域, 每通道)', fontsize=10.5, color=C_DAC, fontweight='bold')
box(7.8, 0.3, 9.7, 7.7, '', color=C_CLK, ls='--'); ax.text(7.95, 7.75, '时钟 (共享) 与 时序', fontsize=10.5, color=C_CLK, fontweight='bold')
box(17.8, 0.3, 5.9, 7.7, '', color=C_DIG, ls='--', fc='#f7fbf5'); ax.text(17.95, 7.75, '数字 (DVDD): 状态机 + 小 RISC-V', fontsize=10.5, color=C_DIG, fontweight='bold')

# ===================== 1. 发射 =====================
box(0.5, 12.3, 1.3, 0.7, 'Vref\n带隙', fs=9); box(0.5, 11.2, 1.3, 0.7, 'buf_TX', fs=9); arrow((1.15, 12.3), (1.15, 11.9))
box(2.2, 11.0, 2.0, 1.9, '正弦 DAC\n8-bit 电流舵\n125 MSps\n16 点/周期', color=C_TX, fs=9)
arrow((1.8, 11.55), (2.2, 11.55)); ax.text(2.0, 11.75, 'I_ref', fontsize=8, ha='center')
wire([(4.2, 12.0), (5.0, 12.0)], color=C_TX); dot(5.0, 12.0); ax.text(4.6, 12.2, 'I_dac(t)', fontsize=8.5, color=C_TX, ha='center')
ax.text(5.15, 12.5, 'I_dac = I_bias + I0·sin ωt', fontsize=8, color=C_TX)
nmos(5.0, 11.3, 'M0', color=C_TX); wire([(5.0, 11.75), (5.0, 12.0)], color=C_TX)
wire([(5.0, 12.0), (4.4, 12.0), (4.4, 11.3), (4.55, 11.3)], color=C_TX); dot(4.4, 12.0)
wire([(5.0, 10.85), (5.0, 10.6)], color=C_TX); gnd(5.0, 10.6, 'GND_TX')
wire([(4.4, 11.3), (4.4, 9.3), (6.0, 9.3)], color=C_TX); dot(4.4, 11.3); dot(4.4, 9.3)
nmos(6.45, 9.3, 'Mk ×k', color=C_TX); wire([(6.45, 8.85), (6.45, 8.7)], color=C_TX); gnd(6.45, 8.7)
nmos(6.45, 10.5, 'Mc 共栅', color=C_TX); wire([(6.45, 9.75), (6.45, 10.05)], color=C_TX)
wire([(6.0, 10.5), (5.75, 10.5)], color=C_TX); ax.text(5.85, 10.68, 'Vb', fontsize=8, color=C_TX, ha='center', va='bottom')
wire([(6.45, 10.95), (6.45, 12.0), (8.4, 12.0)], color=C_TX); ax.text(7.0, 12.2, 'I_drive = k·I_dac', fontsize=8.5, color=C_TX, ha='center')
wire([(2.6, 11.0), (2.6, 9.3), (2.45, 9.3)], color=C_TX); ax.text(2.75, 10.15, 'V_g(I_ref)\nDAC 内部基准镜像', fontsize=7.5, color=C_TX, va='center')
nmos(2.9, 9.3, '', color=C_TX); ax.text(3.05, 9.6, 'M1 (镜 I_ref)', fontsize=8, color=C_TX); wire([(2.9, 8.85), (2.9, 8.7)], color=C_TX); gnd(2.9, 8.7)
wire([(2.9, 9.75), (2.9, 10.3), (1.0, 10.3), (1.0, 8.5)], color=C_DAC); ax.text(1.4, 10.5, 'I_rep ∝ I_ref (直流副本,\n与 I0 同比, 不经线圈)', fontsize=8, color=C_DAC, ha='center')
ax.text(4.4, 8.58, '电流源输出阻抗高 (Mc 共栅)\n→ 线圈端只测电压', fontsize=8, color=C_TX, ha='center', va='bottom')

# ===================== 2. 线圈 + 环 + 高阻缓冲 =====================
dot(8.4, 12.0)
coil(8.4, 10.3); wire([(8.4, 11.5), (8.4, 12.0)]); wire([(8.4, 10.3), (8.4, 10.0)]); gnd(8.4, 10.0, 'GND_RX')
ax.text(8.65, 10.9, 'L_coil\n1.3 µH', fontsize=8.5, va='center')
ax.add_patch(Rectangle((7.8, 12.9), 1.2, 0.13, fc='#c0504d', ec='k')); ax.text(8.4, 13.2, '无源铜环 (动层)', ha='center', fontsize=9)
wire([(8.4, 12.9), (8.4, 12.0)], color='#c0504d', ls=':'); ax.text(8.55, 12.45, 'M(位姿)', fontsize=8, color='#c0504d')
wire([(8.4, 12.0), (9.3, 12.0)]); dot(9.3, 12.0); wire([(9.3, 12.0), (9.3, 11.58)]); cap(9.3, 11.5); wire([(9.3, 11.42), (9.3, 11.2)]); gnd(9.3, 11.2)
ax.text(9.55, 11.5, 'C_pad ~0.3p', fontsize=7.5, va='center')
wire([(9.3, 12.0), (9.9, 12.0), (9.9, 11.8)])
opamp(9.9, 12.0, size=0.45, label='×1')
wire([(10.49, 12.0), (10.6, 12.0), (10.6, 12.7), (9.75, 12.7), (9.75, 12.2), (9.9, 12.2)]); dot(10.6, 12.0)
ax.text(10.2, 13.05, '高阻缓冲 (源随/单位增益)', fontsize=8, ha='center')
wire([(10.6, 12.0), (11.0, 12.0)]); dot(11.0, 12.0); ax.text(11.0, 11.65, 'V_a', fontsize=9, color=C_SIG, ha='center')
ax.text(7.95, 13.45, 'V_coil = Z_eff·I_drive, 幅度 <0.5 V (如 5 mA)', fontsize=8, color=C_SIG)

# ===================== 3. 底板采样 + 累加器 =====================
def sampler(rows, yc, name, sub):
    ys = [r[0] for r in rows]
    wire([(11.0, min(ys + [12.0])), (11.0, max(ys + [12.0]))])
    for y, ph in rows:
        dot(11.0, y); wire([(11.0, y), (11.3, y)]); sw(11.3, y, ph + '_e', fs=7.5); wire([(11.75, y), (12.07, y)])
        cap(12.15, y, horiz=True); ax.text(12.15, y + 0.32, 'Cs 2p', fontsize=7.5, ha='center')
        wire([(12.23, y), (12.55, y)]); dot(12.55, y)
        swv(12.55, y - 0.5, ph, length=0.45, fs=7.5); wire([(12.55, y - 0.5), (12.55, y - 0.6)]); gnd(12.55, y - 0.6)
        wire([(12.55, y), (12.85, y)]); sw(12.85, y, ph + '_t', length=0.45, fs=7.5); wire([(13.3, y), (13.5, y)])
    wire([(13.5, ys[0]), (13.5, ys[1])]); dot(13.5, yc); wire([(13.5, yc), (13.8, yc)])
    opamp(13.8, yc, size=0.42)
    wire([(13.8, yc - 0.2), (13.65, yc - 0.2), (13.65, yc - 0.75)]); gnd(13.65, yc - 0.75)
    wire([(13.8, yc + 0.2), (13.8, yc + 0.6), (14.12, yc + 0.6)]); cap(14.2, yc + 0.6, horiz=True); wire([(14.28, yc + 0.6), (14.7, yc + 0.6), (14.7, yc)])
    ax.text(14.2, yc + 0.35, 'C_acc 20p', fontsize=7.5, ha='center')
    wire([(13.8, yc + 0.6), (13.8, yc + 0.95), (14.0, yc + 0.95)]); sw(14.0, yc + 0.95, 'rst', color='grey', length=0.4, fs=7.5); wire([(14.4, yc + 0.95), (14.7, yc + 0.95), (14.7, yc + 0.6)])
    wire([(14.35, yc), (15.4, yc)]); dot(14.7, yc); dot(15.4, yc)
    ax.text(15.05, yc + 0.28, name, fontsize=9, color=C_SIG, ha='center', fontweight='bold'); ax.text(15.05, yc - 0.28, sub, fontsize=7.5, color=C_SIG, ha='center')

sampler([(13.2, 'φ0'), (12.3, 'φ180')], 12.75, 'I 累加', '过零残差')
sampler([(11.1, 'φ90'), (10.2, 'φ270')], 10.65, 'Q 累加', '幅度残差')
ax.text(16.55, 13.5, '底板采样: φ_e 先断顶板, φ 再断底板\n→ 电荷注入/时钟馈通只留常数项;\nφ_t 把 Cs 电荷转进虚地\n(极性由相位对实现)', fontsize=7.5, color=C_CLK, ha='center', va='center')
wire([(13.5, 10.65), (13.5, 8.5)], color=C_DAC); arrow((13.5, 9.6), (13.5, 10.35), color=C_DAC)
ax.text(13.65, 9.4, 'I_null 注入 Q 虚地\n(φ90 窗 +, φ270 窗 −)', fontsize=8, color=C_DAC, va='center')

# ===================== 4. 调零 DAC =====================
wire([(1.0, 8.5), (1.0, 7.0)], color=C_DAC); dot(1.0, 7.0)
nmos(1.0, 6.5, 'Md0', color=C_DAC); wire([(1.0, 6.95), (1.0, 7.0)], color=C_DAC)
wire([(1.0, 7.0), (0.45, 7.0), (0.45, 6.5), (0.55, 6.5)], color=C_DAC)
wire([(1.0, 6.05), (1.0, 5.85)], color=C_DAC); gnd(1.0, 5.85)
wire([(0.45, 6.5), (0.45, 4.6), (6.4, 4.6)], color=C_DAC); dot(0.45, 6.5)
legs = [(2.0, 'b1', 'Md1'), (3.0, 'b2', 'Md2'), (4.0, 'b3', 'Md3'), (5.8, 'b31', 'Md31')]
for x, b, m in legs:
    nmos(x, 4.6, '', color=C_DAC); dot(x - 0.45, 4.6)
    wire([(x, 4.15), (x, 3.95)], color=C_DAC); gnd(x, 3.95); ax.text(x, 3.6, m, fontsize=7.5, color=C_DAC, ha='center', va='top')
    wire([(x, 5.05), (x, 5.15)], color=C_DAC); swv(x, 5.15, b, color=C_DAC, length=0.4, fs=7.5); wire([(x, 5.55), (x, 5.9)], color=C_DAC); dot(x, 5.9)
ax.text(4.9, 4.7, '···', fontsize=13, color=C_DAC, ha='center')
wire([(2.0, 5.9), (6.6, 5.9)], color=C_DAC); dot(6.6, 5.9)
ax.text(3.9, 6.15, 'ΣI_u = code/32 · I_rep', fontsize=8.5, color=C_DAC, ha='center')
ax.text(4.0, 6.75, '31 个单元支路 (温度计码 b1..b31, 各 I_u = I_rep/32, 直流)', fontsize=8.5, color=C_DAC, ha='center')
wire([(6.6, 5.9), (6.6, 3.5)], color=C_DAC)
box(5.85, 2.4, 1.5, 1.1, '极性/窗口\nφ90: +  φ270: −\n其余 → dump', color=C_DAC, fs=7.5)
wire([(7.35, 2.95), (7.65, 2.95), (7.65, 8.25), (13.5, 8.25), (13.5, 8.5)], color=C_DAC)
ax.text(11.0, 8.3, 'I_null (跨域走线, 屏蔽)', fontsize=7.5, color=C_DAC, ha='center', va='bottom')
ax.text(0.55, 3.3, '每周期扣除 Q_null = code/32 · I_rep · t_w  ∝ I0 (比率型)\n→ I_ref 漂移与载波同比抵消\nI_rep 是直流: 幅度由 code 匹配, 相位由 PI 锁定的 φ90/φ270 窗给出\n(不需要知道 Z 的相位, 也不用正弦参考)\n\ncode 由数字序列在相邻两码间按占空比切换 (一阶 ΔΣ),\n累加 100 周期 → 等效 11.6 bit\n\n32 码真实权重 W(code) 由共享 ADC 逐码标定 (查表)\n开关切换在采样瞬间之外; DAC 支路不接触线圈',
        fontsize=8, color=C_DAC, va='top')

# ===================== 5. 时钟链 + 时序 =====================
box(8.0, 7.0, 1.3, 0.6, '125 MHz', color=C_CLK, fs=9)
box(9.6, 7.0, 1.7, 0.6, 'PI 1/512 (数字码)', color=C_CLK, fs=8)
box(11.6, 7.0, 1.3, 0.6, '÷16', color=C_CLK, fs=9)
box(13.2, 7.0, 3.6, 0.6, '相位发生器: φ0/90/180/270, _e 提前, _t 转移 (不交叠)', color=C_CLK, fs=7.5)
arrow((9.3, 7.3), (9.6, 7.3), color=C_CLK); arrow((11.3, 7.3), (11.6, 7.3), color=C_CLK); arrow((12.9, 7.3), (13.2, 7.3), color=C_CLK)
ax.text(12.3, 7.72, 'f0 = 125/16 = 7.8125 MHz, 一周期 16 拍', fontsize=8, color=C_CLK, ha='center', va='bottom')
X0, XL = 9.6, 6.4; t = np.linspace(0, 2, 400); y0 = 5.95
ax.plot(X0 + t * XL / 2, y0 + 0.35 * np.sin(2 * np.pi * t), color=C_SIG, lw=1.2); ax.text(8.2, y0, 'V_a', fontsize=8, color=C_SIG)
ax.text(12.8, 6.45, '← 一个载波周期 128 ns →', fontsize=7.5, color=C_CLK, ha='center')
rows = [('φ0_e / φ0', 0.0), ('φ90_e / φ90', 0.25), ('φ180_e / φ180', 0.5), ('φ270_e / φ270', 0.75)]
for k, (nm, ph) in enumerate(rows):
    yy = y0 - 0.65 - k * 0.42; ax.text(8.2, yy, nm, fontsize=7.5, color=C_CLK)
    ax.plot([X0, X0 + XL], [yy, yy], color=C_CLK, lw=0.8)
    for cyc in range(2):
        x0 = X0 + (cyc + ph) * XL / 2
        ax.plot([x0, x0, x0 + 0.2, x0 + 0.2], [yy, yy + 0.24, yy + 0.24, yy], color=C_CLK, lw=1.1)
        ax.plot([x0 + 0.07, x0 + 0.07], [yy, yy + 0.24], color=C_CLK, lw=0.6, ls=':')
yy = y0 - 0.65 - 4 * 0.42; ax.text(8.2, yy, 'I_null 窗 t_w', fontsize=7.5, color=C_DAC); ax.plot([X0, X0 + XL], [yy, yy], color=C_DAC, lw=0.8)
for cyc in range(2):
    for ph, s in [(0.25, '+'), (0.75, '−')]:
        x0 = X0 + (cyc + ph) * XL / 2 - 0.3
        ax.plot([x0, x0, x0 + 0.6, x0 + 0.6], [yy, yy + 0.24, yy + 0.24, yy], color=C_DAC, lw=1.1); ax.text(x0 + 0.3, yy + 0.3, s, fontsize=8, ha='center', color=C_DAC)
yy -= 0.42; ax.text(8.2, yy, 'code[4:0]', fontsize=7.5, color=C_DAC); ax.text(X0, yy, ' c  |  c+1  |  c  |  c  |  c+1 ...   ΔΣ 占空比序列, 7.8 MHz 更新', fontsize=7.5, color=C_DAC, va='center')
yy -= 0.42; ax.text(8.2, yy, 'read / rst', fontsize=7.5, color=C_DIG); ax.text(X0, yy, ' 每 100 周期: MUX 选中 → ADC 读 I,Q 累加器 → rst (或连续累加不清零)', fontsize=7.5, color=C_DIG, va='center')
yy -= 0.42; ax.text(8.2, yy, '驻留', fontsize=7.5, color=C_DIG); ax.text(X0, yy, ' 1600 周期 = 200 µs (或 1 ms) → 16~100 次读数; 多通道并行, ADC 轮询', fontsize=7.5, color=C_DIG, va='center')

# ===================== 6. 共享 MUX / ADC =====================
box(18.0, 9.8, 1.1, 3.4, 'MUX\n×38\n(I,Q)', fc='#f2f2f2', fs=9)
wire([(15.4, 12.75), (18.0, 12.75)]); wire([(15.4, 10.65), (17.6, 10.65), (17.6, 10.3), (18.0, 10.3)])
box(19.8, 10.9, 2.2, 1.6, 'S/H + 7-bit SAR\n125 MSps, 差分\n输入 Cin 2p', color=C_DIG, fs=9, bold=True)
arrow((19.1, 11.5), (19.8, 11.5)); ax.text(19.45, 11.72, 'S/H', fontsize=7.5, ha='center')
box(22.3, 10.9, 1.3, 1.6, 'Vref_ADC\n2 档\n±1 / ±4\nDAC 步', color=C_DIG, fs=7.5); arrow((22.3, 11.7), (22.0, 11.7), color=C_DIG)
box(18.0, 8.7, 5.5, 0.9, 'Vref 分配: 带隙 → buf_TX / buf_DAC / buf_ADC 三路独立缓冲\n(比率法消慢漂移; 独立缓冲隔离快速反冲)', fc='#fbfbfb', fs=7.8)
arrow((20.9, 10.9), (20.9, 9.65), color=C_DIG, ls=':'); 

# ===================== 7. 数字 =====================
lines = [
    '① 读数累加/平均; ADC 量程 ±1 步, 溢出 → 切 ±4 步再捕获',
    '② 跟踪环: Q 残差 → code 与 ΔΣ 占空比 (7.8 MHz 序列)',
    '③ PI 环: I 残差 → PI 码, 带宽 <1 kHz, 锁定后可冻结',
    '④ |Z| = code_avg·W(code) + Q 残差/G_acc   → 位姿',
    '⑤ arg Z = PI 码·2π/512 + I 残差/|Z|        → 温度 (实部)',
    '⑥ 自标: DAC 32 码权重 W, PI 台阶, 通道互易性 Z_ij = Z_ji',
    '⑦ 多频点: 各通道 f0 = 125/16 ± n·Δf, Δf = 整数/驻留',
    '⑧ 输出: 每帧 19 自 + 42 边 复数, 或 (w,u,v) 场',
]
for i, tl in enumerate(lines): ax.text(17.95, 7.2 - i * 0.5, tl, fontsize=8.2, color=C_DIG, va='center')
arrow((20.9, 8.5), (20.9, 7.98), color=C_DIG)
# 反馈线
wire([(17.8, 2.6), (17.65, 2.6), (17.65, 6.72), (10.45, 6.72)], color=C_CLK, ls='--'); arrow((10.45, 6.72), (10.45, 7.0), color=C_CLK, ls='--')
ax.text(16.9, 6.85, 'PI 码', fontsize=7.5, color=C_CLK, ha='center')
wire([(17.8, 2.0), (17.55, 2.0), (17.55, 1.3), (7.7, 1.3)], color=C_DAC, ls='--'); arrow((7.7, 1.3), (7.5, 1.3), color=C_DAC, ls='--')
ax.text(12.6, 1.45, 'code[4:0] + 占空比序列 (7.8 MHz) → b1..b31', fontsize=7.5, color=C_DAC, ha='center')
wire([(17.8, 1.4), (17.45, 1.4), (17.45, 0.15), (0.15, 0.15), (0.15, 13.55), (3.2, 13.55)], color=C_TX, ls='--'); arrow((3.2, 13.55), (3.2, 12.9), color=C_TX, ls='--')
ax.text(12.6, 0.2, '正弦 DAC 幅度 / 频点码 (每驻留一次)', fontsize=7.5, color=C_TX, ha='center', va='bottom')
ax.text(3.95, 13.6, '幅度/频点码', fontsize=7.5, color=C_TX, ha='center', va='bottom')

fig.suptitle('图表36  方案C ASIC 单通道详细原理图: 电流镜副本 → 5-bit 电流舵调零 DAC → 底板四相采样 → 电荷累加 → 共享 7-bit SAR', fontsize=13.5, fontweight='bold', y=0.985)
out = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/图表36_ASIC接收电路详细.png'
fig.savefig(out, bbox_inches='tight', facecolor='white'); print('saved', out)
