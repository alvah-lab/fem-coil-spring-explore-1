#!/usr/bin/env python3
"""图表35: ASIC 接收电路简易原理图 (7-bit ADC / 5-bit DAC / 四相开关电容累加 / PI 过零锁相)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 10.5), dpi=110)
ax.set_xlim(0, 18); ax.set_ylim(0, 10.5); ax.axis('off')
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
LW = 1.6
C_SIG = '#1f4e79'; C_CLK = '#c55a11'; C_DIG = '#375623'; C_DAC = '#7030a0'

def wire(pts, color='k', lw=LW, ls='-'):
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, solid_capstyle='round')

def box(x, y, w, h, text, color='k', fc='white', fs=10.5, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal')

def cap(x, y, horiz=False, label=None, dx=0.09):
    if horiz:
        wire([(x - dx, y - 0.22), (x - dx, y + 0.22)]); wire([(x + dx, y - 0.22), (x + dx, y + 0.22)])
    else:
        wire([(x - 0.22, y - dx), (x + 0.22, y - dx)]); wire([(x - 0.22, y + dx), (x + 0.22, y + dx)])
    if label: ax.text(x + 0.28, y, label, fontsize=9, va='center')

def switch(x, y, label, color=C_CLK, length=0.5):
    # 水平开关: 左触点 x, 右触点 x+length
    ax.plot([x, x + length * 0.75], [y, y + 0.22], color='k', lw=LW)
    ax.add_patch(Circle((x, y), 0.04, fc='k')); ax.add_patch(Circle((x + length, y), 0.04, fc='k'))
    ax.text(x + length / 2, y + 0.36, label, ha='center', fontsize=9.5, color=color, fontweight='bold')

def gnd(x, y):
    wire([(x, y), (x, y - 0.18)]); wire([(x - 0.18, y - 0.18), (x + 0.18, y - 0.18)])
    wire([(x - 0.11, y - 0.26), (x + 0.11, y - 0.26)]); wire([(x - 0.04, y - 0.34), (x + 0.04, y - 0.34)])

def opamp(x, y, label='', size=0.55):
    tri = Polygon([(x, y - size), (x, y + size), (x + 1.3 * size, y)], closed=True, fc='white', ec='k', lw=LW)
    ax.add_patch(tri); ax.text(x + 0.18, y + 0.28, '−', fontsize=11); ax.text(x + 0.18, y - 0.36, '+', fontsize=11)
    if label: ax.text(x + 0.45, y, label, fontsize=9, ha='center', va='center')

def coil(x, y, n=4, r=0.16, vertical=True):
    for i in range(n):
        if vertical:
            ax.add_patch(Arc((x, y + (2 * i + 1) * r), 2 * r, 2 * r, theta1=-90, theta2=90, lw=LW, color='k'))
        else:
            ax.add_patch(Arc((x + (2 * i + 1) * r, y), 2 * r, 2 * r, theta1=0, theta2=180, lw=LW, color='k'))

def arrow(p0, p1, color='k', lw=1.4, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=12, color=color, lw=lw, linestyle=ls))

# ===== 1. 线圈 + 铜环 + 电流源驱动 =====
ax.text(0.3, 10.1, '每通道 (×19 或 ×38)', fontsize=12, fontweight='bold', color=C_SIG)
ax.add_patch(Rectangle((0.2, 0.45), 12.4, 9.45, fc='none', ec=C_SIG, lw=1.2, ls='--'))
ax.add_patch(Rectangle((1.0, 8.2), 1.6, 0.16, fc='#c0504d', ec='k')); ax.text(1.8, 8.6, '无源铜环 (动)', ha='center', fontsize=9.5)
ax.text(1.8, 7.75, 'M(位姿)', ha='center', fontsize=9, color='#c0504d')
wire([(1.8, 8.1), (1.8, 7.4)], color='#c0504d', ls=':')
coil(1.8, 5.6, n=4, r=0.16)
ax.text(2.3, 5.75, 'L_coil\n(PCB 螺旋)', fontsize=9, va='center')
wire([(1.8, 5.6), (1.8, 5.2)]); gnd(1.8, 5.2)
wire([(1.8, 6.88), (1.8, 7.4)]); ax.add_patch(Circle((1.8, 7.4), 0.05, fc='k'))
wire([(1.8, 7.4), (0.9, 7.4), (0.9, 6.6)])
ax.add_patch(Circle((0.9, 6.25), 0.35, fc='white', ec='k', lw=LW)); arrow((0.9, 6.55), (0.9, 5.95))
ax.text(0.9, 6.95, 'I_drive 电流源', fontsize=8.5, ha='center', va='center')
wire([(0.9, 5.9), (0.9, 5.2)]); gnd(0.9, 5.2)
box(0.4, 4.0, 1.6, 0.7, '正弦 DAC\n8-bit 125MSps', color=C_DAC, fs=9)
arrow((1.2, 4.7), (0.95, 5.15), color=C_DAC); ax.text(1.35, 4.95, 'gm', fontsize=9, color=C_DAC)
wire([(1.25, 6.25), (2.9, 6.25)], color=C_DAC, ls='--'); ax.text(2.05, 6.52, 'I_drive 复制 ×1/k', fontsize=8.5, color=C_DAC, ha='center')

# ===== 2. LNA =====
wire([(1.8, 7.4), (3.2, 7.4)])
opamp(3.2, 7.4, size=0.5); ax.text(3.5, 7.4, '缓冲\n×1~2', fontsize=8.5, ha='center', va='center')
wire([(3.2 + 0.65, 7.4), (4.6, 7.4)]); ax.add_patch(Circle((4.6, 7.4), 0.05, fc='k'))
ax.text(4.55, 7.65, 'V_a = Z·I  (载波 <0.5 V, 如 5 mA)', fontsize=9, ha='center', color=C_SIG)

# ===== 3. 四相采样开关 + 采样电容 =====
ys = {'φ0': 8.9, 'φ180': 7.9, 'φ90': 5.6, 'φ270': 4.6}
for ph, y in ys.items():
    wire([(4.6, 7.4), (4.6, y), (5.0, y)])
    switch(5.0, y, ph); wire([(5.5, y), (6.1, y)])
    cap(6.1, y, horiz=True, label='Cs'); wire([(6.19, y), (6.7, y)])
wire([(6.7, 8.9), (6.7, 7.9)]); wire([(6.7, 8.4), (7.3, 8.4)])
ax.text(6.85, 8.62, '+', fontsize=10); ax.text(6.85, 7.98, '−', fontsize=10)
opamp(7.3, 8.4, size=0.45); cap(7.9, 9.15, horiz=True, label='C_acc,I')
wire([(7.3, 8.85), (7.3, 9.15), (7.81, 9.15)]); wire([(7.99, 9.15), (8.5, 9.15), (8.5, 8.4)])
wire([(7.3 + 0.585, 8.4), (9.4, 8.4)]); ax.add_patch(Circle((9.4, 8.4), 0.05, fc='k'))
ax.text(8.95, 8.62, 'I 累加器', fontsize=9.5, color=C_SIG)
ax.text(8.95, 8.1, '(过零残差, 相位轴)', fontsize=8.5, color=C_SIG)
switch(7.55, 9.5, 'rst', color='grey', length=0.4); wire([(7.3, 9.15), (7.3, 9.5), (7.55, 9.5)]); wire([(7.95, 9.5), (8.5, 9.5), (8.5, 9.15)])
wire([(6.7, 5.6), (6.7, 4.6)]); wire([(6.7, 5.1), (7.3, 5.1)])
ax.text(6.85, 5.32, '+', fontsize=10); ax.text(6.85, 4.68, '−', fontsize=10)
opamp(7.3, 5.1, size=0.45); cap(7.9, 5.85, horiz=True, label='C_acc,Q')
wire([(7.3, 5.55), (7.3, 5.85), (7.81, 5.85)]); wire([(7.99, 5.85), (8.5, 5.85), (8.5, 5.1)])
wire([(7.3 + 0.585, 5.1), (9.4, 5.1)]); ax.add_patch(Circle((9.4, 5.1), 0.05, fc='k'))
ax.text(8.95, 5.32, 'Q 累加器', fontsize=9.5, color=C_SIG)
ax.text(8.95, 4.8, '(幅度残差, 位姿轴)', fontsize=8.5, color=C_SIG)
switch(7.55, 6.2, 'rst', color='grey', length=0.4); wire([(7.3, 5.85), (7.3, 6.2), (7.55, 6.2)]); wire([(7.95, 6.2), (8.5, 6.2), (8.5, 5.85)])

# ===== 4. 5-bit 电荷 DAC =====
box(3.6, 1.9, 2.6, 1.3, '5-bit 电荷 DAC\n32 个 Cu 单元\n(比率型, 输入=I_drive 复制)', color=C_DAC, fs=9)
wire([(2.9, 6.25), (2.9, 2.55), (3.6, 2.55)], color=C_DAC, ls='--')
wire([(6.2, 2.55), (6.6, 2.55)], color=C_DAC); switch(6.6, 2.55, 'φ90/φ270', color=C_CLK, length=0.5)
wire([(7.1, 2.55), (7.3, 2.55), (7.3, 4.65)], color=C_DAC)
ax.text(7.45, 4.15, '−Q_null\n每周期扣除', fontsize=8.5, color=C_DAC, va='center')
ax.text(4.9, 1.75, '码序列: 相邻两码占空比切换 (ΔΣ), 累加 100 周期 → 等效 11.6 bit', fontsize=8.5, color=C_DAC, ha='center', va='top')

# ===== 5. 时钟 =====
box(0.4, 0.75, 1.6, 0.6, '125 MHz', color=C_CLK, fs=9.5)
box(0.4, 1.6, 1.6, 0.75, '相位插值器\nPI 1/512', color=C_CLK, fs=9.5)
box(0.4, 2.6, 1.6, 0.75, '÷16 四相\nf0 = 7.8125 MHz', color=C_CLK, fs=9)
arrow((1.2, 1.35), (1.2, 1.6), color=C_CLK); arrow((1.2, 2.35), (1.2, 2.6), color=C_CLK)
wire([(2.0, 2.97), (2.5, 2.97), (2.5, 1.5)], color=C_CLK)
arrow((2.5, 1.5), (9.0, 1.5), color=C_CLK, ls=':')
ax.text(5.8, 1.28, 'φ0 φ90 φ180 φ270 → 采样开关 / DAC 开关 / 累加器 rst', fontsize=8.5, color=C_CLK, ha='center', va='top')

# ===== 6. 复用 + 共享 ADC =====
box(10.0, 6.3, 1.2, 3.0, 'MUX\n×38', fc='#f2f2f2', fs=10)
wire([(9.4, 8.4), (10.0, 8.4)]); wire([(9.4, 5.1), (9.7, 5.1), (9.7, 7.0), (10.0, 7.0)])
ax.text(12.9, 10.1, '共享 (每瓦片一个)', fontsize=12, fontweight='bold', color=C_DIG)
box(13.0, 7.2, 2.0, 1.4, '7-bit SAR ADC\n125 MSps\n量程 = ±1 DAC 步', color=C_DIG, fs=9.5, bold=True)
arrow((11.2, 7.9), (13.0, 7.9)); ax.text(12.1, 8.1, '每驻留每通道\n读 ~100~3300 次', fontsize=8.5, ha='center')

# ===== 7. 数字 =====
box(13.0, 2.0, 4.6, 4.4, '', color=C_DIG, fc='#f7fbf5')
ax.text(15.3, 6.1, '数字 (RISC-V / 状态机)', fontsize=11, fontweight='bold', color=C_DIG, ha='center')
lines = [
    '① 读数累加 / 平均 (N^0.5)',
    '② 跟踪环: Q 残差 → DAC 占空比码',
    '③ PI 环: I 残差 → PI 码 (带宽 <1 kHz)',
    '④ |Z| ∝ DAC 码 + Q 残差   → 位姿',
    '⑤ arg Z = PI 码 + I 残差 / |Z| → 温度',
    '⑥ DAC 32 码自标 (ADC 逐码测), 互易性标通道',
    '⑦ 输出: 每帧 19 自 + 42 边 复数, 或直接 (w,u,v)',
]
for i, t in enumerate(lines):
    ax.text(13.2, 5.6 - i * 0.5, t, fontsize=9.2, color=C_DIG, va='center')
arrow((14.0, 7.2), (14.0, 6.4), color=C_DIG)
# 反馈线
wire([(13.0, 3.7), (5.9, 3.7)], color=C_DAC, ls='--'); arrow((5.9, 3.7), (5.9, 3.2), color=C_DAC, ls='--')
ax.text(10.2, 3.85, 'DAC 占空比码 (7.8 MHz 更新)', fontsize=8.5, color=C_DAC, ha='center')
wire([(14.0, 2.0), (14.0, 0.95), (2.25, 0.95)], color=C_CLK, ls='--'); arrow((2.25, 0.95), (2.25, 1.97), color=C_CLK, ls='--'); wire([(2.25, 1.97), (2.0, 1.97)], color=C_CLK, ls='--')
ax.text(8.0, 1.05, 'PI 码', fontsize=8.5, color=C_CLK, ha='center')
wire([(15.0, 2.0), (15.0, 0.65), (2.45, 0.65), (2.45, 4.35)], color=C_DAC, ls='--'); arrow((2.45, 4.35), (2.0, 4.35), color=C_DAC, ls='--')
ax.text(11.0, 0.72, '正弦 DAC 幅度 / 频点 (三者共用 Vref → 比率测量)', fontsize=8.5, color=C_DAC, ha='center')

ax.text(0.3, 0.2,
        '要点: 电流源驱动 → 只测电压 |  0/180 样本=鉴相 (推 PI), 90/270 样本=幅度 |  载波在 Q 累加器内按周期扣除 (5-bit DAC 占空比 ΔΣ), 放大靠累加不靠 LNA |  ADC 只量化残差, 分辨率靠读数次数',
        fontsize=9, color='k', va='center')
fig.suptitle('图表35  方案C ASIC 接收通道简易原理图: 7-bit ADC + 5-bit 电荷 DAC + 四相开关电容累加 + PI 过零锁相',
             fontsize=13.5, fontweight='bold', y=0.985)
out = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/图表35_ASIC接收电路.png'
fig.savefig(out, bbox_inches='tight', facecolor='white'); print('saved', out)
