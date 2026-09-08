#!/usr/bin/env python3
"""图表34: 方案选型多维对比 (A/B/C)"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
COLS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), facecolor='white')

# ---------- (a) 双基准 FoM ----------
ax = axes[0, 0]
# 绝对基准 (um per nH) 与 相对基准 (um @1e-3载波)
abs_fom = {'A': [0.96, 2.3], 'B': [0.53, 0.68], 'C': [0.025, 6.3]}
rel_res = {'A': [1.07, 1.3], 'B': [9.0, 13.0], 'C': [23.6, 15.0]}
x = np.array([0, 1, 2.6, 3.6])
for k, (s, c) in enumerate(COLS.items()):
    v = abs_fom[s] + rel_res[s]
    ax.bar(x + (k - 1) * 0.26, v, 0.26, color=c, label=f'方案{s}')
    for xi, vi in zip(x + (k - 1) * 0.26, v):
        ax.text(xi, vi * 1.15, f'{vi:g}', ha='center', fontsize=9)
ax.set_yscale('log')
ax.set_ylim(0.01, 300)
ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '10']))
ax.set_xticks([0, 1, 2.6, 3.6])
ax.set_xticklabels(['w 法向', '面内差分', 'w 法向', '面内差分'], fontsize=11)
ax.axvline(1.8, color='k', lw=0.8, ls='--')
ax.text(0.5, 90, '绝对噪声基准\n(µm per nH, 调零后)\nC 胜出 38 倍', ha='center',
        fontsize=11, bbox=dict(boxstyle='round', fc='#e8f4e8'))
ax.text(3.1, 90, '相对噪声基准\n(µm @1e-3载波, 无调零)\nA 胜出', ha='center',
        fontsize=11, bbox=dict(boxstyle='round', fc='#e8ecf4'))
ax.legend(fontsize=10, loc='lower right')
ax.set_title('34a  比较基准翻转排名: AFE 投入(调零)是方案选择的一部分',
             fontsize=12)
ax.grid(alpha=0.3, axis='y')

# ---------- (b) 同分辨率驱动能耗比 ----------
ax = axes[0, 1]
cats = ['同 w 分辨率', '同面内分辨率']
en = {'A': [1, 1], 'B': [0.25, 0.07], 'C': [3e-4, 3.1]}
x = np.arange(2)
for k, (s, c) in enumerate(COLS.items()):
    ax.bar(x + (k - 1) * 0.26, en[s], 0.26, color=c, label=f'方案{s}')
    for xi, vi in zip(x + (k - 1) * 0.26, en[s]):
        lab = f'{vi:g}' if vi >= 0.01 else '1/3000'
        ax.text(xi, vi * 1.3, lab, ha='center', fontsize=10)
ax.set_yscale('log')
ax.set_ylim(1e-4, 30)
ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 10]))
ax.yaxis.set_major_formatter(FixedFormatter(['1e-4', '0.01', '1', '10']))
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=12)
ax.set_ylabel('驱动能量 (归一到 A)')
ax.legend(fontsize=10)
ax.set_title('34b  同等效果驱动能耗 (绝对基准, 含驻留数与R差异)\n'
             '法向主导应用: C 省 ~3000 倍 (先撞模型/机械地板为止)', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# ---------- (c) 密度标度律 ----------
ax = axes[1, 0]
p = np.linspace(1.5, 6, 100)          # pitch mm
p0 = 5.2
drive0 = 0.001                         # A方案 19元@280Hz 平均驱动 mW/cm2 量级示意
afe0 = 1.0                             # AFE mW/cm2 示意
drive = drive0 * (p0 / p) ** 5         # 1/(N^3 p^5), N固定
afe = afe0 * (p0 / p) ** 2
ax.semilogy(p, drive, color='#d62728', lw=2,
            label=r'单位面积驱动功耗 $\propto 1/p^5$ (线径随pitch缩)')
ax.semilogy(p, afe, color='#1f77b4', lw=2,
            label=r'单位面积 AFE 功耗 $\propto 1/p^2$')
ax.semilogy(p, drive / 38 ** 2, color='#2ca02c', lw=2, ls='--',
            label='方案C 法向通道等效 (38x 灵敏度余量)')
ax.axvline(p0, color='k', lw=0.8, ls=':')
ax.text(p0 - 0.75, 3.0, '当前设计 p=5.2mm', fontsize=10)
ax.set_xlabel('阵列 pitch p (mm)')
ax.set_ylabel('单位面积功耗 (相对值)')
ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 100, 1e4]))
ax.yaxis.set_major_formatter(FixedFormatter(['1e-4', '0.01', '1', '100', '1e4']))
ax.legend(fontsize=10, loc='upper right')
ax.set_title('34c  高密度的功耗惩罚: 驱动 $p^{-5}$ vs AFE $p^{-2}$\n'
             '(缓解: 榨满匝数 N³ 红利 / 提频 f² / 选C法向余量)', fontsize=12)
ax.grid(alpha=0.3)

# ---------- (d) 决策矩阵 ----------
ax = axes[1, 1]
ax.axis('off')
rows = [
    ['场景', '首选', '关键理由'],
    ['研发/标定/最高精度全六维', 'A', '全可测+线性一步, AFE 最宽容'],
    ['成本极敏感, ~10µm 够用', 'B', '端口减半, AFE 最少'],
    ['长寿命/高弯折 (皮肤/穿戴)', 'C', '运动层零引线'],
    ['高密度 pitch<3mm', 'C', '顶层可印刷, 38x 法向余量抗 $p^{-5}$'],
    ['电池供电+法向为主', 'C', '同分辨率驱动能耗 ~1/3000'],
    ['面内力为主 (剪切仪)', 'A', 'C 面内贵 3 倍且共模慢'],
    ['共模切向绝对量必需', 'A', 'B 不可测, C 慢通道'],
    ['AFE 无调零能力', 'A', 'C 不抵消载波即崩'],
]
tbl = ax.table(cellText=rows, loc='center', cellLoc='center',
               colWidths=[0.42, 0.1, 0.48])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 1.8)
for j in range(3):
    tbl[0, j].set_facecolor('#e8e2d5')
for i in range(1, len(rows)):
    s = rows[i][1]
    tbl[i, 1].set_facecolor({'A': '#dce8f4', 'B': '#fdeedd',
                             'C': '#d9f2e0'}[s])
ax.set_title('34d  决策矩阵: 精度选A, 省钱选B, 量产/密度/寿命/功耗选C',
             fontsize=12.5)

plt.tight_layout()
out = REP + '图表34_方案选型指南.png'
plt.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)
from PIL import Image
im = Image.open(out)
if im.width > 2000:
    r = 2000 / im.width
    im.resize((2000, int(im.height * r)), Image.LANCZOS).save(out)
print(out, Image.open(out).size)
