#!/usr/bin/env python3
"""图表32: 方案B (串联半端口) 判决汇总
a: 接线/观测打包示意  b: 0.2mm邻缘二次机理  c: 可解性对比  d: A vs B 汇总
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
js = json.load(open(REP + 'honeycomb_series.json'))

fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), facecolor='white')

# ---------- (a) 接线示意 ----------
ax = axes[0, 0]
ax.set_facecolor('white')
# 单元电路: T 与 B 串联
for y_, lab, col in [(3.0, '表层环 T (动)', '#d62728'), (1.0, '锚点环 B (定)', '#1f77b4')]:
    ax.add_patch(Circle((2.0, y_), 0.55, fill=False, color=col, lw=3))
    ax.text(3.0, y_, lab, fontsize=12, va='center')
ax.plot([2.0, 2.0], [1.55, 2.45], color='k', lw=1.5)          # 串联线
ax.plot([1.45, 0.7, 0.7], [3.0, 3.0, 0.2], color='k', lw=1.5)
ax.plot([1.45, 0.9, 0.9], [1.0, 1.0, 0.2], color='k', lw=1.5)
ax.add_patch(Rectangle((0.55, -0.35), 0.5, 0.55, fill=False, lw=1.5))
ax.text(0.8, -0.65, '单端口\nS参数+驱动', ha='center', fontsize=10.5)
ax.annotate('互感增强方向串联\nL_port = L_T + L_B + 2M(T,B)\n'
            '≈ 24.5nH(单匝) x225 = 5.5µH', xy=(2.0, 2.0), xytext=(4.6, 2.0),
            fontsize=11, va='center',
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(4.6, 0.3,
        '端口数: 38 -> 19 (减半)\n'
        '自观测: L_port 含 2M(T,B), 对压缩一阶敏感\n'
        '邻边观测: 3个独立互感被打包为1个和\n'
        '  M_edge = TT + TB(i>j) + TB(j>i) + BB(常数)\n'
        '总观测: 145 -> 61 (19自感+42邻边)',
        fontsize=11, va='top',
        bbox=dict(boxstyle='round', fc='#f5f5f0', ec='#999'))
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-1.4, 4.2)
ax.axis('off')
ax.set_title('32a  方案B: 上下环串联单端口', fontsize=13)

# ---------- (b) 边缘二次机理 ----------
ax = axes[0, 1]
vals = [0.0, 5.812, 4.480, 18.449]
labs = ['共同 z\n(-0.05mm)', '差分 dz\n(46um)', '双环倾斜\n(约0.5°)', '实际组合']
cols = ['#999', '#1f77b4', '#ff7f0e', '#d62728']
ax.bar(range(4), vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f'{v:+.1f}', ha='center', fontsize=11)
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=10.5)
ax.set_ylabel('相邻表层环互感变化 dTT (pH)')
ax.set_title('32b  0.2mm 邻缘间隙 => TT 耦合标称点纯二次\n'
             '(一阶恰为零, 标称 Jacobian 看不见; 线性预测全部为 0)\n'
             '工作点对称破缺后才出现一阶斜率', fontsize=11.5)
ax.grid(alpha=0.3, axis='y')

# ---------- (c) 可解性对比 ----------
ax = axes[1, 0]
cats = ['B 无先验\n61观测/95未知', 'B+柔性先验\n标称点',
        'B+柔性先验\n工作点', 'A+柔性先验\n(对照)']
# 用规范固定后的条件数; B无先验用无穷(画为截断柱)
conds = [None, js['fixed']['cond_gauge_fixed'],
         js['local_test']['cond_gauge_fixed'], 18.0]
xs = np.arange(4)
bars = [1e4 if c is None else c for c in conds]
cc = ['#8b0000', '#ff7f0e', '#2ca02c', '#1f77b4']
ax.bar(xs, bars, color=cc)
ax.set_yscale('log')
ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 1e4]))
ax.yaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1000', '不可解']))
labels_on = ['不可解\n34维盲空间', f"{conds[1]:.0f}\n(+3规范模式)",
             f"{conds[2]:.0f}\n(+3规范模式)", f"{conds[3]:.0f}\n(+3规范模式)"]
for x, b, t in zip(xs, bars, labels_on):
    ax.text(x, b * 1.2, t, ha='center', fontsize=10.5)
ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=10)
ax.set_ylim(1, 1e5)
ax.set_ylabel('去规范条件数')
ax.set_title('32c  可解性判决\n规范模式=面内共模平移x2+整体旋转 (串联打包抵消绝对参考,\n'
             '由边界/力学约束固定; A 的规范模式为独立评估同一先验所得)', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# ---------- (d) 汇总表 ----------
ax = axes[1, 1]
ax.axis('off')
lt = js['local_test']
tv2 = js.get('tracking_v2', None)
rows = [
    ['', '方案A 独立双端口', '方案B 串联单端口'],
    ['端口数 (19单元)', '38', '19 (减半)'],
    ['观测数', '145', '61'],
    ['无先验可解性', '满秩 cond=55', '不可解 (34维盲)'],
    ['柔性先验下', 'cond=18', f"cond={lt['cond_gauge_fixed']:.0f} +3规范模式"],
    ['w场噪声分辨率', '约 1.1 um', '约 9 um'],
    ['面内噪声分辨率', '约 1.3 um', '约 13 um'],
    ['反演算法要求', '标称J拟牛顿即可', '自感自举 + 中心差分J高斯牛顿\n(边缘二次强非线性, 收敛盆地约30um)'],
    ['局部收敛验证', 'z误差 0.04um', '30um扰动1步GN -> 0.53um'],
]
bs = js.get('bootstrap', None)
if bs:
    rows.append(['冷启动端到端', 'z 0.04um / alpha 0.3001',
                 f"自举初值{bs['w_init_max_err_um']:.0f}um -> "
                 f"w {bs['w_final_max_err_um']:.1f}um\n"
                 f"压深 {bs['depth_um']:.0f}/400um, 面内 {bs['uv_final_max_err_um']:.1f}um"])
tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 1.75)
for j in range(3):
    tbl[0, j].set_facecolor('#e8e2d5')
ax.set_title('32d  方案 A vs B 汇总', fontsize=13)

plt.tight_layout()
out = REP + '图表32_串联方案判决.png'
plt.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)
from PIL import Image
im = Image.open(out)
if im.width > 2000:
    r = 2000 / im.width
    im.resize((2000, int(im.height * r)), Image.LANCZOS).save(out)
print(out, Image.open(out).size)
