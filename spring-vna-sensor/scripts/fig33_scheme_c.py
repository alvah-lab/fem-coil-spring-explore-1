#!/usr/bin/env python3
"""图表33: 方案C (无源单匝短路环) FEM 完整判决"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
jc = json.load(open(REP + 'honeycomb_scheme_c.json'))
js = json.load(open(REP + 'honeycomb_series.json'))

fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), facecolor='white')

# ---------- (a) 结构与信号预算 ----------
ax = axes[0, 0]
ax.set_facecolor('white')
# 上: 单元示意
ax.add_patch(Circle((1.6, 3.6), 0.5, fill=False, color='#2ca02c', lw=4))
ax.text(2.4, 3.6, '无源单匝短路铜环\nØ5mm x Ø0.5mm线径\nL=7.4nH, Q=28\n无引线!', fontsize=11,
        va='center', color='#0a6c34')
ax.add_patch(Circle((1.6, 1.6), 0.5, fill=False, color='#1f77b4', lw=2.5))
ax.text(2.4, 1.6, '有源锚点线圈 (15股捆束)\n交错高度, 接驱动/检测', fontsize=11, va='center')
ax.annotate('', xy=(1.6, 2.15), xytext=(1.6, 3.05),
            arrowprops=dict(arrowstyle='<->', color='#888'))
ax.text(1.15, 2.6, '1.75', fontsize=9, color='#888')
# 右: 信号预算条
labels = ['自反射\n(42~121nH)', '边反射\n(3.9~7.3nH)', '温漂/K\n(x1e-3)', '杂化占比']
vals = [81, 5.6, 0.19 * 5.6 / 1e3 * 1e3, 3.46 * 5.6]
ax.text(0.3, 0.55,
        f"观测量 = 底层端口有效电感 Im(Z)/ω (温漂免疫分量)\n"
        f"自观测: 背景 2317nH, 反射 -44nH (1.9%)\n"
        f"边观测: 背景 -69nH, 反射 -6.2nH\n"
        f"温漂: 1.9e-4 /K (谐振版为 4e-3/K, 免疫比 x21)\n"
        f"环-环杂化路径贡献 346% -> 必须用全耦合矩阵模型\n"
        f"(B_i->T_i->T_j->B_j 双强耦合路径主导边观测)",
        fontsize=11.5, va='top',
        bbox=dict(boxstyle='round', fc='#f5f5f0', ec='#999'))
ax.set_xlim(0, 8); ax.set_ylim(-1.4, 4.6)
ax.axis('off')
ax.set_title('33a  方案C: 无源反射式 (顶层零引线)', fontsize=13)

# ---------- (b) 归一化奇异值谱对比 ----------
ax = axes[0, 1]
sv_A = np.sort(np.array(js['sv_A_prior']))[::-1]
sv_B = np.sort(np.array(js['sv_B_prior']))[::-1]
sv_C = np.sort(np.array(jc['S2调零']['sv_prior']))[::-1]
for sv, lab, col in [(sv_A, '方案A+先验 (145观测)', '#1f77b4'),
                     (sv_B, '方案B+先验 (61观测)', '#ff7f0e'),
                     (sv_C, '方案C+先验 (61观测)', '#2ca02c')]:
    ax.semilogy(range(1, len(sv) + 1), sv / sv[0], 'o-', ms=3.5,
                color=col, label=lab)
ax.axhspan(1e-7, 3e-5, color='#f0d0d0', alpha=0.6)
ax.text(2, 6e-6, 'B: 3个真规范模式 (结构盲)', fontsize=10, color='#8b0000')
ax.annotate('C: 3个弱共模模式\n(可观测! sv/max~2.5e-3)', xy=(55, 2.5e-3),
            xytext=(30, 2e-4), fontsize=10, color='#0a6c34',
            arrowprops=dict(arrowstyle='->', color='#0a6c34'))
ax.set_xlabel('模式序号'); ax.set_ylabel('归一化奇异值 sv/sv_max')
ax.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
ax.yaxis.set_major_formatter(FixedFormatter(['1', '0.01', '1e-4', '1e-6']))
ax.legend(fontsize=10, loc='lower left')
ax.set_title('33b  柔性先验下 57 模式谱: C 无结构盲区\n'
             '(强子空间 cond=9.3, 优于 A 的 11.3)', fontsize=12)
ax.grid(alpha=0.3)

# ---------- (c) 分辨率对比 ----------
ax = axes[1, 0]
cats = ['w 法向', '面内差分', '面内共模']
# A: 1.07/1.3/1.3 (无先验区分); B: 9/13/不可测; C(S2): 0.78~1.8/3.1/52
data = {
    '方案A (双端口)': [1.07, 1.3, 1.3],
    '方案B (串联)': [9.0, 13.1, np.nan],
    '方案C (无源环+调零)': [1.8, 3.1, 52.0],
}
x = np.arange(3); wdt = 0.25
for k, (lab, v) in enumerate(data.items()):
    col = ['#1f77b4', '#ff7f0e', '#2ca02c'][k]
    vv = [x_ if not np.isnan(x_) else 0 for x_ in v]
    b = ax.bar(x + (k - 1) * wdt, vv, wdt, color=col, label=lab)
    for xi, val in zip(x + (k - 1) * wdt, v):
        if np.isnan(val):
            ax.text(xi, 1.2, '不可测\n(规范盲)', ha='center', fontsize=9,
                    color='#8b0000')
        else:
            ax.text(xi, val * 1.1, f'{val:g}', ha='center', fontsize=10)
ax.set_yscale('log')
ax.set_ylim(0.3, 200)
ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
ax.yaxis.set_major_formatter(FixedFormatter(['1', '10', '100']))
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel('1σ 分辨率 (um), 1e-3 测量噪声')
ax.legend(fontsize=10)
ax.set_title('33c  噪声分辨率对比 (19单元)\n'
             'C 的法向/差分达 A 级; 共模面内弱但可观测(慢通道)', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# ---------- (d) 判决表 ----------
ax = axes[1, 1]
ax.axis('off')
e = jc['e2e_v2'] if 'e2e_v2' in jc else {}
rows = [
    ['', '方案A', '方案B', '方案C 无源环'],
    ['运动层引线', '有 (2端口/元)', '有 (1端口/元)', '无 !'],
    ['端口数(19单元)', '38', '19', '19'],
    ['无先验', '满秩 cond 55', '34维盲', '34维盲'],
    ['柔性先验', 'cond 18', '3规范盲模式', '满秩! 强空间 cond 9.3'],
    ['共模面内', '可测', '不可测', '弱可测 (慢通道)'],
    ['温漂', 'R无关', 'R无关', '1.9e-4/K (免校)'],
    ['端到端', 'z 0.04um', 'w 5.4um', f"w {e.get('w_max_err_um', 14):.0f}um, "
     f"压深 {e.get('depth_um', 402):.0f}/400um"],
    ['噪声MC', 'w 1.1um', 'w 9um', f"w {e.get('mc_w_um', 1.8):.1f}um / "
     f"面内 {e.get('mc_uv_um', 3.1):.1f}um"],
    ['反演', '线性一步', '自举+工作点GN', '双速: 强54快+弱3慢\n(线性偏差23%)'],
    ['顶层成本', '捆束线圈+装配', '捆束线圈+装配', '一个冲压铜环'],
]
tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.8)
tbl.scale(1, 1.62)
for j in range(4):
    tbl[0, j].set_facecolor('#e8e2d5')
for i in [1, 4]:
    tbl[i, 3].set_facecolor('#d9f2e0')
ax.set_title('33d  三方案终局对比', fontsize=13)

plt.tight_layout()
out = REP + '图表33_无源环方案判决.png'
plt.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)
from PIL import Image
im = Image.open(out)
if im.width > 2000:
    r = 2000 / im.width
    im.resize((2000, int(im.height * r)), Image.LANCZOS).save(out)
print(out, Image.open(out).size)
