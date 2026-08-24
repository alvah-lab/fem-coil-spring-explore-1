#!/usr/bin/env python3
"""图表31: 交错锚点高度设计方案 3D 图
19单元蜂窝 patch, 三角格三色化取一色下沉 0.9mm
左: 3D 视图; 右上: 俯视三色格; 右下: 剖面尺寸标注
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

R = 2.5
PITCH = 5.2
GAP_REST = 2.5
STAG = 0.9          # 下沉深度
BASE_TOP = 0.0      # 基座表面

# 三角格 + 三色化 (c = (i-j) mod 3), 取 c==0 下沉
a1 = np.array([PITCH, 0.0])
a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
units = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            units.append((p[0], p[1], (i - j) % 3))
print(f'{len(units)} units, lowered: {sum(1 for u in units if u[2]==0)}')

def ring(cx, cy, z, npt=120):
    t = np.linspace(0, 2 * np.pi, npt)
    return cx + R * np.cos(t), cy + R * np.sin(t), np.full_like(t, z)

def spring(cx, cy, z0, z1, nturn=4, npt=300):
    t = np.linspace(0, 2 * np.pi * nturn, npt)
    u = t / t[-1]
    return cx + R * np.cos(t), cy + R * np.sin(t), z0 + (z1 - z0) * u

C_TOP, C_NORM, C_LOW, C_SPR = '#d62728', '#1f77b4', '#0a9c4a', '#999999'

fig = plt.figure(figsize=(17, 9), facecolor='white')
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.15, 1])

# ================= 左: 3D 视图 =================
ax = fig.add_subplot(gs[:, 0], projection='3d')
ax.set_facecolor('white')
order = sorted(units, key=lambda u: -u[1])   # 后排先画
for cx, cy, c in order:
    low = (c == 0)
    zanc = BASE_TOP - (STAG if low else 0.0)
    x, y, z = ring(cx, cy, zanc)
    ax.plot(x, y, z, color=C_LOW if low else C_NORM, lw=3.5, alpha=0.95)
    x, y, z = spring(cx, cy, BASE_TOP, GAP_REST)
    ax.plot(x, y, z, color=C_SPR, lw=0.7, alpha=0.5)
    x, y, z = ring(cx, cy, GAP_REST)
    ax.plot(x, y, z, color=C_TOP, lw=3.5, alpha=0.9)

ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
ax.set_box_aspect((1, 1, 0.38))
ax.view_init(elev=20, azim=-64)
ax.set_zlim(-1.5, 4.5)
ax.set_title('31a  交错锚点方案 3D (19 单元)\n'
             f'绿=下沉锚点环 (z = -{STAG}mm)   蓝=标称锚点环 (z=0)   '
             '红=表层环 (静息 z=2.5mm)', fontsize=13)
ax.legend(handles=[
    Line2D([], [], color=C_TOP, lw=3, label='表层环 (随弹簧动)'),
    Line2D([], [], color=C_NORM, lw=3, label='标称锚点 z=0'),
    Line2D([], [], color=C_LOW, lw=3, label=f'下沉锚点 z=-{STAG}mm'),
], loc='upper left', fontsize=10)

# ================= 右上: 俯视三色格 =================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')
for cx, cy, c in units:
    low = (c == 0)
    col = C_LOW if low else C_NORM
    ax2.add_patch(Circle((cx, cy), R, fill=low, color=col,
                         alpha=0.35 if low else 1.0, lw=1.6, ec=col))
    ax2.text(cx, cy, '沉' if low else '标', ha='center', va='center',
             fontsize=11, color='#0a6c34' if low else C_NORM)
# 标注一个单元的邻域
hx, hy = a1  # (5.2, 0)
ax2.add_patch(Circle((hx, hy), R + 0.35, fill=False, color='#d62728',
                     lw=2, ls='--'))
ax2.annotate('任一单元: 自身+6邻居锚点中必含两种高度',
             xy=(hx + R * 0.5, hy - R * 0.8), xytext=(2.5, -14.6),
             fontsize=10.5, color='#d62728', ha='center',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
ax2.set_aspect('equal')
ax2.set_xlim(-13, 13); ax2.set_ylim(-15.6, 12.5)
ax2.set_title('31b  俯视: 三角格三色化, 取一色下沉 (1/3)\n'
              '标称单元: 邻居含3个下沉 | 下沉单元: 自身即下沉', fontsize=11.5)
ax2.set_xlabel('x (mm)'); ax2.set_ylabel('y (mm)')

# ================= 右下: 剖面 =================
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('white')
# 基座
ax3.add_patch(Rectangle((-9, -1.8), 18, 1.8, fc='#e8e2d5', ec='#998', hatch='//'))
row = [(-PITCH, True), (0.0, False), (PITCH, True)]   # 沉,标,沉 示意
for cx, low in row:
    zanc = -STAG if low else 0.0
    if low:   # 凹槽
        ax3.add_patch(Rectangle((cx - R - 0.3, zanc - 0.15), 2 * R + 0.6,
                                -zanc + 0.16, fc='white', ec='none'))
    col = C_LOW if low else C_NORM
    ax3.plot([cx - R, cx + R], [zanc, zanc], color=col, lw=1.2)
    ax3.plot([cx - R, cx + R], [zanc, zanc], 'o', color=col, ms=10)
    # 弹簧 (虚线示意)
    for xx in (cx - R * 0.75, cx + R * 0.75):
        ax3.plot([xx, xx], [0.05, GAP_REST - 0.05], color=C_SPR, lw=1, ls=':')
    ax3.plot([cx - R, cx + R], [GAP_REST, GAP_REST], color=C_TOP, lw=1.2)
    ax3.plot([cx - R, cx + R], [GAP_REST, GAP_REST], 'o', color=C_TOP, ms=10)

# 尺寸标注
def dim(ax, x, z0, z1, text, dx=0.25):
    ax.annotate('', xy=(x, z1), xytext=(x, z0),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1))
    ax.text(x + dx, (z0 + z1) / 2, text, fontsize=10, va='center')

dim(ax3, PITCH + R + 0.5, -STAG, 0.0, f'下沉 {STAG}mm')
dim(ax3, R + 0.5, 0.0, GAP_REST, '静息 2.5mm\n(全压缩 1.0mm)')
ax3.annotate('', xy=(PITCH - R, 3.6), xytext=(R, 3.6),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1))
ax3.text(PITCH / 2 + 0.1, 3.85, '净间隙 0.2mm', ha='center', fontsize=10)
ax3.text(0, -2.5, '下沉锚点嵌入固定基座凹槽: 不占压缩行程, 不碰活动件\n'
         '两个接收高度 => 共模平移/倾斜比值方程解耦', ha='center', fontsize=10.5,
         bbox=dict(boxstyle='round', fc='#f5f5f0', ec='#999'))
ax3.set_xlim(-9.3, 11.5); ax3.set_ylim(-3.6, 4.3)
ax3.set_aspect('equal')
ax3.set_title('31c  剖面 (沉-标-沉)', fontsize=11.5)
ax3.set_xlabel('x (mm)'); ax3.set_ylabel('z (mm)')

plt.tight_layout()
out = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/图表31_交错锚点3D方案.png'
plt.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)

from PIL import Image
im = Image.open(out)
if im.width > 2000:
    r = 2000 / im.width
    im.resize((2000, int(im.height * r)), Image.LANCZOS).save(out)
print(out, Image.open(out).size)
