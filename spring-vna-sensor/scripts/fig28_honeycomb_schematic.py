#!/usr/bin/env python3
"""图表28: 蜂窝阵列双层线圈方案示意图 (v2: 束绕圆环)
线圈 = 15股 Ø0.05mm 漆包线捆成一束, 绕成 Ø5mm 圆环 (束径~0.22mm, 环面)
28a: 结构 3D 示意图 (中心单元 + 6 邻居, pitch 5.2mm)
28b: 联动变形模式图 (中心点压, 邻居联动 alpha=0.3)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------------- 参数 ----------------
D_COIL = 5.0        # 线圈(环)中径 mm
R_COIL = D_COIL / 2
N_TURNS = 15
WIRE_D = 0.05       # 漆包线径 mm
BUNDLE_D = 0.22     # 15股捆成圆束的束径 mm (sqrt(15)*0.05/sqrt(0.78))
COIL_H = BUNDLE_D   # 环的"高度"就是束径
PITCH = 5.2         # 蜂窝中心距 mm
GAP_REST = 2.5      # 静息法向间距 mm (环心到环心)
GAP_MIN = 1.0
GAP_NOM = 1.75

# 蜂窝 7 单元中心 (中心 + 6 邻居)
def hex_centers(pitch):
    cs = [(0.0, 0.0)]
    for k in range(6):
        a = np.pi / 3 * k
        cs.append((pitch * np.cos(a), pitch * np.sin(a)))
    return cs

CENTERS = hex_centers(PITCH)

# ---------------- 几何辅助 ----------------
def coil_pts(cx, cy, z0, r=R_COIL, tilt_ax=0.0, tilt_ay=0.0, npt=200):
    """束绕圆环中心线 (单圆); tilt_ax/ay: 绕 x/y 轴倾角(rad), 绕环心旋转"""
    t = np.linspace(0, 2 * np.pi, npt)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.zeros_like(t)
    if tilt_ax or tilt_ay:
        ca, sa = np.cos(tilt_ax), np.sin(tilt_ax)
        y, z = y * ca - z * sa, y * sa + z * ca
        cb, sb = np.cos(tilt_ay), np.sin(tilt_ay)
        x, z = x * cb + z * sb, -x * sb + z * cb
    return x + cx, y + cy, z + z0

def spring_pts(cx, cy, z_bot, z_top, r=R_COIL, nturn=4, npt=400,
               tilt_ay=0.0, shear_x=0.0):
    """弹簧示意 (稀疏螺旋), 底固定、顶可平移/倾斜"""
    t = np.linspace(0, 2 * np.pi * nturn, npt)
    u = t / (2 * np.pi * nturn)          # 0->1
    x = r * np.cos(t) + cx + shear_x * (3 * u**2 - 2 * u**3)
    y = r * np.sin(t) + cy
    z = z_bot + (z_top - z_bot) * u
    return x, y, z

# ================================================================
# 图 28a: 结构 3D 示意
# ================================================================
fig = plt.figure(figsize=(16, 8), facecolor='white')

ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('white')

RING_LW = 4.5   # 粗线表现 0.22mm 束径圆环
for i, (cx, cy) in enumerate(CENTERS):
    is_center = (i == 0)
    # 内层(固定)环 环心 z=0
    xb, yb, zb = coil_pts(cx, cy, 0.0)
    ax.plot(xb, yb, zb, color='#1f77b4', lw=RING_LW, alpha=0.9,
            solid_capstyle='round')
    # 表层环 环心 z=GAP_REST
    xt, yt, zt = coil_pts(cx, cy, GAP_REST)
    ax.plot(xt, yt, zt, color='#d62728' if is_center else '#ff9896',
            lw=RING_LW, solid_capstyle='round')
    # 弹簧
    xs, ys, zs = spring_pts(cx, cy, 0.0, GAP_REST)
    ax.plot(xs, ys, zs, color='#7f7f7f', lw=0.8, alpha=0.55)

# 驱动/接收箭头: 中心表层 -> 自己内层 + 6 邻居
ax.scatter([0], [0], [GAP_REST + 0.5], color='#d62728', s=40, marker='v')
ax.text(0, 0, GAP_REST + 1.1, '驱动: 中心表层线圈', color='#d62728',
        fontsize=11, ha='center')
for (cx, cy) in CENTERS[1:]:
    ax.plot([0, cx], [0, cy], [GAP_REST, GAP_REST],
            color='#2ca02c', lw=0.8, ls='--', alpha=0.7)
ax.plot([0, 0], [0, 0], [GAP_REST, 0], color='#2ca02c', lw=1.4, ls='--')

ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
ax.set_title('28a  蜂窝阵列结构 (静息态, 间距 2.5mm)\n'
             '线圈=15股Ø0.05mm漆包线捆束绕成的粗圆环 (束径约0.22mm)\n'
             '红=表层环(随弹簧动)  蓝=内层环(固定锚点)  灰=弹簧(漆包线,开路)',
             fontsize=12)
ax.set_box_aspect((1, 1, 0.45))
ax.view_init(elev=22, azim=-60)
ax.set_zlim(-1, 5)

# 俯视布局小图
ax2 = fig.add_subplot(122)
ax2.set_facecolor('white')
for i, (cx, cy) in enumerate(CENTERS):
    is_center = (i == 0)
    circ = plt.Circle((cx, cy), R_COIL, fill=False,
                      color='#d62728' if is_center else '#1f77b4',
                      lw=2 if is_center else 1.4)
    ax2.add_patch(circ)
    ax2.text(cx, cy, '驱' if is_center else f'收{i}', ha='center', va='center', fontsize=13,
             color='#d62728' if is_center else '#1f77b4')
# pitch 标注
ax2.annotate('', xy=(PITCH, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
ax2.text(PITCH / 2, 0.4, f'pitch = {PITCH} mm', ha='center', fontsize=12)
ax2.annotate('', xy=(R_COIL + 0.2 + np.cos(np.pi/3)*0, PITCH*np.sin(np.pi/3) - R_COIL),
             xytext=(0, R_COIL),
             arrowprops=dict(arrowstyle='-', color='none'))
ax2.text(0, -PITCH - 3.6,
         f'线圈: 中径 Ø{D_COIL}mm 粗圆环 = {N_TURNS}股 x Ø{WIRE_D}mm 漆包线捆束\n'
         f'束径约 {BUNDLE_D}mm (环面, 无叠绕高度), L 约 2.3µH, R_DC 约 2.0Ω\n'
         f'相邻净间隙 = {PITCH - D_COIL:.1f} mm\n'
         f'环心法向间距: 静息 {GAP_REST}mm, 全压缩 {GAP_MIN}mm, 标称 {GAP_NOM}mm\n'
         '每线圈独立接 S参数检测 + 正弦驱动\n'
         '观测量 = 互感 M: 表层0 到 {自己内层, 6邻居表层, 6邻居内层} 共13个',
         ha='center', va='top', fontsize=12,
         bbox=dict(boxstyle='round', fc='#f5f5f0', ec='#999'))
ax2.set_xlim(-9.5, 9.5); ax2.set_ylim(-15.5, 9.5)
ax2.set_aspect('equal')
ax2.set_title('俯视布局 (蜂窝六邻域)', fontsize=13)
ax2.set_xlabel('x (mm)'); ax2.set_ylabel('y (mm)')

plt.tight_layout()
plt.savefig('/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/图表28a_蜂窝结构示意.png',
            dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)

# ================================================================
# 图 28b: 联动变形模式
# ================================================================
ALPHA = 0.3          # 邻居联动比例
DELTA0 = 0.75        # 中心下压量 mm (静息2.5 -> 1.75 标称)
TILT_N = np.deg2rad(6.0)   # 邻居向心倾角 (示意值, 与 alpha*delta0/pitch 同量级)

fig = plt.figure(figsize=(16, 8), facecolor='white')
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('white')

poses = []   # (cx, cy, dz, tilt_ax, tilt_ay)
for i, (cx, cy) in enumerate(CENTERS):
    if i == 0:
        dz, tax, tay = -DELTA0, 0.0, 0.0
    else:
        dz = -ALPHA * DELTA0
        # 向心倾斜: 顶面法向偏向中心. 邻居方位角 phi
        phi = np.arctan2(cy, cx)
        # 绕垂直于径向的水平轴倾斜 TILT_N: 分解到 x/y 倾角
        tax = TILT_N * np.sin(phi)      # 绕x轴
        tay = -TILT_N * np.cos(phi)     # 绕y轴
    poses.append((cx, cy, dz, tax, tay))

RING_LW = 4.5
for i, (cx, cy, dz, tax, tay) in enumerate(poses):
    is_center = (i == 0)
    xb, yb, zb = coil_pts(cx, cy, 0.0)
    ax.plot(xb, yb, zb, color='#1f77b4', lw=RING_LW, alpha=0.9)
    # 静息位置 (虚影)
    xg, yg, zg = coil_pts(cx, cy, GAP_REST)
    ax.plot(xg, yg, zg, color='#bbbbbb', lw=1.2, alpha=0.5)
    # 变形后表层
    xt, yt, zt = coil_pts(cx, cy, GAP_REST + dz, tilt_ax=tax, tilt_ay=tay)
    ax.plot(xt, yt, zt, color='#d62728' if is_center else '#ff7f0e', lw=RING_LW)
    xs, ys, zs = spring_pts(cx, cy, 0.0, GAP_REST + dz)
    ax.plot(xs, ys, zs, color='#7f7f7f', lw=0.8, alpha=0.5)

# 力箭头
ax.quiver(0, 0, GAP_REST + 2.2, 0, 0, -1.4, color='k', lw=2, arrow_length_ratio=0.25)
ax.text(0.4, 0, GAP_REST + 2.4, 'F (点压)', fontsize=13)

ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
ax.set_title(f'28b  联动变形模式 (中心点压)\n'
             f'中心下压 $\\delta_0$={DELTA0}mm, 邻居联动 $\\alpha\\delta_0$'
             f'={ALPHA*DELTA0:.2f}mm + 向心倾斜 {np.rad2deg(TILT_N):.0f}°',
             fontsize=12)
ax.set_box_aspect((1, 1, 0.45))
ax.view_init(elev=18, azim=-60)
ax.set_zlim(-1, 5)

# 剖面图 (y=0 截面: 单元 4(-x), 0, 1(+x))
ax2 = fig.add_subplot(122)
ax2.set_facecolor('white')
sel = [(CENTERS[4][0], poses[4]), (0.0, poses[0]), (CENTERS[1][0], poses[1])]
labels = ['邻居 (x<0)', '中心', '邻居 (x>0)']
for (cx, (px, py, dz, tax, tay)), lab in zip(sel, labels):
    # 内层环截面: 两个束截面圆点 + 环面连线
    ax2.plot([cx - R_COIL, cx + R_COIL], [0, 0], color='#1f77b4', lw=1.2, ls='-')
    ax2.plot([cx - R_COIL, cx + R_COIL], [0, 0], 'o', color='#1f77b4', ms=9)
    # 静息表层 (虚)
    ax2.plot([cx - R_COIL, cx + R_COIL], [GAP_REST, GAP_REST], color='#bbbbbb',
             lw=1.2, ls='--')
    ax2.plot([cx - R_COIL, cx + R_COIL], [GAP_REST, GAP_REST], 'o',
             color='#bbbbbb', ms=9, mfc='none')
    # 变形后表层: 倾斜的两个束截面圆点
    zc = GAP_REST + dz
    dz_edge = R_COIL * np.tan(tay) if abs(tay) > 1e-9 else 0.0
    col = '#d62728' if cx == 0 else '#ff7f0e'
    ax2.plot([cx - R_COIL, cx + R_COIL], [zc + dz_edge, zc - dz_edge],
             color=col, lw=1.2)
    ax2.plot([cx - R_COIL, cx + R_COIL], [zc + dz_edge, zc - dz_edge], 'o',
             color=col, ms=9)
    ax2.annotate('', xy=(cx + R_COIL + 0.15, zc), xytext=(cx + R_COIL + 0.15, GAP_REST),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1))
    ax2.text(cx, -0.9, lab, ha='center', fontsize=12)
    ax2.text(cx + R_COIL + 0.25, (zc + GAP_REST) / 2, f'{dz:+.2f}', fontsize=10, color='#555')

# 弹簧示意 (竖线)
for cx, (px, py, dz, tax, tay) in sel:
    for xx in (cx - R_COIL * 0.6, cx + R_COIL * 0.6):
        ax2.plot([xx, xx], [0.05, GAP_REST + dz - 0.05], color='#999', lw=1, ls=':')

ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlim(-9.5, 9.5); ax2.set_ylim(-1.6, 4.2)
ax2.set_xlabel('x (mm)'); ax2.set_ylabel('z (mm)')
ax2.set_title('y=0 剖面: 圆点=束截面(Ø0.22mm, 放大画)  蓝=内层(固定)\n'
              f'灰虚=静息表层  红/橙=变形后表层; 中心 dz={-DELTA0:.2f}mm, '
              f'邻居 dz={-ALPHA*DELTA0:.2f}mm + 向心倾斜', fontsize=11)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/图表28b_联动变形模式.png',
            dpi=110, facecolor='white', bbox_inches='tight')
plt.close(fig)

# 尺寸检查
from PIL import Image
for f in ['图表28a_蜂窝结构示意.png', '图表28b_联动变形模式.png']:
    p = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/' + f
    im = Image.open(p)
    if im.width > 2000:
        r = 2000 / im.width
        im = im.resize((2000, int(im.height * r)), Image.LANCZOS)
        im.save(p)
    print(f, Image.open(p).size)
