#!/usr/bin/env python3
"""E1 专利候选附图 (按 v0.5 §十七 编号: 图1 阵列轴测, 图2 共享关系, 图3 剖视, 图4 两种读取电路, 图5 网络, 图6 磁+电容, 图7 流程).
黑线白底, 图内数字标号, 中文只在标号表; 图1/2/3 以 patent_E1_cad.py 的 SVG→PNG 为底并按其 json 变换精确叠加.
用项目 venv 跑: python scripts/patent_E1_figs.py [1 2 3 4 5 6 7 md]"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc, Ellipse
from PIL import Image

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'reports', 'patent_E1')
LAYOUT = json.load(open(os.path.join(ROOT, 'host', 'honeycomb_host', 'data', 'layout.json')))
XY = np.array([[u['x_mm'], u['y_mm']] for u in LAYOUT]); SUNK = np.array([u['sunk'] for u in LAYOUT])
PITCH = 5.2
Z_NOM, Z_SUNK, GAP, PCB_T, T_RING, R_ISL, T_ISL = 0.0, -0.72, 2.53, 1.0, 0.2, 3.1, 0.35
LW = 1.1

def unit_at(x, y):
    return int(np.argmin(np.hypot(XY[:, 0] - x, XY[:, 1] - y)))

class CadView:
    def __init__(self, name):
        self.meta = json.load(open(os.path.join(OUT, f'{name}.json')))
        self.img = Image.open(os.path.join(OUT, f'{name}.png')).convert('L')
        self.W, self.H = self.img.size
        self.R = np.array(self.meta['R'])
    def px(self, p):
        p = np.array(p, float); p[2] *= self.meta['z_exag']
        X, Y, _ = self.R @ p
        m = self.meta
        return (m['scale'] * (X + m['tx'])) * self.W / m['width'], (-m['scale'] * (Y + m['ty'])) * self.H / m['height']
    def fig(self, pts=None, pad=(120, 120, 120, 120), figsize=(16, 10)):
        """pts: 世界坐标点列表 → 自动裁剪范围 (含 pad 像素: 左右上下)."""
        fig, ax = plt.subplots(figsize=figsize, dpi=110)
        ax.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        if pts is not None:
            P = np.array([self.px(p) for p in pts])
            ax.set_xlim(P[:, 0].min() - pad[0], P[:, 0].max() + pad[1]); ax.set_ylim(P[:, 1].max() + pad[3], P[:, 1].min() - pad[2])
        ax.axis('off'); fig.patch.set_facecolor('white')
        return fig, ax

def leader(ax, p_px, txt, off=(60, -60), fs=15, circle=True):
    x, y = p_px; tx, ty = x + off[0], y + off[1]
    ax.annotate('', xy=(x, y), xytext=(tx, ty), arrowprops=dict(arrowstyle='-', lw=LW, color='k', shrinkA=13, shrinkB=1))
    ax.text(tx, ty, txt, fontsize=fs, ha='center', va='center', color='k',
            bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec='k', lw=0.9) if circle else dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

def dim(ax, p0, p1, txt, tick=8, fs=13, offset_txt=(0, 0)):
    (x0, y0), (x1, y1) = p0, p1
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='<->', lw=LW, color='k', shrinkA=0, shrinkB=0))
    d = np.array([x1 - x0, y1 - y0]); n = np.array([-d[1], d[0]]); n = n / (np.linalg.norm(n) + 1e-9)
    for (x, y) in ((x0, y0), (x1, y1)):
        ax.plot([x - n[0] * tick, x + n[0] * tick], [y - n[1] * tick, y + n[1] * tick], 'k', lw=LW)
    ax.text((x0 + x1) / 2 + offset_txt[0], (y0 + y1) / 2 + offset_txt[1], txt, fontsize=fs, ha='center', va='center', bbox=dict(fc='white', ec='none', pad=1))

def savefig(fig, name):
    path = os.path.join(OUT, name); fig.savefig(path, dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    im = Image.open(path)
    if im.width > 2000:
        im = im.resize((2000, int(im.height * 2000 / im.width)), Image.LANCZOS); im.save(path)
    print('saved', name)

def caption(ax, num, note=''):
    ax.text(0.02, 0.02, f'图{num}', transform=ax.transAxes, fontsize=18, ha='left', va='bottom')
    if note:
        ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=11, ha='right', va='bottom', color='0.3')

# ---------------- 图1: 阵列轴测 ----------------
def fig1():
    v = CadView('cad_fig8_array_iso')
    pts = [(-20, -18, Z_SUNK - 0.9), (20, -18, Z_SUNK - 0.9), (-20, 18, Z_SUNK - 0.9), (20, 18, Z_SUNK - 0.9), (0, 0, GAP + 1.0), (0, 9, GAP + 1.0)]
    fig, ax = v.fig(pts, pad=(60, 120, 80, 140), figsize=(16, 11))
    leader(ax, v.px((-20, -18, Z_SUNK - 0.9)), '1', off=(-60, 70))
    front = sorted(range(19), key=lambda i: XY[i, 1])[:3]           # 前排三个单元 (y 最小)
    u_nom = [i for i in front if not SUNK[i]][0]; u_sunk = [i for i in front if SUNK[i]][0]
    leader(ax, v.px((XY[u_nom, 0] - 1.6, XY[u_nom, 1] - 1.6, Z_NOM)), '2a', off=(-90, 100))
    leader(ax, v.px((XY[u_sunk, 0] + 1.8, XY[u_sunk, 1] - 1.4, Z_SUNK)), '2b', off=(110, 100))
    u_r = front[1]
    leader(ax, v.px((XY[u_r, 0] + 2.3, XY[u_r, 1], GAP + T_RING)), '3', off=(150, 20))
    leader(ax, v.px((XY[u_r, 0] + R_ISL, XY[u_r, 1], GAP - T_ISL / 2)), '4', off=(150, 70))
    leader(ax, v.px((XY[u_r, 0] + (R_ISL - 0.6) * np.cos(np.deg2rad(330)) + 0.45, XY[u_r, 1] + (R_ISL - 0.6) * np.sin(np.deg2rad(330)), GAP / 2)), '5', off=(150, 120))
    c = unit_at(0, 0); n = unit_at(2.6, 4.503)
    leader(ax, v.px((XY[c, 0], XY[c, 1] - 2.3, GAP - 0.6 + T_RING)), '12(P1)', off=(-170, 120), circle=False)
    leader(ax, v.px((XY[n, 0] + 0.4 + 2.3, XY[n, 1], GAP - 0.3 + T_RING)), '12(P2)', off=(180, 30), circle=False)
    caption(ax, 1, '(z 方向放大, 不按比例)')
    savefig(fig, '图1_阵列总体轴测.png')

# ---------------- 图2: 局部共享关系 ----------------
def fig2():
    v = CadView('cad_fig8b_subset_iso')
    S0, S1, S2 = unit_at(0, 0), unit_at(2.6, 4.503), unit_at(-2.6, 4.503)
    pts = [(-8, -4, Z_SUNK - 0.9), (8, -4, Z_SUNK - 0.9), (-8, 8, Z_SUNK - 0.9), (8, 8, Z_SUNK - 0.9), (XY[S1, 0], XY[S1, 1], GAP + 0.8), (XY[S2, 0], XY[S2, 1], GAP + 0.8)]
    fig, ax = v.fig(pts, pad=(160, 160, 120, 140), figsize=(15, 11))
    zc = {i: (Z_SUNK if SUNK[i] else Z_NOM) for i in (S0, S1, S2)}
    leader(ax, v.px((XY[S0, 0] + 2.2, XY[S0, 1] - 0.8, zc[S0])), 'S0 (2)', off=(150, 90), circle=False)
    leader(ax, v.px((XY[S1, 0] + 2.4, XY[S1, 1] + 0.3, zc[S1])), 'S1 (2)', off=(170, 30), circle=False)
    leader(ax, v.px((XY[S2, 0] - 2.4, XY[S2, 1] - 0.5, zc[S2])), 'S2 (2)', off=(-160, 70), circle=False)
    p1 = np.array([XY[S1, 0], XY[S1, 1], GAP - 0.4 + T_RING]); p2 = np.array([XY[S2, 0] + 0.3, XY[S2, 1], GAP + T_RING])
    leader(ax, v.px((p1[0] + 2.3, p1[1], p1[2])), 'P1 (3)', off=(170, -50), circle=False)
    leader(ax, v.px((p2[0] - 2.3, p2[1] + 0.3, p2[2])), 'P2 (3)', off=(-170, -50), circle=False)
    leader(ax, v.px((p2[0] - R_ISL, p2[1], p2[2] - T_RING - T_ISL / 2)), '4', off=(-140, 40))
    leader(ax, v.px((-7.5, -3.5, Z_SUNK - 0.9)), '1', off=(-70, 60))
    def path(a, b, txt, rad, txt_off=(0, 0)):
        pa, pb = v.px(a), v.px(b)
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle=f'arc3,rad={rad}', arrowstyle='<->', mutation_scale=16, lw=1.6, color='k', ls='--'))
        ax.text((pa[0] + pb[0]) / 2 + txt_off[0], (pa[1] + pb[1]) / 2 + txt_off[1], txt, fontsize=13, ha='center', va='center', bbox=dict(fc='white', ec='none', pad=1))
    c0 = (XY[S0, 0], XY[S0, 1], zc[S0] + 0.1); c1 = (XY[S1, 0], XY[S1, 1], zc[S1] + 0.1); c2 = (XY[S2, 0], XY[S2, 1], zc[S2] + 0.1)
    path(c0, p1, r'$M_{S0P1}$', -0.35, (50, 30))
    path(c1, p1, r'$M_{S1P1}$', 0.0, (70, 0))
    path(c0, p2, r'$M_{S0P2}$', 0.35, (-50, 30))
    path(c2, p2, r'$M_{S2P2}$', 0.0, (-70, 0))
    path(p1, p2, r'$M_{P1P2}$', 0.3, (0, -50))
    ax.text(0.5, 0.03, '观测组 1 (P1): {S0–P1, S1–P1}    观测组 2 (P2): {S0–P2, S2–P2}    S0 为两组实际共享的固定线圈; P1–P2 环间耦合计入网络模型',
            transform=ax.transAxes, fontsize=12, ha='center', va='bottom')
    caption(ax, 2)
    savefig(fig, '图2_局部共享关系.png')

# ---------------- 图3: 剖视 ----------------
def fig3():
    v = CadView('cad_fig9_section')
    cs, cn = unit_at(0, 0), unit_at(PITCH, 0)
    xl, xr = XY[cs, 0], XY[cn, 0]; y = 0.0
    pts = [(xl - 8, y, -PCB_T), (xr + 8, y, -PCB_T), (xl, y, GAP + 0.9), (xr, y, GAP + 0.9)]
    fig, ax = v.fig(pts, pad=(170, 200, 120, 100), figsize=(17, 9))
    leader(ax, v.px((xr + 6.0, y, -PCB_T * 0.55)), '1', off=(120, 40))
    leader(ax, v.px((xr + 1.2, y, Z_NOM - 0.03)), '2a', off=(150, 110))
    leader(ax, v.px((xl - 1.2, y, Z_SUNK - 0.03)), '2b', off=(-150, 110))
    leader(ax, v.px((xl - 2.0, y, GAP + T_RING / 2)), '3', off=(-120, -90))
    leader(ax, v.px((xl, y, GAP + T_RING / 2)), '11', off=(40, -110))
    leader(ax, v.px((xl + 2.6, y, GAP - T_ISL / 2)), '4', off=(110, -70))
    leader(ax, v.px((xl - (R_ISL - 0.6) - 0.45, y, GAP / 2)), '5', off=(-110, 0))
    for zl, name in ((-0.12, 'L1/L2'), (-0.84, 'L3/L4')):
        px = v.px((xl - 8, y, zl)); ax.text(px[0] - 10, px[1], name, fontsize=11, ha='right', va='center')
    # 空气间隙 g: L1 顶 → 环底 (左单元, 静息)
    xg = xl + 3.9
    dim(ax, v.px((xg, y, 0.0)), v.px((xg, y, GAP - T_ISL)), 'g', offset_txt=(24, 0))
    ax.text(*v.px((xl + 0.0, y, 1.2)), '空气间隙', fontsize=11, ha='center', va='center', color='0.3')
    xh = xl - 5.5
    dim(ax, v.px((xh, y, Z_SUNK)), v.px((xh, y, Z_NOM)), 'h', offset_txt=(-24, 0))
    # 右单元: 位移与倾斜示意
    pr = v.px((xr - 2.0, y, GAP + T_RING + 0.35))
    ax.annotate('', xy=(pr[0], pr[1] + 60), xytext=(pr[0], pr[1] - 30), arrowprops=dict(arrowstyle='<->', lw=1.6, color='k'))
    ax.text(pr[0] + 16, pr[1] + 15, r'$\Delta z$', fontsize=14, va='center')
    pt = v.px((xr + 1.0, y, GAP + T_RING + 0.55))
    ax.add_patch(Arc((pt[0], pt[1] + 30), 160, 70, theta1=195, theta2=345, lw=1.4, color='k'))
    ax.annotate('', xy=(pt[0] + 80, pt[1] + 24), xytext=(pt[0] + 74, pt[1] + 10), arrowprops=dict(arrowstyle='->', lw=1.4, color='k'))
    ax.text(pt[0], pt[1] - 22, r'$\theta$', fontsize=14, ha='center')
    caption(ax, 3, '(沿 y=0 剖切; z 方向放大 2×, 不按比例; 右单元示意位移与倾斜)')
    savefig(fig, '图3_结构剖视.png')

# ---------------- 电路原语 ----------------
def wire(ax, pts, lw=LW, ls='-'):
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color='k', lw=lw, ls=ls, solid_capstyle='round')
def box(ax, x, y, w, h, text, fs=11):
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec='k', lw=LW)); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
def coil_v(ax, x, y, n=4, r=0.3):
    for i in range(n):
        ax.add_patch(Arc((x, y + (2 * i + 1) * r), 2 * r, 2 * r, theta1=-90, theta2=90, lw=LW, color='k'))
    return y + 2 * n * r
def ring_sym(ax, x, y, w=1.3, h=0.42):
    ax.add_patch(Ellipse((x, y), w, h, fc='white', ec='k', lw=1.6)); ax.add_patch(Ellipse((x, y), w * 0.55, h * 0.55, fc='white', ec='k', lw=1.0))
def meter(ax, x, y, txt, r=0.28):
    ax.add_patch(Circle((x, y), r, fc='white', ec='k', lw=LW)); ax.text(x, y, txt, ha='center', va='center', fontsize=10)
def dot(ax, x, y): ax.add_patch(Circle((x, y), 0.05, fc='k'))
def arrow(ax, p0, p1, ls='-', lw=1.2, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color='k', linestyle=ls))

# ---------------- 图4: 两种读取电路 ----------------
def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5), dpi=110)
    for ax in axs:
        ax.set_xlim(0, 9); ax.set_ylim(0, 7); ax.axis('off')
    # (a) 自端口反射: 激励 7 → 电流表 → 线圈; 电压表并联在线圈两端
    ax = axs[0]
    box(ax, 0.4, 2.6, 1.3, 1.0, '7\n激励', fs=11)
    yb, yt = 1.6, 4.2
    wire(ax, [(1.7, 3.6), (2.6, 3.6), (2.6, yt), (3.6, yt)]); meter(ax, 3.9, yt, 'I'); wire(ax, [(4.2, yt), (4.6, yt)])
    wire(ax, [(1.7, 2.6), (2.6, 2.6), (2.6, yb), (4.6, yb)])
    top = coil_v(ax, 4.6, yb, n=4, r=0.325); dot(ax, 4.6, yt); dot(ax, 4.6, yb)
    # 电压表跨端
    wire(ax, [(4.6, yt), (5.6, yt), (5.6, 3.2)]); wire(ax, [(4.6, yb), (5.6, yb), (5.6, 2.6)]); meter(ax, 5.6, 2.9, 'V')
    ax.text(4.9, 2.9 + 0.55, r'$S_i$ (2)', fontsize=12)
    ax.text(2.1, 0.9, r'$Z_i=R_i+j\omega L_i$  (固定线圈自阻抗)', fontsize=11)
    # 环
    ring_sym(ax, 4.6, 5.6); ax.text(5.5, 5.6, 'P (3)', fontsize=12)
    ax.annotate('', xy=(4.6, 5.3), xytext=(4.6, yt + 0.1), arrowprops=dict(arrowstyle='<->', lw=1.2, color='k', ls='--')); ax.text(4.75, 4.85, r'$M_{iP}$', fontsize=12)
    ax.text(3.0, 6.5, r'$I_P=-\,j\omega M_{Pi}\,I_i/Z_P$,   $Z_P=R_P+j\omega L_P$', fontsize=12)
    ax.text(6.3, 5.1, r'$Z_{\mathrm{in},i}=\dfrac{V}{I}=Z_i+\dfrac{\omega^2 M_{iP}M_{Pi}}{Z_P}$', fontsize=13)
    ax.text(6.3, 4.2, '6 端口选通; 8 接收 (V、I 相干检测)', fontsize=10.5)
    box(ax, 6.4, 1.5, 2.3, 1.2, '9 处理电路\n由 Z_in 变化反演位姿', fs=10.5)
    ax.text(0.3, 0.3, '(a) 固定线圈自端口读取 (§7.1, 权8)', fontsize=13)
    # (b) TX-环-RX
    ax = axs[1]
    box(ax, 0.3, 1.5, 1.2, 1.0, '7\n激励', fs=11)
    wire(ax, [(1.5, 2.3), (2.4, 2.3), (2.4, 3.6)]); wire(ax, [(1.5, 1.7), (2.4, 1.7), (2.4, 0.8)])
    coil_v(ax, 2.4, 0.8, n=4, r=0.35); dot(ax, 2.4, 3.6); dot(ax, 2.4, 0.8)
    ax.text(1.5, 3.9, r'$S_i$ (TX, 2)', fontsize=12); ax.text(1.9, 2.55, r'$I_i$', fontsize=12)
    coil_v(ax, 6.2, 0.8, n=4, r=0.35); dot(ax, 6.2, 3.6); dot(ax, 6.2, 0.8)
    wire(ax, [(6.2, 3.6), (7.6, 3.6), (7.6, 2.9)]); wire(ax, [(6.2, 0.8), (7.6, 0.8), (7.6, 1.5)])
    box(ax, 7.0, 1.5, 1.4, 1.4, '8\n接收\n(高阻)', fs=10); ax.text(5.3, 3.9, r'$S_j$ (RX, 2)', fontsize=12); ax.text(7.75, 3.1, r'$V_j$', fontsize=12)
    ring_sym(ax, 4.3, 5.4, w=1.6, h=0.5); ax.text(5.3, 5.4, 'P (3)', fontsize=12)
    ax.annotate('', xy=(3.8, 5.1), xytext=(2.6, 3.7), arrowprops=dict(arrowstyle='<->', lw=1.2, color='k', ls='--')); ax.text(2.6, 4.6, r'$M_{Pi}$', fontsize=12)
    ax.annotate('', xy=(4.8, 5.1), xytext=(6.0, 3.7), arrowprops=dict(arrowstyle='<->', lw=1.2, color='k', ls='--')); ax.text(5.6, 4.6, r'$M_{jP}$', fontsize=12)
    ax.annotate('', xy=(5.8, 2.2), xytext=(2.8, 2.2), arrowprops=dict(arrowstyle='<->', lw=1.2, color='k', ls=':')); ax.text(4.3, 2.4, r'$M_{ji}$ (直接耦合, 基线)', fontsize=11, ha='center')
    ax.text(0.4, 6.4, r'$\dfrac{V_j}{I_i}=j\omega M_{ji}+\dfrac{\omega^2 M_{jP}M_{Pi}}{Z_P}$', fontsize=13)
    ax.text(0.4, 5.6, '环 P 只作无源中继, 不设接收引线', fontsize=11)
    ax.text(0.3, 0.3, '(b) 固定 TX–无源环–固定 RX 传递读取 (§7.2, 权9)', fontsize=13)
    fig.text(0.02, 0.01, '图4', fontsize=18)
    savefig(fig, '图4_两种磁读取电路.png')

# ---------------- 图5: 网络 ----------------
def fig5():
    fig, ax = plt.subplots(figsize=(17, 8), dpi=110); ax.set_xlim(0, 17); ax.set_ylim(0, 8); ax.axis('off')
    xs = [1.5, 3.7, 5.9]
    for k, x in enumerate(xs):
        coil_v(ax, x, 1.4, n=3, r=0.3); dot(ax, x, 1.4); dot(ax, x, 3.2)
        wire(ax, [(x, 3.2), (x, 3.5)]); wire(ax, [(x, 1.4), (x, 0.9)])
        ax.add_patch(Circle((x, 3.75), 0.22, fc='white', ec='k', lw=LW)); ax.add_patch(Circle((x, 0.65), 0.22, fc='white', ec='k', lw=LW))
        ax.text(x + 0.3, 3.75, f'F{k+1}', fontsize=10, va='center'); ax.text(x + 0.3, 0.65, "F{}'".format(k + 1), fontsize=10, va='center')
        ax.text(x - 0.55, 1.05, f'$S_{k}$', fontsize=12, ha='right')
        ring_sym(ax, x + 0.3, 5.9, w=1.4, h=0.45); ax.text(x - 0.5, 6.5, f'$P_{k+1}$ (3)', fontsize=11)
        ax.annotate('', xy=(x + 0.3, 5.6), xytext=(x, 4.0), arrowprops=dict(arrowstyle='<->', lw=1.0, color='k', ls='--'))
    for (a, b) in ((0, 1), (1, 0), (1, 2), (2, 1)):
        ax.annotate('', xy=(xs[b] + 0.3, 5.55), xytext=(xs[a], 4.0), arrowprops=dict(arrowstyle='<->', lw=0.9, color='k', ls='--'))
    for k in range(2):
        ax.add_patch(FancyArrowPatch((xs[k] + 1.0, 5.9), (xs[k + 1] - 0.4, 5.9), arrowstyle='<->', mutation_scale=12, lw=1.0, color='k', ls='--'))
        ax.text((xs[k] + xs[k + 1]) / 2 + 0.3, 6.1, r'$M_{P%dP%d}$' % (k + 1, k + 2), fontsize=10, ha='center')
        ax.add_patch(FancyArrowPatch((xs[k] + 0.45, 2.3), (xs[k + 1] - 0.45, 2.3), arrowstyle='<->', mutation_scale=12, lw=1.0, color='k', ls=':'))
        ax.text((xs[k] + xs[k + 1]) / 2, 2.5, r'$M_{%d%d}$' % (k, k + 1), fontsize=10, ha='center')
    ax.add_patch(FancyArrowPatch((xs[0] + 0.3, 6.3), (xs[2] + 0.3, 6.3), arrowstyle='<->', mutation_scale=12, lw=1.0, color='k', ls='--', connectionstyle='arc3,rad=-0.25'))
    ax.text(xs[1] + 0.3, 7.15, r'$M_{P1P3}$', fontsize=10, ha='center')
    ax.text(0.4, 7.6, 'F: 固定线圈回路 (端口可访问, 由 6 选通)      P: 无源单匝环回路 (回路电压恒为 0, 无引线)', fontsize=12)
    ax.text(7.2, 5.9, '环回路: 闭合\n(V_P = 0)', fontsize=10, va='center')
    def bracket(x0, y0, h, w=0.12, left=True):
        sgn = 1 if left else -1
        wire(ax, [(x0 + sgn * w, y0), (x0, y0), (x0, y0 + h), (x0 + sgn * w, y0 + h)], lw=1.3)
    def matrix(x, y, rows, colw, rowh, fs=14):
        n, m = len(rows), len(rows[0])
        bracket(x, y, n * rowh, left=True); bracket(x + m * colw, y, n * rowh, left=False)
        for i, r in enumerate(rows):
            for j, t in enumerate(r):
                ax.text(x + (j + 0.5) * colw, y + (n - i - 0.5) * rowh, t, fontsize=fs, ha='center', va='center')
        return x + m * colw
    xe = matrix(8.8, 5.7, [[r'$\mathbf{V}_F$'], [r'$\mathbf{0}$']], 0.9, 0.7)
    ax.text(xe + 0.15, 6.4, '=', fontsize=16, va='center')
    xe = matrix(xe + 0.45, 5.7, [[r'$\mathbf{Z}_{FF}$', r'$\mathbf{Z}_{FP}$'], [r'$\mathbf{Z}_{PF}$', r'$\mathbf{Z}_{PP}$']], 1.1, 0.7)
    xe = matrix(xe + 0.15, 5.7, [[r'$\mathbf{I}_F$'], [r'$\mathbf{I}_P$']], 0.9, 0.7)
    ax.text(8.8, 4.7, r'$\mathbf{Z}_{\mathrm{eff}}=\mathbf{Z}_{FF}-\mathbf{Z}_{FP}\,\mathbf{Z}_{PP}^{-1}\,\mathbf{Z}_{PF}$', fontsize=18, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='k', lw=1.2))
    ax.text(8.8, 3.7, r'$\mathbf{Z}_{FF}$: 固定线圈自阻抗与直接耦合 $j\omega M_{ij}$ (刚性几何, 标定基线)', fontsize=11)
    ax.text(8.8, 3.2, r'$\mathbf{Z}_{FP}=\mathbf{Z}_{PF}^{T}$: 固定线圈–环耦合 $j\omega M_{iP}(\mathbf{q})$, 随位姿变化', fontsize=11)
    ax.text(8.8, 2.7, r'$\mathbf{Z}_{PP}$: 对角 $R_P+j\omega L_P$ (损耗、温度); 非对角 $j\omega M_{PP^\prime}$ (环间互感, 不可略)', fontsize=11)
    ax.text(8.8, 2.1, r'测量接口 6 取得 $\mathbf{Z}_{\mathrm{eff}}$ 的自响应 (对角) 与传递响应 (非对角);', fontsize=11)
    ax.text(8.8, 1.7, r'非访问端口的端接 (开路/已知负载) 并入 $\mathbf{Z}_{FF}$ 后再求端口响应', fontsize=11)
    ax.text(8.8, 1.1, r'读出的是网络投影, 不是逐环 $M_{iP}$ 直读 → 多环联合估计 (图7)', fontsize=11)
    fig.text(0.02, 0.01, '图5', fontsize=18)
    savefig(fig, '图5_多固定线圈多无源环网络.png')

# ---------------- 图6: 电磁 + 电容 (固定侧 D / E+ / E−, 浮置环) ----------------
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5), dpi=110, gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    # 基座与固定侧: 线圈 2 (中心), D 电极 (中心线圈外侧环形? 简化: D 在中心, E± 在两侧)
    ax.add_patch(Rectangle((0.5, 0.6), 9.0, 1.0, fc='white', ec='k', lw=LW)); ax.text(9.2, 1.1, '1', fontsize=13, va='center', bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec='k'))
    # 线圈 (剖面: 两组导体截面)
    for xx in np.linspace(3.9, 4.6, 4):
        ax.add_patch(Rectangle((xx - 0.08, 1.6), 0.16, 0.12, fc='white', ec='k', lw=0.9))
        ax.add_patch(Rectangle((10 - xx - 0.08 - 0.0, 1.6), 0.16, 0.12, fc='white', ec='k', lw=0.9))
    ax.text(5.0, 1.95, '2 (磁线圈)', fontsize=11, ha='center')
    # 电极: D 中央 (线圈内侧), E+ E− 两侧
    ax.add_patch(Rectangle((4.75, 1.6), 0.5, 0.1, fc='k', ec='k')); ax.text(5.0, 1.3, 'D (10a)', fontsize=11, ha='center', va='top', bbox=dict(fc='white', ec='none', pad=0))
    ax.add_patch(Rectangle((1.6, 1.6), 1.3, 0.1, fc='k', ec='k')); ax.text(2.25, 1.35, r'$E_-$ (10c)', fontsize=11, ha='center', va='top')
    ax.add_patch(Rectangle((7.1, 1.6), 1.3, 0.1, fc='k', ec='k')); ax.text(7.75, 1.35, r'$E_+$ (10b)', fontsize=11, ha='center', va='top')
    # 浮置单匝环 (剖面: 两个矩形), 倾斜 θ 绕中心
    th = np.deg2rad(6); cx, cz = 5.0, 4.3; L = 3.6
    def rot(x, z): return cx + (x - cx) * np.cos(th) - (z - cz) * np.sin(th), cz + (x - cx) * np.sin(th) + (z - cz) * np.cos(th)
    for x0 in (cx - L / 2 - 0.5, cx + L / 2 - 0.5):
        pts = [rot(x0, cz - 0.15), rot(x0 + 1.0, cz - 0.15), rot(x0 + 1.0, cz + 0.15), rot(x0, cz + 0.15)]
        ax.add_patch(plt.Polygon(pts, fc='white', ec='k', lw=1.6))
    p0, p1 = rot(cx - L / 2 - 0.9, cz), rot(cx + L / 2 + 0.9, cz); ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=0.8)
    ax.text(cx + L / 2 + 1.0, cz + 0.45, '3 (浮置导体 P)', fontsize=11)
    ax.text(cx, cz + 0.55, r'$V_P$', fontsize=12, ha='center')
    # 转轴与杠杆
    ax.plot([cx], [cz], 'ko', ms=4); ax.text(cx + 0.1, cz - 0.55, '转轴', fontsize=10)
    ax.annotate('', xy=(cx + L / 2, 5.6), xytext=(cx, 5.6), arrowprops=dict(arrowstyle='<->', lw=1.0, color='k')); ax.text(cx + L / 4, 5.75, r'$+\ell$', fontsize=11, ha='center')
    ax.annotate('', xy=(cx - L / 2, 5.6), xytext=(cx, 5.6), arrowprops=dict(arrowstyle='<->', lw=1.0, color='k')); ax.text(cx - L / 4, 5.75, r'$-\ell$', fontsize=11, ha='center')
    ax.add_patch(Arc((cx, cz), 3.0, 1.2, theta1=0, theta2=6, lw=1.2, color='k')); ax.text(cx + 1.6, cz + 0.12, r'$\theta$', fontsize=12)
    # 电容 (虚线 + 符号)
    def capsym(x, z0, z1, txt, side=1):
        ax.plot([x, x], [z0, z1], 'k:', lw=1.2)
        zm = (z0 + z1) / 2
        ax.plot([x - 0.18, x + 0.18], [zm - 0.06, zm - 0.06], 'k', lw=1.6); ax.plot([x - 0.18, x + 0.18], [zm + 0.06, zm + 0.06], 'k', lw=1.6)
        ax.text(x + 0.25 * side, zm, txt, fontsize=12, va='center', ha='left' if side > 0 else 'right')
    capsym(2.25, 1.72, rot(2.25, cz - 0.15)[1], r'$C_-$', -1)
    capsym(7.75, 1.72, rot(7.75, cz - 0.15)[1], r'$C_+$', 1)
    capsym(5.0, 1.72, cz - 0.15, r'$C_D$', 1)
    # 间隙 d
    ax.annotate('', xy=(9.0, 1.72), xytext=(9.0, cz - 0.15), arrowprops=dict(arrowstyle='<->', lw=1.0, color='k')); ax.text(9.15, (1.72 + cz - 0.15) / 2, r'$d=g_0-z$', fontsize=11, va='center')
    ax.text(0.5, 6.6, '固定侧: 磁线圈 2 + 电容激励极 D + 检测极 E+/E− (转轴两侧 ±ℓ); 运动侧: 同一单匝环 3 作浮置导体, 无引线', fontsize=11)
    ax.text(0.3, 0.1, '(a) 结构 (剖面示意)', fontsize=13)
    # (b) 节点电路
    ax = axs[1]; ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.axis('off')
    def cap2(x0, y0, x1, y1, txt, toff=(0.25, 0)):
        ax.plot([x0, x1], [y0, y1], 'k', lw=1.0)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        d = np.array([x1 - x0, y1 - y0]); d = d / np.linalg.norm(d); n = np.array([-d[1], d[0]])
        for sgn in (-1, 1):
            c = np.array([mx, my]) + sgn * 0.07 * d
            ax.plot([c[0] - 0.2 * n[0], c[0] + 0.2 * n[0]], [c[1] - 0.2 * n[1], c[1] + 0.2 * n[1]], 'k', lw=1.8)
        ax.add_patch(Rectangle((mx - 0.06, my - 0.06), 0.12, 0.12, fc='white', ec='none'))
        ax.text(mx + toff[0], my + toff[1], txt, fontsize=12, va='center')
    P = (4.0, 5.0)
    dot(ax, *P); ax.text(4.0, 5.35, r'$P$ (3, 浮置, $V_P$)', fontsize=12, ha='center')
    cap2(4.0, 2.4, 4.0, 5.0, r'$C_D$'); box(ax, 3.4, 1.4, 1.2, 1.0, 'D\n激励 $V_D$', fs=10.5)
    cap2(1.5, 2.4, 4.0, 5.0, r'$C_-$', (-0.55, 0.15)); box(ax, 0.9, 1.4, 1.2, 1.0, r'$E_-$' + '\n虚地', fs=10.5)
    cap2(6.5, 2.4, 4.0, 5.0, r'$C_+$', (0.25, 0.15)); box(ax, 5.9, 1.4, 1.2, 1.0, r'$E_+$' + '\n虚地', fs=10.5)
    cap2(4.0, 5.0, 7.2, 5.0, r'$C_0$', (0.0, 0.3)); wire(ax, [(7.2, 5.0), (7.2, 4.4)]); wire(ax, [(7.0, 4.4), (7.4, 4.4)]); wire(ax, [(7.1, 4.28), (7.3, 4.28)]); wire(ax, [(7.16, 4.16), (7.24, 4.16)])
    ax.plot([1.5, 1.5], [2.4, 2.4], 'k'); 
    ax.text(0.4, 6.6, r'$V_P=\dfrac{C_D}{C_\Sigma}V_D,\quad C_\Sigma=C_D+C_++C_-+C_0$', fontsize=12)
    ax.text(0.4, 0.95, r'$H_\pm=\dfrac{C_D C_\pm}{C_\Sigma},\qquad s_C=\dfrac{H_+-H_-}{H_++H_-}=\dfrac{C_+-C_-}{C_++C_-}\approx\dfrac{\ell\,\theta}{d}$', fontsize=12)
    ax.text(0.4, 0.3, '与磁观测 m(z, θ) 联合: det J = (∂m/∂z)·ℓ/d ≠ 0 (§10.2)', fontsize=11)
    ax.text(0.3, 0.1 - 0.5, '', fontsize=1)
    ax.text(0.3, 5.9, '(b) 电容网络 (D 共同激励, E± 分别接收)', fontsize=13)
    fig.text(0.02, 0.01, '图6', fontsize=18)
    savefig(fig, '图6_电磁与电容混合实施.png')

# ---------------- 图7: 联合估计流程 ----------------
def fig7():
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=110); ax.set_xlim(0, 17); ax.set_ylim(0, 8.5); ax.axis('off')
    def b(x, y, w, h, t, fs=11): box(ax, x, y, w, h, t, fs)
    def a(p0, p1): arrow(ax, p0, p1, lw=1.3, ms=14)
    b(0.4, 6.2, 2.4, 1.4, '2/3\n固定线圈阵列\n+ 无源单匝环', 11)
    b(3.4, 6.2, 2.2, 1.4, '6/7/8\n端口选通\n激励 / 接收', 11)
    b(6.2, 6.2, 2.6, 1.4, '端口响应\n自响应 $Z_{ii}$\n传递响应 $Z_{ij}$', 11)
    a((2.8, 6.9), (3.4, 6.9)); a((5.6, 6.9), (6.2, 6.9))
    b(9.6, 6.2, 3.0, 1.4, '参考测量 / 标定\n直接耦合基线 $j\\omega M_{ij}$\n端接状态、链路增益', 10)
    a((8.8, 6.9), (9.6, 6.9))
    b(13.3, 6.2, 3.3, 1.4, '响应模型 (图5)\n$\\mathbf{Z}_{\\mathrm{eff}}(\\mathbf{q},\\eta)$\n含 $\\mathbf{Z}_{PP}$ 环间耦合与损耗', 10)
    a((12.6, 6.9), (13.3, 6.9))
    b(4.0, 3.4, 4.4, 1.7, '9 联合估计\n$\\Delta\\mathbf{y}=\\mathbf{J}_q\\Delta\\mathbf{q}+\\mathbf{J}_\\eta\\Delta\\eta+\\mathbf{n}$\n目标 $\\mathbf{q}$ 与干扰 $\\eta$ 同时/分离估计', 11)
    a((7.5, 6.2), (6.2, 5.1)); a((14.9, 6.2), (8.4, 4.6))
    b(9.6, 3.4, 3.0, 1.7, '可辨识判据\n$\\mathrm{rank}[\\mathbf{J}_\\eta\\ \\mathbf{J}_q]-\\mathrm{rank}(\\mathbf{J}_\\eta)=k$\n(工作域内检查, §8.4)', 10)
    a((8.4, 4.25), (9.6, 4.25))
    b(0.4, 3.4, 2.9, 1.7, '先验 / 约束\n运动约束、承载部\n形状关联 (权13)', 10)
    a((3.3, 4.25), (4.0, 4.25))
    b(4.0, 0.6, 4.4, 1.6, '输出: 各局部被测部 12 的\n目标位移 $\\Delta z_i$ 与转角 $\\theta_i$\n(或可辨识子集, 权14)', 11)
    a((6.2, 3.4), (6.2, 2.2))
    b(9.6, 0.6, 3.0, 1.6, '干扰量输出/更新\n邻域运动、环损耗\n温度、链路漂移', 10)
    a((7.0, 3.4), (10.0, 2.2))
    b(13.3, 0.6, 3.3, 1.6, '迭代: 工作点更新\n重线性化 / 重标定\n(慢通道)', 10)
    a((12.6, 1.4), (13.3, 1.4)); a((14.9, 2.2), (14.9, 6.2))
    ax.text(0.4, 8.1, '固定端口取得的是多环网络的合成响应; 各局部环的运动经联合模型区分, 不将某固定线圈的响应无条件归属于一个环 (§9.3, 权10)', fontsize=12)
    fig.text(0.02, 0.01, '图7', fontsize=18)
    savefig(fig, '图7_多区域联合空间状态估计流程.png')

def legend_md():
    md = '''# E1 v0.5 候选附图 标号表 (内部讨论版)

| 标号 | 部件 | 对应 v0.5 |
|---|---|---|
| 1 | 基座 (PCB) | §5.2 测量基准 |
| 2 | 固定线圈 (平面螺旋); 2a 标称层 (L1+L2), 2b 下沉层 (L3+L4), 固定高度差 h | §6.1, §9.2, 权6/7 |
| 3 | 单匝导电环: 无源闭合、无引线、周向连续导通; 环孔 11 | §6.1, 权3/4 |
| 4 | 局部承载部 (承载岛, 非导电), 使环轮廓与法向跟随局部被测部 | §6.1, 权13 |
| 5 | 机械支承 (弹性支柱; 亦可为泡棉/柔性梁/膜片), 只限定行程 | §6.2, 权13 |
| 6 / 7 / 8 / 9 | 端口选通 / 激励电路 / 接收电路 / 处理电路 (均在固定侧) | §6.1, §7 |
| 10a / 10b / 10c | 电容激励极 D / 检测极 E+ / E− (固定侧, 转轴两侧 ±ℓ) | §10.1, 权11/12 |
| 12 | 局部被测部 P1、P2 | §5.1 |
| S0 | 共享固定线圈 (实际参与 P1 与 P2 两个观测组) | §6.3, 权1 |
| S1、S2 | 补充观测固定线圈 (不同位置基线) | §9.1, 权5/6 |
| g | 静息空气间隙 (L1 顶 → 环底); d = g0 − z 电容间隙 | §6.2, §10.2 |
| Δz、θ | 目标位移、目标转角 | §8.4 |

| 图 | 文件 | v0.5 §十七 | 生成 |
|---|---|---|---|
| 图1 | 图1_阵列总体轴测.png | 1 阵列总体轴测图 | CadQuery (`scripts/patent_E1_cad.py` fig8) + 标注 |
| 图2 | 图2_局部共享关系.png | 2 局部共享关系图 | CadQuery (fig8b) + 耦合路径叠加 |
| 图3 | 图3_结构剖视.png | 3 结构剖视图 (空气间隙、承载部、支承、两种高度) | CadQuery 剖切 (fig9) + 尺寸 |
| 图4 | 图4_两种磁读取电路.png | 4 两种磁读取电路图 | matplotlib |
| 图5 | 图5_多固定线圈多无源环网络.png | 5 网络图 | matplotlib |
| 图6 | 图6_电磁与电容混合实施.png | 6 电磁+电容混合实施图 | matplotlib |
| 图7 | 图7_多区域联合空间状态估计流程.png | (可选第七张) 流程图 | matplotlib |

正式版改法: 图内中文与公式说明移入说明书, 只留数字标号与字母符号; 线宽 0.3~0.4mm; 300dpi TIFF; 图1/3 的 z 放大改按比例或在说明书注明。
STEP 源文件 (`reports/patent_E1/cad_*.step`) 可交代理人/机械重出正式图。
'''
    open(os.path.join(OUT, '标号表.md'), 'w').write(md); print('saved 标号表.md')

if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7', 'md']
    for w in which:
        {'1': fig1, '2': fig2, '3': fig3, '4': fig4, '5': fig5, '6': fig6, '7': fig7, 'md': legend_md}[w]()
