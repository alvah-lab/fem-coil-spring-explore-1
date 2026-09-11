#!/usr/bin/env python3
"""E1 专利结构图: CadQuery 参数化装配体 → STEP + SVG (隐藏线消除黑线图).
用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/patent_E1_cad.py
输出 reports/patent_E1/cad_*.svg / .step
几何: 19 单元蜂窝 pitch 5.2; 线圈 Ø5 平面螺旋 (示意为 5 匝同心环); 标称层 z=0, 下沉层 z=-0.73;
      单匝环 Ø5/Ø3/0.2 在 z=gap; 泡棉 2.38; 表皮 0.3. 轴测图 z 方向放大 Z_EXAG 倍 (不按比例).
"""
import os, sys, json, math
import numpy as np
import cadquery as cq
from cadquery import Compound, exporters

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'reports', 'patent_E1')
os.makedirs(OUT, exist_ok=True)
LAYOUT = json.load(open(os.path.join(ROOT, 'host', 'honeycomb_host', 'data', 'layout.json')))
XY = np.array([[u['x_mm'], u['y_mm']] for u in LAYOUT]); SUNK = np.array([u['sunk'] for u in LAYOUT])
PITCH = 5.2

# ---- 参数 ----
R_RING_O, R_RING_I, T_RING = 2.5, 1.5, 0.2
COIL_TURNS, COIL_W, COIL_R_OUT = 5, 0.22, 2.4          # 示意 5 匝 (真实 11.7 匝, 线图不可辨)
T_CU = 0.06
Z_NOM, Z_SUNK = 0.0, -0.72          # K18 叠层: L1 0 / L2 -0.236 / L3 -0.719 / L4 -0.955 (铜质心)
GAP = 2.53                                              # 环底面到 L1 (静息)
T_FOAM, T_SKIN = 2.38, 0.3
PCB_T = 1.0

def unit_at(x, y):
    return int(np.argmin(np.hypot(XY[:, 0] - x, XY[:, 1] - y)))

def annulus(ro, ri, t, z=0.0, cx=0.0, cy=0.0):
    return cq.Workplane('XY').circle(ro).circle(ri).extrude(t).translate((cx, cy, z))

def coil_solid(cx, cy, ztop, turns=COIL_TURNS):
    """同心环示意螺旋 (导体顶面在 ztop, 向下厚 T_CU); 加一小段径向连线表示螺旋接续."""
    parts = []
    for k in range(turns):
        r = COIL_R_OUT - k * (2 * COIL_W)
        parts.append(annulus(r + COIL_W / 2, r - COIL_W / 2, T_CU, ztop - T_CU, cx, cy).val())
    # 径向桥接 (从最内到最外), 让线图能看出是绕组而非同心环
    bridge = (cq.Workplane('XY').center(cx + (COIL_R_OUT - (turns - 1) * COIL_W), cy)
              .rect((turns - 1) * 2 * COIL_W, COIL_W).extrude(T_CU).translate((0, 0, ztop - T_CU)).val())
    parts.append(bridge)
    return parts

def ring_solid(cx, cy, zbot, dx=0.0, dy=0.0, dz=0.0, tax=0.0, tay=0.0):
    s = annulus(R_RING_O, R_RING_I, T_RING, 0.0)
    s = s.rotate((0, 0, 0), (1, 0, 0), math.degrees(tax)).rotate((0, 0, 0), (0, 1, 0), math.degrees(tay))
    return s.translate((cx + dx, cy + dy, zbot + dz)).val()

R_ISL, T_ISL = 3.1, 0.35          # 局部承载岛 (非导电) 直径 6.2, 厚 0.35, 环贴其顶面
R_POST, N_POST = 0.45, 3          # 弹性支柱

def island_solids(cx, cy, z_ring_bot, dx=0.0, dy=0.0, dz=0.0, tax=0.0, tay=0.0, post_angles=None):
    """承载岛 (顶面 = 环底面) + 支柱 (PCB 顶到岛底). 岛随环位姿; 支柱固定于基座, 顶端随岛."""
    isl = (cq.Workplane('XY').circle(R_ISL).extrude(T_ISL).translate((0, 0, -T_ISL))
           .rotate((0, 0, 0), (1, 0, 0), math.degrees(tax)).rotate((0, 0, 0), (0, 1, 0), math.degrees(tay))
           .translate((cx + dx, cy + dy, z_ring_bot + dz)).val())
    out = [isl]
    angs = post_angles if post_angles is not None else [90 + 120 * k for k in range(N_POST)]
    for a in angs:
        rx, ry = (R_ISL - 0.6) * math.cos(math.radians(a)), (R_ISL - 0.6) * math.sin(math.radians(a))
        px, py = cx + dx + rx, cy + dy + ry
        # 支柱顶随岛底面 (含倾斜): 绕 x 转 tax → z += y·sin(tax); 绕 y 转 tay → z -= x·sin(tay)
        ztop = z_ring_bot + dz - T_ISL + ry * math.sin(tax) - rx * math.sin(tay)
        h = ztop - 0.06
        out.append(cq.Workplane('XY').circle(R_POST).extrude(h).translate((px, py, 0.06)).val())
    return out

def zscale(shape, k):
    """z 方向缩放 (非均匀) — 用 OCC 变换; CadQuery 无直接 API, 用 gp_GTrsf."""
    from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ
    from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
    g = gp_GTrsf(gp_Mat(1, 0, 0, 0, 1, 0, 0, 0, k), gp_XYZ(0, 0, 0))
    return cq.Shape.cast(BRepBuilderAPI_GTransform(shape.wrapped, g, True).Shape())

def view_rotate(shape, az_deg, el_deg):
    """把模型旋转到: 相机方向 (方位角 az, 仰角 el, 世界 Z 朝上) 变成 +Z 视线, 世界 Z 在图像里朝上. 之后用 projectionDir=(0,0,1)."""
    from OCP.gp import gp_Trsf, gp_Mat, gp_XYZ
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    R = view_matrix(az_deg, el_deg)        # 行: 图像 x, 图像 y, 视线
    t = gp_Trsf(); t.SetValues(*[float(v) for v in np.hstack([R, np.zeros((3, 1))]).ravel()])
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, t, True).Shape())

def export(shapes, name, proj=None, width=1800, height=1200, stroke=0.04, hidden=False, z_exag=1.0, az=None, el=None):
    comp = Compound.makeCompound([s for s in shapes if s is not None])
    if z_exag != 1.0:
        comp = zscale(comp, z_exag)
    R = np.eye(3)
    if az is not None:
        comp = view_rotate(comp, az, el); proj = (0, 0, 1)
        R = view_matrix(az, el)
    wp = cq.Workplane('XY').add(comp)
    exporters.export(wp, os.path.join(OUT, f'{name}.step'))
    exporters.export(wp, os.path.join(OUT, f'{name}.svg'), exportType='SVG',
                     opt=dict(width=width, height=height, marginLeft=30, marginTop=30, showAxes=False,
                              projectionDir=proj, strokeWidth=stroke, strokeColor=(0, 0, 0),
                              hiddenColor=(140, 140, 140), showHidden=hidden))
    # 解析 svg 的 scale/translate, 连同视角矩阵写 json, 供 matplotlib 精确叠加标号
    import re
    svg = open(os.path.join(OUT, f'{name}.svg')).read()
    m = re.search(r'scale\(([-\d.e]+), *([-\d.e]+)\)\s+translate\(([-\d.e]+), *([-\d.e]+)\)', svg)
    meta = dict(scale=float(m.group(1)), tx=float(m.group(3)), ty=float(m.group(4)), width=width, height=height,
                R=R.tolist(), z_exag=z_exag, proj=list(proj) if proj else None)
    json.dump(meta, open(os.path.join(OUT, f'{name}.json'), 'w'))
    print('wrote', name)

def view_matrix(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0, 0, 1.0]); right = np.cross(up, d); right /= np.linalg.norm(right); up2 = np.cross(d, right)
    return np.stack([right, up2, d])

# ---------------- 图8: 全阵列轴测 (基座薄板 + 19 线圈 + 19 环 + 局部剖开的泡棉/表皮) ----------------
def fig8_array():
    shapes = []
    # 基座: 薄板置于最深线圈之下 (透明基座画法: 只画板边)
    base = cq.Workplane('XY').rect(40, 36).extrude(0.4).translate((0, 0, Z_SUNK - 0.9)).val()
    shapes.append(base)
    for (x, y), s in zip(XY, SUNK):
        shapes += coil_solid(x, y, Z_SUNK if s else Z_NOM)
    # 环: 全部同高, 其中中心单元下压+倾斜示意运动 (P1), 相邻一个侧移 (P2)
    for i, ((x, y), s) in enumerate(zip(XY, SUNK)):
        if i == 9:
            shapes.append(ring_solid(x, y, GAP, dz=-0.6, tax=0.12))
        elif i == 10:
            shapes.append(ring_solid(x, y, GAP, dx=0.4, dz=-0.3))
        else:
            shapes.append(ring_solid(x, y, GAP))
    # 泡棉与表皮: 只画一个角 (局部剖开), 避免遮挡
    export(shapes, 'cad_fig8_array_top', proj=(0, 0, 1), width=1400, height=1300)
    # 局部承载岛 + 弹性支柱 (空气间隙)
    for i, ((x, y), s_) in enumerate(zip(XY, SUNK)):
        if i == 9:
            shapes += island_solids(x, y, GAP, dz=-0.6, tax=0.12)
        elif i == 10:
            shapes += island_solids(x, y, GAP, dx=0.4, dz=-0.3)
        else:
            shapes += island_solids(x, y, GAP)
    export(shapes, 'cad_fig8_array_iso', az=-125, el=28, z_exag=2.5)

# ---------------- 图8b: 3 线圈 + 2 环 子集 ----------------
def fig8b_subset():
    # S0 中心 (标称), S1 左下沉? 取单元 9(中心,下沉), 10, 4 → 用真实层高
    S0, S1, S2 = unit_at(0, 0), unit_at(2.6, 4.503), unit_at(-2.6, 4.503)
    ids = [S0, S1, S2]
    shapes = []
    base = cq.Workplane('XY').rect(16, 12).extrude(0.4).translate((0, 2.0, Z_SUNK - 0.9)).val()
    shapes.append(base)
    for i in ids:
        shapes += coil_solid(XY[i, 0], XY[i, 1], Z_SUNK if SUNK[i] else Z_NOM)
    # 两个环 P1 (S1 上方, 下压+倾斜) P2 (S2 上方, 侧移): 各自随局部承载岛
    shapes.append(ring_solid(XY[S1, 0], XY[S1, 1], GAP, dz=-0.4, tay=-0.1))
    shapes += island_solids(XY[S1, 0], XY[S1, 1], GAP, dz=-0.4, tay=-0.1)
    shapes.append(ring_solid(XY[S2, 0], XY[S2, 1], GAP, dx=0.3))
    shapes += island_solids(XY[S2, 0], XY[S2, 1], GAP, dx=0.3)
    export(shapes, 'cad_fig8b_subset_iso', az=-120, el=30, width=1500, height=1100, z_exag=2.0)

# ---------------- 图9: 剖视 (穿过单元 9 下沉 与 10 标称, 沿 x 轴) ----------------
def fig9_section():
    ids = [unit_at(0, 0), unit_at(PITCH, 0)]          # 下沉(中心) + 标称(右邻), 同在 y=0 行
    cx = PITCH / 2
    shapes = []
    pcb = cq.Workplane('XY').rect(16, 10).extrude(PCB_T).translate((cx, 0, -PCB_T)).val()
    shapes.append(pcb)
    for i in ids:
        shapes += coil_solid(XY[i, 0], XY[i, 1], Z_SUNK if SUNK[i] else Z_NOM, turns=6)
    for k, i in enumerate(ids):
        # 右单元 (标称) 示意运动: 下压 + 倾斜
        dz, tay = (0.0, 0.0) if k == 0 else (-0.35, 0.10)
        shapes.append(ring_solid(XY[i, 0], XY[i, 1], GAP, dz=dz, tay=tay))
        shapes += island_solids(XY[i, 0], XY[i, 1], GAP, dz=dz, tay=tay, post_angles=(0, 180))   # 支柱在剖面内
    # 加两条铜层示意线 (L2 与 L4 层深处的薄片), 让剖面能看出四层
    for zl in (-0.236, -0.955):
        shapes.append(cq.Workplane('XY').rect(16, 10).extrude(0.02).translate((cx, 0, zl)).val())
    comp = Compound.makeCompound(shapes)
    # 切掉 y<0 半, 露出剖面 (相机在 -y 侧看向 +y)
    cutter = cq.Workplane('XY').rect(40, 20).extrude(10).translate((cx, -10, -5)).val()
    half = cq.Workplane('XY').add(comp).cut(cq.Workplane().add(cutter)).val()
    exporters.export(cq.Workplane().add(half), os.path.join(OUT, 'cad_fig9_section.step'))
    export([half], 'cad_fig9_section', az=-90, el=0, width=1800, height=900, stroke=0.03, z_exag=2.0)
    export([half], 'cad_fig9_section_iso', az=-70, el=22, width=1800, height=1000, stroke=0.03, z_exag=2.0)
    print('wrote fig9')

if __name__ == '__main__':
    which = sys.argv[1:] or ['8', '8b', '9']
    if '8' in which: fig8_array()
    if '8b' in which: fig8b_subset()
    if '9' in which: fig9_section()
