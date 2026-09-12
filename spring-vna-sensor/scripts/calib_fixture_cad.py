#!/usr/bin/env python3
"""标定拼图件 (K18 线圈板): 六边形垫片 + 19 格外框, CadQuery 参数化 → STEP/STL.

用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/calib_fixture_cad.py [--frame-only|--tiles-only]

设计要点
- 六边形片: 对边 AF = PITCH − 2·CLR, 实心块, 顶面开环形凹槽放 Ø5.0/Ø3.0×0.2 铜环 (环在片顶, 片底贴 PCB 阻焊面)。
  间隙 (环质心 → L1 铜面) = 片高 H + T_RING/2 + T_MASK  →  H = gap − 0.13。
- 倾斜片: 顶面绕片心倾斜 θ (方位 φ), 环质心仍在片心正上方 gap 处; 高侧边上有 V 形缺口标方向。
- 偏移片: 凹槽中心偏移 (dx, dy)。
- 底面刻字 "gap/θ" (贴板一侧, 读数时翻过来看)。
- 外框: 19 格蜂窝外轮廓 (AF+2·CLR_FRAME) 的一圈围墙, 用板上 Ø2.2 定位孔 (尼龙 M2) 定位, 围墙内侧净空
  避开阵列周边元件 (HMC544A / 0402)。
"""
import os, sys, math, argparse
import numpy as np
import cadquery as cq
from cadquery import exporters

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'calib_fixture')
os.makedirs(OUT, exist_ok=True)

# ---- 几何常数 (mm) ----
PITCH = 5.2
CLR_TILE = 0.075            # 片与片 / 片与框 单边配合间隙 → 相邻片间 0.15
AF = PITCH - 2 * CLR_TILE   # 5.05 对边
R_HEX = AF / math.sqrt(3)   # 2.916 外接圆半径
R_RING_O, R_RING_I, T_RING = 2.5, 1.5, 0.2
T_MASK = 0.03               # 阻焊 + 铜面到片底
R_RECESS_O, R_RECESS_I = 2.53, 1.47   # 凹槽比环大 0.03 单边; 2.53 > AF/2=2.525 → 六个平边处露 0.005 (可忽略)
D_RECESS = T_RING
GAPS = (1.0, 1.35, 1.75, 2.0, 2.53)
TILTS = (0.0, 2.0, 5.0)
OFFSETS = (0.0, 0.5)

# 单元坐标 (与 host geometry.py 一致: 主机单元 i == 板 COILii)
sys.path.insert(0, os.path.join(HERE, '..', 'host'))
from honeycomb_host import geometry as G   # noqa: E402
XY = G.XY

# 板上定位孔 (相对阵列中心的坐标, 由 K18 track3.kicad_pcb 提取; 见 fill_from_pcb())
# K18 track3.kicad_pcb: 阵列中心 (62.0, 40.0), 板 y 向下 → 主机 y = -(y_board-40); HLOC1..6 NPTH Ø2.2 (尼龙 M2)
HOLES = [(73.68 - 62.0, -(36.33 - 40.0), 2.2), (60.02 - 62.0, -(27.08 - 40.0), 2.2), (51.62 - 62.0, -(32.62 - 40.0), 2.2),
         (50.29 - 62.0, -(44.76 - 40.0), 2.2), (59.47 - 62.0, -(52.57 - 40.0), 2.2), (71.90 - 62.0, -(48.35 - 40.0), 2.2)]
# 孔心到格子边界 1.11~1.71 mm (孔缘到格子 0.01~0.6): 不能做螺钉凸台, 改为框底向下伸出的定位销 (Ø2.0 入 Ø2.2 孔)
# 周边元件 (0402 电容 / HMC544A) 包络边缘都在半径 19.6 mm 以外, 外框外径 ≤ 16.3 mm, 无需净空槽
KEEPOUT = []
PIN_D, PIN_LEN, PIN_CHAMFER = 2.0, 1.3, 0.3   # 板厚 1.0


def hex_pts(r=R_HEX, cx=0.0, cy=0.0, rot_deg=30.0):
    """平边法向指向相邻单元 (0°, 60°, …) 的六边形: 顶点在 30°+k·60°."""
    return [(cx + r * math.cos(math.radians(rot_deg + 60 * k)), cy + r * math.sin(math.radians(rot_deg + 60 * k))) for k in range(6)]


def hex_prism(h, r=R_HEX, cx=0.0, cy=0.0):
    return cq.Workplane('XY').polyline(hex_pts(r, cx, cy)).close().extrude(h)


def annulus(ro, ri, t):
    return cq.Workplane('XY').circle(ro).circle(ri).extrude(t)


def tile(gap=1.75, tilt_deg=0.0, tilt_dir_deg=0.0, dx=0.0, dy=0.0, ring=True, label=None):
    """一片: 实心六棱柱, 顶面(可倾斜)开环槽, 底面刻字, 倾斜高侧 V 缺口."""
    H = gap - T_RING / 2 - T_MASK                     # 片心处环槽底面高度
    th = math.radians(tilt_deg); ph = math.radians(tilt_dir_deg)
    Hmax = H + R_HEX * math.tan(th) + 0.3
    body = hex_prism(Hmax + D_RECESS)
    # 顶面: 过 (0,0,H+D_RECESS) 法向 n 的平面以上切掉 (环槽底在 H, 槽深 D_RECESS → 顶面在 H+D_RECESS)
    n = np.array([-math.sin(th) * math.cos(ph), -math.sin(th) * math.sin(ph), math.cos(th)])
    cutter = (cq.Workplane('XY').rect(20, 20).extrude(10)
              .rotate((0, 0, 0), (-math.sin(ph), math.cos(ph), 0), math.degrees(th))
              .translate((0, 0, H + D_RECESS)))
    body = body.cut(cutter)
    if ring:
        rec = (annulus(R_RECESS_O, R_RECESS_I, D_RECESS + 1.0)
               .rotate((0, 0, 0), (-math.sin(ph), math.cos(ph), 0), math.degrees(th))
               .translate((dx, dy, H)))
        body = body.cut(rec)
    # 倾斜方向: 高侧平边中点上开 V 缺口 (顶面)
    if tilt_deg > 0:
        notch = (cq.Workplane('XY').polyline([(-0.5, 0), (0.5, 0), (0, -0.6)]).close().extrude(5)
                 .translate((0, AF / 2 + 0.01, H - 0.4))
                 .rotate((0, 0, 0), (0, 0, 1), tilt_dir_deg - 90))
        body = body.cut(notch)
    # 底面刻字 (深 0.25, 镜像使从底部看正向)
    txt = label or (f'{gap:g}' + (f'/{tilt_deg:g}' if tilt_deg else '') + (f'/x{dx:g}' if dx else '') + ('' if ring else 'B'))
    try:
        t = cq.Workplane('XY').text(txt, 1.1, 0.25, combine=False, halign='center', valign='center')
        # 字从 z=0 向 +z 刻进片底 0.25; 从底部看 (绕 y 翻转) 时字要正向 → 先绕 YZ 镜像一次
        t = t.mirror('YZ')
        body = body.cut(t)
    except Exception as e:      # 字体缺失时跳过刻字
        print('text skipped:', e)
    return body


def tile_name(gap, tilt, tdir, dx, ring):
    s = f'tile_g{gap:g}'
    if tilt: s += f'_t{tilt:g}d{tdir:g}'
    if dx: s += f'_x{dx:g}'
    if not ring: s += '_blank'
    return s.replace('.', 'p')


def frame(t_frame=2.0, wall=3.0):
    """19 格外框: 蜂窝外轮廓 (片外接圆 + 间隙) 向外 wall 的围墙; 6 根定位销从框底伸入板上 Ø2.2 孔."""
    r_cell = (PITCH + 2 * CLR_TILE) / math.sqrt(3)
    r_out = (PITCH + 2 * CLR_TILE + 2 * wall) / math.sqrt(3)
    cells = outer = None
    for x, y in XY:
        c = hex_prism(t_frame, r_cell, x, y); o = hex_prism(t_frame, r_out, x, y)
        cells = c if cells is None else cells.union(c)
        outer = o if outer is None else outer.union(o)
    fr = outer.cut(cells)
    for (hx, hy, drill) in HOLES:
        pin = (cq.Workplane('XY').circle(PIN_D / 2).extrude(-(PIN_LEN - PIN_CHAMFER))
               .faces('<Z').workplane().circle(PIN_D / 2).workplane(offset=PIN_CHAMFER).circle(PIN_D / 2 - PIN_CHAMFER).loft()
               .translate((hx, hy, 0)))
        fr = fr.union(pin)
    # 方向标记: +x 侧外壁一个小三角凸起 (对应主机 x 轴 / 板上 L19 方向)
    mark = cq.Workplane('XY').polyline([(0, -0.8), (0, 0.8), (1.0, 0)]).close().extrude(t_frame).translate((13.08 + wall * 1.1547 - 0.2, 0, 0))
    return fr.union(mark)


def export(wp, name, stl=True):
    exporters.export(wp, os.path.join(OUT, name + '.step'))
    if stl:
        exporters.export(wp, os.path.join(OUT, name + '.stl'), tolerance=0.01, angularTolerance=0.1)
    bb = wp.val().BoundingBox()
    print(f'{name:28s} {bb.xlen:6.2f} x {bb.ylen:6.2f} x {bb.zlen:5.2f} mm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles-only', action='store_true'); ap.add_argument('--frame-only', action='store_true')
    a = ap.parse_args()
    if not a.frame_only:
        parts = []
        for gap in GAPS:
            export(tile(gap), tile_name(gap, 0, 0, 0, True))
        for tilt in TILTS[1:]:
            export(tile(1.75, tilt, 0), tile_name(1.75, tilt, 0, 0, True))
        export(tile(1.75, 0, 0, 0.5), tile_name(1.75, 0, 0, 0.5, True))
        export(tile(1.75, ring=False), tile_name(1.75, 0, 0, 0, False))
        # 装配示意: 19 片 (中心 1.75/5°, 邻居各高度, 其余空白) 供检查拼合
        asm = None
        for i, (x, y) in enumerate(XY):
            t = tile(1.75, 5, 0) if i == 9 else tile(GAPS[i % len(GAPS)], ring=(i in G.ADJ[9]))
            t = t.translate((x, y, 0))
            asm = t if asm is None else asm.union(t)
        export(asm, 'assembly_tiles', stl=False)
    if not a.tiles_only:
        if HOLES:
            export(frame(), 'frame_19')
        else:
            print('HOLES empty: frame skipped (fill from K18 PCB first)')


if __name__ == '__main__':
    main()
