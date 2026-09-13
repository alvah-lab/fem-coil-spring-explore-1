#!/usr/bin/env python3
"""标定拼图件 (K18 线圈板): 六边形垫片 + 19 格外框, CadQuery 参数化 → STEP/STL.

用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/calib_fixture_cad.py [--frame-only|--tiles-only]

设计要点
- 六边形片: 对边 AF = PITCH − 2·CLR, 实心块, 顶面平整 + 中心 Ø2.94×0.2 凸柱卡住 Ø5.0/Ø3.0×0.2 铜环的内孔 (环在片顶, 片底贴 PCB 阻焊面)。
  间隙 (环质心 → L1 铜面) = 片高 H + T_RING/2 + T_MASK  →  H = gap − 0.13。
- 倾斜片: 顶面绕片心倾斜 θ (方位 φ), 凸柱沿顶面法向, 环质心仍在片心正上方 gap 处; 高侧边上有 V 形缺口标方向。
- 偏移片: 凸柱中心偏移 (dx, dy)。
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
R_BOSS = 1.47               # 中心凸柱半径: 环内孔 Ø3.0 − 0.06 配合; 环外区域为平面
GAPS = (1.0, 1.35, 1.75, 2.0, 2.53)
TILTS = (0.0, 2.0, 5.0)
OFFSETS = (0.0, 0.5)
CH_BOTTOM, CH_BOSS = 0.25, 0.05   # 片底外缘导入倒角 / 凸柱顶缘倒角
LETTER_H, LETTER_D = 2.0, 0.10    # 凸柱顶面字母: 字高 / 刻深 (凸柱高 0.2)
# 型号字母 (查表; README 有对照): (gap, tilt, tilt_dir, dx, ring)
CODES = {
    'A': dict(gap=1.0),  'B': dict(gap=1.35), 'C': dict(gap=1.75), 'D': dict(gap=2.0), 'E': dict(gap=2.53),
    'F': dict(gap=1.75, tilt_deg=2.0), 'G': dict(gap=1.75, tilt_deg=5.0),
    'H': dict(gap=1.75, dx=0.5),
    'Z': dict(gap=1.75, ring=False),
}

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


def tile(gap=1.75, tilt_deg=0.0, tilt_dir_deg=0.0, dx=0.0, dy=0.0, ring=True, code=''):
    """一片: 实心六棱柱, 顶面(可倾斜)平整, 只留中心凸柱 (Ø2·R_BOSS × T_RING) 卡住环的内孔;
    型号字母刻在凸柱顶面 (空白片刻在顶面中心); 片底外缘倒角, 倾斜高侧 V 缺口."""
    H = gap - T_RING / 2 - T_MASK                     # 片心处顶面高度 = 环底面高度
    th = math.radians(tilt_deg); ph = math.radians(tilt_dir_deg)
    axis = (-math.sin(ph), math.cos(ph), 0)           # 倾斜轴 (过片心, 垂直于方位)
    Hmax = H + R_HEX * math.tan(th) + 0.3
    body = hex_prism(Hmax)
    cutter = (cq.Workplane('XY').rect(20, 20).extrude(10).rotate((0, 0, 0), axis, math.degrees(th)).translate((0, 0, H)))
    body = body.cut(cutter)
    body = body.edges('<Z').chamfer(CH_BOTTOM)
    def on_top(wp):   # 把在 z=0 平面上建的体放到 (可倾斜的) 顶面上, 中心 (dx,dy)
        return wp.rotate((0, 0, 0), axis, math.degrees(th)).translate((dx, dy, H))
    if ring:
        boss = cq.Workplane('XY').circle(R_BOSS).extrude(T_RING).edges('>Z').chamfer(CH_BOSS)
        body = body.union(on_top(boss))
        z_txt = T_RING
    else:
        z_txt = 0.0
    if code:
        try:
            t = (cq.Workplane('XY').text(code, LETTER_H, -LETTER_D, combine=False, halign='center', valign='center')
                 .translate((0, 0, z_txt)))
            body = body.cut(on_top(t))
        except Exception as e:      # 字体缺失时跳过刻字
            print('text skipped:', e)
    # 倾斜方向: 高侧平边中点上开 V 缺口 (顶面)
    if tilt_deg > 0:
        notch = (cq.Workplane('XY').polyline([(-0.5, 0), (0.5, 0), (0, -0.6)]).close().extrude(5)
                 .translate((0, AF / 2 + 0.01, H - 0.6))
                 .rotate((0, 0, 0), (0, 0, 1), tilt_dir_deg - 90))
        body = body.cut(notch)
    return body


def tile_by_code(code):
    return tile(code=code, **CODES[code])


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


GAUGE_GO = AF + 0.05        # 5.10: 打磨到刚好落入 → 拼装后相邻片间隙 ≥ 0.10
GAUGE_REF = PITCH           # 5.20: 必须自由落入 (= 单元间距), 否则进不了阵列


def gauge(t=3.0, depth=2.2, lead=0.3):
    """六边形尺寸量规: 两个六边形通槽 (G = GO 5.10 / R = REF 5.20), 槽口导入倒角, 槽底 Ø3 顶出孔, 顶面刻字母."""
    blk = cq.Workplane('XY').rect(24, 12).extrude(t).edges('|Z').fillet(1.0)
    for x, af, txt in ((-6.0, GAUGE_GO, 'G'), (6.0, GAUGE_REF, 'R')):
        r = af / math.sqrt(3)
        blk = blk.cut(hex_prism(depth, r, x, 0).translate((0, 0, t - depth)))
        # 导入倒角: 槽口从 af+2·lead 收到 af (loft)
        lead_in = (cq.Workplane('XY').workplane(offset=t - lead).polyline(hex_pts(r, x, 0)).close()
                   .workplane(offset=lead).polyline(hex_pts(r + lead * 2 / math.sqrt(3), x, 0)).close().loft())
        blk = blk.cut(lead_in)
        blk = blk.cut(cq.Workplane('XY').circle(1.5).extrude(t).translate((x, 0, 0)))
        try:
            lab = (cq.Workplane('XY').text(txt, 2.5, -0.15, combine=False, halign='center', valign='center')
                   .translate((x, -4.4, t)))
            blk = blk.cut(lab)
        except Exception as e:
            print('text skipped:', e)
    return blk


def export(wp, name, stl=True):
    exporters.export(wp, os.path.join(OUT, name + '.step'))
    if stl:
        exporters.export(wp, os.path.join(OUT, name + '.stl'), tolerance=0.01, angularTolerance=0.1)
    bb = wp.val().BoundingBox()
    print(f'{name:28s} {bb.xlen:6.2f} x {bb.ylen:6.2f} x {bb.zlen:5.2f} mm')


def ring_solid(cx, cy, z, tilt_deg=0.0, tilt_dir_deg=0.0):
    th = math.radians(tilt_deg); ph = math.radians(tilt_dir_deg)
    return (annulus(R_RING_O, R_RING_I, T_RING).rotate((0, 0, 0), (-math.sin(ph), math.cos(ph), 0), math.degrees(th))
            .translate((cx, cy, z)))


def overview(layout=None):
    """装配总览: PCB 片段 (带 6 孔、19 个线圈示意圆环) + 外框 (销入孔) + 19 片 + 铜环. layout: {unit: dict(tile kwargs)}."""
    if layout is None:
        layout = {9: dict(code='G', **CODES['G'])}
        for k, u in enumerate(G.ADJ[9]):
            c = 'ABCDE'[k % 5]; layout[u] = dict(code=c, **CODES[c])
    pcb = cq.Workplane('XY').rect(40, 36).extrude(-1.0)
    for (hx, hy, d) in HOLES:
        pcb = pcb.cut(cq.Workplane('XY').circle(d / 2).extrude(-1.0).translate((hx, hy, 0)))
    for x, y in XY:      # 线圈示意: 阻焊面上 0.02 的浅环
        pcb = pcb.union(annulus(2.4555, 0.37, 0.02).translate((x, y, 0)))
    parts = [pcb.val(), frame().val()]
    for i, (x, y) in enumerate(XY):
        kw = layout.get(i, dict(code='Z', **CODES['Z']))
        t = tile(**kw).translate((x, y, 0))
        parts.append(t.val())
        if kw.get('ring', True):
            H = kw.get('gap', 1.75) - T_RING / 2 - T_MASK
            parts.append(ring_solid(x + kw.get('dx', 0.0), y + kw.get('dy', 0.0), H, kw.get('tilt_deg', 0.0), kw.get('tilt_dir_deg', 0.0)).val())
    return cq.Workplane('XY').newObject([cq.Compound.makeCompound(parts)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles-only', action='store_true'); ap.add_argument('--frame-only', action='store_true')
    a = ap.parse_args()
    if not a.frame_only:
        parts = []
        for code, kw in CODES.items():
            export(tile(code=code, **kw), f'tile_{code}_' + tile_name(kw.get('gap', 1.75), kw.get('tilt_deg', 0), kw.get('tilt_dir_deg', 0), kw.get('dx', 0), kw.get('ring', True)))
        # 装配示意: 19 片 (中心 1.75/5°, 邻居各高度, 其余空白) 供检查拼合
        asm = None
        for i, (x, y) in enumerate(XY):
            t = tile_by_code('G') if i == 9 else (tile_by_code('ABCDE'[i % 5]) if i in G.ADJ[9] else tile_by_code('Z'))
            t = t.translate((x, y, 0))
            asm = t if asm is None else asm.union(t)
        export(asm, 'assembly_tiles', stl=False)
    if not a.tiles_only:
        export(frame(), 'frame_19')
        export(gauge(), 'gauge_hex')
        export(overview(), 'assembly_overview')


if __name__ == '__main__':
    main()
