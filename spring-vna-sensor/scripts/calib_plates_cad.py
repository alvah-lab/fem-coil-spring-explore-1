#!/usr/bin/env python3
"""K18 标定整板 (一体打印, N=8, 转位复用): 19 格实心蜂窝 + 围框 + 12 定位通孔 + 板号, 每块板一个预先固定的环布局.

用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/calib_plates_cad.py [--only A1,B2]
输出 reports/calib_plates/: plate_<code>.step/.stl, plates.json (布局表, 供主机 GUI/孪生), README.md 由脚本写表格.

几何 (mm)
- 格: 对边 = PITCH 5.2 无缝拼接, 有环格顶面在 H = gap − 0.13 (环底面; 环质心→L1 铜面 = gap), 中心凸柱 Ø2.94×0.2 卡环内孔 + 其上完整扁圆锥 (母线 30°, 高 0.85, 收尖) 导入;
  无环格实心低平台 H_BLANK = 0.8 (无凸柱). 倾斜格顶面绕格心倾斜 (凸柱沿法向), 偏移格凸柱偏移 (dx,dy).
- 围框: 蜂窝外轮廓外扩 WALL, 高 T_RIM; 12 个 Ø2.15 通孔 = HLOC2/HLOC5 两颗 M2 螺钉 (板背面穿出) 的 6 个转位像; +x 侧三角方向标 (取向 k=0); 板号刻在 +x 方向标旁的围框顶面 (竖排, 两孔之间).
- 板底 z=0 整面贴 PCB 阻焊面. 每个有环格在环座平面以上切一圈 Ø(5.0+0.4) 的余隙 (穿过更高的邻格/围框), 保证偏移/倾斜的环也放得进去.
- 压板 (clamp_<code>): 与该板互补, 同外形、同 12 孔; 每个有环格一个 Ø3.5/Ø4.8 环形压脚落在环顶面 (倾斜格压脚同样倾斜, 偏移格压脚同样偏移),
  压脚实心, 端面中心为与凸柱锥互补的 30° 锥形凹 (Ø3.5, 深 1.01); 围框脚比围框顶高 0.1 (先压环, 再顶围框). 板厚 4.0. 顶面刻 C+板号, +x 三角方向标.
- 打印 (SLA/MSLA): 板 = 平放, 板底直接贴平台 (无支撑, 顶面全部朝上, 无悬空, 孔垂直); 压板 = 顶面贴平台, 压脚朝上. 贴平台面的孔口有 0.3 倒角抵消大象脚.
  层高取 0.02 mm (所有环座高 1.22/1.62/2.40、围框 2.0、平台 0.8 都是整数层).
"""
import os, sys, math, json, argparse
import numpy as np
import cadquery as cq
from cadquery import exporters

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'calib_plates')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'host'))
from honeycomb_host import geometry as G   # noqa: E402
XY = G.XY

PITCH = 5.2
R_CELL = PITCH / math.sqrt(3)
R_RING_O, R_RING_I, T_RING = 2.5, 1.5, 0.2
T_MASK = 0.03
R_BOSS = 1.47
A_CONE = 30.0                     # 凸柱顶部完整扁圆锥: 母线与水平面 30°, 高 R_BOSS·tan30 = 0.85, 收到尖
H_BLANK = 0.8
WALL, T_RIM = 3.5, 2.0
CH_LETTER = 0.05
CLR_RING = 0.2                    # 环外沿到任何更高结构 (邻格/围框) 的余隙
CH_FOOT = 0.3                     # 贴打印平台那一面的孔口倒角 (大象脚)
T_CLAMP, CLR_TOP, CLR_RIM = 4.0, 0.5, 0.1   # 压板厚; 压板体底面离最高环顶/围框顶; 围框脚离围框顶
R_PAD_O, R_PAD_I = 2.4, 1.75              # 压脚: 实心 Ø4.8 圆柱, 端面中心开与凸柱锥互补的 30° 锥形凹 (面上 Ø3.5, 深 1.01; 凸柱锥 Ø2.94 高 0.85, 径向余隙 0.28, 顶余隙 0.16)
# 定位: 板背面从 HLOC2 (60.02,27.08) 与 HLOC5 (59.47,52.57) 伸出两颗 M2 螺钉 (相隔 160°), 整板绕阵列中心按 60° 转位复用;
# 板上开这两点各自 6 个旋转像共 12 个通孔 (Ø2.15, 相互 ≥4.5 mm). 六个 HLOC 本身不是 60° 对称, 不能六颗都用.
STUDS = [(60.02 - 62.0, -(27.08 - 40.0)), (59.47 - 62.0, -(52.57 - 40.0))]
HOLE_D = 2.15
def stud_holes():
    out = []
    for (x, y) in STUDS:
        for k in range(6):
            a = math.radians(60 * k)
            out.append((x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)))
    return out
HOLES = stud_holes()


def rotation_maps():
    """取向 k (板逆时针转 60k°) 下 板格 u → 板上单元 (按格心旋转后最近单元)."""
    maps = {}
    for k in range(6):
        a = math.radians(60 * k); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        rot = XY @ R.T
        m = {}
        for u in range(G.NU):
            d = np.hypot(XY[:, 0] - rot[u, 0], XY[:, 1] - rot[u, 1]); v = int(np.argmin(d)); assert d[v] < 1e-6
            m[u] = v
        maps[k] = m
    return maps

CLASS = {c: [int(i) for i in np.where(G.COLOR == c)[0]] for c in (0, 1, 2)}   # 三色: 同类互不相邻


def ring(gap, tilt=0.0, tdir=0.0, dx=0.0, dy=0.0):
    return dict(gap=gap, tilt=tilt, tdir=tdir, dx=dx, dy=dy)


def isolated_pairs(n_pairs=4):
    """贪心选 n 对相邻单元, 使任一对的成员与其它对的成员都不相邻 (对之间至少隔一个单元)."""
    pairs = []
    used_nb = set()
    edges = sorted(G.EDGES, key=lambda e: (abs(XY[e[0]][0] + XY[e[1]][0]) + abs(XY[e[0]][1] + XY[e[1]][1])))
    # 从外围往里挑, 保证互相隔开
    for i, j in sorted(G.EDGES, key=lambda e: -(np.hypot(*XY[e[0]]) + np.hypot(*XY[e[1]]))):
        if i in used_nb or j in used_nb:
            continue
        pairs.append((i, j))
        for u in (i, j):
            used_nb.add(u)
            used_nb.update(G.ADJ[u])
        if len(pairs) == n_pairs:
            break
    return pairs


def by_angle(units):
    return sorted(units, key=lambda u: math.atan2(XY[u][1], XY[u][0]) % (2 * math.pi))


def build_layouts():
    """N = 8: 转位复用 (板绕中心转 60k°) 后的最小集. 同一轨道的 6 格做两种状态各 3 个, 转位后每个单元两种都测到."""
    L = {}
    L['A0'] = dict(desc='空白: 19 格实心无环 (塑料件对基线的影响)', cells={})
    outer0 = by_angle([u for u in CLASS[0] if u != 9])
    cells = {9: ring(1.75)}
    for k, u in enumerate(outer0):
        cells[u] = ring(1.75) if k % 2 == 0 else ring(1.75, 5.0, 120.0 * (k // 2))
    L['A1C'] = dict(desc='类 0: 中心 + 外圈 3 格平放 @1.75, 外圈另 3 格倾斜 5° (方位 0/120/240°); 转位后每单元得平放 + 3 个倾斜方位', cells=cells)
    c1 = by_angle(CLASS[1])
    cells = {}
    for k, u in enumerate(c1):
        cells[u] = ring(1.75) if k % 2 == 0 else ring(1.75, 0, 0, 0.5 * math.cos(math.radians(120 * (k // 2))), 0.5 * math.sin(math.radians(120 * (k // 2))))
    L['A2C'] = dict(desc='类 1 (转 60° 即类 2): 3 格平放 @1.75, 3 格环座偏移 0.5 mm (方向 0/120/240°)', cells=cells)
    for k, gap in zip((1, 2, 3), (1.35, 1.75, 2.53)):
        L[f'B{k}'] = dict(desc=f'全阵列均匀 @{gap} (19 环): 间隙曲线点 + 网络相互作用', cells={u: ring(gap) for u in range(G.NU)})
    pairs = isolated_pairs(4)
    cells = {}
    for p in pairs[:2]:
        cells[p[0]] = ring(1.75); cells[p[1]] = ring(1.75)
    for p in pairs[2:]:
        cells[p[0]] = ring(1.35); cells[p[1]] = ring(2.53)
    L['D'] = dict(desc=f'相邻对: 等高 1.75/1.75 ×2 {pairs[:2]}, 阶差 1.35/2.53 ×2 {pairs[2:]}; 转位覆盖 24 条边', cells=cells)
    L['E1'] = dict(desc='按压轮廓: 中心 1.35, 一圈 1.75, 外圈 2.53 (跟踪器真实形变)',
                   cells={**{u: ring(2.53) for u in range(G.NU)}, **{u: ring(1.75) for u in G.ADJ[9]}, 9: ring(1.35)})
    return L


def hex_pts(r, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(math.radians(30 + 60 * k)), cy + r * math.sin(math.radians(30 + 60 * k))) for k in range(6)]


def hex_prism(h, r, cx=0.0, cy=0.0):
    return cq.Workplane('XY').polyline(hex_pts(r, cx, cy)).close().extrude(h)


def engrave(wp, txt, z_face, size, depth, ch, cx, cy, box, rot=0.0):
    """在 z_face 面刻字 (中心 cx,cy, 绕 z 转 rot°); box = 未旋转时的选边半宽, 只在字槽边倒角 ch (ch<=0 不倒角)."""
    t = (cq.Workplane('XY').text(txt, size, -depth, combine=False, halign='center', valign='center')
         .rotate((0, 0, 0), (0, 0, 1), rot).translate((cx, cy, z_face)))
    wp = wp.cut(t)
    if ch <= 0:
        return wp
    bx, by = (box[1], box[0]) if abs(rot % 180 - 90) < 1e-6 else box
    try:
        sel = cq.selectors.BoxSelector((cx - bx, cy - by, z_face - 1e-3), (cx + bx, cy + by, z_face + 1e-3), boundingbox=True)
        wp = wp.edges(sel).chamfer(ch)
    except Exception as e:
        print('  letter chamfer skipped:', e)
    return wp


def cell_solid(u, spec):
    """一个格: 实心六棱柱 (对边 PITCH), 顶面 (可倾斜) 在 H, 有环时加凸柱."""
    x, y = XY[u]
    if spec is None:
        return hex_prism(H_BLANK, R_CELL, x, y)
    H = spec['gap'] - T_RING / 2 - T_MASK
    th = math.radians(spec['tilt']); ph = math.radians(spec['tdir'])
    axis = (-math.sin(ph), math.cos(ph), 0)
    body = hex_prism(H + R_CELL * math.tan(th) + 0.3, R_CELL, x, y)
    cutter = (cq.Workplane('XY').rect(20, 20).extrude(10).rotate((0, 0, 0), axis, math.degrees(th)).translate((x, y, H)))
    body = body.cut(cutter)
    h_cone = R_BOSS * math.tan(math.radians(A_CONE))
    boss = cq.Workplane('XY').circle(R_BOSS).extrude(T_RING)
    cone = cq.Workplane('XY').add(cq.Solid.makeCone(R_BOSS, 0.0, h_cone, cq.Vector(0, 0, T_RING), cq.Vector(0, 0, 1)))
    boss = boss.union(cone).rotate((0, 0, 0), axis, math.degrees(th)).translate((x + spec['dx'], y + spec['dy'], H))
    return body.union(boss)


def seat_h(spec):
    return spec['gap'] - T_RING / 2 - T_MASK


def tilt_axis(spec):
    th = math.radians(spec['tilt']); ph = math.radians(spec['tdir'])
    return math.degrees(th), (-math.sin(ph), math.cos(ph), 0)


def ring_clearance(u, spec):
    """环座平面以上、环外沿 + CLR_RING 以内的环形区域 (不含凸柱): 从中挖掉更高的邻格/围框."""
    x, y = XY[u]; deg, axis = tilt_axis(spec)
    ann = cq.Workplane('XY').circle(R_RING_O + CLR_RING).circle(R_BOSS + 0.02).extrude(10)
    return ann.rotate((0, 0, 0), axis, deg).translate((x + spec['dx'], y + spec['dy'], seat_h(spec)))


def hole_cutter(hx, hy, z_face, up):
    """通孔 + 贴平台面 (z_face) 的孔口倒角; up=+1 表示实体在 z_face 之上."""
    c = cq.Workplane('XY').circle(HOLE_D / 2).extrude(30).translate((hx, hy, -15))
    cone = cq.Solid.makeCone(HOLE_D / 2 + CH_FOOT, HOLE_D / 2, CH_FOOT, cq.Vector(hx, hy, z_face), cq.Vector(0, 0, up))
    return c.union(cq.Workplane('XY').add(cone))


def outline(h, z0=0.0):
    """围框外轮廓 (19 个外扩六边形的并) 与内轮廓 (19 格) 的柱体, 高 h, 底 z0."""
    r_out = (PITCH + 2 * WALL) / math.sqrt(3)
    outer = None; inner = None
    for x, y in XY:
        o = hex_prism(h, r_out, x, y); i = hex_prism(h, R_CELL, x, y)
        outer = o if outer is None else outer.union(o)
        inner = i if inner is None else inner.union(i)
    return outer.translate((0, 0, z0)), inner.translate((0, 0, z0))


def marker(z0, h):
    x_out = XY[18][0] + (PITCH + 2 * WALL) / 2
    return cq.Workplane('XY').polyline([(-0.3, -0.8), (-0.3, 0.8), (1.0, 0)]).close().extrude(h).translate((x_out, 0, z0))


# 板号: +x 方向标旁的围框顶面 (r=14.8, 角度 0°), 竖排 (转 90°). 12 孔在 18.6/38.7 + 60k°, 0° 附近 ±20° 无孔.
LAB_XY, LAB_ROT = (14.8, 0.0), 90.0


def clamp(code, layout):
    """压板: 与板互补. 返回 (装配位姿实体, z_bottom, z_top)."""
    cells = layout['cells']
    tops = [T_RIM] + [seat_h(s) + T_RING + R_PAD_O * math.tan(math.radians(s['tilt'])) for s in cells.values()]
    z_b = max(tops) + CLR_TOP; z_t = z_b + T_CLAMP
    outer, _ = outline(T_CLAMP, z_b)
    body = outer
    z_foot = T_RIM + (CLR_RIM if cells else 0.0)
    fo, fi = outline(z_b + 0.5 - z_foot, z_foot)
    foot = fo.cut(fi)
    for u, s in cells.items():
        x, y = XY[u]
        foot = foot.cut(cq.Workplane('XY').circle(R_RING_O + 0.3).extrude(seat_h(s) + T_RING + 0.3).translate((x + s['dx'], y + s['dy'], 0)))
    body = body.union(foot)
    for u, s in cells.items():
        x, y = XY[u]; H = seat_h(s); deg, axis = tilt_axis(s)
        pad = cq.Workplane('XY').circle(R_PAD_O).extrude(z_b + 0.5 - (H - 0.5)).translate((x + s['dx'], y + s['dy'], H - 0.5))
        below = cq.Workplane('XY').rect(20, 20).extrude(-10).rotate((0, 0, 0), axis, deg).translate((x, y, H + T_RING))
        body = body.union(pad.cut(below))
    for u, s in cells.items():   # 锥形互补凹: 并入压脚后再挖 (实心压脚, 无薄环壁)
        x, y = XY[u]; H = seat_h(s); deg, axis = tilt_axis(s)
        t30 = math.tan(math.radians(A_CONE)); ext = 0.3
        r0 = R_PAD_I + ext / t30
        cone = cq.Workplane('XY').add(cq.Solid.makeCone(r0, 0.0, r0 * t30, cq.Vector(0, 0, -ext), cq.Vector(0, 0, 1)))
        body = body.cut(cone.rotate((0, 0, 0), axis, deg).translate((x + s['dx'], y + s['dy'], H + T_RING)))
    for (hx, hy) in HOLES:
        body = body.cut(hole_cutter(hx, hy, z_t, -1))
    body = body.union(marker(z_b, T_CLAMP))
    body = engrave(body, 'C' + code, z_t, 2.0, 0.5, 0.0, LAB_XY[0], LAB_XY[1], (4.0, 1.3), LAB_ROT)
    assert len(body.solids().vals()) == 1, f'{code}: clamp is not a single solid'
    return body, z_b, z_t


def clamp_print_pose(body, z_t):
    """打印位姿: 顶面贴平台 (z=0), 压脚朝上."""
    return body.rotate((0, 0, 0), (1, 0, 0), 180).translate((0, 0, z_t))


def plate(code, layout):
    cells = layout['cells']
    body = None
    for u in range(G.NU):
        c = cell_solid(u, cells.get(u))
        body = c if body is None else body.union(c)
    # 围框
    outer, inner = outline(T_RIM)
    body = body.union(outer.cut(inner))
    # 环余隙: 环座平面以上挖掉更高的邻格/围框 (偏移/倾斜环放得进去)
    for u, s in cells.items():
        body = body.cut(ring_clearance(u, s))
    for (hx, hy) in HOLES:
        body = body.cut(hole_cutter(hx, hy, 0.0, +1))
    body = body.union(marker(0.0, T_RIM))
    # 板号: 方向标旁围框顶面, 竖排
    body = engrave(body, code, T_RIM, 1.6, 0.15, CH_LETTER, LAB_XY[0], LAB_XY[1], (3.0, 1.0), LAB_ROT)
    assert len(body.solids().vals()) == 1, f'{code}: plate is not a single solid'
    return body


def export(wp, name):
    exporters.export(wp, os.path.join(OUT, name + '.step'))
    exporters.export(wp, os.path.join(OUT, name + '.stl'), tolerance=0.01, angularTolerance=0.1)
    bb = wp.val().BoundingBox()
    print(f'{name:10s} {bb.xlen:6.2f} x {bb.ylen:6.2f} x {bb.zlen:5.2f} mm')


def write_json(layouts):
    d = {}
    for code, L in layouts.items():
        d[code] = dict(desc=L['desc'], H_blank=H_BLANK,
                       T_rim=T_RIM, T_clamp=T_CLAMP,
                       cells={str(u): dict(gap_mm=s['gap'], tilt_deg=s['tilt'], tilt_dir_deg=s['tdir'], dx_mm=s['dx'], dy_mm=s['dy'],
                                           H_seat_mm=round(s['gap'] - T_RING / 2 - T_MASK, 4)) for u, s in L['cells'].items()})
    d['_rotation_maps'] = {str(k): {str(u): v for u, v in m.items()} for k, m in rotation_maps().items()}
    d['_studs'] = dict(board_refs=['HLOC2', 'HLOC5'], host_xy=STUDS, note='M2 螺钉从板背面穿出; 板取向 k = 逆时针转 60k°, 方向标指向 +x 为 k=0')
    json.dump(d, open(os.path.join(OUT, 'plates.json'), 'w'), ensure_ascii=False, indent=1)
    host_copy = os.path.join(HERE, '..', 'host', 'honeycomb_host', 'data', 'plates_K18.json')   # 主机 GUI 整板模式读取
    json.dump(d, open(host_copy, 'w'), ensure_ascii=False, indent=1)
    rows = ['| 板 | 环数 | 布局 |', '|---|---|---|']
    for code, L in layouts.items():
        rows.append(f'| {code} | {len(L["cells"])} | {L["desc"]} |')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); ap.add_argument('--json-only', action='store_true')
    ap.add_argument('--no-clamp', action='store_true'); ap.add_argument('--clamp-only', action='store_true')
    a = ap.parse_args()
    layouts = build_layouts()
    table = write_json(layouts)
    print(table)
    if a.json_only:
        return
    for code, L in layouts.items():
        if a.only and code not in a.only.split(','):
            continue
        if not a.clamp_only:
            export(plate(code, L), f'plate_{code}')
        if not a.no_clamp:
            c, z_b, z_t = clamp(code, L)
            print(f'  clamp {code}: 体底面 z={z_b:.2f}, 顶面 z={z_t:.2f} (装配坐标, 板底=0)')
            export(clamp_print_pose(c, z_t), f'clamp_{code}')


if __name__ == '__main__':
    main()
