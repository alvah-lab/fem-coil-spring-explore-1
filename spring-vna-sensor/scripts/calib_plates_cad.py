#!/usr/bin/env python3
"""K18 标定整板 (一体打印, N=8, 转位复用): 19 格实心蜂窝 + 围框 + 12 定位通孔 + 板号 + 2 测量座, 每块板一个预先固定的环布局.

用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/calib_plates_cad.py [--only A1,B2]
输出 reports/calib_plates/: plate_<code>.step/.stl, plates.json (布局表, 供主机 GUI/孪生), README.md 由脚本写表格.

几何 (mm)
- 格: 对边 = PITCH 5.2 无缝拼接, 有环格顶面在 H = gap − 0.13 (环底面; 环质心→L1 铜面 = gap), 中心凸柱 Ø2.94×0.2 卡环内孔 + 其上 0.2 高 30° 圆锥导入;
  无环格实心低平台 H_BLANK = 0.8 (无凸柱). 倾斜格顶面绕格心倾斜 (凸柱沿法向), 偏移格凸柱偏移 (dx,dy).
- 围框: 蜂窝外轮廓外扩 WALL, 高 T_RIM; 12 个 Ø2.15 通孔 = HLOC2/HLOC5 两颗 M2 螺钉 (板背面穿出) 的 6 个转位像; +x 侧三角方向标 (取向 k=0); −y 侧刻板号.
- 测量耳: 围框 ±x 外壁向外伸出的 3×3 小耳, 高度 = 该板环座最低 / 最高 H, 上下面外露, 千分尺直接量作整板高度基准.
- 板底 z=0 整面贴 PCB 阻焊面. 打印: 以一个侧沿做支撑 (SLA).
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
H_CONE, A_CONE = 0.2, 30.0        # 凸柱顶部圆锥导入: 高 0.2, 半角 30° (顶径 2.94 − 2·0.2·tan30 = 2.71)
H_BLANK = 0.8
WALL, T_RIM = 3.5, 2.0
CH_LETTER = 0.05
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


def engrave(wp, txt, z_face, size, depth, ch, cx, cy, box):
    t = cq.Workplane('XY').text(txt, size, -depth, combine=False, halign='center', valign='center').translate((cx, cy, z_face))
    wp = wp.cut(t)
    try:
        sel = cq.selectors.BoxSelector((cx - box[0], cy - box[1], z_face - 1e-3), (cx + box[0], cy + box[1], z_face + 1e-3), boundingbox=True)
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
    r_top = R_BOSS - H_CONE * math.tan(math.radians(A_CONE))
    boss = cq.Workplane('XY').circle(R_BOSS).extrude(T_RING)
    cone = (cq.Workplane('XY').workplane(offset=T_RING).circle(R_BOSS).workplane(offset=H_CONE).circle(r_top).loft())
    boss = boss.union(cone).rotate((0, 0, 0), axis, math.degrees(th)).translate((x + spec['dx'], y + spec['dy'], H))
    return body.union(boss)


def plate(code, layout):
    cells = layout['cells']
    body = None
    for u in range(G.NU):
        c = cell_solid(u, cells.get(u))
        body = c if body is None else body.union(c)
    # 围框
    r_out = (PITCH + 2 * WALL) / math.sqrt(3)
    outer = None; inner = None
    for x, y in XY:
        o = hex_prism(T_RIM, r_out, x, y); i = hex_prism(T_RIM, R_CELL, x, y)
        outer = o if outer is None else outer.union(o)
        inner = i if inner is None else inner.union(i)
    rim = outer.cut(inner)
    body = body.union(rim)
    for (hx, hy) in HOLES:
        body = body.cut(cq.Workplane('XY').circle(HOLE_D / 2).extrude(10).translate((hx, hy, -1)))
    # 方向标 (+x 外壁)
    x_out = XY[18][0] + (PITCH + 2 * WALL) / 2
    body = body.union(cq.Workplane('XY').polyline([(-0.3, -0.8), (-0.3, 0.8), (1.0, 0)]).close().extrude(T_RIM).translate((x_out, 0, 0)))
    # 测量耳: 从 ±x 外壁向外伸出 3×3 小耳, 高 = 该板环座最低 / 最高 H (无环时 H_BLANK); 上下两面外露, 千分尺直接量
    Hs = sorted({round(s_['gap'] - T_RING / 2 - T_MASK, 4) for s_ in cells.values()}) or [H_BLANK]
    x_wall = XY[18][0] + (PITCH + 2 * WALL) / 2          # +x 外壁平边
    for sign, Hm, yoff in ((-1, Hs[0], -4.0), (1, Hs[-1], 4.0)):
        tab = cq.Workplane('XY').rect(3.6, 3.0).extrude(Hm).translate((sign * (x_wall + 1.5), yoff, 0))
        body = body.union(tab)
    # 板号: −y 外壁顶面
    y_lab = XY[12][1] - PITCH / 2 - WALL / 2   # 最下一排 (单元 7/12/16) 外壁
    body = engrave(body, code, T_RIM, 1.6, 0.15, CH_LETTER, 0.0, y_lab, (3.0, 1.0))
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
                       cells={str(u): dict(gap_mm=s['gap'], tilt_deg=s['tilt'], tilt_dir_deg=s['tdir'], dx_mm=s['dx'], dy_mm=s['dy'],
                                           H_seat_mm=round(s['gap'] - T_RING / 2 - T_MASK, 4)) for u, s in L['cells'].items()})
    d['_rotation_maps'] = {str(k): {str(u): v for u, v in m.items()} for k, m in rotation_maps().items()}
    d['_studs'] = dict(board_refs=['HLOC2', 'HLOC5'], host_xy=STUDS, note='M2 螺钉从板背面穿出; 板取向 k = 逆时针转 60k°, 方向标指向 +x 为 k=0')
    json.dump(d, open(os.path.join(OUT, 'plates.json'), 'w'), ensure_ascii=False, indent=1)
    rows = ['| 板 | 环数 | 布局 |', '|---|---|---|']
    for code, L in layouts.items():
        rows.append(f'| {code} | {len(L["cells"])} | {L["desc"]} |')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); ap.add_argument('--json-only', action='store_true')
    a = ap.parse_args()
    layouts = build_layouts()
    table = write_json(layouts)
    print(table)
    if a.json_only:
        return
    for code, L in layouts.items():
        if a.only and code not in a.only.split(','):
            continue
        export(plate(code, L), f'plate_{code}')


if __name__ == '__main__':
    main()
