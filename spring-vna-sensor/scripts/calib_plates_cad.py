#!/usr/bin/env python3
"""K18 标定整板 (一体打印, N=8, 转位复用): 19 格实心蜂窝 + 围框 + 12 定位通孔 + 板号, 每块板一个预先固定的环布局.

用 coil-5 的 .venv 跑:  /work/alvah-labs/spiral-coil/coil-5/.venv/bin/python3 scripts/calib_plates_cad.py [--only A1,B2]
输出 reports/calib_plates/: plate_<code>.step/.stl, plates.json (布局表, 供主机 GUI/孪生), README.md 由脚本写表格.

几何 (mm)
- 格: 对边 = PITCH 5.2 无缝拼接, 有环格顶面在 H = gap − 0.13 (环底面; 环质心→L1 铜面 = gap), 中心凸柱 Ø2.94×0.2 卡环内孔 + 其上完整扁圆锥 (母线 30°, 高 0.85, 收尖) 导入;
  无环格实心低平台 H_BLANK = 2.0                     # 无环格与围框齐平 (方案 3): 台阶只出现在有环格周围 (无凸柱). 倾斜格顶面绕格心倾斜 (凸柱沿法向), 偏移格凸柱偏移 (dx,dy).
- 围框: 蜂窝外轮廓外扩 WALL, 高 T_RIM; 12 个 Ø2.15 通孔 = HLOC2/HLOC5 两颗 M2 螺钉 (板背面穿出) 的 6 个转位像; +x 侧三角方向标 (取向 k=0); 板号刻在 +x 方向标旁的围框顶面 (竖排, 两孔之间).
- 无环格与围框齐平 2.0 (平台), 台阶只在有环格周围. 每个等高区域的每圈台阶边界做等宽 45° 带 (拐角斜接): 高侧切顶角, 低侧是平台时加根部 (两侧各 Δz/2), 低侧是有环格时只切高侧 (≤Δz);
  带宽 ≤0.4 (有环格环座保平到 r≥2.2, 压脚接触带 Ø3.5–Ø4.4; 平台顶角不进定位孔锥口), 每圈取最小值以保证斜接精确. 倾斜格的脚线随倾斜面.
- 定位孔 Ø2.2, 两面锥口 Ø2.8 各深 40% 厚度, 中间 20% 直段.
- 板底 z=0 整面贴 PCB 阻焊面. 每个有环格在环座平面以上切一圈 Ø(5.0+0.4) 的余隙 (穿过更高的邻格/围框), 保证偏移/倾斜的环也放得进去.
- 压板 (clamp_<code>): 与该板互补, 同外形、同 12 孔; 每个有环格一个 Ø3.5/Ø4.8 环形压脚落在环顶面 (倾斜格压脚同样倾斜, 偏移格压脚同样偏移),
  压脚实心, 端面中心为与凸柱锥互补的 30° 锥形凹 (Ø3.5, 深 1.01); 围框脚比围框顶高 0.1 (先压环, 再顶围框). 板厚 4.0. 顶面刻 C+板号, +x 三角方向标.
- 打印 (SLA/MSLA): 板 = 平放, 板底直接贴平台 (无支撑, 顶面全部朝上, 无悬空, 孔垂直); 压板 = 顶面贴平台, 压脚朝上. 
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
H_BLANK = 2.0                     # 无环格与围框齐平 (方案 3): 台阶只出现在有环格周围
WALL, T_RIM = 3.5, 2.0
CH_LETTER = 0.05
CLR_RING = 0.2                    # 环外沿到任何更高结构 (邻格/围框) 的余隙
HOLE_MOUTH_D, HOLE_LAND = 2.5, 0.2   # 定位孔: 两面都开深锥口 (各 40% 厚度), 中间 20% 厚度留 Ø2.2 直段; 锥口 Ø2.5 (孔心离围框内沿 1.75, 平台顶倒角 ≤0.4 不进锥口)
CAP_BAND = 0.4                    # 台阶倒角带宽上限: 有环格顶角保环座平到 r≥2.2; 平台顶角不进定位孔锥口
T_CLAMP, CLR_TOP, CLR_RIM = 4.0, 0.5, 0.1   # 压板厚; 压板体底面离最高环顶/围框顶; 围框脚离围框顶
R_PAD_O, R_PAD_I = 2.2, 1.75              # 压脚: 实心 Ø4.4 圆柱 (接触环带 Ø3.5–Ø4.4, 与倒角后环座 r≤2.2 的平面对应), 端面中心开与凸柱锥互补的 30° 锥形凹 (面上 Ø3.5, 深 1.01; 凸柱锥 Ø2.94 高 0.85, 径向余隙 0.28, 顶余隙 0.16)
# 定位: 板背面从 HLOC2 (60.02,27.08) 与 HLOC5 (59.47,52.57) 伸出两颗 M2 螺钉 (相隔 160°), 整板绕阵列中心按 60° 转位复用;
# 板上开这两点各自 6 个旋转像共 12 个通孔 (Ø2.15, 相互 ≥4.5 mm). 六个 HLOC 本身不是 60° 对称, 不能六颗都用.
STUDS = [(60.02 - 62.0, -(27.08 - 40.0)), (59.47 - 62.0, -(52.57 - 40.0))]
HOLE_D = 2.2
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


def hole_cutter(hx, hy, z0, T):
    """定位孔: Ø2.2 通孔 + 两面锥口 (各深 0.4T, 口径 HOLE_MOUTH_D), 中间 0.2T 直段."""
    c = cq.Workplane('XY').circle(HOLE_D / 2).extrude(30).translate((hx, hy, -15))
    d = (1 - HOLE_LAND) / 2 * T
    lo = cq.Solid.makeCone(HOLE_MOUTH_D / 2, HOLE_D / 2, d, cq.Vector(hx, hy, z0), cq.Vector(0, 0, 1))
    hi = cq.Solid.makeCone(HOLE_MOUTH_D / 2, HOLE_D / 2, d, cq.Vector(hx, hy, z0 + T), cq.Vector(0, 0, -1))
    return c.union(cq.Workplane('XY').add(lo)).union(cq.Workplane('XY').add(hi))


def polyhedron(faces):
    """由平面多边形面 (点列表) 围成的实体."""
    fs = [cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*p) for p in f], close=True)) for f in faces]
    sol = cq.Solid.makeSolid(cq.Shell.makeShell(fs)).fix()
    if sol.Volume() < 0:
        sol = cq.Solid.makeSolid(cq.Shell.makeShell([f.reversed() if hasattr(f, 'reversed') else f for f in fs])).fix()
    return cq.Workplane('XY').add(sol)


def seat_plane_z(u, cells, x, y):
    """格 u 顶面在 (x,y) 处的高度 (倾斜格按倾斜面; 无环格 = H_BLANK)."""
    if u not in cells:
        return H_BLANK
    s = cells[u]; H = seat_h(s); th = math.radians(s['tilt']); ph = math.radians(s['tdir'])
    if th == 0:
        return H
    # 顶面 = 过 (cx,cy,H) 的平面, 绕轴 (-sinφ, cosφ) 转 θ (与 cell_solid 同约定): 梯度沿 -(cosφ, sinφ)·tanθ 或 +, 用旋转求
    axis = np.array([-math.sin(ph), math.cos(ph), 0.0]); v = np.array([x - XY[u][0], y - XY[u][1], 0.0])
    vr = v * math.cos(th) + np.cross(axis, v) * math.sin(th) + axis * np.dot(axis, v) * (1 - math.cos(th))
    return H + vr[2] * 1.0 / max(math.cos(th), 1e-9) * math.cos(th)   # 小角近似: 取旋转后 z


FLAT_DIRS = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))) for k in range(6)]
RIM = -1   # 围框伪单元


def cell_top(u, cells):
    return H_BLANK if u == RIM or u not in cells else seat_h(cells[u])


def neighbor_in_dir(u, d):
    for v in range(G.NU):
        if np.hypot(XY[v][0] - XY[u][0] - PITCH * d[0], XY[v][1] - XY[u][1] - PITCH * d[1]) < 1e-3:
            return v
    return RIM


def edge_endpoints(u, d):
    """格 u 在方向 d 的那条边的两个端点 (六边形顶点), 按绕 u 逆时针顺序."""
    x, y = XY[u]; a = math.atan2(d[1], d[0])
    p1 = (x + R_CELL * math.cos(a - math.radians(30)), y + R_CELL * math.sin(a - math.radians(30)))
    p2 = (x + R_CELL * math.cos(a + math.radians(30)), y + R_CELL * math.sin(a + math.radians(30)))
    return p1, p2


def regions(cells):
    """等高连通区域: 有环格按环座高连通 (相邻且高度差<0.05 且都有环), 无环格 + 围框 = 平台 'P'."""
    lab = {}
    for u in range(G.NU):
        lab[u] = 'P' if u not in cells else None
    for u in range(G.NU):
        if lab[u] is None:
            lab[u] = f'R{u}'; stack = [u]
            while stack:
                a = stack.pop()
                for b in G.ADJ[a]:
                    if lab[b] is None and abs(cell_top(a, cells) - cell_top(b, cells)) < 0.05 and cells[a]['tilt'] == 0 and cells[b]['tilt'] == 0:
                        lab[b] = lab[u]; stack.append(b)
    lab[RIM] = 'P'
    return lab


def step_chamfers(cells):
    """方案 2: 对每个等高区域的每一圈台阶边界做等宽 45° 带 (拐角斜接): 高侧切顶角; 低侧是平台时加根部.
    带宽 = 该圈所有边允许值的最小 (有环格顶角 ≤0.4; 面向平台的台阶两侧各 ≤Δz/2; 面向有环格的台阶 ≤Δz). 返回 (cuts, adds)."""
    lab = regions(cells)
    edges = []   # 每条台阶边: 高侧区域, 低侧, 端点 (按高侧逆时针), 法向 (低→高), 各端顶高/脚高
    seen = set()
    for u in range(G.NU):
        for d in FLAT_DIRS:
            v = neighbor_in_dir(u, d)
            key = tuple(sorted((u, v))) + (round(d[0] * (1 if u < v or v == RIM else -1), 3), round(d[1] * (1 if u < v or v == RIM else -1), 3)) if v != RIM else (u, RIM, round(d[0], 3), round(d[1], 3))
            if key in seen:
                continue
            seen.add(key)
            p1, p2 = edge_endpoints(u, d)
            zu = [seat_plane_z(u, cells, *p1), seat_plane_z(u, cells, *p2)]
            zv = [seat_plane_z(v, cells, *p1), seat_plane_z(v, cells, *p2)] if v != RIM else [H_BLANK, H_BLANK]
            if max(zu) - min(zv) < 0.05 and max(zv) - min(zu) < 0.05:
                continue
            # 高侧 = 平均更高者
            if np.mean(zu) >= np.mean(zv):
                hi, lo, n, top, foot = u, v, (-d[0], -d[1]), zu, zv          # n: 低→高 = 从 v 指向 u
                pts = (p1, p2)
            else:
                hi, lo, n, top, foot = v, u, (d[0], d[1]), zv, zu
                pts = (p2, p1)
            # 端点顺序: 使高侧在左手 (沿 pts[0]->pts[1] 走, n 在左侧) —— 用于统一斜接方向
            t = np.array(pts[1]) - np.array(pts[0]); left = (-t[1], t[0])
            if left[0] * n[0] + left[1] * n[1] < 0:
                pts = (pts[1], pts[0]); top = top[::-1]; foot = foot[::-1]
            dz = min(np.array(top) - np.array(foot))
            ramp = (lo != RIM and lo in cells and cells[lo]['tilt'] != 0)   # 低侧倾斜格: 从脚线起坡到顶 (坡度随墙高变化), 带宽不受墙高限制
            allowed = CAP_BAND if ramp else min(CAP_BAND, dz / 2 if lab[lo] == 'P' else dz)
            edges.append(dict(hi=lab[hi], lo=lab[lo], lo_is_P=(lab[lo] == 'P'), p=pts, n=n, top=top, foot=foot, w=allowed, dz=dz, ramp=ramp))
    cuts, adds = [], []
    # 按高侧区域分组, 连成链/环 (端点匹配)
    from collections import defaultdict
    by_hi = defaultdict(list)
    for e in edges:
        by_hi[e['hi']].append(e)
    def key(p):
        return (round(p[0], 3), round(p[1], 3))
    for R, es in by_hi.items():
        # 链接: 边 i 的终点 == 边 j 的起点 (高侧在左手 => 沿边界逆时针绕高侧区域)
        start = {key(e['p'][0]): e for e in es}
        end = {key(e['p'][1]): e for e in es}
        used = set(); chains = []
        for e in es:
            if id(e) in used:
                continue
            # 回溯到链头
            head = e
            while key(head['p'][0]) in end and id(end[key(head['p'][0])]) != id(head):
                prev = end[key(head['p'][0])]
                if prev is e:
                    break
                head = prev
                if id(head) == id(e):
                    break
            chain = [head]; used.add(id(head)); cur = head
            while key(cur['p'][1]) in start:
                nxt = start[key(cur['p'][1])]
                if id(nxt) in used:
                    break
                chain.append(nxt); used.add(id(nxt)); cur = nxt
            closed = key(chain[-1]['p'][1]) == key(chain[0]['p'][0])
            chains.append((chain, closed))
        for chain, closed in chains:
            w = min(e['w'] for e in chain)
            if w < 0.03:
                continue
            m = len(chain)
            # 每条边两端的斜接偏移点 (高侧, 偏移 w) 与 低侧 (偏移 w) 
            def miter(e_prev, e_next, P, w, side):
                """P 处偏移点: side=+1 高侧 (沿 n), -1 低侧 (沿 -n)."""
                n1 = np.array(e_prev['n']) * side if e_prev is not None else None
                n2 = np.array(e_next['n']) * side if e_next is not None else None
                if n1 is None or n2 is None:
                    nn = n1 if n2 is None else n2
                    return np.array(P) + w * nn
                dot = float(np.dot(n1, n2))
                return np.array(P) + w * (n1 + n2) / max(1 + dot, 0.2)
            for i, e in enumerate(chain):
                prv = chain[i - 1] if (i > 0 or closed) else None
                nxt = chain[(i + 1) % m] if (i < m - 1 or closed) else None
                P1, P2 = e['p']
                T1 = miter(prv, e, P1, w, +1); T2 = miter(e, nxt, P2, w, +1)
                zt1, zt2 = e['top']; zf1, zf2 = e['foot']
                # 切带: 脚线高 = max(脚, 顶-w) (墙高于 w 时是顶角 45°带, 否则从脚起坡)
                zb1, zb2 = (zf1, zf2) if e['ramp'] else (max(zf1, zt1 - w), max(zf2, zt2 - w))
                eps = 0.03
                F1 = (P1[0] - e['n'][0] * eps, P1[1] - e['n'][1] * eps, zb1 - eps); F2 = (P2[0] - e['n'][0] * eps, P2[1] - e['n'][1] * eps, zb2 - eps)
                Tt1 = (T1[0], T1[1], zt1 + (zb1 - zt1) * 0 + eps * 0); Tt2 = (T2[0], T2[1], zt2)
                # 顶点在 T 处的高度 = 高侧面高 (倾斜高侧时沿面), 用顶高近似
                U1 = (T1[0], T1[1], zt1 + 1.0); U2 = (T2[0], T2[1], zt2 + 1.0)
                W1 = (F1[0], F1[1], zt1 + 1.0); W2 = (F2[0], F2[1], zt2 + 1.0)
                faces = [[F1, F2, Tt2, Tt1], [F1, W1, W2, F2], [Tt1, Tt2, U2, U1], [W1, U1, U2, W2], [F1, Tt1, U1, W1], [F2, W2, U2, Tt2]]
                cuts.append(polyhedron(_planarize(faces)))
                if e['lo_is_P']:
                    # 根部加料 (低侧平台): 三棱柱, 墙上高 w, 地上宽 w
                    G1 = miter(prv, e, P1, w, -1); G2 = miter(e, nxt, P2, w, -1)
                    zl1, zl2 = zf1, zf2
                    A1 = (P1[0] + e['n'][0] * eps, P1[1] + e['n'][1] * eps, zl1 - eps); A2 = (P2[0] + e['n'][0] * eps, P2[1] + e['n'][1] * eps, zl2 - eps)
                    B1 = (A1[0], A1[1], zl1 + w + eps); B2 = (A2[0], A2[1], zl2 + w + eps)
                    C1 = (G1[0], G1[1], zl1 - eps); C2 = (G2[0], G2[1], zl2 - eps)
                    faces = [[A1, C1, B1], [A2, B2, C2], [A1, A2, C2, C1], [A1, B1, B2, A2], [B1, C1, C2, B2]]
                    adds.append(polyhedron(_planarize(faces)))
    return cuts, adds


def _planarize(faces):
    """四边形若不共面 (倾斜脚线) 拆成两个三角形."""
    out = []
    for f in faces:
        if len(f) == 4:
            p = np.array(f); nrm = np.cross(p[1] - p[0], p[2] - p[0])
            if abs(np.dot(nrm / (np.linalg.norm(nrm) + 1e-12), p[3] - p[0])) > 1e-4:
                out.append([f[0], f[1], f[2]]); out.append([f[0], f[2], f[3]]); continue
        out.append(f)
    return out


def outline(h, z0=0.0, inner_margin=0.0):
    """围框外轮廓 (19 个外扩六边形的并) 与内轮廓 (19 格, 可外扩 inner_margin) 的柱体, 高 h, 底 z0."""
    r_out = (PITCH + 2 * WALL) / math.sqrt(3)
    r_in = R_CELL + inner_margin / math.cos(math.radians(30))
    outer = None; inner = None
    for x, y in XY:
        o = hex_prism(h, r_out, x, y); i = hex_prism(h, r_in, x, y)
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
    fo, fi = outline(z_b + 0.5 - z_foot, z_foot, inner_margin=CAP_BAND + 0.2)   # 围框脚只落在围框外侧, 避开根部倒角
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
        body = body.cut(hole_cutter(hx, hy, z_foot, z_t - z_foot))   # 锥口从围框脚底面到顶面
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
    # 台阶倒角带 (方案 2): 每圈台阶等宽 45° 带, 拐角斜接; 高侧切顶角, 低侧为平台时加根部 (根部先减去所有切带, 避免在拐角处重新填回)
    cuts, adds = step_chamfers(cells)
    for c in cuts:
        body = body.cut(c)
    for a in adds:
        for c in cuts:
            a = a.cut(c)
        body = body.union(a)
    # 环余隙: 环座平面以上挖掉更高的邻格/围框 (偏移/倾斜环放得进去; 也清掉环附近的根部倒角)
    for u, s in cells.items():
        body = body.cut(ring_clearance(u, s))
    for (hx, hy) in HOLES:
        body = body.cut(hole_cutter(hx, hy, 0.0, T_RIM))
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
