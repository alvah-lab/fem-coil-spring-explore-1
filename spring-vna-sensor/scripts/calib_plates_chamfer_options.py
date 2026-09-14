#!/usr/bin/env python3
"""A0 板三种台阶倒角做法对比 (只出 A0): 1 OCC 原生边倒角; 2 按闭合环放样倒角带; 3 无环格抬到 2.0 (A0 无台阶).
输出 reports/calib_plates/compare_A0/plate_A0_opt{1,2,3}.stl/.step + 轴测/局部图. 在 spring-vna-sensor/ 下用 coil-5 .venv 运行."""
import os, sys, math, subprocess
sys.path.insert(0, 'scripts'); import calib_plates_cad as P
import cadquery as cq
from cadquery import exporters
from PIL import Image
OUT = os.path.join(P.OUT, 'compare_A0'); os.makedirs(OUT, exist_ok=True)
C = 0.6   # A0: 围框 2.0 vs 平台 0.8, 台阶 1.2, 两侧各 0.6 (方案 2)
C1 = 0.5  # 方案 1 (OCC): 顶 0.6 + 根 0.5, 留 0.1 竖壁 (0.6+0.6 把竖壁吃光, OCC 报错)


def base_a0(h_blank=P.H_BLANK):
    body = None
    for u in range(P.G.NU):
        c = P.hex_prism(h_blank, P.R_CELL, *P.XY[u]); body = c if body is None else body.union(c)
    outer, inner = P.outline(P.T_RIM)
    return body.union(outer.cut(inner))


def finish(body, code='A0'):
    for (hx, hy) in P.HOLES:
        body = body.cut(P.hole_cutter(hx, hy, 0.0, P.T_RIM))
    body = body.union(P.marker(0.0, P.T_RIM))
    body = P.engrave(body, code, P.T_RIM, 1.6, 0.15, P.CH_LETTER, P.LAB_XY[0], P.LAB_XY[1], (3.0, 1.0), P.LAB_ROT)
    assert len(body.solids().vals()) == 1
    return body


class OnInnerOutline(cq.Selector):
    """选中点在内围框轮廓线上、且在指定高度的边."""
    def __init__(self, z, tol=0.03):
        o1, i1 = P.outline(1.0, -0.5, inner_margin=+tol); o2, i2 = P.outline(1.0, -0.5, inner_margin=-tol)
        self.outer_side, self.inner_side, self.z = i1.val(), i2.val(), z
    def filter(self, objs):
        out = []
        for e in objs:
            c = e.Center()
            if abs(c.z - self.z) > 1e-3: continue
            pts = [e.startPoint(), e.endPoint(), c]
            if all(abs(p.z - self.z) < 1e-3 for p in pts) and all(self.outer_side.isInside(cq.Vector(p.x, p.y, 0)) and not self.inner_side.isInside(cq.Vector(p.x, p.y, 0)) for p in pts):
                out.append(e)
        return out


def opt1():
    """OCC 原生: 围框顶内沿 (凸, z=2.0) 与平台根部 (凹, z=0.8) 两圈边各倒 0.6."""
    body = base_a0()
    n_top = len(body.edges(OnInnerOutline(P.T_RIM)).vals()); n_bot = len(body.edges(OnInnerOutline(P.H_BLANK)).vals())
    print(f'opt1: 选中顶边 {n_top} 条, 根部边 {n_bot} 条')
    body = body.edges(OnInnerOutline(P.T_RIM)).chamfer(C)
    # 凹边 OCC 不能直接填: 对"空气"实体 (盒 − 板) 做凸边倒角, 再用盒减回去
    box = cq.Workplane('XY').box(60, 60, 6, centered=(True, True, False)).translate((0, 0, -1))
    air = box.cut(body)
    n = len(air.edges(OnInnerOutline(P.H_BLANK)).vals()); print(f'opt1: 空气实体根部边 {n} 条')
    air = air.edges(OnInnerOutline(P.H_BLANK)).chamfer(C1)   # 两侧倒角之和必须 < 台阶高, 否则竖壁消失 OCC 失败
    body = box.cut(air)
    print('opt1: 凹边经空气实体倒角成功')
    return finish(body)


def inner_wire(z):
    """内围框轮廓 (19 格并集) 的外轮廓线, 放在高度 z."""
    _, inner = P.outline(1.0)
    f = inner.faces('>Z').val()
    return f.outerWire().translate(cq.Vector(0, 0, z - 1.0))


def loft_band(w_lo, w_hi):
    return cq.Workplane('XY').add(w_lo).add(w_hi).toPending().loft(ruled=True)


def opt2():
    """放样倒角带: 凸带 = loft(轮廓@2.0−c, 轮廓外扩c@2.0) 切; 凹带 = loft(轮廓内缩c@0.8, 轮廓外扩0.05@0.8+c+0.05) − 内缩c柱 加."""
    body = base_a0()
    w0 = inner_wire(0.0)
    def off(d, z):
        w = w0.offset2D(d, 'intersection')[0] if abs(d) > 1e-9 else w0
        return w.translate(cq.Vector(0, 0, z))
    cut = loft_band(off(0.0, P.T_RIM - C), off(C, P.T_RIM))
    cap = cq.Workplane('XY').add(off(C, P.T_RIM - 0.01)).toPending().extrude(1.0)
    body = body.cut(cut.union(cap))
    # 根部带: 在高度 z 占据 offset ∈ [−(C − (z − 0.8)), 0], 底宽 C、到墙处收零 → 墙内柱体 − 放样体
    slant = loft_band(off(-C, P.H_BLANK), off(0.05, P.H_BLANK + C + 0.05))
    wall_prism = cq.Workplane('XY').add(off(0.05, P.H_BLANK - 0.05)).toPending().extrude(C + 0.05)
    band = wall_prism.cut(slant)
    body = body.union(band)
    print('opt2: 放样带 ok')
    return finish(body)


def opt3():
    """无环格抬到 2.0 = 与围框齐平: A0 没有台阶."""
    return finish(base_a0(h_blank=P.T_RIM))


def render(wp, name, crop=(0.30, 0.10, 0.75, 0.45)):
    svg = os.path.join(OUT, name + '.svg'); png = os.path.join(OUT, name + '.png')
    exporters.export(wp, svg, exporters.ExportTypes.SVG, opt=dict(width=2400, height=2000, marginLeft=5, marginTop=5, showAxes=False, projectionDir=(1, -1, 1.2), strokeWidth=0.08, showHidden=False))
    subprocess.run(['inkscape', svg, '-o', png, '-w', '2400', '-b', 'white'], check=True, capture_output=True)
    im = Image.open(png); W, H = im.size
    im.crop((int(W * crop[0]), int(H * crop[1]), int(W * crop[2]), int(H * crop[3]))).save(os.path.join(OUT, name + '_corner.png'))
    os.remove(svg)


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else '123'
    for k, fn in ((1, opt1), (2, opt2), (3, opt3)):
        if str(k) not in only: continue
        try:
            b = fn()
        except Exception as e:
            print(f'opt{k} 失败: {e}'); continue
        exporters.export(b, os.path.join(OUT, f'plate_A0_opt{k}.step')); exporters.export(b, os.path.join(OUT, f'plate_A0_opt{k}.stl'), tolerance=0.01, angularTolerance=0.1)
        bad = [f for f in b.faces().vals() if f.normalAt().z < -0.72 and f.Center().z > 0.05]
        print(f'opt{k}: 面数 {len(b.faces().vals())}, 朝下>45° 面 {len(bad)}')
        render(b, f'view_A0_opt{k}')
