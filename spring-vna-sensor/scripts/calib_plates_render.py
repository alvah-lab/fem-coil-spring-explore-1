"""轴测线图: 板 (装配位姿) 与压板 (打印位姿). 在 spring-vna-sensor/ 下运行: coil-5 .venv python3 scripts/calib_plates_render.py A0,A1C,...  [单元号: 另出 SVG 剖面]"""
import sys, os, math, subprocess
sys.path.insert(0, 'scripts')
import calib_plates_cad as P
import cadquery as cq
from cadquery import exporters
OUT = P.OUT
def svg_png(wp, name, proj=(1, -1, 0.8), w=900):
    svg = os.path.join(OUT, name + '.svg'); png = os.path.join(OUT, name + '.png')
    exporters.export(wp, svg, exporters.ExportTypes.SVG, opt=dict(width=w, height=int(w * 0.8), marginLeft=8, marginTop=8, showAxes=False, projectionDir=proj, strokeWidth=0.2, strokeColor=(0, 0, 0), hiddenColor=(200, 200, 200), showHidden=False))
    subprocess.run(['inkscape', svg, '-o', png, '-w', str(w), '-b', 'white'], check=True, capture_output=True)
    return png
def rings_for(layout):
    body = None
    for u, s in layout['cells'].items():
        x, y = P.XY[u]; deg, axis = P.tilt_axis(s)
        r = cq.Workplane('XY').circle(P.R_RING_O).circle(P.R_RING_I).extrude(P.T_RING).rotate((0, 0, 0), axis, deg).translate((x + s['dx'], y + s['dy'], P.seat_h(s)))
        body = r if body is None else body.union(r)
    return body
codes = sys.argv[1].split(',')
L = P.build_layouts()
for code in codes:
    lay = L[code]
    pl = P.plate(code, lay)
    cl, z_b, z_t = P.clamp(code, lay)
    svg_png(pl, f'view_{code}_iso')
    svg_png(P.clamp_print_pose(cl, z_t), f'view_clamp_{code}_print')
    # 装配剖面: 过指定单元中心的竖直面 (法向 y), 看 −y 方向
    if len(sys.argv) > 2:
        u_sec = int(sys.argv[2]); ysec = P.XY[u_sec][1]
        asm = pl.union(cl)
        rg = rings_for(lay)
        if rg is not None: asm = asm.union(rg)
        half = cq.Workplane('XY').box(80, 80, 40, centered=(True, False, False)).translate((0, ysec, -10))
        sec = asm.cut(half)
        svg_png(sec, f'view_{code}_section_u{u_sec}', proj=(0, -1, 0), w=1400)
        print('section at y =', ysec)
    print(code, 'done')
