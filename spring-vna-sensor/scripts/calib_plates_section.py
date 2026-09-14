"""剖面图 (matplotlib): 板 + 铜环 + 压板. 在 spring-vna-sensor/ 下运行: coil-5 .venv python3 scripts/calib_plates_section.py <板号> <单元号|tilt>"""
import sys, os, math, numpy as np
sys.path.insert(0, 'scripts'); import calib_plates_cad as P, cadquery as cq
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    try: fm.fontManager.addfont(_f)
    except Exception as _e: print('font', _e)
plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
def rings_for(lay):
    body = None
    for u, s in lay['cells'].items():
        x, y = P.XY[u]; deg, axis = P.tilt_axis(s)
        r = cq.Workplane('XY').circle(P.R_RING_O).circle(P.R_RING_I).extrude(P.T_RING).rotate((0, 0, 0), axis, deg).translate((x + s['dx'], y + s['dy'], P.seat_h(s)))
        body = r if body is None else body.union(r)
    return body
def cut_polys(wp, normal, off):
    """返回剖切面上的边采样 (u,v) 列表. normal: 'y' 或 'z'."""
    if normal == 'y':
        half = cq.Workplane('XY').box(200, 200, 200, centered=(True, False, True)).translate((0, off, 0))
    else:
        half = cq.Workplane('XY').box(200, 200, 200, centered=(True, True, False)).translate((0, 0, off))
    sec = wp.cut(half)
    out = []
    for f in sec.faces().vals():
        n = f.normalAt(); c = f.Center()
        ok = (abs(n.y) > 0.99 and abs(c.y - off) < 1e-3) if normal == 'y' else (abs(n.z) > 0.99 and abs(c.z - off) < 1e-3)
        if not ok: continue
        for e in f.Edges():
            pts = e.positions(np.linspace(0, 1, 24).tolist(), mode='length')
            out.append([(p.x, p.z) if normal == 'y' else (p.x, p.y) for p in pts])
    return out
code = sys.argv[1]
L = P.build_layouts(); lay = L[code]
u_sec = int(sys.argv[2]) if sys.argv[2] != 'tilt' else next(u for u, s in lay['cells'].items() if s['tilt'] > 0)
pl = P.plate(code, lay); cl, z_b, z_t = P.clamp(code, lay); rg = rings_for(lay)
ysec = P.XY[u_sec][1]
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
for wp, col, lab in ((pl, 'k', '板'), (cl, 'tab:blue', '压板'), (rg, 'tab:red', '铜环')):
    if wp is None: continue
    for poly in cut_polys(wp, 'y', ysec):
        xs, zs = zip(*poly); ax[0].plot(xs, zs, color=col, lw=1)
    ax[0].plot([], [], color=col, label=lab)
ax[0].set_aspect('equal'); ax[0].grid(alpha=.3); ax[0].legend(loc='upper right')
ax[0].set_title(f'{code}: 过单元 {u_sec} 中心 (y={ysec:.2f}) 的竖直剖面, 装配位姿 (板底 z=0)'); ax[0].set_xlabel('x (mm)'); ax[0].set_ylabel('z (mm)')
# 水平切片: 压板 (装配位姿) 在环顶面 +0.3 处
z_slice = z_b - 0.05
for poly in cut_polys(cl, 'z', z_slice):
    xs, ys = zip(*poly); ax[1].plot(xs, ys, color='tab:blue', lw=0.8)
for poly in cut_polys(pl, 'z', 0.5):
    xs, ys = zip(*poly); ax[1].plot(xs, ys, color='k', lw=0.4, alpha=0.5)
ax[1].set_aspect('equal'); ax[1].grid(alpha=.3); ax[1].set_title(f'压板在 z={z_slice:.2f} 的水平切片 (蓝: 压脚/围框脚) + 板 z=0.5 切片 (黑)')
fig.tight_layout(); out = os.path.join(P.OUT, f'view_{code}_section_u{u_sec}.png'); fig.savefig(out, dpi=110); print(out, 'z_b', z_b, 'z_t', z_t)
