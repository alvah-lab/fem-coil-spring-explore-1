"""3D 场形变视图: 表面 w 场网格 (由 19 单元 (x+u, y+v, w) 插值, z 放大) + 19 个铜环 (5-DOF 位姿, 含倾斜) + PCB 线圈 (标称/下沉).
优先 pyqtgraph.opengl (GLViewWidget); 无 OpenGL 时退化为 2D 等轴投影."""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QCheckBox
from PyQt6.QtCore import Qt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from .. import geometry as G
from ..fastmodel import rot_matrix

try:
    import pyqtgraph.opengl as gl
    HAVE_GL = True
except Exception:       # noqa
    HAVE_GL = False

_ang = np.linspace(0, 2 * np.pi, 49)
_circ = np.stack([2.0 * np.cos(_ang), 2.0 * np.sin(_ang), np.zeros_like(_ang)], 1)   # 环中径 r=2
_coil = np.stack([2.45 * np.cos(_ang), 2.45 * np.sin(_ang), np.zeros_like(_ang)], 1)

class View3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout(); lay.addLayout(top)
        top.addWidget(QLabel('z 放大:')); self.zs = QSlider(Qt.Orientation.Horizontal); self.zs.setRange(1, 50); self.zs.setValue(10); top.addWidget(self.zs)
        self.zlbl = QLabel('×10'); top.addWidget(self.zlbl); self.zs.valueChanged.connect(lambda v: self.zlbl.setText(f'×{v}'))
        top.addWidget(QLabel('着色:')); self.cmode = QComboBox(); self.cmode.addItems(['w', 'ΔT 环', '|u,v|']); top.addWidget(self.cmode)
        self.show_truth = QCheckBox('叠加真值 (孪生)'); top.addWidget(self.show_truth); top.addStretch()
        self.gap = G.GAP_REST
        self.cmap = pg.colormap.get('viridis')
        # 插值网格 (六边形阵列凸包内)
        r = G.PITCH * 2.9
        gx, gy = np.meshgrid(np.linspace(-r, r, 45), np.linspace(-r, r, 45))
        self.gx, self.gy = gx, gy
        self.inside = np.hypot(gx, gy) < G.PITCH * 2.3 + G.PITCH * 0.5
        # 一次性重心插值权重: 19 单元 + 24 个固定边界零点 (泡棉边缘) 的 Delaunay 三角剖分
        rb = G.PITCH * 2.3 + G.PITCH * 0.5
        a = np.linspace(0, 2 * np.pi, 25)[:-1]
        pts = np.vstack([G.XY, np.stack([rb * np.cos(a), rb * np.sin(a)], 1)])
        tri = Delaunay(pts)
        gp = np.stack([gx.ravel(), gy.ravel()], 1)
        simp = tri.find_simplex(gp)
        W = np.zeros((len(gp), len(pts)))
        ok = simp >= 0
        T = tri.transform[simp[ok]]
        b = np.einsum('ijk,ik->ij', T[:, :2], gp[ok] - T[:, 2])
        bary = np.hstack([b, 1 - b.sum(1, keepdims=True)])
        for k in range(3):
            W[np.where(ok)[0], tri.simplices[simp[ok], k]] += bary[:, k]
        self.W = W[:, :G.NU]                     # 边界点值恒 0, 只保留 19 列
        self.inside &= ok.reshape(gx.shape)
        # 单元着色的最近邻索引
        d = np.hypot(gx.ravel()[:, None] - G.XY[None, :, 0], gy.ravel()[:, None] - G.XY[None, :, 1])
        self.nearest = np.argmin(d, axis=1).reshape(gx.shape)
        self.gl_ok = False
        if HAVE_GL:
            try:
                self._init_gl(lay)
                self.gl_ok = True
            except Exception as e:      # 无 GL 上下文等
                print('view3d: GL 不可用, 退化 2D 等轴投影:', e)
        if not self.gl_ok:
            self._init_2d(lay)

    # ---------- GL ----------
    def _init_gl(self, lay):
        self.view = gl.GLViewWidget(); lay.addWidget(self.view)
        self.view.setBackgroundColor('w')
        self.view.setCameraPosition(distance=45, elevation=28, azimuth=-60)
        grid = gl.GLGridItem(); grid.setSize(30, 30); grid.setSpacing(5.2, 5.2); grid.setColor((150, 150, 150, 80))
        grid.translate(0, 0, 0); self.view.addItem(grid)
        self.surf = gl.GLSurfacePlotItem(x=self.gx[0], y=self.gy[:, 0], z=np.zeros_like(self.gx), shader='shaded', smooth=True,
                                         computeNormals=True, drawEdges=False, glOptions='translucent')
        self.view.addItem(self.surf)
        self.rings = []
        for u in range(G.NU):
            it = gl.GLLinePlotItem(pos=_circ + [G.XY[u, 0], G.XY[u, 1], self.gap], color=(0.75, 0.3, 0.1, 1), width=2, antialias=True, mode='line_strip')
            self.view.addItem(it); self.rings.append(it)
        self.coils = []
        for u in range(G.NU):
            c = (0.85, 0.2, 0.2, 0.9) if G.SUNK[u] else (0.2, 0.2, 0.2, 0.9)
            it = gl.GLLinePlotItem(pos=_coil + [G.XY[u, 0], G.XY[u, 1], G.COIL_Z0[u]], color=c, width=1.5, antialias=True, mode='line_strip')
            self.view.addItem(it); self.coils.append(it)
        self.truth_rings = []
        for u in range(G.NU):
            it = gl.GLLinePlotItem(pos=_circ + [G.XY[u, 0], G.XY[u, 1], self.gap], color=(0.1, 0.5, 0.1, 0.7), width=1, antialias=True, mode='line_strip')
            it.setVisible(False); self.view.addItem(it); self.truth_rings.append(it)

    # ---------- 2D 退化 ----------
    def _init_2d(self, lay):
        self.pw = pg.PlotWidget(); lay.addWidget(self.pw); self.pw.setAspectLocked(True); self.pw.setBackground('w')
        self.pw.hideAxis('left'); self.pw.hideAxis('bottom')
        self.surf2d = pg.PlotDataItem(pen=pg.mkPen((120, 120, 120), width=0.6), connect='pairs'); self.pw.addItem(self.surf2d)
        self.rings2d = pg.PlotDataItem(pen=pg.mkPen((190, 80, 20), width=2), connect='finite'); self.pw.addItem(self.rings2d)
        self.coils2d = pg.PlotDataItem(pen=pg.mkPen((60, 60, 60), width=1), connect='finite'); self.pw.addItem(self.coils2d)
        self.truth2d = pg.PlotDataItem(pen=pg.mkPen((30, 130, 30), width=1, style=Qt.PenStyle.DashLine), connect='finite'); self.pw.addItem(self.truth2d)
    @staticmethod
    def _iso(p):
        """等轴投影 (x,y,z) → (X,Y)."""
        return p[..., 0] - 0.5 * p[..., 1], 0.5 * p[..., 1] * 0.9 + p[..., 2]

    # ---------- 更新 ----------
    def _ring_pts(self, poses, zscale, base=_circ):
        out = np.empty((G.NU, len(_ang), 3))
        for u in range(G.NU):
            dx, dy, dz, tax, tay = poses[u]
            R = rot_matrix(tax * zscale, tay * zscale)      # 倾斜随 z 放大同比夸张
            out[u] = base @ R.T + [G.XY[u, 0] + dx, G.XY[u, 1] + dy, self.gap + dz * zscale]
        return out
    def _surface(self, q, zscale):
        w = q[:G.NU]
        z = (self.W @ w).reshape(self.gx.shape)
        z = np.where(self.inside, z, np.nan)
        return z * zscale + self.gap
    def update_result(self, res):
        tr = res.track
        if tr is None:
            return
        zscale = float(self.zs.value())
        q = tr.q
        poses = G.pose_of(q)
        z = self._surface(q, zscale)
        mode = self.cmode.currentText()
        if mode == 'w':
            cv, lo, hi = tr.w * 1e3, -500, 50
        elif mode.startswith('ΔT'):
            cv, lo, hi = res.dT_ring, -5, 20
        else:
            cv, lo, hi = np.hypot(tr.u, tr.v) * 1e3, 0, 200
        cgrid = cv[self.nearest]
        cols = self.cmap.map(np.clip((cgrid - lo) / (hi - lo), 0, 1), mode='float')   # (ny,nx,4)
        rp = self._ring_pts(poses, zscale)
        truth = res.truth if (self.show_truth.isChecked() and res.truth is not None) else None
        if self.gl_ok:
            zz = np.where(np.isnan(z), self.gap, z)
            cols[..., 3] = np.where(np.isnan(z), 0.0, 0.85)
            self.surf.setData(z=zz.T, colors=cols.transpose(1, 0, 2))
            for u in range(G.NU):
                self.rings[u].setData(pos=rp[u])
            if truth is not None:
                tp = self._ring_pts(G.pose_of(truth.q), zscale)
                for u in range(G.NU):
                    self.truth_rings[u].setData(pos=tp[u]); self.truth_rings[u].setVisible(True)
            else:
                for it in self.truth_rings:
                    it.setVisible(False)
        else:
            X, Y = self._iso(np.stack([self.gx, self.gy, np.nan_to_num(z, nan=self.gap)], -1))
            segs = []
            for i in range(0, X.shape[0], 3):
                m = self.inside[i]
                xs, ys = X[i][m], Y[i][m]
                segs.append(np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], 1))
            for j in range(0, X.shape[1], 3):
                m = self.inside[:, j]
                xs, ys = X[:, j][m], Y[:, j][m]
                segs.append(np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], 1))
            s = np.vstack(segs)
            self.surf2d.setData(s[:, [0, 2]].ravel(), s[:, [1, 3]].ravel())
            def polys(pts):
                X, Y = self._iso(pts)
                nanc = np.full((pts.shape[0], 1), np.nan)
                return np.hstack([X, nanc]).ravel(), np.hstack([Y, nanc]).ravel()
            self.rings2d.setData(*polys(rp))
            cp = np.stack([_coil + [G.XY[u, 0], G.XY[u, 1], G.COIL_Z0[u]] for u in range(G.NU)])
            self.coils2d.setData(*polys(cp))
            if truth is not None:
                self.truth2d.setData(*polys(self._ring_pts(G.pose_of(truth.q), zscale)))
            else:
                self.truth2d.setData([], [])
