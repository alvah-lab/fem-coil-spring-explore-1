#!/usr/bin/env python3
# ⚠ 已知缺陷 (2026-09-06): 双层螺旋绕向部分抵消 (L=0.072µH) 且单位换算使 R 下溢, 本脚本的 L/Q 输出无效; 分辨率结论以 pcb_verdict.py + 验证板规模方案 §4 解析推导为准. 待修绕向后重跑.
"""投板前验算: 真实 PCB 配置的方案C 19单元判决
- 有源线圈: Ø5mm 双层平面螺旋 (L1+L2 或 L3+L4), 12匝/层, 1oz
- 层深交错: 标称单元螺旋质心 z=0.08mm, 下沉单元 z=-0.65mm (高差0.73mm)
- 无源环: 紫铜 Ø5/Ø3/0.2mm 单匝短路环, 贴泡棉顶面(全部同高 z=+1.75)
- 频率 8MHz (65M ADC 甜点), 判决可分性/条件数/分辨率 vs 理想仿真
真实螺旋按多匝建模 (不再单环x225近似), 检验降规格影响
"""
import numpy as np
import subprocess, re, os, time
import gen_ring_array as G

WD = G.WORKDIR
F = 8e6
SIGMA_CU = 5.8e7
NSEG_TURN = 24        # 每匝分段

# ---- 几何参数 (mm) ----
DO, DI = 5.0, 0.7     # 螺旋外/内径
NT = 12               # 匝/层
TRACE = 0.0889        # 3.5mil
PITCH_R = (DO - DI) / 2 / NT       # 径向匝距
LAYER_GAP = 0.10      # L1-L2 层间距 (1.0mm 4层板 prepreg)
COIL_TOP_Z = 0.08     # 标称单元双层螺旋质心
COIL_SUNK_Z = COIL_TOP_Z - 0.73   # 下沉单元
RING_Z = 1.75         # 铜环质心 (泡棉顶, 全部同高)
RING_R = 2.5          # 环中径半径 Ø5
RING_W = 1.0          # 环径向宽 (5-3)/2*2? 宽1mm
RING_H = 0.2          # 环厚
RING_SIG = SIGMA_CU   # 紫铜
PITCH = 5.2

# ---- 蜂窝19单元 + 三色交错 ----
a1 = np.array([PITCH, 0.0]); a2 = np.array([PITCH / 2, PITCH * np.sqrt(3) / 2])
UNITS = []
for i in range(-3, 4):
    for j in range(-3, 4):
        p = i * a1 + j * a2
        if np.linalg.norm(p) < PITCH * 2.3:
            UNITS.append((p[0], p[1], (i - j) % 3))
NU = len(UNITS)
SUNK = [c == 0 for _, _, c in UNITS]      # 下沉单元
ADJ = {i: [j for j in range(NU) if i != j and
           np.hypot(UNITS[i][0] - UNITS[j][0],
                    UNITS[i][1] - UNITS[j][1]) < PITCH * 1.05] for i in range(NU)}
EDGES = sorted({tuple(sorted((i, j))) for i in range(NU) for j in ADJ[i]})
CI = min(range(NU), key=lambda u: np.hypot(UNITS[u][0], UNITS[u][1]))

def spiral_nodes(cx, cy, z0, layer_gap, nt=NT):
    """双层平面螺旋 (L1顶+L2底, 串联) 节点; 两层间过孔连接. 单层nt匝阿基米德螺旋"""
    pts = []
    # 顶层: 外->内
    for L, zz, rev in [(0, z0 + layer_gap / 2, False), (1, z0 - layer_gap / 2, True)]:
        th = np.linspace(0, 2 * np.pi * nt, nt * NSEG_TURN + 1)
        r = DO / 2 - PITCH_R * th / (2 * np.pi)
        seg = np.stack([cx + r * np.cos(th), cy + r * np.sin(th),
                        np.full_like(th, zz)], 1)
        if rev:
            seg = seg[::-1]
        pts.append(seg)
    # 顶层内端 -> 底层内端 (过孔, 已由 rev 对齐), 拼接
    return np.vstack([pts[0], pts[1]])

def ring_nodes_c(cx, cy, z0, nseg=48):
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    return np.stack([cx + RING_R * np.cos(th), cy + RING_R * np.sin(th),
                     np.full_like(th, z0)], 1)

def write_and_solve(top_poses, tag):
    """top_poses: (NU,5) 铜环位姿偏移. 有源螺旋固定. 求全 (NU螺旋 + NU环) Z矩阵"""
    assert np.all(np.abs(top_poses[:, :2]) < 1.0) and \
           np.all(np.abs(top_poses[:, 2]) < 1.2) and \
           np.all(np.abs(top_poses[:, 3:]) < 0.35), 'pose 超界'
    L = ['* pcb valid', '.units mm', '', '']
    # 有源螺旋 (sigma铜, 1oz厚35um -> h=0.035; 线宽TRACE)
    for u, (cx, cy, c) in enumerate(UNITS):
        z0 = COIL_SUNK_Z if SUNK[u] else COIL_TOP_Z
        p = spiral_nodes(cx, cy, z0, LAYER_GAP)
        L.append(f'* coil {u}')
        L.append(f'.default sigma={SIGMA_CU} nhinc=1 nwinc=1')
        for i, (x, y, z) in enumerate(p):
            L.append(f'NC{u}_{i} x={x:.5f} y={y:.5f} z={z:.5f}')
        for i in range(len(p) - 1):
            L.append(f'EC{u}_{i} NC{u}_{i} NC{u}_{i+1} w={TRACE} h=0.035')
    # 无源环 (紫铜, 0.2厚)
    for u, (cx, cy, c) in enumerate(UNITS):
        dx, dy, dz, tax, tay = top_poses[u]
        p = ring_nodes_c(cx + dx, cy + dy, RING_Z + dz)
        # 倾斜
        if tax or tay:
            ctr = np.array([cx + dx, cy + dy, RING_Z + dz])
            q = p - ctr
            ca, sa = np.cos(tax), np.sin(tax)
            q = q @ np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]]).T
            cb, sb = np.cos(tay), np.sin(tay)
            q = q @ np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]).T
            p = q + ctr
        L.append(f'* ring {u}')
        L.append(f'.default sigma={RING_SIG} nhinc=1 nwinc=1')
        for i, (x, y, z) in enumerate(p):
            L.append(f'NR{u}_{i} x={x:.5f} y={y:.5f} z={z:.5f}')
        for i in range(len(p) - 1):
            L.append(f'ER{u}_{i} NR{u}_{i} NR{u}_{i+1} w={RING_W} h={RING_H}')
    # 端口: 只有源螺旋 (环短路, 由.external闭合即短路环? 需环自身external闭合)
    for u in range(NU):
        p = spiral_nodes(0, 0, 0, LAYER_GAP)
        L.append(f'.external NC{u}_0 NC{u}_{len(p)-1}')
    for u in range(NU):
        pr = ring_nodes_c(0, 0, 0)
        L.append(f'.external NR{u}_0 NR{u}_{len(pr)-1}')
    L.append(f'.freq fmin={F} fmax={F} ndec=1')
    L.append('.end')
    inp = os.path.join(WD, f'{tag}.inp')
    open(inp, 'w').write('\n'.join(L) + '\n')
    Z, _ = G.run_fasthenry(inp)
    return Z          # 2NU x 2NU 复阻抗

if __name__ == '__main__':
    t0 = time.time()
    print(f'{NU}单元, {sum(SUNK)}下沉; 螺旋{NT}匝x2层, 环紫铜Ø5/0.2, F={F/1e6}MHz',
          flush=True)
    Z0 = write_and_solve(np.zeros((NU, 5)), 'pcbv_base')
    n = Z0.shape[0]
    print(f'求解规模 {n}端口, 单解 {time.time()-t0:.0f}s', flush=True)
    # 有源螺旋自感 & 环参数
    w = 2 * np.pi * F
    Lc = np.imag(Z0[CI, CI]) / w * 1e6
    Rc = np.real(Z0[CI, CI])
    print(f'螺旋自感 {Lc:.3f}uH, R {Rc:.2f}Ω, Q {w*Lc*1e-6/Rc:.1f}')
    Lr = np.imag(Z0[NU + CI, NU + CI]) / w * 1e9
    Rr = np.real(Z0[NU + CI, NU + CI])
    print(f'紫铜环 L {Lr:.2f}nH, R {Rr*1e3:.2f}mΩ, Q {w*Lr*1e-9/Rr:.1f}', flush=True)
    np.savez('/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/pcb_valid_base.npz',
             Z0=Z0, units=np.array([(u[0], u[1], u[2]) for u in UNITS]),
             sunk=SUNK)
    print(f'基线存档 ({time.time()-t0:.0f}s)')
