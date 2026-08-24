#!/usr/bin/env python3
"""蜂窝阵列束绕圆环线圈 FastHenry 几何生成器 + 基线验证
线圈 = Ø5mm 中径单匝环 (束截面 0.2x0.2mm 等效), 互感结果 x N^2=225
步骤2: (a) 双环验证 L_self vs 环公式, M(g) vs 椭圆积分解析解
       (b) 7单元x2层 14环标称基线, 输出全 M 矩阵与耦合量级
"""
import numpy as np
import subprocess, re, json, os, sys, time

FASTHENRY = '/work/alvah-labs/fem/fem-2/FastHenry2/bin/fasthenry'
WORKDIR = os.path.dirname(os.path.abspath(__file__))
MU0 = 4e-7 * np.pi

# ---------------- 结构参数 (mm) ----------------
R_COIL = 2.5        # 环中径半径
N_TURNS = 15        # 每束股数 -> M,L 缩放 N^2
BUNDLE_W = 0.2      # 等效方束截面
PITCH = 5.2
GAP_REST, GAP_MIN, GAP_NOM = 2.5, 1.0, 1.75
NSEG = 72           # 每环分段
GAP_ARC = 0.005     # 端口开口弧长 mm (足够小以保持六重对称)
SIGMA = 5.8e7       # 铜 (R 数值单位约定不准, 只用 L)
FREQ = 1e6

def hex_centers(pitch=PITCH):
    cs = [(0.0, 0.0)]
    for k in range(6):
        a = np.pi / 3 * k
        cs.append((pitch * np.cos(a), pitch * np.sin(a)))
    return cs

# ---------------- 环节点生成 ----------------
def ring_nodes(pose, r=R_COIL, nseg=NSEG):
    """pose=(x,y,z,tax,tay): 环心位置 + 绕x/y轴倾角(rad). 返回 (nseg+1,3) 数组,
    首尾之间留 GAP_ARC 开口 (端口跨接处)"""
    x0, y0, z0, tax, tay = pose
    gap_ang = GAP_ARC / r
    th = np.linspace(gap_ang / 2, 2 * np.pi - gap_ang / 2, nseg + 1)
    p = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)
    # 绕 x 轴 -> 绕 y 轴 旋转 (绕环心)
    ca, sa = np.cos(tax), np.sin(tax)
    p = p @ np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]]).T
    cb, sb = np.cos(tay), np.sin(tay)
    p = p @ np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]).T
    return p + np.array([x0, y0, z0])

def write_inp(path, rings):
    """rings: list of (name, pose). 每环一个 .external, 端口顺序 = rings 顺序"""
    L = ['* honeycomb ring array', '.units mm',
         f'.default sigma={SIGMA:.4g} nhinc=1 nwinc=1', '']
    for name, pose in rings:
        p = ring_nodes(pose)
        for i, (x, y, z) in enumerate(p):
            L.append(f'N{name}_{i} x={x:.6f} y={y:.6f} z={z:.6f}')
        for i in range(len(p) - 1):
            L.append(f'E{name}_{i} N{name}_{i} N{name}_{i+1} '
                     f'w={BUNDLE_W} h={BUNDLE_W}')
        L.append('')
    for name, _ in rings:
        L.append(f'.external N{name}_0 N{name}_{NSEG}')
    L.append(f'.freq fmin={FREQ:.4g} fmax={FREQ:.4g} ndec=1')
    L.append('.end')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')

def run_fasthenry(inp):
    t0 = time.time()
    r = subprocess.run([FASTHENRY, os.path.basename(inp),
                        '-m', 'direct', '-s', 'ludecomp'],
                       cwd=WORKDIR, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + '\n' + r.stderr[-2000:] + '\n')
        raise RuntimeError('fasthenry failed')
    return parse_zc(os.path.join(WORKDIR, 'Zc.mat')), time.time() - t0

def parse_zc(path):
    """解析 Zc.mat -> (nport,nport) 复矩阵 (最后一个频点)"""
    txt = open(path).read()
    blocks = re.split(r'Impedance matrix for frequency[^\n]*\n', txt)[1:]
    rows = []
    for line in blocks[-1].strip().splitlines():
        nums = re.findall(r'[-+]?[\d.]+(?:e[-+]?\d+)?', line.replace('j', ' '))
        if not nums:
            break
        v = [float(x) for x in nums]
        rows.append([complex(v[i], v[i + 1]) for i in range(0, len(v), 2)])
    n = len(rows[0])
    return np.array(rows[:n])

def z_to_L(Z, freq=FREQ):
    """Im(Z)/w -> 电感矩阵 (H), 乘 N^2 得束绕值"""
    return np.imag(Z) / (2 * np.pi * freq)

# ---------------- 解析参照: 完全椭圆积分 (AGM) ----------------
def ellipKE(m):
    a, b, c = 1.0, np.sqrt(1 - m), np.sqrt(m)
    s, n = c * c / 2, 0
    while abs(c) > 1e-15:
        a, b, c = (a + b) / 2, np.sqrt(a * b), (a - b) / 2
        n += 1
        s += 2 ** (n - 1) * c * c
    K = np.pi / (2 * a)
    return K, K * (1 - s)

def M_coax_analytic(a_mm, b_mm, d_mm):
    """共轴双环互感 (H), Maxwell 公式"""
    a, b, d = a_mm * 1e-3, b_mm * 1e-3, d_mm * 1e-3
    k2 = 4 * a * b / ((a + b) ** 2 + d ** 2)
    k = np.sqrt(k2)
    K, E = ellipKE(k2)
    return MU0 * np.sqrt(a * b) * ((2 / k - k) * K - 2 / k * E)

def L_ring_analytic(a_mm=R_COIL, w_mm=BUNDLE_W):
    """单环自感 (H), 方截面 GMD=0.2235*(w+h)"""
    a, r = a_mm * 1e-3, 0.2235 * 2 * w_mm * 1e-3
    return MU0 * a * (np.log(8 * a / r) - 2)

# ---------------- 场景构建 ----------------
def build_pair(gap):
    return [('B', (0, 0, 0, 0, 0)), ('T', (0, 0, gap, 0, 0))]

def build_patch(top_poses=None, gap=GAP_NOM):
    """7单元x2层. top_poses: dict unit->(x,y,z,tax,tay) 相对标称的位姿偏移
    端口顺序: T0..T6 然后 B0..B6"""
    cs = hex_centers()
    rings = []
    for u, (cx, cy) in enumerate(cs):
        dx = dy = dz = tax = tay = 0.0
        if top_poses and u in top_poses:
            dx, dy, dz, tax, tay = top_poses[u]
        rings.append((f'T{u}', (cx + dx, cy + dy, gap + dz, tax, tay)))
    for u, (cx, cy) in enumerate(cs):
        rings.append((f'B{u}', (cx, cy, 0.0, 0.0, 0.0)))
    return rings

def solve(rings, tag):
    inp = os.path.join(WORKDIR, f'{tag}.inp')
    write_inp(inp, rings)
    Z, dt = run_fasthenry(inp)
    return z_to_L(Z), dt

# ================================================================
if __name__ == '__main__':
    out = {}

    # ---- (a1) 单环自感验证 ----
    Lm, dt = solve(build_pair(GAP_NOM), 'pair_nom')
    L_fh = Lm[0, 0]
    L_an = L_ring_analytic()
    print(f'[pair {dt:.1f}s] L_ring: FastHenry={L_fh*1e9:.3f}nH  '
          f'analytic={L_an*1e9:.3f}nH  diff={(L_fh/L_an-1)*100:+.1f}%')
    print(f'   束绕 L = {L_fh*225*1e6:.3f} uH (x225)')

    # ---- (a2) M(g) 共轴互感 vs 解析 ----
    Mg = {}
    for g in [1.0, 1.5, 1.75, 2.0, 2.5, 3.0]:
        Lm2, dt = solve(build_pair(g), f'pair_g{g}')
        m_fh, m_an = Lm2[0, 1], M_coax_analytic(R_COIL, R_COIL, g)
        Mg[g] = dict(M_fh_nH=m_fh * 1e9, M_an_nH=m_an * 1e9,
                     diff_pct=(m_fh / m_an - 1) * 100,
                     k=m_fh / np.sqrt(Lm2[0, 0] * Lm2[1, 1]))
        print(f'[g={g}] M: FH={m_fh*1e9:.4f}nH  an={m_an*1e9:.4f}nH  '
              f'diff={(m_fh/m_an-1)*100:+.2f}%  k={Mg[g]["k"]:.4f}')
    out['L_ring_nH'] = dict(fasthenry=L_fh * 1e9, analytic=L_an * 1e9)
    out['L_bundle_uH'] = L_fh * 225 * 1e6
    out['M_coax_vs_gap'] = Mg

    # ---- (b) 7单元标称基线 ----
    rings = build_patch()
    Lm14, dt = solve(rings, 'patch_nom')
    print(f'\n[patch 14 rings {dt:.1f}s]')
    names = [r[0] for r in rings]
    idx = {n: i for i, n in enumerate(names)}
    # 中心表层 T0 的 13 个观测
    obs = {}
    for tgt in ['B0'] + [f'T{j}' for j in range(1, 7)] + [f'B{j}' for j in range(1, 7)]:
        m = Lm14[idx['T0'], idx[tgt]]
        obs[f'T0-{tgt}'] = m * 1e9
    print('中心表层 T0 观测 (单匝 nH, 束绕 x225):')
    for k_, v in obs.items():
        print(f'  M({k_}) = {v:.5f} nH  (束绕 {v*225/1000:.3f} uH)')
    # 对称性: 6个邻居应相同
    mt = [obs[f'T0-T{j}'] for j in range(1, 7)]
    mb = [obs[f'T0-B{j}'] for j in range(1, 7)]
    print(f'对称性: T0-Tj spread={np.ptp(mt)/abs(np.mean(mt))*100:.3f}%  '
          f'T0-Bj spread={np.ptp(mb)/abs(np.mean(mb))*100:.3f}%')
    out['patch_obs_T0_nH'] = obs
    out['patch_symmetry_spread_pct'] = dict(
        TT=float(np.ptp(mt) / abs(np.mean(mt)) * 100),
        TB=float(np.ptp(mb) / abs(np.mean(mb)) * 100))
    out['patch_L_matrix_nH'] = (Lm14 * 1e9).tolist()
    out['ring_names'] = names
    out['solve_time_s'] = dt

    jp = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/honeycomb_baseline.json'
    with open(jp, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f'\nsaved {jp}')
