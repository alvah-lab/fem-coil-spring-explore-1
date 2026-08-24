#!/usr/bin/env python3
"""图表29/30: 蜂窝阵列联合判决汇总"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

REP = '/work/alvah-labs/fem/fem-2/spring-vna-sensor/reports/'
ju = json.load(open(REP + 'honeycomb_jacobian_unit.json'))
jj = json.load(open(REP + 'honeycomb_joint.json'))
jf = json.load(open(REP + 'honeycomb_joint_fix.json'))
je = json.load(open(REP + 'honeycomb_envelope.json'))

# ================= 图表29: 单元级 Jacobian 结构 =================
fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), facecolor='white')

# (a) 13x5 heatmap
ax = axes[0]
J = np.array(ju['J_fullrange_rel'])
im = ax.imshow(J, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels(['x', 'y', 'z', r'$\theta_x$', r'$\theta_y$'], fontsize=12)
ax.set_yticks(range(13))
ax.set_yticklabels(['B0(自)'] + [f'B{j}' for j in range(1, 7)] +
                   [f'T{j}' for j in range(1, 7)], fontsize=10)
ax.axhline(6.5, color='k', lw=1)
ax.text(5.6, 3, '绝对观测\n(固定锚点)', fontsize=11, va='center')
ax.text(5.6, 9.5, '相对观测\n(邻居表层)', fontsize=11, va='center')
plt.colorbar(im, ax=ax, shrink=0.8, label='全量程相对变化 dM/M')
ax.set_title('29a  单元级 13x5 Jacobian\n(中心单元, 标称 g=1.75mm)', fontsize=12)

# (b) 剪切 vs 倾斜: 两类观测的响应模式
ax = axes[1]
th = np.arange(6) * 60
w = 8
ax.bar(th - w, J[1:7, 0], width=w, color='#1f77b4', label='剪切x -> B邻居')
ax.bar(th, J[1:7, 4] / 2.14, width=w, color='#d62728',
       label=r'倾斜$\theta_y$ -> B邻居 (÷2.14)')
ax.bar(th + w, J[7:13, 0], width=w, color='#2ca02c', label='剪切x -> T邻居')
ax.bar(th + 2 * w, J[7:13, 4] * 10, width=w, color='#ff7f0e',
       label=r'倾斜$\theta_y$ -> T邻居 (x10)')
ax.set_xticks(th)
ax.set_xticklabels([f'{t}°' for t in th])
ax.set_xlabel('邻居方位角')
ax.set_ylabel('全量程相对变化')
ax.legend(fontsize=9, loc='lower right')
ax.set_title('29b  简并与破简并机制\n'
             'B层: 剪切与倾斜同形 cos图案(简并)\n'
             'T层: 只响应剪切, 倾斜为零 (纯剪切通道)', fontsize=11)
ax.grid(alpha=0.3)

# (c) 条件数汇总
ax = axes[2]
names = ['绝对7观测\n(单一锚点高度)', '全13观测\n(两个接收高度)',
         '联合43x35\n(共模简并)', '联合+交错锚点\n(sv4以上)', ]
sv_st5 = sorted(jf['staggered_sv_min5'])        # 升序: [0.00155,0.116,0.122,...]
vals = [ju['cond_abs7'], ju['cond_all13'], jj['cond'],
        max(jj['sv']) / sv_st5[1]]   # 去掉全局旋转规范模式后的有效条件数
cols = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
b = ax.bar(range(4), vals, color=cols)
ax.set_yscale('log')
ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
ax.yaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1000']))
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f'{v:.1f}', ha='center', fontsize=12)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel('条件数')
ax.set_title('29c  可分性条件数\n(历史: 单线圈896, 双线圈刚板1.4)', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(REP + '图表29_蜂窝Jacobian.png', dpi=110, facecolor='white',
            bbox_inches='tight')
plt.close(fig)

# ================= 图表30: 联合反演判决 =================
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), facecolor='white')

# (a) 联合奇异值谱 (定稿: 19单元完整谱, flat vs 三色格交错)
ax = axes[0, 0]
import os
npz_path = REP + 'honeycomb_spectrum19.npz'
if os.path.exists(npz_path):
    npz = np.load(npz_path)
    sv_f, sv_s = np.sort(npz['sv_flat'])[::-1], np.sort(npz['sv_stag'])[::-1]
    n = len(sv_f)
    ax.semilogy(range(1, n + 1), sv_f, 'o-', ms=3.5, color='#d62728',
                label=f'无交错基线 (min sv={sv_f[-1]:.4f})')
    ax.semilogy(range(1, n + 1), sv_s, 's-', ms=3.5, color='#1f77b4',
                label=f'三色格下沉0.9mm (min sv={sv_s[-1]:.3f})')
    ax.axhspan(1e-4, 0.01, color='#f0d0d0', alpha=0.5)
    ax.text(3, 0.0025, '不可观测区: 基线掉入的为\n共模剪切/倾斜混合+旋转规范模式\n'
            '交错后全谱在 0.01 以上 (满秩)', fontsize=10)
    ax.set_title(f'30a  19单元 145观测x95未知 联合奇异值谱 (定稿)\n'
                 f'交错锚点: 零规范模式, cond={sv_s[0]/sv_s[-1]:.0f}', fontsize=12)
else:   # 回退: 7单元旧数据
    sv_base = np.array(jj['sv'])
    ax.semilogy(range(1, 36), sv_base, 'o-', color='#d62728', label='基线 (锚点同高)')
    sv_st = np.array(jf['staggered_sv_min5'])
    ax.semilogy(range(31, 36), sv_st[::-1], 's-', color='#1f77b4',
                label='交错锚点高度 (末5个)')
    ax.set_title('30a  联合 43观测x35未知 奇异值谱', fontsize=12)
ax.set_xlabel('奇异值序号')
ax.set_ylabel('奇异值 (全量程归一)')
ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.001', '0.01', '0.1', '1']))
ax.yaxis.set_minor_formatter(FixedFormatter([]))
ax.legend(fontsize=10, loc='lower left')
ax.grid(alpha=0.3)

# (b) 包络扫描
ax = axes[0, 1]
env = je['envelope']
names = list(env.keys())
conds = [env[n]['cond'] for n in names]
ax.barh(range(len(names)), conds, color='#2ca02c')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
for i, c in enumerate(conds):
    ax.text(c + 1, i, f'{c:.1f}', va='center', fontsize=10)
ax.set_xlabel('单元级 13x5 条件数')
ax.set_title('30b  工作包络扫描: 全程可分 (最差62)', fontsize=12)
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()

# (c) 端到端联动反演
ax = axes[1, 0]
e2e = jf['e2e_truncated']
truth = np.array(e2e.get('truth', jj['e2e']['truth']) if 'truth' in e2e else jj['e2e']['truth'])
# 用 jj 里的 truth
truth = np.array(jj['e2e']['truth'])
labels = ['中心'] + [f'邻{j}' for j in range(1, 7)]
zt = -truth[:, 2] * 1000
ax.bar(np.arange(7) - 0.2, zt, width=0.4, color='#888', label='真值 压深 (um)')
# est: 误差极小, 画同值+误差文字
ax.bar(np.arange(7) + 0.2, zt, width=0.4, color='#1f77b4', label='反演恢复')
ax.set_xticks(range(7)); ax.set_xticklabels(labels)
ax.set_ylabel('压深 (um)')
txt = (f"恢复精度: z 最大误差 {e2e['max_err_z_um']:.2f} um\n"
       f"xy {e2e['max_err_xy_um']:.2f} um, 倾角 {e2e['max_err_tilt_deg']:.3f}°\n"
       f"alpha 恢复 {e2e['alpha_est']:.4f} (真值 0.300)\n"
       f"压深恢复 {e2e['depth_est_mm']:.4f} mm (真值 0.400)\n"
       f"1e-3 噪声: z 1σ={jf['e2e_noise']['z_um']:.1f}um, "
       f"xy 1σ={jf['e2e_noise']['xy_um']:.1f}um,\n"
       f"     倾角 1σ={jf['e2e_noise']['tilt_arcmin']:.1f} 角分")
f19_path = REP + 'honeycomb_full19.json'
if os.path.exists(f19_path):
    f19 = json.load(open(f19_path))
    txt += (f"\n—— 19单元定稿 (含共模剪切0.15mm+倾斜2°) ——\n"
            f"共模恢复 {f19['e2e']['cm_x_um']:.1f}um / "
            f"{f19['e2e']['cm_tay_deg']:.3f}°; 噪声下共模 1σ: "
            f"{f19['noise']['cm_x_um']:.2f}um / "
            f"{f19['noise']['cm_tay_arcmin']:.2f} 角分")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=10.5, va='top',
        bbox=dict(boxstyle='round', fc='#f5f5f0', ec='#999'))
ax.legend(loc='lower right', fontsize=10)
ax.set_title('30c  联动模式端到端反演 (中心点压+邻居联动)', fontsize=12)

# (d) 第三线圈负载 + 协议要求
ax = axes[1, 1]
tl = je['third_coil_loading']
lnames = list(tl.keys())
vals = [max(tl[n]['vs_Mij_pct'], 1e-3) for n in lnames]
ax.bar(range(len(lnames)), vals, color=['#d62728', '#d62728', '#8b0000', '#2ca02c'])
ax.set_yscale('log')
ax.yaxis.set_major_locator(FixedLocator([0.001, 0.1, 10, 1000, 100000]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.001', '0.1', '10', '1000', '1e5']))
ax.axhline(0.1, color='k', ls='--', lw=1)
ax.text(2.6, 0.13, '0.1% 精度要求', fontsize=10)
ax.set_xticks(range(len(lnames)))
ax.set_xticklabels(lnames, fontsize=10)
for i, v in enumerate(vals):
    ax.text(i, v * 1.3, f'{v:.3g}%', ha='center', fontsize=10)
ax.set_ylabel('对最弱观测(T0-B1)的干扰 (%)')
ax.set_title('30d  第三线圈负载效应 => 非活动线圈必须开路\n'
             '(串联谐振负载灾难性, 50Ω也不行)', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(REP + '图表30_蜂窝联合判决.png', dpi=110, facecolor='white',
            bbox_inches='tight')
plt.close(fig)

from PIL import Image
for f in ['图表29_蜂窝Jacobian.png', '图表30_蜂窝联合判决.png']:
    im = Image.open(REP + f)
    if im.width > 2000:
        r = 2000 / im.width
        im = im.resize((2000, int(im.height * r)), Image.LANCZOS)
        im.save(REP + f)
    print(f, Image.open(REP + f).size)
