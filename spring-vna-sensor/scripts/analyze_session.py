#!/usr/bin/env python3
"""离线分析 GUI 会话日志 (host/logs/session_*/): 事件时间线 + 逐帧跟踪器状态 + 跳变检测 + 图.

用法: python3 scripts/analyze_session.py host/logs/session_20260914_120000 [--plot out.png]
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'host'))


def load(d):
    ev = [json.loads(l) for l in open(os.path.join(d, 'events.jsonl'))]
    fr = [json.loads(l) for l in open(os.path.join(d, 'frames.jsonl'))]
    meta = json.load(open(os.path.join(d, 'meta.json')))
    return meta, ev, fr


def summarize(meta, ev, fr):
    print(f"session {meta.get('started')} git {meta.get('git')} source {meta.get('source')} frames {len(fr)}")
    print('\n== 事件 ==')
    for e in ev:
        kv = {k: v for k, v in e.items() if k not in ('t', 'seq', 'name')}
        print(f"  t={e['t']:8.2f}s seq={e['seq']}  {e['name']}  {json.dumps(kv, ensure_ascii=False)[:160]}")
    if not fr:
        return
    t = np.array([f['t'] for f in fr]); seq = np.array([f['seq'] for f in fr])
    has_tr = np.array(['q' in f for f in fr])
    chi2 = np.array([f.get('chi2', np.nan) for f in fr], float)
    resets = np.array([f.get('reset', False) for f in fr]); relin = np.array([f.get('relin', False) for f in fr])
    clamped = np.array([f.get('clamped', False) for f in fr])
    pres = np.array([sum(f['present']) if f['present'] is not None else 19 for f in fr])
    fps = np.array([f['fps'] for f in fr]); gap = np.array([f['seq_gap'] for f in fr])
    q = np.array([f['q'] if 'q' in f else [np.nan] * 57 for f in fr], float)
    print('\n== 逐帧汇总 ==')
    print(f"  时长 {t[-1]-t[0]:.1f}s  帧 {len(fr)}  seq 缺口累计 {gap.sum()}  fps 中位 {np.nanmedian(fps):.1f}")
    print(f"  跟踪器运行帧 {has_tr.sum()}  复位 {resets.sum()}  重线性化 {relin.sum()}  钳位帧 {clamped.sum()}")
    print(f"  χ² 中位 {np.nanmedian(chi2):.3g}  p90 {np.nanpercentile(chi2, 90):.3g}  max {np.nanmax(chi2):.3g}")
    # present 掩码变化
    ch = np.where(np.diff(pres) != 0)[0]
    for i in ch:
        print(f"  在位数 {pres[i]} → {pres[i+1]} @ t={t[i+1]:.2f}s seq={seq[i+1]}")
    # 跳变检测: 相邻帧 w 变化 > 100 µm 的次数与位置
    dw = np.abs(np.diff(q[:, :19], axis=0)); dw = np.where(np.isnan(dw), 0, dw)
    jumps = np.where(dw.max(axis=1) > 100)[0]
    print(f"  w 单帧跳变 >100 µm: {len(jumps)} 次" + (f"  首次 t={t[jumps[0]+1]:.2f}s" if len(jumps) else ''))
    if len(jumps):
        units, cnt = np.unique(np.argmax(dw[jumps], axis=1), return_counts=True)
        print('  跳变最多的单元:', dict(zip(units.tolist(), cnt.tolist())))
    # 在位单元 w 的稳定性 (最后 100 帧)
    tail = q[-100:, :19]
    if not np.all(np.isnan(tail)):
        print('  最后 100 帧 w 均值 (µm):', np.round(np.nanmean(tail, axis=0)).astype(int).tolist())
        print('  最后 100 帧 w σ    (µm):', np.round(np.nanstd(tail, axis=0), 1).tolist())
    if any('truth_q' in f for f in fr):
        tq = np.array([f['truth_q'] if 'truth_q' in f else [np.nan] * 57 for f in fr], float)
        err = np.abs(q[:, :19] - tq[:, :19])
        print(f"  相对真值 |w 误差| 最后 100 帧: 中位 {np.nanmedian(err[-100:]):.1f} µm  max {np.nanmax(err[-100:]):.1f} µm")
    return dict(t=t, chi2=chi2, q=q, resets=resets, pres=pres, fps=fps)


def plot(s, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK SC'
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].semilogy(s['t'], np.maximum(s['chi2'], 1e-3), lw=0.8); ax[0].set_ylabel('χ²')
    for tt in s['t'][s['resets']]:
        ax[0].axvline(tt, color='r', lw=0.5, alpha=0.5)
    ax[1].plot(s['t'], s['q'][:, :19], lw=0.6); ax[1].set_ylabel('w (µm)')
    ax[2].plot(s['t'], s['pres'], lw=1, label='在位环数'); ax[2].plot(s['t'], s['fps'], lw=0.6, label='fps'); ax[2].legend(); ax[2].set_xlabel('t (s)')
    fig.tight_layout(); fig.savefig(out, dpi=120); print('plot', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    s = summarize(*load(a.dir))
    if a.plot and s:
        plot(s, a.plot)
