#!/usr/bin/env python3
"""M4: External tuning capacitor C_ext optimization for the spring-coil VNA sensor.

Adds an external capacitor C_ext in parallel with the coil self-capacitance C0.
Sweeps C_ext, evaluates:
  - SRF location
  - bending / compression sensitivity spectra (peak dB spread)
  - blind-inversion grid-search MLE RMSE @ 60 dB SNR
and identifies the optimal C_ext (minimizing bending kappa RMSE while keeping
full-rank identifiability). Verifies whether the dilution tradeoff optimum appears.

numpy-only (no scipy).
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'AR PL UKai CN']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physics model (must match prior milestones exactly)
# ---------------------------------------------------------------------------
a = 1.5e-3          # design coil radius (m)
EPS_MAX = 0.333
KAP_MAX = 100.0
L0_BASE = 1e-9
C0_BASE = 5e-12


def S11_ext(eps, kap, freq, C_ext):
    """S11 with external cap C_ext in parallel. C_ext=0 -> baseline."""
    omega = 2 * np.pi * freq
    eta = (a * kap) ** 2
    L = 1e-9 * (1 - 0.4 * eps) * (1 - 0.6 * eta)
    C0 = 5e-12 * (1 + 0.3 * eps) * (1 + 0.8 * eta)
    C = C0 + C_ext
    f1 = 1 / (2 * np.pi * np.sqrt(L * C + 1e-32)) * (1 + 0.05 * eps) * (1 + 0.1 * eta)
    w1 = 2 * np.pi * f1
    Q = 150 * (1 - 0.3 * eps)
    Z = 5.0 + 1j * (omega * L - 1 / (omega * C + 1e-16))
    Z = Z / (1 + 1j * omega * (1 / (Q * w1)))
    return (Z - 50) / (Z + 50 + 1e-16)


def s11_db(s):
    return 20 * np.log10(np.abs(s) + 1e-16)


def srf_of(C_ext):
    """Self-resonant frequency using base L0,C0 (nominal, eps=kap=0)."""
    return 1.0 / (2 * np.pi * np.sqrt(L0_BASE * (C0_BASE + C_ext)))


def make_freq(C_ext):
    """1 MHz - 10 GHz log grid; extend low end if SRF drops below ~1 MHz margin."""
    srf = srf_of(C_ext)
    lo_exp = 6
    # keep at least ~1.5 decades below SRF so the lower flank is captured
    need_lo = np.log10(srf) - 2.5
    if need_lo < lo_exp:
        lo_exp = int(np.floor(need_lo))
    return np.logspace(lo_exp, 10, 31)


# ---------------------------------------------------------------------------
# Sensitivity spectra
# ---------------------------------------------------------------------------
def sensitivity_spectra(C_ext, freq):
    """Return (bend_peak, comp_peak, bend_flank_freq) in dB / Hz.

    spread(f) = max_over_param(|S11|dB) - min_over_param(|S11|dB); peak over f.
    """
    kap_grid = np.linspace(0, KAP_MAX, 11)
    eps_grid = np.linspace(0, EPS_MAX, 11)

    # bending: vary kappa at eps=0
    bend = np.array([s11_db(S11_ext(0.0, k, freq, C_ext)) for k in kap_grid])  # (11, nf)
    bend_spread = bend.max(axis=0) - bend.min(axis=0)
    bend_peak = bend_spread.max()
    # flank frequency = conventional LOWER-flank operating point (below SRF),
    # where the resonance-notch slope is steepest. Restrict argmax to f < SRF.
    srf = srf_of(C_ext)
    below = freq < srf
    if below.any():
        idx = np.where(below)[0]
        bend_flank_freq = freq[idx[np.argmax(bend_spread[idx])]]
    else:
        bend_flank_freq = freq[np.argmax(bend_spread)]

    # compression: vary eps at kappa=0
    comp = np.array([s11_db(S11_ext(e, 0.0, freq, C_ext)) for e in eps_grid])
    comp_spread = comp.max(axis=0) - comp.min(axis=0)
    comp_peak = comp_spread.max()

    return bend_peak, comp_peak, bend_flank_freq


# ---------------------------------------------------------------------------
# Blind-inversion grid-search MLE Monte-Carlo
# ---------------------------------------------------------------------------
def build_library(C_ext, freq, ne=81, nk=81):
    """Reference library: observation vectors concat(Re,Im) over freq for (eps,kap) grid."""
    eps_ax = np.linspace(0, EPS_MAX, ne)
    kap_ax = np.linspace(0, KAP_MAX, nk)
    EE, KK = np.meshgrid(eps_ax, kap_ax, indexing='ij')
    epsf = EE.ravel()
    kapf = KK.ravel()

    omega = 2 * np.pi * freq
    eta = (a * kapf[:, None]) ** 2                       # (N,1)
    L = 1e-9 * (1 - 0.4 * epsf[:, None]) * (1 - 0.6 * eta)
    C0 = 5e-12 * (1 + 0.3 * epsf[:, None]) * (1 + 0.8 * eta)
    C = C0 + C_ext
    f1 = 1 / (2 * np.pi * np.sqrt(L * C + 1e-32)) * (1 + 0.05 * epsf[:, None]) * (1 + 0.1 * eta)
    w1 = 2 * np.pi * f1
    Q = 150 * (1 - 0.3 * epsf[:, None])
    Z = 5.0 + 1j * (omega[None, :] * L - 1 / (omega[None, :] * C + 1e-16))
    Z = Z / (1 + 1j * omega[None, :] * (1 / (Q * w1)))
    S = (Z - 50) / (Z + 50 + 1e-16)                      # (N, nf)
    lib = np.concatenate([S.real, S.imag], axis=1)       # (N, 2nf)
    return lib, epsf, kapf


def mc_rmse(C_ext, freq, snr_db=60.0, ne=81, nk=81, n_test=5, n_trials=15, seed=0):
    lib, epsf, kapf = build_library(C_ext, freq, ne, nk)
    rng = np.random.default_rng(seed)

    # interior test points (avoid exact grid nodes / edges)
    te = np.linspace(0.12, 0.88, n_test) * EPS_MAX
    tk = np.linspace(0.12, 0.88, n_test) * KAP_MAX

    err_e = []
    err_k = []
    for e0 in te:
        for k0 in tk:
            y0 = S11_ext(e0, k0, freq, C_ext)
            y0v = np.concatenate([y0.real, y0.imag])
            sigma = np.sqrt(np.mean(y0v ** 2)) * 10 ** (-snr_db / 20)
            for _ in range(n_trials):
                noise = rng.normal(0, sigma, size=y0v.shape)
                yv = y0v + noise
                d2 = np.sum((lib - yv[None, :]) ** 2, axis=1)
                j = np.argmin(d2)
                err_e.append(epsf[j] - e0)
                err_k.append(kapf[j] - k0)
    err_e = np.array(err_e)
    err_k = np.array(err_k)
    rmse_e_pct = np.sqrt(np.mean(err_e ** 2)) / EPS_MAX * 100
    rmse_k_pct = np.sqrt(np.mean(err_k ** 2)) / KAP_MAX * 100
    return rmse_e_pct, rmse_k_pct


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(outdir, exist_ok=True)

    C_ext_sweep = np.concatenate([[0.0], np.logspace(-13, -10, 25)])  # 0, 0.1pF..100pF

    res = {
        "C_ext": [], "srf": [], "bend_flank_freq": [],
        "bend_sens": [], "comp_sens": [],
        "rmse_e_pct": [], "rmse_k_pct": [],
    }

    print("=" * 68)
    print("M4 External Capacitor Optimization sweep")
    print("=" * 68)
    print(f"{'C_ext(pF)':>10} {'SRF(MHz)':>10} {'bend(dB)':>9} {'comp(dB)':>9} "
          f"{'RMSE_e%':>8} {'RMSE_k%':>8}")

    for C_ext in C_ext_sweep:
        freq = make_freq(C_ext)
        bend_peak, comp_peak, bend_flank = sensitivity_spectra(C_ext, freq)
        rmse_e, rmse_k = mc_rmse(C_ext, freq, snr_db=60.0, seed=12345)
        srf = srf_of(C_ext)

        res["C_ext"].append(float(C_ext))
        res["srf"].append(float(srf))
        res["bend_flank_freq"].append(float(bend_flank))
        res["bend_sens"].append(float(bend_peak))
        res["comp_sens"].append(float(comp_peak))
        res["rmse_e_pct"].append(float(rmse_e))
        res["rmse_k_pct"].append(float(rmse_k))

        print(f"{C_ext*1e12:>10.3f} {srf/1e6:>10.1f} {bend_peak:>9.4f} "
              f"{comp_peak:>9.4f} {rmse_e:>8.3f} {rmse_k:>8.3f}")

    # ---- identify optimum: minimize bending kappa RMSE ----
    rmse_k_arr = np.array(res["rmse_k_pct"])
    rmse_e_arr = np.array(res["rmse_e_pct"])
    C_arr = np.array(res["C_ext"])
    i_opt = int(np.argmin(rmse_k_arr))
    C_opt = C_arr[i_opt]

    baseline = {
        "C_ext": res["C_ext"][0], "srf": res["srf"][0],
        "bend_sens": res["bend_sens"][0], "comp_sens": res["comp_sens"][0],
        "rmse_e_pct": res["rmse_e_pct"][0], "rmse_k_pct": res["rmse_k_pct"][0],
        "bend_flank_freq": res["bend_flank_freq"][0],
    }
    optimum = {
        "C_ext": res["C_ext"][i_opt], "srf": res["srf"][i_opt],
        "bend_sens": res["bend_sens"][i_opt], "comp_sens": res["comp_sens"][i_opt],
        "rmse_e_pct": res["rmse_e_pct"][i_opt], "rmse_k_pct": res["rmse_k_pct"][i_opt],
        "bend_flank_freq": res["bend_flank_freq"][i_opt],
        "index": i_opt,
    }

    # dilution tradeoff detection: does bending sensitivity peak then fall?
    bend_arr = np.array(res["bend_sens"])
    i_bend_max = int(np.argmax(bend_arr))
    # tradeoff optimum present if the bending-sens (or kappa-rmse) interior extremum
    # is strictly better than both ends
    kappa_interior_opt = (0 < i_opt < len(rmse_k_arr) - 1)
    sens_interior_peak = (0 < i_bend_max < len(bend_arr) - 1)

    findings = {
        "baseline": baseline,
        "optimum": optimum,
        "bend_sens_peak_index": i_bend_max,
        "bend_sens_peak_C_ext": float(C_arr[i_bend_max]),
        "kappa_rmse_optimum_is_interior": bool(kappa_interior_opt),
        "bend_sens_peak_is_interior": bool(sens_interior_peak),
    }

    out = {"sweep": res, "findings": findings,
           "meta": {"snr_db": 60, "ref_grid": "81x81", "test_pts": "5x5",
                    "trials": 15, "eps_max": EPS_MAX, "kap_max": KAP_MAX,
                    "a_radius_m": a}}
    json_path = os.path.join(outdir, "m4_capacitor_optimization.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    from matplotlib.ticker import FuncFormatter, LogLocator

    def _plainfmt(val, _pos):
        if val >= 1:
            return f'{val:.0f}'
        return f'{val:g}'
    plainfmt = FuncFormatter(_plainfmt)

    def style_logx(ax):
        ax.xaxis.set_major_formatter(plainfmt)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ''))

    C_pF = C_arr * 1e12
    # x-axis: use index for the 0 point handling; plot vs C_pF but 0 -> small
    x = np.where(C_pF <= 0, 0.03, C_pF)   # place baseline at 0.03 pF on log axis

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) sensitivity vs C_ext
    ax = axes[0, 0]
    ax.semilogx(x, res["bend_sens"], 'o-', color='#d1495b', label='弯曲灵敏度 (κ)')
    ax.semilogx(x, res["comp_sens"], 's-', color='#2e86ab', label='压缩灵敏度 (ε)')
    ax.axvline(x[i_opt], color='green', ls='--', alpha=0.7, label='最优 $C_{ext}$')
    ax.set_xlabel('外部电容 $C_{ext}$ (pF)   [最左点 = 基线 0]')
    ax.set_ylabel('灵敏度峰值 (dB)')
    ax.set_title('(a) 灵敏度 vs 外部电容')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    style_logx(ax)

    # (b) RMSE vs C_ext
    ax = axes[0, 1]
    ax.semilogx(x, res["rmse_k_pct"], 'o-', color='#d1495b', label='弯曲 κ RMSE')
    ax.semilogx(x, res["rmse_e_pct"], 's-', color='#2e86ab', label='压缩 ε RMSE')
    ax.axvline(x[i_opt], color='green', ls='--', alpha=0.7)
    ax.plot(x[i_opt], rmse_k_arr[i_opt], '*', color='green', markersize=20,
            label='最优点')
    ax.set_xlabel('外部电容 $C_{ext}$ (pF)   [最左点 = 基线 0]')
    ax.set_ylabel('盲反演 RMSE (占量程 %)  @ SNR=60dB')
    ax.set_title('(b) 参数反演误差 vs 外部电容')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    style_logx(ax)

    # (c) SRF & flank freq vs C_ext
    ax = axes[1, 0]
    ax.loglog(x, np.array(res["srf"]) / 1e6, 'o-', color='#6a4c93', label='自谐振频率 SRF')
    ax.loglog(x, np.array(res["bend_flank_freq"]) / 1e6, '^-', color='#e07a5f',
              label='弯曲灵敏度峰值频率 (侧翼)')
    ax.axvline(x[i_opt], color='green', ls='--', alpha=0.7, label='最优 $C_{ext}$')
    ax.set_xlabel('外部电容 $C_{ext}$ (pF)   [最左点 = 基线 0]')
    ax.set_ylabel('频率 (MHz)')
    ax.set_title('(c) SRF 与最优测量频率 vs 外部电容')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    style_logx(ax)

    # (d) S11 spectrum comparison: baseline (=optimum) vs a naive large cap.
    # The optimum is the baseline (monotonic), so illustrate the DILUTION penalty
    # by contrasting against a naive "SRF-lowering" choice C_ext ~ C0 (=5 pF).
    ax = axes[1, 1]
    C_naive = 5e-12
    freq_b = make_freq(0.0)
    freq_n = make_freq(C_naive)
    for (lbl, cx, fr, col) in [
        ('最优=基线 $C_{ext}=0$', 0.0, freq_b, '#2e86ab'),
        (f'朴素大电容 $C_{{ext}}$=5 pF (稀释)', C_naive, freq_n, '#d1495b')]:
        ax.semilogx(fr / 1e6, s11_db(S11_ext(0.0, 0.0, fr, cx)), '-', color=col,
                    label=f'{lbl}, κ=0')
        ax.semilogx(fr / 1e6, s11_db(S11_ext(0.0, KAP_MAX, fr, cx)), '--', color=col,
                    alpha=0.7, label=f'{lbl}, κ=100 m$^{{-1}}$')
    ax.set_xlabel('频率 (MHz)')
    ax.set_ylabel('|S11| (dB)')
    ax.set_title('(d) S11 谱: 基线 vs 朴素大电容 (弯曲κ对比)')
    ax.legend(fontsize=7.5)
    ax.grid(True, which='both', alpha=0.3)

    # summary text box
    txt = (
        "===== M4 外部电容优化结论 =====\n"
        f"最优 C_ext = {C_opt*1e12:.3f} pF\n"
        f"SRF: {baseline['srf']/1e6:.0f} MHz  ->  {optimum['srf']/1e6:.0f} MHz\n"
        f"弯曲峰值频率(最优): {optimum['bend_flank_freq']/1e6:.0f} MHz\n"
        f"κ RMSE@60dB: {baseline['rmse_k_pct']:.2f}%  ->  {optimum['rmse_k_pct']:.2f}%\n"
        f"ε RMSE@60dB: {baseline['rmse_e_pct']:.2f}%  ->  {optimum['rmse_e_pct']:.2f}%\n"
        f"弯曲灵敏度: {baseline['bend_sens']:.4f}  ->  {optimum['bend_sens']:.4f} dB\n"
        f"稀释权衡最优(内部极值): {'是' if kappa_interior_opt else '否(单调)'}"
    )
    fig.text(0.5, -0.01, txt, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', fc='#f5f5f5', ec='gray'))

    fig.suptitle('图表10  M4 外部调谐电容优化 (弹簧线圈 VNA 传感器, a=1.5mm)',
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    png_path = os.path.join(outdir, "图表10_M4外部电容优化.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("FINDINGS")
    print("=" * 68)
    print(f"Optimal C_ext = {C_opt*1e12:.3f} pF (index {i_opt})")
    print(f"Baseline (C_ext=0): SRF={baseline['srf']/1e6:.0f} MHz, "
          f"kappa RMSE={baseline['rmse_k_pct']:.3f}%, eps RMSE={baseline['rmse_e_pct']:.3f}%, "
          f"bend_sens={baseline['bend_sens']:.4f} dB")
    print(f"Optimal:            SRF={optimum['srf']/1e6:.0f} MHz, "
          f"kappa RMSE={optimum['rmse_k_pct']:.3f}%, eps RMSE={optimum['rmse_e_pct']:.3f}%, "
          f"bend_sens={optimum['bend_sens']:.4f} dB")
    print(f"Optimal measurement (bending flank) freq = {optimum['bend_flank_freq']/1e6:.0f} MHz")
    print(f"Bending-sensitivity peak at C_ext={C_arr[i_bend_max]*1e12:.3f} pF "
          f"({bend_arr[i_bend_max]:.4f} dB); interior peak = {sens_interior_peak}")
    print(f"kappa-RMSE optimum is interior (dilution tradeoff present) = {kappa_interior_opt}")
    improvement = (baseline['rmse_k_pct'] - optimum['rmse_k_pct']) / baseline['rmse_k_pct'] * 100
    print(f"kappa RMSE improvement vs baseline = {improvement:.1f}%")
    print(f"\nFiles:\n  {png_path}\n  {json_path}")


if __name__ == "__main__":
    main()
