#!/usr/bin/env python3
"""
Sweep mutual inductance as two planar spirals move apart.
Generates series of .inp files with increasing separation,
simulates each, and plots inductance evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.coilgen import gen_dual_planar_spiral
from tools.run_fh import FastHenryRunner


def sweep_dual_coils(output_dir="sweep_data", num_points=11,
                     turns=10, inner_r=1, outer_r=5, wire_r=0.3):
	"""
	Generate dual planar spiral configurations at different separations.

	Args:
		output_dir: directory for .inp files and results
		num_points: number of separation points to sweep
		turns: number of turns per coil
		inner_r: inner radius (mm)
		outer_r: outer radius (mm)
		wire_r: wire radius (mm)

	Returns:
		gaps, L0_list, L1_list, M_list, k_list (all vs gap)
	"""
	os.makedirs(output_dir, exist_ok=True)

	max_gap = outer_r  # Sweep from 0 to outer_radius
	gaps = np.linspace(0, max_gap, num_points)

	L0_list = []
	L1_list = []
	M_list = []
	k_list = []
	runner = FastHenryRunner()

	print(f"\n{'='*70}")
	print(f"SWEEPING DUAL PLANAR SPIRALS")
	print(f"{'='*70}")
	print(f"Turns: {turns}")
	print(f"Inner radius: {inner_r} mm")
	print(f"Outer radius: {outer_r} mm")
	print(f"Wire radius: {wire_r} mm")
	print(f"Sweep range: gap = 0 to {max_gap} mm ({num_points} points)")
	print(f"{'='*70}\n")

	for idx, gap in enumerate(gaps):
		print(f"[{idx+1}/{len(gaps)}] Gap = {gap:.2f} mm ... ", end="", flush=True)

		inp_file = os.path.join(output_dir, f"dual_gap_{gap:.2f}.inp")
		gen_dual_planar_spiral(
			inp_file,
			turns=turns,
			inner_radius=inner_r,
			outer_radius=outer_r,
			wire_radius=wire_r,
			gap=gap,
			title=f"Dual Planar Spiral (gap={gap:.2f}mm)"
		)

		if runner.run(inp_file, verbose=False):
			freq, Z = runner.load_zc_matrix()
			if freq is not None:
				result = runner.extract_inductance(freq, Z)

				# Extract at 1 GHz (last frequency)
				L0 = result['inductance_uh']['L0'][-1]
				L1 = result['inductance_uh']['L1'][-1]
				M = result['inductance_uh']['L01'][-1]
				k = result['coupling_factor'][-1]

				L0_list.append(L0)
				L1_list.append(L1)
				M_list.append(M)
				k_list.append(k)

				print(f"L0={L0:.3f} µH, L1={L1:.3f} µH, M={M:.3f} µH, k={k:.3f}")
			else:
				print("FAILED to extract results")
				L0_list.append(np.nan)
				L1_list.append(np.nan)
				M_list.append(np.nan)
				k_list.append(np.nan)
		else:
			print("FAILED to run simulation")
			L0_list.append(np.nan)
			L1_list.append(np.nan)
			M_list.append(np.nan)
			k_list.append(np.nan)

	return np.array(gaps), np.array(L0_list), np.array(L1_list), np.array(M_list), np.array(k_list)


def plot_sweep_results(gaps, L0, L1, M, k, output_file="sweep_results.png"):
	"""Plot inductance evolution during sweep."""
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

	# L0 and L1
	ax1.plot(gaps, L0, 'b-o', linewidth=2, markersize=6, label='L₀ (Coil 1)')
	ax1.plot(gaps, L1, 'r-s', linewidth=2, markersize=6, label='L₁ (Coil 2)')
	ax1.set_xlabel('Gap between coils (mm)', fontsize=11)
	ax1.set_ylabel('Self-inductance (µH)', fontsize=11)
	ax1.set_title('Self-Inductance vs Separation', fontsize=12, fontweight='bold')
	ax1.legend(fontsize=10)
	ax1.grid(True, alpha=0.3)
	ax1.set_xlim([gaps.min(), gaps.max()])

	# Mutual inductance
	ax2.plot(gaps, M, 'g-^', linewidth=2.5, markersize=7, label='M (measured)')
	sqrt_prod = np.sqrt(np.abs(L0 * L1))
	ax2.plot(gaps, sqrt_prod, 'k--', alpha=0.5, linewidth=2, label='√(L₀·L₁) (max possible)')
	ax2.fill_between(gaps, 0, M, alpha=0.2, color='green')
	ax2.set_xlabel('Gap between coils (mm)', fontsize=11)
	ax2.set_ylabel('Inductance (µH)', fontsize=11)
	ax2.set_title('Mutual Inductance vs Separation', fontsize=12, fontweight='bold')
	ax2.legend(fontsize=10)
	ax2.grid(True, alpha=0.3)
	ax2.set_xlim([gaps.min(), gaps.max()])

	# Coupling factor
	ax3.plot(gaps, k, 'purple', marker='o', linewidth=2.5, markersize=7)
	ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='k=1 (perfect)')
	ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='k=0 (no coupling)')
	ax3.fill_between(gaps, 0, k, alpha=0.2, color='purple')
	ax3.set_xlabel('Gap between coils (mm)', fontsize=11)
	ax3.set_ylabel('Coupling factor k', fontsize=11)
	ax3.set_title('Coupling Factor vs Separation', fontsize=12, fontweight='bold')
	ax3.set_ylim([0, 1.1])
	ax3.legend(fontsize=10)
	ax3.grid(True, alpha=0.3)
	ax3.set_xlim([gaps.min(), gaps.max()])

	# Energy coupling ratio
	energy_ratio = (M**2 / (L0 * L1)) * 100
	ax4.plot(gaps, energy_ratio, 'orange', marker='s', linewidth=2.5, markersize=7)
	ax4.fill_between(gaps, 0, energy_ratio, alpha=0.2, color='orange')
	ax4.set_xlabel('Gap between coils (mm)', fontsize=11)
	ax4.set_ylabel('Coupled Energy %', fontsize=11)
	ax4.set_title('M²/(L₀·L₁) vs Separation', fontsize=12, fontweight='bold')
	ax4.grid(True, alpha=0.3)
	ax4.set_xlim([gaps.min(), gaps.max()])

	plt.tight_layout()
	plt.savefig(output_file, dpi=120)
	print(f"\n✓ Saved sweep plot: {output_file}")
	plt.close()


def print_summary(gaps, L0, L1, M, k):
	"""Print summary statistics."""
	print(f"\n{'='*70}")
	print("SWEEP SUMMARY")
	print(f"{'='*70}")
	print(f"\nGap range: {gaps.min():.2f} - {gaps.max():.2f} mm")
	print(f"\nAt close proximity (gap={gaps[0]:.2f} mm):")
	print(f"  L₀ = {L0[0]:.3f} µH")
	print(f"  L₁ = {L1[0]:.3f} µH")
	print(f"  M  = {M[0]:.3f} µH")
	print(f"  k  = {k[0]:.3f}")

	print(f"\nAt maximum separation (gap={gaps[-1]:.2f} mm):")
	print(f"  L₀ = {L0[-1]:.3f} µH")
	print(f"  L₁ = {L1[-1]:.3f} µH")
	print(f"  M  = {M[-1]:.3f} µH")
	print(f"  k  = {k[-1]:.3f}")

	print(f"\nChange from close to far:")
	print(f"  ΔL₀ = {(L0[-1] - L0[0])/L0[0] * 100:+.1f}%")
	print(f"  ΔL₁ = {(L1[-1] - L1[0])/L1[0] * 100:+.1f}%")
	print(f"  ΔM  = {(M[-1] - M[0])/M[0] * 100:+.1f}%")
	print(f"  Δk  = {k[-1] - k[0]:+.3f}")
	print(f"{'='*70}\n")


def main():
	print("\n" + "="*70)
	print("MUTUAL INDUCTANCE SWEEP - Planar Spirals Moving Apart")
	print("="*70)

	# Parameters
	turns = 10
	inner_radius = 1.0
	outer_radius = 5.0
	wire_radius = 0.3
	num_sweep_points = 11

	# Run sweep
	gaps, L0, L1, M, k = sweep_dual_coils(
		output_dir="sweep_data",
		num_points=num_sweep_points,
		turns=turns,
		inner_r=inner_radius,
		outer_r=outer_radius,
		wire_r=wire_radius
	)

	# Plot results
	plot_sweep_results(gaps, L0, L1, M, k, output_file="tests/sweep_planar_spirals.png")

	# Print summary
	print_summary(gaps, L0, L1, M, k)

	# Save data
	np.savez("tests/sweep_data.npz", gaps=gaps, L0=L0, L1=L1, M=M, k=k)
	print("✓ Saved sweep data: tests/sweep_data.npz")


if __name__ == "__main__":
	main()
