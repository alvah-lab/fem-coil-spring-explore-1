#!/usr/bin/env python3
"""
Compare three coil types: straight, planar spiral, cylindrical spring.
Run FEM on each and visualize frequency response.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.coilgen import (
	gen_straight_wire,
	gen_single_coil,
	gen_planar_spiral
)
from tools.run_fh import FastHenryRunner


def compare_coil_types():
	"""Generate and simulate three coil geometries."""

	print("\n" + "="*70)
	print("COMPARING THREE COIL TYPES")
	print("="*70)

	# Common parameters
	turns = 10
	wire_radius = 0.3

	# Test case 1: Straight wire (100mm, same "length" as spirals)
	print("\n[1/3] Straight wire (100mm)...")
	gen_straight_wire(
		"examples/compare_straight.inp",
		length=100,
		segs=20,
		wire_radius=wire_radius,
		title="Straight Wire 100mm"
	)

	# Test case 2: Planar spiral (Archimedean)
	print("[2/3] Planar spiral (10 turns, 1-5mm radius)...")
	from tools.coilgen import gen_planar_spiral
	gen_planar_spiral(
		"examples/compare_planar_spiral.inp",
		turns=turns,
		inner_radius=1.0,
		outer_radius=5.0,
		wire_radius=wire_radius,
		title="Planar Spiral 10 turns"
	)

	# Test case 3: Cylindrical spring (helix)
	print("[3/3] Cylindrical spring (10 turns, 5mm radius)...")
	gen_single_coil(
		"examples/compare_spring.inp",
		radius=5.0,
		pitch=1.0,
		num_turns=turns,
		segs_per_turn=16,
		wire_radius=wire_radius,
		title="Spring Coil 10 turns"
	)

	# Simulate each
	runner = FastHenryRunner()
	results = {}

	configs = [
		("Straight Wire", "examples/compare_straight.inp"),
		("Planar Spiral", "examples/compare_planar_spiral.inp"),
		("Spring Coil", "examples/compare_spring.inp"),
	]

	print("\n" + "="*70)
	print("RUNNING SIMULATIONS")
	print("="*70)

	for name, inp_file in configs:
		print(f"\nSimulating {name}...")
		if runner.run(inp_file, verbose=False):
			freq, Z = runner.load_zc_matrix()
			if freq is not None:
				result = runner.extract_inductance(freq, Z)
				results[name] = result
				L_dc = result['inductance_uh']['L0'][0]
				L_1ghz = result['inductance_uh']['L0'][-1]
				print(f"  ✓ L(DC)   = {L_dc:.4f} µH")
				print(f"  ✓ L(1GHz) = {L_1ghz:.4f} µH")
			else:
				print(f"  ✗ Failed to load results")
		else:
			print(f"  ✗ Simulation failed")

	return results


def plot_comparison(results, output_file="tests/coil_comparison.png"):
	"""Plot frequency response comparison of three coil types."""

	if not results or len(results) < 3:
		print("Not enough data to plot")
		return

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

	colors = {'Straight Wire': 'blue', 'Planar Spiral': 'green', 'Spring Coil': 'red'}
	markers = {'Straight Wire': 'o', 'Planar Spiral': 's', 'Spring Coil': '^'}

	# Plot 1: Inductance vs frequency (linear scale)
	for name, result in results.items():
		freq = result['frequencies_hz']
		L = result['inductance_uh']['L0'] * 1e6  # Convert to nH for visibility
		ax1.semilogx(freq, L, color=colors[name], marker=markers[name],
		            linewidth=2, markersize=4, label=name, alpha=0.7)

	ax1.set_xlabel('Frequency (Hz)', fontsize=11)
	ax1.set_ylabel('Inductance (nH)', fontsize=11)
	ax1.set_title('Self-Inductance vs Frequency', fontsize=12, fontweight='bold')
	ax1.legend(fontsize=10)
	ax1.grid(True, alpha=0.3)

	# Plot 2: Inductance (µH)
	for name, result in results.items():
		freq = result['frequencies_hz']
		L = result['inductance_uh']['L0']
		ax2.semilogx(freq, L, color=colors[name], marker=markers[name],
		            linewidth=2, markersize=4, label=name, alpha=0.7)

	ax2.set_xlabel('Frequency (Hz)', fontsize=11)
	ax2.set_ylabel('Inductance (µH)', fontsize=11)
	ax2.set_title('Self-Inductance (µH) vs Frequency', fontsize=12, fontweight='bold')
	ax2.legend(fontsize=10)
	ax2.grid(True, alpha=0.3)

	# Plot 3: Resistance (showing skin effect)
	for name, result in results.items():
		freq = result['frequencies_hz']
		Z = result['raw_impedance']
		R = np.real(Z[:, 0, 0]) * 1e6  # Convert to µΩ
		ax3.loglog(freq, R, color=colors[name], marker=markers[name],
		          linewidth=2, markersize=4, label=name, alpha=0.7)

	ax3.set_xlabel('Frequency (Hz)', fontsize=11)
	ax3.set_ylabel('Resistance (µΩ)', fontsize=11)
	ax3.set_title('Resistance vs Frequency (Skin Effect)', fontsize=12, fontweight='bold')
	ax3.legend(fontsize=10)
	ax3.grid(True, alpha=0.3, which='both')

	# Plot 4: Relative change from DC
	for name, result in results.items():
		freq = result['frequencies_hz']
		L = result['inductance_uh']['L0']
		L_dc = L[0]
		rel_change = (L - L_dc) / L_dc * 100
		ax4.semilogx(freq, rel_change, color=colors[name], marker=markers[name],
		            linewidth=2, markersize=4, label=name, alpha=0.7)

	ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
	ax4.set_xlabel('Frequency (Hz)', fontsize=11)
	ax4.set_ylabel('Relative change from DC (%)', fontsize=11)
	ax4.set_title('Inductance Change vs DC Value', fontsize=12, fontweight='bold')
	ax4.legend(fontsize=10)
	ax4.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(output_file, dpi=120)
	print(f"\n✓ Saved comparison plot: {output_file}")
	plt.close()


def print_summary(results):
	"""Print detailed comparison table."""

	print("\n" + "="*70)
	print("INDUCTANCE COMPARISON TABLE")
	print("="*70)

	print(f"\n{'Coil Type':<20} {'DC (µH)':<12} {'1 GHz (µH)':<12} {'Change %':<12}")
	print("-" * 70)

	for name, result in results.items():
		L_dc = result['inductance_uh']['L0'][0]
		L_1ghz = result['inductance_uh']['L0'][-1]
		change_pct = (L_1ghz - L_dc) / L_dc * 100

		print(f"{name:<20} {L_dc:<12.4f} {L_1ghz:<12.4f} {change_pct:<+12.2f}%")

	print("="*70 + "\n")


def main():
	# Run simulations
	results = compare_coil_types()

	if results:
		# Create comparison plot
		plot_comparison(results)

		# Print summary
		print_summary(results)

		# Save data
		import json
		summary = {}
		for name, result in results.items():
			summary[name] = {
				'L_DC_uH': float(result['inductance_uh']['L0'][0]),
				'L_1GHz_uH': float(result['inductance_uh']['L0'][-1]),
				'change_pct': float((result['inductance_uh']['L0'][-1] - result['inductance_uh']['L0'][0]) / result['inductance_uh']['L0'][0] * 100),
			}

		with open('tests/coil_comparison_summary.json', 'w') as f:
			json.dump(summary, f, indent=2)

		print("✓ Saved summary: tests/coil_comparison_summary.json")

		return 0
	else:
		print("✗ No valid results to plot")
		return 1


if __name__ == "__main__":
	sys.exit(main())
