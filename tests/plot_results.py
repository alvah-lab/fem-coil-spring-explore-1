#!/usr/bin/env python3
"""
Visualize FastHenry results and compare with analytical formulas.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.run_fh import (
	FastHenryRunner,
	analytical_straight_wire,
	analytical_solenoid
)


def plot_straight_wire():
	"""Plot straight wire inductance vs frequency"""
	inp_file = "examples/test_straight_wire.inp"
	runner = FastHenryRunner()
	runner.run(inp_file, verbose=False)

	freq, Z = runner.load_zc_matrix()
	result = runner.extract_inductance(freq, Z)

	L_fh = result["inductance_uh"]["L0"]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	ax1.semilogx(freq, L_fh * 1e6, 'b-o', label='FastHenry', markersize=3)
	L_analytical = analytical_straight_wire(100.0, 0.5)
	ax1.axhline(L_analytical, color='r', linestyle='--', label='Analytical')
	ax1.set_xlabel('Frequency (Hz)')
	ax1.set_ylabel('Inductance (nH)')
	ax1.set_title('Straight Wire: 100mm × 0.5mm radius')
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	R_real = np.real(Z[:, 0, 0])
	ax2.loglog(freq, R_real * 1e6, 'g-o', markersize=3)
	ax2.set_xlabel('Frequency (Hz)')
	ax2.set_ylabel('Resistance (µΩ)')
	ax2.set_title('Resistance vs Frequency (Skin Effect)')
	ax2.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('tests/plot_straight_wire.png', dpi=100)
	print("✓ Saved: tests/plot_straight_wire.png")
	plt.close()


def plot_single_coil():
	"""Plot single coil inductance vs frequency"""
	inp_file = "examples/test_single_coil.inp"
	runner = FastHenryRunner()
	runner.run(inp_file, verbose=False)

	freq, Z = runner.load_zc_matrix()
	result = runner.extract_inductance(freq, Z)

	L_fh = result["inductance_uh"]["L0"]

	fig, ax = plt.subplots(figsize=(10, 5))

	ax.semilogx(freq, L_fh, 'b-o', label='FastHenry', markersize=4)

	L_wheeler = analytical_solenoid(5.0, 1.0, 10, 0.5)
	ax.axhline(L_wheeler, color='r', linestyle='--', linewidth=2,
	           label=f'Wheeler Formula ({L_wheeler:.3f} µH)')

	ax.fill_between(freq, L_wheeler*0.5, L_wheeler*1.5, alpha=0.2, color='red',
	                label='±50% of Wheeler')

	ax.set_xlabel('Frequency (Hz)', fontsize=11)
	ax.set_ylabel('Inductance (µH)', fontsize=11)
	ax.set_title('Single Spiral Coil: 10 turns, 5mm radius, 1mm pitch', fontsize=12)
	ax.legend(fontsize=10)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('tests/plot_single_coil.png', dpi=100)
	print("✓ Saved: tests/plot_single_coil.png")
	plt.close()


def plot_dual_coil():
	"""Plot dual coil inductances and coupling factor"""
	inp_file = "examples/test_dual_coil.inp"
	runner = FastHenryRunner()
	runner.run(inp_file, verbose=False)

	freq, Z = runner.load_zc_matrix()
	result = runner.extract_inductance(freq, Z)

	L0 = result["inductance_uh"]["L0"]
	L1 = result["inductance_uh"]["L1"]
	M = result["inductance_uh"]["L01"]
	k = result["coupling_factor"]

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

	# Inductances
	ax1.semilogx(freq, L0, 'b-o', label='L₀ (Coil 1)', markersize=3)
	ax1.semilogx(freq, L1, 'r-s', label='L₁ (Coil 2)', markersize=3)
	ax1.set_ylabel('Inductance (µH)')
	ax1.set_title('Self-Inductances')
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# Mutual inductance
	max_M = np.sqrt(np.abs(L0 * L1))
	ax2.semilogx(freq, M, 'g-o', label='M (measured)', markersize=3)
	ax2.semilogx(freq, max_M, 'k--', alpha=0.5, label='√(L₀·L₁) (max possible)')
	ax2.set_ylabel('Mutual Inductance (µH)')
	ax2.set_title('Mutual Inductance')
	ax2.legend()
	ax2.grid(True, alpha=0.3)

	# Coupling factor
	ax3.semilogx(freq, k, 'purple', marker='o', markersize=3)
	ax3.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='k=1 (perfect coupling)')
	ax3.set_ylabel('Coupling Factor k')
	ax3.set_xlabel('Frequency (Hz)')
	ax3.set_title('Coupling Factor')
	ax3.set_ylim([0, 1.1])
	ax3.legend()
	ax3.grid(True, alpha=0.3)

	# Energy ratio
	ax4.semilogx(freq, (M**2 / (L0*L1)) * 100, 'orange', marker='o', markersize=3)
	ax4.set_ylabel('M²/(L₀·L₁) × 100 (%)')
	ax4.set_xlabel('Frequency (Hz)')
	ax4.set_title('Coupled Energy Ratio')
	ax4.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('tests/plot_dual_coil.png', dpi=100)
	print("✓ Saved: tests/plot_dual_coil.png")
	plt.close()


def main():
	print("\nGenerating plots...")
	try:
		plot_straight_wire()
		plot_single_coil()
		plot_dual_coil()
		print("\n✓ All plots generated successfully!")
		return 0
	except Exception as e:
		print(f"\n✗ Error: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == "__main__":
	sys.exit(main())
