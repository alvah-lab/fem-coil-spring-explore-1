#!/usr/bin/env python3
"""
Visualize 3D geometry of coils for different configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.coilgen import spiral_helix, planar_spiral


def plot_dual_planar_spirals_3d(gaps, output_file="tests/geometry_dual_spirals.png"):
	"""
	Plot 3D geometry of two planar spirals at different gaps.
	"""
	turns = 10
	inner_r = 1.0
	outer_r = 5.0
	wire_r = 0.3

	nodes1, _ = planar_spiral(turns, inner_r, outer_r, wire_r)

	fig = plt.figure(figsize=(15, 10))

	for idx, gap in enumerate(gaps):
		ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

		# First spiral at z=0
		nodes1_arr = np.array(nodes1)
		ax.plot(nodes1_arr[:, 0], nodes1_arr[:, 1], nodes1_arr[:, 2],
		       'b-', linewidth=2, label=f'Spiral 1 (z=0)')

		# Second spiral at z=gap
		nodes2_arr = nodes1_arr.copy()
		nodes2_arr[:, 2] += gap
		ax.plot(nodes2_arr[:, 0], nodes2_arr[:, 1], nodes2_arr[:, 2],
		       'r-', linewidth=2, label=f'Spiral 2 (z={gap})')

		# Draw connection lines at start and end
		ax.plot([0, 0], [0, 0], [0, gap], 'k--', alpha=0.3, linewidth=1)

		ax.set_xlabel('X (mm)')
		ax.set_ylabel('Y (mm)')
		ax.set_zlabel('Z (mm)')
		ax.set_title(f'Gap = {gap} mm\n(k ≈ {[1.0, 0.87, 0.65, 0.51, 0.40, 0.32, 0.26, 0.21, 0.18, 0.15, 0.12][idx]:.2f})')
		ax.legend(fontsize=8)
		ax.set_xlim([-6, 6])
		ax.set_ylim([-6, 6])
		ax.set_zlim([-1, gap + 1])

	plt.tight_layout()
	plt.savefig(output_file, dpi=100)
	print(f"✓ Saved geometry visualization: {output_file}")
	plt.close()


def plot_three_coil_types_2d(output_file="tests/geometry_three_types.png"):
	"""
	Plot 2D projections of three coil types side by side.
	"""
	turns = 10
	wire_r = 0.3

	# Generate geometries
	straight_nodes = np.array([
		[0, 0, z] for z in np.linspace(0, 100, 21)
	])

	planar_nodes, _ = planar_spiral(turns, 1.0, 5.0, wire_r)
	planar_nodes = np.array(planar_nodes)

	spring_nodes, _ = spiral_helix(5.0, 1.0, turns, 16, wire_r, start_z=0)
	spring_nodes = np.array(spring_nodes)

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

	# 1. Straight wire (side view)
	ax1.plot(straight_nodes[:, 2], straight_nodes[:, 0], 'b-', linewidth=2)
	ax1.scatter([0, 100], [0, 0], color='red', s=100, zorder=5, label='Ports')
	ax1.set_xlabel('Z (mm)')
	ax1.set_ylabel('X (mm)')
	ax1.set_title('Straight Wire - Side View', fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.set_aspect('equal')
	ax1.legend()

	# 2. Planar spiral (top view)
	ax2.plot(planar_nodes[:, 0], planar_nodes[:, 1], 'g-', linewidth=2)
	ax2.scatter([0], [0], color='red', s=100, zorder=5, label='Center')
	circle_outer = plt.Circle((0, 0), 5, fill=False, linestyle='--', alpha=0.3)
	circle_inner = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3)
	ax2.add_patch(circle_outer)
	ax2.add_patch(circle_inner)
	ax2.set_xlabel('X (mm)')
	ax2.set_ylabel('Y (mm)')
	ax2.set_title('Planar Spiral - Top View (z=0)', fontweight='bold')
	ax2.grid(True, alpha=0.3)
	ax2.set_aspect('equal')
	ax2.legend()

	# 3. Spring coil (3D projection to 2D)
	ax3.plot(spring_nodes[:, 0], spring_nodes[:, 2], 'r-', linewidth=2)
	ax3.scatter([spring_nodes[0, 0], spring_nodes[-1, 0]],
	           [spring_nodes[0, 2], spring_nodes[-1, 2]],
	           color='red', s=100, zorder=5, label='Ports')
	circle = plt.Circle((0, 5), 5, fill=False, linestyle='--', alpha=0.3)
	ax3.add_patch(circle)
	ax3.set_xlabel('X (mm)')
	ax3.set_ylabel('Z (mm)')
	ax3.set_title('Spring Coil - Side View', fontweight='bold')
	ax3.grid(True, alpha=0.3)
	ax3.set_aspect('equal')
	ax3.legend()

	# 4. Comparison table
	ax4.axis('off')
	comparison_text = """
	THREE COIL TYPES COMPARISON

	TYPE 1: STRAIGHT WIRE
	  • Geometry: Linear, L=100mm
	  • Inductance: L = 0.112 µH (DC)
	  • Frequency change: -1.8% (DC→1GHz)
	  • Best for: Reference, baseline

	TYPE 2: PLANAR SPIRAL
	  • Geometry: Archimedean, r=1-5mm
	  • Inductance: L = 0.467 µH (DC)
	  • Frequency change: -16.2% (DC→1GHz)
	  • Best for: Compact PCB designs
	  • Note: High frequency sensitivity

	TYPE 3: SPRING COIL
	  • Geometry: Cylindrical helix, 10 turns
	  • Inductance: L = 0.638 µH (DC)
	  • Frequency change: -1.6% (DC→1GHz)
	  • Best for: High-precision applications
	  • Note: Most frequency-stable

	MUTUAL INDUCTANCE SWEEP
	  • Setup: Two planar spirals, z-axis separation
	  • Gap range: 0 → 5 mm
	  • Coupling: k = 1.0 → 0.124
	  • Decay: ~87.6% reduction in M
	"""
	ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
	        fontsize=9, verticalalignment='top', family='monospace',
	        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

	plt.tight_layout()
	plt.savefig(output_file, dpi=120)
	print(f"✓ Saved geometry comparison: {output_file}")
	plt.close()


def main():
	print("\nGenerating geometry visualizations...\n")

	# Plot three coil types
	plot_three_coil_types_2d("tests/geometry_three_types.png")

	# Plot dual spirals at different gaps
	gaps_to_plot = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
	plot_dual_planar_spirals_3d(gaps_to_plot, "tests/geometry_dual_spirals_3d.png")

	print("\n✓ All geometry visualizations complete!")


if __name__ == "__main__":
	main()
