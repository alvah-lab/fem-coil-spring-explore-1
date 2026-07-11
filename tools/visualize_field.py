#!/usr/bin/env python3
"""
Compute and visualize magnetic field for a coil geometry using Biot-Savart law.
Note: Assumes 1A excitation; for actual field use FastHenry current distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.coilgen import spiral_helix, straight_wire


def biot_savart_segment(point, r1, r2, current=1.0):
	"""
	Magnetic field at 'point' due to current element from r1 to r2.
	Uses Biot-Savart law for finite wire segment.

	Args:
		point: observation point (x, y, z)
		r1: segment start
		r2: segment end
		current: current (A)

	Returns:
		B field vector at point
	"""
	mu0 = 4 * np.pi * 1e-7

	dl = r2 - r1
	dl_mag = np.linalg.norm(dl)

	if dl_mag < 1e-9:
		return np.zeros(3)

	R1 = point - r1
	R2 = point - r2
	R1_mag = np.linalg.norm(R1)
	R2_mag = np.linalg.norm(R2)

	if R1_mag < 1e-9 or R2_mag < 1e-9:
		return np.zeros(3)

	numerator = np.cross(dl, R1) / (R1_mag**3) - np.cross(dl, R2) / (R2_mag**3)
	B = (mu0 * current / (4 * np.pi * dl_mag)) * numerator

	return B


def compute_field_plane(nodes, segments, plane_z=0.0, grid_size=50, extent=15):
	"""
	Compute magnetic field in a horizontal plane at z=plane_z.

	Args:
		nodes: list of (x, y, z) node coordinates
		segments: list of (node_idx1, node_idx2) segment indices
		plane_z: z-coordinate of plane to visualize
		grid_size: resolution (grid_size x grid_size points)
		extent: ±extent (mm) in x and y

	Returns:
		x_grid, y_grid, B_magnitude, B_x, B_y
	"""
	x = np.linspace(-extent, extent, grid_size)
	y = np.linspace(-extent, extent, grid_size)
	xx, yy = np.meshgrid(x, y)

	B_mag = np.zeros((grid_size, grid_size))
	B_x = np.zeros((grid_size, grid_size))
	B_y = np.zeros((grid_size, grid_size))

	for i in range(grid_size):
		for j in range(grid_size):
			point = np.array([xx[i, j], yy[i, j], plane_z])
			B_total = np.zeros(3)

			for seg_i1, seg_i2 in segments:
				r1 = np.array(nodes[seg_i1])
				r2 = np.array(nodes[seg_i2])
				B_total += biot_savart_segment(point, r1, r2, current=1.0)

			B_mag[i, j] = np.linalg.norm(B_total)
			B_x[i, j] = B_total[0]
			B_y[i, j] = B_total[1]

	return xx, yy, B_mag, B_x, B_y


def plot_field(nodes, segments, geometry_name, plane_z=0.0, output_file=None):
	"""
	Plot magnetic field in a plane with coil cross-section overlay.
	"""
	print(f"Computing field for {geometry_name}...")
	xx, yy, B_mag, B_x, B_y = compute_field_plane(nodes, segments, plane_z=plane_z)

	print("Creating visualization...")
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

	# Plot 1: Magnitude with contours
	levels = np.linspace(B_mag.min(), B_mag.max(), 20)
	contourf = ax1.contourf(xx, yy, B_mag, levels=levels, cmap='hot', extend='both')
	contour = ax1.contour(xx, yy, B_mag, levels=levels[::3], colors='black',
	                        linewidths=0.5, alpha=0.3)
	ax1.clabel(contour, inline=True, fontsize=8)
	cbar1 = plt.colorbar(contourf, ax=ax1, label='|B| (Tesla)')

	# Overlay coil geometry on plane
	for seg_i1, seg_i2 in segments:
		n1 = np.array(nodes[seg_i1])
		n2 = np.array(nodes[seg_i2])
		if abs(n1[2] - plane_z) < 0.5 and abs(n2[2] - plane_z) < 0.5:
			ax1.plot([n1[0], n2[0]], [n1[1], n2[1]], 'b-', linewidth=2, alpha=0.7)

	ax1.set_xlabel('x (mm)')
	ax1.set_ylabel('y (mm)')
	ax1.set_title(f'{geometry_name}: Magnetic Field Magnitude at z={plane_z:.1f}mm')
	ax1.grid(True, alpha=0.2)
	ax1.set_aspect('equal')

	# Plot 2: Vector field (quiver)
	scale = 50
	skip = 3
	q = ax2.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
	               B_x[::skip, ::skip], B_y[::skip, ::skip],
	               B_mag[::skip, ::skip], cmap='hot', scale=scale)
	plt.colorbar(q, ax=ax2, label='|B| (Tesla)')

	# Overlay coil geometry
	for seg_i1, seg_i2 in segments:
		n1 = np.array(nodes[seg_i1])
		n2 = np.array(nodes[seg_i2])
		if abs(n1[2] - plane_z) < 0.5 and abs(n2[2] - plane_z) < 0.5:
			ax2.plot([n1[0], n2[0]], [n1[1], n2[1]], 'b-', linewidth=2, alpha=0.7)

	ax2.set_xlabel('x (mm)')
	ax2.set_ylabel('y (mm)')
	ax2.set_title(f'{geometry_name}: Vector Field at z={plane_z:.1f}mm')
	ax2.grid(True, alpha=0.2)
	ax2.set_aspect('equal')

	plt.tight_layout()

	if output_file:
		plt.savefig(output_file, dpi=120)
		print(f"✓ Saved: {output_file}")
	else:
		plt.show()

	plt.close()


def main():
	print("\nGenerating magnetic field visualizations...\n")

	# Test 1: Straight wire (x-y plane at midpoint)
	print("1. Straight wire...")
	nodes_w, segs_w = straight_wire(100, 10, 0.5, start=(0, 0, 0), direction=(0, 0, 1))
	plot_field(nodes_w, segs_w, "Straight Wire (100mm)", plane_z=50,
	          output_file="tests/field_straight_wire.png")

	# Test 2: Single coil (top view)
	print("2. Single coil...")
	nodes_c, segs_c = spiral_helix(5, 1, 10, 16, 0.5, start_z=0)
	coil_z_mid = (10 * 1.0) / 2
	plot_field(nodes_c, segs_c, "Single Spiral Coil", plane_z=coil_z_mid,
	          output_file="tests/field_single_coil.png")

	# Test 3: Single coil (side view, x-z plane at y=0)
	print("3. Single coil (side view)...")
	# For side view, need to recompute in different plane
	# This requires rewriting the function to handle arbitrary planes
	print("   (Side view requires 3D visualization - skipped for now)")

	print("\n✓ Magnetic field visualizations complete!")
	print("   Note: Assumes 1A excitation; actual field proportional to current")


if __name__ == "__main__":
	main()
