#!/usr/bin/env python3
import numpy as np
import math


def spiral_helix(radius, pitch, num_turns, segs_per_turn, wire_radius, start_z=0.0):
	"""
	Generate 3D coordinates for a cylindrical helix (spring coil).

	Args:
		radius: Helix radius (mm)
		pitch: Vertical spacing per turn (mm)
		num_turns: Number of complete turns
		segs_per_turn: Number of segments per turn
		wire_radius: Wire radius for cross-section (mm)
		start_z: Starting z coordinate (mm)

	Returns:
		nodes: List of (x, y, z) tuples
		segments: List of (node_idx1, node_idx2) tuples
	"""
	total_segs = num_turns * segs_per_turn
	nodes = []
	segments = []

	for i in range(total_segs + 1):
		t = 2 * np.pi * i / segs_per_turn
		z = start_z + (pitch / segs_per_turn) * i
		x = radius * np.cos(t)
		y = radius * np.sin(t)
		nodes.append((x, y, z))

	for i in range(total_segs):
		segments.append((i, i + 1))

	return nodes, segments


def straight_wire(length, segs, wire_radius, start=(0, 0, 0), direction=(0, 0, 1)):
	"""
	Generate straight wire coordinates.

	Args:
		length: Wire length (mm)
		segs: Number of segments
		wire_radius: Wire radius (mm)
		start: Starting position (x, y, z)
		direction: Unit direction vector

	Returns:
		nodes, segments
	"""
	direction = np.array(direction) / np.linalg.norm(direction)
	nodes = []
	segments = []

	for i in range(segs + 1):
		pos = np.array(start) + direction * (length * i / segs)
		nodes.append(tuple(pos))

	for i in range(segs):
		segments.append((i, i + 1))

	return nodes, segments


def gen_inp_file(filename, nodes_list, segments_list, external_ports, freq_list=None,
				 conductivity=5.8e7, wire_radius=None, title="FEM Inductance Simulation"):
	"""
	Generate FastHenry .inp file from node/segment lists.

	Args:
		filename: Output .inp file path
		nodes_list: List of (coil_name, nodes) tuples
		segments_list: List of (coil_name, segments, wire_radius) tuples
		external_ports: List of (coil_name, port_idx_list) tuples defining external nodes
		freq_list: Frequencies to simulate (Hz), default [0.1, 1e19]
		conductivity: Conductivity (S/m), default 5.8e7 for copper
		wire_radius: Default wire radius if not specified per coil (mm)
		title: Simulation title
	"""
	if freq_list is None:
		freq_list = [0.1, 1e19]

	with open(filename, 'w') as f:
		f.write(f"* {title}\n")
		f.write(".units mm\n")
		f.write(".default sigma={} nhinc=1 nwinc=4\n".format(conductivity))
		f.write("\n")

		node_id = 1
		node_map = {}
		seg_id = 1

		for coil_name, nodes in nodes_list:
			f.write(f"* Coil: {coil_name}\n")
			for i, (x, y, z) in enumerate(nodes):
				nid = f"{coil_name}_{i}"
				node_map[nid] = node_id
				f.write(f"N{node_id:d} x={x:.6f} y={y:.6f} z={z:.6f}\n")
				node_id += 1

		f.write("\n")

		for coil_name, segments, r_wire in segments_list:
			f.write(f"* Segments: {coil_name}\n")
			for i, (n1_idx, n2_idx) in enumerate(segments):
				n1_id = f"{coil_name}_{n1_idx}"
				n2_id = f"{coil_name}_{n2_idx}"
				f.write(f"E{seg_id:d} N{node_map[n1_id]:d} N{node_map[n2_id]:d} W={r_wire*2:.6f} H={r_wire*2:.6f}\n")
				seg_id += 1

		f.write("\n")

		for coil_name, port_indices in external_ports:
			if len(port_indices) == 2:
				nid1 = f"{coil_name}_{port_indices[0]}"
				nid2 = f"{coil_name}_{port_indices[1]}"
				f.write(f".external N{node_map[nid1]:d} N{node_map[nid2]:d}\n")

		f.write("\n")
		if len(freq_list) == 1:
			f.write(f".freq fmin={freq_list[0]} fmax={freq_list[0]}\n")
		else:
			f.write(f".freq fmin={freq_list[0]} fmax={freq_list[-1]} ndec={10/len(freq_list)}\n")

		f.write(".end\n")


def gen_single_coil(output_inp, radius=5.0, pitch=1.0, num_turns=10, segs_per_turn=16,
					 wire_radius=0.5, title="Single Coil"):
	"""
	Generate a single spiral coil test case.
	"""
	nodes, segments = spiral_helix(radius, pitch, num_turns, segs_per_turn, wire_radius)

	gen_inp_file(
		output_inp,
		[("coil1", nodes)],
		[("coil1", segments, wire_radius)],
		[("coil1", [0, len(nodes)-1])],
		freq_list=[1.0, 1e5, 1e9],
		title=title
	)
	print(f"✓ Generated {output_inp}")


def gen_dual_coil(output_inp, radius1=5.0, radius2=3.0, gap=2.0, pitch=1.0, num_turns=10,
				   segs_per_turn=16, wire_radius=0.5, title="Dual Coaxial Coil"):
	"""
	Generate two coaxial coils for mutual inductance testing.
	"""
	nodes1, segs1 = spiral_helix(radius1, pitch, num_turns, segs_per_turn, wire_radius, start_z=0.0)
	nodes2, segs2 = spiral_helix(radius2, pitch, num_turns, segs_per_turn, wire_radius, start_z=gap)

	gen_inp_file(
		output_inp,
		[("coil1", nodes1), ("coil2", nodes2)],
		[("coil1", segs1, wire_radius), ("coil2", segs2, wire_radius)],
		[("coil1", [0, len(nodes1)-1]), ("coil2", [0, len(nodes2)-1])],
		freq_list=[1.0, 1e5, 1e9],
		title=title
	)
	print(f"✓ Generated {output_inp}")


def gen_straight_wire(output_inp, length=100.0, segs=10, wire_radius=0.5, title="Straight Wire"):
	"""
	Generate straight wire for basic inductance test.
	"""
	nodes, segments = straight_wire(length, segs, wire_radius, start=(0, 0, 0), direction=(0, 0, 1))

	gen_inp_file(
		output_inp,
		[("wire", nodes)],
		[("wire", segments, wire_radius)],
		[("wire", [0, len(nodes)-1])],
		freq_list=[1.0, 1e5, 1e9],
		title=title
	)
	print(f"✓ Generated {output_inp}")


def planar_spiral(turns, inner_radius, outer_radius, wire_radius, segs_per_turn=20):
	"""
	Generate a planar spiral (Archimedean spiral in xy-plane, z=0).
	Spiral expands from inner_radius to outer_radius as θ increases.
	"""
	nodes = []
	segments = []

	total_segs = turns * segs_per_turn

	for i in range(total_segs + 1):
		theta = 2 * np.pi * i / segs_per_turn
		r = inner_radius + (outer_radius - inner_radius) * i / total_segs
		x = r * np.cos(theta)
		y = r * np.sin(theta)
		z = 0.0
		nodes.append((x, y, z))

	for i in range(total_segs):
		segments.append((i, i + 1))

	return nodes, segments


def gen_planar_spiral(output_inp, turns=10, inner_radius=1, outer_radius=5,
					   wire_radius=0.3, title="Planar Spiral"):
	"""Generate planar spiral coil test case."""
	nodes, segments = planar_spiral(turns, inner_radius, outer_radius, wire_radius)

	gen_inp_file(
		output_inp,
		[("spiral", nodes)],
		[("spiral", segments, wire_radius)],
		[("spiral", [0, len(nodes)-1])],
		freq_list=[1.0, 1e5, 1e9],
		title=title
	)
	print(f"✓ Generated {output_inp}")


def gen_dual_planar_spiral(output_inp, turns=10, inner_radius=1, outer_radius=5,
						   wire_radius=0.3, gap=0.0, title="Dual Planar Spiral"):
	"""
	Generate two planar spirals at different z heights for mutual inductance.
	gap: vertical separation between the two spirals
	"""
	nodes1, segs1 = planar_spiral(turns, inner_radius, outer_radius, wire_radius, segs_per_turn=20)

	# Second spiral at z=gap
	nodes2_raw, segs2 = planar_spiral(turns, inner_radius, outer_radius, wire_radius, segs_per_turn=20)
	nodes2 = [(x, y, z + gap) for x, y, z in nodes2_raw]

	gen_inp_file(
		output_inp,
		[("spiral1", nodes1), ("spiral2", nodes2)],
		[("spiral1", segs1, wire_radius), ("spiral2", segs2, wire_radius)],
		[("spiral1", [0, len(nodes1)-1]), ("spiral2", [0, len(nodes2)-1])],
		freq_list=[1.0, 1e5, 1e9],
		title=title
	)
	print(f"✓ Generated {output_inp}")


if __name__ == "__main__":
	print("Generating test cases...")
	gen_straight_wire("examples/test_straight_wire.inp")
	gen_single_coil("examples/test_single_coil.inp")
	gen_dual_coil("examples/test_dual_coil.inp", gap=0.5)
	print("Done!")
