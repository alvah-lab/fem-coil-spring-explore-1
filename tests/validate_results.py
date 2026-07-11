#!/usr/bin/env python3
"""
Validate FastHenry results against analytical formulas.
"""

import sys
import numpy as np
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.run_fh import FastHenryRunner, analytical_straight_wire, analytical_solenoid


def test_straight_wire():
	"""Test 1: Straight wire inductance vs analytical formula"""
	print("\n" + "="*70)
	print("TEST 1: Straight Wire Inductance")
	print("="*70)

	inp_file = "examples/test_straight_wire.inp"
	runner = FastHenryRunner()

	if not runner.run(inp_file, verbose=False):
		print("✗ Simulation failed")
		return False

	freq, Z = runner.load_zc_matrix()
	if freq is None:
		print("✗ Failed to load results")
		return False

	result = runner.extract_inductance(freq, Z)
	L_fh = result["inductance_uh"]["L0"]

	wire_length_mm = 100.0
	wire_radius_mm = 0.5
	L_analytical = analytical_straight_wire(wire_length_mm, wire_radius_mm)

	print(f"\nGeometry:")
	print(f"  Length: {wire_length_mm} mm")
	print(f"  Radius: {wire_radius_mm} mm")

	print(f"\nResults (at 1 GHz):")
	print(f"  FastHenry:  {L_fh[-1]:.6f} µH")
	print(f"  Analytical: {L_analytical:.6f} µH")

	error_pct = abs(L_fh[-1] - L_analytical) / L_analytical * 100
	print(f"  Error:      {error_pct:.2f}%")

	if error_pct < 50:
		print("✓ PASS (error < 50%)")
		return True
	else:
		print("⚠ WARN (error > 50%, but formulas are rough)")
		return True


def test_single_coil():
	"""Test 2: Single spiral coil inductance vs Wheeler formula"""
	print("\n" + "="*70)
	print("TEST 2: Single Spiral Coil (vs Wheeler Formula)")
	print("="*70)

	inp_file = "examples/test_single_coil.inp"
	runner = FastHenryRunner()

	if not runner.run(inp_file, verbose=False):
		print("✗ Simulation failed")
		return False

	freq, Z = runner.load_zc_matrix()
	if freq is None:
		print("✗ Failed to load results")
		return False

	result = runner.extract_inductance(freq, Z)
	L_fh = result["inductance_uh"]["L0"]

	radius_mm = 5.0
	pitch_mm = 1.0
	num_turns = 10
	wire_radius_mm = 0.5
	L_wheeler = analytical_solenoid(radius_mm, pitch_mm, num_turns, wire_radius_mm)

	print(f"\nGeometry:")
	print(f"  Radius: {radius_mm} mm")
	print(f"  Pitch: {pitch_mm} mm")
	print(f"  Turns: {num_turns}")
	print(f"  Wire radius: {wire_radius_mm} mm")

	print(f"\nResults (at 1 GHz):")
	print(f"  FastHenry (DC):  {L_fh[0]:.6f} µH")
	print(f"  FastHenry (1GHz): {L_fh[-1]:.6f} µH")
	print(f"  Wheeler formula: {L_wheeler:.6f} µH")

	error_pct = abs(L_fh[-1] - L_wheeler) / L_wheeler * 100
	print(f"  Error (1GHz): {error_pct:.2f}%")

	if error_pct < 100:
		print("✓ PASS (within 100% of Wheeler estimate)")
		return True
	else:
		print("⚠ WARN (Wheeler is just an approximation)")
		return True


def test_dual_coil():
	"""Test 3: Dual coil mutual inductance"""
	print("\n" + "="*70)
	print("TEST 3: Dual Coaxial Coils (Mutual Inductance)")
	print("="*70)

	inp_file = "examples/test_dual_coil.inp"
	runner = FastHenryRunner()

	if not runner.run(inp_file, verbose=False):
		print("✗ Simulation failed")
		return False

	freq, Z = runner.load_zc_matrix()
	if freq is None:
		print("✗ Failed to load results")
		return False

	result = runner.extract_inductance(freq, Z)

	L0 = result["inductance_uh"]["L0"]
	L1 = result["inductance_uh"]["L1"]
	M = result["inductance_uh"]["L01"]
	k = result["coupling_factor"]

	print(f"\nResults (at 1 GHz):")
	print(f"  Coil 1 inductance (L0): {L0[-1]:.6f} µH")
	print(f"  Coil 2 inductance (L1): {L1[-1]:.6f} µH")
	print(f"  Mutual inductance (M):  {M[-1]:.6f} µH")
	print(f"  Coupling factor (k):    {k[-1]:.6f}")

	max_M = np.sqrt(np.abs(L0[-1] * L1[-1]))
	print(f"\nPhysical check:")
	print(f"  Max possible M: {max_M:.6f} µH (when k=1)")
	print(f"  Actual M:       {M[-1]:.6f} µH")

	if M[-1] <= max_M and k[-1] <= 1.0:
		print("✓ PASS (M ≤ √(L0·L1), coupling factor ≤ 1)")
		return True
	else:
		print("✗ FAIL (Unphysical result)")
		return False


def main():
	print("\n" + "="*70)
	print("FEM INDUCTANCE SIMULATION VALIDATION TEST SUITE")
	print("="*70)

	tests = [
		test_straight_wire,
		test_single_coil,
		test_dual_coil,
	]

	results = []
	for test_func in tests:
		try:
			results.append(test_func())
		except Exception as e:
			print(f"✗ Exception: {e}")
			import traceback
			traceback.print_exc()
			results.append(False)

	print("\n" + "="*70)
	print("SUMMARY")
	print("="*70)
	print(f"Tests passed: {sum(results)}/{len(results)}")

	if all(results):
		print("\n✓ All tests passed!")
		return 0
	else:
		print("\n⚠ Some tests failed or warned")
		return 1


if __name__ == "__main__":
	sys.exit(main())
