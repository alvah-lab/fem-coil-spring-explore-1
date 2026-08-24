#!/usr/bin/env python3
"""
Palace Mock Backend: Simulates Palace FEM solver output for M2 validation.

Generates realistic electromagnetic data for spring coil configurations
without requiring actual Palace installation.

Provides:
  - Electrostatic (capacitance)
  - Magnetostatic (inductance)
  - Eigenmode (resonant frequencies)
  - Driven (S-parameters)
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Tuple


class PalaceMockSolver:
    """Mock Palace solver that generates physically realistic EM data."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = np.logspace(6, 10, 31)  # 1 MHz to 10 GHz (includes SRF)
        self.omega = 2 * np.pi * self.freq

    def solve_magnetostatic(
        self,
        case_id: str,
        epsilon: float,
        kappa: float,
    ) -> Dict[str, float]:
        """
        Simulate magnetostatic solution (inductance extraction).

        Args:
            case_id: Configuration identifier
            epsilon: Compression strain
            kappa: Curvature (m⁻¹)

        Returns:
            dict with inductance values
        """
        # Base inductance (spring geometry)
        L_base = 1.0e-9  # 1 nH (enables SRF ~1.3 GHz with high C)

        # Compression reduces inductance (turn pitch increases)
        # Enhanced sensitivity: 40% change over range
        L_comp_factor = 1 - 0.4 * epsilon

        # Curvature reduces inductance (field distortion)
        # Enhanced sensitivity: 60% change over range
        eta = (5e-3 * kappa) ** 2
        L_bend_factor = 1 - 0.6 * eta

        L = L_base * L_comp_factor * L_bend_factor

        # Self and mutual inductance (single port)
        inductance = {
            "L11": float(L),
            "L_real": float(L),
            "quality_factor": 150.0,
        }

        return inductance

    def solve_electrostatic(
        self,
        case_id: str,
        epsilon: float,
        kappa: float,
    ) -> Dict[str, float]:
        """
        Simulate electrostatic solution (capacitance extraction).

        Args:
            case_id: Configuration identifier
            epsilon: Compression strain
            kappa: Curvature

        Returns:
            dict with capacitance values
        """
        # Base distributed capacitance
        C_base = 5.0e-12  # 5.0 pF (enables SRF ~1.4 GHz with 1nH inductance)

        # Compression increases capacitance (turns closer together)
        # Enhanced sensitivity: 30% change over range
        C_comp_factor = 1 + 0.3 * epsilon

        # Bending increases capacitance (distorted field)
        # Enhanced sensitivity: 80% change over range
        eta = (5e-3 * kappa) ** 2
        C_bend_factor = 1 + 0.8 * eta

        C = C_base * C_comp_factor * C_bend_factor

        capacitance = {
            "C11": float(C),
            "C_real": float(C),
        }

        return capacitance

    def solve_eigenmode(
        self,
        case_id: str,
        epsilon: float,
        kappa: float,
        num_modes: int = 3,
    ) -> Dict:
        """
        Simulate eigenmode solution (resonant frequencies).

        Args:
            case_id: Configuration identifier
            epsilon: Compression strain
            kappa: Curvature
            num_modes: Number of modes to compute

        Returns:
            dict with eigenfrequencies
        """
        # Get L and C
        L_data = self.solve_magnetostatic(case_id, epsilon, kappa)
        C_data = self.solve_electrostatic(case_id, epsilon, kappa)

        L = L_data["L11"]
        C = C_data["C11"]

        # First resonant frequency (LC resonance)
        f_1 = 1.0 / (2 * np.pi * np.sqrt(L * C + 1e-32))

        # Compression shifts frequency up
        f_1 *= 1 + 0.05 * epsilon

        # Bending shifts frequency up (increased capacitance, decreased inductance)
        eta = (5e-3 * kappa) ** 2
        f_1 *= 1 + 0.1 * eta

        # Higher modes (empirical relationships)
        f_2 = f_1 * 3.2
        f_3 = f_1 * 5.8

        modes = {
            "frequencies_hz": [float(f_1), float(f_2), float(f_3)][:num_modes],
            "quality_factors": [150.0, 80.0, 40.0][:num_modes],
        }

        return modes

    def solve_driven(
        self,
        case_id: str,
        epsilon: float,
        kappa: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate driven frequency-domain solution (S-parameters).

        Args:
            case_id: Configuration identifier
            epsilon: Compression strain
            kappa: Curvature

        Returns:
            (frequency_array, S11_complex)
        """
        # Get circuit parameters
        L_data = self.solve_magnetostatic(case_id, epsilon, kappa)
        C_data = self.solve_electrostatic(case_id, epsilon, kappa)
        modes = self.solve_eigenmode(case_id, epsilon, kappa, num_modes=3)

        L = L_data["L11"]
        C = C_data["C11"]
        Q = L_data["quality_factor"]
        f_1 = modes["frequencies_hz"][0]

        # Lumped impedance (first mode dominated)
        Z0 = 50.0  # Reference impedance
        R_base = Z0 * 0.1  # Base series resistance

        # Frequency-dependent impedance
        omega_1 = 2 * np.pi * f_1
        Z = R_base + 1j * (self.omega * L - 1 / (self.omega * C + 1e-16))

        # Add resonance peak
        Q_effective = Q * (1 - 0.3 * epsilon)  # Q decreases with compression
        Z = Z / (1 + 1j * self.omega * (1 / (Q_effective * omega_1)))

        # S11 from impedance
        S11 = (Z - Z0) / (Z + Z0 + 1e-16)

        return self.freq, S11

    def export_s_parameter_csv(
        self,
        case_id: str,
        freq: np.ndarray,
        s11: np.ndarray,
    ) -> Path:
        """
        Export S-parameter in Palace CSV format.

        Args:
            case_id: Configuration ID
            freq: Frequency array
            s11: S11 parameter (complex)

        Returns:
            Path to exported CSV
        """
        output_file = self.output_dir / f"{case_id}_s11.csv"

        # Palace format: freq, S11_dB, S11_phase
        s11_db = 20 * np.log10(np.abs(s11) + 1e-16)
        s11_phase = np.angle(s11, deg=True)

        with open(output_file, "w") as f:
            f.write("Frequency (Hz),S11 (dB),S11 (deg)\n")
            for i in range(len(freq)):
                f.write(f"{freq[i]:.6e},{s11_db[i]:.6f},{s11_phase[i]:.6f}\n")

        return output_file

    def solve_case(
        self,
        case_id: str,
        epsilon: float,
        kappa: float,
        output_format: str = "csv",
    ) -> Dict:
        """
        Solve complete electromagnetic problem for one state.

        Args:
            case_id: Configuration ID
            epsilon: Compression strain
            kappa: Curvature
            output_format: "csv" or "native"

        Returns:
            dict with all results
        """
        # Solve all sub-problems
        L_data = self.solve_magnetostatic(case_id, epsilon, kappa)
        C_data = self.solve_electrostatic(case_id, epsilon, kappa)
        modes = self.solve_eigenmode(case_id, epsilon, kappa)
        freq, S11 = self.solve_driven(case_id, epsilon, kappa)

        # Export
        s11_file = self.export_s_parameter_csv(case_id, freq, S11) if output_format == "csv" else None

        result = {
            "case_id": case_id,
            "epsilon": float(epsilon),
            "kappa": float(kappa),
            "eta": float((5e-3 * kappa) ** 2),
            "magnetostatic": L_data,
            "electrostatic": C_data,
            "eigenmode": modes,
            "driven": {
                "frequencies_hz": freq.tolist(),
                "s11_real": np.real(S11).tolist(),
                "s11_imag": np.imag(S11).tolist(),
            },
            "output_file": str(s11_file) if s11_file else None,
        }

        return result


def batch_solve(
    output_dir: str,
    epsilon_grid: np.ndarray,
    kappa_grid: np.ndarray,
) -> Dict:
    """
    Solve complete state grid.

    Args:
        output_dir: Output directory
        epsilon_grid: Compression values
        kappa_grid: Curvature values

    Returns:
        dict with all solutions
    """
    solver = PalaceMockSolver(output_dir)
    results = []

    print(f"\nSolving {len(epsilon_grid) * len(kappa_grid)} configurations...")
    for i, eps in enumerate(epsilon_grid):
        for j, kap in enumerate(kappa_grid):
            case_id = f"case_eps{i:02d}_kap{j:02d}"
            result = solver.solve_case(case_id, eps, kap)
            results.append(result)

            if (i * len(kappa_grid) + j + 1) % 5 == 0:
                print(f"  {i*len(kappa_grid)+j+1}/{len(epsilon_grid)*len(kappa_grid)} complete")

    # Save summary
    summary_file = Path(output_dir) / "palace_mock_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "solver": "Palace Mock v0.1",
                "num_cases": len(results),
                "frequency_range": {"start_hz": 1e6, "end_hz": 1e9, "points": 31},
                "sample_case": results[0] if results else None,
            },
            f,
            indent=2,
        )

    print(f"✓ Saved {len(results)} solutions to {output_dir}")
    print(f"✓ Summary: {summary_file}")

    return {"solver": solver, "results": results, "summary_file": summary_file}


if __name__ == "__main__":
    # Test run
    solver = PalaceMockSolver("./test_output")

    # Test one configuration
    print("\nTesting Palace Mock Solver...")
    result = solver.solve_case("test_case", epsilon=0.05, kappa=10.0)

    print(f"\nCase: {result['case_id']}")
    print(f"  L = {result['magnetostatic']['L11']*1e9:.2f} nH")
    print(f"  C = {result['electrostatic']['C11']*1e12:.2f} pF")
    print(f"  f_1 = {result['eigenmode']['frequencies_hz'][0]/1e6:.2f} MHz")
    print(f"  Output: {result['output_file']}")
