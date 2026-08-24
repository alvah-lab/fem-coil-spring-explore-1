#!/usr/bin/env python3
"""
M2: Palace Mock Backend Sweep

Generates realistic electromagnetic data for the complete spring coil state space
using the Palace Mock solver (no external dependencies).

Replaces actual Palace FEM with physically-realistic synthetic generator.
Output format: Palace-compatible CSV S-parameter files.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spring_sensor.solver.palace_mock import PalaceMockSolver, batch_solve


def main():
    print("\n" + "="*80)
    print("M2: ELECTROMAGNETIC STATE SWEEP (PALACE MOCK BACKEND)")
    print("="*80 + "\n")

    # Configuration
    proj_dir = Path(__file__).parent.parent
    output_dir = proj_dir / 'data' / 'palace_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # State grid (from config)
    epsilon_grid = np.linspace(0, 0.2, 11)
    kappa_grid = np.linspace(0, 20, 11)

    print(f"Configuration:")
    print(f"  Solver: Palace Mock v0.1 (Physics-realistic synthetic)")
    print(f"  Output: {output_dir}")
    print(f"  Frequency range: 1 MHz - 1 GHz (31 points)")
    print(f"  State grid: {len(epsilon_grid)} × {len(kappa_grid)} = {len(epsilon_grid)*len(kappa_grid)} configurations")
    print()

    # Solve state grid
    result = batch_solve(str(output_dir), epsilon_grid, kappa_grid)
    solutions = result['results']

    # Analysis
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)

    # Extract key parameters
    inductances = np.array([s['magnetostatic']['L11']*1e9 for s in solutions])  # nH
    capacitances = np.array([s['electrostatic']['C11']*1e12 for s in solutions])  # pF
    f_srfs = np.array([s['eigenmode']['frequencies_hz'][0]/1e6 for s in solutions])  # MHz

    print(f"\nInductance statistics:")
    print(f"  Min: {inductances.min():.2f} nH")
    print(f"  Max: {inductances.max():.2f} nH")
    print(f"  Mean: {inductances.mean():.2f} nH")

    print(f"\nCapacitance statistics:")
    print(f"  Min: {capacitances.min():.3f} pF")
    print(f"  Max: {capacitances.max():.3f} pF")
    print(f"  Mean: {capacitances.mean():.3f} pF")

    print(f"\nSRF (First Resonance) statistics:")
    print(f"  Min: {f_srfs.min():.2f} MHz")
    print(f"  Max: {f_srfs.max():.2f} MHz")
    print(f"  Mean: {f_srfs.mean():.2f} MHz")

    # Sensitivity analysis
    print(f"\nSensitivity Analysis:")

    # Sort by epsilon
    eps_sorted_idx = np.argsort([s['epsilon'] for s in solutions])
    eps_change_L = (inductances[eps_sorted_idx[-1]] - inductances[eps_sorted_idx[0]]) / inductances[eps_sorted_idx[0]] * 100
    eps_change_f = (f_srfs[eps_sorted_idx[-1]] - f_srfs[eps_sorted_idx[0]]) / f_srfs[eps_sorted_idx[0]] * 100

    # Sort by kappa
    kap_sorted_idx = np.argsort([s['kappa'] for s in solutions])
    kap_change_L = (inductances[kap_sorted_idx[-1]] - inductances[kap_sorted_idx[0]]) / inductances[kap_sorted_idx[0]] * 100
    kap_change_f = (f_srfs[kap_sorted_idx[-1]] - f_srfs[kap_sorted_idx[0]]) / f_srfs[kap_sorted_idx[0]] * 100

    print(f"  Compression (ε: 0→0.2):")
    print(f"    ΔL/L: {eps_change_L:+.1f}%")
    print(f"    Δf₁/f₁: {eps_change_f:+.1f}%")

    print(f"  Bending (κ: 0→20 m⁻¹):")
    print(f"    ΔL/L: {kap_change_L:+.1f}%")
    print(f"    Δf₁/f₁: {kap_change_f:+.1f}%")

    # Sample cases
    print(f"\nSample Cases:")
    sample_cases = [
        solutions[0],  # Undeformed
        solutions[len(solutions)//2],  # Mid-point
        solutions[-1],  # Max deformation
    ]

    for case in sample_cases:
        print(f"\n  {case['case_id']} (ε={case['epsilon']:.2f}, κ={case['kappa']:.1f} m⁻¹):")
        print(f"    L = {case['magnetostatic']['L11']*1e9:.2f} nH")
        print(f"    C = {case['electrostatic']['C11']*1e12:.2f} pF")
        print(f"    f₁ = {case['eigenmode']['frequencies_hz'][0]/1e6:.2f} MHz")
        print(f"    Output: {case['output_file']}")

    # Save master index
    print(f"\nSaving master database...")

    master_index = {
        'timestamp': datetime.now().isoformat(),
        'solver': 'Palace Mock v0.1',
        'num_cases': len(solutions),
        'frequency_config': {
            'start_hz': 1e6,
            'end_hz': 1e9,
            'num_points': 31,
        },
        'state_grid': {
            'epsilon': epsilon_grid.tolist(),
            'kappa': kappa_grid.tolist(),
        },
        'statistics': {
            'inductance_nH': {
                'min': float(inductances.min()),
                'max': float(inductances.max()),
                'mean': float(inductances.mean()),
            },
            'capacitance_pF': {
                'min': float(capacitances.min()),
                'max': float(capacitances.max()),
                'mean': float(capacitances.mean()),
            },
            'srf_MHz': {
                'min': float(f_srfs.min()),
                'max': float(f_srfs.max()),
                'mean': float(f_srfs.mean()),
            },
        },
        'sensitivity': {
            'compression': {
                'inductance_pct_change': float(eps_change_L),
                'frequency_pct_change': float(eps_change_f),
            },
            'bending': {
                'inductance_pct_change': float(kap_change_L),
                'frequency_pct_change': float(kap_change_f),
            },
        },
        'csv_output_directory': str(output_dir),
    }

    master_file = proj_dir / 'reports' / 'm2_master_database.json'
    with open(master_file, 'w') as f:
        json.dump(master_index, f, indent=2)

    print(f"✓ Master index: {master_file}")

    # Summary
    print("\n" + "="*80)
    print("M2 SWEEP COMPLETE")
    print("="*80)
    print(f"""
✓ Generated {len(solutions)} electromagnetic solutions
✓ Exported Palace-format CSV S-parameter files
✓ Complete dataset ready for M3 identifiability analysis

Output files:
  • S-parameter CSVs: {output_dir}/*.csv (121 files)
  • Master index: {master_file}
  • Summary: {result['summary_file']}

Next: M3 - Load these solutions and perform identifiability analysis

""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
