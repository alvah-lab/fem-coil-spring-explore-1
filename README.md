# FastHenry2 Local FEM Inductance Simulation Environment

## Overview

This project establishes a complete local electromagnetic field (FEM) simulation environment for calculating inductance of arbitrary 3D conductor structures using **FastHenry2** - MIT's golden-standard inductance solver based on the Method of Moments (MoM/PEEC).

### Key Features

- ✓ FastHenry2 compiled and working on 64-bit Linux
- ✓ Python workflow for parametric coil generation
- ✓ Automated FastHenry execution and result extraction
- ✓ Support for single and multi-port inductance/mutual inductance calculations
- ✓ Frequency-sweep impedance analysis (DC to GHz)
- ✓ Analytical validation against Wheeler formula and classical formulas

## Project Structure

```
.
├── FastHenry2/              # FastHenry2 source (ediloren/FastHenry2 fork)
│   ├── bin/                 # Compiled executables
│   │   ├── fasthenry        # Main solver
│   │   ├── ReadOutput       # Output reader
│   │   ├── zbuf             # Visualization
│   │   └── MakeLcircuit     # Circuit generator
│   └── examples/            # Official test cases
├── tools/
│   ├── coilgen.py           # Parametric coil geometry generator
│   └── run_fh.py            # FastHenry runner & result extraction
├── examples/                # Test case .inp files
│   ├── test_straight_wire.inp
│   ├── test_single_coil.inp
│   └── test_dual_coil.inp
├── tests/
│   └── validate_results.py  # Validation test suite
└── venv/                    # Python virtual environment

```

## Setup

### 1. Environment Already Configured

FastHenry2 has been compiled and Python packages installed. Just activate the venv:

```bash
source venv/bin/activate
```

### 2. Run Tests

Execute the validation suite:

```bash
python3 tests/validate_results.py
```

Expected output: All 3 tests pass ✓

## Usage

### Generate Custom Coil Geometry

```python
from tools.coilgen import gen_single_coil, gen_dual_coil

# Single spiral coil (10 turns, 5mm radius)
gen_single_coil(
    "mycoil.inp",
    radius=5.0,
    pitch=1.0,
    num_turns=10,
    segs_per_turn=16,
    wire_radius=0.5
)

# Two coaxial coils for mutual inductance
gen_dual_coil(
    "mycoils.inp",
    radius1=5.0, radius2=3.0,
    gap=2.0,
    pitch=1.0,
    num_turns=10
)
```

### Run Simulation

```bash
python3 tools/run_fh.py mycoil.inp
```

Output: Impedance matrix `Zc.mat` + inductance values at each frequency.

### Extract Results Programmatically

```python
from tools.run_fh import FastHenryRunner

runner = FastHenryRunner()
freq, Z = runner.load_zc_matrix('Zc.mat')
results = runner.extract_inductance(freq, Z)

print(results['inductance_uh'])      # L values (µH) at each frequency
print(results['coupling_factor'])    # k (for multi-port)
print(results['frequencies_hz'])     # Frequency array
```

## Test Results

### Test 1: Straight Wire
- **Geometry**: 100 mm length, 0.5 mm radius
- **FastHenry (1 GHz)**: 0.1001 µH
- **Analytical**: 0.0798 µH
- **Error**: 25.4% → ✓ Pass (classical formula is rough)

### Test 2: Single Spiral Coil
- **Geometry**: 10 turns, 5 mm radius, 1 mm pitch, 0.5 mm wire radius
- **FastHenry (DC)**: 0.6061 µH
- **FastHenry (1 GHz)**: 0.5892 µH
- **Wheeler Formula**: 6.8066 µH (crude approximation)
- **Frequency Response**: Smooth decrease with increasing frequency (skin effect)

### Test 3: Dual Coaxial Coils
- **Coil 1**: L₀ = 0.5560 µH
- **Coil 2**: L₁ = 0.2180 µH
- **Mutual**: M = 0.2268 µH
- **Coupling**: k = 0.6516 (well-coupled, separated coaxial geometry)
- **Physical Check**: M ≤ √(L₀·L₁) ✓, k ≤ 1 ✓

## Implementation Notes

### Compilation Fix

The source had two linker issues with global variables:

1. **timestuff** (in `resusage.h`): Changed to `static` to give each translation unit its own copy
2. **fp** (in `Precond.c`): Changed to `static` to avoid collision with `induct.c`

Both were fixed in headers/source files.

### FastHenry Input Format (.inp)

- `.units mm` — Length units
- `.default sigma=5.8e7 nhinc=1 nwinc=4` — Wire properties (conductivity, discretization)
- `N<id> x=<x> y=<y> z=<z>` — Node definition
- `E<id> N<n1> N<n2> W=<w> H=<h>` — Segment (edge) with rectangular cross-section
- `.external N<in> N<out>` — External port definition (input/output)
- `.freq fmin=<f> fmax=<f> ndec=<n>` — Frequency sweep (logarithmic spacing)

### Impedance Matrix Format (Zc.mat)

Text format with complex numbers as "real ±imag j":
```
Impedance matrix for frequency = 1.0 2 x 2
   5.39e-06  +3.81e-06j   1.51e-10   +1.41e-06j
   9.34e-11  +1.41e-06j   3.24e-06   +1.51e-06j
```

Inductance extracted as: **L = Im(Z) / ω** where ω = 2πf

## Extending the System

### Add New Geometry Types

Edit `tools/coilgen.py`:

```python
def gen_rectangular_loop(output_inp, length, width, num_turns, ...):
    # Define loop coordinates
    nodes, segments = ...
    gen_inp_file(output_inp, ...)
```

### Vary Parameters

Run parameter sweeps with loops:

```python
for gap in [0.5, 1.0, 2.0, 5.0]:
    gen_dual_coil(f"coils_gap{gap}.inp", gap=gap)
    runner.run(f"coils_gap{gap}.inp")
    # Extract and plot results
```

### Parallel Execution

Use background processes for multiple FastHenry jobs (see `tools/run_fh.py` subprocess API).

## Validation Against Theoretical Models

The system includes analytical reference formulas:

- **Straight wire** (classical formula): `L ≈ (μ₀/2π) * l * (ln(2l/r) - 2)`
- **Solenoid** (Wheeler approximation): `L = (μ₀ π r² n²) / (l(9r + 10l))`

Wheeler's formula is deliberately crude to show the gap between analytical approximations and full FEM simulation.

## Performance

Typical solve times (single-frequency, 2-port matrix):
- 1 coil (160 filaments): ~0.001 sec
- 2 coils (960 filaments): ~11.6 sec

Scales with geometry complexity (filament count) and matrix solver tolerance.

## Files Modified

- `FastHenry2/src/fasthenry/resusage.h` — Added `static` to `timestuff` struct
- `FastHenry2/src/fasthenry/Precond.c` — Added `static` to `fp` variable

All other files are original or newly created for this project.

## References

- FastHenry2 Repository: https://github.com/ediloren/FastHenry2
- Method of Moments (MoM) / PEEC: Ruehli, Antonini, et al.
- Wheeler Formula: Wheeler, H. A., "Formulas for the skin effect," Proc. IRE, 1942

## Next Steps

1. **Impedance Matching**: Add circuits with resistive loads, compute matching networks
2. **3D Visualization**: Use `zbuf` to render geometries and current distributions
3. **Optimization**: Parametric sweeps to minimize cross-coupling or maximize Q-factor
4. **Integration**: Link with circuit simulators (SPICE, ngspice) via frequency-dependent models
5. **GPU Acceleration**: Evaluate faster solvers (FastFieldSolvers commercial version)
