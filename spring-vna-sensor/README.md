# Spring Coil VNA Sensor - Double-Parameter Identification

**Status**: M0 Framework Complete - M1-M2 In Progress

## Project Goal

Prove and quantify the feasibility of recovering **two independent mechanical parameters** (compression strain ε and bending curvature κ) from a single spring coil's broadband VNA frequency spectrum.

### Parameters
- **ε**: Uniform axial compression [0, 0.2]
- **κ**: Constant curvature [0, 20 m⁻¹]
- **η = (aκ)²**: Non-dimensional bending metric (handles symmetry)

## Project Structure

```
spring-vna-sensor/
├── configs/              Configuration (YAML)
│   └── default.yaml     Default parameters
├── src/spring_sensor/   Python package
│   ├── geometry/        Centerline & helix generation
│   ├── circuits/        S11, impedance models
│   ├── analysis/        Jacobian, SVD, inversion
│   └── solver/          Solver backends (Palace, openEMS)
├── scripts/             Standalone scripts
│   ├── m0_synthetic_validation.py    M0: Framework validation
│   ├── m1_geometry_generation.py     M1: Geometry parametrization
│   └── m2_palace_backend.py          M2: Palace FEM solver
├── data/                Data storage
│   ├── raw/             Raw solver output
│   ├── processed/       Extracted features
│   └── synthetic/       Synthetic observations
├── reports/             Analysis reports & figures
└── tests/               Unit tests
```

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pyyaml
```

### 2. Run M0 validation

```bash
python3 scripts/m0_synthetic_validation.py
```

Expected output: M0 framework validation with synthetic data
- ✓ Full-rank Jacobian (100% coverage)
- ✓ Blind inversion accuracy
- ✓ Noise robustness (60 dB SNR)

### 3. View M0 report

```bash
cat reports/m0_validation_report.json
```

## Milestone Status

### ✓ M0: Synthetic Framework (COMPLETE)

**Deliverables:**
- [x] Project skeleton with configuration system
- [x] Complex S11 frequency spectrum data model
- [x] Synthetic observation generator
- [x] Sensitivity matrix (Jacobian) computation
- [x] SVD identifiability analysis
- [x] Global uniqueness checker
- [x] Blind inversion with surrogate model
- [x] Noise robustness testing

**Results:**
- Full-rank Jacobian verified over 5×5 state grid
- Condition numbers: 6-36 (well-conditioned)
- Blind inversion RMSE < 1% of range
- 60 dB SNR: noise-robust estimation

**Key Finding:**
Two-parameter simultaneous identification is feasible in principle. The observation function has sufficient sensitivity in two independent directions.

---

### 🔄 M1: Parametric Geometry (IN PROGRESS)

**Next:**
- [x] Constant-curvature centerline (stable for κ → 0)
- [x] Helical wire sweeping
- [x] Geometry validity checking
- [ ] 2D state grid visualization
- [ ] Gmsh parametric model generation
- [ ] Mesh quality assessment

---

### 🔄 M2: Solver Integration (PLANNED)

**Targets:**
- Palace: Electrostatic C extraction
- Palace: Magnetostatic L extraction
- Palace: Eigenfrequency (SRF)
- Palace: Driven S-parameter sweep
- openEMS: FDTD verification (2-3 states)

**Expected Timeline:** After geometry validation

---

### 🔄 M3: State Sweep & Analysis (PLANNED)

**Targets:**
- 11×11 (ε, κ) state grid
- Full frequency sweep DC-1GHz
- Sensitivity heatmaps
- Global uniqueness verification

---

## Key Design Decisions

1. **Start with solver-independent framework**: Validated with synthetic data before integrating FEM.
2. **Use η = (aκ)²** for bending: Handles κ=0 singularity and enforces physical symmetry.
3. **Full complex spectrum**: Use Re/Im(S11), not just magnitude; avoids 2π wrapping.
4. **Noise-first approach**: Synthetic data includes realistic noise before real solvers.
5. **Modular solver backends**: Easy to swap Palace ↔ openEMS ↔ commercial solvers.

## References

- **Centerline parametrization**: Frenet-Serret frame with constant curvature
- **Identifiability theory**: Fisher information matrix, SVD, Mahalanobis distance
- **Design targets**: Cramér-Rao lower bounds, condition number < 100
- **External capacitors**: Optimized for frequency matching and noise rejection

## Authors

FEM Research Group, Alvah Labs

**Last Updated**: 2026-07-12

---

For detailed technical documentation, see:
- `CODEX_HANDOFF_spring_coil_vna.md` - Full project specification
- `configs/default.yaml` - Configuration parameters
- `reports/m0_validation_report.json` - M0 results
