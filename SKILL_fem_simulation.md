---
name: fem-em-simulation
description: >
  First-principles FEM / electromagnetic-quasistatic simulation for coils, inductors,
  and small 3D conductor/elastomer structures. Covers tool selection (FastHenry PEEC,
  Palace FEM, CalculiX structural), build/install fixes (esp. Palace + OpenBLAS),
  gmsh meshing recipes, and — most importantly — the METHODOLOGY for parameter
  separability / identifiability analysis and the symmetry physics that governs it.
  Use when asked to compute inductance/capacitance/S-parameters of a coil, verify a
  lumped model against real EM, judge whether two mechanical deformations are
  separable from an electrical measurement, or run nonlinear structural FEA.
---

# FEM / EM Quasistatic Simulation — Field-tested Skill

Hard-won lessons from a spring-coil VNA-sensor study (double-parameter identification:
compression ε vs bending κ / shear γ) plus a CalculiX lip-spring structural study.
Everything below was actually run and debugged — the pitfalls are real.

---

## 0. THE #1 LESSON — validate analytical models with a real solver

Hand-derived coupling coefficients are **routinely wrong in sign and magnitude**.
On this project a plausible analytical model said "bending *decreases* inductance
(−0.6·η)"; FastHenry showed bending *increases* it (+0.21·η) — **wrong sign**. That
error had propagated into a whole separability conclusion. **Do not trust a
hand-waved ∂L/∂(param) or coupling constant. Compute it.** A cheap PEEC/FEM run
settles in minutes what argument cannot.

Corollary: when a result flips a prior conclusion, re-verify (mesh convergence, a
κ-sweep to check the power law, a sign check) before believing it — but believe the
solver over the intuition.

---

## 1. Tool selection (pick physics, not habit)

| Question | Right tool | Why |
|---|---|---|
| Inductance / mutual inductance of 3D wires at DC–~GHz | **FastHenry** (PEEC) | Purpose-built, seconds, no meshing of air. Full L matrix incl. skin/proximity. |
| Self / mutual **capacitance**, electrostatic fields | **Palace** electrostatic (or FastCap) | Real Laplace FEM → Maxwell C matrix. |
| Full-wave S-params, eigenmodes, high freq | **Palace** driven/eigenmode | Overkill for MHz magnetoquasistatics — don't reach for it first. |
| Nonlinear large-deformation **mechanics** (elastomer, contact) | **CalculiX** (`*NLGEOM`,`*HYPERELASTIC`) | Hyperelastic + self-contact force-displacement. |

**Key judgment**: a MHz inductance question does **not** need full-wave FEM. We wasted
effort configuring Palace before realizing FastHenry was already built and correct for
the job. Match the solver to the dominant physics.

---

## 2. Environment / build (the part that eats a day)

### Palace (AWS-Labs FEM) — the BLAS trap
- Palace is a CMake **superbuild** (pulls MFEM, HYPRE, PETSc/SLEPc, SuperLU_DIST,
  STRUMPACK, libCEED, METIS…). Full build ≈ **30–60 min on ~18 cores**, several GB.
- It defaults to finding **OpenBLAS** and **REQUIRES `cblas.h`** (STRUMPACK needs the
  CBLAS interface). The stock Ubuntu `libblas3` runtime has neither the `.so` dev
  symlink nor `cblas.h`.
- **Fix (clean, needs sudo):** `sudo apt-get install -y libopenblas-dev`
  → provides `libopenblas.so` + `/usr/include/x86_64-linux-gnu/cblas.h` + `lapacke.h`.
- **Fix (no sudo, partial):** symlink runtime libs so `find_package(BLAS)` passes —
  `ln -s /lib/x86_64-linux-gnu/libblas.so.3 <dir>/libblas.so` (+lapack), pass
  `-DBLAS_LIBRARIES/-DLAPACK_LIBRARIES`. But this still fails STRUMPACK's `cblas.h`
  requirement, so OpenBLAS-dev is the real answer.
- Configure then build:
  ```bash
  export OPENBLAS_DIR=/usr/lib/x86_64-linux-gnu
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PWD/install -DPALACE_WITH_OPENMP=ON
  cmake --build build --parallel 18     # run in background; it's long
  ```
  Look for `-- Using BLAS/LAPACK from OpenBLAS` in configure output.
- Binary: `install/bin/palace`; run `palace -np <ncores> config.json`.
- Smoke-test with the bundled `examples/spheres` (electrostatic → writes `terminal-C.csv`).

### FastHenry
- Often already compiled (`FastHenry2/bin/fasthenry`, FastFieldSolvers build). No deps.
- **Units gotcha**: with `.units mm`, giving `sigma` in **S/m** makes the reported
  **resistance 1000× too low** (it wants per-length-unit). Inductance is geometry-only,
  so L is unaffected — but fix sigma (→ S/mm) if you need R/Q.

### gmsh (meshing for Palace)
- `pip install gmsh` gives the full Python API (self-contained SDK, ~4.15). The CLI
  alone can't script helices comfortably.

### Python plotting venv
- A minimal `numpy + matplotlib` venv is enough; **scipy not required** for any of this
  (SVD/lstsq via numpy). Reuse one venv across FastHenry driving + plotting.

---

## 3. FastHenry recipe (inductance & mutual)

Netlist = nodes + segments + ports:
```
* comment
.units mm
.default sigma=<S/m or S/mm> nhinc=1 nwinc=1     # nhinc/nwinc=1 → low-freq DC-like L
N0 x=.. y=.. z=..                                 # nodes
E0 N0 N1 w=<width> h=<height>                     # rectangular wire segment
.external N0 N<last>                              # port 1 (repeat for 2-port → mutual)
.freq fmin=1e6 fmax=1e6 ndec=1                    # 1 MHz ≈ DC inductance (skin depth ≫ wire)
.end
```
- Round wire ≈ square of equal area: `side = sqrt(pi)*d/2`.
- Discretize a turn into ~24–36 segments; a 24-turn coil (≈600 segs) solves instantly.
- Output `Zc.mat`: parse the impedance matrix; `L = Im(Z)/(2πf)`.
  For a 2-port: `L = [[L1, M],[M, L2]]`; series-aiding total `L_tot = L1+L2+2|M|`.
- **Double-layer winding must be same-sense** or the two layers cancel! Wind up at r1
  (phase `q·s`), then down at r2 with **continuous phase** `q·H + q·s` (NOT the reversed
  up-helix — reversing flips current direction → fields subtract → L collapses to a
  fraction of the true ~4× single-layer value). This bug is easy to miss; sanity-check
  L_double ≈ 4× L_single.

Reusable helix generator (bottom-clamped, supports bend κ and shear γ):
```python
def helix(r, H, N, kap_per_m=0, gamma=0, zbase=0.0, spt=30):
    s = np.linspace(0, H, int(N*spt)); q = 2*np.pi*N/H; k = kap_per_m/1000  # →1/mm
    if abs(k) < 1e-12:  cx = 0*s; cz = s; nx = 1+0*s; nz = 0*s
    else: ks=k*s; cx=(1-np.cos(ks))/k; cz=np.sin(ks)/k; nx=np.cos(ks); nz=-np.sin(ks)
    u = s/H; xc = gamma*(3*u**2 - 2*u**3)     # shear: parallel top translation, S-curve
    return np.column_stack([xc + cx + r*np.cos(q*s)*nx, r*np.sin(q*s),
                            zbase + cz + r*np.cos(q*s)*nz])
```
- Compression = fix N, shrink H (pitch H/N drops, turns denser → L up). **Do NOT** let
  N vary with H (a common bug: `N=H/pitch` makes compression *remove* turns → L drops,
  wrong sign).

---

## 4. Palace electrostatic recipe (capacitance)

Config (`Type: Electrostatic`): mesh in mm → set `L0: 1.0e-3`; conductors are
`Terminal` electrodes; far box is `Ground`; `SurfaceFlux` gives charge → Maxwell C.
```json
{"Problem":{"Type":"Electrostatic","Output":"postpro"},
 "Model":{"Mesh":"coil.msh","L0":1.0e-3},
 "Domains":{"Materials":[{"Attributes":[1],"Permittivity":1.0}]},
 "Boundaries":{"Ground":{"Attributes":[2]},
   "Terminal":[{"Index":1,"Attributes":[3]}]},
 "Solver":{"Order":2,"Electrostatic":{"Save":0},
   "Linear":{"Type":"BoomerAMG","KSPType":"CG","Tol":1e-8,"MaxIts":200}}}
```
Output `postpro/terminal-C.csv` = K×K Maxwell matrix (diag +, off-diag −).

### Self-resonant (turn-to-turn) capacitance — the RIGOROUS version
Coil-to-ground C (1 terminal) is a convenient **proxy but optimistic**. The physically
relevant self-C is turn-to-turn. Extract it by **multi-terminal + energy method**:
1. Split the coil into K contiguous electrode segments (small gaps), tag each a Terminal.
2. Palace → K×K matrix `C` (comes out clean tridiagonal: adjacent-segment coupling = the
   turn-to-turn C).
3. Effective self-C under the fundamental (≈linear voltage ramp `v=[0..1]`):
   **`C_self = v @ C @ v`**.
On this project coil-to-ground gave condition number ~107; the rigorous turn-to-turn gave
~896 — **a 8× more pessimistic (and correct) answer.** Always state which you used.

---

## 5. gmsh meshing recipe + the pitfalls

Build the air region = box − wire-tube(s), tag surfaces, export **.msh v2.2** (Palace).
```python
import gmsh, numpy as np
gmsh.initialize(); occ = gmsh.model.occ
def make_pipe(P, rw=0.1):                              # P = Nx3 centerline points (mm)
    pts=[occ.addPoint(*p) for p in P]; wire=occ.addWire([occ.addSpline(pts)])
    t=P[1]-P[0]; t/=np.linalg.norm(t)
    dsk=occ.addDisk(*P[0], rw, rw, zAxis=list(t))      # disk ⟂ start tangent
    return occ.addPipe([(2,dsk)], wire)[0][1]
# ... make pipes, occ.synchronize(), addBox, occ.cut([box],[pipes]), synchronize()
# classify boundary faces: box faces have bbox extent > box size → Ground; else Terminal
gmsh.option.setNumber("Mesh.MeshSizeMin",0.06)         # ~half wire radius
gmsh.option.setNumber("Mesh.MeshSizeMax",2.5)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",8)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion",2.2); gmsh.write("coil.msh")
```
**Pitfalls that cost hours:**
- A single continuous pipe **self-intersects / makes sliver tets** when the path comes
  within ~a wire-diameter of itself (e.g. a double-layer coil with 0.2 mm inter-layer
  gap → PLC "segment and facet intersect" error). **Fix:** build up-layer, down-layer,
  and any inner coil as **separate pipes**, boolean-cut all from the box, tag them all as
  the same Terminal (the thin electrical connectors carry negligible charge). Widening the
  gap to ≥0.3 mm also helps meshing.
- Classify Ground vs Terminal surfaces by `getBoundingBox` extent (box faces span the
  whole domain), or by center-of-mass distance to each segment for multi-terminal.
- A 24-turn coil + air box ≈ 250–550k tets; converges in a BoomerAMG-CG in ~10–15 s
  per terminal solve. Run 3-state batches in the background.

---

## 6. METHODOLOGY — separability / identifiability (the real deliverable)

Goal: can two parameters (say compression ε, bending κ) be recovered from an electrical
measurement? Reduce to a **2×2 Jacobian condition number**.

1. Pick observables (e.g. `L`, `C_self`, or a mutual `M`, or sum/difference of two coils).
2. Compute each observable at baseline and at +Δε, +Δκ (real solver, range-normalized
   fractional changes).
3. Form `J = [[dO1/dε, dO1/dκ],[dO2/dε, dO2/dκ]]` (columns = params).
4. `cond = σ_max/σ_min` (numpy SVD). Also report the **angle between the two parameter
   direction-vectors** in observation space — that's the intuitive picture.

Reading the number: cond ~1–20 excellent, ~30–100 usable (needs good SNR on the weak
axis), >100 ill-conditioned (rank-2 but the weak parameter needs very high SNR),
→∞ degenerate. Convert to an actual RMSE-vs-SNR with a grid-search MLE + Monte-Carlo if
you want measurable numbers (a full-complex multi-frequency IQ fit is the realistic VNA
readout; use a WIDE frequency window — a narrow window inflates the condition number and
is falsely pessimistic).

### The governing physics — SYMMETRY THEOREM (memorize this)
> A **symmetric / isotropic scalar observable** (a single coil's self-L or self-C)
> responds to a **directional (vector) deformation** — bending, shear — only at
> **second order** (`∝ (a·κ)²`, an even function, because +κ and −κ are mirror images).
> Compression (a scalar/isotropic deformation) is first order. Therefore a single
> symmetric coil sees bending/shear weakly, and — worse — **all its deformation modes
> move (L, C) along nearly the same line** (they all just change the coil's overall
> compactness), so the (L,C) pair separates modes *poorly* (cond ~100–900).

Consequences proven on this project:
- Compression↔bending, single coil: cond ~896 (turn-to-turn C). Poor.
- Compression↔shear, single coil: L,C nearly **anti-parallel** (182°) → cond ~117. Poor.
  (Opposite *sign* ≠ different *direction*; anti-parallel is still collinear.)

### What actually WORKS — break the symmetry / engineer orthogonality
- **Differential / gradiometric pickup**: two side coils at ±x, observable
  `D = M⁺ − M⁻`. Bending leans the coil toward one → **first-order** signal; compression
  is common-mode and cancels. Took cond 896 → **36**.
- **Dual coil + rigid top plate** (best): two coils side-by-side, tops joined by a rigid
  plate. Overall compression = common-mode (both compress); bending tilts the plate →
  one compresses, one stretches = **differential compression, lever-amplified** by
  spacing/height. Sum `S=L1+L2+2M` reads compression, difference `D=L1−L2` reads bending;
  they are **naturally orthogonal** → cond **1.4**. Compression is easy to measure, so
  converting bending *into* differential compression is the winning move.
- **Readout caveat**: a pure **series** (2-terminal) connection returns only the SUM →
  it **common-mode-rejects the differential (bending) info**. To get both you need a
  **center-tap / 2-port / bridge** (sum AND difference). Many "clever structures" fail
  only because the series readout throws away the very signal they create.
- A **symmetric coaxial** add-on (e.g. a bottom pancake spiral) does NOT help a directional
  deformation (its mutual is even in κ → ~0 first-order) and merely **dilutes** sensitivity
  by adding fixed inductance. Asymmetric placement (inner coil only at the bottom half)
  recovers *some* first-order M signal but the series sum still buries it.

---

## 7. CalculiX nonlinear structural FEA (complementary path)

For elastomer/spring force-displacement (from the lip-spring evidence package):
- Solver **CalculiX (ccx) 2.19**: `*NLGEOM` large-deformation + `*HYPERELASTIC, NEO HOOKE`
  + frictionless **self-contact**.
- Material: neo-Hooke `C10 = μ/2`, `D1 = 2/κ_bulk`; from Shore hardness → μ (e.g. Shore
  25A ≈ μ 0.177 MPa → C10 0.0885, D1 0.929 at ν≈0.46).
- Mesh **C3D10** (quadratic tets). Load = rigid platen, **displacement-controlled** to
  target strain, read base reaction → `F(δ)`.
- Pipeline: STL → C3D10 → ccx self-contact → F(δ) curve (`fea_ccx.py`).
- Pitfalls: ccx **diverges at first self-contact** (lip closure) — expect it, bracket the
  usable range. Always run a **solid-cylinder case = absolute material ceiling**: any
  holed/lipped geometry is softer, so if the solid can't hit the force target, no geometry
  can (material/hardness must change). This "material ceiling" framing turns a failed
  force-match into an actionable spec trade-off (harder durometer / relax force band /
  allow wall constraint).

---

## 8. Matplotlib gotchas (recurring, CJK + preview)
- Chinese font: `rcParams['font.sans-serif']=['AR PL UMing CN','AR PL UKai CN']`.
- Superscripts/minus via **mathtext**: write `m$^{-1}$`, never literal `⁻¹` (renders as a
  box — the CJK font lacks U+207B). Also missing: `−`(U+2212), `✓`(U+2713), `✗`, `≠` in
  text boxes → use ASCII `-`, `(OK)`, `!=`. For log-axis ticks that auto-use `10^{-1}`,
  set explicit `FixedFormatter(['0.1','1','10'...])` to dodge the minus glyph.
- `rcParams['axes.unicode_minus']=False`; `mathtext.fontset='cm'`.
- Set `facecolor='white'` on figure AND `savefig` (default can look transparent/odd).
- Keep saved images **≤ ~1900 px wide** if you (or a tool) will preview them (2000 px hard
  cap in some viewers); down-res with PIL if needed.
- `ax.axis('off')` text-box "dashboard" panels overflow easily — size the figure for the
  text, not vice-versa.

---

## 9. One-paragraph reusable checklist
Right solver for the physics (FastHenry L / Palace C / CalculiX F). Build Palace with
`libopenblas-dev` (cblas.h!). Fix FastHenry sigma units if you need R. Same-sense
double-layer winding. Fix N (not pitch-count) under compression. Separate pipes for
close-spaced conductors in gmsh; tag by bbox; export .msh 2.2. Verify every analytical
coupling with a solver — expect sign surprises. Reduce separability to a 2×2 Jacobian
condition number; remember the symmetry theorem (directional deformations are 2nd-order
for symmetric observables); break symmetry with differential/gradiometric or
rigid-plate-coupled dual structures; don't let a series readout common-mode-reject your
signal. Prefer turn-to-turn self-C over coil-to-ground (it's the honest, pessimistic one).
