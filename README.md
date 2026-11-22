# STEM Racing Pro-Class Wheel: Multi-Process Design and FEA

Research code for designing the lightest robust wheel for a STEM Racing Professional Class car
(2026–27 UAE Technical Regulations). Process options are milled/turned Al and Ti, SLA/DLP resins,
SLS PA12, FDM ABS and carbon fibre. The analysis is a scripted, reproducible open-source FEA
pipeline. Final designs are meant to be cross-checked independently in Ansys Student.

> **Status (2026-11-22): stage 5 done for SLS PA12.** A general 3D pipeline (spokes, variable rim thickness,
> fillets, buckling) is built and verified, and the SLS PA12 rear wheel is sized, converged and exported
> (`analysis/pa12_final.step`): a closed drum with tapered faces and powder-escape holes, **m_eq 1.37 g, mass 2.83 g**.
> See [§8](#8-3d-pipeline-and-the-sls-pa12-wheel) and [`DESIGN_LOG.md`](DESIGN_LOG.md). The hex-pipeline re-sizing of the
> other material families with the new rim-edge load cases is paused (PA12 first, at the team's request).

---

## 1. Problem definition

### Regulation envelope (T7, T3)
| Rule | Requirement | Design target |
|---|---|---|
| T7.5 diameter | 28.0–32.0 mm on the contact surface | 28.2 mm (proposed, not yet confirmed) |
| T7.4 contact width | front ≥ 13.0, rear ≥ 17.0 mm (fillets/chamfers excluded) | 13.2 / 17.2 mm |
| T7.7 rolling surface | constant diameter, no tread | — |
| T7.8 rotation | free-spins on a 3° incline | — |
| T7.13 safety test | 100 g hung from a claw on the inner wheel edge | retention via bearing fit/adhesive |
| T7.9–T7.11 | wheels visible in all views | no fairings |
| T3.6 car mass | ≥ 48.0 g (+ 28.9–29.4 g cartridge) | — |
| T1.1 / T1.8 | wheel may be a multi-part assembly, adhesives allowed | enables hybrids |

### Decisions taken with the team
1. **20 N = general handling load** (pinch, push, knock). It is a no-yield / no-crack check, not a running load.
   Running load ≈ 0.19 N static per wheel plus imbalance; 20 000 rpm.
2. **Bearings:** MR63 (3×6×2.5 mm) on a **fixed axle**. Screening models use two per wheel (still to confirm).
3. **Processes:** all of the above, including lathe and 4th axis.
4. **Toolchain:** open-source scripted FEA (this repo) + Ansys Student verification.

### Load cases
| ID | Description | Boundary conditions |
|---|---|---|
| LC1 | 1 N radial at contact (5× static), ±2° line patch | bearing seats fixed |
| LC2 | 20 N radial at contact (LC1 × 20) | bearing seats fixed |
| LC3 | 20 N diametral pinch, ±6° fingertip patches, wheel off the car | symmetry only |
| LC4 | 20 N axial thumb push, Ø8 mm patch at r = 8.5 mm on outermost side face | bearing seats fixed |
| LC5 | 20 000 rpm centrifugal | bearing seats fixed |

**Pass criteria:** metals SF ≥ 1.5 on von Mises vs yield. Polymers SF ≥ 2 on max principal vs UTS. CF SF ≥ 2 on von Mises
vs a conservative 400 MPa.

---

## 2. Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.12 | scriptable, reproducible, publishable |
| Mesh topology, quadrature, facet integration | **scikit-fem 12** (`MeshHex2`, `Basis`, `FacetBasis`) | pure Python, transparent |
| Element stiffness/mass assembly | **NumPy batched BLAS** (custom, `wheelfem.py`) | ~20× faster than scikit-fem's per-dof-pair loop for 27-node hexes |
| Linear solve | **MKL PARDISO** via `pypardiso` | direct, multithreaded, handles thin-shell conditioning (AMG/CG stalled) |
| Eigen solve | SciPy `eigsh` shift-invert with PARDISO factorisation | natural frequencies |
| Verification (planned) | Ansys Student (128k-node limit) | independent solver cross-check |

Installed but currently unused: `gmsh`, `meshio`, `pyamg`, `matplotlib`. They were kept for a possible return to unstructured meshes,
AMG and plotting.

### Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd analysis && ../.venv/bin/python validate.py
```

---

## 3. Architecture

```
analysis/
├── 01_first_pass.py      closed-form screening (spin, imbalance, merit indices, bare-rim check)
├── 02_losses.py          order-of-magnitude resistance budget (aero, windage, bearings, rolling)
├── wheelfem.py           FEA core: mesh generation, assembly, loads, BCs, solve, post-processing
├── validate.py           V1 ring pinch deflection, V2 rotating disc displacement
├── validate_stress.py    V4 ring bending stress (Gauss vs surface evaluation)
├── modal.py              eigen solver + V3 ring natural-frequency validation
├── concepts.py           materials, concept geometry builders, load-case driver run()
├── size.py               fully-stressed-design sizing of each concept family
├── summarise.py          summary table + first natural frequency of sized designs
└── *.txt / *.json / summary.md   saved outputs of the runs above
```

Data flow: `concepts.py` (geometry, materials, `run()`) → `size.py` (iterates `run()`, writes `size_results.json`)
→ `summarise.py` (rebuilds the sized designs, adds modal results, writes `summary.md`). Everything is built on
`wheelfem.Model`.

### 3.1 `wheelfem.py`: the FEA core
**Geometry model.** A wheel is an axisymmetric `Design`: a list of rectangular `Block`s in (r, z), each with a
material index, plus the rolling radius, contact band, bearing-seat radius/extent and lumped extras (bearing mass
and inertia).

**Meshing (`Model.__init__`).**
1. Build tensor grids in r and z whose lines include every block edge (`_grid`). Thin walls therefore get exactly
   one element through the thickness, aligned with the wall.
2. Build the θ grid over a **half model** θ ∈ [90°, 270°]. A cubic map clusters elements near 90° and 270°
   (end spacing 0.2 × mean), where loads are applied.
3. Create a linear hex tensor mesh in (r, θ, z) and delete elements whose centroid lies in void. Each element is
   tagged with a material and a block.
4. Promote to **27-node triquadratic hexes** (`MeshHex2`), then map *every node* (including mid-edge/face/centre
   nodes) with (r, θ, z) → (r cos θ, r sin θ, z). Curved walls are therefore represented isoparametrically to
   quadratic order.

**Assembly.** Shape-function gradients come from scikit-fem at 3×3×3 Gauss points (full integration, so no
hourglass modes). The strain–displacement matrix **B** (Voigt, 6×81) is built per Gauss point. Element matrices
**Kₑ = Σ Bᵀ C B |J| w** are formed with batched `matmul` in chunks of 600 elements and summed into a CSR matrix.
Dof numbering: `3·node + component`. A consistent mass matrix is assembled the same way.

**Loads.** `traction_patch(selector, direction, F)` selects boundary facets by vertex-midpoint position, integrates
their exact area, and applies a uniform traction so the resultant is exactly `F` (half of the physical load,
because of the half model). `centrifugal(ω)` applies the body force ρω²(x, y).

**Boundary conditions.** Bearing seats are fixed in all components (`dofs_fixed_bore`). Symmetry is uₓ = 0 on x = 0
(`dofs_symmetry`). `dofs_where` constrains any component at nodes matching a predicate, including interior
symmetry planes (y = 0 for pinch, z = W/2 mid-plane).

**Solve.** Condense out the constrained dofs, then PARDISO direct solve. Several load cases sharing K and the BCs
are solved in one call with one factorisation.

**Post-processing.** `stresses(u, by="mat"|"block", at="gauss"|"nodes", el_filter=...)` returns max von Mises
and max principal stress per material or per block. `at="nodes"` evaluates at the 27 element nodes, which
includes the wall surfaces (see §5). `mass_props()` integrates mass and polar inertia over the full 360° and
returns **m_eq = I_zz / R²**, the translational mass equivalent of the wheel's spin inertia. `disp_at()` reads the
displacement at the nearest node.

### 3.2 Concept families (`concepts.py`, `size.py`)
| Builder | Topology |
|---|---|
| `drum()` | rim + two flush side faces + full-width hub tube, bearings at both ends (aero-clean, closed) |
| `web()` | rim + single central web + 6 mm hub, two bearings back-to-back (open side cavities) |
| `solid()` | solid disc baseline |

Families: Al drum/web, Ti drum, SLA drum/web, glass-filled SLA drum, PA12 drum, and hybrids H1/H3/H4/H5
(CF rim with SLA or Al faces/web and an Al hub).

### 3.3 Sizing method (`size.py`)
Fully-stressed design: each role (rim / face / web / hub) has its own thickness. Each iteration runs all load
cases, finds the worst stress ratio q = σ / (strength / SF) per role, and updates
t ← max(t_min, t·√q), since bending stress scales as 1/t². The step is clipped to ×0.6–1.6. The loop stops when every
role has q ≤ 1 and either q ≥ 0.85 or the role is at its process minimum gauge
(Al/CF 0.25, Ti 0.30, SLA 0.40, PA12 0.70 mm; **these minima are assumptions**).

---

## 4. Methods per script

| Script | Method | Key output |
|---|---|---|
| `01_first_pass.py` | thin-ring hoop stress ρv²; imbalance F = m e ω²; merit indices E^⅓/ρ, σ^½/ρ, ρ·t_min; free-ring bending under 20 N (Roark M = 0.318FR, δ = 0.149FR³/EI); race-time sensitivity Δt/t ≈ Δm/2m … Δm/m | spin is harmless; balance and inertia matter; a bare thin rim cannot carry 20 N |
| `02_losses.py` | pressure drag C_d q A; von Kármán laminar disc windage C_m = 3.87/√Re; bearing torque / R; C_rr·W | aero ≫ bearings ≫ windage ≫ rolling |
| `validate.py` | V1 ring pinch ΔD vs (π/4 − 2/π)FR³/EI; V2 rotating annulus u(R_o) vs Timoshenko plane-stress solution | −0.92 % and +0.02 % |
| `modal.py` | shift-invert Lanczos; V3 ring in-plane modes f_n ∝ n(n²−1)/√(n²+1) | n = 2–4 within 0.3 % |
| `validate_stress.py` | V4 ring bending stress vs 6M/(bt²), M = 0.182FR | Gauss −2…−10 %, surface nodes +12…+17 % (unresolved, see §5) |
| `concepts.py` | 5 load cases per design, stresses per material/block | `concepts_results.*` |
| `size.py` | fully-stressed sizing per role | `size_log.txt`, `size_results.json` |
| `summarise.py` | sized designs + first natural frequency (bore fixed) | `summary.md` |

---

## 5. Results so far

### Sized rear wheels (Ø28.2 × 17.2 mm, 2 bearings included)
| Concept | m_eq g | Mass g | Governed by | f₁ / f_rot |
|---|---|---|---|---|
| H4 CF rim + Al web/hub | 0.87 | 1.85 | min gauge | 4× |
| H3 CF rim + Al flush faces | 1.01 | 2.26 | min gauge | 5× |
| H5 CF rim + SLA web + Al hub | 1.11 | 2.33 | web, thumb push | 6× |
| SLA web | 1.18 | 2.41 | web, thumb push | 5× |
| SLA drum | 1.19 | 2.59 | faces, thumb push | 4× |
| Al web | 1.33 | 2.32 | min gauge | 3× |
| Al drum | 1.47 | 2.74 | min gauge | 4× |
| Ti drum | 2.76 | 4.60 | min gauge | 5× |
| Solid PA12 baseline | 5.43 | 11.0 | — | — |

**Caveat: do not quote these as final.** They were sized with Gauss-point stresses and with load cases biased
towards web designs. Both issues are being fixed (§6).

### Findings
- **Spin is harmless:** hoop stress 1–4 MPa, diameter growth ~1 µm (metal) to 4–15 µm (polymer).
- **Balance is critical:** a 1.5 g wheel with 25 µm CG offset produces 164 mN at 20 k rpm, 87 % of the static wheel load.
  Runout and balance targets are ≤ 5–10 µm.
- **Metals and CF are limited by manufacturing, not strength.** They sit at minimum wall with 3–6× margin, so the
  ranking is driven by ρ·t_min. That makes the achievable minimum wall the key input.
- **Polymer side faces and webs are governed by the 20 N thumb push**, which forces 0.7–1.5 mm walls.
- **Web vs drum:** the web is lighter and stiffer at the contact but leaves open cavities (aero penalty not yet quantified).
  Flush faces are aero-clean but cost 0.1–0.2 g of m_eq.
- **Resistance budget:** wheel pressure drag (~320–580 mN, area fixed by regulation minima) ≫ bearings (11–57 mN)
  ≫ face windage (~8 mN) ≫ rolling (1–4 mN).
- All sized designs have their first natural frequency ≥ 3× the 333 Hz rotation frequency.

---

## 6. What's left and why

### Done since the pause (details in DESIGN_LOG stages 4–5)
- V4 resolved (curved-element parasitic membrane stress; new `at="surface"` recovery within 0.5 % of Winkler).
- Rim-edge load cases LC6–LC8 added (both pipelines).
- 3D pipeline with spokes / variable rim thickness / fillets / buckling; SLS PA12 wheel finished (§8).

### Still open
- Re-size the Al / Ti / SLA / CF / hybrid families with LC6–LC8 (hex results in `size_results.json` are partial)
  and decide whether any of them should move to the 3D pipeline (spokes are now possible).
- Minimum-wall crossover sweep (`sweep_tmin.py`, written, not run).
- Bearing seat press-fit / bonded joint model (the clamped seat is currently excluded within 0.5 mm).

### Original notes (kept for history)
1. **Resolve stress evaluation (V4).** Gauss points under-read surface bending stress by up to 10 % at one element
   per wall. Nodal (surface) values over-read the ring theory by 12–17 %. The cause is not yet separated; candidates are
   load-patch proximity, curved-beam/axial terms, and the vM vs uniaxial comparison. *Why it matters:* sizing runs
   directly on stress, so a 10–17 % bias moves wall thicknesses. Note that `wheelfem.stresses()` now **defaults to
   `at="nodes"`** (conservative), while the saved results above used Gauss points. Re-running `size.py` will give
   different numbers.
2. **Add rim-edge load cases:** 20 N radial on a ~3 mm band at one rim edge (on-axle and pinch), 20 N axial on the rim
   lip, and a displacement probe at the edge. *Why:* every current radial load is spread over the full width and
   centred over the web. A web design's 8.4 mm overhanging 0.25 mm rim edge is never loaded, which is close to the
   failing bare-ring case. This could re-rank web vs drum.
3. **Re-size all families** with (1) and (2), then sweep the minimum wall per process and present **crossover
   curves** instead of a single winner. *Why:* the ranking is an input (t_min), not a result.

### Questions for the team (next decision round)
- Achievable minimum wall per process, CF tube source/wall/OD, lathe roundness. These set the ranking.
- One or two bearings per wheel (screening assumed two).
- Diameter: 28.2 mm is proposed, not confirmed.
- Is the car at the 48 g floor? If so, only m_eq matters, not mass. What wheel do you run now (real baseline)?
- Whether to add a race-time simulator and CFD (OpenFOAM) to trade web vs drum aero against inertia.
- Can a non-rotating hubcap on the fixed axle (p. 33 figure / T1.9) close web cavities at zero inertia?

### Later stages
- **Non-axisymmetric features** (spokes, ribbed faces) need a general 3D mesh path (gmsh) instead of the swept mesh.
- **Fillets and stress concentrations** at rim–face and hub–face junctions. Sharp corners are currently idealised.
- **Thin-wall buckling** of 0.25 mm walls under the handling loads (linear buckling or Ansys eigen-buckling).
- **CF anisotropy:** the CF rim is modelled isotropic (E = 50 GPa hoop approximation). Laminate properties are needed.
- **Bearing fit and retention:** press-fit/adhesive stresses, T7.13 claw-test retention, thermal mismatch for Al/CF.
- **Front wheel** (13.2 mm wide) using the same pipeline.
- **Manufacturing plan:** "assemble then finish-turn the OD on a bearing arbor" to hit ≤ 5 µm runout regardless of
  print accuracy, plus balancing.
- **Ansys Student cross-check** of the final design, and mesh-convergence study for the paper.
- **Experimental validation:** spin-down rig (inertia, bearing friction), runout measurement, 20 N handling tests.
- CAD/STEP export of the final geometry.

---

## 7. Reproducing
```bash
cd analysis
../.venv/bin/python 01_first_pass.py
../.venv/bin/python validate.py          # V1, V2
../.venv/bin/python modal.py             # V3
../.venv/bin/python validate_stress.py   # V4
../.venv/bin/python concepts.py          # unsized screening (~2 min)
../.venv/bin/python size.py              # sizing (~8 min)
../.venv/bin/python summarise.py         # summary.md + modal (~3 min)
```
Memory: the largest thin-wall models are ~130–150k dof and need ~2–3 GB. Solid designs are meshed coarser
automatically to avoid running out of memory.

The 2026–27 Technical Regulations PDF is copyright STEM Racing Ltd and is not included.

---

## 8. 3D pipeline and the SLS PA12 wheel

| File | Role |
|---|---|
| `geom3d.py` | parametric OCC geometry: rim thickness profile + lips, θ-varying pads, spoke sets (N, width/axial taper, phase), web, drum (tapered faces, powder holes), automatic fillets of every concave edge, half model, local mesh refinement |
| `fem3d.py` | P2 tetrahedra: vectorised K / M / K_G, PARDISO, nodal-averaged stresses, modes, linear buckling |
| `lc3d.py` | LC1–LC8 with smooth (mesh-independent) patch profiles; two spoke phases; pinch cases with minimal rigid-body restraint |
| `verify3d.py` | ring vs Winkler/Roark (< 0.6 %), PA12 web vs the hex solver (< 3 %) → `verify3d.txt` |
| `pa12.py` | SLS PA12 material/criteria, concept builders, fully-stressed sizing → `pa12_explore.json` |
| `drum_study.py`, `conv_local.py`, `pa12_final.py` | face-taper/powder-hole study, local peak convergence, final verification (convergence, buckling, modes, STEP) |
| `render3d.py`, `render_final.py` | renders → `renders3d/` |

**Result:** for SLS PA12, a closed drum (rim 0.70, faces 0.70 → 1.12 mm, hub 1.0, 1 mm fillets, 2 × Ø3 holes in the inner face)
beats every spoked/web layout by ~0.6 g of m_eq per wheel (1.37 vs ≥ 2.0 g), because its faces carry the rim-edge loads at
the rim edges. Governing check: thumb push at the face–hub fillet, q = 0.92 converged (SF 2 on Z-direction UTS, ≈ 8 % spare).
Buckling ≥ 7.7 × the handling loads, f₁ = 1700 Hz (5.1 × rotation).

```bash
cd analysis
../.venv/bin/python verify3d.py
../.venv/bin/python pa12.py                 # concept sizing (~1 h)
../.venv/bin/python drum_study.py
../.venv/bin/python pa12_final.py "DF3 drum, faces 0.70->1.12 + powder holes"   # ~45 min, ~8 GB
../.venv/bin/python render_final.py
```
gmsh needs the system OpenGL libraries: `apt-get install -y libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1`.
