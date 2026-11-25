# STEM Racing Pro wheel — design log

## Regulation envelope (2026-27 Pro Tech Regs)
| Rule | Requirement | Target (margin) |
|---|---|---|
| T7.5 diameter | 28.0–32.0 mm, measured on contact surface | 28.2 mm (proposed) |
| T7.4 contact width | front ≥ 13.0, rear ≥ 17.0 mm, chamfers/fillets excluded | 13.2 / 17.2 mm |
| T7.7 rolling surface | constant diameter across contact width, no tread | — |
| T7.8 rotation | must free-spin on a 3° incline | — |
| T7.13 safety test | 100 g hung from 3-finger claw on inner wheel edge; no detachment or shape change; 1.5 mm claw clearance to body | — |
| T7.12 / T1.9 support | support system inside the cylinder swept by the wheel diameter | — |
| T7.9–T7.11 | wheels visible in top/bottom/side; front view obstruction ≤ 20 mm high | no fairings |
| T3.6 car mass | ≥ 48.0 g without cartridge (cartridge 28.9–29.4 g) | — |
| T3.9 spares | 4 front + 4 rear wheels, identical geometry | — |
| T1.1 | wheel may be a multi-part assembly; adhesives allowed | — |

## Stage 1 — first-pass physics (`analysis/01_first_pass.py`, output in `01_first_pass.txt`)
- 20 000 rpm at Ø28.2 → ω = 2094 rad/s, rim speed 29.5 m/s.
- **Spin is not the problem.** Hoop stress ρv² = 0.9–3.9 MPa (SF 33–228). Diameter growth ~1 µm metal, 4–15 µm polymer.
- **Balance is a big deal.** 1.5 g wheel with 25 µm CG offset → 164 mN, ~87 % of the 189 mN static wheel load. Runout/balance targets need to be ≤ 5–10 µm.
- **Rotational inertia is the main performance lever.** Each 1 g of m_eq = I/R² removed across the car ≈ 0.6–1.3 % race time.
- **20 N is 106× static wheel load and governs structure.** An unsupported min-gauge rim fails (Al 0.25 mm: ~500 MPa, ~5 mm ovalisation). A self-supporting rim needs 1.4–5.5 mm wall (m_eq 4.7–8 g per wheel), which is unacceptable. So the rim must be **supported internally** (web / spokes / core), and support layout sets the mass. This is the FEA/optimisation stage.
- At min gauge, SLA resin gives the lightest rim (0.47 kg/m²). Al/Ti win where stiffness or precision govern (hub, bearing seat).

## Decisions (2026-11-22)
1. 20 N = **general handling** load → no-yield/no-crack check (metals SF 1.5 on vM; polymers SF 2 on max principal vs UTS; CF SF 2). Stiffness only matters at running load.
2. Bearings: **MR63 (3×6×2.5) on a fixed axle**, two per wheel in the screening models.
3. Processes: all available (lathe, 4th axis, SLA/DLP, SLS, FDM, CF).
4. Toolchain: scripted open-source FEA (reproducible, goes in the paper) + user cross-checks in **Ansys Student**.

## Stage 2 — FEA pipeline (`analysis/wheelfem.py`)
- 27-node hex, cross-section swept in θ and mapped isoparametrically; half model with symmetry; vectorised assembly; PARDISO solve.
- Validation (`validate.py`, `modal.py`): thin-ring pinch −0.9 % vs Roark; rotating disc +0.02 %; ring modes n = 2–4 within 0.3 %.

## Stage 3 — concept screening + fully-stressed sizing (`concepts.py`, `size.py`, `summary.md`)
Load cases: LC1 1 N running radial; LC2 20 N radial on axle; LC3 20 N pinch off-car; LC4 20 N thumb push on the side face (Ø8 mm at r = 8.5); LC5 20 000 rpm.
Rear wheel Ø28.2 × 17.2, including 2 bearings (0.64 g mass, 0.013 g m_eq) in this stage's table; later stages exclude them. Min walls assumed: Al/CF 0.25, Ti 0.30, SLA 0.40, PA12 0.70 mm.

| concept | m_eq g | mass g | governs |
|---|---|---|---|
| H4 CF rim + Al web/hub | **0.87** | 1.85 | min gauge |
| H3 CF rim + Al flush faces | 1.01 | 2.26 | min gauge |
| H5 CF rim + SLA web + Al hub | 1.11 | 2.33 | web: thumb push |
| SLA web | 1.18 | 2.41 | web: thumb push |
| SLA drum (flush faces) | 1.19 | 2.59 | faces: thumb push |
| Al web | 1.33 | 2.32 | min gauge |
| Al drum | 1.47 | 2.74 | min gauge |
| Ti drum | 2.76 | 4.60 | min gauge |
| Baseline solid PA12 | 5.43 | 11.0 | — |

Findings:
- **Metals and CF are limited by manufacturing, not by strength**: walls sit at the minimum gauge with 3–6× strength margin. Areal density at minimum wall (ρ·t_min) ranks the rim: CF 0.25 mm (0.39 kg/m²) < SLA 0.4 (0.47) < Al 0.25 (0.70).
- **Polymer side faces are governed by the 20 N thumb push**, which forces 0.7–1.5 mm faces/webs.
- A central web gives a stiffer contact (0.05 vs 1–11 µm/N) and lower inertia, but leaves open cavities (aero/windage penalty, needs CFD to quantify). Flush faces are aero-clean but cost 0.1–0.2 g m_eq.
- All sized designs have their first natural frequency ≥ 3× the 333 Hz rotation frequency.
- Loss budget (`02_losses.py`): wheel pressure drag ~320–580 mN ≫ bearings 11–57 mN ≫ face windage ~8 mN ≫ rolling ~1–4 mN.
- Every designed wheel cuts m_eq by 4–5× vs a solid wheel: ~4.5 g m_eq per wheel, ~15 g over the car.

## Open decisions (asked 2026-11-22, round 2)
Face architecture (drum vs web), realistic minimum walls / CF tube sourcing, whether to add a race-time simulator and CFD.

## Stage 4 — stress recovery, rim-edge loads (2026-11-22, `wheelfem.py`, `validate_stress.py`)
- **V4 resolved.** Nodal stresses over-read by 3–8 %: curved quadratic hexes carry a parasitic membrane stress that is parabolic in θ, largest at the element ends. Gauss points under-read by up to 20 % (inside the wall). New default `at="surface"`: wall surfaces in r/z, 2-point Gauss stations in θ → narrow ring within 0.5 % of Winkler on every mesh. The remaining +7 % on a 5 mm-wide ring is the physical anticlastic free-edge effect (mesh-converged).
- Stress recovery also averages per material, charges junction points to the thinner wall, and skips 0.5 mm around sharp re-entrant corners (singular).
- **Rim-edge load cases added:** LC6 edge press, LC7 edge pinch (20 N on a 3 mm band at one rim edge), LC8 rim-lip axial push (20 N, outer 1 mm of the end face). Old LC1–LC4 never loaded an overhanging rim edge; with them, a central web loses to a drum (Al web 1.67 g vs Al drum 1.47 g m_eq). The hex re-sizing of all families was paused at the user's request (SLS PA12 first).

## Stage 5 — 3D pipeline and the SLS PA12 wheel (2026-11-22)
**Why no spokes before:** the hex pipeline sweeps an r–z cross-section around the axis, so it could only represent axisymmetric wheels. Spokes and θ-varying rim thickness were impossible by construction, not rejected.

**New 3D pipeline** (`geom3d.py`, `fem3d.py`, `lc3d.py`, `verify3d.py`): gmsh/OpenCASCADE parametric CAD (rim thickness profile along the width, inward lips, θ-varying rim pads at spoke roots, spoke sets with N / width taper / axial taper / phase, web, drum faces with radial taper, powder-escape holes, automatic fillets on every concave edge), P2 tetrahedra, PARDISO, linear buckling (symmetric + antisymmetric mode families), modes, STEP export.
- Verification: ring pinch vs Winkler/Roark within 0.6 % / 0.1 % from 1 to 4 elements through the wall; PA12 web vs the validated hex solver within 3 % on every load case.
- Loads are smooth patch profiles (cos² in θ, smoothed band edges): uniform patches narrower than an element gave mesh-dependent load distributions (edge deflection drifted 14 %).
- Spoked wheels are evaluated with a spoke under the load and between spokes (worst case used).

**SLS PA12 (PA2200-class):** E 1700 MPa, ρ 930 kg/m³, UTS 42 MPa (Z, conservative). Allowable max(σ₁, σ_vM) = 21 MPa (SF 2). Min wall 0.7 mm; hub wall ≥ 1.0 mm (bearing retention); 1.0 mm junction fillets (at 0.5 mm the spoke-root Kt grew with spoke thickness and sizing stalled).

| concept (sized to LC2–LC8) | m_eq g | mass g | governing |
|---|---|---|---|
| central web | 2.10 | 3.45 | rim LC7, web LC8 |
| central web + lips | 2.13 | 3.61 | LC7 |
| 6 spokes + lips | ~1.94–2.05 (oscillated, not converged) | 3.2–3.4 | LC8 at spoke roots, LC7 rim |
| 6 spokes + lips + θ-pads | 2.01 | 3.22 | LC8, LC7 |
| twin 6-spoke planes at the edges | > 1.9 and still failing (stopped; meshing fragile) | — | LC6/LC7 |
| **closed drum** | **1.36** | **2.78** | faces LC4, everything else at min gauge |

**Why spokes lose for PA12:** the 20 N rim-edge and rim-lip loads act on an 8.6 mm overhang (rim edge to a central web/spokes) or produce 20 N × 14 mm ≈ 280 N·mm at the spoke roots. Low-modulus PA12 then needs 1.2–1.4 mm rims and 2.5–3 mm spokes. A drum's faces sit at the rim edges and carry those loads in-plane, so the rim stays at the 0.7 mm minimum and only the faces are sized (by the thumb push).

**Final SLS PA12 rear wheel** (`pa12_final.step`, `renders3d/pa12_final.png`): closed drum, rim 0.70 mm, faces 0.70 mm at the rim tapering to 1.12 mm at the hub, hub 1.0 mm, 1 mm fillets, 2 × Ø3 powder-escape holes at r = 8.5 in the inner face (needed: a sealed SLS cavity cannot be de-powdered; inner face keeps the outer face aero-clean). **m_eq 1.357 g, mass 2.19 g** (wheel only; 1.370 / 2.83 g with the 2 bearings).
- Stress (converged): governing q = 0.92 at the face–hub fillet under the thumb push (local refinement 0.25 → 0.20 mm: 0.903 → 0.918; extrapolated ≈ 0.93). Rim 0.75 (edge pinch), hub 0.62. All other cases lower. Hole position (under/between loads) changes q by < 1 %.
- Global convergence (h 1.4 → 1.0 → 0.8 mm, 0.18–0.42 M dof): deflections and m_eq within 1 %.
- Buckling (taper-1.5 design, i.e. conservative): lowest load factor 7.7 (pinch / edge pinch, antisymmetric), ≥ 13 for everything else.
- First natural frequency 1700 Hz = 5.1 × the 333 Hz rotation.
- Stiffness: running 5.7 µm/N at the contact centre, 0.15 µm/N at the edge; spin growth 13 µm on diameter at 20 000 rpm (≤ runout target scale: check balance/runout together).
- Not yet modelled: bearing press-fit / bonded seat stresses (clamped-seat singularity excluded within 0.5 mm), SLS anisotropy beyond using the Z strength, creep, real contact.


## Stage 6 — organic topology optimisation (2026-11-23/24)
Engine: `analysis/topo.py` (minimum rotational inertia subject to the stress limit, density method, MMA, adjoint
surface stresses, filter + robust projection, cyclic symmetry), `extract.py` (STL, min wall by medial-surface balls,
powder holes into enclosed cavities), `verify_organic.py` (full-model P2 tets, all load angles, buckling, modes).
Fixed parts only: bearing hub and a thin rim skin; the optimiser decides everything else, including rim thickening.
Loads (as instructed): radial press + diametral pinch, 20 N, full width, no edge / axial loads.

**Mass and inertia are reported for the wheel body only, bearings excluded (from 2026-11-24).**

Reference result (coarse grid, `fine_runs/reference/`): organic branching spokes, **m_eq 1.202 g, mass 1.53 g**
(parametric drum 1.357 g / 2.19 g), max stress 0.72 x allowable (mesh-checked), buckling >= 6.1, f1 1015 Hz,
walls >= 0.70 mm. Load angles must be sampled finely: 20 deg spacing over-fitted (1.38 x allowable between angles).
Fine runs: `fine_runs/` (GitHub Actions workflow `.github/workflows/fine-runs.yml`).

### Stage 6 — fine run r02 verified (PA12, N 5, 0.6 mm)
- After fixing the extraction (the rim skin stays at its exact 0.6 mm; tiny voids are filled; the min-wall field is smoothed),
  r02 is a single watertight shell.
- Full P2 check, rolling every 1°:
  - m_eq 1.075 g, mass 1.46 g
  - stress ratio 0.80 (pinch)
  - buckling 5.7
  - f1 1080 Hz
- That is 11 % lower m_eq than the coarse reference (1.202 g).
- Topology: a thin central web (diaphragm), not discrete spokes.
- The resin run r07 (0.3 mm) was still optimising at its last checkpoint. That checkpoint geometry is a hollow two-wall box
  (m_eq ≈ 1.06 g). Its final result is pending on GitHub Actions.

### Stage 6 — fine-run batch (GitHub Actions run #10)

| run | m_eq (g) | stress / allowable | buckling | f1 (Hz) | note |
|---|---|---|---|---|---|
| r03 N4 | 1.249 | 0.49 | 16.5 | 1340 | min-wall rule added 0.27 g |
| r04 N8 | 1.247 | 0.63 | 13.5 | 1617 | min-wall rule added 0.23 g |
| r05 N6, 0.4 mm | 0.885 | **1.27** (fail) | 3.1 | 436 | organic spokes |

- Neither r03 nor r04 beats r02 (1.075).
- r05 fails on stress: the optimiser under-reads 0.4 mm members on its 0.5 mm grid by about 27 %. Queued **r05b**: the same setup
  with sigma_target 14 MPa.
- r01's runner was lost at iteration 104; it is resuming.
- r07 (resin) is still optimising.
- **Resume bug fixed.** Checkpoints did not store the MMA state (asymptotes, previous iterates) or the stress normalisation.
  - Every cross-job resume restarted with wide asymptotes, and the optimiser cut load paths: r07 went 23 → 891 MPa, then recovered
    heavier at m_eq 1.14 g instead of 1.02 g. r04 went 29 → 79 MPa.
  - Now saved and restored; a stop/resume equals the uninterrupted run (max |Δx| = 0).
  - r01b and r07b continue from the last healthy checkpoints (`fine_runs/seeds/`).
  - r03 and r04 were also hit by the bug, but at the iterations where they resumed their designs were already heavier than r02, so they were not re-run.
