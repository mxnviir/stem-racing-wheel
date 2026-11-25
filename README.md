# STEM Racing Pro-Class Wheel: organic topology optimisation

Research code for designing the lowest-inertia robust wheel for a STEM Racing Professional Class car
(2026–27 UAE Technical Regulations). The wheel shape between the bearing hub and a thin rim skin is found by
3D topology optimisation (no shape family is prescribed), extracted as a printable STL and checked at full
accuracy. Final designs are meant to be cross-checked independently in Ansys Student.

> **All masses and m_eq are for the wheel body only (bearings excluded).** Headline figure: moment of inertia
> about the axle, I = m_eq · R² (g·mm²); lower is faster.
>
> Results, run list and how to run: [`fine_runs/README.md`](fine_runs/README.md). History: [`DESIGN_LOG.md`](DESIGN_LOG.md).

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
| T1.1 / T1.8 | wheel may be a multi-part assembly, adhesives allowed | — |

### Requirements
1. **20 000 rpm** spin.
2. **20 N on each wheel** — the general design force on a wheel with the car stationary or racing. It must not
   break (SF 2 on UTS for polymers). For reference, the real static load is ≈ 0.19 N per wheel, so 20 N is a
   ~100× envelope.
3. **Bearings:** MR63 (3×6×2.5 mm) on a fixed axle, two per wheel (still to confirm).

How the 20 N is applied in the optimiser and the check (which load cases, contact-patch width, how often around
the wheel) is set per run in `fine_runs/runs.json` (`"lcs"`, `n_phi`, `verify_step_deg`).

---

## 2. Pipeline

| File | Role |
|---|---|
| `analysis/topo.py` | topology optimisation: minimum rotational inertia subject to the stress limit under every load case at every sampled angle; density method, MMA, adjoint stresses, density filter + robust projection, N-fold cyclic symmetry (static balance) |
| `analysis/extract.py` | density field → smooth watertight STL; minimum wall enforced; powder-escape holes into enclosed cavities |
| `analysis/verify_organic.py` | full-accuracy check of the STL: gmsh quadratic tets, full 360° model, loads rolled every 1°, spin, buckling, natural frequencies |
| `analysis/fem3d.py` | P2-tet solver used by the check (PARDISO; stresses, buckling, modes) |
| `analysis/validate_fem3d.py` | solver validation: thin ring pinch vs Winkler / Roark → `validate_fem3d.txt` (within 0.6 % / 0.1 %) |
| `fine_runs/run_all.py` | runs optimise → extract → verify → render for each entry in `runs.json`; resumable |
| `.github/workflows/fine-runs.yml` | the same on GitHub Actions (320-min jobs, resume across jobs) |
| `analysis/01_first_pass.py`, `02_losses.py` | closed-form physics: spin stress, imbalance, resistance budget |

### Closed-form findings (`01_first_pass.py`, `02_losses.py`)
- **Spin is harmless:** hoop stress 1–4 MPa, diameter growth a few µm.
- **Balance is critical:** a 1.5 g wheel with 25 µm CG offset produces 164 mN at 20 000 rpm, 87 % of the static wheel load.
  Runout and balance targets are ≤ 5–10 µm.
- **Resistance budget:** wheel pressure drag (~320–580 mN) ≫ bearings (11–57 mN) ≫ face windage (~8 mN) ≫ rolling (1–4 mN).

---

## 3. Setup
```bash
python3 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
On Linux gmsh needs the OpenGL libraries:
```bash
sudo apt-get install -y libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1
```
Solver check:
```bash
cd analysis && ../.venv/bin/python validate_fem3d.py
```
Then follow [`fine_runs/README.md`](fine_runs/README.md).

The 2026–27 Technical Regulations PDF is copyright STEM Racing Ltd and is not included.

---

## 4. Open questions
- Diameter (28.2 mm proposed), one or two bearings, achievable minimum wall per process.
- Front wheel (13.2 mm wide) not designed yet.
- Bearing seat press-fit / bonded joint (the seat is currently clamped and excluded within 0.5 mm).
- Ansys Student cross-check and physical tests (mass, runout, balance, spin-down, 20 N load test) of a printed wheel.
