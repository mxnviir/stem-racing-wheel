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
Rear wheel Ø28.2 × 17.2, including 2 bearings (0.64 g). Min walls assumed: Al/CF 0.25, Ti 0.30, SLA 0.40, PA12 0.70 mm.

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
