# Fine runs — organic wheel topology optimisation (run on your own PC)

Everything here is driven by `run_all.py` and the list in `runs.json`. Each run does, in order:

1. **Optimise** (`analysis/topo.py`, `optimise_inertia`). The optimiser decides where material goes between
   the fixed bearing hub and a thin fixed rim skin. It can also thicken the rim inwards from the skin, wherever
   that helps. It minimises the **rotational inertia** (m_eq = I/R²), subject to the **stress limit** under
   every load case at several angles. The limit is von Mises ≤ 0.9 × UTS/2 on SLS PA12. Minimum feature size
   comes from the density filter and robust projection.
2. **Extract** (`analysis/extract.py`) a smooth, watertight STL. Every wall is forced to be **≥ t_min**
   (0.6 mm for SLS PA12). If the optimiser closed a cavity, **powder-escape holes** are drilled to it from the
   inner face. Loose, unattached bodies make a run *invalid*.
3. **Verify** (`analysis/verify_organic.py`) the real STL at full accuracy:
   - quadratic tetrahedra, full 360° model
   - every load case every 1° across a symmetry sector (rolling)
   - running stiffness and 20 000 rpm spin
   - buckling of the 3 worst cases
   - natural frequencies
4. **Render** `wheel.png` (outside, inside, cut-away, mid-width section).

Loads (decided 2026-11-25): **20 N radial press at the contact, full width** — the design force on each wheel
with the car stationary or racing — plus 20 000 rpm spin. No pinch, edge, lip or axial loads. Edit `"lcs"` in
`runs.json` to change that: the available cases are `press`, `pinch`, `press_e0`, `press_eW`, `pinch_e0`,
`lip0` and `lipW`.

## Minimum wall: 0.6 mm by default (SLS PA12)

0.6 mm is the thinnest printable vertical SLS PA12 wall in the published guides:
- [Formlabs Fuse](https://formlabs.com/support/Design-specifications-for-3D-models-Fuse-1/): 0.6 mm
- [Stratasys](https://stratasys.com/uk/stratasysdirect/resources/articles/ls-design-tips-wall-thickness): 0.6 mm possible, 0.8 mm recommended
- [Murtfeldt](https://murtfeldt.de/en-GB/Additive-manufacturing/Design-tips/): > 0.6 mm
- service bureaus quote 0.7–1.0 mm for reliability ([Xometry](https://xometry.eu/en/?p=1593), [Protolabs](https://3dhubs.com/knowledge-base/designing-parts-sls-3d-printing), [Materialise](https://www.materialise.com/en/manufacturing/materials/pa-12-sls))

The rim is a vertical wall when the wheel prints lying flat. The fixed rim skin and the minimum wall on the
output geometry are therefore both 0.6 mm by default, with filter radius `rf` = 1.2 mm (2 × the minimum).
At 1.2 mm the optimiser left some members thinner than 0.6 mm in three of four press + pinch runs, and the
min-wall rule then added ~0.25 g; rf 1.5 (r08) avoids that but doubles the optimised material, which is worse. Raise `t_rim`/`t_min` in `runs.json` if your printer or bureau needs more.

**The rim thickens itself where needed.** Only the 0.6 mm skin is fixed. The 2.5 mm band inside it
(`rim_zone`, meshed at 0.15 mm) is design space, so the optimiser chooses the rim thickness at every angle
and every position across the width. In the reference result (`reference/rim_thickness.png`), 93 % of the
rim stays at the skin thickness. The other 7 % grows into 1–3 mm pads exactly where the spokes join.

## Loads: the wheel rolling on flat ground

The 20 N radial press is applied at every position the wheel can roll into.
Thanks to the N-fold symmetry, one sector covers all of them.

- **Optimiser:** a load every ~5° across the sector (`n_phi`; 14 per 72° sector for N 5). 20° spacing failed
  between the sampled angles and 10° needed a hand-lowered stress target; 5° is about half the load patch width.
- **Final check:** a load every 1° (`verify_step_deg`), which is effectively continuous.

There is no uniform pressure around the wheel, because there's no tyre to create one. Spin (20 000 rpm)
is the only uniform load, and it is included. Masses and m_eq are for the **wheel body only; bearings are
excluded**.

## Running on GitHub Actions

Use the `.github/workflows/fine-runs.yml` workflow (Actions tab, "Run workflow"). A job stops at 320 minutes,
uploads its checkpoint and **dispatches the workflow again by itself**; the new job resumes exactly from the
checkpoint. This repeats up to `max_attempts` (default 5) jobs per run.

- The job fails (red) if `run_all.py` crashes, the optimiser hits its iteration cap without converging, or the
  full check fails (stress ratio > 1, buckling < 2). f1 below 3 × rotation (1000 Hz) is a warning.
- `run_all.py` discards a checkpoint whose `runs.json` settings have changed.
- Small result files (summary, log, render, density field) are pushed to the `fine-run-results` branch.
- `extract` reports `min_wall_added_g`; if it's large, raise `rf`.

## Setup (once)

Python 3.11–3.13, 64-bit. From the repository root:

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r fine_runs/requirements.txt
```

On Windows the interpreter is `.venv\Scripts\python` instead of `.venv/bin/python`.

On Linux, gmsh also needs the OpenGL libraries:

```bash
sudo apt-get install -y libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1
```

- **Intel/AMD (Windows or Linux):** the MKL PARDISO solver (`pypardiso`) is used, which is fast.
- **Apple Silicon:** there's no MKL, so the code falls back to SciPy's SuperLU. That works but is much slower
  and needs much more RAM. Use a Windows or Linux x86 machine if you can.

## Check the install (2–3 min)

```bash
cd fine_runs
```

```bash
../.venv/bin/python run_all.py --smoke
```

It runs a 40-iteration very coarse version of `r01`. That is not converged, so the extract step is expected
to report *"loose bodies … verification skipped"*. It checks that the optimiser and extraction run.

To check that the verification stage (gmsh, P2 solver, buckling) also works on your machine, run the
reference case in `reference/`. It re-verifies a converged design that is shipped with the package:

```bash
../.venv/bin/python run_all.py --reference
```

## Run

```bash
../.venv/bin/python run_all.py
```

To run only some of them:

```bash
../.venv/bin/python run_all.py r01 r02
```

To optimise and extract without the verification step:

```bash
../.venv/bin/python run_all.py --no-verify
```

Everything is resumable. If you stop it with Ctrl-C or the PC sleeps, start the same command again. The
optimiser resumes from its last save (every 5 iterations), and finished stages are skipped.

## Runs in `runs.json`

| run | what it is |
|---|---|
| r09_N5_press_skin0.6 | SLS PA12, 0.6 mm skin and min wall, 5-fold symmetry, radial press only, rf 1.2, up to 300 iterations |
| r10_N5_press_from_r02 | r09's settings, started from r02's converged design (`seeds/r02_it117.npz`): r09 from scratch settled into a heavier two-web layout, while r02's single web already satisfies press-only |
| r11_N5_resin_rim0.3_from_r10 | SLA resin (team TDS: E 2100 MPa, UTS 52 MPa, 1.15 g/cc; nu 0.38 assumed), 0.3 mm rim skin and min wall, press only, N5, rf 1.2, warm start from r10 (`init_from`, `beta0` 8) |
| r12_N5_resin_rim0.3 | the same from scratch |

r08 (the same with rf 1.5) was stopped after 190 iterations: the larger filter doubled the optimised material
(m_eq 1.31 g vs 1.06 g for r02) and peak stress kept spiking 28–45 MPa, so it would not have converged.

The earlier runs r01–r07 (press **and pinch**) are recorded below and on the `fine-run-results` branch; their
settings are in git history.

Symmetry order N is a balance requirement: N-fold symmetry makes the wheel statically balanced. Inside each
sector the shape is completely free.

## Reference result computed here (coarse, `reference/`): a valid organic wheel

r01's setup on a coarse optimisation grid (h 0.9 mm, loads every 10°). The optimiser grew **organic,
branching spokes** between the hub and the thin rim (`reference/wheel.png`). Printable mesh:
`reference/organic_wheel_r01.stl`. Full-accuracy check of that STL:

| m_eq (wheel only) | mass (wheel only) | max stress / allowable | buckling | f1 | min wall |
|---|---|---|---|---|---|
| **1.202 g** | 1.53 g | **0.72** (0.73 at h 1.0, 0.72 at h 0.8) | ≥ 6.1 | 1015 Hz | no ray < 0.70 mm (60k rays) |

How it got there:
- **Load spacing.** A first attempt with loads every 20° failed (1.38) at the angles between the sampled ones.
- **Stress target.** With loads every 10°, the coarse grid under-read the real peak by about 5 %, so the
  stress target was lowered in two continuations, 16.8 → 15.5 → 14.5 MPa, until the full check passed.

The fine runs use a finer grid and loads every ~10° from the start; if one still fails, lower `sigma_target`.

## Fine result: r02 (N 5, PA12, 0.6 mm skin and min wall) — passes, best so far

Optimised on GitHub Actions (117 iterations, loads every 5°). Extracted and checked here with the
rim skin kept at its exact thickness. Files are in `results_verified/r02_N5_skin0.6/`.

| m_eq (wheel only) | mass | max stress / allowable (rolling, every 1°) | buckling | f1 | wall p1 |
|---|---|---|---|---|---|
| **1.075 g** (coarse reference 1.202 g) | 1.46 g | **0.80** (pinch@21) | 5.7 | 1080 Hz | 0.59 mm |

On the fine grid with N 5, the optimiser converged to a **single thin central web**: a near-axisymmetric
diaphragm with a thickened bearing seat. It did not grow discrete spokes. Under radial press and pinch,
a continuous web costs less inertia than spokes at this wall thickness.

The resin run (r07, Accura AMX, 0.3 mm) was at iteration 134 at its last checkpoint and is not yet final.
There, the optimiser formed a hollow box of two ~0.45 mm side walls (m_eq ≈ 1.06 g at the checkpoint).

## Resin result: r11 (SLA resin from the team TDS, 0.3 mm rim, radial press only, N 5) — passes, lightest so far

Files: `results_verified/r11_N5_resin_rim0.3_from_r10/`. Material: 1.15 g/cc, E 2100 MPa, UTS 52 MPa (ASTM D638),
nu 0.38 assumed; allowable 26 MPa (SF 2). Warm start from r10, converged at iteration 201. Full check: P2 tets
(407k dof), press rolled every 1°, spin, buckling, modes; wheel body only.

| I (g·mm²) | m_eq | mass | stress / allowable | buckling | f1 | min-wall rule added | wall p1 / median |
|---|---|---|---|---|---|---|---|
| **196** (r10: 214, −8 %) | 0.985 g | 1.26 g | **0.81** (press@69) | 3.3 | **574 Hz (1.7 × rotation)** | 0 g | 0.30 / 0.42 mm |

Shape: 0.3 mm rim skin with thickened bands at both rim edges, five organic branching spokes, bearing hub.

- **Watch f1:** the first two modes (574/577 Hz, a pair, i.e. rim ovalisation) are only 1.7 × the 333 Hz rotation
  frequency, below the 3 × guideline. r10 has 1085 Hz. Running stiffness is 1.7–3.7 µm/N (r10 0.6–0.8).
- The thin rim needs much more support than the 0.6 mm PA12 rim (optimised material ≈ 0.47 g of m_eq vs 0.26 g),
  and the resin is 24 % denser, so halving the rim saves only 8 % of I.
- **r12** (the same from scratch) was stopped at iteration ~177 at m_eq 1.25 g: like r09, the from-scratch run
  settles into a heavier layout than the warm start.
- The first check of r11 failed in the volume mesher: the extraction's snap tolerance (0.08 mm) was close to the
  0.1 mm voxel and folded 166 rim triangles. Fixed (tol ≤ voxel / 2) and re-checked with `--re-extract`.

## Current result: r10 (radial press only, N 5, PA12, 0.6 mm) — passes

Files: `results_verified/r10_N5_press_from_r02/` (summary, log, render). Full check: P2 tets, press rolled every 1°
(72 positions), spin, buckling, modes; wheel body only.

| I (g·mm²) | m_eq | mass | stress / allowable | buckling | f1 | min-wall rule added | wall p1 |
|---|---|---|---|---|---|---|---|
| **214** | 1.076 g | 1.46 g | **0.74** (press@47) | 6.2 | 1085 Hz (3.3 × rotation) | 0 g | 0.60 mm |

Shape: 0.6 mm rim, a single thin central web (diaphragm) and the bearing hub — no spokes.

How it was reached:
- **r08** (rf 1.5) was stopped at iteration 190: the larger filter doubled the optimised material (m_eq 1.31 g).
- **r09** (rf 1.2, from scratch) settled into a two-web layout at m_eq ≈ 1.24 g and was stopped at iteration ~115+.
  Topology optimisation finds a local optimum; the first ~20 iterations matched r02 exactly, then it branched.
- **r10** starts from r02's converged design with press only. It converged on its first iteration: the design did
  not move (max change 0.004) and the press stress was already on target (18.4 of 18.9 MPa).
  **Dropping pinch frees no material for this wheel:** in r02's 117 iterations the worst case was a press on all but
  one iteration, so the press always set the wall thicknesses.

### Earlier fine runs, press + pinch loads (full check: P2 tets, rolling every 1°, wheel only)

Headline figure: **moment of inertia I about the axle, in g·mm²** (wheel only, no bearings). Lower is better.

| run | **I (g·mm²)** | mass | stress / allowable | buckling | f1 | min-wall rule added | status |
|---|---|---|---|---|---|---|---|
| r02 N5, 0.6 mm | **214** | 1.46 g | 0.80 | 5.7 | 1080 Hz | 0 g | **pass, best valid** |
| r03 N4, 0.6 mm | 248 | 1.89 g | 0.49 | 16.5 | 1340 Hz | 0.27 g | pass, heavy |
| r04 N8, 0.6 mm | 248 | 1.79 g | 0.63 | 13.5 | 1617 Hz | 0.23 g | pass, heavy (iteration cap) |
| r05 N6, 0.4 mm | 176 | 1.25 g | **1.27** | 3.1 | 436 Hz | 0 g | FAIL: stress (pinch) |
| r01 N6, 0.6 mm | — | — | — | — | — | — | runner lost at iteration 104 → continued as **r01b** |
| r01b N6, 0.6 mm | 249 | 1.83 g | 0.53 | 17.3 | 1589 Hz | 0.25 g | pass, heavy (clean resume, converged at 164) |
| r05b N6, 0.4 mm, target 14 MPa | 207 | 1.46 g | 0.77 | 5.5 | **462 Hz** | 0 g | pass on stress, but f1 only 1.4 × rotation; no real process matches 0.4 mm walls with PA12 properties |
| r07 resin, 0.3 mm | ≈211 (checkpoint) | ≈1.60 g | — | — | — | — | its resume diverged → continued as **r07b** |
| r07b resin, 0.3 mm | — | — | — | — | — | — | stopped: oscillated (peak stress 22–132 MPa vs 22.9 target) for 80 iterations, never converged |
| *coarse reference (spoked)* | *239* | 1.53 g | 0.72 | 6.1 | 1015 Hz | — | pass |

- **r03/r04:** the optimiser drew features thinner than 0.6 mm. The min-wall rule then added 0.23–0.27 g at extraction,
  which makes both wheels heavy and over-strong. Even before that rule ran, their m_eq (1.15 g) was above r02's,
  so they are not worth re-running.
- **r05 (0.4 mm wall) is the only fine run with clear organic spokes** (6 branching spokes between two side walls).
  It also has the lowest inertia. The optimiser under-read its peak stress by about 27 %: 0.4 mm members sit on a
  0.5 mm grid. **r05b** re-runs it with `sigma_target` 14 MPa instead of 18.9 MPa.

### Resume bug (fixed): why r01b and r07b exist

Checkpoints used to save only the design. A resumed job restarted MMA with wide initial asymptotes and reset the
stress normalisation, so its first steps were over-confident and could cut a load path:
- r07 went from 23 to 891 MPa peak stress within 15 iterations, then recovered into a heavier design (m_eq 1.02 → 1.14 g).
- r04 jumped from 29 to 79 MPa.

Checkpoints now hold the full optimiser state, and a stop/resume reproduces the uninterrupted run bit for bit.
**r01b** and **r07b** continue r01 and r07 from their last healthy checkpoints (`seeds/`, via the `seed` field in
`runs.json`). They replace r01 and r07 in the workflow matrix.

## Rim thickness vs process — read this

The rim skin can't be thinner than the process's minimum printable wall. Extraction forces every wall,
including the skin, to at least `t_min`. A full-width rim at R = 14.1 has m_eq of:

| rim skin | rim m_eq on its own |
|---|---|
| 0.6 mm (SLS PA12, default) | 0.79 g, the floor for r01–r04 |
| 0.4 mm | 0.54 g |
| 0.25 mm | 0.35 g |

**r05 (0.4 mm skin) is the run that matches "make the rim really thin"**, but only if your process can
print 0.4 mm walls. The optimiser thickens the rim inwards from the skin only where the loads need it.

## Time and memory (estimates, not yet measured on the fine grid)

These are extrapolated: the fine grid (h 0.5 with the 0.15 mm rim band, ~220k elements, ~700k DOF) has not
been run to completion on the development box.

The reference is the 4-core / 15 GB cloud box this was developed on.

| stage | h = 0.5 mm grid (r01–r05) | h = 0.4 mm grid (r06) |
|---|---|---|
| optimise, per iteration | ~60–75 s (4 cores) | ~2–3 min |
| optimise, total (~120–200 iterations) | ~2.5–4 h on 4 cores, ~1–1.5 h on 12–16 cores | 6–10 h |
| optimise RAM | ~10 GB | ~20 GB |
| verify (verify_h 0.8 mm) | ~10–20 min, ~12–16 GB | same |

On a 32 GB machine you can lower `verify_h` to 0.6 for a more accurate check (~25–30 GB). On a 16 GB machine
keep 0.8, or use `--no-verify` and send back the STLs.

## What to send back

The whole `fine_runs/results/` folder: `summary.csv`, plus per run `summary.json`, `log.txt`, `wheel.png`,
`wheel.stl` and `density.npz`. Commit it on a branch with `git add -f fine_runs/results` (the folder is git-ignored so half-finished runs are not committed by accident), or zip it. The STLs can be large (50–80 MB), so zip
them or leave them out if they're too big for git.

## Reading the results

`results/summary.csv` columns:

| column | meaning |
|---|---|
| `m_eq_g` | rotational inertia as equivalent mass of the **wheel body only, bearings excluded** (lower = faster car) |
| `stress_ratio` | max(σ₁, σ_vM) / allowable over all load cases and angles; must be ≤ 1 |
| `buckling_min` | lowest buckling load factor of the checked cases; must be ≥ 2 |
| `f1_Hz` | first natural frequency; should be ≫ 333 Hz (20 000 rpm) |
| `wall_p1_mm` | 1st-percentile wall thickness of the output; should be ≥ t_min |
| `holes` | powder holes added because the optimiser closed a cavity |

If `stress_ratio` comes out above 1, the coarse optimisation grid under-read a stress peak. Re-run that case
with a lower `sigma_target`: add `"sigma_target": 16.0` to its entry, next to `"setup"`.
