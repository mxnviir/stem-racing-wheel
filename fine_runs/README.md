# Fine runs — organic wheel topology optimisation (run on your own PC)

Everything here is driven by `run_all.py` and the list in `runs.json`. Each run does, in order:

1. **Optimise** (`analysis/topo.py`, `optimise_inertia`). The optimiser decides where material goes between
   the fixed bearing hub and a thin fixed rim skin. It can also thicken the rim inwards from the skin, wherever
   that helps. It minimises the **rotational inertia** (m_eq = I/R²), subject to the **stress limit** under
   every load case at several angles. The limit is von Mises ≤ 0.9 × UTS/2 on SLS PA12. Minimum feature size
   comes from the density filter and robust projection.
2. **Extract** (`analysis/extract.py`) a smooth, watertight STL. Every wall is forced to be **≥ t_min**
   (0.7 mm for SLS). If the optimiser closed a cavity, **powder-escape holes** are drilled to it from the
   inner face. Loose, unattached bodies make a run *invalid*.
3. **Verify** (`analysis/verify_organic.py`) the real STL at full accuracy:
   - quadratic tetrahedra, full 360° model
   - every load case at 6 angles across a symmetry sector
   - running stiffness and 20 000 rpm spin
   - buckling of the 3 worst cases
   - natural frequencies
4. **Render** `wheel.png` (outside, inside, cut-away, mid-width section).

Loads, as agreed: **radial press and diametral pinch only, 20 N, full width**. There are no edge, lip or axial
loads. Edit `"lcs"` in `runs.json` to change that: the available cases are `press`, `pinch`, `press_e0`,
`press_eW`, `pinch_e0`, `lip0` and `lipW`.

## Minimum wall: 0.6 mm by default (SLS PA12)

0.6 mm is the thinnest printable vertical SLS PA12 wall in the published guides:
- [Formlabs Fuse](https://formlabs.com/support/Design-specifications-for-3D-models-Fuse-1/): 0.6 mm
- [Stratasys](https://stratasys.com/uk/stratasysdirect/resources/articles/ls-design-tips-wall-thickness): 0.6 mm possible, 0.8 mm recommended
- [Murtfeldt](https://murtfeldt.de/en-GB/Additive-manufacturing/Design-tips/): > 0.6 mm
- service bureaus quote 0.7–1.0 mm for reliability ([Xometry](https://xometry.eu/en/?p=1593), [Protolabs](https://3dhubs.com/knowledge-base/designing-parts-sls-3d-printing), [Materialise](https://www.materialise.com/en/manufacturing/materials/pa-12-sls))

The rim is a vertical wall when the wheel prints lying flat. The fixed rim skin and the minimum wall on the
output geometry are therefore both 0.6 mm by default, with filter radius `rf` = 1.2 mm (about 2 × the
minimum). Raise `t_rim`/`t_min` in `runs.json` if your printer or bureau needs more.

**The rim thickens itself where needed.** Only the 0.6 mm skin is fixed. The 2.5 mm band inside it
(`rim_zone`, meshed at 0.15 mm) is design space, so the optimiser chooses the rim thickness at every angle
and every position across the width. In the reference result (`reference/rim_thickness.png`), 93 % of the
rim stays at the skin thickness. The other 7 % grows into 1–3 mm pads exactly where the spokes join.

## Loads: the wheel rolling on flat ground

The 20 N radial press and the 20 N diametral pinch are applied at every position the wheel can roll into.
Thanks to the N-fold symmetry, one sector covers all of them.

- **Optimiser:** a load every 5° across the sector (`n_phi`).
- **Final check:** a load every 1° (`verify_step_deg`), which is effectively continuous.

There is no uniform pressure around the wheel, because there's no tyre to create one. Spin (20 000 rpm)
is the only uniform load, and it is included. Masses and m_eq are for the **wheel body only; bearings are
excluded**.

## Running on GitHub Actions

Use the `.github/workflows/fine-runs.yml` workflow (Actions tab, "Run workflow"). A job stops at 320 minutes
and uploads its checkpoint. Dispatch the workflow again and it **resumes from the newest checkpoint**.

- `run_all.py` discards a checkpoint whose `runs.json` settings have changed.
- Small result files (summary, log, render, density field) are pushed to the `fine-run-results` branch.
- r06 needs about 20 GB and is not in the CI matrix.

The filter radius `rf` is 1.4 mm for the 0.7 mm-wall runs. With 1.0 mm on the 0.5 mm grid, the optimiser
made 0.5 mm features. The 0.7 mm minimum-wall rule then added about 26 % mass, and the result came out
over-strong at 0.40× the allowable. `extract` now reports `min_wall_added_g`; if it's large, raise `rf`.

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

| run | what it tests |
|---|---|
| r01_N6_skin0.7 | baseline: SLS PA12, 0.7 mm rim skin, 6-fold symmetry |
| r02_N5_skin0.7 | 5-fold symmetry (5/10/15 spokes possible) |
| r03_N4_skin0.7 | 4-fold symmetry |
| r04_N8_skin0.7 | 8-fold symmetry |
| r05_N6_skin0.4 | 0.4 mm skin and 0.4 mm min wall (a process that can print 0.4 mm walls) |
| r06_N6_skin0.7_h0.4 | mesh-refinement check of r01 (big) |

Symmetry order N is a balance requirement: N-fold symmetry makes the wheel statically balanced. Inside each
sector the shape is completely free.

## Reference result computed here (coarse, `reference/`): a valid organic wheel

r01's setup on a coarse optimisation grid (h 0.9 mm, loads every 10°). The optimiser grew **organic,
branching spokes** between the hub and the thin rim (`reference/wheel.png`). Printable mesh:
`reference/organic_wheel_r01.stl`. Full-accuracy check of that STL:

| m_eq (wheel only) | mass (wheel only) | max stress / allowable | buckling | f1 | min wall |
|---|---|---|---|---|---|
| **1.202 g** (parametric drum: 1.357 g) | 1.53 g | **0.72** (0.73 at h 1.0, 0.72 at h 0.8) | ≥ 6.1 | 1015 Hz | no ray < 0.70 mm (60k rays) |

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
| **1.075 g** (coarse reference 1.202 g, parametric drum 1.357 g) | 1.46 g | **0.80** (pinch@21) | 5.7 | 1080 Hz | 0.59 mm |

On the fine grid with N 5, the optimiser converged to a **single thin central web**: a near-axisymmetric
diaphragm with a thickened bearing seat. It did not grow discrete spokes. Under radial press and pinch,
a continuous web costs less inertia than spokes at this wall thickness.

The resin run (r07, Accura AMX, 0.3 mm) was at iteration 134 at its last checkpoint and is not yet final.
There, the optimiser formed a hollow box of two ~0.45 mm side walls (m_eq ≈ 1.06 g at the checkpoint).

### All fine runs so far (full check: P2 tets, rolling every 1°, wheel only)

| run | m_eq | mass | stress / allowable | buckling | f1 | min-wall rule added | status |
|---|---|---|---|---|---|---|---|
| r02 N5, 0.6 mm | **1.075 g** | 1.46 g | 0.80 | 5.7 | 1080 Hz | 0 g | **pass, best valid** |
| r03 N4, 0.6 mm | 1.249 g | 1.89 g | 0.49 | 16.5 | 1340 Hz | 0.27 g | pass, heavy |
| r04 N8, 0.6 mm | 1.247 g | 1.79 g | 0.63 | 13.5 | 1617 Hz | 0.23 g | pass, heavy (iteration cap) |
| r05 N6, 0.4 mm | 0.885 g | 1.25 g | **1.27** | 3.1 | 436 Hz | 0 g | FAIL: stress (pinch) |
| r01 N6, 0.6 mm | — | — | — | — | — | — | runner lost at iteration 104; resuming |
| r07 resin, 0.3 mm | ≈1.06 g (checkpoint) | ≈1.60 g | — | — | — | — | still optimising |

- **r03/r04:** the optimiser drew features thinner than 0.6 mm. The min-wall rule then added 0.23–0.27 g at extraction,
  which makes both wheels heavy and over-strong. Even before that rule ran, their m_eq (1.15 g) was above r02's,
  so they are not worth re-running.
- **r05 (0.4 mm wall) is the only fine run with clear organic spokes** (6 branching spokes between two side walls).
  It also has the lowest inertia. The optimiser under-read its peak stress by about 27 %: 0.4 mm members sit on a
  0.5 mm grid. **r05b** re-runs it with `sigma_target` 14 MPa instead of 18.9 MPa.

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
