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

## Rim thickness vs process — read this

The rim skin can't be thinner than the process's minimum printable wall. Extraction forces every wall,
including the skin, to at least `t_min`. A full-width rim at R = 14.1 has m_eq of:

| rim skin | rim m_eq on its own |
|---|---|
| 0.7 mm (SLS PA12) | 0.92 g, the floor for r01–r04 |
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
`wheel.stl` and `density.npz`. Commit it on a branch, or zip it. The STLs can be large (50–80 MB), so zip
them or leave them out if they're too big for git.

## Reading the results

`results/summary.csv` columns:

| column | meaning |
|---|---|
| `m_eq_g` | rotational inertia as equivalent mass, **including the 2 bearings** (lower = faster car) |
| `stress_ratio` | max(σ₁, σ_vM) / allowable over all load cases and angles; must be ≤ 1 |
| `buckling_min` | lowest buckling load factor of the checked cases; must be ≥ 2 |
| `f1_Hz` | first natural frequency; should be ≫ 333 Hz (20 000 rpm) |
| `wall_p1_mm` | 1st-percentile wall thickness of the output; should be ≥ t_min |
| `holes` | powder holes added because the optimiser closed a cavity |

If `stress_ratio` comes out above 1, the coarse optimisation grid under-read a stress peak. Re-run that case
with a lower `sigma_target`: add `"sigma_target": 16.0` to its entry, next to `"setup"`.
