"""
Run the fine topology-optimisation studies end to end:
  optimise (min inertia, stress-constrained)  ->  extract STL (min wall enforced, powder holes if the
  optimiser closed a cavity)  ->  full-accuracy check (P2 tets, all load angles, buckling, modes)  ->  renders

Every stage is skipped if its output already exists, and the optimiser resumes from its last save, so
the script can be stopped and restarted at any time.

    python run_all.py                 # all runs in runs.json, in order
    python run_all.py r01 r03         # only runs whose name starts with these
    python run_all.py --smoke         # 2-minute install check on a very coarse grid
    python run_all.py --no-verify     # optimise + extract only
    python run_all.py r11 --re-extract   # redo extract + verify of a finished optimisation
"""
import sys, os, json, time, traceback, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import topo, extract                                      # noqa: E402

RESULTS = os.path.join(HERE, "results")


def log_to(path):
    fh = open(path, "a")
    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); fh.write(line + "\n"); fh.flush()
    return log


def do_extract(npz, stl, t_min, voxel, log):
    s, r, z, nt, rho = extract.load(npz)
    phi0, axes = extract.field(s, r, z, nt, rho, voxel)
    phi = extract.enforce_min_thickness(phi0, axes, s, voxel, t_min) if t_min > 0 else phi0
    added = float(((phi > 0) & ~(phi0 > 0)).sum() * voxel**3 * s["rho"] * 1e-6)   # g added by the min-wall rule
    log(f"min-wall rule added {added:.3f} g ({'OK' if added < 0.05 * max(1e-9, float((phi0 > 0).sum() * voxel**3 * s['rho'] * 1e-6)) else 'LARGE: optimiser features thinner than t_min, raise rf'})")
    phi, n_small, v_small = extract.fill_small_voids(phi, axes, s)
    if n_small:
        log(f"filled {n_small} tiny enclosed voids ({v_small:.2f} mm^3 in total)")
    tol = min(0.08, 0.5 * voxel)                         # snap tolerance below half a voxel (no folds)
    m = extract.snap(extract.surface(phi, axes), s, tol=tol)
    holes = None
    if m.metadata.get("n_cavities", 0) > 0:             # enclosed cavity: SLS powder cannot get out
        phi, holes = extract.cavity_holes(phi, axes, s)
        m = extract.snap(extract.surface(phi, axes), s, tol=tol)
        log(f"closed cavity found -> powder holes (2 x dia 3.0 from the inner face) {holes}; cavities now "
            f"{m.metadata.get('n_cavities', 0)} ({'open' if m.metadata.get('n_cavities', 0) == 0 else 'STILL CLOSED - check by hand'})")
    m.export(stl)
    t, _ = extract.ray_thickness(m, n=20000)
    t = t[np.isfinite(t)]
    if m.metadata.get("n_loose", 0):
        log(f"WARNING: {m.metadata['n_loose']} loose solid bodies not attached to the wheel (design not converged?)")
    info = dict(triangles=len(m.faces), shells=m.metadata.get("n_shells", 1), cavities=m.metadata.get("n_cavities", 0),
                loose_bodies=m.metadata.get("n_loose", 0), watertight=bool(m.is_watertight),
                volume_mm3=float(m.volume), mass_g=float(m.volume * s["rho"] * 1e-6), min_wall_added_g=added,
                wall_p1_mm=float(np.percentile(t, 1)), wall_median_mm=float(np.median(t)), holes=holes)
    log(f"extract: {info}")
    return info


def render(stl, png, title):
    import trimesh, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import colors
    m = trimesh.load(stl)
    def show(ax, mm, elev, azim, t):
        T = mm.vertices[mm.faces][:, :, [0, 2, 1]]
        n = mm.face_normals[:, [0, 2, 1]]
        L = np.array([0.35, -0.75, 0.55]); L /= np.linalg.norm(L)
        k = 0.35 + 0.65 * np.clip(np.abs(n @ L), 0, 1)
        ax.add_collection3d(Poly3DCollection(T, facecolors=np.clip(np.outer(k, colors.to_rgb("#d9cdb5")), 0, 1), edgecolors="none"))
        R = 14.5; ax.set_xlim(-R, R); ax.set_ylim(8.6 - R, 8.6 + R); ax.set_zlim(-R, R); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(t, fontsize=9)
    f = plt.figure(figsize=(16, 5))
    show(f.add_subplot(1, 4, 1, projection="3d"), m, 20, -60, "outer side")
    show(f.add_subplot(1, 4, 2, projection="3d"), m, 20, 120, "inner side")
    show(f.add_subplot(1, 4, 3, projection="3d"), m.slice_plane([0, 0, 0], [-1, 0, 0]), 20, -30, "half cut-away")
    show(f.add_subplot(1, 4, 4, projection="3d"), m.slice_plane([0, 0, 8.6], [0, 0, 1]), 80, -90, "mid-width section")
    f.suptitle(title, fontsize=10); f.tight_layout(); f.savefig(png, dpi=120); plt.close(f)


def run_one(spec, smoke=False, verify=True, re_extract=False):
    name = spec["name"] + ("_smoke" if smoke else "")
    d = os.path.join(RESULTS, name); os.makedirs(d, exist_ok=True)
    log = log_to(os.path.join(d, "log.txt"))
    st = dict(spec["setup"])
    if smoke:
        st.update(h=1.2, h_rim=0.4, rf=1.5, iters=40)
    npz, stl = os.path.join(d, "density.npz"), os.path.join(d, "wheel.stl")
    summary_f = os.path.join(d, "summary.json")
    summary = json.load(open(summary_f)) if os.path.exists(summary_f) else dict(name=name, spec=spec)
    if summary.get("spec", spec) != spec and not smoke:      # settings changed: never resume a stale checkpoint
        import shutil
        old_dir = d + "_stale_" + time.strftime("%Y%m%d%H%M%S")
        shutil.move(d, old_dir); os.makedirs(d, exist_ok=True)
        print(f"settings of {name} changed since the saved checkpoint -> old results moved to {old_dir}, starting fresh")
        summary = dict(name=name, spec=spec)
    summary["spec"] = spec
    if re_extract and "optimise" in summary:             # redo extract + verify + render from the saved design
        for k in ("extract", "verify"):
            summary.pop(k, None)
        for fn in ("wheel.stl", "wheel.png"):
            if os.path.exists(os.path.join(d, fn)):
                os.remove(os.path.join(d, fn))
    log(f"=== {name}: {spec.get('note', '')} | {platform.platform()}, {os.cpu_count()} cpus")
    # 1. optimise (resumes automatically from density.npz)
    if spec.get("seed") and not os.path.exists(npz) and "optimise" not in summary:
        import shutil                                       # start from a committed checkpoint of an earlier run
        shutil.copy(os.path.join(HERE, spec["seed"]), npz)
        log(f"seeded from {spec['seed']}")
    if "optimise" not in summary:
        t0 = time.time()
        init = os.path.join(HERE, spec["init_from"]) if spec.get("init_from") and not smoke else None
        w, x, pb, hist = topo.optimise_inertia(topo.Setup(**st), npz, log=log, sigma_target=spec.get("sigma_target"),
                                               init_from=init, beta0=spec.get("beta0", 1.0) if init else 1.0)
        converged = len(hist) < st["iters"]                 # topo stops early only when converged
        summary["optimise"] = dict(iterations=len(hist), converged=converged, final=hist[-1],
                                   minutes=(time.time() - t0) / 60)
        if not converged:
            log(f"WARNING: iteration cap {st['iters']} reached without convergence")
        json.dump(summary, open(summary_f, "w"), indent=1, default=float)
    # 2. extract the geometry
    if "extract" not in summary:
        summary["extract"] = do_extract(npz, stl, spec.get("t_min", 0.6), spec.get("voxel", 0.15) if not smoke else 0.3, log)
        json.dump(summary, open(summary_f, "w"), indent=1, default=float)
    # 3. full-accuracy verification of the real geometry
    if summary["extract"].get("loose_bodies", 0):
        log("design has loose (unattached) bodies -> not a valid wheel; verification skipped. "
            "Run longer / check the density field.")
        summary["verify"] = dict(invalid="loose bodies")
    if verify and "verify" not in summary:
        import verify_organic
        s = json.loads(str(np.load(npz, allow_pickle=True)["setup"]))
        h = spec.get("verify_h", 0.6) if not smoke else 1.2
        n_ang = max(2, int(round(360.0 / s["N"] / spec.get("verify_step_deg", 1.0))))   # rolling sweep
        log(f"verification: load every {360.0 / s['N'] / n_ang:.2f} deg across the sector ({n_ang} positions)")
        try:
            vres = verify_organic.run(stl, s, h=h, h_min=min(0.25, h / 2), n_ang=n_ang if not smoke else 2,
                                       buckle=3 if not smoke else 0, log=log, modes=not smoke)
        except Exception as e:                      # record it instead of failing silently
            import traceback
            log("verification FAILED: " + repr(e) + "\n" + traceback.format_exc())
            raise
        summary["verify"] = vres
        json.dump(summary, open(summary_f, "w"), indent=1, default=float)
    # 4. render
    png = os.path.join(d, "wheel.png")
    if not os.path.exists(png):
        v = summary.get("verify", {}) if "invalid" not in summary.get("verify", {}) else {}
        title = (f"{name}: m_eq {v.get('m_eq_g', float('nan')):.3f} g (wheel only), mass {v.get('mass_g', float('nan')):.2f} g, "
                 f"max stress / allowable {v.get('q_max', float('nan')):.2f} ({v.get('governing', '-')})")
        render(stl, png, title)
    log(f"=== {name} done")
    return summary


def write_table():
    rows = []
    for n in sorted(os.listdir(RESULTS)):
        f = os.path.join(RESULTS, n, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f)); v = s.get("verify", {}); e = s.get("extract", {}); o = s.get("optimise", {})
        b = min([x[0] for x in v.get("buckling", [])], default=float("nan"))
        rows.append([n, o.get("iterations", ""), f"{o.get('minutes', float('nan')):.0f}",
                     f"{v.get('m_eq_g', float('nan')):.3f}", f"{v.get('mass_g', float('nan')):.3f}",
                     f"{v.get('q_max', float('nan')):.2f}", v.get("governing", ""), f"{b:.1f}",
                     f"{(v.get('modes_Hz') or [float('nan')])[0]:.0f}", f"{e.get('wall_p1_mm', float('nan')):.2f}",
                     e.get("shells", ""), e.get("holes", "")])
    hdr = ["run", "iters", "opt_min", "m_eq_g", "mass_g", "stress_ratio", "governing", "buckling_min", "f1_Hz",
           "wall_p1_mm", "shells", "holes"]
    with open(os.path.join(RESULTS, "summary.csv"), "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for r in rows:
            fh.write(",".join(str(c).replace(",", ";") for c in r) + "\n")
    print(open(os.path.join(RESULTS, "summary.csv")).read())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke, verify = "--smoke" in sys.argv, "--no-verify" not in sys.argv
    re_extract = "--re-extract" in sys.argv
    if "--reference" in sys.argv:          # install check of the verification stage on a shipped STL
        import shutil
        ref = os.path.join(HERE, "reference")
        d = os.path.join(RESULTS, "reference_check")
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        shutil.copy(os.path.join(ref, "density.npz"), os.path.join(d, "density.npz"))
        # verify the shipped STL itself (the geometry the reference numbers belong to); re-extracting the density
        # field with today's extract.py gives a different, lighter wheel, so it would not test the install
        shutil.copy(os.path.join(ref, "organic_wheel_r01.stl"), os.path.join(d, "wheel.stl"))
        rs = json.load(open(os.path.join(ref, "summary.json")))
        spec = dict(rs["spec"], name="reference_check", verify_h=1.2)
        json.dump(dict(name="reference_check", spec=spec, optimise=rs["optimise"], extract=rs["extract"]),
                  open(os.path.join(d, "summary.json"), "w"))
        out = run_one(spec)
        a, b = out.get("verify", {}), rs.get("verify", {})
        print(f"reference (h {rs['spec'].get('verify_h')}): m_eq {b.get('m_eq_g', float('nan')):.3f} g, stress ratio "
              f"{b.get('q_max', float('nan')):.2f} | your machine (h 1.2): m_eq {a.get('m_eq_g', float('nan')):.3f} g, "
              f"stress ratio {a.get('q_max', float('nan')):.2f}  (expect m_eq within ~1 %, stress within ~10 %)")
        sys.exit(0)
    runs = json.load(open(os.path.join(HERE, "runs.json")))["runs"]
    if smoke:
        runs = runs[:1]
    failed = []
    for spec in runs:
        if args and not any(spec["name"].startswith(a) for a in args):
            continue
        try:
            run_one(spec, smoke=smoke, verify=verify, re_extract=re_extract)
        except Exception:
            traceback.print_exc()
            failed.append(spec["name"])
            print(f"!!! {spec['name']} failed; continuing with the next run", flush=True)
    write_table()
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")
