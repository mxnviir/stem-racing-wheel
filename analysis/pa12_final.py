"""
Final verification of the selected SLS PA12 wheel:
  1. mesh convergence (3 levels) of every stress/stiffness result, both spoke phases
  2. linear buckling of every handling case (symmetric and antisymmetric mode families: the half model
     with u_x = 0 only finds symmetric modes, so the antisymmetric BC u_y = u_z = 0 on x = 0 is solved too)
  3. natural frequencies (bore fixed, both families) vs rotation (333 Hz) and spoke passing (N x 333 Hz)
  4. STEP export of the full wheel, renders of geometry and stress
Usage: python pa12_final.py "<candidate name in pa12_explore.json>"
"""
import sys, json, time
import numpy as np
from geom3d import build
from fem3d import Model3D
import lc3d
from pa12 import BUILDERS, PA12_SLS, ALLOW, BEAR_M, BEAR_I, evaluate, phases

LEVELS = [(1.4, 0.6), (1.0, 0.42), (0.8, 0.32)]
F_ROT = 20000 / 60


def buckling(g, h, h_load, log):
    P, T, _ = build(g, h=h, h_load=h_load)
    m = Model3D(P, T, PA12_SLS)
    K = m.stiffness()
    r = lc3d.run(g, m, K=K, keep_u=True)
    U, S = r["_U"], r["_S"]
    # antisymmetric pinch restraint: u_y = u_z = 0 on x = 0 leaves x-translation and rotations about y, z
    Dx0 = m.dofs_where(lambda x: np.abs(x[0]) < 1e-7, [1, 2])
    pts = [np.array([[0.0], [g.r_bore], [g.seats[0][0]]]), np.array([[0.0], [-g.r_bore], [g.seats[0][0]]]),
           np.array([[0.0], [g.r_bore], [g.seats[-1][1]]])]
    nodes = [int(np.argmin(np.sum((m.X - p)**2, 0))) for p in pts]
    Dpinch_anti = [Dx0, np.array([3 * n for n in nodes])]
    out = {}
    for lc in ("LC2", "LC3", "LC4", "LC6", "LC7", "LC8"):
        if lc not in U:
            continue
        KG = m.geometric_stiffness(U[lc])
        bcs = ((S["Dbore"], S["Dbore_anti"]) if lc not in ("LC3", "LC7") else (S["Dpinch"], Dpinch_anti))
        lam = []
        for D, fam in zip(bcs, ("sym", "anti")):
            t0 = time.time()
            l, _ = m.eigs(K, KG, D, k=4, mode="buckling")
            lam.append(float(l[0]) if len(l) else float("inf"))
            log(f"    {lc} {fam:4s}: lambda_1 = {lam[-1]:.2f}  ({time.time()-t0:.0f}s)")
        out[lc] = min(lam)
    Mm = m.mass_matrix()
    S0 = lc3d.setup(g, m)
    f_s, _ = m.eigs(K, Mm, S0["Dbore"], k=4, mode="modes")
    f_a, _ = m.eigs(K, Mm, S0["Dbore_anti"], k=4, mode="modes")
    out["modes_Hz"] = sorted(np.concatenate([f_s, f_a]).tolist())
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    cand = json.load(open("pa12_explore.json"))[name]
    g = BUILDERS[cand["kind"]](cand["p"])
    rep = {"name": name, "p": cand["p"]}
    fh = open("pa12_final.txt", "w")
    def log(s):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()
    log(f"Final verification: {name}\nparams {cand['p']}\nallowable (max of s1, vM) = {ALLOW:.1f} MPa")
    log("1. mesh convergence (q = max(s1, vM) / allowable, worst over load cases and spoke phases)")
    conv = []
    for h, hl in LEVELS:
        t0 = time.time()
        r = evaluate(g, h, hl)
        nd = max(rr["ndofs"] for rr in r["phases"].values())
        conv.append(dict(h=h, hl=hl, ndofs=nd, q=r["q"], gov=r["gov"], stiff=r["stiff"], m_eq=r["m_eq_g"], mass=r["mass_g"]))
        log(f"  h {h:.2f}/{hl:.2f} ({nd} dof): " + " ".join(f"{k} {v:.3f} ({r['gov'][k]})" for k, v in sorted(r["q"].items()))
            + f" | m_eq {r['m_eq_g']:.4f} g | " + " ".join(f"{k} {v:.2f}" for k, v in r["stiff"].items()) + f" [{time.time()-t0:.0f}s]")
    rep["convergence"] = conv
    log("2./3. buckling (load factor on the 20 N handling loads) and natural frequencies, middle mesh level")
    for dph in phases(g):
        log(f"  phase +{dph:.0f}")
        rep[f"buckling_{dph:.0f}"] = buckling(g.rotated(dph), *LEVELS[1], log)
        fm = rep[f"buckling_{dph:.0f}"]["modes_Hz"]
        log(f"    modes Hz: {', '.join(f'{f:.0f}' for f in fm)}  (rotation {F_ROT:.0f} Hz)")
    build(g, h=2.0, half=False, step_file="pa12_final.step")
    json.dump(rep, open("pa12_final.json", "w"), indent=1, default=float)
    log("wrote pa12_final.json, pa12_final.step")
