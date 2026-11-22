"""
SLS PA12 rear wheel: evaluation of 3D concepts over all load cases and spoke phases, fully-stressed
sizing, and topology exploration.

Material: SLS PA12 (EOS PA2200-class laser-sintered): E 1700 MPa, nu 0.40, part density 930 kg/m^3,
UTS 48 MPa (XY) / 42 MPa (Z, build direction). Conservative: the Z value is used everywhere.
Criterion (team decision): SF >= 2 on max principal stress vs UTS. Also checked: von Mises <= UTS / 2
(covers compressive/shear-dominated points under contact patches).
Minimum printable wall / spoke dimension: 0.7 mm (assumption; typical SLS guidance 0.7-1.0 mm).
"""
import sys, json, time, dataclasses
import numpy as np
from wheelfem import Material
from geom3d import Wheel, SpokeSet, build
from fem3d import Model3D
import lc3d

PA12_SLS = Material("SLS PA12 (PA2200)", 1700, 0.40, 930, 42)
SF = 2.0
ALLOW = PA12_SLS.strength / SF
TMIN = 0.7
BEAR_M, BEAR_I = 2 * 0.32, 2 * 0.18 * 2.7**2     # two MR63 (as in concepts.py)


def phases(g):
    Ns = [s.N for s in g.spokes if s.N > 0]
    return [0.0] if not Ns else [0.0, 180.0 / Ns[0]]


def evaluate(g, h=1.4, h_load=0.6, verbose=False):
    """Worst case over load cases and spoke phases. q = max(s1, vM) / allowable per role."""
    res = {"phases": {}}
    for dph in phases(g):
        gg = g.rotated(dph)
        t0 = time.time()
        for k in range(4):                 # mesher failures are rare and size-dependent: perturb and retry
            try:
                P, T, info = build(gg, h=h * (1 - 0.07 * k), h_load=h_load * (1 - 0.05 * k))
                break
            except Exception as e:
                if k == 3:
                    raise
                print(f"    mesh retry {k + 1}: {e}", flush=True)
        m = Model3D(P, T, PA12_SLS, extra_mass_g=BEAR_M, extra_Izz=BEAR_I)
        r = lc3d.run(gg, m)
        r["mesh"] = dict(nodes=P.shape[1], tets=T.shape[1], **info)
        r["time_s"] = time.time() - t0
        res["phases"][f"{dph:.1f}"] = r
        if verbose:
            print(f"    phase +{dph:.0f}: {m.ndofs} dof, fillets {info}, {r['time_s']:.0f}s", flush=True)
    q, gov = {}, {}
    for ph, r in res["phases"].items():
        for lc in lc3d.LCS:
            for role, s in r.get(lc, {}).items():
                v = max(s["s1"], s["vm"]) / ALLOW
                if v > q.get(role, -1):
                    q[role], gov[role] = v, f"{lc}@+{ph}"
    r0 = next(iter(res["phases"].values()))
    res.update(q=q, gov=gov, mass_g=r0["mass_g"], m_eq_g=r0["m_eq_g"],
               ok=all(v <= 1.0 for v in q.values()),
               stiff={k: max(r[k] for r in res["phases"].values() if k in r) for k in
                      ("LC1_um_per_N", "LC1_edge_um_per_N", "LC3_dD_um", "LC4_um", "LC5_dD_um",
                       "LC6_edge_um", "LC7_dD_edge_um", "LC8_lip_um") if any(k in r for r in res["phases"].values())})
    return res


# ---------------- concept builders (params dict -> Wheel) ----------------
R, W, RB = 14.1, 17.2, 3.0


def web_wheel(p):
    return Wheel(name="web", rim_t=[(0, p["rim"]), (W, p["rim"])], t_hub=p["hub"], hub_z=(W / 2 - 3, W / 2 + 3),
                 seats=[(W / 2 - 2.5, W / 2 + 2.5)], lip_h=p.get("lip_h", 0.0), lip_t=p.get("lip_t", 0.8),
                 spokes=[SpokeSet(N=0, zc=W / 2, ts_hub=p["sup"] * p.get("taper", 1.0), ts_rim=p["sup"])],
                 rf=p.get("rf", 0.5))


def spoke_wheel(p):
    N = p["N"]
    return Wheel(name=f"spokes{N}", rim_t=[(0, p["rim"]), (W, p["rim"])], t_hub=p["hub"], hub_z=(W / 2 - 3, W / 2 + 3),
                 seats=[(W / 2 - 2.5, W / 2 + 2.5)], lip_h=p.get("lip_h", 0.0), lip_t=p.get("lip_t", 0.8),
                 spokes=[SpokeSet(N=N, zc=W / 2, ts_hub=p["ts"] * p.get("taper", 1.0), ts_rim=p["ts"],
                                  w_hub=p["w"] * p.get("wtaper", 1.0), w_rim=p["w"])],
                 pad_t=p.get("pad_t", 0.0), pad_deg=p.get("pad_deg", 15.0), rf=p.get("rf", 0.5))


def drum_wheel(p):
    return Wheel(name="drum", rim_t=[(0, p["rim"]), (W, p["rim"])], t_hub=p["hub"], hub_z=(0, W),
                 seats=[(0, 2.5), (W - 2.5, W)], spokes=[], drum_t=p["sup"], rf=p.get("rf", 0.5))


def twin_spoke_wheel(p):
    """Two spoke planes near the rim edges (support both edges), hub over the full width."""
    N, zo = p["N"], p.get("zoff", 3.0)
    s = lambda zc, ph: SpokeSet(N=N, zc=zc, ts_hub=p["ts"] * p.get("taper", 1.0), ts_rim=p["ts"],
                                w_hub=p["w"] * p.get("wtaper", 1.0), w_rim=p["w"], phase_deg=ph)
    stag = 180.0 / N if p.get("stagger", False) else 0.0
    return Wheel(name=f"twin{N}", rim_t=[(0, p["rim"]), (W, p["rim"])], t_hub=p["hub"], hub_z=(0, W),
                 seats=[(0, 2.5), (W - 2.5, W)], spokes=[s(zo, 90.0), s(W - zo, 90.0 + stag)],
                 lip_h=p.get("lip_h", 0.0), lip_t=p.get("lip_t", 0.8), rf=p.get("rf", 0.5))


BUILDERS = {"web": web_wheel, "spokes": spoke_wheel, "drum": drum_wheel, "twin": twin_spoke_wheel}

# which sizing parameter each stress role drives (radial-load roles drive spoke width, axial ones thickness)
ROLE_PARAMS = {
    "web": {"rim": ["rim"], "support": ["sup"], "hub": ["hub"]},
    "drum": {"rim": ["rim"], "support": ["sup"], "hub": ["hub"]},
    "spokes": {"rim": ["rim"], "support": ["ts", "w"], "hub": ["hub"]},
    "twin": {"rim": ["rim"], "support": ["ts", "w"], "hub": ["hub"]},
}
AXIAL_LCS = ("LC4", "LC8")


def size(kind, p0, iters=8, h=1.4, h_load=0.6, log=print):
    """Fully-stressed sizing: each parameter scaled by sqrt(q) of its role (bending: sigma ~ 1/t^2).
    For spokes, an axially governed role scales the axial thickness ts, a radially governed one the
    width w (in-plane bending of the spoke). Stops when all q <= 1 and each q >= 0.85 or at minimum."""
    p = dict(p0); hist = []
    for it in range(iters):
        g = BUILDERS[kind](p)
        t0 = time.time()
        r = evaluate(g, h, h_load)
        hist.append(dict(p=dict(p), q=r["q"], gov=r["gov"], m_eq=r["m_eq_g"], mass=r["mass_g"], ok=r["ok"]))
        log(f"  it{it} " + " ".join(f"{k}={v:.3f}" for k, v in p.items() if isinstance(v, float)) +
            " | " + " ".join(f"{k} q{r['q'][k]:.2f} ({r['gov'][k]})" for k in sorted(r["q"])) +
            f" | m_eq {r['m_eq_g']:.3f} g mass {r['mass_g']:.2f} g [{time.time()-t0:.0f}s]")
        params = {k: list(v) for k, v in ROLE_PARAMS[kind].items()}
        if p.get("lip_h", 0) > 0:
            params["rim"] += ["lip_t", "lip_h"]
        tight = all(r["q"][ro] >= 0.85 or all(p[k] <= TMIN + 1e-9 for k in params[ro]) for ro in r["q"] if ro in params)
        if r["ok"] and tight:
            break
        for ro, q in r["q"].items():
            if ro not in params:
                continue
            ks = params[ro]
            if ro == "support" and len(ks) == 2:      # spokes: choose by governing load case
                ks = [ks[0]] if r["gov"][ro].split("@")[0] in AXIAL_LCS else [ks[1]]
            f = np.clip(np.sqrt(q) * (1.03 if q > 1 else 1.0), 0.65, 1.5)
            for k in ks:
                p[k] = round(max(TMIN, p[k] * f), 3)
    return dict(kind=kind, p=hist[-1]["p"], final=r, hist=hist)


CANDIDATES = [
    ("W  central web",              "web",    dict(rim=0.9, hub=0.9, sup=1.4, taper=1.3)),
    ("WL central web + lips",       "web",    dict(rim=0.8, hub=0.9, sup=1.4, taper=1.3, lip_h=1.0, lip_t=0.8)),
    ("S6L 6 spokes + lips",         "spokes", dict(N=6, rim=0.8, hub=0.9, ts=1.6, w=2.0, taper=1.3, wtaper=1.3, lip_h=1.0, lip_t=0.8)),
    ("S6LP 6 spokes + lips + pads", "spokes", dict(N=6, rim=0.8, hub=0.9, ts=1.6, w=2.0, taper=1.3, wtaper=1.3, lip_h=1.0, lip_t=0.8,
                                                   pad_t=0.6, pad_deg=12.0)),
    ("T6 twin 6-spoke planes",      "twin",   dict(N=6, rim=0.8, hub=0.9, ts=1.2, w=1.6, taper=1.2, wtaper=1.3, zoff=2.0)),
    ("D  closed drum",              "drum",   dict(rim=0.8, hub=0.9, sup=0.9)),
]


if __name__ == "__main__":
    out_f = "pa12_explore.json"
    try:
        out = json.load(open(out_f))
    except Exception:
        out = {}
    sel = sys.argv[1:]
    for name, kind, p0 in CANDIDATES:
        if (sel and not any(name.startswith(x) for x in sel)) or name in out:
            continue
        print(f"== {name}", flush=True)
        r = size(kind, p0, log=lambda s: print(s, flush=True))
        f = r["final"]
        out[name] = dict(kind=kind, p=r["p"], ok=f["ok"], q=f["q"], gov=f["gov"], m_eq=f["m_eq_g"], mass=f["mass_g"],
                         stiff=f["stiff"], hist=r["hist"])
        print(f"=> {name}: {'PASS' if f['ok'] else 'FAIL'} m_eq {f['m_eq_g']:.3f} g mass {f['mass_g']:.2f} g p={r['p']}", flush=True)
        json.dump(out, open(out_f, "w"), indent=1, default=float)
