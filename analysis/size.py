"""
Fully-stressed-design sizing of each concept family.
Each role (rim / face / web / hub) thickness is scaled by sqrt(stress ratio) (bending-dominated:
sigma ~ 1/t^2) until every role is at its allowable or at its process minimum gauge.
Allowable = strength / SF_req (metals: von Mises vs yield, SF 1.5; polymers: max principal vs UTS,
SF 2.0; CF: von Mises vs conservative strength, SF 2.0).  Load cases as in concepts.py.
"""
import sys, json, time
import numpy as np
from concepts import run, drum, web, AL, TI, SLA, SLA10K, CF, PA12

TMIN = {AL.name: 0.25, TI.name: 0.30, SLA.name: 0.40, SLA10K.name: 0.40, CF.name: 0.25, PA12.name: 0.70}

def brittle(name): return ("SLA" in name) or ("PA12" in name)
def need(name): return 2.0 if (brittle(name) or "CF" in name) else 1.5

# family: (builder(t: dict) -> Design, roles per block index, material per role)
FAM = {
 "A Al drum":            (lambda t: drum("A", t["rim"], t["face"], t["hub"], 0, 0, 0, [AL]),
                          ["rim", "face", "face", "hub"], {"rim": AL, "face": AL, "hub": AL}),
 "A Al web":             (lambda t: web("Aw", t["rim"], t["web"], t["hub"], 0, 0, 0, [AL]),
                          ["rim", "web", "hub"], {"rim": AL, "web": AL, "hub": AL}),
 "T Ti drum":            (lambda t: drum("T", t["rim"], t["face"], t["hub"], 0, 0, 0, [TI]),
                          ["rim", "face", "face", "hub"], {"rim": TI, "face": TI, "hub": TI}),
 "S SLA drum":           (lambda t: drum("S", t["rim"], t["face"], t["hub"], 0, 0, 0, [SLA]),
                          ["rim", "face", "face", "hub"], {"rim": SLA, "face": SLA, "hub": SLA}),
 "S SLA web":            (lambda t: web("Sw", t["rim"], t["web"], t["hub"], 0, 0, 0, [SLA]),
                          ["rim", "web", "hub"], {"rim": SLA, "web": SLA, "hub": SLA}),
 "G SLA-10K drum":       (lambda t: drum("G", t["rim"], t["face"], t["hub"], 0, 0, 0, [SLA10K]),
                          ["rim", "face", "face", "hub"], {"rim": SLA10K, "face": SLA10K, "hub": SLA10K}),
 "P PA12 drum":          (lambda t: drum("P", t["rim"], t["face"], t["hub"], 0, 0, 0, [PA12]),
                          ["rim", "face", "face", "hub"], {"rim": PA12, "face": PA12, "hub": PA12}),
 "H1 CF rim/SLA faces/Al hub": (lambda t: drum("H1", t["rim"], t["face"], t["hub"], 0, 1, 2, [CF, SLA, AL]),
                          ["rim", "face", "face", "hub"], {"rim": CF, "face": SLA, "hub": AL}),
 "H3 CF rim/Al faces+hub": (lambda t: drum("H3", t["rim"], t["face"], t["hub"], 0, 1, 1, [CF, AL]),
                          ["rim", "face", "face", "hub"], {"rim": CF, "face": AL, "hub": AL}),
 "H4 CF rim/Al web+hub": (lambda t: web("H4", t["rim"], t["web"], t["hub"], 0, 1, 1, [CF, AL]),
                          ["rim", "web", "hub"], {"rim": CF, "web": AL, "hub": AL}),
 "H5 CF rim/SLA web/Al hub": (lambda t: web("H5", t["rim"], t["web"], t["hub"], 0, 1, 2, [CF, SLA, AL]),
                          ["rim", "web", "hub"], {"rim": CF, "web": SLA, "hub": AL}),
}

def ratios(res, roles):
    r = {}
    for lc in ("LC2", "LC3", "LC4", "LC5"):
        for ib, s in res[lc].items():
            meas = s["s1"] if brittle(s["mat"]) else s["vm"]
            q = meas / (s["strength"] / need(s["mat"]))
            role = roles[int(ib)]
            if q > r.get(role, (0,))[0]:
                r[role] = (q, lc)
    return r

def size(fam, iters=7):
    build, roles, rmat = FAM[fam]
    t = {ro: max(0.5, TMIN[rmat[ro].name]) for ro in set(roles)}
    hist = []
    for it in range(iters):
        res = run(build(t), by="block")
        q = ratios(res, roles)
        hist.append(dict(t=dict(t), m_eq=res["m_eq_g"], mass=res["mass_g"], q={k: v[0] for k, v in q.items()}))
        ok = all(v[0] <= 1.0 for v in q.values())
        tight = all(v[0] >= 0.85 or t[k] <= TMIN[rmat[k].name] + 1e-9 for k, v in q.items())
        print(f"  it{it} " + " ".join(f"{k}={t[k]:.3f}(q{q[k][0]:.2f},{q[k][1]})" for k in sorted(t))
              + f"  m_eq={res['m_eq_g']:.3f} g", flush=True)
        if ok and tight:
            break
        for k, (qq, _) in q.items():
            f = np.clip(np.sqrt(qq) * (1.03 if qq > 1 else 1.0), 0.6, 1.6)
            t[k] = round(max(TMIN[rmat[k].name], t[k] * f), 3)
    return dict(final=hist[-1], ok=ok, res={k: res[k] for k in ("mass_g", "m_eq_g", "LC1_um_per_N", "LC3_dD_um", "LC4_face_um", "LC5_dD_um")}, hist=hist)

if __name__ == "__main__":
    keys = sys.argv[1:] or list(FAM)
    out = {}
    try:
        out = json.load(open("size_results.json"))
    except Exception:
        pass
    for k in keys:
        print(k, flush=True)
        t0 = time.time()
        out[k] = size(k)
        r = out[k]
        print(f"=> {k}: {'PASS' if r['ok'] else 'FAIL'} t={r['final']['t']} mass {r['res']['mass_g']:.2f} g m_eq {r['res']['m_eq_g']:.3f} g "
              f"run {r['res']['LC1_um_per_N']:.2f} um/N [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open("size_results.json", "w"), indent=1)
