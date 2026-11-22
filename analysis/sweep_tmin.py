"""
Minimum-wall sensitivity: the ranking of concepts is an input (achievable minimum wall per process),
not a result. For each process group, sweep its minimum wall while the others stay at baseline (TMIN),
re-size every family that uses it, and plot m_eq vs t_min (crossover curves).

Writes sweep_tmin.json (resumable) and sweep_tmin.png.
"""
import sys, json, time
import numpy as np
from size import FAM, TMIN, size
from concepts import AL, TI, SLA, SLA10K, CF, PA12

GROUPS = {   # group: (materials moved together, sweep values for the first one, scale for the others)
    "Al (milled/turned)": ([AL, TI], [0.15, 0.25, 0.35, 0.50], [1.0, 1.2]),
    "CF tube":            ([CF], [0.15, 0.25, 0.35, 0.50], [1.0]),
    "SLA resin":          ([SLA, SLA10K], [0.30, 0.40, 0.60, 0.80], [1.0, 1.0]),
    "SLS PA12":           ([PA12], [0.50, 0.70, 1.00], [1.0]),
}


def families_using(mats):
    names = {m.name for m in mats}
    return [k for k, (b, roles, rmat) in FAM.items() if any(rmat[r].name in names for r in set(roles))]


if __name__ == "__main__":
    base = json.load(open("size_results.json"))
    try:
        out = json.load(open("sweep_tmin.json"))
    except Exception:
        out = {}
    for g, (mats, vals, scale) in GROUPS.items():
        for fam in families_using(mats):
            for v in vals:
                key = f"{g}|{fam}|{v:.2f}"
                if key in out:
                    continue
                tmin = {m.name: round(v * s, 3) for m, s in zip(mats, scale)}
                if all(abs(TMIN[m.name] - tmin[m.name]) < 1e-9 for m in mats):
                    r = base[fam]                     # baseline point already sized
                else:
                    t0 = time.time()
                    r = size(fam, tmin=tmin, t0=base[fam]["final"]["t"], verbose=False)
                    print(f"{g:20s} {fam:28s} t_min {v:.2f}: m_eq {r['res']['m_eq_g']:.3f} g "
                          f"{'PASS' if r['ok'] else 'FAIL'} t={r['final']['t']} [{time.time()-t0:.0f}s]", flush=True)
                out[key] = dict(group=g, fam=fam, tmin=v, ok=r["ok"], m_eq=r["res"]["m_eq_g"],
                                mass=r["res"]["mass_g"], t=r["final"]["t"])
                json.dump(out, open("sweep_tmin.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(4.2 * len(GROUPS), 4.2), sharey=True)
    for ax, g in zip(axs, GROUPS):
        pts = [p for p in out.values() if p["group"] == g]
        for fam in sorted({p["fam"] for p in pts}):
            P = sorted((p["tmin"], p["m_eq"], p["ok"]) for p in pts if p["fam"] == fam)
            x, y, ok = map(np.array, zip(*P))
            line, = ax.plot(x, y, "o-", ms=4, label=fam)
            if (~ok).any():
                ax.plot(x[~ok], y[~ok], "x", color=line.get_color(), ms=9)
        ax.axvline(TMIN[GROUPS[g][0][0].name], color="0.6", ls=":", lw=1)
        ax.set_title(g, fontsize=10); ax.set_xlabel("minimum wall t_min (mm)")
        ax.grid(alpha=0.3); ax.legend(fontsize=6.5, loc="upper left")
    axs[0].set_ylabel("m_eq = I/R² per rear wheel (g)")
    fig.suptitle("Sized rear wheel m_eq vs process minimum wall (others at baseline; dotted = baseline; x = not converged)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("sweep_tmin.png", dpi=150)
    print("wrote sweep_tmin.png")
