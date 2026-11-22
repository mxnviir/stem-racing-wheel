"""SLS PA12 drum: parametric study of face taper and powder-escape holes (all load cases, both hole phases)."""
import json, sys, time
from pa12 import evaluate, drum_wheel
pts = []
for sup, ft, holes in [(0.806, 1.0, 0), (0.806, 1.0, 2), (0.70, 1.4, 2), (0.70, 1.7, 2), (0.75, 1.4, 2), (0.70, 2.0, 2)]:
    p = dict(rim=0.7, hub=1.0, sup=sup, ftaper=ft, holes_n=holes, holes_d=3.0, holes_r=8.5)
    t0 = time.time(); r = evaluate(drum_wheel(p))
    pts.append(dict(p=p, q=r["q"], gov=r["gov"], ok=r["ok"], m_eq=r["m_eq_g"], mass=r["mass_g"], stiff=r["stiff"]))
    print(f"face {sup:.3f} (hub x{ft:.1f}) holes {holes}: " + " ".join(f"{k} q{v:.2f} ({r['gov'][k]})" for k, v in sorted(r["q"].items()))
          + f" | m_eq {r['m_eq_g']:.3f} g mass {r['mass_g']:.2f} g {'PASS' if r['ok'] else 'FAIL'} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(pts, open("drum_study.json", "w"), indent=1, default=float)
