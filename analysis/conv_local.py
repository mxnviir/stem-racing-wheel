"""Local refinement study of the governing peak of the final PA12 drum (face root under the thumb, LC4):
global h fixed, the thumb/contact refinement zones refined to h_load -> 0.2 mm (both hole phases)."""
import json, time
import numpy as np
from pa12 import drum_wheel, PA12_SLS, BEAR_M, BEAR_I, ALLOW, phases
from geom3d import build
from fem3d import Model3D
import lc3d

p = json.load(open("pa12_explore.json"))["DF2 drum, faces 0.70->1.05 + powder holes"]["p"]
g = drum_wheel(p)
out = []
fh = open("conv_local.txt", "w")
for hl in (0.42, 0.32, 0.25, 0.2):
    for dph in phases(g):
        gg = g.rotated(dph); t0 = time.time()
        P, T, _ = build(gg, h=1.0, h_load=hl)
        m = Model3D(P, T, PA12_SLS, extra_mass_g=BEAR_M, extra_Izz=BEAR_I)
        r = lc3d.run(gg, m)
        row = dict(h_load=hl, phase=dph, ndofs=m.ndofs)
        for lc in lc3d.LCS:
            for ro, s in r[lc].items():
                q = max(s["s1"], s["vm"]) / ALLOW
                if q > row.get(ro, (0,))[0]:
                    X = np.array(s["at_s1"] if s["s1"] >= s["vm"] else s["at_vm"])
                    row[ro] = (q, lc, round(float(np.hypot(X[0], X[1])), 2), round(float(np.degrees(np.arctan2(X[1], X[0]))), 0), round(float(X[2]), 2))
        out.append(row)
        line = (f"h_load {hl:.2f} phase +{dph:.0f} ({m.ndofs} dof): " + " | ".join(
            f"{ro} q {row[ro][0]:.3f} {row[ro][1]} at r {row[ro][2]} th {row[ro][3]:.0f} z {row[ro][4]}" for ro in ("support", "hub", "rim"))
            + f" [{time.time()-t0:.0f}s]")
        print(line, flush=True); fh.write(line + "\n"); fh.flush()
json.dump(out, open("conv_local.json", "w"), indent=1, default=float)
