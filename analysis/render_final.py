"""Renders of the PA12 concepts (geometry) and of the final design (max principal stress, worst cases)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pa12 import BUILDERS, PA12_SLS, BEAR_M, BEAR_I, ALLOW
from geom3d import build
from fem3d import Model3D, _vm_s1
import lc3d
from render3d import render

os.makedirs("renders3d", exist_ok=True)
E = json.load(open("pa12_explore.json"))
names = ["W  central web", "WL central web + lips", "S6L 6 spokes + lips", "S6LP 6 spokes + lips + pads",
         "DF2 drum, faces 0.70->1.05 + powder holes"]

# 1. concept overview (geometry, sized parameters)
f = plt.figure(figsize=(4.0 * len(names), 4.6))
for i, n in enumerate(names):
    c = E[n]
    P, T, _ = build(BUILDERS[c["kind"]](c["p"]), h=1.4)
    m = Model3D(P, T, PA12_SLS)
    ok = "PASS" if c["ok"] else "did not converge"
    render(f.add_subplot(1, len(names), i + 1, projection="3d"), m,
           title=f"{n}\nm_eq {c['m_eq']:.2f} g, mass {c['mass']:.2f} g ({ok})")
f.suptitle("SLS PA12 rear-wheel concepts, sized to all 8 load cases (Ø28.2 × 17.2 mm)", fontsize=12)
f.tight_layout(); f.savefig("renders3d/pa12_concepts.png", dpi=130); plt.close(f)

# 2. final design: geometry front/back + stress for the governing cases
c = E[names[-1]]
g = BUILDERS[c["kind"]](c["p"])
P, T, _ = build(g, h=1.0, h_load=0.32)
m = Model3D(P, T, PA12_SLS, extra_mass_g=BEAR_M, extra_Izz=BEAR_I)
r = lc3d.run(g, m, keep_u=True)
f = plt.figure(figsize=(16, 8.4))
render(f.add_subplot(2, 3, 1, projection="3d"), m, title="final: outer face (thumb side)", elev=22, azim=-60)
render(f.add_subplot(2, 3, 4, projection="3d"), m, title="inner face: 2 x Ø3 powder-escape holes", elev=22, azim=120)
for k, (lc, title) in enumerate([("LC4", "LC4 thumb push 20 N"), ("LC7", "LC7 rim-edge pinch 20 N"),
                                 ("LC8", "LC8 rim-lip push 20 N"), ("LC3", "LC3 diametral pinch 20 N")]):
    _, s1, _ = _vm_s1(m.nodal_stress(r["_U"][lc]))
    ax = f.add_subplot(2, 3, [2, 3, 5, 6][k], projection="3d")
    sm = render(ax, m, field=np.clip(s1, 0, None)[:m.mesh.nvertices], vmax=ALLOW,
                title=f"{title}: max principal (MPa), allowable {ALLOW:.0f}", elev=22, azim=-60 if lc != "LC8" else -60)
    f.colorbar(sm, ax=ax, shrink=0.6, pad=0.0)
f.suptitle(f"Final SLS PA12 rear wheel: closed drum, rim 0.70, faces 0.70 -> 1.05 mm, hub 1.0 mm, 1 mm fillets | "
           f"m_eq {c['m_eq']:.3f} g, mass {c['mass']:.2f} g (incl. 2 x MR63)", fontsize=11)
f.tight_layout(); f.savefig("renders3d/pa12_final.png", dpi=130); plt.close(f)
print("wrote renders3d/")
