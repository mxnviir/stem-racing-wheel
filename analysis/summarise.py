"""Summary table of sized concepts + lowest natural frequency (bore fixed, mounted on axle)."""
import json
import numpy as np
from size import FAM
from wheelfem import Model
from modal import modes

res = json.load(open("size_results.json"))
f_rot = 20000 / 60
rows = []
for k, r in res.items():
    build, roles, rmat = FAM[k]
    d = build(r["final"]["t"])
    m = Model(d, h_r=0.8, h_z=0.8, n_theta=48)
    K, M = m.stiffness(), m.mass_matrix()
    f, *_ = modes(m, K, M, [m.dofs_fixed_bore(), m.dofs_symmetry()], k=4, sigma_hz=10.0)
    rows.append((k, r["ok"], r["final"]["t"], r["res"], float(f[0])))

rows.sort(key=lambda x: x[3]["m_eq_g"])
lines = ["| concept | pass | thicknesses mm | mass g | m_eq g | run um/N | pinch dD um | thumb um | spin dD um | f1 Hz (x rot) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for k, ok, t, s, f1 in rows:
    ts = ", ".join(f"{a} {b:.2f}" for a, b in sorted(t.items()))
    lines.append(f"| {k} | {'yes' if ok else 'NO'} | {ts} | {s['mass_g']:.2f} | {s['m_eq_g']:.3f} | {s['LC1_um_per_N']:.2f} | "
                 f"{s['LC3_dD_um']:.0f} | {s['LC4_face_um']:.0f} | {s['LC5_dD_um']:.2f} | {f1:.0f} ({f1/f_rot:.0f}x) |")
open("summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
