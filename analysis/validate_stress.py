"""V4: bending stress in a thin ring under diametral pinch, at 90 deg from the loads.
Theory (Roark): M = F R (1/2 - 1/pi) = 0.182 F R; sigma = 6 M / (b t^2) (+ small curved-beam/axial terms)."""
import numpy as np
from wheelfem import Material, Block, Design, Model
Al = Material("Al", 71700, 0.33, 2810, 503)
Ro, t, b, F = 14.1, 0.5, 5.0, 2.0
for h, nth in [(0.5, 48), (0.25, 96)]:
    d = Design("ring", [Block(Ro - t, Ro, 0, b, 0)], [Al], Ro, (0, b), Ro - t, (0, b))
    m = Model(d, h_r=h, h_z=1.0, n_theta=nth)
    K = m.stiffness()
    th = lambda x: np.arctan2(x[1], x[0]); a = np.deg2rad(2)
    onR = lambda x: np.hypot(x[0], x[1]) > Ro - 0.1 * t
    f1, *_ = m.traction_patch(lambda x: onR(x) & (th(x) > np.pi/2) & (th(x) < np.pi/2 + a), lambda x: np.array([0*x[0], -1+0*x[0], 0*x[0]]), F/2)
    f2, *_ = m.traction_patch(lambda x: onR(x) & (th(x) < -np.pi/2) & (th(x) > -np.pi/2 - a), lambda x: np.array([0*x[0], 1+0*x[0], 0*x[0]]), F/2)
    D = [m.dofs_symmetry(), m.dofs_where(lambda x: np.abs(x[1]) < 1e-9, 1), m.dofs_where(lambda x: np.isclose(x[2], b/2), 2)]
    u = m.solve(K, f1 + f2, D)
    c = m.mesh.p[:, m.mesh.t].mean(axis=1)
    side = np.abs(np.abs(np.arctan2(c[1], c[0])) - np.pi) < np.deg2rad(4)
    Rm = Ro - t/2
    s_th = 6 * (0.182 * F * Rm) / (b * t**2)
    for at in ("gauss", "nodes"):
        s = m.stresses(u, at=at, el_filter=side)["Al"]["vm"]
        print(f"h={h} at={at:5s}: FE {s:7.2f} MPa  theory {s_th:7.2f} MPa  ratio {s/s_th:.3f}")
