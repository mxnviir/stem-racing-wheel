"""V4: stress recovery in a thin ring under diametral pinch, section at 90 deg from the loads.

Ring Ro = 14.1, t = 0.5 mm (R/t = 28), Al, pinch F over +-2 deg patches; half model with symmetry.

Reference (exact for a slender curved beam):
  section resultants at 90 deg:  N = -F/2,  M = F Rc (1/2 - 1/pi)   (thin-ring theory, Roark)
  Winkler-Bach curved-beam stress  sigma(r) = N/A + M (r - Rn) / (A e r),  Rn = t / ln(Ro/Ri), e = Rc - Rn
  i.e. inner fibre |sigma| 3 % above and outer fibre 3 % below the straight-beam 6M/(b t^2).

Checks
  (a) resultants from internal nodal forces of the upper half (exact equilibrium, no stress recovery)
  (b) narrow ring b = 1 mm (b^2/(R t) = 0.14, beam theory applies): peak von Mises from the three
      recovery modes vs Winkler, for a mesh study in theta and through the thickness
  (c) wide ring b = 5 mm (b^2/(R t) = 3.6): hoop stress at the free edges vs mid-width. This is the
      physical anticlastic edge effect of wide curved beams, not a discretisation error (it is
      mesh-converged), so the FE value is the one to design to.

Finding: 'nodes' over-reads by +3..8 % because curved quadratic elements carry a parasitic membrane
stress that is parabolic in theta and largest at the element ends; 'gauss' under-reads by up to 10 %
because the outermost Gauss point is inside the wall. 'surface' (wall surfaces in r/z, 2-point Gauss
stations in theta) removes both and is the default in wheelfem.stresses().
"""
import numpy as np
from wheelfem import Material, Block, Design, Model, _Bmat

Al = Material("Al", 71700, 0.33, 2810, 503)
Ro, t, F = 14.1, 0.5, 2.0
Ri, Rc = Ro - t, Ro - t / 2


def ring(b, h, nth, hz):
    d = Design("ring", [Block(Ri, Ro, 0, b, 0)], [Al], Ro, (0, b), Ri, (0, b))
    m = Model(d, h_r=h, h_z=hz, n_theta=nth)
    K = m.stiffness()
    th = lambda x: np.arctan2(x[1], x[0]); a = np.deg2rad(2)
    onR = lambda x: np.hypot(x[0], x[1]) > Ro - 0.1 * t
    f1, *_ = m.traction_patch(lambda x: onR(x) & (th(x) > np.pi/2) & (th(x) < np.pi/2 + a),
                              lambda x: np.array([0*x[0], -1 + 0*x[0], 0*x[0]]), F / 2)
    f2, *_ = m.traction_patch(lambda x: onR(x) & (th(x) < -np.pi/2) & (th(x) > -np.pi/2 - a),
                              lambda x: np.array([0*x[0], 1 + 0*x[0], 0*x[0]]), F / 2)
    f = f1 + f2
    D = [m.dofs_symmetry(), m.dofs_where(lambda x: np.abs(x[1]) < 1e-9, 1),
         m.dofs_where(lambda x: np.isclose(x[2], b / 2), 2)]
    u = m.solve(K, f, D)
    return m, f, u


def resultants(m, f, u):
    """N, M on the theta = 180 deg section from the internal forces of the upper-half elements."""
    c = m.mesh.p[:, m.mesh.t].mean(axis=1)
    keep = m.mat.copy(); m.mat[c[1] < 0] = 99
    Kup = m.stiffness(); m.mat = keep
    Rf = Kup @ u - f * np.repeat(m.X[1] > 1e-9, 3)
    n = np.nonzero((np.abs(m.X[1]) < 1e-9) & (m.X[0] < 0))[0]
    Ry, r = Rf[3 * n + 1], -m.X[0, n]
    return -Ry.sum(), -(Ry * (r - Rc)).sum()        # force/moment of the lower part on the upper part


def side_elements(m, deg=4):
    c = m.mesh.p[:, m.mesh.t].mean(axis=1)
    return np.abs(np.abs(np.arctan2(c[1], c[0])) - np.pi) < np.deg2rad(deg)


def hoop_at_y0(m, u, el):
    """Hoop stress (sigma_yy) at element nodes lying on the theta = 180 deg plane: {(r, z): value}."""
    C = Al.C(); out = {}
    for bb, G, gd in m._chunks(None, np.nonzero(el)[0], at="nodes"):
        s = np.einsum('eqia,ea->eqi', _Bmat(G), u[gd]) @ C.T
        X = np.asarray(bb.global_coordinates().value)
        on = np.abs(X[1]) < 1e-6
        for r, z, v in zip(-X[0][on], X[2][on], s[..., 1][on]):
            out.setdefault((round(r, 6), round(z, 6)), []).append(v)
    return {k: np.mean(v) for k, v in out.items()}


if __name__ == "__main__":
    A_ = lambda b: b * t
    Rn = t / np.log(Ro / Ri); e = Rc - Rn
    N_th, M_th = -F / 2, F * Rc * (0.5 - 1 / np.pi)
    wink = lambda b, r: N_th / A_(b) + M_th * (r - Rn) / (A_(b) * e * r)

    print("(a)+(b) narrow ring b = 1.0 mm; Winkler: outer %.2f, inner %.2f MPa (6M/bt^2 = %.2f)"
          % (wink(1, Ro), wink(1, Ri), 6 * M_th / t**2))
    print(f"{'h_r':>6} {'n_th':>5} {'dth deg':>7} | {'N/N_th':>7} {'M/M_th':>7} | "
          f"{'gauss':>6} {'nodes':>6} {'surface':>7}   (peak vM / Winkler peak |sigma|)")
    ref = abs(wink(1, Ri))
    for h, nth in [(0.5, 48), (0.25, 48), (0.25, 96), (0.125, 96), (0.125, 192)]:
        m, f, u = ring(1.0, h, nth, 0.25)
        N, M = resultants(m, f, u)
        side = side_elements(m)
        r = {at: m.stresses(u, at=at, el_filter=side)["Al"]["vm"] / ref for at in ("gauss", "nodes", "surface")}
        print(f"{h:6.3f} {nth:5d} {180/nth*1.4:7.2f} | {N/N_th:7.4f} {M/M_th:7.4f} | "
              f"{r['gauss']:6.3f} {r['nodes']:6.3f} {r['surface']:7.3f}")

    print("\n(c) wide ring b = 5.0 mm: hoop stress (MPa) on the 90-deg section, F = 2 N")
    print(f"{'h_r':>6} {'h_z':>5} {'n_th':>5} | {'out edge':>8} {'out mid':>8} {'in edge':>8} {'in mid':>8} | "
          f"{'edge/Wink':>9} | {'surface vM / Winkler':>20}")
    F5 = F                                                  # same total load on the wider ring
    for h, hz, nth in [(0.25, 1.0, 96), (0.125, 0.5, 96), (0.125, 0.5, 192)]:
        m, f, u = ring(5.0, h, nth, hz)
        S = hoop_at_y0(m, u, side_elements(m))
        g = lambda r, z: S[min(S, key=lambda k: abs(k[0] - r) + abs(k[1] - z))]
        oe, om, ie, im = g(Ro, 0), g(Ro, 2.5), g(Ri, 0), g(Ri, 2.5)
        w = max(abs(wink(5, Ro)), abs(wink(5, Ri)))
        vs = m.stresses(u, at="surface", el_filter=side_elements(m))["Al"]["vm"]
        print(f"{h:6.3f} {hz:5.2f} {nth:5d} | {oe:8.2f} {om:8.2f} {ie:8.2f} {im:8.2f} | "
              f"{max(abs(oe), abs(ie)) / w:9.3f} | {vs / w:20.3f}")
    print(f"Winkler (b = 5): outer {wink(5, Ro):.2f}, inner {wink(5, Ri):.2f} MPa")
