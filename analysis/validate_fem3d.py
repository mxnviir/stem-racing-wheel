"""
Validation of the P2-tet solver (fem3d) used by verify_organic: thin ring pinch (Ro 14.1, t 0.5, b 1 mm, Al),
half model, section stress at 90 deg from the loads vs Winkler curved-beam theory and diameter change vs
Roark, under mesh refinement. Writes validate_fem3d.txt.
"""
import sys, time
import numpy as np
from fem3d import Model3D, Material

Al = Material("Al", 71700, 0.33, 2810, 503)


def ring_mesh(Ro, t, b, h):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    o = occ.addCylinder(0, 0, 0, 0, 0, b, Ro); i = occ.addCylinder(0, 0, 0, 0, 0, b, Ro - t)
    ring, _ = occ.cut([(3, o)], [(3, i)])
    box = occ.addBox(-2 * Ro, -2 * Ro, -1, 2 * Ro, 4 * Ro, b + 2)
    occ.intersect(ring, [(3, box)]); occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h / 3)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, en = gmsh.model.mesh.getElements(3)
    T = np.array(en[list(et).index(4)], np.int64).reshape(-1, 4)
    idx = np.full(int(nt.max()) + 1, -1); idx[nt.astype(int)] = np.arange(len(nt))
    P = xyz.reshape(-1, 3).T; T = idx[T].T
    u = np.unique(T); rm = np.full(P.shape[1], -1); rm[u] = np.arange(len(u))
    gmsh.finalize()
    return P[:, u], rm[T]


def check_ring(log):
    Ro, t, b, F = 14.1, 0.5, 1.0, 2.0
    Ri, Rc = Ro - t, Ro - t / 2
    Rn = t / np.log(Ro / Ri); e = Rc - Rn; A = b * t
    N, M = -F / 2, F * Rc * (0.5 - 1 / np.pi)
    wink = lambda r: N / A + M * (r - Rn) / (A * e * r)
    ref = max(abs(wink(Ro)), abs(wink(Ri)))
    log(f"(A) ring pinch, b = {b}: Winkler outer {wink(Ro):.2f}, inner {wink(Ri):.2f} MPa")
    for h in (0.5, 0.35, 0.25, 0.18):
        P, T = ring_mesh(Ro, t, b, h)
        m = Model3D(P, T, Al)
        K = m.stiffness()
        th = lambda x: np.arctan2(x[1], x[0])
        onR = lambda x: np.hypot(x[0], x[1]) > Ro - 0.02
        f1, _ = m.traction_patch(m.boundary_facets(lambda x: onR(x) & (th(x) > np.pi / 2) & (th(x) < np.pi / 2 + np.deg2rad(2))),
                                 lambda x: np.array([0 * x[0], -1 + 0 * x[0], 0 * x[0]]), F / 2)
        f2, _ = m.traction_patch(m.boundary_facets(lambda x: onR(x) & (th(x) < -np.pi / 2) & (th(x) > -np.pi / 2 - np.deg2rad(2))),
                                 lambda x: np.array([0 * x[0], 1 + 0 * x[0], 0 * x[0]]), F / 2)
        n0 = np.argmin(np.sum((m.X - np.array([[-Ro], [0], [0]]))**2, 0))
        D = [m.dofs_where(lambda x: np.abs(x[0]) < 1e-7, 0), m.dofs_where(lambda x: np.abs(x[1]) < 1e-7, 1),
             np.array([3 * n0 + 2])]
        u = m.solve(K, f1 + f2, D)
        side = (np.abs(m.X[1]) < 0.6) & (m.X[0] < 0)
        s = m.stresses(u, {"side": side})["side"]
        dD = -2 * m.disp_at(u, np.array([[0.0], [Ro], [b / 2]]))[1, 0]
        I = b * t**3 / 12
        dD_th = (np.pi / 4 - 2 / np.pi) * F * Rc**3 / (Al.E * I)
        log(f"    h={h:.2f} dofs {m.ndofs:7d}: peak vM / Winkler {s['vm'] / ref:.3f}   pinch dD / Roark {dD / dD_th:.3f}")


if __name__ == "__main__":
    fh = open("validate_fem3d.txt", "w")
    def log(s):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()
    check_ring(log)
