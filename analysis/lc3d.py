"""
Load cases for the 3D wheel models (half model x <= 0, loads in the x = 0 plane, see geom3d).
Forces in the half model are half the physical forces.

  (patches: cos^2 profiles across theta of half-width 2x the nominal +-angle, smoothed band edges)
  LC1 run        1 N radial at the contact, +-2 deg, full width, bearing seats fixed (5x static load)
  LC2 press      20 N version of LC1 (= 20 x LC1)
  LC3 pinch      20 N diametral pinch, +-6 deg fingertips, full width, wheel off the car
  LC4 thumb      20 N axial, dia 8 mm at r = 8.5 on the outermost -z facing surface under the thumb
  LC5 spin       20 000 rpm
  LC6 edge press 20 N radial on a 3 mm band at the z = 0 rim edge, +-6 deg, seats fixed
  LC7 edge pinch 20 N diametral pinch on the z = 0 edge bands, off the car
  LC8 lip push   20 N axial on the z = 0 end face, outer 1 mm, +-16 deg (8 mm finger), seats fixed
Spoked wheels are evaluated at two phases: a spoke under the load (phase 90) and mid-way between
spokes (phase 90 + 180/N); the worst of the two is used.
Pinch cases: symmetry u_x = 0 on x = 0 plus the minimum rigid-body restraint (3 dofs at the bore).
"""
import numpy as np
from fem3d import Model3D

F = 20.0
EDGE = 3.0
OMEGA = 20000 * 2 * np.pi / 60
LCS = ("LC2", "LC3", "LC4", "LC5", "LC6", "LC7", "LC8")


def _th(x):
    return np.arctan2(x[1], x[0])


def bump(d, a):
    """cos^2 profile of half-width a (equal area to a uniform patch of half-width a/2... scaled below)."""
    return np.where(np.abs(d) < a, np.cos(0.5 * np.pi * d / a)**2, 0.0)


def ramp(d, w=0.5):
    """1 for d < -w/2, smooth cos^2 fall to 0 at d = +w/2 (smoothed edge of a band at d = 0)."""
    s = np.clip((d + w / 2) / w, 0, 1)
    return np.cos(0.5 * np.pi * s)**2


def setup(g, m: Model3D):
    """Load vectors and BC sets for wheel g on model m.
    Patch profiles are smooth (cos^2 across theta with the same total width as the nominal patch,
    smoothed band/circle edges) so loads converge with the mesh instead of depending on which facets
    happen to fall inside a narrow patch."""
    R, W = g.R, g.W
    down = lambda x: np.array([0 * x[0], -1 + 0 * x[0], 0 * x[0]])
    up = lambda x: np.array([0 * x[0], 1 + 0 * x[0], 0 * x[0]])
    axial = lambda x: np.array([0 * x[0], 0 * x[0], 1 + 0 * x[0]])
    r_ = lambda x: np.hypot(x[0], x[1])
    onR = lambda x: r_(x) > R - 0.05
    near = lambda c, a: (lambda x: onR(x) & (np.abs(np.angle(np.exp(1j * (_th(x) - c)))) < np.deg2rad(a) + 0.15 / R * 4))
    prof = lambda c, a, band=None: (lambda x: bump(np.angle(np.exp(1j * (_th(x) - c))), np.deg2rad(a))
                                    * (1.0 if band is None else ramp(x[2] - band)))
    T, B = np.pi / 2, -np.pi / 2
    f = {}
    # nominal patch +-a  ->  cos^2 of half-width 2a (same second moment order, same total force)
    f["LC1"], _ = m.traction_weighted(m.boundary_facets(near(T, 4)), prof(T, 2), down, 0.5)
    f["LC6"], _ = m.traction_weighted(m.boundary_facets(lambda x: near(T, 12)(x) & (x[2] < EDGE + 0.6)),
                                      prof(T, 6, EDGE), down, F / 2)
    f["LC3"] = (m.traction_weighted(m.boundary_facets(near(T, 12)), prof(T, 6), down, F / 2)[0]
                + m.traction_weighted(m.boundary_facets(near(B, 12)), prof(B, 6), up, F / 2)[0])
    f["LC7"] = (m.traction_weighted(m.boundary_facets(lambda x: near(T, 12)(x) & (x[2] < EDGE + 0.6)), prof(T, 6, EDGE), down, F / 2)[0]
                + m.traction_weighted(m.boundary_facets(lambda x: near(B, 12)(x) & (x[2] < EDGE + 0.6)), prof(B, 6, EDGE), up, F / 2)[0])
    # thumb: -z facing facets near the dia-8 circle at r = 8.5, outermost layer (min z) only
    rc = lambda x: np.sqrt(x[0]**2 + (x[1] - 8.5)**2)
    bf = m.boundary_facets(lambda x: (rc(x) < 4.6) & ~g.roles(x)["rim"])   # the rim lip is LC8's job
    n = m.facet_normals(bf)
    bf = bf[n[2] < -0.95]
    zf = m.mesh.p[2, m.mesh.facets[:, bf]].mean(axis=0)
    thumb = bf[zf < zf.min() + 0.3] if len(bf) else bf
    zt = zf.min() if len(bf) else 0.0
    wt = lambda x: ramp(rc(x) - 4.0) * (x[2] < zt + 0.3)
    f["LC4"], a4 = m.traction_weighted(thumb, wt, axial, F / 2) if len(thumb) else (None, 0.0)
    # lip: z = 0 end face, outer 1 mm, +-16 deg (smoothed edges)
    bf = m.boundary_facets(lambda x: (x[2] < 1e-6) & (r_(x) > R - 1.4) &
                           (_th(x) > np.pi / 2 - 0.01) & (_th(x) < np.pi / 2 + np.deg2rad(18)))
    wl = lambda x: ramp(R - 1.0 - r_(x), 0.3) * ramp(_th(x) - np.pi / 2 - np.deg2rad(16), np.deg2rad(2))
    f["LC8"], _ = m.traction_weighted(bf, wl, axial, F / 2)
    f["LC5"] = m.centrifugal(OMEGA)
    seat = lambda x: (np.hypot(x[0], x[1]) < g.r_bore + 1e-6) & np.any(
        [(x[2] > z0 - 1e-6) & (x[2] < z1 + 1e-6) for z0, z1 in g.seats], axis=0)
    Dsym = m.dofs_where(lambda x: np.abs(x[0]) < 1e-7, 0)
    Dbore = m.dofs_where(seat, [0, 1, 2])
    zc = np.mean(g.seats[0])
    pa = np.argmin(np.sum((m.X - np.array([[0], [g.r_bore], [zc]]))**2, 0))
    pb = np.argmin(np.sum((m.X - np.array([[0], [-g.r_bore], [zc]]))**2, 0))
    Drb = np.array([3 * pa + 1, 3 * pa + 2, 3 * pb + 2])
    return dict(f=f, Dbore=[Dbore, Dsym], Dpinch=[Dsym, Drb], thumb_area=2 * a4,
                Dbore_anti=[Dbore, m.dofs_where(lambda x: np.abs(x[0]) < 1e-7, [1, 2])])


def run(g, m: Model3D, K=None, keep_u=False):
    """All load cases on one mesh (one phase). Returns per-LC stresses by role and displacements."""
    K = m.stiffness() if K is None else K
    S = setup(g, m)
    f = S["f"]
    on = ["LC1", "LC5", "LC6", "LC8"] + (["LC4"] if f["LC4"] is not None else [])
    U = dict(zip(on, m.solve(K, [f[k] for k in on], S["Dbore"])))
    U.update(zip(["LC3", "LC7"], m.solve(K, [f["LC3"], f["LC7"]], S["Dpinch"])))
    U["LC2"] = 20 * U["LC1"]
    groups = g.roles(m.X)
    out = dict(**m.mass_props(), ndofs=m.ndofs, thumb_area=S["thumb_area"])
    out["m_eq_g"] = out["Izz_gmm2"] / g.R**2
    for k, u in U.items():
        if k != "LC1":
            out[k] = m.stresses(u, groups)
    R, W = g.R, g.W
    pc, pe = np.array([[0.0], [R], [W / 2]]), np.array([[0.0], [R], [0.0]])
    out["LC1_um_per_N"] = float(-m.disp_at(U["LC1"], pc)[1, 0] * 1e3)
    out["LC1_edge_um_per_N"] = float(-m.disp_at(U["LC1"], pe)[1, 0] * 1e3)
    pcb, peb = pc * np.array([[1], [-1], [1]]), pe * np.array([[1], [-1], [1]])
    dDy = lambda u, a, b: float(-(m.disp_at(u, a)[1, 0] - m.disp_at(u, b)[1, 0]) * 1e3)   # diameter change
    out["LC3_dD_um"] = dDy(U["LC3"], pc, pcb)
    out["LC5_dD_um"] = float(2 * m.disp_at(U["LC5"], pc)[1, 0] * 1e3)
    out["LC6_edge_um"] = float(-m.disp_at(U["LC6"], pe)[1, 0] * 1e3)
    out["LC7_dD_edge_um"] = dDy(U["LC7"], pe, peb)
    out["LC8_lip_um"] = float(m.disp_at(U["LC8"], pe)[2, 0] * 1e3)
    if "LC4" in U:
        out["LC4_um"] = float(np.max(np.abs(U["LC4"][2::3])) * 1e3)
    if keep_u:
        out["_U"], out["_S"], out["_K"] = U, S, K
    return out
