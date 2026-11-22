"""
geom3d: parametric 3D wheel geometry (gmsh / OpenCASCADE) and tetrahedral meshing.

Wheel axis = z, outer (thumb-side) face at z = 0, inner face at z = W. Parts:
  rim    : revolved r-z profile; thickness can vary along the width (piecewise-linear stations)
           plus optional inward lips (flanges) at both rim edges
  hub    : tube r_bore .. r_bore + t_hub over [hub_z0, hub_z1]
  support: any number of spoke sets (N spokes, planform width taper hub->rim, axial thickness taper
           hub->rim, axial position), a full web (N = 0), or drum faces
  pads   : optional local rim thickening at the spoke roots (theta-varying rim thickness)
  fillets: every concave (re-entrant) edge of the fused solid is filleted with radius rf; concave
           edges are found by probing solid occupancy around each edge (fraction > 0.6)

The solid is cut to the half x <= 0. All loads are applied in the x = 0 plane, and the spoke pattern is
mirror-symmetric about it (phase 90 deg = spoke at the top load point, 90 + 180/N = between spokes),
so the half model with symmetry (or antisymmetry for buckling/modes) is exact.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpokeSet:
    N: int = 6                  # 0 = full web (axisymmetric disc)
    zc: float = 8.6             # axial centre, mm
    ts_hub: float = 1.2         # axial thickness at the hub, mm
    ts_rim: float = 1.2         # axial thickness at the rim, mm
    w_hub: float = 1.5          # planform width at the hub, mm (ignored for a web)
    w_rim: float = 1.5          # planform width at the rim, mm
    phase_deg: float = 90.0     # angle of the first spoke (90 = at the top load point)


@dataclass
class Wheel:
    name: str = "wheel"
    R: float = 14.1
    W: float = 17.2
    r_bore: float = 3.0
    t_hub: float = 0.8
    hub_z: tuple = (5.6, 11.6)
    seats: list = field(default_factory=lambda: [(6.1, 11.1)])      # fixed bearing seats (z ranges)
    rim_t: list = field(default_factory=lambda: [(0.0, 0.8), (17.2, 0.8)])   # (z, thickness) stations
    lip_h: float = 0.0          # radial depth of inward lips at both rim edges (0 = none)
    lip_t: float = 0.8          # axial thickness of the lips
    spokes: list = field(default_factory=lambda: [SpokeSet()])
    drum_t: float = 0.0         # >0: closed drum faces of this thickness at z = 0 and z = W
    drum_t_hub: float = 0.0     # face thickness at the hub (0 = same as drum_t, i.e. no taper)
    holes_n: int = 0            # powder-escape holes (SLS) in the inner face (z = W) of a drum
    holes_d: float = 3.0        # hole diameter, mm
    holes_r: float = 8.5        # pitch radius, mm
    holes_phase_deg: float = 90.0
    pad_t: float = 0.0          # extra rim thickness at spoke roots, smooth cos^2 in theta (0 = none)
    pad_deg: float = 15.0       # half-angle of the pads
    rf: float = 0.5             # fillet radius at concave junctions (0 = sharp)

    def rim_thickness(self, z):
        zs, ts = np.array(self.rim_t).T
        return np.interp(z, zs, ts)

    def bump(self, theta):
        """Pad height (theta-varying extra rim thickness) at angle theta."""
        b = np.zeros_like(np.asarray(theta, float)); hw = np.deg2rad(self.pad_deg)
        if self.pad_t <= 0:
            return b
        for s in self.spokes:
            for k in range(s.N):
                a = np.deg2rad(s.phase_deg) + 2 * np.pi * k / s.N
                d = np.angle(np.exp(1j * (np.asarray(theta) - a)))
                b = np.maximum(b, np.where(np.abs(d) < hw, self.pad_t * np.cos(0.5 * np.pi * d / hw)**2, 0.0))
        return b

    def roles(self, X, tol=0.3):
        """Role of each point: 'rim' (inside the rim, lips and pads), 'hub', 'support' (spokes/web/faces
        and their junction fillets)."""
        r, th, z = np.hypot(X[0], X[1]), np.arctan2(X[1], X[0]), X[2]
        t = self.rim_thickness(z)
        lip = (self.lip_h > 0) & ((z < self.lip_t + 1e-6) | (z > self.W - self.lip_t - 1e-6))
        r_in = self.R - t - np.where(lip, self.lip_h, 0.0)
        tb = float(np.max(self.rim_thickness(np.linspace(0, self.W, 50))))
        r_in = np.minimum(r_in, np.where(self.bump(th) > 0, self.R - tb - self.bump(th), np.inf))
        rim = r >= r_in - 1e-3
        hub = r <= self.r_bore + self.t_hub + 1e-3
        return {"rim": rim, "hub": hub & ~rim, "support": ~rim & ~hub}

    def rotated(self, dphase):
        return dataclasses.replace(self, spokes=[dataclasses.replace(s, phase_deg=s.phase_deg + dphase)
                                                 for s in self.spokes],
                                   holes_phase_deg=self.holes_phase_deg + dphase)


def _rim(occ, g):
    R, W, lt = g.R, g.W, g.lip_t
    t = g.rim_thickness
    if g.lip_h > 0:
        zs = [z for z in sorted({z for z, _ in g.rim_t}) if lt < z < W - lt]
        inner = [(R - t(z), z) for z in [lt] + zs + [W - lt]]
        rl0, rl1 = R - t(0) - g.lip_h, R - t(W) - g.lip_h
        pts = ([(R, 0.0), (R, W), (rl1, W), (rl1, W - lt)] + list(reversed(inner)) + [(rl0, lt), (rl0, 0.0)])
    else:
        zs = sorted({z for z, _ in g.rim_t} | {0.0, W})
        pts = [(R, 0.0), (R, W)] + [(R - t(z), z) for z in reversed(zs)]
    return _revolve_poly(occ, pts)


def _revolve_poly(occ, rz, angle=2 * np.pi, a0=0.0):
    """Revolve a closed polygon given in (r, z) about the z axis, starting at angle a0."""
    clean = [rz[0]]
    for p in rz[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    if np.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) < 1e-9:
        clean.pop()
    P = [occ.addPoint(r * np.cos(a0), r * np.sin(a0), z) for r, z in clean]
    L = [occ.addLine(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
    s = occ.addPlaneSurface([occ.addCurveLoop(L)])
    out = occ.revolve([(2, s)], 0, 0, 0, 0, 0, 1, angle)
    return [dt for dt in out if dt[0] == 3]


def _spokes(occ, g, s):
    r0 = g.r_bore + g.t_hub / 2
    r1 = g.R - g.rim_thickness(s.zc) / 2
    rz = [(r0, s.zc - s.ts_hub / 2), (r1, s.zc - s.ts_rim / 2), (r1, s.zc + s.ts_rim / 2), (r0, s.zc + s.ts_hub / 2)]
    if s.N == 0:
        return _revolve_poly(occ, rz)
    vols = []
    tz = max(s.ts_hub, s.ts_rim)
    for k in range(s.N):
        a = np.deg2rad(s.phase_deg) + 2 * np.pi * k / s.N
        P = [occ.addPoint(x, y, s.zc - tz / 2) for x, y in
             [(r0 - 0.3, -s.w_hub / 2), (r1, -s.w_rim / 2), (r1, s.w_rim / 2), (r0 - 0.3, s.w_hub / 2)]]
        L = [occ.addLine(P[i], P[(i + 1) % 4]) for i in range(4)]
        sf = occ.addPlaneSurface([occ.addCurveLoop(L)])
        ex = [dt for dt in occ.extrude([(2, sf)], 0, 0, tz) if dt[0] == 3]
        if abs(s.ts_hub - s.ts_rim) > 1e-9:      # axial taper: intersect with a revolved r-z envelope
            env = _revolve_poly(occ, [(r0 - 0.3, s.zc - s.ts_hub / 2), (r1, s.zc - s.ts_rim / 2),
                                      (r1, s.zc + s.ts_rim / 2), (r0 - 0.3, s.zc + s.ts_hub / 2)])
            ex, _ = occ.intersect(ex, env)
        occ.rotate(ex, 0, 0, 0, 0, 0, 1, a)
        vols += ex
    return vols


def _pads(occ, g):
    """Theta-varying rim thickness: a band whose inner surface is r_in(theta) = R - t - bump(theta),
    bump = pad_t * cos^2 hump of half-width pad_deg centred on each spoke root, over the rim width
    between the lips (or the full width). Smooth in theta, so it fuses and fillets cleanly."""
    sets = [s for s in g.spokes if s.N > 0]
    if g.pad_t <= 0 or not sets:
        return []
    z0, z1 = (g.lip_t, g.W - g.lip_t) if g.lip_h > 0 else (0.0, g.W)
    tb = float(np.max(g.rim_thickness(np.linspace(z0, z1, 50))))
    th = np.linspace(0, 2 * np.pi, 721)[:-1]
    bump = np.zeros_like(th); hw = np.deg2rad(g.pad_deg)
    for s in sets:
        for k in range(s.N):
            a = np.deg2rad(s.phase_deg) + 2 * np.pi * k / s.N
            d = np.angle(np.exp(1j * (th - a)))
            bump = np.maximum(bump, np.where(np.abs(d) < hw, g.pad_t * np.cos(0.5 * np.pi * d / hw)**2, 0.0))
    r_in = g.R - tb - bump
    P = [occ.addPoint(r * np.cos(a), r * np.sin(a), z0) for r, a in zip(r_in, th)]
    inner = occ.addCurveLoop([occ.addBSpline(P + [P[0]])])
    outer = occ.addCurveLoop([occ.addCircle(0, 0, z0, g.R - 0.5 * tb)])
    sf = occ.addPlaneSurface([outer, inner])
    return [dt for dt in occ.extrude([(2, sf)], 0, 0, z1 - z0) if dt[0] == 3]


def _concave_edges(gmsh, vol, eps=0.03, n=24):
    """Curves of volume `vol` where the solid occupies > 60 % of a small circle around the edge."""
    out = []
    curves = {abs(t) for d, t in gmsh.model.getEntities(1)}
    for c in sorted(curves):
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        hits = tot = 0
        for u in (0.3, 0.5, 0.7):
            p = lo[0] + u * (hi[0] - lo[0])
            x = np.array(gmsh.model.getValue(1, c, [p]))
            d = np.array(gmsh.model.getDerivative(1, c, [p]))
            if np.linalg.norm(d) < 1e-12:
                continue
            d /= np.linalg.norm(d)
            a = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]); a /= np.linalg.norm(a)
            b = np.cross(d, a)
            for k in range(n):
                q = x + eps * (np.cos(2 * np.pi * k / n) * a + np.sin(2 * np.pi * k / n) * b)
                hits += gmsh.model.isInside(3, vol, list(q)) > 0
                tot += 1
        if tot and hits / tot > 0.6:
            out.append(c)
    return out


def _fillet_all(gmsh, v, rf):
    """Fillet every concave edge of volume v in one OCC operation (rf, then 0.6 rf on failure).
    Returns (volume, n filleted, n concave edges left sharp)."""
    occ = gmsh.model.occ
    E = _concave_edges(gmsh, v)
    if not E:
        return v, 0, 0
    for r in (rf, 0.6 * rf):
        try:
            out = occ.fillet([v], E, [r], removeVolume=False)
            nv = [t for d, t in out if d == 3]
            if len(nv) != 1:
                raise RuntimeError
            occ.remove([(3, v)], recursive=True)
            occ.synchronize()
            return nv[0], len(E), 0
        except Exception:
            occ.synchronize()
    return v, 0, len(E)


def build(g: Wheel, h=0.6, h_load=None, half=True, step_file=None, verbose=False):
    """Mesh the wheel. Returns (points (3, n), tets (4, ne), info). Linear tets (straight-sided);
    the solver uses quadratic (P2) displacement on them."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.model.add(g.name)
    occ = gmsh.model.occ
    parts = _rim(occ, g)
    parts += _revolve_poly(occ, [(g.r_bore, g.hub_z[0]), (g.r_bore + g.t_hub, g.hub_z[0]),
                                 (g.r_bore + g.t_hub, g.hub_z[1]), (g.r_bore, g.hub_z[1])])
    for s in g.spokes:
        parts += _spokes(occ, g, s)
    if g.drum_t > 0:      # faces flush with the wheel ends; optional radial taper (thicker at the hub)
        ri, ro = g.r_bore + g.t_hub / 2, g.R - g.rim_thickness(0.0) / 2
        th = g.drum_t_hub if g.drum_t_hub > 0 else g.drum_t
        parts += _revolve_poly(occ, [(ri, 0.0), (ro, 0.0), (ro, g.drum_t), (ri, th)])
        parts += _revolve_poly(occ, [(ri, g.W - th), (ro, g.W - g.drum_t), (ro, g.W), (ri, g.W)])
    parts += _pads(occ, g)
    vol, _ = occ.fuse(parts[:1], parts[1:]) if len(parts) > 1 else (parts, None)
    occ.removeAllDuplicates()
    occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        gmsh.finalize()
        raise RuntimeError(f"fused geometry has {len(vols)} volumes (parts not connected?)")
    v = vols[0][1]
    nfil = nfail = 0
    if g.rf > 0:
        v, nfil, nfail = _fillet_all(gmsh, v, g.rf)
    if g.holes_n > 0 and g.drum_t > 0:
        cyl = []
        for k in range(g.holes_n):
            a = np.deg2rad(g.holes_phase_deg) + 2 * np.pi * k / g.holes_n
            cyl.append((3, occ.addCylinder(g.holes_r * np.cos(a), g.holes_r * np.sin(a), g.W - g.drum_t - 0.2,
                                           0, 0, g.drum_t + 0.4, g.holes_d / 2)))
        out, _ = occ.cut([(3, v)], cyl)
        occ.synchronize()
        v = out[0][1]
    if step_file:
        gmsh.write(step_file)
    if half:
        box = occ.addBox(-2 * g.R, -2 * g.R, -1, 2 * g.R, 4 * g.R, g.W + 2)
        out, _ = occ.intersect([(3, v)], [(3, box)])
        occ.synchronize()
        v = out[0][1]
    stray = [e for e in gmsh.model.getEntities(3) if e[1] != v]
    if stray:
        occ.remove(stray, recursive=True); occ.synchronize()

    # mesh sizes
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min(0.25, h, h_load or h))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)        # HXT
    gmsh.option.setNumber("Mesh.Optimize", 1)
    if h_load and h_load < h:
        f = gmsh.model.mesh.field
        boxes = []
        hl2 = min(h_load, 0.2)                  # contact lines: resolve the +-4 deg running-load profile
        for sgn in (1, -1):
            b = f.add("Box")
            for k, val in zip(("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "VIn", "VOut"),
                              (-g.R * np.sin(np.deg2rad(5)) - 0.2, 0.2,
                               *((g.R - 1.6, g.R + 0.3) if sgn > 0 else (-g.R - 0.3, -g.R + 1.6)), -0.5, g.W + 0.5, hl2, h)):
                f.setNumber(b, k, val)
            f.setNumber(b, "Thickness", 1.0)
            boxes.append(b)
        th = np.deg2rad(14)                     # contact patches (+-12 deg pinch profiles) plus margin
        xs = g.R * np.sin(th) + 0.5
        for (x0, x1, y0, y1, z0, z1) in [(-xs, xs, g.R * np.cos(th) - 3.0, g.R + 0.5, -0.5, g.W + 0.5),    # top contact
                                         (-xs, xs, -g.R - 0.5, -g.R * np.cos(th) + 3.0, -0.5, g.W + 0.5),  # bottom
                                         (-4.5, 4.5, 4.0, 13.0, -0.5, 3.5)]:                            # thumb
            b = f.add("Box")
            for k, val in zip(("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "VIn", "VOut"),
                              (x0, x1, y0, y1, z0, z1, h_load, h)):
                f.setNumber(b, k, val)
            f.setNumber(b, "Thickness", 1.5)
            boxes.append(b)
        m = f.add("Min"); f.setNumbers(m, "FieldsList", boxes); f.setAsBackgroundMesh(m)
    for alg in (10, 1, 4):                 # HXT, then Delaunay, then Frontal on failure
        try:
            gmsh.option.setNumber("Mesh.Algorithm3D", alg)
            gmsh.model.mesh.generate(3)
            et_ = gmsh.model.mesh.getElements(3)[0]
            if 4 not in list(et_):
                raise RuntimeError("no tetrahedra")
            break
        except Exception:
            gmsh.model.mesh.clear()
            if alg == 4:
                gmsh.finalize()
                raise
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    et, etags, enodes = gmsh.model.mesh.getElements(3)
    tets = np.array(enodes[list(et).index(4)], dtype=np.int64).reshape(-1, 4)
    idx = np.full(int(nt.max()) + 1, -1, np.int64); idx[nt.astype(np.int64)] = np.arange(len(nt))
    P = xyz.reshape(-1, 3).T.copy()
    T = idx[tets].T.copy()
    used = np.unique(T)
    remap = np.full(P.shape[1], -1, np.int64); remap[used] = np.arange(len(used))
    P, T = P[:, used], remap[T]
    gmsh.finalize()
    if P[0].max() > (1e-6 if half else g.R + 0.01) or np.hypot(P[0], P[1]).max() > g.R + 0.01:
        raise RuntimeError("mesh outside the wheel envelope: x max %.3f" % P[0].max())
    return P, T, dict(fillets=nfil, fillet_failures=nfail)
