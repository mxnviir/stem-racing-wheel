"""
Extract the optimised wheel as a watertight, smooth triangle surface (STL) from the density field.

* element densities on the (r, theta, z) grid are interpolated (periodic in theta) to a fine Cartesian
  voxel grid and combined with the exact analytic hub, rim and bore as a level-set union/difference, so
  the rim and the bearing seat come out exactly cylindrical
* marching cubes at the 0.5 iso-level, then Taubin smoothing (volume-preserving), largest connected body
* local wall thickness is measured on the voxel solid (twice the distance-to-surface on the medial axis)
"""
import json, sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage


def load(npz):
    d = np.load(npz, allow_pickle=True)
    s = json.loads(str(d["setup"]))
    r, z, nt = d["r"], d["z"], int(d["nt"])
    rho = d["rho"].reshape(len(r) - 1, nt, len(z) - 1)       # element order (i, j, k) = (r, theta, z)
    return s, r, z, nt, rho


def field(s, r, z, nt, rho, voxel):
    rc = 0.5 * (r[1:] + r[:-1]); zc = 0.5 * (z[1:] + z[:-1]); dth = 2 * np.pi / nt
    tc = (np.arange(nt) + 0.5) * dth
    # pad periodic in theta, constant in r/z
    A = rho
    A = np.concatenate([A[:, -1:], A, A[:, :1]], 1)
    tcp = np.concatenate([[tc[0] - dth], tc, [tc[-1] + dth]])
    f = RegularGridInterpolator((rc, tcp, zc), A, bounds_error=False, fill_value=None)
    R, W = s["R"], s["W"]
    n = int(np.ceil(2 * (R + 1) / voxel)); nz = int(np.ceil((W + 2) / voxel))
    xs = np.linspace(-R - 1, R + 1, n); zs = np.linspace(-1, W + 1, nz)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    rr = np.hypot(X, Y); th = np.mod(np.arctan2(Y, X), 2 * np.pi)
    pts = np.stack([np.clip(rr, rc[0], rc[-1]), th, np.clip(Z, zc[0], zc[-1])], -1)
    phi = (f(pts.reshape(-1, 3)).reshape(rr.shape) - 0.5) * 2.0 * s["rf"]   # ~signed distance, mm
    rim = s["t_rim"] / 2 - np.abs(rr - (R - s["t_rim"] / 2))
    hub = s["r_hub"] - rr
    seat_z = np.max([np.minimum(Z - a, b - Z) for a, b in s["seats"]], axis=0)
    hub = np.minimum(hub, seat_z)
    phi = np.maximum.reduce([phi, rim, hub])
    phi = np.minimum.reduce([phi, R - rr, rr - s["r_bore"], Z, W - Z])        # outer cylinder, bore, ends
    return phi, (xs, xs, zs)


def add_holes(phi, axes, s, n, d, r_h, side="W", depth=4.0):
    """Powder-escape holes: n axial cylinders (dia d at radius r_h) cut through the face on one side."""
    xs, ys, zs = axes
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    for k in range(n):
        a = np.pi / 2 + 2 * np.pi * k / n
        dist = np.hypot(X - r_h * np.cos(a), Y - r_h * np.sin(a)) - d / 2
        inz = (Z > s["W"] - depth) if side == "W" else (Z < depth)
        phi = np.where(inz, np.minimum(phi, dist), phi)
    return phi


def enforce_min_thickness(phi, axes, s, voxel, t_min):
    """Every wall >= t_min: union of the solid with balls of diameter t_min centred on its medial
    (centre) surface. Thick walls already contain those balls, so only thin walls grow; the result is
    clipped back to the wheel envelope (walls grow inwards from the end planes and the rim)."""
    solid = phi > 0
    din = ndimage.distance_transform_edt(solid)
    mx = ndimage.maximum_filter(din, size=3)
    ridge = solid & (din >= mx - 1e-9) & (din > 0)
    xs, ys, zs = axes
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rr = np.hypot(X, Y)
    # the rim skin is an exact analytic wall of thickness t_rim >= t_min: leave it alone (thickening it,
    # at the largest radius, is the most expensive place to add material)
    ridge &= rr < s["R"] - s["t_rim"] - voxel
    d_ridge = ndimage.distance_transform_edt(~ridge).astype(np.float32) * voxel
    # balls of diameter t_min (+1 voxel margin) on the medial surface; the voxel distance map is stepped and
    # leaves exact zeros that marching cubes turns into open, zero-volume slivers -> smooth it slightly
    ball = ndimage.gaussian_filter((t_min / 2 + 0.25 * voxel) - d_ridge, 0.7)
    phi2 = np.maximum(phi, ball)
    return np.minimum.reduce([phi2, s["R"] - rr, rr - s["r_bore"], Z, s["W"] - Z])


def fill_small_voids(phi, axes, s, v_max=2.0):
    """Enclosed voids smaller than v_max mm^3 (tiny bubbles left by the density field) are filled:
    they cannot be de-powdered, carry nothing, and make the surface non-watertight. Returns (phi, n, mm3)."""
    xs, ys, zs = axes
    vox = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rr = np.hypot(X, Y)
    inside = (rr < s["R"]) & (rr > s["r_bore"]) & (Z > 0) & (Z < s["W"])
    void = (phi <= 0) & inside
    lab, nlab = ndimage.label(void)
    outer = set(np.unique(lab[~ndimage.binary_erosion(inside, iterations=1) & void]).tolist()) - {0}
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
    small = [k for k in range(1, nlab + 1) if k not in outer and sizes[k - 1] * vox < v_max]
    if small:
        mask = np.isin(lab, small)
        phi = np.where(mask, np.abs(phi) + 0.5 * (xs[1] - xs[0]), phi)
    return phi, len(small), float(sum(sizes[k - 1] for k in small) * vox)


def cavity_holes(phi, axes, s, n=2, d=3.0):
    """Find enclosed cavities (void not connected to the outside) and drill n axial powder holes of
    diameter d from the inner face (z = W) to each cavity, at the cavity's radius. Returns (phi, info)."""
    xs, ys, zs = axes
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rr = np.hypot(X, Y)
    inside = (rr < s["R"]) & (rr > s["r_bore"]) & (Z > 0) & (Z < s["W"])
    void = (phi <= 0) & inside
    lab, nlab = ndimage.label(void)
    outer = set(np.unique(lab[~ndimage.binary_erosion(inside, iterations=1) & void]).tolist()) - {0}
    info = []
    for k in range(1, nlab + 1):
        if k in outer:
            continue
        cav = lab == k
        if cav.sum() < 50:
            continue
        r_c = float(np.median(rr[cav]))
        ztop = float(Z[cav & (np.abs(rr - r_c) < 0.5)].max()) if (cav & (np.abs(rr - r_c) < 0.5)).any() else float(Z[cav].max())
        depth = s["W"] - ztop + 0.6
        phi = add_holes(phi, axes, s, n, d, r_c, depth=depth)
        info.append(dict(r=round(r_c, 2), depth=round(depth, 2), voxels=int(cav.sum())))
    return phi, info


def surface(phi, axes, smooth=10):
    from skimage.measure import marching_cubes
    import trimesh
    sp = [a[1] - a[0] for a in axes]
    v, f, _, _ = marching_cubes(phi, 0.0, spacing=sp)
    v += np.array([a[0] for a in axes])
    m = trimesh.Trimesh(v, f, process=True)
    # keep the outer skin AND the walls of enclosed internal cavities; drop only small floating debris
    parts = m.split(only_watertight=False)
    main = max(parts, key=lambda p: abs(p.volume))
    sgn = np.sign(main.volume)
    cav = [p for p in parts if p is not main and np.sign(p.volume) != sgn and abs(p.volume) > 0.05]
    flo = [p for p in parts if p is not main and np.sign(p.volume) == sgn]
    big_flo = [p for p in flo if abs(p.volume) > 0.02 * abs(main.volume)]   # loose bodies worth reporting
    m = trimesh.util.concatenate([main] + cav + big_flo)
    if m.volume < 0:
        m.invert()
    m.metadata["n_shells"] = 1 + len(cav) + len(big_flo)
    m.metadata["n_cavities"] = len(cav)
    m.metadata["n_loose"] = len(big_flo)
    m.metadata["dropped_debris_mm3"] = float(sum(abs(p.volume) for p in flo if p not in big_flo))
    if smooth:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=smooth)
    return m


def snap(m, s, tol=0.08):
    """Put vertices back exactly on the analytic end planes, rim OD and bore after smoothing.
    Keep tol below ~half the voxel: with neighbouring vertices closer than tol, only some of them snap and
    triangles fold (voxel 0.1 with tol 0.08 left 166 folds on the rim of r11 and the volume mesher failed)."""
    v = m.vertices.copy()
    r = np.hypot(v[:, 0], v[:, 1])
    v[np.abs(v[:, 2]) < tol, 2] = 0.0
    v[np.abs(v[:, 2] - s["W"]) < tol, 2] = s["W"]
    for target in (s["R"], s["r_bore"]):
        k = np.abs(r - target) < tol
        v[k, :2] *= (target / r[k])[:, None]
    m.vertices = v
    return m


def thickness(phi, voxel):
    """Local thickness map of the solid: 2 x distance to the surface, sampled on the medial axis."""
    solid = phi > 0
    d = ndimage.distance_transform_edt(solid) * voxel
    # medial-axis-like points: local maxima of d
    mx = ndimage.maximum_filter(d, size=3)
    ridge = solid & (d >= mx - 1e-9) & (d > 0)
    return 2 * d[ridge], ridge, d


def ray_thickness(m, n=30000, seed=0):
    """Wall thickness at random surface points: distance along the inward normal to the opposite wall."""
    import trimesh
    np.random.seed(seed)
    pts, fi = trimesh.sample.sample_surface(m, n)
    nrm = m.face_normals[fi]
    t = np.full(n, np.nan)
    for a in range(0, n, 500):                    # chunks: candidate ray/triangle pairs grow with ray count
        o, d = pts[a:a + 500] - 1e-4 * nrm[a:a + 500], -nrm[a:a + 500]
        loc, idx_ray, _ = m.ray.intersects_location(o, d, multiple_hits=False)
        t[a + idx_ray] = np.linalg.norm(loc - pts[a + idx_ray], axis=1)
    return t, pts


if __name__ == "__main__":
    npz = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else npz.replace(".npz", ".stl")
    voxel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
    s, r, z, nt, rho = load(npz)
    phi, axes = field(s, r, z, nt, rho, voxel)
    t_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6       # SLS PA12 minimum wall
    if t_min > 0:
        phi = enforce_min_thickness(phi, axes, s, voxel, t_min)
    if len(sys.argv) > 4 and sys.argv[4] not in ("", "0", "none"):   # holes n,d,r_h (cut after thickening)
        n_h, d_h, r_h = sys.argv[4].split(",")
        phi = add_holes(phi, axes, s, int(n_h), float(d_h), float(r_h))
    m = snap(surface(phi, axes), s, tol=min(0.08, 0.5 * voxel))
    m.export(out)
    t, pts = ray_thickness(m)
    rr = np.hypot(pts[:, 0], pts[:, 1])
    # sharp convex edges (hole rims, face/rim corners) give short rays by construction; report the walls
    t = t[np.isfinite(t)]
    print(f"{out}: {len(m.faces)} triangles, {m.metadata['n_shells']} shells (>1 = enclosed cavities), watertight {m.is_watertight}, volume {m.volume:.1f} mm^3, "
          f"mass {m.volume * s['rho'] * 1e-6:.3f} g; ray wall thickness 0.5/1/5th percentile {np.percentile(t, 0.5):.2f}/"
          f"{np.percentile(t, 1):.2f}/{np.percentile(t, 5):.2f} mm, median {np.median(t):.2f} mm")
