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


def surface(phi, axes, smooth=10):
    from skimage.measure import marching_cubes
    import trimesh
    sp = [a[1] - a[0] for a in axes]
    v, f, _, _ = marching_cubes(phi, 0.0, spacing=sp)
    v += np.array([a[0] for a in axes])
    m = trimesh.Trimesh(v, f, process=True)
    # keep the outer skin AND the walls of enclosed internal cavities; drop only small floating debris
    parts = m.split(only_watertight=False)
    big = max(abs(p.volume) for p in parts)
    keep = [p for p in parts if abs(p.volume) > 1e-3 * big or len(p.faces) > 2000]
    m = trimesh.util.concatenate(keep)
    if m.volume < 0:
        m.invert()
    m.metadata["n_shells"] = len(keep)
    if smooth:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=smooth)
    return m


def snap(m, s, tol=0.08):
    """Put vertices back exactly on the analytic end planes, rim OD and bore after smoothing."""
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


if __name__ == "__main__":
    npz = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else npz.replace(".npz", ".stl")
    voxel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
    s, r, z, nt, rho = load(npz)
    phi, axes = field(s, r, z, nt, rho, voxel)
    if len(sys.argv) > 4:                                   # holes: n,d,r_h
        n_h, d_h, r_h = sys.argv[4].split(",")
        phi = add_holes(phi, axes, s, int(n_h), float(d_h), float(r_h))
    m = snap(surface(phi, axes), s)
    m.export(out)
    t, _, _ = thickness(phi, voxel)
    print(f"{out}: {len(m.faces)} triangles, {m.metadata['n_shells']} shells (>1 = enclosed cavities), watertight {m.is_watertight}, volume {m.volume:.1f} mm^3, "
          f"mass {m.volume * s['rho'] * 1e-6:.3f} g; wall thickness 1st/5th percentile {np.percentile(t, 1):.2f}/"
          f"{np.percentile(t, 5):.2f} mm, min {t.min():.2f} mm")
