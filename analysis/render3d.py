"""
Renders of 3D wheel models: boundary surface of the tet mesh (half model mirrored to the full wheel),
flat-shaded geometry or coloured by a nodal field (e.g. max principal stress), painter's-sorted.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors


def surface(m):
    """Boundary triangles (3, 3, nf) as vertex coordinates and their vertex indices."""
    bf = m.mesh.boundary_facets()
    tri = m.mesh.facets[:, bf]                       # (3, nf) vertex ids
    keep = ~np.all(np.abs(m.mesh.p[0, tri]) < 1e-7, axis=0)   # drop the symmetry-plane cut faces
    return tri[:, keep]


def render(ax, m, field=None, vmax=None, base="#d9cdb5", mirror=True, elev=24, azim=-58, cmap="turbo",
           cut=False, title=None):
    tri = surface(m)
    P = m.mesh.p
    tris = [np.stack([P[:, tri[k]] for k in range(3)], 0)]           # (3 verts, 3 xyz, nf)
    vals = [None if field is None else field[tri].mean(0)]
    if mirror:
        Q = P * np.array([[-1], [1], [1]])
        if not cut:
            tris.append(np.stack([Q[:, tri[k]] for k in range(3)], 0)); vals.append(vals[0])
    T = np.concatenate(tris, -1).transpose(2, 0, 1)                   # (nf, 3, 3)
    T = T[:, :, [0, 2, 1]]                                            # view with the axle horizontal
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-30
    light = np.array([0.35, -0.75, 0.55]); light /= np.linalg.norm(light)
    k = 0.38 + 0.62 * np.abs(n @ light)
    if field is None:
        fc = np.clip(np.outer(k, colors.to_rgb(base)), 0, 1)
    else:
        v = np.concatenate([x for x in vals if x is not None])
        vmax = vmax or np.percentile(v, 99.8)
        rgb = cm.get_cmap(cmap)(colors.Normalize(0, vmax, clip=True)(v))[:, :3]
        fc = np.clip(rgb * (0.55 + 0.45 * k)[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(T, facecolors=fc, edgecolors="none", linewidths=0))
    R = np.abs(P[:2]).max() * 1.02; W = P[2].max()
    ax.set_xlim(-R, R); ax.set_ylim(W / 2 - R, W / 2 + R); ax.set_zlim(-R, R)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    if field is not None:
        sm = cm.ScalarMappable(norm=colors.Normalize(0, vmax), cmap=cmap)
        return sm
