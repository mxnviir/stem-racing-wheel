"""
Cut-away renders of the sized wheel designs (axisymmetric blocks revolved over 270 deg, coloured by
material, MR63 bearings in grey), plus the r-z half-section.

Thicknesses come from size_results.json; families missing there fall back to an older results file
given as argv[1] (shown as 'provisional'). Writes renders/<family>.png and renders/overview.png.
"""
import sys, os, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from size import FAM
from concepts import W, R, RB

COL = {"Al": "#a9b4c2", "Ti": "#8c7f9e", "SLA rigid": "#e8c66a", "SLA glass": "#c9d98b",
       "CF": "#3a3a3a", "PA12": "#f2efe6"}
BEAR = "#6f6f6f"


def colour(name):
    for k, c in COL.items():
        if name.startswith(k):
            return c
    return "#cccccc"


def ring_faces(r0, r1, z0, z1, ta, tb, n, h=0.6):
    """Quads of a rectangular (r, z) section revolved from ta to tb, with the two cut faces.
    Surfaces are subdivided (~h mm) so the painter's sort works. Wheel axis is drawn along y."""
    th = np.linspace(ta, tb, n + 1)
    rs = np.linspace(r0, r1, max(2, int(np.ceil((r1 - r0) / h)) + 1))
    zs = np.linspace(z0, z1, max(2, int(np.ceil((z1 - z0) / h)) + 1))
    P = lambda r, t, z: (r * np.cos(t), z, r * np.sin(t))
    quads, normals = [], []
    for a, b in zip(th[:-1], th[1:]):
        m = 0.5 * (a + b)
        for za, zb in zip(zs[:-1], zs[1:]):
            quads += [[P(r1, a, za), P(r1, b, za), P(r1, b, zb), P(r1, a, zb)],
                      [P(r0, a, za), P(r0, b, za), P(r0, b, zb), P(r0, a, zb)]]
            normals += [(np.cos(m), 0, np.sin(m)), (-np.cos(m), 0, -np.sin(m))]
        for ra, rb in zip(rs[:-1], rs[1:]):
            quads += [[P(ra, a, z0), P(ra, b, z0), P(rb, b, z0), P(rb, a, z0)],
                      [P(ra, a, z1), P(ra, b, z1), P(rb, b, z1), P(rb, a, z1)]]
            normals += [(0, -1, 0), (0, 1, 0)]
    for t, sgn in ((ta, -1), (tb, 1)):
        for ra, rb in zip(rs[:-1], rs[1:]):
            for za, zb in zip(zs[:-1], zs[1:]):
                quads.append([P(ra, t, za), P(rb, t, za), P(rb, t, zb), P(ra, t, zb)])
                normals.append((-sgn * np.sin(t), 0, sgn * np.cos(t)))
    return quads, np.array(normals, float)


def shade(c, normals, light=np.array([0.4, -0.7, 0.6])):
    light = light / np.linalg.norm(light)
    k = 0.35 + 0.65 * np.abs(normals @ light)
    return np.clip(np.outer(k, to_rgb(c)), 0, 1)


def draw(ax3, ax2, d, title):
    parts = [(b.r0, b.r1, b.z0, b.z1, colour(d.materials[b.mat].name), d.materials[b.mat].name) for b in d.blocks]
    bearings = [(1.5, RB, z0, z1, BEAR, "MR63 bearing") for z0, z1 in d.seats]
    ta, tb = np.deg2rad(78), np.deg2rad(348)           # 270 deg revolved; the missing quarter faces the camera
    Q, C = [], []
    for r0, r1, z0, z1, c, _ in parts + bearings:
        q, n = ring_faces(r0, r1, z0, z1, ta, tb, 90)
        Q += q; C.append(shade(c, n))
    ax3.add_collection3d(Poly3DCollection(Q, facecolors=np.concatenate(C), edgecolors="none", linewidths=0))
    lim = R * 1.0
    ax3.set_xlim(-lim, lim); ax3.set_ylim(W / 2 - lim, W / 2 + lim); ax3.set_zlim(-lim, lim)
    ax3.set_box_aspect((1, 1, 1)); ax3.view_init(elev=28, azim=-35); ax3.set_axis_off()
    ax3.set_title(title, fontsize=9)
    if ax2 is None:
        return
    seen = {}
    for r0, r1, z0, z1, c, name in parts + bearings:
        ax2.add_patch(Rectangle((z0, r0), z1 - z0, r1 - r0, facecolor=c, edgecolor="k", lw=0.4,
                                label=None if name in seen else name))
        seen[name] = 1
    ax2.axhline(0, color="0.5", lw=0.6, ls="-.")
    ax2.set_xlim(-0.5, W + 0.5); ax2.set_ylim(0, R + 0.6); ax2.set_aspect("equal")
    ax2.set_xlabel("z (mm)", fontsize=7); ax2.set_ylabel("r (mm)", fontsize=7); ax2.tick_params(labelsize=6)
    ax2.legend(fontsize=6, loc="lower center", ncol=2, frameon=False)


if __name__ == "__main__":
    res = json.load(open("size_results.json")) if os.path.exists("size_results.json") else {}
    old = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    os.makedirs("renders", exist_ok=True)
    items = []
    for k in FAM:
        src = res.get(k) or old.get(k)
        if src:
            items.append((k, src, k not in res))
    n = len(items); cols = 4; rows = int(np.ceil(n / cols))
    big = plt.figure(figsize=(4.2 * cols, 4.4 * rows))
    for i, (k, r, prov) in enumerate(items):
        t = r["final"]["t"]; d = FAM[k][0](t)
        info = (f"{k}{'  [provisional]' if prov else ''}\nm_eq {r['res']['m_eq_g']:.2f} g, mass {r['res']['mass_g']:.2f} g\n"
                + ", ".join(f"{a} {b:.2f}" for a, b in sorted(t.items())) + " mm")
        f = plt.figure(figsize=(9, 4.2))
        draw(f.add_subplot(1, 2, 1, projection="3d"), f.add_subplot(1, 2, 2), d, info)
        f.tight_layout(); f.savefig(f"renders/{re.sub(r'[^A-Za-z0-9]+', '_', k).strip('_')}.png", dpi=160); plt.close(f)
        ax = big.add_subplot(rows, cols, i + 1, projection="3d")
        draw(ax, None, d, info)
    big.suptitle("Rear wheel concepts, Ø28.2 × 17.2 mm, cut-away (grey = MR63 bearings)", fontsize=12, y=0.995)
    big.tight_layout(rect=(0, 0, 1, 0.97)); big.savefig("renders/overview.png", dpi=130)
    print("wrote", len(items), "renders")
