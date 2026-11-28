"""
Trim the rim edge bands of an optimised wheel and re-check it at full accuracy.

The optimiser's filter radius sets a minimum feature size of ~1 mm, so the stiffening bands it grows at the free
rim edges can come out deeper than the load needs. This keeps only `keep` mm of material under the rim skin within
`edge_w` mm of each rim edge, re-extracts the STL (optionally with a 45-degree edge chamfer, press on the flat
tread only) and runs the same check as the fine runs (press every 1 deg, spin, buckling, modes).

    python trim_rim_bands.py density.npz out_dir keep_mm [chamfer_mm] [t_min] [voxel]
"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "fine_runs"))
import verify_organic, run_all


def trim(npz, out, keep, edge_w=2.0):
    d = dict(np.load(npz, allow_pickle=True))
    s = json.loads(str(d["setup"])); r, z, nt = d["r"], d["z"], int(d["nt"])
    rho = d["rho"].reshape(len(r) - 1, nt, len(z) - 1).copy()
    rc = 0.5 * (r[1:] + r[:-1]); zc = 0.5 * (z[1:] + z[:-1])
    cut = (rc > s["R"] - s["t_rim"] - s["rim_zone"]) & (rc < s["R"] - s["t_rim"] - keep)
    edge = (zc < edge_w) | (zc > s["W"] - edge_w)
    rho[np.ix_(cut, np.ones(nt, bool), edge)] = 0.0
    d["rho"] = rho.ravel()
    np.savez_compressed(os.path.join(out, "density.npz"), **d)
    return s


if __name__ == "__main__":
    npz, out, keep = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ch = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    t_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    voxel = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    os.makedirs(out, exist_ok=True)
    s = trim(npz, out, keep)
    ext = run_all.do_extract(os.path.join(out, "density.npz"), os.path.join(out, "wheel.stl"), t_min, voxel, print, chamfer=ch)
    s["chamfer"] = ch
    res = verify_organic.run(os.path.join(out, "wheel.stl"), s, h=0.6, h_min=0.25, n_ang=int(round(360 / s["N"])),
                             buckle=3, log=print, modes=True)
    json.dump(dict(keep_mm=keep, chamfer_mm=ch, extract=ext, verify=res), open(os.path.join(out, "summary.json"), "w"),
              indent=1, default=float)
