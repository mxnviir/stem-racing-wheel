"""
Concept screening: candidate rear-wheel architectures (Ø28.2 x 17.2 mm, MR63 3x6x2.5 bearings,
fixed axle) under the agreed load cases.

Load cases (bore = bearing seats fixed unless stated)
  LC1 run   : 1 N radial at contact (5x static 0.19 N, covers imbalance/bumps), line patch +-2 deg
  LC2 press : 20 N radial at contact, bore fixed (handling: car pressed onto a surface)
  LC3 pinch : 20 N diametral pinch, wheel off the car (fingertip patch +-6 deg), bore free
  LC4 push  : 20 N axial thumb push, dia 8 mm patch on outer face at r = 8.5 mm
  LC5 spin  : 20 000 rpm centrifugal
  LC6 edge press : 20 N radial on a 3 mm band at one rim edge (+-6 deg), bore fixed
  LC7 edge pinch : 20 N diametral pinch on the same edge bands, wheel off the car
  LC8 lip push   : 20 N axial on the rim lip (outer 1 mm of the end face, +-16 deg ~ 8 mm finger), bore fixed
  LC1-LC4 load the full width and are centred on a web, so they never load an overhanging rim edge.
  LC6-LC8 close that gap.
Pass criteria: metals SF >= 1.5 on von Mises vs yield; polymers SF >= 2 on max principal vs UTS;
CF SF >= 2 on von Mises vs (conservative) strength.
"""
import sys, json, time
import numpy as np
from wheelfem import Material, Block, Design, Model

# ---- materials (typical datasheet values; replace with your exact grades) ----
AL = Material("Al 7075-T6", 71700, 0.33, 2810, 503)
TI = Material("Ti-6Al-4V", 113800, 0.34, 4430, 880)
SLA = Material("SLA rigid resin", 2800, 0.38, 1180, 65)
SLA10K = Material("SLA glass-filled", 10000, 0.35, 1650, 65)
CF = Material("CF tube (hoop approx)", 50000, 0.30, 1550, 400)
PA12 = Material("SLS PA12", 1700, 0.40, 1010, 48)

R, W = 14.1, 17.2
RB = 3.0                     # bearing seat radius (6 mm OD)
MR63_M = 2 * 0.32            # g, two MR63ZZ per wheel (for reference only)
MR63_I = 2 * 0.18 * 2.7**2   # g mm^2, their rotating outer races (for reference only)
BEAR_M = BEAR_I = 0.0        # bearings are NOT included in reported mass / inertia (wheel body only)
SEATS = [(0, 2.5), (W - 2.5, W)]
EDGE = 3.0                   # mm, width of the rim-edge load band (LC6, LC7)
LCS = ("LC2", "LC3", "LC4", "LC5", "LC6", "LC7", "LC8")   # load cases checked against strength


def drum(name, t_rim, t_face, t_hub, m_rim, m_face, m_hub, mats):
    """Closed drum: rim + two flat faces + full-width hub tube, bearings at both ends."""
    rh = RB + t_hub
    blocks = [Block(R - t_rim, R, 0, W, m_rim),
              Block(rh, R - t_rim, 0, t_face, m_face),
              Block(rh, R - t_rim, W - t_face, W, m_face),
              Block(RB, rh, 0, W, m_hub)]
    return Design(name, blocks, mats, R, (0, W), RB, SEATS, BEAR_M, BEAR_I)


def web(name, t_rim, t_web, t_hub, m_rim, m_web, m_hub, mats, hub_len=6.0):
    """Rim + single central web + short hub, two bearings back-to-back in the hub."""
    rh, zc = RB + t_hub, W / 2
    blocks = [Block(R - t_rim, R, 0, W, m_rim),
              Block(rh, R - t_rim, zc - t_web / 2, zc + t_web / 2, m_web),
              Block(RB, rh, zc - hub_len / 2, zc + hub_len / 2, m_hub)]
    return Design(name, blocks, mats, R, (0, W), RB, [(zc - 2.5, zc + 2.5)], BEAR_M, BEAR_I)


def solid(name, mat):
    return Design(name, [Block(RB, R, 0, W, 0)], [mat], R, (0, W), RB, SEATS, BEAR_M, BEAR_I)


CONCEPTS = {
    "B0 solid PA12 (baseline)":      lambda: solid("B0", PA12),
    "A1 Al drum 0.3/0.3":            lambda: drum("A1", 0.30, 0.30, 0.5, 0, 0, 0, [AL]),
    "A2 Al web 0.3/0.4":             lambda: web("A2", 0.30, 0.40, 0.5, 0, 0, 0, [AL]),
    "T1 Ti drum 0.3/0.3":            lambda: drum("T1", 0.30, 0.30, 0.5, 0, 0, 0, [TI]),
    "S1 SLA drum 0.5/0.45":          lambda: drum("S1", 0.50, 0.45, 0.8, 0, 0, 0, [SLA]),
    "S2 SLA-10K drum 0.5/0.45":      lambda: drum("S2", 0.50, 0.45, 0.8, 0, 0, 0, [SLA10K]),
    "P1 PA12 drum 0.8/0.8":          lambda: drum("P1", 0.80, 0.80, 1.0, 0, 0, 0, [PA12]),
    "H1 CF rim + SLA faces + Al hub": lambda: drum("H1", 0.30, 0.45, 0.5, 0, 1, 2, [CF, SLA, AL]),
    "H2 SLA drum + Al hub":          lambda: drum("H2", 0.50, 0.45, 0.5, 0, 0, 1, [SLA, AL]),
}


def run(design, n_theta=48, by="mat"):
    solid_like = len(design.blocks) == 1
    m = Model(design, h_r=1.6 if solid_like else 0.8, h_z=1.6 if solid_like else 0.8,
              n_theta=32 if solid_like else n_theta, z_extra=(EDGE, W - EDGE))
    K = m.stiffness()
    th = lambda x: np.arctan2(x[1], x[0])
    on_R = lambda x: np.hypot(x[0], x[1]) > R - 0.05
    down = lambda x: np.array([0 * x[0], -1 + 0 * x[0], 0 * x[0]])
    up = lambda x: np.array([0 * x[0], 1 + 0 * x[0], 0 * x[0]])
    top = lambda a: (lambda x: on_R(x) & (th(x) > np.pi / 2) & (th(x) < np.pi / 2 + np.deg2rad(a)))
    bot = lambda a: (lambda x: on_R(x) & (th(x) < -np.pi / 2) & (th(x) > -np.pi / 2 - np.deg2rad(a)))
    Dbore, Dsym = m.dofs_fixed_bore(), m.dofs_symmetry()
    out = dict(ndofs=m.ndofs, **m.mass_props())
    probe = np.array([[0.0], [R], [W / 2]])

    f1, *_ = m.traction_patch(top(2), down, 1.0 / 2)
    zmin = min(b.z0 for b in design.blocks if b.r0 < 8.5 < b.r1)   # outermost surface under the thumb
    thumb = lambda x: (np.abs(x[2] - zmin) < 1e-6) & (x[0]**2 + (x[1] - 8.5)**2 < 16.0)
    f4, *_ = m.traction_patch(thumb, lambda x: np.array([0 * x[0], 0 * x[0], 1 + 0 * x[0]]), 20.0 / 2)
    f5 = m.centrifugal(20000 * 2 * np.pi / 60)
    edge = lambda sel: (lambda x: sel(x) & (x[2] < EDGE + 1e-6))
    f6, *_ = m.traction_patch(edge(top(6)), down, 20.0 / 2)
    lip = lambda x: ((np.abs(x[2]) < 1e-6) & (np.hypot(x[0], x[1]) > R - 1.0 - 1e-6)
                     & (th(x) > np.pi / 2) & (th(x) < np.pi / 2 + np.deg2rad(16)))
    f8, *_ = m.traction_patch(lip, lambda x: np.array([0 * x[0], 0 * x[0], 1 + 0 * x[0]]), 20.0 / 2)
    u1, u4, u5, u6, u8 = m.solve(K, [f1, f4, f5, f6, f8], [Dbore, Dsym])
    pedge = np.array([[0.0], [R], [0.0]])
    out["LC1_edge_um_per_N"] = float(-m.disp_at(u1, pedge)[1, 0] * 1e3)
    out["LC6"] = m.stresses(u6, by=by)
    out["LC6_edge_um"] = float(-m.disp_at(u6, pedge)[1, 0] * 1e3)
    out["LC8"] = m.stresses(u8, by=by)
    out["LC8_lip_um"] = float(m.disp_at(u8, pedge)[2, 0] * 1e3)
    out["LC1_um_per_N"] = float(-m.disp_at(u1, probe)[1, 0] * 1e3)
    out["LC1"] = m.stresses(u1, by=by)
    out["LC2"] = m.stresses(20 * u1, by=by)
    out["LC4"] = m.stresses(u4, by=by)
    out["LC4_face_um"] = float(m.disp_at(u4, np.array([[0.0], [8.5], [zmin]]))[2, 0] * 1e3)
    out["LC5"] = m.stresses(u5, by=by)
    out["LC5_dD_um"] = float(2 * m.disp_at(u5, probe)[1, 0] * 1e3)

    f3a, *_ = m.traction_patch(top(6), down, 20.0 / 2)
    f3b, *_ = m.traction_patch(bot(6), up, 20.0 / 2)
    f7a, *_ = m.traction_patch(edge(top(6)), down, 20.0 / 2)
    f7b, *_ = m.traction_patch(edge(bot(6)), up, 20.0 / 2)
    Dy = m.dofs_where(lambda x: np.abs(x[1]) < 1e-9, 1)
    # z rigid-body mode: fix u_z at one node (loads are self-equilibrated; LC7 is not z-symmetric)
    n0 = int(np.argmin(np.sum((m.X - np.array([[-R], [0.0], [W / 2]]))**2, axis=0)))
    u3, u7 = m.solve(K, [f3a + f3b, f7a + f7b], [Dsym, Dy, np.array([3 * n0 + 2])])
    out["LC3"] = m.stresses(u3, by=by)
    out["LC3_dD_um"] = float(-2 * m.disp_at(u3, probe)[1, 0] * 1e3)
    out["LC7"] = m.stresses(u7, by=by)
    out["LC7_dD_edge_um"] = float(-2 * m.disp_at(u7, pedge)[1, 0] * 1e3)
    return out


def governing_SF(res):
    worst = {}
    for lc in LCS:
        for mat, r in res[lc].items():
            brittle = "SLA" in mat or "PA12" in mat
            sf = r["strength"] / max(r["s1"] if brittle else r["vm"], 1e-12)
            need = 2.0 if (brittle or "CF" in mat) else 1.5
            if mat not in worst or sf / need < worst[mat][0] / worst[mat][1]:
                worst[mat] = (sf, need, lc)
    return worst


if __name__ == "__main__":
    keys = sys.argv[1:] or list(CONCEPTS)
    allres = {}
    for k in keys:
        t0 = time.time()
        res = run(CONCEPTS[k]())
        res["worst"] = {m: dict(SF=v[0], need=v[1], lc=v[2]) for m, v in governing_SF(res).items()}
        allres[k] = res
        w = "; ".join(f"{m}: SF {v['SF']:.2f} ({v['lc']}) {'OK' if v['SF'] >= v['need'] else 'FAIL'}"
                      for m, v in res["worst"].items())
        print(f"{k:32s} mass {res['mass_g']:5.2f} g  m_eq {res['m_eq_g']:5.2f} g  "
              f"run {res['LC1_um_per_N']:6.2f} um/N  pinch dD {res['LC3_dD_um']:7.1f} um  "
              f"spin dD {res['LC5_dD_um']:5.2f} um | {w}   [{time.time()-t0:.0f}s, {res['ndofs']} dof]", flush=True)
    json.dump(allres, open("concepts_results.json", "w"), indent=1)
