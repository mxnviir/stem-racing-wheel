"""
Full-accuracy check of an optimised (organic) wheel from its STL.

* the STL is re-meshed by gmsh (surface classification + reparametrisation, then quadratic tetrahedra,
  fem3d P2 solver), full 360 deg model: the organic layout has no mirror symmetry
* load cases (20 N handling loads unless stated), each applied at n_ang angles across one symmetry sector
  so every relative position of loads and spokes is covered:
    press / press_e0 / press_eW   radial at the contact, full width / 3 mm band at either rim edge
    pinch / pinch_e0              diametral pinch, full width / at the z = 0 edge
    lip0 / lipW                   axial push on the rim end faces (outer 1 mm, +-16 deg)
    thumb                         axial push, dia-8 mm at r = 8.5 on the outermost surface under the thumb
                                  (any material within 2 mm of the outer plane; deeper material is not reached)
    spin                          20 000 rpm
    run                           1 N radial (stiffness only)
  bearing seats clamped (points within 0.5 mm of the clamped seat excluded from the stress check)
* criteria: max(s1, vM) <= UTS / 2 (SF 2), linear buckling load factor >= 2 for the worst load cases
"""
import sys, json, time
import numpy as np
from fem3d import Model3D, _vm_s1
from wheelfem import Material

PA12_SLS = Material("SLS PA12 (PA2200)", 1700, 0.40, 930, 42)
ALLOW = PA12_SLS.strength / 2
BEAR_M, BEAR_I = 2 * 0.32, 2 * 0.18 * 2.7**2


def mesh_stl(stl, h=0.6, h_min=0.25, angle=40):
    """Tet mesh of a closed STL. First try: gmsh reparametrises the surface and remeshes it at size h.
    Fallback (complex topology): decimate the STL to ~h triangles and fill the volume from them directly."""
    import gmsh
    try:
        return _mesh_stl(stl, h, h_min, angle, reparam=True)
    except Exception as e:
        print(f"    reparametrisation failed ({e}); using the decimated discrete surface", flush=True)
        try:
            gmsh.finalize()
        except Exception:
            pass
        import trimesh, tempfile, os
        m = trimesh.load(stl)
        target = int(max(2000, 2.3 * m.area / (0.433 * h**2)))     # ~equilateral triangles of edge h
        if len(m.faces) > target:
            m = m.simplify_quadric_decimation(face_count=target)
        f = os.path.join(tempfile.mkdtemp(), "dec.stl"); m.export(f)
        return _mesh_stl(f, h, h_min, angle, reparam=False)


def _mesh_stl(stl, h, h_min, angle, reparam):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.merge(stl)
    if reparam:
        gmsh.model.mesh.classifySurfaces(angle * np.pi / 180, True, True, np.pi)
        gmsh.model.mesh.createGeometry()
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    for alg in (10, 1, 4):                       # HXT, Delaunay, Frontal
        try:
            gmsh.option.setNumber("Mesh.Algorithm3D", alg)
            gmsh.model.mesh.generate(3)
            break
        except Exception:
            gmsh.model.mesh.clear()
            if alg == 4:
                gmsh.finalize(); raise
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, en = gmsh.model.mesh.getElements(3)
    T = np.array(en[list(et).index(4)], np.int64).reshape(-1, 4)
    idx = np.full(int(nt.max()) + 1, -1); idx[nt.astype(int)] = np.arange(len(nt))
    P = xyz.reshape(-1, 3).T; T = idx[T].T
    u = np.unique(T); rm = np.full(P.shape[1], -1); rm[u] = np.arange(len(u))
    gmsh.finalize()
    return P[:, u].copy(), rm[T].copy()


def loads(m, s, n_ang=6):
    R, W, N = s["R"], s["W"], s["N"]
    X = m.X
    th_ = lambda x: np.arctan2(x[1], x[0])
    r_ = lambda x: np.hypot(x[0], x[1])
    d_ = lambda x, c: np.angle(np.exp(1j * (th_(x) - c)))
    cos2 = lambda d, a: np.where(np.abs(d) < a, np.cos(0.5 * np.pi * d / a)**2, 0.0)
    ramp = lambda d, w=0.5: np.cos(0.5 * np.pi * np.clip((d + w / 2) / w, 0, 1))**2
    outer = lambda x: r_(x) > R - 0.06
    F, names = [], []
    sector = 2 * np.pi / N
    bf_all = m.mesh.boundary_facets()
    mid_all = m.mesh.p[:, m.mesh.facets[:, bf_all]].mean(axis=1)
    nrm_all = m.facet_normals(bf_all)
    for k in range(n_ang):
        phi = np.pi / 2 + k * sector / n_ang
        tag = f"@{np.degrees(phi - np.pi / 2):.0f}"
        def radial(c, band=None, F_=s["F"]):
            sel = outer(mid_all) & (np.abs(d_(mid_all, c)) < np.deg2rad(14))
            if band == "e0":
                sel &= mid_all[2] < s["edge"] + 0.6
            if band == "eW":
                sel &= mid_all[2] > W - s["edge"] - 0.6
            wf = lambda x: cos2(d_(x, c), np.deg2rad(12)) * (1.0 if band is None else
                                                             (ramp(x[2] - s["edge"]) if band == "e0" else ramp(W - s["edge"] - x[2])))
            dirn = lambda x: np.array([-np.cos(c) + 0 * x[0], -np.sin(c) + 0 * x[0], 0 * x[0]])
            return m.traction_weighted(bf_all[sel], wf, dirn, F_)[0]
        want = set(s.get("lcs", [])) | set(s.get("verify_extra", []))
        F.append(radial(phi, F_=1.0)); names.append("run" + tag)
        for nm, f in (("press", lambda: radial(phi)), ("press_e0", lambda: radial(phi, "e0")),
                      ("press_eW", lambda: radial(phi, "eW")), ("pinch", lambda: radial(phi) + radial(phi + np.pi)),
                      ("pinch_e0", lambda: radial(phi, "e0") + radial(phi + np.pi, "e0"))):
            if nm in want:
                F.append(f()); names.append(nm + tag)
        for zz, sg, nm in ((0.0, 1, "lip0"), (W, -1, "lipW")):
            if nm not in want:
                continue
            sel = (np.abs(mid_all[2] - zz) < 0.05) & (np.abs(nrm_all[2]) > 0.9) & (r_(mid_all) > R - s["t_rim"] - 0.1) & (np.abs(d_(mid_all, phi)) < np.deg2rad(18))
            wf = lambda x, c=phi: cos2(d_(x, c), np.deg2rad(16))
            F.append(m.traction_weighted(bf_all[sel], wf, lambda x, g=sg: np.array([0 * x[0], 0 * x[0], g + 0 * x[0]]), s["F"])[0])
            names.append(nm + tag)
        # thumb: outermost -z facing surface within 2 mm of the outer plane under a dia-8 circle at r 8.5
        cx, cy = 8.5 * np.cos(phi), 8.5 * np.sin(phi)
        rc = lambda x: np.hypot(x[0] - cx, x[1] - cy)
        sel = (rc(mid_all) < 4.3) & (nrm_all[2] < -0.9) & (mid_all[2] < 2.0) & (r_(mid_all) < R - s["t_rim"] - 0.05)
        if "thumb" in want and sel.sum() > 3:
            wf = lambda x: ramp(rc(x) - 4.0, 0.5)
            F.append(m.traction_weighted(bf_all[sel], wf, lambda x: np.array([0 * x[0], 0 * x[0], 1 + 0 * x[0]]), s["F"])[0])
            names.append("thumb" + tag)
    F.append(m.centrifugal(20000 * 2 * np.pi / 60)); names.append("spin")
    seat = lambda x: (r_(x) < s["r_bore"] + 0.03) & np.any([(x[2] > a - 0.03) & (x[2] < b + 0.03) for a, b in s["seats"]], axis=0)
    D = [m.dofs_where(seat, [0, 1, 2])]
    return F, names, D


def run(stl, setup, h=0.6, h_min=0.25, n_ang=6, buckle=3, log=print, modes=True):
    s = setup
    t0 = time.time()
    P, T = mesh_stl(stl, h, h_min)
    m = Model3D(P, T, PA12_SLS, extra_mass_g=BEAR_M, extra_Izz=BEAR_I)
    mp = m.mass_props(half=False)
    log(f"mesh: {P.shape[1]} nodes, {T.shape[1]} tets, {m.ndofs} dof [{time.time()-t0:.0f}s]; "
        f"mass {mp['mass_g']:.3f} g, m_eq {mp['Izz_gmm2'] / s['R']**2:.3f} g (incl. bearings)")
    K = m.stiffness()
    F, names, D = loads(m, s, n_ang)
    U = m.solve(K, F, D)
    r = np.hypot(m.X[0], m.X[1]); z = m.X[2]
    excl = (r < s["r_bore"] + 0.5) & np.any([(z > a - 0.5) & (z < b + 0.5) for a, b in s["seats"]], axis=0)
    ok_pts = ~excl
    res = dict(mass_g=mp["mass_g"], m_eq_g=mp["Izz_gmm2"] / s["R"]**2, ndofs=m.ndofs, cases={})
    worst = []
    for nm, u in zip(names, U):
        if nm.startswith("run"):
            ph = np.pi / 2 + np.deg2rad(float(nm.split("@")[1]))          # probe under the load
            top = np.argmin(np.sum((m.X - np.array([[s["R"] * np.cos(ph)], [s["R"] * np.sin(ph)], [s["W"] / 2]]))**2, 0))
            res["cases"][nm] = dict(um_per_N=float(np.linalg.norm(u[3 * top:3 * top + 3]) * 1e3))
            continue
        vm, s1, _ = _vm_s1(m.nodal_stress(u))
        q = np.maximum(s1, vm) / ALLOW
        q[~ok_pts] = 0
        i = int(np.argmax(q))
        res["cases"][nm] = dict(q=float(q[i]), s1=float(s1[i]), vm=float(vm[i]), at=m.X[:, i].round(2).tolist(),
                                umax_um=float(np.abs(u).max() * 1e3))
        worst.append((q[i], nm))
    worst.sort(reverse=True)
    res["q_max"], res["governing"] = float(worst[0][0]), worst[0][1]
    log(f"stress: q_max {worst[0][0]:.3f} ({worst[0][1]}); next: " + ", ".join(f"{n} {q:.2f}" for q, n in worst[1:6]))
    run_c = [v["um_per_N"] for k, v in res["cases"].items() if k.startswith("run")]
    log(f"running stiffness {min(run_c):.2f}..{max(run_c):.2f} um/N over the sector (spoke-passing ripple "
        f"{(max(run_c) - min(run_c)) / np.mean(run_c) * 100:.0f} %)")
    res["run_um_per_N"] = [min(run_c), max(run_c)]
    if buckle:
        lam = []
        for q, nm in worst[:buckle]:
            if nm == "spin":
                continue
            KG = m.geometric_stiffness(U[names.index(nm)])
            l, _ = m.eigs(K, KG, D, k=4, mode="buckling")
            lam.append((float(l[0]) if len(l) else float("inf"), nm))
            log(f"buckling {nm}: lambda_1 = {lam[-1][0]:.2f}")
            del KG
        res["buckling"] = lam
    if modes:
        f, _ = m.eigs(K, m.mass_matrix(), D, k=6, mode="modes")
        res["modes_Hz"] = sorted(np.abs(f).tolist())
        log(f"natural frequencies (Hz): {', '.join(f'{x:.0f}' for x in res['modes_Hz'])} (rotation 333 Hz)")
    log(f"done [{time.time()-t0:.0f}s]")
    return res


if __name__ == "__main__":
    stl, npz = sys.argv[1], sys.argv[2]
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    d = np.load(npz, allow_pickle=True)
    s = json.loads(str(d["setup"]))
    res = run(stl, s, h=h)
    json.dump(res, open(stl.replace(".stl", "_verify.json"), "w"), indent=1, default=float)
