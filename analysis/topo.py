"""
topo: 3D topology optimisation of a wheel. The optimiser decides where material goes; nothing between
the bearing hub and the rim is prescribed.

Domain and fixed parts
  * design domain: the whole annular volume r_bore .. R - t_rim, over the full width W
  * non-design solid: the bearing hub tube (r_bore .. r_hub over the bearing seats) and a thin rim of
    thickness t_rim (R - t_rim .. R, full width)
  * bore r < r_bore is void (axle and bearings)

Discretisation
  * structured hexahedra in (r, theta, z) mapped to Cartesian; incompatible-mode (Wilson/Taylor H11) hex8,
    so the thin rim and thin members bend correctly with 2 elements through their thickness
  * stiffness entries are assembled once into a fixed symmetric (upper) CSR pattern; each iteration only
    rescales element contributions (MKL PARDISO, SPD)

Optimisation (density method, SIMP)
  * variables x in one 2*pi/N sector, replicated round the wheel: cyclic symmetry of order N makes the
    wheel statically balanced (shape inside the sector is free, so spokes can be curved/branched/organic)
  * density filter (radius rf, mm) + Heaviside projection with beta continuation. Robust minimum length
    scale: stiffness is evaluated on the ERODED design (eta_e > 0.5) and the manufactured BLUEPRINT
    (eta = 0.5) carries the mass/inertia constraint, so every blueprint member is thicker than what the
    eroded design needs; the realised minimum thickness is measured after extraction
  * objective: p-norm (smooth max) of the compliances of all load cases, each normalised by its value for
    the initial design; every load case is applied at several angles inside the sector, so the design must
    work wherever the wheel is loaded
  * constraint: rotational inertia of the design material <= cap (directly the m_eq that matters)
  * update: optimality criteria with a single weighted linear constraint, move limit 0.1
Units mm, N, MPa, tonne/mm^3.
"""
from __future__ import annotations
import json, time, sys
from dataclasses import dataclass, field, asdict
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


@dataclass
class Setup:
    R: float = 14.1
    W: float = 17.2
    r_bore: float = 3.0
    r_hub: float = 4.0                       # bearing-holder outer radius (hub wall 1.0)
    seats: tuple = ((6.1, 11.1),)            # bearing seats (fixed bore) and the non-design hub length
    t_rim: float = 0.6                       # thin non-design rim skin = SLS minimum wall; the optimiser thickens
                                             # the rim inwards, locally, wherever the loads need it (rim_zone band)
    rim_zone: float = 2.5                    # radial band inside the rim meshed finer so its thickness can vary
    h_rim: float = 0.2                       # element size in that band
    E: float = 1700.0
    nu: float = 0.40
    rho: float = 930.0                       # kg/m^3
    strength: float = 42.0                   # UTS, MPa (allowable = strength / 2, SF 2)
    material: str = "SLS PA12 (PA2200)"
    N: int = 6                               # cyclic symmetry order
    h: float = 0.45                          # target element size, mm
    rf: float = 1.2                          # filter radius, mm (~2 x minimum wall keeps features >= t_min)
    F: float = 20.0                          # handling load, N
    edge: float = 3.0                        # edge band width, mm
    n_phi: int = 3                           # load angles per sector
    p_norm: float = 6.0
    simp: float = 3.0
    eta_e: float = 0.6                       # eroded threshold (blueprint 0.5)
    Emin: float = 1e-6
    I_cap: float = 0.0                       # g mm^2 of design material; 0 -> from meq_cap or frac
    meq_cap: float = 0.0                     # cap on the design material's m_eq = I / R^2 (g)
    frac: float = 0.25                       # initial density
    iters: int = 160
    lcs: tuple = ("press", "press_e0", "press_eW", "pinch", "pinch_e0", "lip0", "lipW")
    lc_scale: dict = field(default_factory=dict)   # optional per-load-case force factor, e.g. {"lip0": 0.25}


def _axis(breaks, h):
    br = np.unique(np.round(np.asarray(breaks, float), 9))
    out = [br[0]]
    for a, b in zip(br[:-1], br[1:]):
        n = max(1, int(np.ceil((b - a) / h - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


class Wheel:
    def __init__(self, s: Setup):
        self.s = s
        r0 = s.R - s.t_rim - s.rim_zone
        r = _axis([s.r_bore, s.r_hub, r0], s.h)
        rz = _axis([r0, s.R - s.t_rim], s.h_rim)
        rim = np.linspace(s.R - s.t_rim, s.R, 2 if s.t_rim < 0.45 else 3)
        r = np.unique(np.round(np.concatenate([r, rz, rim]), 9))
        z = _axis([0, s.W, s.edge, s.W - s.edge] + [v for sz in s.seats for v in sz], s.h)
        m = max(1, int(round(2 * np.pi * (s.R - s.t_rim) / s.N / (1.0 * s.h))))
        nt = s.N * m
        self.r, self.z, self.nt, self.m = r, z, nt, m
        self.dth = 2 * np.pi / nt
        nr, nz = len(r), len(z)
        self.nr, self.nz = nr, nz
        nid = lambda i, j, k: (k * nt + (j % nt)) * nr + i
        self.nn = nr * nt * nz
        self.ndofs = 3 * self.nn
        I, J, K = np.meshgrid(np.arange(nr - 1), np.arange(nt), np.arange(nz - 1), indexing="ij")
        I, J, K = I.ravel(), J.ravel(), K.ravel()
        self.ei, self.ej, self.ek = I, J, K
        self.ne = len(I)
        conn = np.stack([nid(I, J, K), nid(I + 1, J, K), nid(I + 1, J + 1, K), nid(I, J + 1, K),
                         nid(I, J, K + 1), nid(I + 1, J, K + 1), nid(I + 1, J + 1, K + 1), nid(I, J + 1, K + 1)], 1)
        self.conn = conn
        # node coordinates
        # node id ordering: (k*nt + j)*nr + i
        rr = np.tile(r, nt * nz); tt = np.tile(np.repeat(np.arange(nt) * self.dth, nr), nz)
        zz = np.repeat(z, nr * nt)
        self.X = np.vstack([rr * np.cos(tt), rr * np.sin(tt), zz])
        # element centroids, radii, volumes
        rc = 0.5 * (r[I] + r[I + 1]); zc = 0.5 * (z[K] + z[K + 1]); tc = (J + 0.5) * self.dth
        self.rc, self.zc, self.tc = rc, zc, tc
        self.vol = 0.5 * (r[I + 1]**2 - r[I]**2) * self.dth * (z[K + 1] - z[K])
        self.cent = np.vstack([rc * np.cos(tc), rc * np.sin(tc), zc])
        # non-design sets
        hub = (rc < s.r_hub) & np.any([(zc > a) & (zc < b) for a, b in s.seats], axis=0)
        rimm = rc > s.R - s.t_rim
        self.fixed = hub | rimm
        self.design = ~self.fixed
        # sector map: element -> variable index (cyclic symmetry)
        Js = J % m
        self.var_of = (K * m + Js) * (nr - 1) + I
        self.nvar = (nz - 1) * m * (nr - 1)
        self._element_matrices()
        self._pattern()
        self._filter()
        self._bcs_loads()

    # ---------- element matrices ----------
    def _C(self):
        E, nu = 1.0, self.s.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6)); C[:3, :3] = lam; C[[0, 1, 2], [0, 1, 2]] += 2 * mu; C[3:, 3:] = np.eye(3) * mu
        return C

    def _ke(self, Xe):
        """H11 incompatible-mode hex8 for node coords Xe (8, 3), unit E."""
        C = self._C()
        xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
        g = 1 / np.sqrt(3)
        gps = [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]
        def dN(p):
            a, b, c = p
            return 0.125 * np.stack([xi[:, 0] * (1 + xi[:, 1] * b) * (1 + xi[:, 2] * c),
                                     xi[:, 1] * (1 + xi[:, 0] * a) * (1 + xi[:, 2] * c),
                                     xi[:, 2] * (1 + xi[:, 0] * a) * (1 + xi[:, 1] * b)], 1)   # (8, 3) d/dxi
        J0 = dN((0, 0, 0)).T @ Xe; detJ0 = np.linalg.det(J0); J0i = np.linalg.inv(J0)
        Kuu = np.zeros((24, 24)); Kua = np.zeros((24, 9)); Kaa = np.zeros((9, 9))
        for p in gps:
            d = dN(p); J = d.T @ Xe; detJ = np.linalg.det(J); G = d @ np.linalg.inv(J).T     # (8, 3) d/dx
            B = np.zeros((6, 24))
            B[0, 0::3] = G[:, 0]; B[1, 1::3] = G[:, 1]; B[2, 2::3] = G[:, 2]
            B[3, 1::3] = G[:, 2]; B[3, 2::3] = G[:, 1]; B[4, 0::3] = G[:, 2]; B[4, 2::3] = G[:, 0]
            B[5, 0::3] = G[:, 1]; B[5, 1::3] = G[:, 0]
            # incompatible modes P_i = 1 - xi_i^2 ; dP/dxi = -2 xi_i ; Taylor: use J0 and scale detJ0/detJ
            dP = np.diag([-2 * p[0], -2 * p[1], -2 * p[2]])          # rows: modes, cols: d/dxi
            Gp = (dP @ J0i.T) * (detJ0 / detJ)                          # (3 modes, 3) d/dx
            Ba = np.zeros((6, 9))
            Ba[0, 0::3] = Gp[:, 0]; Ba[1, 1::3] = Gp[:, 1]; Ba[2, 2::3] = Gp[:, 2]
            Ba[3, 1::3] = Gp[:, 2]; Ba[3, 2::3] = Gp[:, 1]; Ba[4, 0::3] = Gp[:, 2]; Ba[4, 2::3] = Gp[:, 0]
            Ba[5, 0::3] = Gp[:, 1]; Ba[5, 1::3] = Gp[:, 0]
            Kuu += B.T @ C @ B * detJ; Kua += B.T @ C @ Ba * detJ; Kaa += Ba.T @ C @ Ba * detJ
        Ke = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
        # stress recovery at the 8 corners (surface points): eps = (B - Ba Kaa^-1 Kau) u, sigma = C eps
        A = np.linalg.solve(Kaa, Kua.T)                                 # alpha = -A u
        Sr = np.zeros((8, 6, 24))
        for a in range(8):
            p = xi[a]
            d = dN(p); J = d.T @ Xe; detJ = np.linalg.det(J); G = d @ np.linalg.inv(J).T
            B = np.zeros((6, 24))
            B[0, 0::3] = G[:, 0]; B[1, 1::3] = G[:, 1]; B[2, 2::3] = G[:, 2]
            B[3, 1::3] = G[:, 2]; B[3, 2::3] = G[:, 1]; B[4, 0::3] = G[:, 2]; B[4, 2::3] = G[:, 0]
            B[5, 0::3] = G[:, 1]; B[5, 1::3] = G[:, 0]
            dP = np.diag([-2 * p[0], -2 * p[1], -2 * p[2]])
            Gp = (dP @ J0i.T) * (detJ0 / detJ)
            Ba = np.zeros((6, 9))
            Ba[0, 0::3] = Gp[:, 0]; Ba[1, 1::3] = Gp[:, 1]; Ba[2, 2::3] = Gp[:, 2]
            Ba[3, 1::3] = Gp[:, 2]; Ba[3, 2::3] = Gp[:, 1]; Ba[4, 0::3] = Gp[:, 2]; Ba[4, 2::3] = Gp[:, 0]
            Ba[5, 0::3] = Gp[:, 1]; Ba[5, 1::3] = Gp[:, 0]
            Sr[a] = C @ (B - Ba @ A)
        return Ke, Sr

    def _element_matrices(self):
        """Ke in the element's local frame (element centred at theta = 0) for each (ir, dz) class."""
        r, z, dt = self.r, self.z, self.dth
        dz = np.round(np.diff(z), 9)
        self.dz_classes, dzc = np.unique(dz, return_inverse=True)
        self.cls = self.ei * len(self.dz_classes) + dzc[self.ek]
        ncls = (len(r) - 1) * len(self.dz_classes)
        self.Ke0 = np.zeros((ncls, 24, 24)); self.Sr = np.zeros((ncls, 8, 6, 24))
        for i in range(len(r) - 1):
            for k, h in enumerate(self.dz_classes):
                pts = []
                for (a, b, c) in [(0, -1, 0), (1, -1, 0), (1, 1, 0), (0, 1, 0), (0, -1, 1), (1, -1, 1), (1, 1, 1), (0, 1, 1)]:
                    rr = r[i + a]; tt = b * dt / 2
                    pts.append((rr * np.cos(tt), rr * np.sin(tt), c * h))
                self.Ke0[i * len(self.dz_classes) + k], self.Sr[i * len(self.dz_classes) + k] = self._ke(np.array(pts))

    # ---------- assembly ----------
    def _pattern(self):
        """Upper-triangle CSR pattern and per-element-entry map (element entries rotated to global)."""
        ne = self.ne
        dofs = (3 * self.conn[:, :, None] + np.arange(3)).reshape(ne, 24)
        self.edofs = dofs
        iu, ju = np.triu_indices(24)
        # global Ke for each element = T Ke0 T^T with T = blockdiag(Rz(tc))
        # store base values of the 24x24 upper-triangular entries after rotation (float64)
        c, s = np.cos(self.tc), np.sin(self.tc)
        Rz = np.zeros((ne, 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1
        self.Rz = Rz
        base = np.empty((ne, len(iu)))
        for k in range(len(self.Ke0)):
            idx = np.nonzero(self.cls == k)[0]
            if len(idx) == 0:
                continue
            T = np.zeros((len(idx), 24, 24))
            for a in range(8):
                T[:, 3 * a:3 * a + 3, 3 * a:3 * a + 3] = Rz[idx]
            Kg = T @ self.Ke0[k] @ T.transpose(0, 2, 1)
            base[idx] = Kg[:, iu, ju]
        rows = dofs[:, iu]; cols = dofs[:, ju]
        lo = np.minimum(rows, cols); hi = np.maximum(rows, cols)
        # diagonal-block entries iu<=ju of local can map to lower global; swap to upper (K symmetric)
        key = lo.astype(np.int64) * self.ndofs + hi
        uk, inv = np.unique(key.ravel(), return_inverse=True)
        self.map = inv.astype(np.int64)
        self.base = base.ravel()
        r_ = (uk // self.ndofs).astype(np.int32); c_ = (uk % self.ndofs).astype(np.int32)
        self.nnz = len(uk)
        A = sp.csr_matrix((np.ones(self.nnz), (r_, c_)), shape=(self.ndofs, self.ndofs))
        A.sort_indices()
        # map unique (sorted by key = row-major) -> csr data order (also row-major sorted) : identical
        self.indptr, self.indices = A.indptr, A.indices
        # off-diagonal local entries counted twice when lo/hi swap? local upper (iu<=ju) covers each global
        # pair once per element except the diagonal 3x3 blocks, which are complete -> fine.
        del key, uk, rows, cols, lo, hi

    def K(self, Ee):
        w = np.repeat(Ee, 300)
        data = np.bincount(self.map, weights=self.base * w, minlength=self.nnz) * self.s.E
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.ndofs, self.ndofs))

    # ---------- filter ----------
    def _filter(self):
        tree = cKDTree(self.cent.T)
        pairs = tree.query_pairs(self.s.rf, output_type="ndarray")
        d = np.linalg.norm(self.cent[:, pairs[:, 0]] - self.cent[:, pairs[:, 1]], axis=0)
        w = self.s.rf - d
        i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(self.ne)])
        j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(self.ne)])
        v = np.concatenate([w, w, np.full(self.ne, self.s.rf)])
        v = v * self.vol[j]
        H = sp.csr_matrix((v, (i, j)), shape=(self.ne, self.ne))
        self.Hs = np.asarray(H.sum(1)).ravel()
        self.H = H

    # ---------- loads / BCs ----------
    def _bcs_loads(self):
        s, X = self.s, self.X
        r = np.hypot(X[0], X[1]); th = np.arctan2(X[1], X[0]); z = X[2]
        seat = (np.abs(r - s.r_bore) < 1e-6) & np.any([(z > a - 1e-6) & (z < b + 1e-6) for a, b in s.seats], axis=0)
        fixed_nodes = np.nonzero(seat)[0]
        self.D = np.sort((3 * fixed_nodes[:, None] + np.arange(3)).ravel())
        self.free = np.setdiff1d(np.arange(self.ndofs), self.D)
        outer = np.abs(r - s.R) < 1e-6
        rimface = r > s.R - s.t_rim - 1e-6
        ramp = lambda d, w=0.6: np.cos(0.5 * np.pi * np.clip((d + w / 2) / w, 0, 1))**2
        cos2 = lambda d, a: np.where(np.abs(d) < a, np.cos(0.5 * np.pi * d / a)**2, 0.0)
        # node "areas" for smooth tractions: use tributary weights (dz * r dtheta) approximately
        zt = np.interp(z, self.z, np.gradient(self.z))
        trib = zt * r * self.dth
        F = []; names = []
        sector = 2 * np.pi / s.N
        for k in range(s.n_phi):
            phi = np.pi / 2 + k * sector / s.n_phi
            dth = lambda c: np.angle(np.exp(1j * (th - c)))
            for lc in s.lcs:
                f = np.zeros((3, self.nn))
                def radial(c, band, sign=1.0):
                    w = outer * cos2(dth(c), np.deg2rad(12)) * trib
                    if band == "e0":
                        w = w * ramp(z - s.edge)
                    elif band == "eW":
                        w = w * ramp(s.W - s.edge - z)
                    w = w / w.sum() * s.F
                    f[0] -= sign * w * np.cos(c); f[1] -= sign * w * np.sin(c)
                if lc == "press":
                    radial(phi, None)
                elif lc == "press_e0":
                    radial(phi, "e0")
                elif lc == "press_eW":
                    radial(phi, "eW")
                elif lc == "pinch":
                    radial(phi, None); radial(phi + np.pi, None)
                elif lc == "pinch_e0":
                    radial(phi, "e0"); radial(phi + np.pi, "e0")
                elif lc in ("lip0", "lipW"):
                    zz = 0.0 if lc == "lip0" else s.W
                    w = (np.abs(z - zz) < 1e-6) & rimface
                    w = w * cos2(dth(phi), np.deg2rad(16)) * (r * self.dth)
                    w = w / w.sum() * s.F
                    f[2] += w if lc == "lip0" else -w
                F.append(f.T.ravel() * s.lc_scale.get(lc, 1.0)); names.append(f"{lc}@{np.degrees(phi - np.pi / 2):.0f}")
        self.F = np.column_stack(F)
        self.lc_names = names

    # ---------- physics ----------
    def solve(self, Ee, keep=False, F=None):
        F = self.F if F is None else F
        K = self.K(Ee)
        Kf = K[self.free][:, self.free]
        try:
            import pypardiso
        except ImportError:                       # no MKL (e.g. Apple Silicon): SciPy SuperLU on the full matrix
            Kfull = (Kf + sp.triu(Kf, 1).T).tocsc()
            lu = __import__("scipy.sparse.linalg", fromlist=["splu"]).splu(Kfull)
            U = np.zeros((self.ndofs, F.shape[1])); U[self.free] = lu.solve(np.ascontiguousarray(F[self.free]))
            if keep:
                def again(B):
                    X = np.zeros_like(B); X[self.free] = lu.solve(np.ascontiguousarray(B[self.free])); return X
                return U, again
            return U
        Kf = sp.triu(Kf, format="csr"); Kf.sort_indices()
        solver = pypardiso.PyPardisoSolver(mtype=2)
        solver.factorize(Kf)
        U = np.zeros((self.ndofs, F.shape[1]))
        U[self.free] = solver.solve(Kf, np.ascontiguousarray(F[self.free])).reshape(len(self.free), -1)
        if keep:
            def again(B):
                X = np.zeros_like(B)
                X[self.free] = solver.solve(Kf, np.ascontiguousarray(B[self.free])).reshape(len(self.free), -1)
                solver.free_memory(everything=True)
                return X
            return U, again
        solver.free_memory(everything=True)
        return U

    # ---------- surface stresses (solid-material stress at the 8 corners of every element) ----------
    def _local(self, U, idx):
        ue = U[self.edofs[idx]].reshape(len(idx), 8, 3, -1)
        return np.einsum('eji,enjl->enil', self.Rz[idx], ue).reshape(len(idx), 24, -1)

    def corner_vm(self, U):
        """von Mises (solid-material, i.e. the stress the printed part sees) at the element corners:
        (ne, 8, nlc), plus max principal for reporting."""
        vm = np.zeros((self.ne, 8, U.shape[1])); s1 = np.zeros_like(vm)
        for k in range(len(self.Ke0)):
            idx = np.nonzero(self.cls == k)[0]
            if len(idx) == 0:
                continue
            S = np.einsum('aib,ebl->eail', self.Sr[k], self._local(U, idx)) * self.s.E     # (n, 8, 6, nlc)
            sx, sy, sz, tyz, txz, txy = (S[:, :, i] for i in range(6))
            vm[idx] = np.sqrt(np.maximum(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3 * (tyz**2 + txz**2 + txy**2), 1e-30))
            T = np.stack([np.stack([sx, txy, txz], -1), np.stack([txy, sy, tyz], -1), np.stack([txz, tyz, sz], -1)], -2)
            s1[idx] = np.linalg.eigvalsh(T)[..., -1]
        return vm, s1

    def vm_adjoint_rhs(self, U, coef):
        """sum over elements/corners/load cases of coef * d vM / d u  -> (ndofs, nlc); coef (ne, 8, nlc)."""
        out = np.zeros_like(U)
        V = np.array([[1, -.5, -.5, 0, 0, 0], [-.5, 1, -.5, 0, 0, 0], [-.5, -.5, 1, 0, 0, 0],
                      [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3]], float)
        for k in range(len(self.Ke0)):
            idx = np.nonzero(self.cls == k)[0]
            if len(idx) == 0:
                continue
            S = np.einsum('aib,ebl->eail', self.Sr[k], self._local(U, idx)) * self.s.E
            VS = np.einsum('ij,eajl->eail', V, S)
            vm = np.sqrt(np.maximum(np.einsum('eail,eail->eal', S, VS), 1e-30))
            g = VS * (coef[idx] / vm)[:, :, None, :]                     # d/dsigma
            gl = np.einsum('eail,aib->ebl', g, self.Sr[k]) * self.s.E    # d/du_local (n, 24, nlc)
            gl = gl.reshape(len(idx), 8, 3, -1)
            gg = np.einsum('eij,enjl->enil', self.Rz[idx], gl).reshape(len(idx), 24, -1)
            np.add.at(out, self.edofs[idx], gg)
        return out

    def element_cross(self, A, U):
        """a_e^T Ke0 u_e (unit E) per element and load case."""
        out = np.zeros((self.ne, U.shape[1]))
        for k in range(len(self.Ke0)):
            idx = np.nonzero(self.cls == k)[0]
            if len(idx) == 0:
                continue
            out[idx] = np.einsum('eal,ab,ebl->el', self._local(A, idx), self.Ke0[k], self._local(U, idx))
        return out * self.s.E

    def element_energy(self, U):
        """u_e^T Ke0 u_e (unit E) per element and load case: (ne, nlc)."""
        out = np.zeros((self.ne, U.shape[1]))
        for k in range(len(self.Ke0)):
            idx = np.nonzero(self.cls == k)[0]
            if len(idx) == 0:
                continue
            ue = U[self.edofs[idx]]                                    # (n, 24, nlc)
            # rotate to local frame
            ue = ue.reshape(len(idx), 8, 3, -1)
            ul = np.einsum('eji,enjl->enil', self.Rz[idx], ue).reshape(len(idx), 24, -1)
            out[idx] = np.einsum('eal,ab,ebl->el', ul, self.Ke0[k], ul)
        return out * self.s.E

    # ---------- density chain ----------
    def densities(self, x, beta):
        xf = x[self.var_of]
        xf = np.where(self.fixed, 1.0, xf)
        rt = (self.H @ xf) / self.Hs
        def proj(eta):
            a, b = np.tanh(beta * eta), np.tanh(beta * (1 - eta))
            p = (a + np.tanh(beta * (rt - eta))) / (a + b)
            dp = beta * (1 - np.tanh(beta * (rt - eta))**2) / (a + b)
            return p, dp
        pe, dpe = proj(self.s.eta_e)
        pb, dpb = proj(0.5)
        pe = np.where(self.fixed, 1.0, pe); pb = np.where(self.fixed, 1.0, pb)
        dpe = np.where(self.fixed, 0.0, dpe); dpb = np.where(self.fixed, 0.0, dpb)
        return pe, dpe, pb, dpb

    def back(self, g_rho, dp):
        """d/dx of a function with gradient g_rho wrt projected element density."""
        g = (self.H.T @ (g_rho * dp / self.Hs))
        g = np.where(self.fixed, 0.0, g)
        return np.bincount(self.var_of, weights=g, minlength=self.nvar)

    def inertia_w(self):
        return self.vol * self.rc**2 * self.s.rho * 1e-6        # g mm^2 per unit density

    def mass_props(self, rho_e):
        m = (rho_e * self.vol).sum() * self.s.rho * 1e-6
        I = (rho_e * self.inertia_w()).sum()
        return m, I


def optimise(s: Setup, out="topo_result.npz", log=print):
    t0 = time.time()
    w = Wheel(s)
    log(f"grid: nr {w.nr} nt {w.nt} nz {w.nz} -> {w.ne} elements, {w.ndofs} dof, {w.nvar} variables, "
        f"{w.nnz/1e6:.1f}M nnz, {len(w.lc_names)} load cases; setup {time.time()-t0:.0f}s")
    wI = w.inertia_w()
    I_fixed = (wI * w.fixed).sum()
    if s.I_cap <= 0:
        s.I_cap = s.meq_cap * s.R**2 if s.meq_cap > 0 else (wI * w.design).sum() * s.frac
    s.frac = min(0.9, s.I_cap / (wI * w.design).sum())
    log(f"fixed hub+rim m_eq {I_fixed / s.R**2:.3f} g; cap on designed material m_eq {s.I_cap / s.R**2:.3f} g "
        f"(start density {s.frac:.3f})")
    x = np.full(w.nvar, s.frac)
    beta = 1.0; c0 = None; hist = []
    for it in range(s.iters):
        if it > 0 and it % 20 == 0 and beta < 16:
            beta *= 2
        pe, dpe, pb, dpb = w.densities(x, beta)
        Ee = s.Emin + pe**s.simp * (1 - s.Emin)
        t1 = time.time()
        U = w.solve(Ee)
        c = np.einsum('ij,ij->j', w.F, U)
        if c0 is None:
            c0 = c.copy()
        cn = c / c0
        J = (cn**s.p_norm).sum()**(1 / s.p_norm)
        dJdc = (cn**(s.p_norm - 1)) / J**(s.p_norm - 1) / c0
        en = w.element_energy(U)                                         # (ne, nlc)
        dc_dE = -en                                                      # dc_i/dE_e
        dJ_dpe = (dc_dE @ dJdc) * s.simp * pe**(s.simp - 1) * (1 - s.Emin)
        gJ = w.back(dJ_dpe, dpe)
        gI = w.back(wI * w.design, dpb)
        I_design = (pb * wI * w.design).sum()
        # OC update
        l1, l2, move = 1e-12, 1e12, 0.1
        gJn = np.minimum(gJ, -1e-30)
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(x * np.sqrt(-gJn / (lm * np.maximum(gI, 1e-30))), np.maximum(0, x - move), np.minimum(1, x + move))
            pe_, _, pb_, _ = w.densities(xn, beta)
            if (pb_ * wI * w.design).sum() > s.I_cap:
                l1 = lm
            else:
                l2 = lm
        change = np.abs(xn - x).max(); x = xn
        m_b, I_b = w.mass_props(pb)
        grey = 4 * (pb * (1 - pb) * w.vol).sum() / w.vol.sum()
        worst = w.lc_names[int(np.argmax(cn))]
        hist.append(dict(it=it, J=J, beta=beta, I=I_b, m=m_b, change=change, grey=grey, worst=worst))
        log(f"it {it:3d} beta {beta:4.0f} J {J:8.4f} (worst {worst} {cn.max():.3f}) m_eq {I_b / s.R**2:.3f} g "
            f"mass {m_b:.3f} g grey {grey:.3f} change {change:.3f} [{time.time()-t1:.0f}s]")
        if beta >= 16 and change < 0.01 and it > 60:
            break
    pe, dpe, pb, dpb = w.densities(x, beta)
    np.savez_compressed(out, x=x, rho=pb, rho_e=pe, beta=beta, setup=json.dumps(asdict(s), default=list),
                        r=w.r, z=w.z, nt=w.nt, hist=json.dumps(hist))
    return w, x, pb, hist


def mma_step(x, df0, g, dg, xmin, xmax, L, U, x1, x2, it, move=0.1):
    """One MMA step (Svanberg 1987) for min f0(x) s.t. g(x) <= 0 with a single constraint; the convex
    separable subproblem is solved through its 1-D dual (bisection on the multiplier)."""
    rng = xmax - xmin
    if it < 2 or L is None:
        L = x - 0.2 * rng; U = x + 0.2 * rng
    else:
        sgn = (x - x1) * (x1 - x2)
        gam = np.where(sgn > 0, 1.2, np.where(sgn < 0, 0.7, 1.0))
        L = x - gam * (x1 - L); U = x + gam * (U - x1)
        L = np.clip(L, x - 10 * rng, x - 0.01 * rng); U = np.clip(U, x + 0.01 * rng, x + 10 * rng)
    alpha = np.maximum.reduce([xmin, L + 0.1 * (x - L), x - move])
    beta = np.minimum.reduce([xmax, U - 0.1 * (U - x), x + move])
    ux, xl = U - x, x - L
    eps = 1e-5 / rng
    p0 = ux**2 * (1.001 * np.maximum(df0, 0) + 0.001 * np.maximum(-df0, 0) + eps)
    q0 = xl**2 * (0.001 * np.maximum(df0, 0) + 1.001 * np.maximum(-df0, 0) + eps)
    p1 = ux**2 * (1.001 * np.maximum(dg, 0) + 0.001 * np.maximum(-dg, 0) + eps)
    q1 = xl**2 * (0.001 * np.maximum(dg, 0) + 1.001 * np.maximum(-dg, 0) + eps)
    b = (p1 / ux + q1 / xl).sum() - g
    def xof(lam):
        P, Q = p0 + lam * p1, q0 + lam * q1
        sp_, sq = np.sqrt(P), np.sqrt(Q)
        return np.clip((sp_ * L + sq * U) / (sp_ + sq), alpha, beta)
    def gof(lam):
        xx = xof(lam)
        return (p1 / (U - xx) + q1 / (xx - L)).sum() - b
    lo, hi = 0.0, 1.0
    if gof(0.0) <= 0:
        lam = 0.0
    else:
        while gof(hi) > 0 and hi < 1e12:
            hi *= 10
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gof(mid) > 0 else (lo, mid)
        lam = hi
    return xof(lam), L, U


def _mma_state(L, U, x1, x2, cn):
    """MMA asymptotes, the two previous iterates and the stress normalisation, saved in every checkpoint
    so that a resumed run (e.g. the next 6 h GitHub job) continues exactly where it stopped."""
    if L is None:
        return {}
    return dict(mma_L=L, mma_U=U, mma_x1=x1, mma_x2=x2, cn=cn)


def warm_start(w, npz):
    """Design variables of Wheel w interpolated from the density field of an earlier result (other grid,
    rim thickness or material allowed): trilinear in (r, theta, z), periodic in theta."""
    from scipy.interpolate import RegularGridInterpolator
    d = np.load(npz, allow_pickle=True)
    r, z, nt = d["r"], d["z"], int(d["nt"])
    rho = d["rho"].reshape(len(r) - 1, nt, len(z) - 1)
    rc = 0.5 * (r[1:] + r[:-1]); zc = 0.5 * (z[1:] + z[:-1]); dth = 2 * np.pi / nt
    tc = (np.arange(nt) + 0.5) * dth
    A = np.concatenate([rho[:, -1:], rho, rho[:, :1]], 1)
    tcp = np.concatenate([[tc[0] - dth], tc, [tc[-1] + dth]])
    f = RegularGridInterpolator((rc, tcp, zc), A, bounds_error=False, fill_value=None)
    first = w.ej < w.m                                                    # one sector holds every variable
    pts = np.column_stack([w.rc[first], w.tc[first] % (2 * np.pi), w.zc[first]])
    x = np.zeros(w.nvar)
    x[w.var_of[first]] = np.clip(f(pts), 0.0, 1.0)
    return x


def optimise_inertia(s: Setup, out="topo_inertia.npz", log=print, sigma_target=None, P=10.0, q_relax=0.5,
                     init_from=None, beta0=1.0):
    """Minimum rotational inertia subject to the stress limit under all load cases (and the minimum
    length scale). Constraint: adaptive-normalised P-norm of the relaxed surface von Mises stress of the
    eroded design over all elements, corners and load cases <= sigma_target.
    init_from: density npz of an earlier result (any grid) used as the starting design of a fresh run;
    beta0: projection sharpness to start from (use ~8 for a warm start so it is not blurred)."""
    t0 = time.time()
    w = Wheel(s)
    allow = s.strength / 2.0
    sigma_target = sigma_target or 0.9 * allow
    log(f"grid: nr {w.nr} nt {w.nt} nz {w.nz} -> {w.ne} elements, {w.ndofs} dof, {w.nvar} variables, "
        f"{len(w.lc_names)} load cases; setup {time.time()-t0:.0f}s; stress target {sigma_target:.1f} MPa (vM)")
    wI = w.inertia_w()
    I_fixed = (wI * w.fixed).sum()
    I_full = (wI * w.design).sum()
    # points excluded from the stress measure: within 0.5 mm of the clamped bearing seat (BC singularity)
    excl = (w.rc < s.r_bore + 0.5) & np.any([(w.zc > a - 0.5) & (w.zc < b + 0.5) for a, b in s.seats], axis=0)
    x = np.full(w.nvar, 0.5)
    xmin, xmax = np.zeros(w.nvar), np.ones(w.nvar)
    L = U_ = x1 = x2 = None
    beta, cn, hist, it0 = float(beta0), 1.0, [], 0
    import os
    if init_from and not os.path.exists(out):                            # warm start from another grid
        x = warm_start(w, init_from)
        log(f"warm start from {init_from}: design variables mapped, mean {x.mean():.3f}; beta starts at {beta:g}")
    if os.path.exists(out):                                              # resume an interrupted run
        d = np.load(out, allow_pickle=True)
        if len(d["x"]) == w.nvar:
            x, beta, hist = d["x"], float(d["beta"]), json.loads(str(d["hist"]))
            it0 = len(hist)
            if "mma_L" in d:                    # full optimiser state: the resumed run continues exactly
                L, U_, x1, x2, cn = d["mma_L"], d["mma_U"], d["mma_x1"], d["mma_x2"], float(d["cn"])
                log(f"resuming from iteration {it0} (beta {beta}), optimiser state restored")
            else:                               # old checkpoint without MMA state: restart the asymptotes
                L, U_ = x - 0.02, x + 0.02     # tight (= conservative), not the wide initial 0.2 that let
                x1 = x2 = x.copy()             # resumed runs cut a load path (r07: 23 -> 891 MPa)
                cn = None
                log(f"resuming from iteration {it0} (beta {beta}), old checkpoint: tight MMA restart")
    for it in range(it0, s.iters):
        if it > 0 and it % 25 == 0 and beta < 16:   # continuation (a warm start begins at beta0)
            beta *= 2
        pe, dpe, pb, dpb = w.densities(x, beta)
        Ee = s.Emin + pe**s.simp * (1 - s.Emin)
        t1 = time.time()
        Uu, again = w.solve(Ee, keep=True)
        vm, s1 = w.corner_vm(Uu)                                          # solid stress (ne, 8, nlc)
        pq = np.maximum(pe, 1e-3)
        rel = (pq**q_relax)[:, None, None] * vm
        rel[excl] = 0.0
        true_max = float(rel.max())
        pn = float(((rel / allow)**P).sum()**(1 / P)) * allow
        cn = 0.5 * cn + 0.5 * (true_max / pn) if (it > 0 and cn is not None) else true_max / pn  # adaptive normalisation
        G = cn * pn
        # sensitivities of G: d pn/d rel = (rel/pn)^(P-1)
        dpn = np.where(rel > 0, (rel / pn)**(P - 1), 0.0) * cn
        coef = dpn * (pq**q_relax)[:, None, None]                         # d G / d vm (explicit u-dependence)
        coef[excl] = 0.0
        lam = again(w.vm_adjoint_rhs(Uu, coef))                          # K lambda = dG/du
        dG_dE = -w.element_cross(lam, Uu).sum(1)                          # implicit via K(E)
        dG_dpe = dG_dE * s.simp * pe**(s.simp - 1) * (1 - s.Emin)
        dG_dpe += (dpn * vm).sum((1, 2)) * q_relax * pq**(q_relax - 1) * (~excl) * (pe > 1e-3)
        gG = w.back(dG_dpe, dpe) / sigma_target
        Iw = (pb * wI * w.design).sum()
        gI = w.back(wI * w.design, dpb) / I_full
        g = G / sigma_target - 1.0
        xn, L, U_ = mma_step(x, gI, g, gG, xmin, xmax, L, U_, x1 if x1 is not None else x,
                             x2 if x2 is not None else x, it, move=0.03 if beta < 8 else 0.02)
        x2, x1 = x1, x.copy()
        change = np.abs(xn - x).max(); x = xn
        m_b, I_b = w.mass_props(pb)
        grey = 4 * (pb * (1 - pb) * w.vol * w.design).sum() / (w.vol * w.design).sum()
        jw = np.unravel_index(np.argmax(rel), rel.shape)
        hist.append(dict(it=it, beta=beta, meq=(I_b) / s.R**2, mass=m_b, smax=true_max, G=G, grey=grey,
                         worst=w.lc_names[jw[2]]))
        log(f"it {it:3d} beta {beta:4.0f} m_eq {I_b / s.R**2:.3f} g (design {Iw / s.R**2:.3f}) mass {m_b:.3f} g | "
            f"max vM {true_max:5.1f} MPa ({w.lc_names[jw[2]]}, r {w.rc[jw[0]]:.1f} z {w.zc[jw[0]]:.1f}) "
            f"P-norm {G:5.1f} | grey {grey:.3f} change {change:.3f} [{time.time()-t1:.0f}s]")
        if beta >= 16 and len(hist) > 20:                               # converged: m_eq flat, stress on target
            m10 = [h["meq"] for h in hist[-10:]]
            if (max(m10) - min(m10)) / m10[-1] < 0.002 and abs(true_max / sigma_target - 1) < 0.05:
                log("converged"); break
        if it % 5 == 0:
            np.savez_compressed(out, x=x, rho=pb, rho_e=pe, beta=beta, setup=json.dumps(asdict(s), default=list),
                                r=w.r, z=w.z, nt=w.nt, hist=json.dumps(hist),
                                **_mma_state(L, U_, x1, x2, cn))
    pe, dpe, pb, dpb = w.densities(x, beta)
    np.savez_compressed(out, x=x, rho=pb, rho_e=pe, beta=beta, setup=json.dumps(asdict(s), default=list),
                        r=w.r, z=w.z, nt=w.nt, hist=json.dumps(hist),
                                **_mma_state(L, U_, x1, x2, cn))
    return w, x, pb, hist


if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = sys.argv[2] if len(sys.argv) > 2 else "topo_inertia.npz"
    mode = kw.pop("mode", "inertia")
    st = kw.pop("sigma_target", None)
    if mode == "compliance":
        optimise(Setup(**kw), out)
    else:
        optimise_inertia(Setup(**kw), out, sigma_target=st)
