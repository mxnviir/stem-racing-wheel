"""
wheelfem — 3D linear-elastic FEA of axisymmetric multi-material wheels under
non-axisymmetric loads. Geometry/quadrature from scikit-fem, assembly vectorised (numpy BLAS),
direct solve with MKL PARDISO.

Method
------
* The wheel cross-section is a union of rectangles in (r, z), each with a material.
* A tensor grid in (r, theta, z) is generated whose lines include every block edge;
  elements whose centroid lies in void are removed.
* The linear hex mesh is promoted to 27-node triquadratic hexes (MeshHex2) and every
  node is mapped (r, theta, z) -> (r cos t, r sin t, z): curved geometry is represented
  isoparametrically; one quadratic element through a thin wall captures bending
  (verified in validate.py: thin-ring pinch within 1-2 %, rotating disc within 0.02 %).
* Half model: theta in [90 deg, 270 deg] (x <= 0), symmetry u_x = 0 on the x = 0 plane.
  Loads applied in the symmetry plane at theta = 90 deg (top) / 270 deg (bottom).
* Full 3x3x3 Gauss integration (no reduced-integration hourglassing).

Units: mm, N, MPa; density given in kg/m^3 and converted (tonne/mm^3 = kg/m^3 * 1e-12).
Global dof numbering: dof = 3 * node + component.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from skfem import MeshHex, MeshHex2, Basis, FacetBasis, ElementHex2, LinearForm, Functional, asm


@dataclass
class Material:
    name: str
    E: float        # MPa
    nu: float
    rho: float      # kg/m^3
    strength: float # MPa (yield for metals, UTS for brittle polymers)

    @property
    def rho_t(self):
        return self.rho * 1e-12

    def C(self):
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[[0, 1, 2], [0, 1, 2]] += 2 * mu
        C[[3, 4, 5], [3, 4, 5]] = mu
        return C


@dataclass
class Block:
    r0: float; r1: float; z0: float; z1: float; mat: int


@dataclass
class Design:
    name: str
    blocks: list
    materials: list
    R: float                 # outer (rolling) radius, mm
    contact: tuple           # (z0, z1) of the flat contact band
    r_bore: float            # bearing seat radius (fixed BC), mm
    bore_z: object           # (z0, z1) or list of (z0, z1) bearing seats
    extra_mass_g: float = 0.0
    extra_Izz: float = 0.0   # g mm^2

    @property
    def seats(self):
        return self.bore_z if isinstance(self.bore_z[0], (tuple, list)) else [self.bore_z]


def _grid(breaks, h):
    breaks = np.unique(np.round(np.asarray(breaks, float), 9))
    pts = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(np.ceil((b - a) / h - 1e-9)))
        pts.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(pts)


def _Bmat(G):
    """G: (ne, nq, 27, 3) shape-function gradients -> B (ne, nq, 6, 81), Voigt xx,yy,zz,yz,xz,xy."""
    ne, nq = G.shape[:2]
    B = np.zeros((ne, nq, 6, 81))
    gx, gy, gz = G[..., 0], G[..., 1], G[..., 2]
    B[:, :, 0, 0::3] = gx; B[:, :, 1, 1::3] = gy; B[:, :, 2, 2::3] = gz
    B[:, :, 3, 1::3] = gz; B[:, :, 3, 2::3] = gy
    B[:, :, 4, 0::3] = gz; B[:, :, 4, 2::3] = gx
    B[:, :, 5, 0::3] = gy; B[:, :, 5, 1::3] = gx
    return B


class Model:
    CHUNK = 600

    def __init__(self, d: Design, h_r=0.8, h_z=0.8, n_theta=48, theta_refine=True, z_extra=()):
        self.d = d
        rb = [x for b in d.blocks for x in (b.r0, b.r1)]
        zb = ([x for b in d.blocks for x in (b.z0, b.z1)] + list(d.contact) + [z for s in d.seats for z in s]
              + list(z_extra))      # extra grid lines, e.g. load-patch edges
        r, z = _grid(rb, h_r), _grid(zb, h_z)
        if theta_refine:   # finer near the load points at 90 and 270 deg (end/mean spacing 0.2)
            s = np.linspace(-1, 1, n_theta + 1); k = 0.4
            th = np.pi + (np.pi / 2) * ((1 + k) * s - k * s**3)
        else:
            th = np.linspace(np.pi / 2, 3 * np.pi / 2, n_theta + 1)
        m = MeshHex.init_tensor(r, th, z)
        c = m.p[:, m.t].mean(axis=1)
        mat = -np.ones(m.nelements, int); blk = -np.ones(m.nelements, int)
        for ib, b in enumerate(d.blocks):
            ins = (c[0] > b.r0) & (c[0] < b.r1) & (c[2] > b.z0) & (c[2] < b.z1)
            mat[ins] = b.mat; blk[ins] = ib
        keep = np.nonzero(mat >= 0)[0]
        m = m.restrict(keep)
        self.mat, self.blk = mat[keep], blk[keep]
        m2 = MeshHex2.from_mesh(m)
        rr, tt, zz = m2.doflocs
        self.mesh = dataclasses.replace(m2, doflocs=np.vstack([rr * np.cos(tt), rr * np.sin(tt), zz]))
        self._check_theta_axis()
        self.sb = Basis(self.mesh, ElementHex2(), intorder=4)
        self.X = self.sb.doflocs
        self.nn = self.X.shape[1]
        self.ndofs = 3 * self.nn

    def _check_theta_axis(self):
        """at="surface" assumes reference axis 2 runs along theta (true for MeshHex.init_tensor(r, th, z))."""
        P = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
        b = Basis(self.mesh, ElementHex2(), quadrature=(P, np.ones(2)), elements=np.array([0]))
        X = np.asarray(b.global_coordinates().value)[:, 0, :]
        if not (np.isclose(np.hypot(*X[:2, 0]), np.hypot(*X[:2, 1])) and np.isclose(X[2, 0], X[2, 1])):
            raise RuntimeError("reference axis 2 is not theta; fix the at='surface' stations")

    # ---------------- element data ----------------
    def _chunks(self, i, el=None, at="gauss"):
        if el is None:
            el = np.nonzero(self.mat == i)[0]
        quad = None
        if at in ("nodes", "surface"):
            # "nodes": the 27 element nodes (wall surfaces, but also the element ends in theta).
            # "surface": nodes (0, 1/2, 1) in r and z, so wall surfaces are included, but the 2-point
            # Gauss stations in theta (reference axis 2). Curved quadratic elements carry a parasitic
            # membrane stress that is parabolic in theta, about +2.5..8 % at the element ends and
            # zero at these stations (see validate_stress.py, V4).
            g = np.array([0.0, 0.5, 1.0])
            gt = g if at == "nodes" else 0.5 + np.array([-0.5, 0.5]) / np.sqrt(3)
            P = np.array(np.meshgrid(g, g, gt, indexing='ij')).reshape(3, -1)
            quad = (P, np.ones(P.shape[1]))
        for k in range(0, len(el), self.CHUNK):
            sub = el[k:k + self.CHUNK]
            b = (Basis(self.mesh, ElementHex2(), quadrature=quad, elements=sub) if quad is not None
                 else Basis(self.mesh, ElementHex2(), intorder=4, elements=sub))
            G = np.stack([b.basis[j][0].grad for j in range(27)])      # (27, 3, ne, nq)
            G = G.transpose(2, 3, 0, 1)                                 # (ne, nq, 27, 3)
            ed = b.element_dofs                                         # (27, ne)
            gd = (3 * ed.T[:, :, None] + np.arange(3)).reshape(len(sub), 81)
            yield b, G, gd

    def stiffness(self):
        K = None
        for i, M in enumerate(self.d.materials):
            C = M.C()
            for b, G, gd in self._chunks(i):
                B = _Bmat(G)
                CB = np.einsum('ij,eqjb->eqib', C, B)
                Bw = B * b.dx[:, :, None, None]
                ne = B.shape[0]
                Ke = np.matmul(Bw.reshape(ne, -1, 81).transpose(0, 2, 1), CB.reshape(ne, -1, 81))
                Kc = sp.coo_matrix((Ke.ravel(), (np.repeat(gd, 81, axis=1).ravel(), np.tile(gd, (1, 81)).ravel())),
                                   shape=(self.ndofs, self.ndofs)).tocsr()
                K = Kc if K is None else K + Kc
        return K

    def mass_matrix(self):
        """Consistent mass matrix (tonne, so K u = w^2 M u gives w in rad/s with mm, N, MPa)."""
        rows, cols, vals = [], [], []
        for i, M in enumerate(self.d.materials):
            for b, G, gd in self._chunks(i):
                Nv = np.stack([b.basis[j][0].value for j in range(27)], -1)   # (ne, nq, 27)
                Ms = np.einsum('eqa,eqb,eq->eab', Nv, Nv, b.dx * M.rho_t)    # scalar (ne, 27, 27)
                ne = Ms.shape[0]
                Me = np.zeros((ne, 81, 81))
                for c in range(3):
                    Me[:, c::3, c::3] = Ms
                rows.append(np.repeat(gd, 81, axis=1).ravel())
                cols.append(np.tile(gd, (1, 81)).ravel())
                vals.append(Me.ravel())
        Mm = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.ndofs, self.ndofs)).tocsr()
        Mm.sum_duplicates()
        return Mm

    def _vec(self, comps):
        f = np.zeros(self.ndofs)
        for c, v in enumerate(comps):
            f[c::3] += v
        return f

    def centrifugal(self, omega):
        f = np.zeros(self.ndofs)
        for i, M in enumerate(self.d.materials):
            el = np.nonzero(self.mat == i)[0]
            if len(el) == 0:
                continue
            b = Basis(self.mesh, ElementHex2(), intorder=4, elements=el)
            k = M.rho_t * omega**2
            fx = asm(LinearForm(lambda v, w: k * w.x[0] * v), b)
            fy = asm(LinearForm(lambda v, w: k * w.x[1] * v), b)
            f += self._vec([fx, fy, 0 * fx])
        return f

    def traction_patch(self, sel, direction_fn, F_total):
        """Total force F_total (N, in this half model) spread uniformly over boundary facets whose
        vertex-midpoints satisfy sel(x); direction_fn(x) -> (3, ...) unit vector field."""
        bf = self.mesh.boundary_facets()
        mid = self.mesh.p[:, self.mesh.facets[:, bf]].mean(axis=1)
        facets = bf[sel(mid)]
        if len(facets) == 0:
            raise ValueError("empty load patch")
        fb = FacetBasis(self.mesh, ElementHex2(), facets=facets, intorder=4)
        area = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), fb)
        p = F_total / area
        comps = [asm(LinearForm(lambda v, w, c=c: p * direction_fn(w.x)[c] * v), fb) for c in range(3)]
        return self._vec(comps), area, facets

    # ---------------- boundary conditions ----------
    def dofs_where(self, fn, comp):
        nodes = np.nonzero(fn(self.X))[0]
        comps = [comp] if np.isscalar(comp) else comp
        return np.concatenate([3 * nodes + c for c in comps])

    def dofs_fixed_bore(self):
        d = self.d
        def on_bore(x):
            r = np.hypot(x[0], x[1])
            inz = np.zeros(x.shape[1], bool)
            for z0, z1 in d.seats:
                inz |= (x[2] >= z0 - 1e-9) & (x[2] <= z1 + 1e-9)
            return (r < d.r_bore + 1e-6) & inz
        return self.dofs_where(on_bore, [0, 1, 2])

    def dofs_symmetry(self):
        return self.dofs_where(lambda x: np.abs(x[0]) < 1e-7, 0)

    # ---------------- solve ---------------------
    def solve(self, K, f, D):
        """Direct solve (PARDISO). f: vector or list of vectors sharing K and D."""
        import pypardiso
        D = np.unique(np.concatenate(D))
        I = np.setdiff1d(np.arange(self.ndofs), D)
        Kc = K[I][:, I].tocsr()
        multi = isinstance(f, (list, tuple))
        F = np.ascontiguousarray(np.column_stack(f if multi else [f])[I])
        X = pypardiso.spsolve(Kc, F)
        X = X.reshape(len(I), -1)
        U = np.zeros((self.ndofs, X.shape[1]))
        U[I] = X
        return [U[:, k] for k in range(U.shape[1])] if multi else U[:, 0]

    # ---------------- post ----------------------
    def stresses(self, u, by="mat", at="surface", el_filter=None):
        """Max von Mises and max principal stress, grouped by material (by="mat", keyed by material
        name) or by block (by="block", keyed by block index). at = "surface" (default: wall surfaces,
        theta-Gauss stations), "nodes" (27 element nodes) or "gauss" (3x3x3 interior Gauss points)."""
        out = {}
        groups = ([(M.name, M, np.nonzero(self.mat == i)[0]) for i, M in enumerate(self.d.materials)] if by == "mat"
                  else [(ib, self.d.materials[b.mat], np.nonzero(self.blk == ib)[0]) for ib, b in enumerate(self.d.blocks)])
        for key, M, el in groups:
            if el_filter is not None:
                el = el[el_filter[el]]
            C = M.C(); vm_max = 0.0; s1_max = -np.inf; any_el = False
            for b, G, gd in self._chunks(None, el, at=at):
                any_el = True
                B = _Bmat(G)
                eps = np.einsum('eqia,ea->eqi', B, u[gd])
                s = eps @ C.T                                   # (ne, nq, 6)
                sx, sy, sz, tyz, txz, txy = np.moveaxis(s, -1, 0)
                vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3 * (tyz**2 + txz**2 + txy**2))
                S = np.stack([np.stack([sx, txy, txz], -1), np.stack([txy, sy, tyz], -1), np.stack([txz, tyz, sz], -1)], -2)
                s1 = np.linalg.eigvalsh(S)[..., -1]
                vm_max = max(vm_max, float(vm.max())); s1_max = max(s1_max, float(s1.max()))
            if any_el:
                out[key] = dict(vm=vm_max, s1=s1_max, strength=M.strength, mat=M.name)
        return out

    def mass_props(self):
        """Mass (g) and polar inertia Izz (g mm^2) of the FULL 360 deg wheel."""
        m = 0.0; I = 0.0
        for i, M in enumerate(self.d.materials):
            el = np.nonzero(self.mat == i)[0]
            if len(el) == 0:
                continue
            b = Basis(self.mesh, ElementHex2(), intorder=4, elements=el)
            vol = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), b)
            j = asm(Functional(lambda w: w.x[0]**2 + w.x[1]**2), b)
            m += 2 * vol * M.rho * 1e-6     # mm^3 * kg/m^3 = 1e-9 kg = 1e-6 g
            I += 2 * j * M.rho * 1e-6
        m += self.d.extra_mass_g; I += self.d.extra_Izz
        return dict(mass_g=m, Izz_gmm2=I, m_eq_g=I / self.d.R**2)

    def disp_at(self, u, pts):
        """Displacement (3, n) at the nodes nearest to physical points pts (3, n)."""
        out = []
        for p in pts.T:
            j = np.argmin(np.sum((self.X - p[:, None])**2, axis=0))
            out.append(u[3 * j:3 * j + 3])
        return np.array(out).T
