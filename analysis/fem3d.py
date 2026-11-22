"""
fem3d: linear elasticity on quadratic tetrahedra (10-node, P2 displacement on straight-sided tets
from geom3d), for wheels of arbitrary 3D shape (spokes, variable rim thickness, fillets).

* Assembly: shape-function gradients from scikit-fem, B (6 x 30) per quadrature point, batched matmul,
  CSR accumulation in chunks. Stiffness with the 4-point rule (exact for affine P2), mass with order 4.
* Solve: MKL PARDISO (several right-hand sides per factorisation).
* Stress recovery: stress tensors evaluated at the 10 element nodes and averaged per global node
  (unweighted), von Mises and max principal from the averaged tensor.
* Geometric stiffness K_G(sigma) for linear buckling; generalised eigen solves for modes and buckling.
Units mm, N, MPa, tonne/mm^3 (Material.rho_t). dof = 3 * node + component.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import MeshTet, Basis, FacetBasis, ElementTetP2, LinearForm, Functional, asm
from wheelfem import Material


def _Bmat(G):
    ne, nq, nb = G.shape[:3]
    B = np.zeros((ne, nq, 6, 3 * nb))
    gx, gy, gz = G[..., 0], G[..., 1], G[..., 2]
    B[:, :, 0, 0::3] = gx; B[:, :, 1, 1::3] = gy; B[:, :, 2, 2::3] = gz
    B[:, :, 3, 1::3] = gz; B[:, :, 3, 2::3] = gy
    B[:, :, 4, 0::3] = gz; B[:, :, 4, 2::3] = gx
    B[:, :, 5, 0::3] = gy; B[:, :, 5, 1::3] = gx
    return B


def _vm_s1(S):
    sx, sy, sz, tyz, txz, txy = np.moveaxis(S, -1, 0)
    vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3 * (tyz**2 + txz**2 + txy**2))
    T = np.stack([np.stack([sx, txy, txz], -1), np.stack([txy, sy, tyz], -1), np.stack([txz, tyz, sz], -1)], -2)
    ev = np.linalg.eigvalsh(T)
    return vm, ev[..., -1], ev[..., 0]


class Model3D:
    CHUNK = 20000

    def __init__(self, P, T, material: Material, extra_mass_g=0.0, extra_Izz=0.0):
        self.mesh = MeshTet(np.ascontiguousarray(P, float), np.ascontiguousarray(T, np.int64))
        self.M = material
        self.extra_mass_g, self.extra_Izz = extra_mass_g, extra_Izz
        self.nn = self.mesh.nvertices + self.mesh.nedges
        self.ndofs = 3 * self.nn
        self.X = self._doflocs()

    def _doflocs(self):
        m = self.mesh
        return np.hstack([m.p, m.p[:, m.edges].mean(axis=1)])

    def _chunks(self, intorder=2, quad=None, elements=None):
        el = np.arange(self.mesh.nelements) if elements is None else elements
        for k in range(0, len(el), self.CHUNK):
            sub = el[k:k + self.CHUNK]
            b = (Basis(self.mesh, ElementTetP2(), quadrature=quad, elements=sub) if quad is not None
                 else Basis(self.mesh, ElementTetP2(), intorder=intorder, elements=sub))
            yield b, sub

    @staticmethod
    def _grads(b):
        G = np.stack([b.basis[j][0].grad for j in range(10)])      # (10, 3, ne, nq)
        return G.transpose(2, 3, 0, 1)                               # (ne, nq, 10, 3)

    @staticmethod
    def _gdofs(b):
        ed = b.element_dofs                                          # (10, ne)
        return (3 * ed.T[:, :, None] + np.arange(3)).reshape(ed.shape[1], 30)

    def _accum(self, blocks):
        A = None
        for vals, gd in blocks:
            n = gd.shape[1]
            Ac = sp.coo_matrix((vals.ravel(), (np.repeat(gd, n, axis=1).ravel(), np.tile(gd, (1, n)).ravel())),
                               shape=(self.ndofs, self.ndofs)).tocsr()
            A = Ac if A is None else A + Ac
        return A

    def stiffness(self):
        C = self.M.C()
        def gen():
            for b, _ in self._chunks(2):
                B = _Bmat(self._grads(b)); ne = B.shape[0]
                CB = np.einsum('ij,eqjb->eqib', C, B)
                Bw = B * b.dx[:, :, None, None]
                yield np.matmul(Bw.reshape(ne, -1, 30).transpose(0, 2, 1), CB.reshape(ne, -1, 30)), self._gdofs(b)
        return self._accum(gen())

    def mass_matrix(self):
        rho = self.M.rho_t
        def gen():
            for b, _ in self._chunks(4):
                Nv = np.stack([b.basis[j][0].value for j in range(10)], -1)
                Ms = np.einsum('eqa,eqb,eq->eab', Nv, Nv, b.dx * rho)
                Me = np.zeros((Ms.shape[0], 30, 30))
                for c in range(3):
                    Me[:, c::3, c::3] = Ms
                yield Me, self._gdofs(b)
        return self._accum(gen())

    def stress_qp(self, u, intorder=2):
        """Stress at quadrature points, list of (b, S (ne, nq, 6))."""
        C = self.M.C()
        for b, sub in self._chunks(intorder):
            yield b, np.einsum('eqia,ea->eqi', _Bmat(self._grads(b)), u[self._gdofs(b)]) @ C.T

    def geometric_stiffness(self, u):
        """K_G for the pre-stress of displacement u: K_G[(a,c),(b,c)] = int grad N_a . sigma . grad N_b."""
        def gen():
            for b, S in self.stress_qp(u):
                G = self._grads(b)
                sx, sy, sz, tyz, txz, txy = np.moveaxis(S, -1, 0)
                Sig = np.stack([np.stack([sx, txy, txz], -1), np.stack([txy, sy, tyz], -1),
                                np.stack([txz, tyz, sz], -1)], -2)          # (ne, nq, 3, 3)
                kg = np.einsum('eqai,eqij,eqbj,eq->eab', G, Sig, G, b.dx)
                Ke = np.zeros((kg.shape[0], 30, 30))
                for c in range(3):
                    Ke[:, c::3, c::3] = kg
                yield Ke, self._gdofs(b)
        return self._accum(gen())

    # ---------------- loads ----------------
    def _vec(self, comps):
        f = np.zeros(self.ndofs)
        for c, v in enumerate(comps):
            f[c::3] += v
        return f

    def centrifugal(self, omega):
        k = self.M.rho_t * omega**2
        b = Basis(self.mesh, ElementTetP2(), intorder=4)
        fx = asm(LinearForm(lambda v, w: k * w.x[0] * v), b)
        fy = asm(LinearForm(lambda v, w: k * w.x[1] * v), b)
        return self._vec([fx, fy, 0 * fx])

    def boundary_facets(self, sel):
        m = self.mesh
        bf = m.boundary_facets()
        mid = m.p[:, m.facets[:, bf]].mean(axis=1)
        return bf[sel(mid)]

    def facet_normals(self, facets):
        m = self.mesh
        a, b, c = (m.p[:, m.facets[i, facets]] for i in range(3))
        n = np.cross((b - a).T, (c - a).T).T
        n /= np.linalg.norm(n, axis=0)
        # orient outward: compare with the vector from the owning element's centroid
        t2f = m.f2t[0, facets]
        cen = m.p[:, m.t[:, t2f]].mean(axis=1)
        s = np.sign(np.sum(n * (a - cen), axis=0))
        return n * s

    def traction_weighted(self, facets, weight_fn, direction_fn, F_total, intorder=6):
        """Traction p(x) = F_total w(x) / int w dA on the given facets (a superset of the patch);
        w is a smooth weight (patch profile), so the load is mesh-independent once resolved."""
        fb = FacetBasis(self.mesh, ElementTetP2(), facets=facets, intorder=intorder)
        W = asm(Functional(lambda w: weight_fn(w.x)), fb)
        if W <= 0:
            raise ValueError("empty load patch")
        comps = [asm(LinearForm(lambda v, w, c=c: (F_total / W) * weight_fn(w.x) * direction_fn(w.x)[c] * v), fb)
                 for c in range(3)]
        return self._vec(comps), W

    def traction_patch(self, facets, direction_fn, F_total):
        if len(facets) == 0:
            raise ValueError("empty load patch")
        fb = FacetBasis(self.mesh, ElementTetP2(), facets=facets, intorder=4)
        area = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), fb)
        p = F_total / area
        comps = [asm(LinearForm(lambda v, w, c=c: p * direction_fn(w.x)[c] * v), fb) for c in range(3)]
        return self._vec(comps), area

    # ---------------- BCs / solve ----------------
    def dofs_where(self, fn, comp):
        nodes = np.nonzero(fn(self.X))[0]
        comps = [comp] if np.isscalar(comp) else comp
        return np.concatenate([3 * nodes + c for c in comps]) if len(nodes) else np.zeros(0, int)

    def solve(self, K, f, D):
        import pypardiso
        D = np.unique(np.concatenate(D)).astype(np.int64)
        I = np.setdiff1d(np.arange(self.ndofs), D)
        Kc = K[I][:, I].tocsr()
        multi = isinstance(f, (list, tuple))
        F = np.ascontiguousarray(np.column_stack(f if multi else [f])[I])
        X = pypardiso.spsolve(Kc, F).reshape(len(I), -1)
        U = np.zeros((self.ndofs, X.shape[1])); U[I] = X
        return [U[:, k] for k in range(U.shape[1])] if multi else U[:, 0]

    def eigs(self, A, B, D, k=6, mode="modes"):
        """mode='modes': A=K, B=M -> lowest frequencies (Hz).  mode='buckling': A=K, B=K_G ->
        lowest positive load factors lambda with (K + lambda K_G) phi = 0."""
        import pypardiso
        D = np.unique(np.concatenate(D)).astype(np.int64)
        I = np.setdiff1d(np.arange(self.ndofs), D)
        Ac, Bc = A[I][:, I].tocsr(), B[I][:, I].tocsr()
        solver = pypardiso.PyPardisoSolver()
        solver.factorize(Ac)
        op = spla.LinearOperator(Ac.shape, matvec=lambda x: solver.solve(Ac, np.ascontiguousarray(x)), dtype=float)
        if mode == "modes":
            # K phi = w2 M phi  ->  K^-1 M phi = (1/w2) phi : largest eigenvalues of K^-1 M
            C = spla.LinearOperator(Ac.shape, matvec=lambda x: op @ (Bc @ x), dtype=float)
            mu, V = spla.eigs(C, k=k, which="LM")
            solver.free_memory(everything=True)
            mu = np.real(mu); o = np.argsort(-mu)
            w2 = 1.0 / mu[o]
            return np.sqrt(np.abs(w2)) / (2 * np.pi), np.real(V[:, o])
        # (K + l KG) phi = 0  ->  K^-1 KG phi = -(1/l) phi ; positive l <-> most negative mu
        C = spla.LinearOperator(Ac.shape, matvec=lambda x: op @ (Bc @ x), dtype=float)
        mu, V = spla.eigs(C, k=k, which="LM")
        solver.free_memory(everything=True)
        mu = np.real(mu)
        lam = np.array(sorted(-1.0 / m for m in mu if m < 0))
        return lam, None

    # ---------------- post ----------------
    def nodal_stress(self, u):
        """Stress tensor (nn, 6) at the P2 nodes, averaged over the elements sharing each node."""
        C = self.M.C()
        ref = ElementTetP2().doflocs.T                     # (3, 10) reference node coordinates
        S = np.zeros((self.nn, 6)); cnt = np.zeros(self.nn)
        for b, sub in self._chunks(quad=(ref, np.ones(10) / 10)):
            s = np.einsum('eqia,ea->eqi', _Bmat(self._grads(b)), u[self._gdofs(b)]) @ C.T   # (ne, 10, 6)
            ed = b.element_dofs.T                          # (ne, 10): node q of element = dof ed[:, q]
            for c in range(6):
                np.add.at(S[:, c], ed.ravel(), s[..., c].ravel())
            np.add.at(cnt, ed.ravel(), 1.0)
        return S / cnt[:, None]

    def stresses(self, u, groups=None):
        """Max vM / max principal / min principal over nodes; groups: {name: bool mask over nodes}."""
        vm, s1, s3 = _vm_s1(self.nodal_stress(u))
        groups = groups or {"all": np.ones(self.nn, bool)}
        out = {}
        for k, msk in groups.items():
            if msk.any():
                i1 = np.argmax(np.where(msk, s1, -np.inf)); iv = np.argmax(np.where(msk, vm, -np.inf))
                out[k] = dict(vm=float(vm[iv]), s1=float(s1[i1]), s3=float(s3[msk].min()),
                              at_s1=self.X[:, i1].round(2).tolist(), at_vm=self.X[:, iv].round(2).tolist())
        return out

    def mass_props(self, half=True):
        b = Basis(self.mesh, ElementTetP2(), intorder=2)
        vol = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), b)
        j = asm(Functional(lambda w: w.x[0]**2 + w.x[1]**2), b)
        f = 2 if half else 1
        m = f * vol * self.M.rho * 1e-6 + self.extra_mass_g
        I = f * j * self.M.rho * 1e-6 + self.extra_Izz
        return dict(mass_g=m, Izz_gmm2=I)

    def disp_at(self, u, pts):
        out = []
        for p in np.atleast_2d(pts.T):
            jn = np.argmin(np.sum((self.X - p[:, None])**2, axis=0))
            out.append(u[3 * jn:3 * jn + 3])
        return np.array(out).T
