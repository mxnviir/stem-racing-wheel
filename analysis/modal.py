"""Natural frequencies (half model -> modes symmetric about x = 0; each ovalisation pair has one
symmetric member, so all n-lobe modes appear).  Shift-invert Lanczos with a PARDISO factorisation.
V3 validation: free thin ring, in-plane flexural mode n:
    f_n = n (n^2 - 1) / sqrt(n^2 + 1) * sqrt(E I / (rho A R^4)) / (2 pi)   (Rao, Vibration of
    Continuous Systems, ring in-plane inextensional modes)."""
import numpy as np, scipy.sparse.linalg as spla
from wheelfem import Material, Block, Design, Model

def modes(m, K, M, D, k=8, sigma_hz=-50.0):
    import pypardiso
    D = np.unique(np.concatenate(D)) if len(D) else np.array([], int)
    I = np.setdiff1d(np.arange(m.ndofs), D)
    Kc, Mc = K[I][:, I].tocsr(), M[I][:, I].tocsr()
    s = (2 * np.pi * sigma_hz)**2 * np.sign(sigma_hz)
    A = (Kc - s * Mc).tocsr()
    ps = pypardiso.PyPardisoSolver()
    ps.factorize(A)
    op = spla.LinearOperator(A.shape, matvec=lambda x: ps.solve(A, np.ascontiguousarray(x)))
    w2, V = spla.eigsh(Kc, k=k, M=Mc, sigma=s, OPinv=op, which='LM')
    f = np.sqrt(np.abs(w2)) / (2 * np.pi) * np.sign(w2)
    o = np.argsort(f)
    return f[o], V[:, o], I

if __name__ == "__main__":
    Al = Material("Al", 71700, 0.33, 2810, 503)
    Ro, t, b = 14.1, 1.0, 5.0
    d = Design("ring", [Block(Ro - t, Ro, 0, b, 0)], [Al], Ro, (0, b), Ro - t, (0, b))
    m = Model(d, h_r=1.0, h_z=1.0, n_theta=72, theta_refine=False)
    K, M = m.stiffness(), m.mass_matrix()
    f, V, I = modes(m, K, M, [m.dofs_symmetry()], k=12)
    Rm = Ro - t / 2; A = b * t; Iy = b * t**3 / 12
    print("FE (Hz):", np.round(f[f > 1], 0))
    for n in (2, 3, 4):
        fn = n * (n * n - 1) / np.sqrt(n * n + 1) * np.sqrt(Al.E * Iy / (Al.rho_t * A * Rm**4)) / (2 * np.pi)
        fe = f[np.argmin(np.abs(f - fn))]
        print(f"  in-plane n={n}: theory {fn:8.0f} Hz  nearest FE {fe:8.0f} Hz  err {(fe/fn-1)*100:+.2f}%")
