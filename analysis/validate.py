"""Validation of wheelfem against closed-form solutions.
V1  thin ring, diametral pinch:  dD = (pi/4 - 2/pi) F R^3 / (E I)   (Roark, Table 9.2 case 1)
V2  rotating annulus (plane stress): u(Ro) = Ro*sigma_theta(Ro)/E   (Timoshenko & Goodier)
Run:  ../.venv/bin/python validate.py
"""
import numpy as np, time
from wheelfem import Material, Block, Design, Model

Al = Material("Al 7075-T6", 71700, 0.33, 2810, 503)

def V1(h, nth):
    Ro, t, b, F = 14.1, 1.0, 5.0, 20.0
    d = Design("ring", [Block(Ro - t, Ro, 0, b, 0)], [Al], Ro, (0, b), Ro - t, (0, b))
    m = Model(d, h_r=h, h_z=h, n_theta=nth)
    K = m.stiffness()
    a = np.deg2rad(2.0)
    th = lambda x: np.arctan2(x[1], x[0])
    on_R = lambda x: np.hypot(x[0], x[1]) > Ro - 0.2 * t
    ftop, *_ = m.traction_patch(lambda x: on_R(x) & (th(x) > np.pi/2) & (th(x) < np.pi/2 + a),
                                lambda x: np.array([0*x[0], -1 + 0*x[0], 0*x[0]]), F / 2)
    fbot, *_ = m.traction_patch(lambda x: on_R(x) & (th(x) < -np.pi/2) & (th(x) > -np.pi/2 - a),
                                lambda x: np.array([0*x[0], 1 + 0*x[0], 0*x[0]]), F / 2)
    D = [m.dofs_symmetry(),
         m.dofs_where(lambda x: np.abs(x[1]) < 1e-9, 1),
         m.dofs_where(lambda x: np.isclose(x[2], b / 2), 2)]
    u = m.solve(K, ftop + fbot, D)
    Rm = Ro - t / 2
    top = m.disp_at(u, np.array([[0.0], [Rm], [b / 2]]))
    dD_fe = -2 * top[1, 0]
    I = b * t**3 / 12
    dD_th = (np.pi / 4 - 2 / np.pi) * F * Rm**3 / (Al.E * I)
    return dD_fe, dD_th, m.ndofs, 0

def V2(h, nth):
    Ri, Ro, b = 10.0, 14.0, 1.0
    w = 20000 * 2 * np.pi / 60
    d = Design("annulus", [Block(Ri, Ro, 0, b, 0)], [Al], Ro, (0, b), Ri, (0, b))
    m = Model(d, h_r=h, h_z=b / 2, n_theta=nth, theta_refine=False)
    K = m.stiffness()
    f = m.centrifugal(w)
    D = [m.dofs_symmetry(),
         m.dofs_where(lambda x: np.abs(x[1]) < 1e-9, 1),
         m.dofs_where(lambda x: np.isclose(x[2], b / 2), 2)]
    u = m.solve(K, f, D)
    p = m.disp_at(u, np.array([[-Ro], [0.0], [b / 2]]))
    nu, rho = Al.nu, Al.rho_t
    st = (3 + nu) / 8 * rho * w**2 * (Ri**2 + Ro**2 + Ri**2 * Ro**2 / Ro**2 - (1 + 3 * nu) / (3 + nu) * Ro**2)
    return -p[0, 0], Ro * st / Al.E, m.ndofs

if __name__ == "__main__":
    print("V1 thin ring pinch (R/t = 13.6), mesh convergence")
    for h, nth in [(1.0, 24), (1.0, 48), (0.5, 72), (0.5, 120)]:
        t0 = time.time()
        fe, th, n, it = V1(h, nth)
        print(f"  h={h} n_theta={nth:4d} dofs={n:7d}  dD FE={fe*1e3:8.3f} um  theory={th*1e3:8.3f} um  err={(fe/th-1)*100:+6.2f}%  ({time.time()-t0:.1f}s)")
    print("V2 rotating annulus 20k rpm, u_r(Ro)")
    for h, nth in [(1.0, 24), (0.5, 48)]:
        fe, th, n = V2(h, nth)
        print(f"  h={h} n_theta={nth:4d} dofs={n:7d}  u FE={fe*1e6:8.3f} nm  theory={th*1e6:8.3f} nm  err={(fe/th-1)*100:+6.2f}%")
