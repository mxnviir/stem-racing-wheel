"""
Order-of-magnitude resistance budget for the wheels at race speed, to rank which wheel
properties matter.  Every coefficient here is an ASSUMPTION with a stated range; the
paper should replace them with measurement (coast-down, spin-down rig).
"""
import numpy as np

rho, nu_air = 1.20, 1.5e-5
V = 25.0                       # m/s, representative mid-race speed (assumption: 20-30)
R = 14.1e-3
w = V / R
q = 0.5 * rho * V**2
Re_disc = w * R**2 / nu_air

print(f"V = {V} m/s, omega = {w:.0f} rad/s ({w*60/2/np.pi:.0f} rpm), Re_disc = {Re_disc:.0f} (laminar < 3e5)\n")

# 1. pressure drag of exposed wheels (rotating cylinder in ground effect), Cd ~ 0.5-0.9
A_rear, A_front = 28.2e-3 * 17.2e-3, 28.2e-3 * 13.2e-3
for cd in (0.5, 0.9):
    F = cd * q * 2 * (A_rear + A_front)
    print(f"[aero]     4 wheels, Cd {cd}: {F*1000:6.0f} mN   (frontal area set by T7.4/T7.5 minima)")

# 2. disc-face windage (von Karman laminar free disc, both faces): Cm = 3.87/sqrt(Re)
Cm = 3.87 / np.sqrt(Re_disc)
M = 0.5 * Cm * rho * w**2 * R**5
print(f"[windage]  flush faces: {4*M/R*1000:6.1f} mN equiv. (x4 wheels); open spokes/cavities typically 2-5x")

# 3. bearing friction, MR63 (3x6x2.5): running torque 0.02-0.10 N*mm per bearing (shield/grease dependent)
for T in (0.02, 0.10):
    print(f"[bearings] 8 bearings at {T} N*mm: {8*T/(R*1e3)*1000:6.1f} mN")

# 4. rolling resistance, hard wheel on track: Crr 0.001-0.005 on total weight 0.757 N
for crr in (0.001, 0.005):
    print(f"[rolling]  Crr {crr}: {crr*0.757*1000:6.1f} mN")

# 5. inertia, expressed as equivalent extra mass during acceleration
print("\n[inertia]  per 1 g of total m_eq (4 wheels): +1.3 % effective mass on a 77 g car;")
print("           wheel frontal area is fixed by the regulation minima, so the wheel designer's levers are")
print("           inertia, face/edge aero shape, bearing choice/preparation, and runout/balance.")
