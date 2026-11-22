"""
First-pass physics screening for a STEM Racing Professional Class wheel.
All SI units internally; printed in practical units.

Regulation inputs (2026-27 Pro Tech Regs, T7):
  D = 28.0..32.0 mm, contact width >= 13.0 (front) / 17.0 (rear) mm,
  car >= 48.0 g without cartridge, cartridge 28.9..29.4 g.
User inputs: 20 N wheel load, 20 000 rpm.
"""
import numpy as np

RPM = 20_000
w = RPM * 2 * np.pi / 60            # rad/s
D = 28.2e-3                          # target diameter (0.2 mm margin over 28.0 min)
R = D / 2
B_rear, B_front = 17.2e-3, 13.2e-3   # contact widths with 0.2 mm margin
m_car = (48.0 + 29.15) * 1e-3        # kg, at min weight with full cartridge
W_static = m_car * 9.81 / 4          # N per wheel

print(f"omega = {w:.0f} rad/s, rim speed = {w*R:.1f} m/s ({w*R*3.6:.0f} km/h)")
print(f"static load per wheel = {W_static*1000:.0f} mN  (design load 20 N = {20/W_static:.0f}x static)\n")

# ---- materials: rho kg/m3, E Pa, strength Pa, practical min wall m, notes
# Values are typical datasheet numbers; verify against your exact grade/resin.
M = {
    "Al 7075-T6 (milled/turned)": (2810, 71.7e9, 503e6, 0.25e-3),
    "Al 6061-T6 (milled/turned)": (2700, 68.9e9, 276e6, 0.25e-3),
    "Ti-6Al-4V (milled)":         (4430, 113.8e9, 880e6, 0.30e-3),
    "SLS PA12":                   (1010, 1.70e9, 48e6, 0.70e-3),
    "FDM ABS (perimeters only)":  (1040, 2.0e9, 30e6, 0.80e-3),
    "SLA standard/rigid resin":   (1180, 2.6e9, 60e6, 0.40e-3),
    "SLA Tough/ABS-like resin":   (1150, 2.2e9, 45e6, 0.40e-3),
    "SLA glass-filled (10K-type)":(1650, 10.0e9, 65e6, 0.40e-3),
}

print("1) Spin-only: thin-ring hoop stress sigma = rho v^2, diameter growth dD = 2 R sigma/E")
print(f"{'material':30s} {'sigma MPa':>9s} {'SF':>6s} {'dD um':>7s}")
v = w * R
for k, (rho, E, S, t) in M.items():
    s = rho * v**2
    print(f"{k:30s} {s/1e6:9.2f} {S/s:6.0f} {2*R*s/E*1e6:7.2f}")

print("\n2) Imbalance force F = m_wheel * e * omega^2 (m_wheel = 1.5 g)")
for e_um in (2, 5, 10, 25, 50):
    F = 1.5e-3 * e_um * 1e-6 * w**2
    print(f"   e = {e_um:3d} um -> F = {F*1000:6.0f} mN  = {F/W_static*100:5.0f}% of static wheel load")

print("\n3) Merit indices (higher = lighter for the same job)")
print(f"{'material':30s} {'E^1/3/rho (shell stiff)':>24s} {'S^1/2/rho (shell strength)':>27s} {'rho*t_min kg/m2 (min-gauge)':>28s}")
for k, (rho, E, S, t) in M.items():
    print(f"{k:30s} {E**(1/3)/rho*1e1:24.3f} {S**0.5/rho:27.2f} {rho*t:28.3f}")

print("\n4) Equivalent translating mass of a thin rim at min gauge (rear, 17.2 mm wide)")
print("   m_eq = I/R^2. Rim only; faces/hub add more. Lower is better.")
for k, (rho, E, S, t) in M.items():
    Ro, Ri = R, R - t
    m = rho * np.pi * (Ro**2 - Ri**2) * B_rear
    I = 0.5 * m * (Ro**2 + Ri**2)
    print(f"   {k:30s} rim mass {m*1000:5.2f} g   m_eq {I/R**2*1000:5.2f} g")

print("\n5) Sensitivity: race time vs effective mass. Constant force: dt/t = dm/2m; constant impulse: dt/t = dm/m")
m_eff = m_car
for dm in (1, 2, 4):
    lo, hi = dm/1000/(2*m_eff), dm/1000/m_eff
    print(f"   -{dm} g total m_eq over 4 wheels -> {lo*100:.1f}-{hi*100:.1f}% time  ({lo*1050:.0f}-{hi*1050:.0f} ms on an assumed 1.05 s run)")

print("\n6) 20 N check on an UNSUPPORTED rim (free ring, diametral point loads - worst case)")
print("   M = 0.318 F R,  delta = 0.149 F R^3/(E I),  I = b t^3/12, rear width 17.2 mm")
F, b = 20.0, B_rear
print(f"{'material':30s} {'t_min mm':>8s} {'sigma MPa':>10s} {'delta mm':>9s} | {'t for SF1.5':>11s} {'t for 20um':>10s} {'rim m_eq g':>10s}")
for k, (rho, E, S, t) in M.items():
    I = b * t**3 / 12
    Mb = 0.318 * F * R
    s = Mb * (t/2) / I
    d = 0.149 * F * R**3 / (E * I)
    t_s = np.sqrt(6 * Mb / (b * S/1.5))
    t_d = (12 * 0.149 * F * R**3 / (E * b * 20e-6))**(1/3)
    tt = max(t, t_s, t_d)
    m = rho * np.pi * (R**2 - (R-tt)**2) * b
    Ieq = 0.5 * m * (R**2 + (R-tt)**2) / R**2
    print(f"{k:30s} {t*1e3:8.2f} {s/1e6:10.0f} {d*1e3:9.2f} | {t_s*1e3:11.2f} {t_d*1e3:10.2f} {Ieq*1e3:10.2f}")
print("   -> an unsupported min-gauge rim cannot carry 20 N. The rim needs internal support (web/spokes/core);")
print("      support spacing, not material alone, sets the mass. That is what the FEA stage optimises.")
