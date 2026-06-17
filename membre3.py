import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
import aero_model
import dynamic
import atmosphere

atm = atmosphere.AtmosphereISA()

# === PARAMÈTRES FIXES — alignés avec Membre 2 ===
h       = 6000.0
mach    = 0.6
ms_list = [0.7, 0.2]
colors  = {0.7: 'steelblue', 0.2: 'tomato'}

# ============================================================
# PARTIE 1 — Plan complexe valeurs propres
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

for ms in ms_list:
    color = colors[ms]
    avion = aero_model.Airbus_A321_200(atm)
    avion.set_options(stall=True, buffeting=False, wave_drag=True)
    avion.set_mass(avion.m_design)
    avion.set_static_margin(ms)

    Va = atm.tas_from_mach_altp(mach, h)
    trim = dynamic.get_trim_level_flight(avion, h, Va, use_saturations=False)
    aoa_e   = trim["aoa"][0]
    dtrim_e = trim["dtrim"][0]
    dthr_e  = trim["dthr"][0]

    Xe = np.array([aoa_e, 0.0, Va, 0.0, h, 0.0, avion.get_mass()])
    Ue = np.array([dtrim_e, 0.0, dthr_e])

    A, B = dynamic.num_jacobian(Xe, Ue, avion)
    A4 = A[0:4, 0:4]
    eigvals = np.linalg.eigvals(A4)

    first = True
    for eig in eigvals:
        ax.plot(eig.real, eig.imag, 'o',
                color=color, markersize=12,
                label=f"ms = {ms}" if first else "_nolegend_")
        first = False

ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label="Axe stabilite")
ax.set_xlabel("Partie reelle Re(lambda)", fontsize=13)
ax.set_ylabel("Partie imaginaire Im(lambda)", fontsize=13)
ax.set_title("Valeurs propres de A4 — ms = 0.7 vs ms = 0.2", fontsize=14)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# --- Version zoomée ---
fig, ax = plt.subplots(figsize=(8, 8))

for ms in ms_list:
    color = colors[ms]
    avion = aero_model.Airbus_A321_200(atm)
    avion.set_options(stall=True, buffeting=False, wave_drag=True)
    avion.set_mass(avion.m_design)
    avion.set_static_margin(ms)

    Va = atm.tas_from_mach_altp(mach, h)
    trim = dynamic.get_trim_level_flight(avion, h, Va, use_saturations=False)
    aoa_e   = trim["aoa"][0]
    dtrim_e = trim["dtrim"][0]
    dthr_e  = trim["dthr"][0]

    Xe = np.array([aoa_e, 0.0, Va, 0.0, h, 0.0, avion.get_mass()])
    Ue = np.array([dtrim_e, 0.0, dthr_e])

    A, B = dynamic.num_jacobian(Xe, Ue, avion)
    A4 = A[0:4, 0:4]
    eigvals = np.linalg.eigvals(A4)

    first = True
    for eig in eigvals:
        ax.plot(eig.real, eig.imag, 'o',
                color=color, markersize=12,
                label=f"ms = {ms}" if first else "_nolegend_")
        first = False

ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label="Axe stabilite")
ax.set_xlabel("Partie reelle Re(lambda)", fontsize=13)
ax.set_ylabel("Partie imaginaire Im(lambda)", fontsize=13)
ax.set_title("Valeurs propres de A4 — ms = 0.7 vs ms = 0.2 (zoom)", fontsize=14)
ax.set_xlim(-0.02, 0.02)
ax.set_ylim(-0.1, 0.1)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PARTIE 2 — Extraction wn, zeta, T pour chaque mode
# ============================================================
print(f"\n{'ms':>6} | {'Mode':>15} | {'wn (rad/s)':>12} | {'zeta':>8} | {'T (s)':>10} | {'Re(eig)':>10}")
print("-" * 75)

results = {}

for ms in ms_list:
    avion = aero_model.Airbus_A321_200(atm)
    avion.set_options(stall=True, buffeting=False, wave_drag=True)
    avion.set_mass(avion.m_design)
    avion.set_static_margin(ms)

    Va = atm.tas_from_mach_altp(mach, h)
    trim = dynamic.get_trim_level_flight(avion, h, Va, use_saturations=False)
    aoa_e   = trim["aoa"][0]
    dtrim_e = trim["dtrim"][0]
    dthr_e  = trim["dthr"][0]

    Xe = np.array([aoa_e, 0.0, Va, 0.0, h, 0.0, avion.get_mass()])
    Ue = np.array([dtrim_e, 0.0, dthr_e])

    A, B = dynamic.num_jacobian(Xe, Ue, avion)
    A4 = A[0:4, 0:4]
    eigvals = np.linalg.eigvals(A4)

    eigvals_sorted = sorted(eigvals, key=lambda x: abs(x))

    modes = []
    seen = set()
    for eig in eigvals_sorted:
        key = round(abs(eig), 4)
        if key in seen:
            continue
        seen.add(key)

        wn   = abs(eig)
        zeta = -eig.real / wn if wn > 1e-10 else 0
        T    = (2 * np.pi / abs(eig.imag)) if abs(eig.imag) > 1e-10 else float('inf')
        tau  = (1 / abs(eig.real)) if abs(eig.real) > 1e-10 else float('inf')
        modes.append((wn, zeta, T, eig.real, tau))

    noms = ["Phugoid", "Courte periode"]
    results[ms] = {}
    for i, (wn, zeta, T, re, tau) in enumerate(modes[:2]):
        nom = noms[i]
        results[ms][nom] = {"wn": wn, "zeta": zeta, "T": T, "re": re, "tau": tau}
        print(f"{ms:>6} | {nom:>15} | {wn:>12.5f} | {zeta:>8.5f} | {T:>10.2f} | {re:>10.6f}")

print(f"\n--- Temps caracteristique d'amortissement (tau = 1/|Re(lambda)|) ---")
for ms in ms_list:
    tau_phug = results[ms]["Phugoid"]["tau"]
    print(f"ms={ms} : tau phugoide = {tau_phug:.1f}s  "
          f"(3*tau = {3*tau_phug:.0f}s pour amortissement quasi-complet)")


# ============================================================
# PARTIE 3 — Placement sur Figure 3
# ============================================================

# -- Graphe PHUGOID --
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Specifications phugoid — placement de l'A321", fontsize=13)
ax.set_xlabel("2 x zeta x wn  (rad/s)", fontsize=12)
ax.set_ylabel("Periode T (s)", fontsize=12)
ax.set_xlim(-0.04, 0.08)
ax.set_ylim(0, 120)

ax.axvspan(-0.04, -0.02, color='red',    alpha=0.15, label="Inacceptable")
ax.axvspan(-0.02,  0.00, color='orange', alpha=0.15, label="Acceptable (urgence)")
ax.axvspan( 0.00,  0.08, color='green',  alpha=0.10, label="Satisfaisant")
ax.axvline(0, color='black', linewidth=1, linestyle='--')

for ms in ms_list:
    color = colors[ms]
    mode = results[ms]["Phugoid"]
    x_val = 2 * mode["zeta"] * mode["wn"]
    y_val = mode["T"]
    if y_val == float('inf'):
        print(f"  ms={ms} phugoid aperiodique (T=inf) — non trace")
        continue
    ax.plot(x_val, y_val, 'o', color=color, markersize=14,
            label=f"ms={ms}  (T={y_val:.1f}s, zeta={mode['zeta']:.3f})")
    ax.annotate(f"ms={ms}", (x_val, y_val),
                textcoords="offset points", xytext=(8, 5), fontsize=11)

ax.legend(fontsize=10)
ax.grid(True)
plt.tight_layout()
plt.show()


# -- Graphe COURTE PERIODE --
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Specifications courte periode — placement de l'A321", fontsize=13)
ax.set_xlabel("Damping ratio zeta (-)", fontsize=12)
ax.set_ylabel("Frequence naturelle wn (rad/s)", fontsize=12)
ax.set_xscale('log')
ax.set_xlim(0.1, 10)
ax.set_ylim(0, 14)

ax.axvspan(0.1,  0.35, color='red',    alpha=0.15, label="Inacceptable")
ax.axvspan(0.35, 1.3,  color='green',  alpha=0.10, label="Satisfaisant")
ax.axvspan(1.3,  10,   color='orange', alpha=0.10, label="Acceptable")

for ms in ms_list:
    color = colors[ms]
    mode = results[ms]["Courte periode"]
    if mode["zeta"] >= 1.0:
        print(f"ms={ms} courte periode aperiodique (zeta={mode['zeta']:.3f}) — non trace")
        ax.annotate(f"ms={ms} : aperiodique\n(zeta >= 1, pas d'oscillation)",
                    xy=(1.0, mode["wn"]),
                    xytext=(1.5, mode["wn"] + 0.5),
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle='->', color=color))
        ax.axvline(1.0, color=color, linewidth=1.5,
                   linestyle=':', alpha=0.6)
        continue
    ax.plot(mode["zeta"], mode["wn"], 'o', color=color, markersize=14,
            label=f"ms={ms}  (wn={mode['wn']:.3f}, zeta={mode['zeta']:.3f})")
    ax.annotate(f"ms={ms}", (mode["zeta"], mode["wn"]),
                textcoords="offset points", xytext=(8, 5), fontsize=11)

ax.legend(fontsize=10)
ax.grid(True, which='both')
plt.tight_layout()
plt.show()