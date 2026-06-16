import aero_model as am
import atmosphere as atm
import display_utils as du
import dynamic as dyn
import units
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration initiale
# ============================================================

h = 6000.0           # altitude [m]
mach = 0.6           # Mach
W_h = 35.0           # rafale de vent [m/s]
km = 0.5             # facteur de modulation
marges_statiques = [0.7, 0.2] 

# Configuration avion
aero_m = am.Airbus_A321_200()
print(f'\n{aero_m.name}')
aero_m.set_options(stall=True, buffeting=False, wave_drag=True)
aero_m.set_mass(aero_m.m_design)

# Calcul de la vitesse vraie
tas = aero_m.atm.tas_from_mach_altp(mach, h)
print(f'\nConditions de vol:')
print(f'  Altitude h = {h} m')
print(f'  Mach = {mach}')
print(f'  TAS = {tas:.2f} m/s')
print(f'  Rafale de vent W_h = {W_h} m/s')

# Trim de reference
trim_ref = dyn.get_trim_level_flight(aero_m, h, tas, use_saturations=False)
aoa_e = trim_ref['aoa'][0]
dtrim_e = trim_ref['dtrim'][0]
dthr_e = trim_ref['dthr'][0]

print(f'  AoA d\'equilibre = {np.rad2deg(aoa_e):.2f} deg')
print(f'  Detrim d\'equilibre = {np.rad2deg(dtrim_e):.2f} deg')

# Changement d'AoA du a la rafale
delta_aoa = np.arctan(W_h / tas)
print(f'\nRafale de vent: Delta_AoA = arctan({W_h}/{tas:.2f}) = {np.rad2deg(delta_aoa):.2f} deg')

# ============================================================
# Fonction pour extraire les caracteristiques de la reponse
# ============================================================

def extract_characteristics(time, X, sv_aoa, sv_q, sv_path):
    """Extrait les caracteristiques de la reponse"""
    aoa = X[:, sv_aoa]
    peaks, _ = find_peaks(aoa[100:], height=np.max(aoa[100:]) * 0.2)
    if len(peaks) > 1:
        periode = 2 * (peaks[1] - peaks[0]) * (time[1] - time[0])
    else:
        periode = np.nan
    return {'periode': periode, 'damping_ratio': np.nan, 'settling_time': np.nan}

# ============================================================
# Boucle sur les deux marges statiques
# ============================================================

results = {}

for ms in marges_statiques:
    print(f'\n\n{"="*70}')
    print(f'SIMULATION POUR MARGE STATIQUE = {ms}')
    print(f'{"="*70}')
    
    aero_m.set_static_margin(ms)
    trim = dyn.get_trim_level_flight(aero_m, h, tas, use_saturations=False)
    aoa_eq = trim['aoa'][0]
    dtrim_eq = trim['dtrim'][0]
    dthr_eq = trim['dthr'][0]
    
    X_eq = np.array([aoa_eq, 0., tas, 0., h, 0., aero_m.get_mass()])
    U_eq = np.array([dtrim_eq, 0., dthr_eq])
    
    X0 = X_eq.copy()
    X0[0] = aoa_eq + delta_aoa
    
    T_sim = 240.0
    T = np.linspace(0, T_sim, int(T_sim * 10) + 1)
    
    print(f'\nTrim avec ms = {ms}:')
    print(f'  AoA trim = {np.rad2deg(aoa_eq):.2f} deg')
    print(f'  Detrim = {np.rad2deg(dtrim_eq):.2f} deg')
    print(f'  Condition initiale: AoA = {np.rad2deg(X0[0]):.2f} deg')
    print(f'\nSimulation NON-LINEAIRE...')
    
    def rhs_nonlin(X, t):
        U = U_eq.copy()
        return dyn.get_state_dot(X, t, U, aero_m)
    
    X_nonlin = odeint(rhs_nonlin, X0, T)
    
    print(f'Simulation LINEARISEE...')
    
    A, B = dyn.num_jacobian(X_eq, U_eq, aero_m)
    dX0 = X0 - X_eq
    
    def rhs_lin(dX, t):
        return A @ dX
    
    dX_lin = odeint(rhs_lin, dX0, T)
    X_lin = X_eq + dX_lin
    
    sv_aoa, sv_q, sv_tas, sv_path, sv_height, sv_xpos, sv_mass = range(7)
    results[ms] = {
        'X_nonlin': X_nonlin,
        'X_lin': X_lin,
        'T': T,
        'X_eq': X_eq
    }

# ============================================================
# GENERATION DES FIGURES
# ============================================================

print(f'\n\n{"="*70}')
print(f'GENERATION DES FIGURES')
print(f'{"="*70}')

sv_aoa, sv_q, sv_tas, sv_path, sv_height, sv_xpos, sv_mass = range(7)

for ms_idx, ms in enumerate(marges_statiques):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Reponse a une rafale de vent - Marge statique ms = {ms}\n' + 
                 f'Altitude {h:.0f}m, Mach {mach}, Rafale {W_h} m/s', 
                 fontsize=16, fontweight='bold')
    
    T = results[ms]['T']
    X_nonlin = results[ms]['X_nonlin']
    X_lin = results[ms]['X_lin']
    X_eq = results[ms]['X_eq']
    
    # Plot 1: Angle d'attaque
    axes[0].plot(T, np.rad2deg(X_nonlin[:, sv_aoa]), 'b-', linewidth=2, label='Non-lineaire')
    axes[0].plot(T, np.rad2deg(X_lin[:, sv_aoa]), 'r--', linewidth=2, label='Linearise')
    axes[0].axhline(np.rad2deg(X_eq[sv_aoa]), color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[0].set_ylabel('AoA (deg)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=12, loc='best')
    axes[0].set_title('Angle d\'attaque', fontsize=14, fontweight='bold')
    
    # Plot 2: Vitesse de tangage
    axes[1].plot(T, np.rad2deg(X_nonlin[:, sv_q]), 'b-', linewidth=2, label='Non-lineaire')
    axes[1].plot(T, np.rad2deg(X_lin[:, sv_q]), 'r--', linewidth=2, label='Linearise')
    axes[1].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].set_ylabel('q (deg/s)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=12, loc='best')
    axes[1].set_title('Vitesse de tangage', fontsize=14, fontweight='bold')
    
    # Plot 3: Vitesse vraie
    axes[2].plot(T, X_nonlin[:, sv_tas], 'b-', linewidth=2, label='Non-lineaire')
    axes[2].plot(T, X_lin[:, sv_tas], 'r--', linewidth=2, label='Linearise')
    axes[2].axhline(X_eq[sv_tas], color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[2].set_ylabel('TAS (m/s)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=12, loc='best')
    axes[2].set_title('Vitesse vraie', fontsize=14, fontweight='bold')
    
    # Plot 4: Pente de trajectoire
    axes[3].plot(T, np.rad2deg(X_nonlin[:, sv_path]), 'b-', linewidth=2, label='Non-lineaire')
    axes[3].plot(T, np.rad2deg(X_lin[:, sv_path]), 'r--', linewidth=2, label='Linearise')
    axes[3].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[3].set_ylabel('gamma (deg)', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Temps (s)', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=12, loc='best')
    axes[3].set_title('Pente de trajectoire', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'slide_{7 if ms == 0.7 else 8}_ms_{ms}_response_240s.png', dpi=150, bbox_inches='tight')
    print(f'Figure slide_{7 if ms == 0.7 else 8}_ms_{ms}_response_240s.png creee')
    plt.show()

# ============================================================
# SLIDE 9 : Zoom 10s sur la courte periode
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'Zoom 20s - Mode courte periode (comparaison ms = 0.7 vs ms = 0.2)\nAltitude {h:.0f}m, Mach {mach}, Rafale {W_h} m/s',
             fontsize=16, fontweight='bold')

T_zoom = (0, 20)

for col_idx, ms in enumerate(marges_statiques):
    T = results[ms]['T']
    X_nonlin = results[ms]['X_nonlin']
    X_lin = results[ms]['X_lin']
    X_eq = results[ms]['X_eq']
    
    mask = (T >= T_zoom[0]) & (T <= T_zoom[1])
    T_zoom_data = T[mask]
    
    ax = axes[0, col_idx]
    ax.plot(T_zoom_data, np.rad2deg(X_nonlin[mask, sv_aoa]), 'b-', linewidth=2.5, label='Non-lineaire')
    ax.plot(T_zoom_data, np.rad2deg(X_lin[mask, sv_aoa]), 'r--', linewidth=2.5, label='Linearise')
    ax.set_ylabel('AoA (deg)', fontsize=14, fontweight='bold')
    ax.set_title(f'Angle d\'attaque - ms = {ms}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    ax = axes[1, col_idx]
    ax.plot(T_zoom_data, np.rad2deg(X_nonlin[mask, sv_q]), 'b-', linewidth=2.5, label='Non-lineaire')
    ax.plot(T_zoom_data, np.rad2deg(X_lin[mask, sv_q]), 'r--', linewidth=2.5, label='Linearise')
    ax.set_ylabel('q (deg/s)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temps (s)', fontsize=14, fontweight='bold')
    ax.set_title(f'Vitesse de tangage - ms = {ms}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('slide_9_zoom_20s_comparison.png', dpi=150, bbox_inches='tight')
print(f'Figure slide_9_zoom_20s_comparison.png creee')
plt.show()

print(f'\n\n{"="*70}')
print(f'FIN DES SIMULATIONS - Personne 2')
print(f'{"="*70}')
print(f'\nFichiers generes:')
print(f'  - slide_7_ms_0.7_response_240s.png')
print(f'  - slide_8_ms_0.2_response_240s.png')
print(f'  - slide_9_zoom_20s_comparison.png')