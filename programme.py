import aero_model 
import numpy as np
import matplotlib.pyplot as plt


"""effet de ms sur Cm=f(alpha), expliquer pente et stabilité statique"""
aero_m = aero_model.Airbus_A321_200()
aero_m.set_options(stall=False, buffeting=False, wave_drag=False)
aero_m.set_mass(aero_m.m_design) 
mach, q, tas,dtrim, dm = aero_m.mach_design, 0, aero_m.atm.tas_from_mach_altp(aero_m.mach_design, aero_m.altp_ref),0, 0        #la TAS ne doit pas être nulle
aoa_range = np.linspace(np.deg2rad(-10), np.deg2rad(20), 100)
for ms in [0.7,0.2,0,-0.3]:
    aero_m.set_static_margin(ms)
    cm_list = []
    for aoa in aoa_range:
        cz, cx, cm = aero_m.get_aero_coefs(aoa, mach, dtrim, dm, q, tas)
        cm_list.append(cm)
        # Tracé
    plt.plot(np.rad2deg(aoa_range), cm_list,
             label=f"ms = {ms}")

plt.xlabel("Angle d'attaque α (deg)")
plt.ylabel("Coefficient de moment de tangage Cm")
plt.title("Cm en fonction de α pour différents ms")
plt.grid(True)
plt.legend()
plt.show()




"""Effet de ms sur delta trim,e : graphe deltatrim,e= f(alpha) pour les mêmes valeurs de ms, 
expliquer que ms est faible  gouverne très déflectée  plus de trainée"""
aero_m = aero_model.Airbus_A321_200()
aero_m.set_options(stall=False, buffeting=False, wave_drag=False)
aero_m.set_mass(aero_m.m_design)

mach,q,tas,dm = aero_m.mach_design,0, aero_m.atm.tas_from_mach_altp(aero_m.mach_design, aero_m.altp_ref),0
aoa_range = np.linspace(np.deg2rad(-10), np.deg2rad(20), 100)

for ms in [0.7, 0.2, 0, -0.3]:
    aero_m.set_static_margin(ms)
    dtrim_list = []
    cm_list = []
    
    for aoa in aoa_range:
        # trim condition Cm = 0
        dtrim_e = -(aero_m.cm0 + aero_m.cma * (aoa - aero_m.a0)) / aero_m.cmtrim

        dtrim_list.append(np.rad2deg(dtrim_e))

        # calcul Cm avec trim équilibré (doit être ~0)
        cz, cx, cm = aero_m.get_aero_coefs(aoa, mach, dtrim_e, dm, q, tas)
        cm_list.append(cm)

    plt.plot(np.rad2deg(aoa_range), dtrim_list, label=f"ms = {ms}")

plt.xlabel("Incidence α (deg)")
plt.ylabel("δtrim,e (deg)")
plt.title("Commande de trim d’équilibre en fonction de α")
plt.grid(True)
plt.legend()
plt.show()
