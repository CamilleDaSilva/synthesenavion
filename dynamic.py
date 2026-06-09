#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT, Dajung KIM
         Aircraft & Systems, Air Transport Department, ENAC
         Antoine Drouin, Thierry Miquel, Sysdyn, Optim Department, ENAC
"""

import numpy as np, scipy.optimize


import atmosphere, display_utils, units
sv_size = 7
sv_aoa, sv_q, sv_tas, sv_path, sv_height, sv_xpos, sv_mass = range(sv_size)
iv_size = 3
iv_dtrim, iv_dm, iv_dthr = range(iv_size)

#=======================================================================================================================
#
#  Dynamic model
#
#=======================================================================================================================

def get_state_dot(X, t, U, aero_m):
    """Compute state vector derivative
    """
    # State vector :
    # aoa: rad, angle of attack
    # q: rad/s, pitch rotation speed
    # tas: m/s, true air speed
    # path: rad, trajectory slope
    # height: m, geometrical altitude
    # xpos: m, ground position along the trajectory
    # mass: kg, airplane mass
    aoa, q, tas, path, height, xpos, mass = list(X)     # State vector
    dtrim, dm, dthr = list(U)

    # Get longitudinal coefficients
    aero_m.set_mass_and_balance(mass, aero_m.rxg)

    # Compute atmospheric data
    pamb, tamb = aero_m.atm.atmosphere_geo(height)
    rho = aero_m.atm.air_density(pamb, tamb)
    sigma = rho / aero_m.atm.rho0
    mach = tas / aero_m.atm.sound_speed(tamb)
    re = aero_m.atm.reynolds_number(pamb, tamb, mach)
    pdyn = 0.5 * rho * tas ** 2

    # Compute aerodynamic coefficients
    cz, cx, cm = aero_m.get_aero_coefs(aoa, mach, dtrim, dm, q, tas)
    
    fn = aero_m.thrust(sigma, mach, dthr)            # Get usable thrust (all engines)

    pitch = pdyn * aero_m.w.s * aero_m.w.mac * cm    # Pitch moment
    drag = pdyn * aero_m.w.s * cx                    # Drag force
    lift = pdyn * aero_m.w.s * cz                    # Lift force

    teta_dot = q                                                                        # Valid only within longitudinal flight
    q_dot = (aero_m.dzf * fn + pitch) / aero_m.iyy                                                # Applying FPD around pitch axis
    tas_dot = (fn * np.cos(aoa) - drag - aero_m.m * aero_m.atm.g * np.sin(path)) / aero_m.m              # Applying FPD along velocity axis
    path_dot = (lift + fn * np.sin(aoa) - aero_m.m * aero_m.atm.g * np.cos(path)) / (aero_m.m * tas)     # Applying FPD perpendicularly to velocity axis
    aoa_dot = teta_dot - path_dot                                                       # Valid only within longitudinal flight
    height_dot = tas * np.sin(path)                                                     # Vertical speed projection on ground axis
    xpos_dot = tas * np.cos(path)                                                       # Horizontal speed projection on ground axis
    m_dot = 0

    # State vector derivative
    X_dot = np.array([aoa_dot, q_dot, tas_dot, path_dot, height_dot, xpos_dot, m_dot])  # State vector derivative
    return X_dot


#https://aviation.stackexchange.com/questions/67897/what-are-the-maximum-possible-stabilizer-and-elevator-deflections-for-the-a320
#  The maximum trimmable horizontal stabiliser deflection is separately 13.5º nose up to 4º nose dow
def get_trim_level_flight(aero_m, altp, tas, use_saturations=False):
    """Compute level flight equilibrium using dtrim
    """
    # Compute atmospheric data
    pamb, tamb = aero_m.atm.atmosphere(altp)
    rho = aero_m.atm.air_density(pamb, tamb)
    sigma = rho / aero_m.atm.rho0
    mach = tas / aero_m.atm.sound_speed(tamb)
    re = aero_m.atm.reynolds_number(pamb, tamb, mach)
    pdyn = 0.5 * rho * tas ** 2


    dm, nz, q, path = 0., 1, 0., 0.     # Steady level flight

    def fct(x):
        dthr, aoa, dtrim = x[0], x[1], x[2]   # Variables to be computed: throttle, angle of attack, dtrim

        # Compute aerodynamic coefficients
        cz, cx, cm = aero_m.get_aero_coefs(aoa, mach, dtrim, dm, q, tas)
        
        fn = aero_m.thrust(sigma, mach, dthr)            # Get usable thrust (all engines)

        lift = pdyn * aero_m.w.s * cz                    # Lift force (FIXME: not the sum of wing + tail surface?)
        drag = pdyn * aero_m.w.s * cx                    # Drag force
        pitch = pdyn * aero_m.w.s * aero_m.w.mac * cm    # Pitch moment

        y1 = fn * np.cos(aoa) - drag                        # Thrust balances drag (supposing thrust is parallel to speed)
        y2 = nz * aero_m.get_mass() * aero_m.atm.g - lift - fn * np.sin(aoa)    # Lift balances weight
        y3 = aero_m.dzf * fn + pitch                             # Pitch moment balances thrust moment

        return [y1, y2, y3]

    xini = [0, 0, 0]
    output_dict = scipy.optimize.fsolve(fct, x0=xini, args=(), full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    dthr = output_dict[0][0]    # Throttle
    aoa = output_dict[0][1]     # Angle of attack
    dtrim = output_dict[0][2]   # dtrim
    if use_saturations:
        if dthr<0 or dthr>1.: raise Exception("Throttle saturation")
        if dtrim < np.deg2rad(-13.5) or dtrim > np.deg2rad(4.): raise Exception("HTP saturation")
    
    
    # Update aerodynamic coefficients
    cz, cx, cm = aero_m.get_aero_coefs(aoa, mach, dtrim, dm, q, tas)

    # Update usable thrust
    fn = aero_m.thrust(sigma, mach, dthr)

    return {"dthr": [dthr, "no_dim"],
            "aoa": [aoa, "deg"],
            "dtrim": [dtrim, "deg"],
            "dm": [dm, "deg"],
            "cz": [cz, "no_dim"],
            "cx": [cx, "no_dim", 4],
            "cm": [cm, "no_dim", 4],
            "fn": [fn, "daN"],
            "lod": [cz/cx, "no_dim"] }#,
            #"czmax": [czmax, "no_dim"]}


def graceful_trim(aero_m, h, tas, use_saturations=False):
    try:
        res = get_trim_level_flight(aero_m, h, tas, use_saturations)
        return [res['aoa'][0], res['dtrim'][0], res['dthr'][0]]
    except:
        return [np.nan, np.nan, np.nan]

def graceful_trim2(aero_m, h, mach, use_saturations=False):
    return graceful_trim(aero_m, h, aero_m.atm.tas_from_mach_altp(mach, h), use_saturations)
    
"""
Compute numerical jacobian 
"""
def num_jacobian(X, U, aero_m, _t=0.):
    s_size, i_size = len(X), len(U)
    epsilonX = (0.1*np.ones(s_size)).tolist()
    dX = np.diag(epsilonX)
    A = np.zeros((s_size, s_size))
    for i in range(0, s_size):
        dx = dX[i,:]
        delta_f = get_state_dot(X+dx/2, _t, U, aero_m) - get_state_dot(X-dx/2, _t, U, aero_m)
        delta_f = delta_f / dx[i]
        A[:,i] = delta_f
    epsilonU = (0.1*np.ones(i_size)).tolist()
    dU = np.diag(epsilonU)
    B = np.zeros((s_size,i_size))
    for i in range(0, i_size):
        du = dU[i,:]
        delta_f = get_state_dot(X, _t, U+du/2, aero_m) - get_state_dot(X, _t, U-du/2, aero_m)
        delta_f = delta_f / du[i]
        B[:,i] = delta_f
    return A,B



def plot_simulation(time, X, U, figure=None, axes=None, window_title=None, label=''):
    figure = display_utils.prepare_fig(figure, window_title=window_title)
    axes = figure.subplots(5,1, sharex=True) if axes is None else axes
    #sv_aoa, sv_q, sv_tas, sv_path, sv_height, sv_xpos, sv_mass = range(sv_size)
    axes[0].plot(time, np.rad2deg(X[:, sv_aoa]), label=label)
    display_utils.decorate(axes[0], 'aoa', ylab='deg', min_yspan=0.1)
    axes[1].plot(time, np.rad2deg(X[:, sv_q]), label=label)
    display_utils.decorate(axes[1], 'q', ylab='deg/s', min_yspan=0.1)
    axes[2].plot(time, X[:, sv_tas], label=label)
    display_utils.decorate(axes[2], 'tas', ylab='m/s', min_yspan=0.5)
    axes[3].plot(time, np.rad2deg(X[:, sv_path]), label=label)
    display_utils.decorate(axes[3], '$\\gamma$', ylab='deg', min_yspan=0.1)
    axes[4].plot(time, X[:, sv_height], label=label)
    display_utils.decorate(axes[4], 'h', 'time in s', 'm', min_yspan=1.)


    return figure, axes



