#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT, Dajung KIM
         Aircraft & Systems, Air Transport Department, ENAC
         Antoine Drouin, Thierry Miquel, SYSDYN, OPTIM, ENAC
"""

import numpy as np
import scipy.optimize

class AtmosphereISA(object):
    """Provide all necessary models related to atmosphere
    """
    def __init__(self, disa=0.):
        self.g = 9.80665    # Gravity acceleration at sea level

        self.rho0 = 1.225   # (kg/m3), air density at sea level
        self.P0 = 101325.   # , standard pressure at sea level
        self.T0 = 288.15    # Kelvins, standard temperature
        self.vc0 = 340.29   # m/s, sound speed at sea level

        self.r = 287.053
        self.gam = 1.40
        self.cv = self.r / (self.gam - 1.)
        self.cp = self.gam * self.cv

        # Mixed gas dynamic viscosity, Sutherland's formula
        self.mu0 = 1.715e-5
        self.Tref = 273.15
        self.S = 110.4

        # Standard atmosphere model
        self.Z = np.array([0., 11000., 20000., 32000., 47000., 50000.])
        self.dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])

        self.P = np.array([self.P0, 0., 0., 0., 0., 0.])
        self.T = np.array([self.T0, 0., 0., 0., 0., 0.])

        for j in range(len(self.Z)-1):
            self.T[j + 1] = self.T[j] + self.dtodz[j] * (self.Z[j + 1] - self.Z[j])
            if (0. < np.abs(self.dtodz[j])):
                self.P[j + 1] = self.P[j] * (1. + (self.dtodz[j] / self.T[j]) * (self.Z[j + 1] - self.Z[j])) ** (-self.g / (self.r * self.dtodz[j]))
            else:
                self.P[j + 1] = self.P[j] * np.exp(-(self.g / self.r) * ((self.Z[j + 1] - self.Z[j]) / self.T[j]))
        self.disa = disa

                
    def gas_data(self):
        return self.r, self.gam, self.cp, self.cv

    def air_density(self, pamb, tamb):
        """Ideal gas density
        """
        r, gam, Cp, Cv = self.gas_data()
        rho0 = self.rho0
        rho = pamb / (r * tamb)
        return rho

    def sound_speed(self, tamb):
        """Sound speed for ideal gas
        """
        r, gam, Cp, Cv = self.gas_data()
        vsnd = np.sqrt(gam * r * tamb)
        return vsnd

    def gas_viscosity(self, tamb):
        """Mixed gas dynamic viscosity, Sutherland's formula
        """
        return (self.mu0 * ((self.Tref + self.S) / (tamb + self.S)) * (tamb / self.Tref) ** 1.5)

    def reynolds_number(self, pamb, tamb, mach):
        """Reynolds number based on Sutherland viscosity model
        """
        vsnd = self.sound_speed(tamb)
        rho = self.air_density(pamb, tamb)
        mu = self.gas_viscosity(tamb)
        re = rho * vsnd * mach / mu
        return re

    def pressure_altitude(self, pamb):
        """Pressure altitude from ground to 50 km
        """
        g = self.g
        R, gam, Cp, Cv = self.gas_data()

        if (pamb < self.P[-1]):
            raise Exception("pressure_altitude, altitude cannot exceed 50km")

        j = np.searchsorted(-self.P[1:], -pamb, side="right")

        if (0. < np.abs(self.dtodz[j])):
            altp = self.Z[j] + ((pamb / self.P[j]) ** (-(R * self.dtodz[j]) / g) - 1) * (self.T[j] / self.dtodz[j])
        else:
            altp = self.Z[j] - (self.T[j] / (g / R)) * np.log(pamb / self.P[j])

        return altp

    def atmosphere(self, altp):
        """Pressure and temperature from pressure altitude from ground to 50 km
        """
        g = self.g
        R, gam, Cp, Cv = self.gas_data()

        if (self.Z[-1] < altp):
            raise Exception("pressure_altitude, altitude cannot exceed 50km")

        j = np.searchsorted(self.Z[1:], altp, side="right")

        if (0. < np.abs(self.dtodz[j])):
            pamb = self.P[j] * (1 + (self.dtodz[j] / self.T[j]) * (altp - self.Z[j])) ** (-g / (R * self.dtodz[j]))
        else:
            pamb = self.P[j] * np.exp(-(g / R) * ((altp - self.Z[j]) / self.T[j]))
        tstd = self.T[j] + self.dtodz[j] * (altp - self.Z[j])
        tamb = tstd + self.disa

        return pamb, tamb

    def atmosphere_geo(self, altg):
        """Pressure and temperature from geometrical altitude (height) from ground to 50 km
        """
        g = self.g
        R, gam, Cp, Cv = self.gas_data()

        Z = np.zeros_like(self.Z)
        dtodz = np.zeros_like(self.dtodz)
        P = np.zeros_like(self.P)
        T = np.zeros_like(self.T)

        P[0] = self.P0
        T[0] = self.T0

        for j in range(len(self.Z)-1):
            K = 1 + self.disa / self.T[j]
            dtodz[j] = self.dtodz[j] / K
            Z[j + 1] = Z[j] + (self.Z[j + 1] - self.Z[j]) * K

            T[j + 1] = T[j] + dtodz[j] * (Z[j + 1] - Z[j])
            if (0. < np.abs(dtodz[j])):
                P[j + 1] = P[j] * (1. + (dtodz[j] / (T[j] + self.disa)) * (Z[j + 1] - Z[j])) ** (-g / (R * dtodz[j]))
            else:
                P[j + 1] = P[j] * np.exp(-(g / R) * ((Z[j + 1] - Z[j]) / (T[j] + self.disa)))

        if (Z[-1] < altg):
            raise Exception("atmosphere_geo, altitude cannot exceed 50km")

        j = np.searchsorted(self.Z[1:], altg, side="right")

        if (0. < np.abs(dtodz[j])):
            pamb = P[j] * (1 + (dtodz[j] / (T[j] + self.disa)) * (altg - Z[j])) ** (-g / (R * dtodz[j]))
        else:
            pamb = P[j] * np.exp(-(g / R) * ((altg - Z[j]) / (T[j] + self.disa)))
        tamb = T[j] + dtodz[j] * (altg - Z[j]) + self.disa

        return pamb, tamb

    def altg_from_altp(self, altp):
        """Geometrical altitude (height) from pressure altitude
        """
        def fct(altg, altp):
            pamb, tamb = self.atmosphere_geo(altg)
            zp = self.pressure_altitude(pamb)
            return altp - zp

        output_dict = scipy.optimize.fsolve(fct, x0=altp, args=(altp), full_output=True)

        altg = output_dict[0][0]
        if (output_dict[2] != 1): raise Exception("Convergence problem")

        return altg

    def tas_from_mach_altp(self, mach, altp):
        pamb, tamb = self.atmosphere(altp)
        return mach * self.sound_speed(tamb)

    def tas_from_mach_altg(self, mach, altg):
        return self.tas_from_mach_altp(mach, self.altg_from_altp(altg))
