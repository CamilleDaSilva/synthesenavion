import numpy as np
import units, atmosphere

##conclusion qui élargie 

def lift_gradiant(ar):
    """Helmbold formula for the lift gradiant of a lifting body with finite span

    :param ar: no_dim, Aspect ratio of the lifting surface
    :return: cza: 1/rad, Lif gradient of the lifting surface
    """
    cza = (np.pi * ar) / (1 + np.sqrt(1 + (ar / 2) ** 2))
    return cza

def wake_deflection(ar, sa, a0, cza):
    """Compute wake deflection angle behind a wing of given aspect ratio
    
    eps0 : rad, deflection when airplane_aoa = 0
    deps : no_dim, deflection derivative versus airplane aoa
    """
    deps = 2 * cza / (np.pi * ar)
    eps0 = deps * (sa - a0)
    return eps0, deps

class Wing(object):
    """This class packages all data related to Wing (+Fuselage FIXME: really??)  body
    """
    def __init__(self,
                 s = 122.6,
                 ar = 9.4,
                 mac = 3.5,
                 dhdr = units.convert_from("deg", 2),
                 set = units.convert_from("deg", 0),
                 a0 = units.convert_from("deg", -3),
                 cm0 = 0,
                 czmax = 1.4,
                 rxf = 0.25):

        self.s = s          # Plan form area
        self.ar = ar        # Aspect ratio
        self.mac = mac      # Mean aero chord
        self.dhdr = dhdr    # Dihedral angle
        self.set = set      # Wing setting angle versus fuselage

        self.a0 = a0        # Wing & fuselage zero lift angle
        self.cm0 = cm0      # Zero lift moment

        self.czmax = czmax
        self.dczm = -0.2
        self.kczm = 1

        self.rxf = rxf                          # Wing lift center relative position versus MAC
        self.xf = - self.mac * self.rxf         # X_wise wing lift center position,  reference point is 0% MAC
        self.span = np.sqrt(self.s * self.ar)   # Span

        self.cza = lift_gradiant(self.ar)                                               # Lift gradient
        self.eps0, self.deps = wake_deflection(self.ar, self.set, self.a0, self.cza)    # Deflection

        # Buffeting effect correction
        self.buffet_fac = np.poly1d([  44.33942717, -133.55785435,  147.53451342,  -73.80265249,   15.50106458,   -1.11246361, 1.01117994])


class HorizTailPlane(object):
    """This class packages all data related to Horizontal tail body
    """
    def __init__(self,
                 wing_data,
                 s = 31,
                 ar = 5,
                 mac = 2.5,
                 la = 17,
                 hrcp = 0.25,
                 rxf = 0.25):

        self.w = wing_data  # Wing class

        self.s = s          # Plan form area
        self.ar = ar        # Aspect ratio
        self.mac = mac      # Mean aero chord
        self.la = la        # Lever arm : distance : w.xf - h.xf
        self.hrcp = hrcp                # Relative chord of the elevator
        self.rxf = rxf                  # HTP lift center relative position versus its own MAC
        self.xf = self.w.xf - self.la   # X-wise HTP lift center position

        self.cza = (self.s / self.w.s) * lift_gradiant(self.ar)  # Referenced versus wing
        self.czdm = self.cza * self.hrcp
        self.ki = (1.15 / (np.pi * self.ar)) / (self.s / self.w.s)**2   # ki will be multiplied by the square of cz_htp in wing reference

#
# Generic airplane, see below for specific configurations
#
class Airplane(object):
    """Assembles wing and horizontal tail plane to provide airplane level coefficients and methods
    """
    def __init__(self,
                 wing_data,
                 htp_data,
                 mtom = 77000,
                 oem = 42100,
                 disa_ref = 0,
                 altp_ref = units.convert_from("ft", 35000),
                 mach_design = 0.78,
                 lod_design = 17,
                 cxc_design = 0.0007,
                 dzf = 1.5,
                 slst = 111205,
                 ne = 2,
                 stall = False,
                 buffeting = False,
                 wave_drag = False,
                 atm=None):

        self.atm = atmosphere.AtmosphereISA() if atm is None else atm
        self.set_options(stall, buffeting, wave_drag)

        self.w = wing_data  # Wing class
        self.h = htp_data   # HTP class

        self.mtom = mtom    # Maximum Take Off Mass
        self.oem = oem      # Operating Empty Mass

        self.disa_ref = disa_ref
        self.altp_ref = altp_ref
        pamb, tamb = self.atm.atmosphere(self.altp_ref)

        self.mach_design = mach_design      # Cruise mach
        self.m_design = 0.97 * self.mtom    # Total mass for reference aero point
        self.cz_design = (self.m_design * self.atm.g) / (0.7 * pamb * self.w.s * self.mach_design**2)     # Cz reference aero point
        self.re_design = self.atm.reynolds_number(pamb, tamb, self.mach_design)                           # Re reference

        self.lod_design = lod_design    # L/D reference point
        self.cxc_design = cxc_design    # Compressibility drag at reference aero point

        self.ki = 1.07 / (np.pi * self.w.ar)
        self.cx0 = self.cz_design * (1/self.lod_design - self.ki * self.cz_design) - self.cxc_design      # Reverse computation
        self.d_mach_div = np.log(0.0025/self.cxc_design) / 40  # estimate mach_div to have cxc_ref at cruise mach (mach_div = mach_design + d_mach_div)

        self.dzf = dzf          # Engine Z-wise position versus CG (Z oriented downward)
        self.n_engine = ne      # Number of engine
        self.slst = slst        # Sea Level Static Thrust, One engine

        
        self.m = None           # Airplane mass
        self.xg = None          # X-wise CG position
        self.iyy = None         # Y-wise inertia moment

        self.cza = None         # Lift coefficient gradient versus AoA
        self.a0 = None          # Zero lift angle of attack
        self.cztrim = None      # Lift coefficient gradient versus dtrim angle
        self.czdm = None        # Lift coefficient gradient versus elevator angle
        self.czq = None         # Lift coefficient gradient versus pitching speed
        self.cm0 = None         # Zero lift moment coefficient
        self.cma = None         # Moment coefficient gradient versus AoA
        self.cmtrim = None      # Moment coefficient gradient versus dtrim angle
        self.cmdm = None        # Moment coefficient gradient versus elevator angle
        self.cmq = None         # Moment coefficient gradient versus pitching speed
        self.xf = None          # X_wise position of the neutral point
        self.xmr = None         # X_wise position of the resource maneuvre point
        self.xmt = None         # X_wise position of the turn maneuvre point

        # static margin
        self.czawh = self.w.cza + (1 - self.w.deps)* self.h.cza  # lift coef for wing plus tail
        self.xfwh = (self.w.cza*self.w.xf+(1 - self.w.deps)*self.h.cza*self.h.xf)/self.czawh # lift + tail neutral point
        self.rxfwh = self.xfwh/self.w.mac # relative lift + tail neutral point
        
        self.set_mass_and_balance(self.m_design, 0.2)
        
    def get_mass(self): return self.m
    def set_mass(self, mass): return self.set_mass_and_balance(mass, self.get_balance())
    # Dimensionless mass coefficient (0: empty airplane, 1: full airplane)
    def set_mass_from_km(self, km):
        mass = (1 - km) * self.oem + km * self.mtom
        self.set_mass_and_balance(mass, self.get_balance())
        return mass

    # Dimensionless balance, referenced to the leading edge
    def get_balance(self): return self.rxg
    def set_balance(self, rxg): return self.set_mass_and_balance(self.get_mass(), rxg)
    # Dimensionless balance, referenced to neutral point 
    def get_static_margin(self): return -(self.rxg+self.rxfwh)
    def set_static_margin(self, sm): return self.set_balance(-sm-self.rxfwh)
    
    def set_options(self, stall, buffeting, wave_drag):
        self.stall = stall          # if True: lift is limited by czmax consequently, lift curve is not linear, if False: lift is linear with angle of attack
        self.buffeting = buffeting  # if True: czmax is driven by buffeting, if False: czmax is constant
        self.wave_drag = wave_drag  # if True: compressibility drag is taken into account, if False: only form, friction and induced drag is taken into acount
    
    def get_aero_coefs(self, aoa, mach, dtrim, dm, q, tas):
        cz, ax, eps, cz_wf, cz_ht, czmax = self.get_cz(aoa, mach, dtrim, dm, q, tas)
        cx = self.get_cx(ax, aoa, eps, cz_wf, cz_ht, mach, self.re_design)
        cm = self.get_cm(aoa, dtrim, dm, q, tas)
        return cz, cx, cm

    def get_aero_forces(self, aoa, mach, dtrim, dm, q, tas, h):
        aero_coefs = np.array(self.get_aero_coefs(aoa, mach, dtrim, dm, q, tas)) # cz, cx, cm
        rho = self.atm.air_density(*self.atm.atmosphere_geo(h))
        pdyn = 0.5 * rho * tas ** 2
        LDM = pdyn * self.w.s * aero_coefs
        LDM[2] *= self.w.mac
        return LDM
        

        
    def set_mass_and_balance(self, mass, rxg):
        """Compute airplane aerodynamic coefficients

        VERY IMPORTANT : Reference point is 0% MAC

        :param mass: kg, current airplane mass
        :param rxg: no_dim, Position of the Center of Gravity given as a roportion of the MAC
        :return:
        dictionary with aerodynamic coefficients in airplane reference
        """
        # Inertia matrix
        self.m = mass
        self.iyy = 0.12 * (self.w.s / self.w.mac) * self.m
        
        # Center of gravity
        self.rxg = rxg
        self.xg = - self.w.mac * rxg    # Reference point is at 0% MAC, X axis is oriented forward
        
        # Aerodynamic coefficients
        # Replace self.xg by self.w.xf to turn coefficients independent from CG position, EXCEPT cma
        # FIXME: does not need to be recomputed when changing mass nor balance???
        self.cza = self.w.cza + self.h.cza * (1 - self.w.deps)

        self.a0 = (self.w.cza * (self.w.a0 - self.w.set) + self.h.cza * self.w.eps0) / (self.w.cza + self.h.cza * (1 - self.w.deps))
        self.cztrim = self.h.cza
        self.czdm = self.h.czdm
        self.czq = - self.h.cza * (self.h.xf - self.xg) / self.w.mac
        self.cma = self.w.cza * (self.w.xf - self.xg) / self.w.mac + self.h.cza * (1 - self.w.deps) * (self.h.xf - self.xg) / self.w.mac
        self.cmtrim = self.h.cza * (self.h.xf - self.xg) / self.w.mac
        self.cmdm = self.h.czdm * (self.h.xf - self.xg) / self.w.mac
        self.cm0 = self.w.cm0 + self.w.cza * (self.a0 + self.w.set - self.w.a0) * (self.h.xf - self.xg) / self.w.mac \
                   + self.h.cza * (self.a0 - self.w.eps0 - self.w.deps * self.a0) * (self.h.xf - self.xg) / self.w.mac
        self.cmq = -self.h.cza * ((self.h.xf - self.xg) / self.w.mac) ** 2
        self.xf = (self.w.cza * self.w.xf + self.h.cza * (1 - self.w.deps) * self.h.xf) / (self.w.cza + self.h.cza * (1 - self.w.deps))

      
        
        return {"mass": [self.m, "kg", 0],
                "xg": [self.xg, "m", 3],
                "iyy": [self.iyy, "kg.m2", 0],
                "cx0": [self.cx0, "no_dim", 4],
                "ki": [self.ki, "no_dim", 4],
                "a0": [self.a0, "deg"],
                "eps0": [self.w.eps0, "deg", 3],
                "deps": [self.w.deps, "no_dim", 3],
                "cza": [self.cza, "1/rad"],
                "cztrim": [self.cztrim, "1/rad"],
                "czdm": [self.czdm, "1/rad"],
                "czq": [self.czq, "1/rad"],
                "cm0": [self.cm0, "no_dim", 4],
                "cma": [self.cma, "1/rad"],
                "cmtrim": [self.cmtrim, "1/rad"],
                "cmdm": [self.cmdm, "1/rad"],
                "cmq": [self.cmq, "1/rad"],
                "xf": [self.xf, "m", 3]}

    def thrust(self, sigma, mach, dthr):
        """Total engine thrust (all engines)

        :param sigma: no_dim, rho / rho0
        :param mach:  mach, Mach number
        :param dthr:  no_dim, 0 <= dthr <= 1, throttle
        :return:
        fn: N, Total engine thrust
        """
        f = self.slst * self.n_engine * sigma**0.6 * (0.568 + 0.25*(1.2-mach)**3) * dthr
        return f

    def get_cz(self, aoa, mach, dtrim, dm, q, tas):
        cz_wf, eps, czmax, ax = self.get_cz_wing(aoa, mach)
        cz_ht = self.get_cz_tail(eps, aoa, dtrim, dm, q, tas)
        return cz_wf + cz_ht, ax, eps, cz_wf, cz_ht, czmax
    
    def get_cz_wing(self, aoa, mach):
        """Compute the wing+fuselage lift coefficient

        :param aoa: rad, airplane angle of attack
        :param mach: mach, Mach number
        :return:
        cz_wf: no_dim, lift coefficient for Wing+Fuselage
        eps: rad, wake deflection angle at tail level
        czmax: no_dim, Cz max voilure
        ax: rad, angle of attack of Czmax
        """
        if self.stall:
            if self.buffeting:
                czmax = self.w.czmax * self.w.buffet_fac(mach)
            else:
                czmax = self.w.czmax
            xczm = self.w.kczm * czmax + self.w.dczm
            fac = 4 * (xczm - czmax)
            ak = self.w.a0 - self.w.set + xczm / self.w.cza  # Angle of attack at which non-linearity is starting
            ax = ak - 0.5 * fac / self.w.cza    # Angle of attack at czmax (stalling angle)
            a = self.w.cza ** 2 / fac
            b = self.w.cza - (2 * self.w.cza ** 2 * ak) / fac
            c = xczm - ak * self.w.cza + (self.w.cza ** 2 * ak ** 2) / fac
            aoa_wing = aoa + self.w.set - self.w.a0
            if aoa <= ak:
                cz_wf = self.w.cza * aoa_wing   # Linear part
            else:
                cz_wf = (a*aoa + b)*aoa + c     # Quadratic part
            eps = cz_wf*2/(np.pi*self.w.ar) # wake deflection is linked to Wing lift coefficient : eps = cz_w * d(eps)/d(Cz)
        else:
            aoa_wing = aoa + self.w.set - self.w.a0
            cz_wf = self.w.cza * aoa_wing  # Linear part
            eps = cz_wf * 2 / (np.pi * self.w.ar)  # wake deflection is linked to Wing lift coefficient : eps = cz_w * d(eps)/d(Cz)
            czmax = np.nan
            ax = np.nan
        return cz_wf, eps, czmax, ax

    def get_cz_tail(self, eps, aoa, dtrim, dm, q, tas):
        """Compute the lift coefficient of the horizontal tail

        :param eps: rad, wake deflection created by the wing at tail level
        :param aoa: rad, airplane angle of attack
        :param dtrim: rad, horizontal tail setting
        :param dm: rad, elevator deflection
        :param q: rad/s, pitch rotation speed
        :param tas: m/s, true air speed
        :return:
        cz_ht: no_dim, Lift coefficient of horizontal tail
        """
        cz_ht = self.h.cza * (aoa + dtrim - eps) + self.czdm * dm + self.czq * (self.w.mac / tas) * q
        return cz_ht

    def get_cx(self, ax, aoa, eps, cz_wf, cz_ht, mach, re):
        """Compute the total drag coefficient for the airplane

        :param ax: rad, angle of attack of Czmax
        :param aoa: rad, airplane angle of attack
        :param eps: rad, wake deflection created by the wing at tail level
        :param cz_wf: no_dim, lift coefficient for Wing+Fuselage
        :param cz_ht: no_dim, Lift coefficient of horizontal tail
        :param mach: mach, Mach number
        :param re: 1/m, metric Reynolds number in current conditions
        :return:
        cx: no_dim, total drag coefficient
        """
        # Reynolds effect on form & friction drag
        cx0 = self.cx0 * (np.log(self.re_design * self.w.mac) / np.log(re * self.w.mac))**2.58

        # Speed independent drag : form & friction + induced + dtrim + interaction
        cx_uc = cx0 + self.ki * cz_wf ** 2 + self.h.ki * cz_ht ** 2 + cz_ht * eps

        # Compressibility drag
        if self.wave_drag:
            mach_div = self.mach_design + (self.d_mach_div + 0.1 * (self.cz_design - cz_wf))
            cxc = 0.0025 * np.exp(40. * (mach - mach_div))  # To have 10% slope and cxc = 25 cts at Mach div
        else:
            cxc = 0

        # During stall and after
        cxmax = 1  # cx when AoA is getting close to 90°
        delta_aoa = np.deg2rad(0.75) # the charate of the transition interval
        if self.stall:
            k = 1 / (1 + np.exp(-(aoa - (ax + units.convert_from("deg", 0.75)))/delta_aoa))     # Sigmoid
            #k = 1 / (1 + np.exp(-30 * (aoa - (ax + units.convert_from("deg", 10)))))     # Sigmoid
        else:
            k = 0

        cx = (cx_uc + cxc) * (1 - k) + cxmax * k
        return cx

    def get_cm(self, aoa, dtrim, dm, q, tas):
        """Compute the total pitch moment coefficient of the airplane

        :param aoa: rad, airplane angle of attack
        :param dtrim: rad, horizontal tail setting
        :param dm: rad, elevator deflection
        :param q: rad/s, pitch rotation speed
        :param tas: m/s, true air speed
        :return:
        cm: no_dim, total pitch moment coefficient
        """
        cm = self.cm0 + self.cma * (aoa - self.a0) + self.cmtrim * dtrim + self.cmdm * dm + self.cmq * (self.w.mac / tas) * q
        return cm


class Airbus_A321_200(Airplane):
    def __init__(self, atm=None):
        self.name = "Airbus_A321_200"

        wing = Wing(s = 128.0,    # Wing reference area
                        ar = 10.0,    # Wing aspect ratio
                        mac = 4.31,   # Wing mean aerodynamic chord
                        dhdr = units.convert_from("deg", 2),   # Wing dihedral
                        set = units.convert_from("deg", 5),    # Wing setting angle
                        a0 = units.convert_from("deg", -3),    # Wing zero lift angle
                        cm0 = 0,      # Wing zero lift moment
                        czmax = 1.6,  # Wing maximum lift coefficient
                        rxf = 0.25)   # Wing neutral point relative position versus MAC
        
        horiz_tp = HorizTailPlane(wing,
                           s = 31,        # HTP reference area
                           ar = 5,        # HTP aspect ratio
                           mac = 2.49,    # HTP mean aerodynamic chord
                           la = 22.25,    # HTP lever arm (25% wing MAC - 25% HTP MAC)
                           hrcp = 0.30,   # HTP elevator chord ratio
                           rxf = 0.25)    # HTP neutral point relative position versus MAC
        
        Airplane.__init__(self, wing, horiz_tp,
                              mtom = 93500,         # kg, Max Take Off Mass
                              oem = 48300,          # kg, Operating Empty Mass
                              disa_ref = 0,         # Reference temperature shift versus ISA
                              altp_ref = units.convert_from("ft", 35000),    # Design cruise altitude
                              mach_design = 0.78,   # Design cruise Mach number
                              lod_design = 17.5,    # Design L/D at cruise altitude and 97% MTOM
                              cxc_design = 0.0008,  # Design compressibility drag coefficient
                              dzf = 0,              # 1.5 m, Vertical engine thrust lever arm versus CG
                              slst = 133446,        # Sea Level Static Thrust
                              ne = 2,
                              atm=atm) 
