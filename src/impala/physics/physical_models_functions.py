"""
physical_models_functions.py

    A module conatining standalone functions that are used as subroutines
    for physical_models_functions.py, a module for
    material strength behavior to be imported into python scripts for
    optimizaton or training emulators. All functions that depend
    on temperature in this module expect a numpy array of temperatures.

    Authors:
        DJ Luscher,    djl@lanl.gov
        Peter Trubey,  ptrubey@lanl.gov
        Devin Francom, dfrancom@lanl.gov
        JeeYeon Plohr, jplohr@lanl.gov
        Sky Sjue, sjue@lanl.gov
        Lauren VanDervort, @lvandervort@lanl.gov
        Daniel N Blaschke, dblaschke@lanl.gov
"""

import math
from math import pi

import numpy as np

try:
    from numba import jit

    @jit(nopython=True)
    def erf(x):
        """numba.jit-compatible erf."""
        out = np.empty(x.shape)
        for i, xi in enumerate(x):
            out[i] = math.erf(xi)
        return out

except ImportError:
    from functools import partial

    from scipy.special import erf

    def jit(func=None, forceobj=True, nopython=False):
        """Dummy decorator if numba is unavailable at runtime."""
        return func or partial(jit, forceobj=forceobj, nopython=nopython)


## constants
Avogadro = 6.022e23
cbrtAvogadro = pow(Avogadro, 1.0 / 3.0)

########################
# Specific Heat Models
########################


def Linear_Specific_Heat(c0: float, c1: float, T: float) -> float:
    """
    Linear Specific Heat Model

    T: temperature
    c0, c1: model parameters
    """
    return c0 + c1 * T


@jit(nopython=True)
def Quadratic_Specific_Heat(c0: float, c1: float, c2: float, T: float) -> float:
    """
    Quadratic Specific Heat Model

    T: temperature
    c0, c1, c2: model parameters
    """
    return c0 + c1 * T + c2 * T**2


@jit(nopython=True)
def Cubic_Specific_Heat(
    c0: float, c1: float, c2: float, c3: float, T: float
) -> float:
    """
    Cubic Specific Heat Model

    T: temperature
    c0, c1, c2, c3: model parameters
    """
    return c0 + c1 * T + c2 * T**2 + c3 * T**3


@jit(nopython=True)
def Piecewise_Linear_Specific_Heat(
    Tt: float, c00: float, c01: float, c10: float, c11: float, T: float
) -> float:
    """
    Piecewise Linear Specific Heat Model

    T: current temperature
    Tt: temperature separating the two linear models
    c00, c10: model parameters used up to temperature Tt
    c01, c11: model parameters used above Tt

    Cv (T) = c00 + c10 * T for T<=Tt
    Cv (T) = c01 + c11 * T for T>Tt
    """
    intercept = np.repeat(c00, len(T))
    slope = np.repeat(c10, len(T))
    intercept[np.where(T > Tt)] = c01
    slope[np.where(T > Tt)] = c11
    return intercept + slope * T


@jit(nopython=True)
def Piecewise_Quadratic_Specific_Heat(
    Tt: float,
    c00: float,
    c01: float,
    c10: float,
    c11: float,
    c20: float,
    c21: float,
    T: float,
) -> float:
    """
    Piecewise Quadratic Specific Heat Model

    T: current temperature
    Tt: temperature separating the two quadratic models
    c00, c10, c20: model parameters used up to temperature Tt
    c01, c11, c21: model parameters used above Tt

    Cv (T) = c00 + c10 * T + c20 * T**2 for T<=Tt
    Cv (T) = c01 + c11 * T + c21 * T**2 for T>Tt
    """
    pow_0_coeff = np.repeat(c00, len(T))
    pow_1_coeff = np.repeat(c10, len(T))
    pow_2_coeff = np.repeat(c20, len(T))

    pow_0_coeff[np.where(T > Tt)] = c01
    pow_1_coeff[np.where(T > Tt)] = c11
    pow_2_coeff[np.where(T > Tt)] = c21

    return pow_0_coeff + pow_1_coeff * T + pow_2_coeff * T * T


@jit(nopython=True)
def Piecewise_Cubic_Specific_Heat(
    Tt: float,
    c00: float,
    c01: float,
    c10: float,
    c11: float,
    c20: float,
    c21: float,
    c30: float,
    c31: float,
    T: float,
) -> float:
    """
    Piecewise Cubic Specific Heat Model

    T: current temperature
    Tt: temperature separating the two linear models
    c00, c10, c20, c30: model parameters used up to temperature Tt
    c01, c11, c21, c31: model parameters used above Tt

    Cv (T) = c0_0 + c1_0 * T + c2_0 * T**2  + c3_0 * T**3 for T<=T_t
    Cv (T) = c0_1 + c1_1 * T + c2_1 * T**2  + c3_1 * T**3 for T>T_t
    """

    pow_0cf = np.repeat(c00, len(T))
    pow_1cf = np.repeat(c10, len(T))
    pow_2cf = np.repeat(c20, len(T))
    pow_3cf = np.repeat(c30, len(T))

    pow_0cf[np.where(T > Tt)] = c01
    pow_1cf[np.where(T > Tt)] = c11
    pow_2cf[np.where(T > Tt)] = c21
    pow_3cf[np.where(T > Tt)] = c31

    return pow_0cf + pow_1cf * T + pow_2cf * T**2 + pow_3cf * T**3


########################
# Density Models
########################


def Linear_Density(r0: float, r1: float, T: float) -> float:
    """
    Linear Density Model

    T: temperature
    r0, r1: model parameters
    """
    return r0 + r1 * T


@jit(nopython=True)
def Quadratic_Density(r0: float, r1: float, r2: float, T: float) -> float:
    """
    Quadratic Density Model

    T: temperature
    r0, r1, r2: model parameters
    """
    return r0 + r1 * T + r2 * T**2


@jit(nopython=True)
def Cubic_Density(
    r0: float, r1: float, r2: float, r3: float, T: float
) -> float:
    """
    Cubic Density Model

    T: temperature
    r0, r1, r2, r3: model parameters
    """
    return r0 + r1 * T + r2 * T**2 + r3 * T**3


########################
# Melt Temperature Models
########################


def Linear_Melt_Temperature(tm0: float, tm1: float, rho: float) -> float:
    """
    Linear Melt Temperature Model

    rho: material density
    tm0, tm1: model parameters
    """
    return tm0 + tm1 * rho


@jit(nopython=True)
def Quadratic_Melt_Temperature(
    tm0: float, tm1: float, tm2: float, rho: float
) -> float:
    """
    Quadratic Melt Temperature Model

    rho: material density
    tm0, tm1, tm2: model parameters
    """
    return tm0 + tm1 * rho + tm2 * rho**2


@jit(nopython=True)
def Cubic_Melt_Temperature(
    tm0: float, tm1: float, tm2: float, tm3: float, rho: float
) -> float:
    """
    Cubic Melt Temperature Model

    rho: material density
    tm0, tm1, tm2, tm3: model parameters
    """
    return tm0 + tm1 * rho + tm2 * rho**2 + tm3 * rho**3


@jit(nopython=True)
def BGP_Melt_Temperature(
    Tm0: float, rhom: float, gamma1: float, gamma3: float, q3: float, rho: float
) -> float:
    """
    Burakovsky-Greeff-Preston Melt Temperature Model
    see doi.org/10.1103/PhysRevB.67.094107

    rho: current material density
    rhom: reference density
    Tm0: melt temperature at reference density rhom
    gamma1, gamma2, q3: model parameters
    """
    melt_temp = (
        Tm0
        * np.cbrt(rho / rhom)
        * np.exp(
            6 * gamma1 * (1 / np.cbrt(rhom) - 1 / np.cbrt(rho))
            + 2.0 * gamma3 / q3 * (np.power(rhom, -q3) - np.power(rho, -q3))
        )
    )
    return melt_temp


########################
# Shear Modulus Models
########################


@jit(nopython=True)
def Linear_Cold_PW_Shear_Modulus(
    g0: float, g1: float, alpha: float, rho: float, T: float, Tmelt: float
) -> float:
    """
    Linear Cold PW Shear Modulus

    rho: current material density
    T: temperature
    Tmelt: melting temperature
    g0, g1, alpha: model parameters
    """
    cold_shear = g0 + g1 * rho
    gnow = cold_shear * (1.0 - alpha * (T / Tmelt))

    gnow[np.where(T >= Tmelt)] = 0.0
    gnow[np.where(gnow < 0)] = 0.0

    return gnow


@jit(nopython=True)
def Quadratic_Cold_PW_Shear_Modulus(
    g0: float,
    g1: float,
    g2: float,
    alpha: float,
    rho: float,
    T: float,
    Tmelt: float,
) -> float:
    """
    Quadratic Cold PW Shear Modulus

    rho: current material density
    T: temperature
    Tmelt: melting temperature
    g0, g1, g2, alpha: model parameters
    """
    cold_shear = g0 + g1 * rho + g2 * rho**2
    gnow = cold_shear * (1.0 - alpha * (T / Tmelt))

    gnow[np.where(T >= Tmelt)] = 0.0
    gnow[np.where(gnow < 0)] = 0.0

    return gnow


@jit(nopython=True)
def Simple_Shear_Modulus(
    G0: float, alpha: float, T: float, Tmelt: float
) -> float:
    """
    Simple Shear Modulus

    T: temperature
    Tmelt: melting temperature
    G0, alpha: model parameters
    """
    return G0 * (1.0 - alpha * (T / Tmelt))


@jit(nopython=True)
def BGP_PW_Shear_Modulus(
    G0: float,
    rho_0: float,
    gamma_1: float,
    gamma_2: float,
    q2: float,
    alpha: float,
    rho: float,
    T: float,
    Tmelt: float,
) -> float:
    """
    BPG model provides cold shear, i.e. shear modulus at zero temperature as a function of density.
    PW describes the (linear) temperature dependence of the shear modulus. (Same dependency as
    in Simple_Shear_modulus.)
    With these two models combined, we get the shear modulus as a function of density and temperature;
    see Burakovsky, Greeff, Preston, Phys. Rev. B67 (2003) 094107, DOI:10.1103/PhysRevB.67.094107

    rho: current material density
    T: temperature
    Tmelt: melting temperature
    rho_0: reference density
    G0: shear modulues at reference density rho_0
    gamma1, gamma2, q2, alpha: model parameters
    """
    cold_shear = (
        G0
        * np.power(rho / rho_0, 4.0 / 3.0)
        * np.exp(
            6.0 * gamma_1 * (1 / np.cbrt(rho_0) - 1 / np.cbrt(rho))
            + 2 * gamma_2 / q2 * (np.power(rho_0, -q2) - np.power(rho, -q2))
        )
    )
    gnow = cold_shear * (1.0 - alpha * (T / Tmelt))

    gnow[np.where(T >= Tmelt)] = 0.0
    gnow[np.where(gnow < 0)] = 0.0
    return gnow


@jit(nopython=True)
def Stein_Shear_Modulus(G0: float, sgB: float, T: float, Tmelt: float) -> float:
    """
    Stein Shear Modulus assuming constant density and pressure,
    so we only include the temperature dependence;
    including aterm = a/eta**(1.0/3.0)*pressure here just for completeness
    and setting aterm = 0

    T: temperature
    Tmelt: melting temperature
    G0, sgB: model parameters
    """
    aterm = 0.0
    bterm = sgB * (T - 300.0)
    gnow = G0 * (1.0 + aterm - bterm)
    gnow[np.where(T >= Tmelt)] = 0.0
    gnow[np.where(gnow < 0)] = 0.0
    return gnow


########################
# Yield Stress Models
########################


@jit(nopython=True)
def pos(a):
    """returns array a with all negative values set to 0"""
    return np.maximum(0, a)


@jit(nopython=True)
def JC_Yield_Stress(
    edot: float,
    A: float,
    B: float,
    C: float,
    n: float,
    m: float,
    Tref: float,
    edot0: float,
    eps: float,
    T: float,
    Tmelt: float,
) -> float:
    """
    JC Yield Stress

    eps: current strain
    T: temperature
    Tmelt: melting temperature
    edot0, Tref: reference strain rate and temperature
    A, B, C, n, m: model parameters
    """
    th = pos((T - Tref) / (Tmelt - Tref))

    Y = (
        (A + B * np.power(eps, n))
        * (1.0 + C * np.log(edot / edot0))
        * (1.0 - np.power(th, m))
    )
    return Y


@jit(nopython=True)
def PTW_goodparam(
    s0: float,
    sInf: float,
    y0: float,
    yInf: float,
    y1: float,
    y2: float,
    beta: float,
) -> float:
    """checks if the given PTW parameter set is valid"""
    return (
        (sInf < s0)
        * (yInf < y0)
        * (y0 < s0)
        * (yInf < sInf)
        * (y1 > s0)
        * (y2 > beta)
    )


@jit(nopython=True)
def PTW_Yield_Stress(
    p: float,
    kappa: float,
    s0: float,
    sInf: float,
    y0: float,
    yInf: float,
    y1: float,
    y2: float,
    beta: float,
    theta: float,
    lgamma: float,
    edot: float,
    rho0: float,
    matomic: float,
    shear: float,
    eps: float,
    T: float,
    Tmelt: float,
    small: float = 1.0e-10,
) -> float:
    """
    This function implements the PTW flow stress model.
    It returns the flow stress at the current material state
    and specified strain rate;
    see Preston, Tonks, Wallace, J. Appl. Phys. 93 (2003) 211,
    doi.org/10.1063/1.1524706

    edot: current strain rate
    eps: current strain
    T: temperature
    Tmelt: melting temperature
    shear: shear modulus
    p, kappa, s0, sInf, y0, yInf, y1, y2, beta, theta, lgamma: model parameters
    """

    t_hom = T / Tmelt
    # this one is commented because it is assumed that
    # the material state computes the temperature dependence of
    # the shear modulus
    # shear = shear * (1.0 - alpha * t_hom)
    # print("ptw shear is "+str(shear))

    afact = (4.0 / 3.0) * pi * rho0 / matomic
    # ainv is 1/a where 4/3 pi a^3 is the atomic volume
    ainv = np.cbrt(afact)

    # transverse wave velocity up to units
    xfact = np.sqrt(shear / rho0)
    # PTW characteristic strain rate [ 1/s ]
    xiDot = 0.5 * ainv * xfact * cbrtAvogadro
    # Note: previous version had xiDot *1e6 as well as edot*1e6
    # since everything below depends only on the ratio, we can drop those factors

    # should be flow stress in units of Mbar
    log_xid_ed = np.log(xiDot / edot)
    argErf = kappa * t_hom * (lgamma + log_xid_ed)
    Erfres = erf(argErf)

    saturation1 = s0 - (s0 - sInf) * Erfres
    saturation2 = s0 * np.exp(beta * (-lgamma - log_xid_ed))
    sat_cond = saturation1 > saturation2
    tau_s = np.copy(saturation2)
    tau_s[np.where(sat_cond)] = saturation1[sat_cond]

    ayield = y0 - (y0 - yInf) * Erfres
    byield = y1 * np.exp(-y2 * (lgamma + log_xid_ed))
    cyield = s0 * np.exp(-beta * (lgamma + log_xid_ed))

    y_cond = byield < cyield
    dyield = np.copy(cyield)
    dyield[np.where(y_cond)] = byield[y_cond]

    y_cond2 = ayield > dyield
    tau_y = np.copy(dyield)
    tau_y[np.where(y_cond2)] = ayield[y_cond2]
    scaled_stress = tau_s
    ind = np.where((p > small) * (np.abs(tau_s - tau_y) > small))
    eArg1 = (p * (tau_s - tau_y) / (s0 - tau_y))[ind]
    eArg2 = (
        (eps * p * theta)[ind] / (s0 - tau_y)[ind] / (np.exp(eArg1) - 1.0)
    )  # eArg1 already subsetted by ind
    if np.any((1.0 - (1.0 - np.exp(-eArg1)) * np.exp(-eArg2)) <= 0) or np.any(
        np.isinf(1.0 - (1.0 - np.exp(-eArg1)) * np.exp(-eArg2))
    ):
        print("bad")
    theLog = np.log(1.0 - (1.0 - np.exp(-eArg1)) * np.exp(-eArg2))
    scaled_stress[ind] = tau_s[ind] + (s0[ind] - tau_y[ind]) * theLog / p[ind]
    ind2 = np.where((p <= small) * (tau_s > tau_y))
    scaled_stress[ind2] = tau_s[ind2] - (tau_s - tau_y)[ind2] * np.exp(
        -eps[ind2] * theta[ind2] / (tau_s - tau_y)[ind2]
    )
    return scaled_stress * shear * 2.0


@jit(nopython=True)
def Stein_Flow_Stress(
    y0: float,
    beta: float,
    n: float,
    ymax: float,
    G0: float,
    epsi: float,
    shear: float,
    eps: float,
    T: float,
    Tmelt: float,
) -> float:
    """this function implements the Stein flow tress model"""
    fnow = np.power((1.0 + beta * (epsi + eps)), n)

    cond1 = fnow * y0 > ymax
    fnow[cond1] = (ymax / y0)[cond1]
    cond2 = T > Tmelt
    fnow[cond2] = 0.0

    return y0 * fnow * shear / G0
