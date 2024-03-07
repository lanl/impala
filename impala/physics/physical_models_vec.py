"""
physical_models_vec.py

    A module for material strength behavior to be imported into python scripts for
    optimizaton or training emulators.  Adapted from strength_models_add_ptw.py

    Authors:
        DJ Luscher,    djl@lanl.gov
        Peter Trubey,  ptrubey@lanl.gov
        Devin Francom, dfrancom@lanl.gov
"""

import numpy as np
np.seterr(all = 'raise')
#import ipdb
import copy
from math import pi
from scipy.special import erf

## Error Definitions

class ConstraintError(ValueError):
    pass

class PTWStressError(FloatingPointError):
    pass

## Model Definitions

class BaseModel(object):
    """
    Base Class for property Models (flow stress, specific heat, melt, density,
    etc.).  Must be instantiated as a child of MaterialModel
    """
    params = []
    consts = []

    def value(self, *args):
        return None

    def update_parameters(self, x):
        self.parent.parameters.update_parameters(x, self.params)
        return

    def __init__(self, parent):
        self.parent = parent
        return

# Specific Heat Models

class Constant_Specific_Heat(BaseModel):
    """
    Constant Specific Heat Model
    """
    consts = ['Cv0']

    def value(self, *args):
        return self.parent.parameters.Cv0

#class Linear_Specific_Heat(BaseModel):
#    """
#    Linear Specific Heat Model
#    """
#    consts = ['Cv0', 'T0', 'dCdT']

#    def value(self, *args):
#        c0=self.parent.parameters.Cv0
#        t0=self.parent.parameters.T0
#        dcdt=self.parent.parameters.dCdT
#        tnow=self.parent.state.T
#        cnow=c0+(tnow-t0)*dcdt
#        return cnow

class Linear_Specific_Heat(BaseModel):
    """
    Linear Specific Heat Model
    """
    consts = ['c0', 'c1']

    def value(self, *args):

        tnow=self.parent.state.T
        cnow=self.parent.parameters.c0+self.parent.parameters.c1*tnow
        return cnow

class Quadratic_Specific_Heat(BaseModel):
    """
    Quadratic Specific Heat Model
    """
    consts = ['c0', 'c1', 'c2']

    def value(self, *args):

        tnow=self.parent.state.T
        cnow=self.parent.parameters.c0+self.parent.parameters.c1*tnow+self.parent.parameters.c2*tnow**2
        return cnow

class Piecewise_Linear_Specific_Heat(BaseModel):
   """
   Piecewise Linear Specific Heat Model
   Cv (T) = c0_0 + c1_0 * T for T<=T_t
   Cv (T) = c0_1 + c1_1 * T for T>T_t          
   """
   consts = ['T_t','c0_0', 'c1_0', 'c0_1', 'c1_1']
   def value(self, *args):
       tnow=self.parent.state.T
       intercept = np.repeat(self.parent.parameters.c0_0,len(tnow))
       slope = np.repeat(self.parent.parameters.c1_0,len(tnow))
       intercept[np.where(tnow > self.parent.parameters.T_t)] =  self.parent.parameters.c0_1
       slope[np.where(tnow > self.parent.parameters.T_t)] =  self.parent.parameters.c1_1
       cnow = intercept + slope * tnow
       return cnow

    
# Density Models

class Constant_Density(BaseModel):
    """
    Constant Density Model
    """
    consts = ['rho0']

    def value(self, *args):
        return self.parent.parameters.rho0 * np.ones(len(self.parent.state.T))

#class Linear_Density(BaseModel):
#    """
#    Linear Density Model
#    """
#    consts = ['rho0', 'T0', 'dRhodT']

#    def value(self, *args):
#        r0=self.parent.parameters.rho0
#        t0=self.parent.parameters.T0
#        drdt=self.parent.parameters.dRhodT
#        tnow=self.parent.state.T
#        rnow=r0+drdt*(tnow-t0)
#        return rnow

class Linear_Density(BaseModel):
    """
    Linear Density Model
    """
    consts = ['r0','r1']

    def value(self, *args):
        
        tnow=self.parent.state.T
        rnow=self.parent.parameters.r0+self.parent.parameters.r1*tnow
        return rnow    

class Quadratic_Density(BaseModel):
    """
    Quadratic Density Model
    """
    consts = ['r0','r1','r2']

    def value(self, *args):
        
        tnow=self.parent.state.T
        rnow=self.parent.parameters.r0+self.parent.parameters.r1*tnow+self.parent.parameters.r2*tnow**2
        return rnow

class Cubic_Density(BaseModel):
    """
    Quadratic Density Model
    """
    consts = ['r0','r1','r2','r3']

    def value(self, *args):
        
        tnow=self.parent.state.T
        rnow=self.parent.parameters.r0+self.parent.parameters.r1*tnow+self.parent.parameters.r2*tnow**2+self.parent.parameters.r3*tnow**3
        return rnow
    
    
# Melt Temperature Models

class Constant_Melt_Temperature(BaseModel):
    """
    Constant Melt Temperature Model
    """
    consts = ['Tmelt0']

    def value(self, *args):
        return self.parent.parameters.Tmelt0

#class Linear_Melt_Temperature(BaseModel):
#    """
#    Linear Melt Temperature Model
#    """
#    consts=['Tmelt0', 'rho0', 'dTmdRho']#

#    def value(self, *args):
#        tm0=self.parent.parameters.Tmelt0
#        rnow=self.parent.state.rho
#        dtdr=self.parent.parameters.dTmdRho
#        r0=self.parent.parameters.rho0
#        tmeltnow=tm0+dtdr*(rnow-r0)
#        return tmeltnow

class Linear_Melt_Temperature(BaseModel):
    """
    Linear Melt Temperature Model
    """

    consts=['tm0', 'tm1']
    def value(self, *args):
        rnow=self.parent.state.rho
        
        tmeltnow=self.parent.parameters.tm0+self.parent.parameters.tm1*rnow
        return tmeltnow

class Quadratic_Melt_Temperature(BaseModel):
    """
    Quadratic Melt Temperature Model
    """
    consts=['tm0', 'tm1', 'tm2']
    def value(self, *args):
        rnow=self.parent.state.rho
        tmeltnow=self.parent.parameters.tm0+self.parent.parameters.tm1*rnow+self.parent.parameters.tm2*rnow**2
        return tmeltnow    
    
class BGP_Melt_Temperature(BaseModel):

    consts = ['Tm_0', 'rho_m', 'gamma_1', 'gamma_3', 'q3']

    def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
       
        melt_temp = mp.Tm_0*np.power(rho/mp.rho_m, 1./3.)*np.exp(6*mp.gamma_1*(np.power(mp.rho_m,-1./3.)-np.power(rho,-1./3.))\
                    +2.*mp.gamma_3/mp.q3*(np.power(mp.rho_m,-mp.q3)-np.power(rho,-mp.q3)))
        return melt_temp

# Shear Modulus Models

class Constant_Shear_Modulus(BaseModel):
    consts = ['G0']

    def value(self, *args):
        return self.parent.parameters.G0

#class Linear_Shear_Modulus(BaseModel):
#    consts =  ['G0', 'rho0', 'dGdRho' ]#

#    def value(self, *args):
#         g0=self.parent.parameters.G0
#         rho0=self.parent.parameters.rho0
#         dgdr=self.parent.parameters.dGdRho
#         rnow=self.parent.state.rho
#         gnow=g0+dgdr*(rnow-rho0)
#         return gnow

class Linear_Cold_PW_Shear_Modulus(BaseModel):
     consts = ['g0', 'g1', 'alpha']
     def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        cold_shear = mp.g0+mp.g1*rho
        gnow = cold_shear*(1.- mp.alpha* (temp/tmelt))

        gnow[np.where(temp >= tmelt)] = 0.
        gnow[np.where(gnow < 0)] = 0.

        return gnow

class Quadratic_Cold_PW_Shear_Modulus(BaseModel):
     consts = ['g0', 'g1', 'g2', 'alpha']
     def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        cold_shear = mp.g0+mp.g1*rho+mp.g2*rho**2
        gnow = cold_shear*(1.- mp.alpha* (temp/tmelt))

        gnow[np.where(temp >= tmelt)] = 0.
        gnow[np.where(gnow < 0)] = 0.

        return gnow
    
class Simple_Shear_Modulus(BaseModel):
    consts = ['G0', 'alpha']

    def value(self, *args):
        mp = self.parent.parameters
        temp = self.parent.state.T
        tmelt = self.parent.state.Tmelt

        return mp.G0 * (1. - mp.alpha * (temp / tmelt))

class BGP_PW_Shear_Modulus(BaseModel):
    #BPG model provides cold shear, i.e. shear modulus at zero temperature as a function of density.
    #PW describes the (lienar) temperature dependence of the shear modulus. (Same dependency as
    #in Simple_Shear_modulus.)
    #With these two models combined, we get the shear modulus as a function of density and temperature.
    
    consts = ['G0', 'rho_0', 'gamma_1', 'gamma_2', 'q2', 'alpha']

    def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
 
        cold_shear  = mp.G0*np.exp(6.*mp.gamma_1*(np.power(mp.rho_0,-1./3.)-np.power(rho,-1./3.))\
                    + 2*mp.gamma_2/mp.q2*(np.power(mp.rho_0,-mp.q2)-np.power(rho,-mp.q2)))
        gnow = cold_shear*(1.- mp.alpha* (temp/tmelt))

        gnow[np.where(temp >= tmelt)] = 0.
        gnow[np.where(gnow < 0)] = 0.

        #if temp >= tmelt: gnow = 0.0
        #if gnow < 0.0:    gnow = 0.0
        return gnow

class Stein_Shear_Modulus(BaseModel):
    #consts = ['G0', 'sgA', 'sgB']
    #assuming constant density and pressure
    #so we only include the temperature dependence
    consts = ['G0', 'sgB']
    eta = 1.0

    def value(self, *args):
        mp    = self.parent.parameters
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        #just putting this here for completeness
        #aterm = a/eta**(1.0/3.0)*pressure
        aterm = 0.0
        bterm = mp.sgB * (temp - 300.0)
        gnow  = mp.G0 * (1.0 + aterm - bterm)
        #if temp >= tmelt: gnow = 0.0
        #if gnow < 0.0:    gnow = 0.0
        gnow[np.where(temp >= tmelt)] = 0.
        gnow[np.where(gnow < 0)] = 0.
        return gnow

# Yield Stress Models

class Constant_Yield_Stress(BaseModel):
    """
    Constant Yield Stress Model
    """
    consts = ['yield_stress']

    def value(self, *args):
        return self.parent.parameters.yield_stress

def fast_pow(a, b):
    """
    Numpy power is slow, this is faster.  Gets a**b for a and b np arrays.
    """
    cond = a>0
    out = a * 0.
    out[cond] = np.exp(b[cond] * np.log(a[cond]))
    return out

pos = lambda a: (abs(a) + a) / 2 # same as max(0,a)

class JC_Yield_Stress(BaseModel):
    params = ['A','B','C','n','m']
    consts = ['Tref','edot0','chi']

    def value(self, edot):
        mp    = self.parent.parameters
        eps   = self.parent.state.strain
        t     = self.parent.state.T
        tmelt = self.parent.state.Tmelt

        #th = np.max([(t - mp.Tref) / (tmelt - mp.Tref), 0.])
        th = pos((t - mp.Tref) / (tmelt - mp.Tref))

        Y = (
            (mp.A + mp.B * fast_pow(eps, mp.n)) *
            (1. + mp.C * np.log(edot / mp.edot0)) *
            (1. - fast_pow(th, mp.m))
            )
        return Y

class PTW_Yield_Stress(BaseModel):
    params = ['theta','p','s0','sInf','kappa','lgamma','y0','yInf','y1', 'y2']
    consts = ['rho0', 'beta', 'matomic', 'chi']

    #@profile
    def value(self, edot):
        """
        function used to define PTW flow stress model
        arguments are:
            - edot: scalar, strain rate
            - material: an instance of MaterialModel class
        returns the flow stress at the current material state
        and specified strain rate
        """
        mp    = self.parent.parameters
        eps   = self.parent.state.strain
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        shear = self.parent.state.G

        #if (np.any(mp.sInf > mp.s0) or np.any(mp.yInf > mp.y0) or
        #        np.any(mp.y0 > mp.s0) or np.any(mp.yInf > mp.sInf) or np.any(mp.y1 < mp.s0) or np.any(mp.y2 < mp.beta)):
        #    raise ConstraintError

        good = (
            (mp.sInf < mp.s0) * (mp.yInf < mp.y0) * (mp.y0 < mp.s0)   *
            (mp.yInf < mp.sInf) * (mp.y1 > mp.s0) * (mp.y2 > mp.beta)
            )
        if np.any(~good):
            #return np.array([-999.]*len(good))
            raise ConstraintError('PTW bad val')


        #convert to 1/s strain rate since PTW rate is in that unit
        edot = edot * 1.0E6

        t_hom = temp / tmelt
        #this one is commented because it is assumed that
        #the material state computes the temperature dependence of
        #the shear modulus
        #shear = shear * (1.0 - mp.alpha * t_hom)
        #print("ptw shear is "+str(shear))

        afact = (4.0 / 3.0) * pi * mp.rho0 / mp.matomic
        #ainv is 1/a where 4/3 pi a^3 is the atomic volume
        ainv  = afact ** (1.0 / 3.0)

        #transverse wave velocity up to units
        xfact = np.sqrt ( shear / mp.rho0 )
        #PTW characteristic strain rate [ 1/s ]
        xiDot = 0.5 * ainv * xfact * pow(6.022E29, 1.0 / 3.0) * 1.0E4

        #if np.any(mp.gamma * xiDot / edot <= 0) or np.any(np.isinf(mp.gamma * xiDot / edot)):
        #    print("bad")
        argErf = mp.kappa * t_hom * (mp.lgamma + np.log( xiDot / edot ))

        saturation1 = mp.s0 - ( mp.s0 - mp.sInf ) * erf( argErf )
        #saturation2 = mp.s0 * np.power( edot / mp.gamma / xiDot , mp.beta )
        #saturation2 = mp.s0 * (edot / mp.gamma / xiDot)**mp.beta
        saturation2 = mp.s0 * np.exp(mp.beta * (-mp.lgamma + np.log(edot / xiDot)))
        #if saturation1 > saturation2:
        #    tau_s=saturation1 # thermal activation regime
        #else:
        #    tau_s=saturation2 # phonon drag regime

        sat_cond = saturation1 > saturation2
        #tau_s = sat_cond*saturation1 + (~sat_cond)*saturation2
        tau_s = np.copy(saturation2)
        tau_s[np.where(sat_cond)] = saturation1[sat_cond]

        ayield = mp.y0 - ( mp.y0 - mp.yInf ) * erf( argErf )
        #byield = mp.y1 * np.power( mp.gamma * xiDot / edot , -mp.y2 )
        #cyield = mp.s0 * np.power( mp.gamma * xiDot / edot , -mp.beta)

        byield = mp.y1 * np.exp( -mp.y2*(mp.lgamma + np.log( xiDot / edot )))
        cyield = mp.s0 * np.exp( -mp.beta*(mp.lgamma + np.log( xiDot / edot )))

        #if byield < cyield:
        #    dyield = byield    # intermediate regime
        #else:
        #    dyield = cyield    # phonon drag regime

        y_cond = (byield < cyield)
        #dyield = y_cond*byield + (~y_cond)*cyield
        dyield = np.copy(cyield)
        dyield[np.where(y_cond)] = byield[y_cond]

        #if ayield > dyield:
        #    tau_y = ayield     # thermal activation regime
        #else:
        #    tau_y = dyield     # intermediate or high rate

        y_cond2 = ayield > dyield
        #tau_y = y_cond2*ayield + (~y_cond2)*dyield
        tau_y = np.copy(dyield)
        tau_y[np.where(y_cond2)] = ayield[y_cond2]


        # if mp.p > 0:
        #     if tau_s == tau_y:
        #         scaled_stress = tau_s
        #     else:
        #         try:
        #             eArg1  = mp.p * (tau_s - tau_y) / (mp.s0 - tau_y)
        #             eArg2  = eps * mp.p * mp.theta / (mp.s0 - tau_y) / (exp(eArg1) - 1.0)
        #             theLog = log(1.0 - (1.0 - exp(- eArg1)) * exp(-eArg2))
        #         except (FloatingPointError, OverflowError) as e:
        #             raise PTWStressError from e
        #         scaled_stress = (tau_s + ( mp.s0 - tau_y ) * theLog / mp.p )
        # else:
        #     if tau_s > tau_y:
        #         scaled_stress = ( tau_s - ( tau_s - tau_y )
        #                         * exp( - eps * mp.theta / (tau_s - tau_y) ) )
        #     else:
        #         scaled_stress = tau_s

        small = 1.0e-10
        scaled_stress = tau_s
        ind = np.where((mp.p > small) * (np.abs(tau_s - tau_y) > small))
        eArg1 = (mp.p * (tau_s - tau_y) / (mp.s0 - tau_y))[ind]
        eArg2 = (eps * mp.p * mp.theta)[ind] / (mp.s0 - tau_y)[ind] / (np.exp(eArg1) - 1.0) # eArg1 already subsetted by ind
        if (np.any((1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2)) <= 0) or \
                np.any(np.isinf(1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2)))):
            print('bad')
        theLog = np.log(1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2))
        scaled_stress[ind] = (tau_s[ind] + ( mp.s0[ind] - tau_y[ind] ) * theLog / mp.p[ind] )
        ind2 = np.where((mp.p <= small) * (tau_s>tau_y))
        scaled_stress[ind2] = (
            + tau_s[ind2]
            - (tau_s - tau_y)[ind2] * np.exp(- eps[ind2] * mp.theta[ind2] / (tau_s - tau_y)[ind2])
            )
        # should be flow stress in units of Mbar
        out = scaled_stress * shear * 2.0
        out[np.where(~good)] = -999.
        return out

class Stein_Flow_Stress(BaseModel):
    params = ['y0', 'a', 'b', 'beta', 'n', 'ymax']
    consts = ['G0', 'epsi', 'chi']

    def value(self, *args):
        mp    = self.parent.parameters
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        shear = self.parent.state.G
        eps   = self.parent.state.strain
        fnow  = fast_pow((1.0+mp.beta*(mp.epsi+eps)), mp.n)
        
        cond1 = fnow*mp.y0 > mp.ymax
        fnow[cond1] = (mp.ymax/mp.y0)[cond1]
        cond2 = temp > tmelt
        fnow[cond2] = 0.0

        #if fnow*mp.y0 > mp.ymax: fnow = mp.ymax/mp.y0
        #if temp > tmelt: fnow = 0.0
        return mp.y0*fnow*shear/mp.G0

## Parameters Definition

class ModelParameters(object):
    params = []
    consts = []
    parent = None

    def update_parameters(self, x):
        if type(x) is np.ndarray:
            self.__dict__.update({x:y for x,y in zip(self.params,x)})
        elif type(x) is dict:
            for key in self.params:
                self.__dict__[key] = x[key]
        elif type(x) is list:
            try:
                assert(len(x) == len(self.params))
            except AssertionError:
                print('Incorrect number of parameters!')
                raise
            for i in range(len(self.params)):
                self.__dict__[self.params[i]] = x[i]
        else:
            raise ValueError('Type {} is not supported.'.format(type(x)))
        return

    def __init__(self, parent):
        self.parent = parent
        return

## State Definition

class MaterialState(object):
    T      = None
    Tmelt  = None
    stress = None
    strain = None
    G      = None

    def set_state(self, T = 300., strain = 0., stress = 0.):
        self.T = T
        self.strain = strain
        self.stress = stress
        return

    def __init__(self, parent, T = 300., strain = 0., stress = 0.):
        self.parent = parent
        self.set_state(T, strain, stress)
        return

## Material Model Definition

class MaterialModel(object):
    def __init__(
            self,
            parameters          = ModelParameters,
            initial_state       = MaterialState,
            flow_stress_model   = Constant_Yield_Stress,
            specific_heat_model = Constant_Specific_Heat,
            shear_modulus_model = Constant_Shear_Modulus,
            melt_model          = Constant_Melt_Temperature,
            density_model       = Constant_Density,
            ):
        """
        Initialization routine for Material Model.  All of the arguments
        supplied are classes, which are then instantiated within the function.

        The reason for doing this is that then we can pass the MaterialModel
        instance to the physical models so that the model's parent can be
        declared at instantiation.  then the model.value() function can reach
        into the parent class to find whatever it needs.
        """
        self.parameters    = parameters(self)
        self.state         = initial_state(self)

        self.flow_stress   = flow_stress_model(self)
        self.specific_heat = specific_heat_model(self)
        self.shear_modulus = shear_modulus_model(self)
        self.melt_model    = melt_model(self)
        self.density       = density_model(self)

        params = (
                self.flow_stress.params +
                self.specific_heat.params +
                self.shear_modulus.params +
                self.melt_model.params +
                self.density.params
                )
        consts = set(
                self.flow_stress.consts +
                self.specific_heat.consts +
                self.shear_modulus.consts +
                self.melt_model.consts +
                self.density.consts
                )

        try:
            assert(len(set(params)) == len(params))
        except AssertionError:
            print('Some Duplicate Parameters between models')
            raise

        try:
            assert(len(set(params).intersection(set(consts))) == 0)
        except AssertionError:
            print('Duplicate item in parameters and constants')
            raise

        self.parameters.params = params
        self.parameters.consts = consts
        return

    def get_parameter_list(self,):
        """
        The list of parameters used in the model.
        This also describes the order of their appearance in the sampling results
        """
        return self.parameters.params

    def get_constants_list(self,):
        """
        List of Constants used in the model
        """
        return self.parameters.consts

    def update_state(self, edot, dt):
        chi = self.parameters.chi
        self.state.Cv  = self.specific_heat.value()
        self.state.rho = self.density.value()
        #if we are working with microseconds, then this is a reasonable value
        #if we work in seconds, it should be changed to ~1.
        edotcrit=1.0e-6
        #if edot > edotcrit:
        #  self.state.T += chi * self.state.stress * edot * dt / (self.state.Cv * self.state.rho)
        cond = edot > edotcrit
        #if any(cond):
        self.state.T = self.state.T + cond * chi * self.state.stress * edot * dt / (self.state.Cv * self.state.rho)
        self.state.strain = self.state.strain + edot * dt

        self.state.Tmelt  = self.melt_model.value()
        self.state.G      = self.shear_modulus.value()
        self.state.stress = self.flow_stress.value(edot)
        return

    def update_parameters(self, x):
        self.parameters.update_parameters(x)
        return

    def initialize(self, parameters, constants):
        """
        Initialize the model at a given set of parameters, constants
        """
        try:
            self.parameters.__dict__.update(
                    {key : parameters[key] for key in self.parameters.params},
                    )
        except KeyError:
            print('{} missing from list of supplied parameters'.format(
                    set(self.parameters.params).difference(set(parameters.keys()))
                    ))
            raise
        try:
            self.parameters.__dict__.update(
                    {key : constants[key] for key in self.parameters.consts},
                    )
        except KeyError:
            print('{} missing from list of supplied constants'.format(
                    set(self.parameters.consts).difference(set(constants.keys()))
                    ))
            raise
        return

    def initialize_state(self, T = 300., stress = 0., strain = 0.):
        self.state.set_state(T, stress, strain)
        return

    def set_history_variables(self, emax, edot, Nhist):
        self.emax = emax
        self.edot = edot
        self.Nhist = Nhist
        return

    def get_history_variables(self):
        return [self.emax, self.edot, self.Nhist]

    def compute_state_history(self, strain_history):
        strains = strain_history['strains']
        times = strain_history['times']
        strain_rate = strain_history['strain_rate']
        # Nhist = len(strains)
        # nrep = len(self.parameters.kappa)
        nrep, Nhist = strains.shape # nexp * nhist array

        results = np.empty((Nhist, 6, nrep))

        state = self.state
        self.update_state(strain_rate[:,0], 0.)

        #import pdb
        #pdb.set_trace()

        results[0] = np.array([times[:,0], state.strain, state.stress, state.T, state.G, state.rho]) #np.repeat(state.rho,nrep)])

        for i in range(1, Nhist):
            self.update_state(strain_rate[:,i-1], times[:,i] - times[:,i-1])
            # self.update_state(strain_rate.T[i-1], times.T[i] - times.T[i-1])
            # results[i] = [times[i], state.strain, state.stress, state.T, state.G, state.rho]
            results[i] = np.array([times[:,i], state.strain, state.stress, state.T, state.G,
                 state.rho]) #np.repeat(state.rho, nrep)])

        return results

## function to generate strain history to calculate along
##  Should probably make this a method of MaterialModel class

def generate_strain_history(emax, edot, Nhist):
    tmax = emax / edot
    strains = np.linspace(0., emax, Nhist)
    nrep=len(edot)
    times = np.empty((nrep, Nhist))
    #for i in range(nrep):
    #    times[i,:] = np.linspace(0., tmax[i], Nhist)
    times = np.linspace(0., tmax, Nhist)
    strain_rate = np.empty((nrep, Nhist - 1))
    #for i in range(nrep):
    #    strain_rate[i, :] = np.diff(strains) / np.diff(times[i, :])

    strain_diffs = np.diff(strains)

    strain_rate = strain_diffs[:, np.newaxis] / np.diff(times,axis=0)
    return dict((['times',times.T], ['strains',strains], ['strain_rate',strain_rate.T]))

def generate_strain_history_new(emax, edot, nhist):
    tmax    = emax / edot     
    strains = np.linspace(0, emax, nhist) # nhist * nexp
    times   = np.linspace(0, tmax, nhist) # nhist * nexp
    rates   = np.diff(strains, axis = 0) / np.diff(times, axis = 0) # (nhist - 1) * nexp
    return {'times' : times.T, 'strains' : strains.T, 'strain_rate' : rates.T}

# EOF
