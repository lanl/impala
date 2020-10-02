cimport numpy as np
import cython
import numpy as np
from libc.math cimport log, exp, erf, sqrt, cbrt, pi, pow

## Declaring Models

cdef class BaseModel(object):
    """ BaseModel provides a template for each of the different types of models """
    parameter_list = []  # list of parameters to sample
    constant_list = []   # list of constants to set at initialization

    cdef int p_idx, c_idx
    cdef MaterialModel parent

    cpdef dict report_parameters(self):
        """ Reports current Parameter values """
        return {}

    cpdef dict report_constants(self):
        """ Reports Constant values as initialized """
        return {}

    cpdef void update_parameters(self, double[:] input):
        """ Update the model parameters """
        return

    cpdef void initialize_constants(self, double[:] input):
        """ set the model constants """
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        """ Return Value """
        return 0.

    cpdef bint check_constraints(self):
        """ Checks the constraints of the model """
        return True

    def __init__(self, MaterialModel parent, int p_idx, int c_idx):
        self.parent = parent
        self.p_idx = p_idx
        self.c_idx = c_idx
        return

# Specific Heat Models

cdef class ConstantSpecificHeat(BaseModel):
    """ Constant Specific Heat Model """
    constant_list = ['Cv0']

    cdef double Cv0

    cpdef dict report_constants(self):
        return {'Cv0' : self.Cv0}

    cpdef void initialize_constants(self, double[:] input):
        self.Cv0 = input[0]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        return self.Cv0

SpecificHeat = {
    'Constant' : ConstantSpecificHeat,
    }
# Density Models

cdef class ConstantDensity(BaseModel):
    """ Constant Density Model """
    constant_list = ['rho0']

    cdef double rho0

    cpdef dict report_constants(self):
        return {'rho0' : self.rho0}

    cpdef void initialize_constants(self, double[:] input):
        self.rho0 = input[0]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        return self.rho0

Density = {
    'Constant' : ConstantDensity,
    }

# Melt Temperature Models

cdef class ConstantMeltTemperature(BaseModel):
    """ Constant Melt Temperature Model """
    constant_list = ['Tmelt0']

    cdef double Tmelt0

    cpdef dict report_constants(self):
        return {'Tmelt0': self.Tmelt0}

    cpdef void initialize_constants(self, double[:] input):
        self.Tmelt0 = input[0]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        return self.Tmelt0

MeltTemperature = {
    'Constant' : ConstantMeltTemperature
    }

# Shear Modulus Models

cdef class ConstantShearModulus(BaseModel):
    """ Constant Shear Modulus Model """
    constant_list = ['G0']

    cdef double G0

    cpdef dict report_constants(self):
        return {'G0' : self.G0}

    cpdef void initialize_constants(self, double[:] input):
        self.G0 = input[0]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        return self.G0

cdef class SimpleShearModulus(BaseModel):
    """ Constant Shear Modulus model exhibiting thermal softening """
    constant_list = ['G0','alpha']

    cdef double G0, alpha

    cpdef dict report_constants(self):
        return {'G0' : self.G0, 'alpha' : self.alpha}

    cpdef void initialize_constants(self, double[:] input):
        self.G0    = input[0]
        self.alpha = input[1]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        cdef double temp = self.parent.state.T
        cdef double tmelt = self.parent.state.Tmelt
        return self.G0 * (1. - self.alpha * (temp / tmelt))

cdef class SteinShearModulus(BaseModel):
    constant_list = ['G0', 'sgB']
    cdef double G0, sgB

    cpdef dict report_constants(self):
        return {'G0' : self.G0, 'sgB' : self.sgB}

    cpdef void initialize_constants(self, double[:] input):
        self.G0  = input[0]
        self.sgB = input[1]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        cdef double temp, tmelt, aterm, bterm, gnow

        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        # Just putting this here for completeness
        # aterm = a / (eta**(1/3)) * pressure
        aterm = 0.
        bterm = self.sgB * (temp - 300.)
        gnow  = self.G0 * (1. + aterm - bterm)

        if (temp >= tmelt) or (gnow < 0.):
            gnow = 0.

        return gnow

ShearModulus = {
    'Constant' : ConstantShearModulus,
    'Simple'   : SimpleShearModulus,
    'Stein'    : SteinShearModulus,
    }

# Yield Stress Models

cdef class ConstantYieldStress(BaseModel):
    """ Constant Yield Stress Model """
    constant_list = ['yield_stress']
    cdef double yield_stress

    cpdef dict report_constants(self):
        return {'yield_stress' : self.yield_stress}

    cpdef void initialize_constants(self, double[:] input):
        self.yield_stress = input[0]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        return self.yield_stress

cdef class JCYieldStress(BaseModel):
    """ Johnson Cook Yield Stress Model """
    parameter_list = ['A','B','C','n','m']
    constant_list  = ['Tref','edot0']

    cdef double A, B, C, n, m
    cdef double Tref, edot0

    cpdef dict report_parameters(self):
        return {'A' : self.A, 'B' : self.B, 'C' : self.C,
                'n' : self.n, 'm' : self.m }

    cpdef dict report_constants(self):
        return {'Tref' : self.Tref, 'edot0' : self.edot0}

    cpdef void update_parameters(self, double[:] input):
        self.A = input[0]
        self.B = input[1]
        self.C = input[2]
        self.n = input[3]
        self.m = input[4]
        return

    cpdef void initialize_constants(self, double[:] input):
        self.Tref  = input[0]
        self.edot0 = input[1]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        cdef double eps   = self.parent.state.strain
        cdef double t     = self.parent.state.T
        cdef double tmelt = self.parent.state.Tmelt

        cdef double th, Y

        th = max((t - self.Tref) / (tmelt - self.Tref), 0.)

        Y = (self.A + self.B * eps ** self.n) * \
            (1. + self.C * log(edot / self.edot0)) * \
            (1 - th ** self.m)
        return Y

cdef class PTWYieldStress(BaseModel):
    """ Preston Tonks Wallace Yield Stress Model """
    parameter_list = ['theta0','p','s0','sInf','kappa','gamma','y0','yInf','y1','y2','beta']
    constant_list  = ['matomic']

    cdef double theta, p, s0, sInf, kappa, gamma, y0, yInf, y1, y2, beta
    cdef double matomic

    cpdef dict report_parameters(self):
        return {'theta0' : self.theta, 'p'     : self.p,     's0'    : self.s0,
                'sInf'  : self.sInf,  'kappa' : self.kappa, 'gamma' : self.gamma,
                'y0'    : self.y0,    'yInf'  : self.yInf,  'y1'    : self.y1,
                'y2'    : self.y2,    'beta'  : self.beta,
                }

    cpdef dict report_constants(self):
        return {'matomic' : self.matomic}

    cpdef void update_parameters(self, double[:] input):
        self.theta = input[0]
        self.p     = input[1]
        self.s0    = input[2]
        self.sInf  = input[3]
        self.kappa = input[4]
        self.gamma = input[5]
        self.y0    = input[6]
        self.yInf  = input[7]
        self.y1    = input[8]
        self.y2    = input[9]
        self.beta  = input[10]
        return

    cpdef void initialize_constants(self, double[:] input):
        # self.beta    = input[0]
        self.matomic = input[0]
        return

    cpdef bint check_constraints(self):
        if (self.sInf > self.s0) or (self.yInf > self.y0) or \
            (self.y0 > self.s0) or (self.yInf > self.sInf) or \
            (self.y1 < self.s0) or (self.y2   < self.beta):
            return False
        else:
            return True

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        cdef:
            double t_hom, afact, ainv, xfact, xiDot, erf_argErf
            double saturation1, saturation2, tau_s, tau_y, scaled_stress
            double a_yield, byield, cyield, dyield, eArg1, eArg2, theLog

        cdef double eps   = self.parent.state.strain
        cdef double temp  = self.parent.state.T
        cdef double tmelt = self.parent.state.Tmelt
        cdef double shear = self.parent.state.G
        cdef double rho   = self.parent.state.rho
        cdef double ledot = edot * 1.e6

        if not self.check_constraints():
            return -999.

        # If shear modulus returns 0, then return 0 (and avoid divide_by_zero)
        if shear < 1.e-12:
            return 0.

        t_hom = temp / tmelt
        ainv = cbrt((4./3.) * pi * rho / self.matomic)
        xfact = sqrt(shear / rho)
        xiDot = 0.5 * ainv * xfact * cbrt(6.022e29) * 1.e4
        erf_argerf = erf(self.kappa * t_hom * log(self.gamma * xiDot / ledot))

        saturation1 = self.s0 - (self.s0 - self.sInf) * erf_argerf
        saturation2 = self.s0 * pow(ledot / self.gamma / xiDot, self.beta)

        tau_s = max(saturation1, saturation2)

        ayield = self.y0 - (self.y0 - self.yInf) * erf_argerf
        byield = self.y1 * pow(self.gamma * xiDot / ledot, -self.y2)
        cyield = self.s0 * pow(self.gamma * xiDot / ledot, -self.beta)

        dyield = min(byield, cyield)
        tau_y = max(ayield, dyield)

        if ayield > dyield:
            tau_y = ayield
        else:
            tau_y = dyield

        if self.p > 0.:
            if tau_s == tau_y:
                scaled_stress = tau_s
            else:
                eArg1 = self.p * (tau_s - tau_y) / (self.s0 - tau_y)
                eArg2 = eps * self.p * self.theta / (self.s0 - tau_y) / (exp(eArg1) - 1.)
                theLog = log(1. - (1. - exp(-eArg1)) * exp(-eArg2))
                scaled_stress = tau_s + (self.s0 - tau_y) * theLog / self.p
        else:
            if tau_s > tau_y:
                scaled_stress = (tau_s - (tau_s - tau_y) *
                                  exp(-eps * self.theta / (tau_s - tau_y)))
            else:
                scaled_stress = tau_s

        return scaled_stress * shear * 2.

cdef class SteinFlowStress(BaseModel):
    parameter_list = ['y0', 'a','b','beta','n','ymax']
    constant_list = ['G0','epsi','chi']

    cdef double y0,a, b, beta, n, ymax
    cdef double G0, epsi, chi

    cpdef dict report_parameters(self):
        return {'y0'   : self.y0,   'a' : self.a, 'b'    : self.b,
                'beta' : self.beta, 'n' : self.n, 'ymax' : self.ymax}

    cpdef dict report_constants(self):
        return {'G0' : self.G0, 'epsi' : self.epsi, 'chi' : self.chi}

    cpdef void update_parameters(self, double[:] input):
        self.y0   = input[0]
        self.a    = input[1]
        self.b    = input[2]
        self.beta = input[3]
        self.n    = input[4]
        self.ymax = input[5]
        return

    cpdef void initialize_constants(self, double[:] input):
        self.G0   = input[0]
        self.epsi = input[1]
        self.chi  = input[2]
        return

    @cython.cdivision(True)
    cpdef double value(self, double edot = 0.):
        cdef double temp, tmelt, G, eps, fnow
        temp  = self.parent.state.T
        tmelt = self.parent.state.tmelt
        G     = self.parent.state.G
        eps   = self.parent.state.strain
        fnow  = (1. + self.beta * (self.epsi + eps)) ** self.n
        if fnow * self.y0 > self.ymax:
            fnow = self.ymax / self.y0
        if temp > tmelt:
            fnow = 0.
        return self.y0 * fnow * G / self.G0

YieldStress = {
    'Constant' : ConstantYieldStress,
    'JC'       : JCYieldStress,
    'PTW'      : PTWYieldStress,
    'Stein'    : SteinFlowStress,
    }

# Material State Model

cdef class MaterialState:
    """ Current Material State, based on outputs of supplied models """
    cdef double T, Tmelt, stress, strain, G, Cv, rho
    cdef MaterialModel parent
    cdef double edotcrit

    cdef update(self, double edot, double dt):
        self.Cv      = self.parent.specific_heat.value()
        self.rho     = self.parent.density_model.value()

        if edot > self.edotcrit:
            self.T += self.parent.chi * self.stress * edot * dt / (self.Cv * self.rho)

        self.strain += edot * dt

        self.Tmelt   = self.parent.melt_model.value()
        self.G       = self.parent.shear_modulus.value()
        self.stress  = self.parent.flow_stress.value(edot)
        return

    cpdef void set(self, double T = 300., double strain = 0., double stress = 0.):
        self.T      = T
        self.strain = strain
        self.stress = stress
        return

    def __init__(self, parent, double T = 300., double strain = 0.,
                  double stress = 0., double edotcrit = 1e-6):
        self.parent = parent
        self.edotcrit = edotcrit
        self.set(T, strain, stress)
        return

# Material Model

cdef class MaterialModel:
    """ Material Model -- contains models for aspects of  """
    cdef MaterialState state
    cdef BaseModel flow_stress, specific_heat, shear_modulus, melt_model, density_model
    cdef (int,int) flow_stress_idx, specific_heat_idx, shear_modulus_idx
    cdef (int,int) melt_model_idx, density_model_idx
    cdef list parameter_list, constant_list
    cdef double chi
    cdef double emax, edot
    cdef int Nhist
    cdef dict models_used

    cpdef dict report_parameters(self):
        return {
            **self.flow_stress.report_parameters(),
            **self.specific_heat.report_parameters(),
            **self.shear_modulus.report_parameters(),
            **self.melt_model.report_parameters(),
            **self.density_model.report_parameters(),
            }

    cpdef dict report_constants(self):
        return {
            **self.flow_stress.report_constants(),
            **self.specific_heat.report_constants(),
            **self.shear_modulus.report_constants(),
            **self.melt_model.report_constants(),
            **self.density_model.report_constants(),
            'chi' : self.chi,
            }

    cpdef dict report_indices(self):
        return {
            'flow_stress'   : self.flow_stress_idx,
            'specific_heat' : self.specific_heat_idx,
            'shear_modulus' : self.shear_modulus_idx,
            'melt_model'    : self.melt_model_idx,
            'density_model' : self.density_model_idx,
            }

    cpdef dict report_models_used(self):
        return self.models_used

    cpdef (double,double,int) report_history_variables(self):
        return (self.emax, self.edot, self.Nhist)

    cpdef set_history_variables(self, double emax, double edot, int Nhist):
        self.emax = emax
        self.edot = edot
        self.Nhist = Nhist
        return

    cpdef compute_state_history(self,):
        """ Computes the state history """
        cdef double tmax = self.emax / self.edot
        cdef np.ndarray[dtype = np.float64_t, ndim = 1] times   = np.linspace(0., tmax, self.Nhist)
        cdef np.ndarray[dtype = np.float64_t, ndim = 1] delta_t = np.diff(times)
        cdef np.ndarray[dtype = np.float64_t, ndim = 1] strains = np.linspace(0., self.emax, self.Nhist)
        cdef np.ndarray[dtype = np.float64_t, ndim = 1] strain_rate = np.diff(strains) / delta_t
        cdef np.ndarray[dtype = np.float64_t, ndim = 2] results = np.empty((self.Nhist, 6))

        self.update_state(strain_rate[0], 0.)
        results[0] = (times[0], self.state.strain, self.state.stress,
                      self.state.T, self.state.G, self.state.rho)
        for i in range(1, self.Nhist):
            self.update_state(strain_rate[i-1], delta_t[i-1])
            results[i] = (times[i], self.state.strain, self.state.stress,
                          self.state.T, self.state.G, self.state.rho)
        return results

    cpdef void initialize_state(self, double T = 300., double strain = 0., double stress = 0.):
        self.state.set(T, strain, stress)
        return

    cdef update_state(self, double edot, double dt):
        """ Updates the state using a given edot and dt """
        self.state.update(edot, dt)
        return

    cpdef list get_parameter_list(self):
        """ Returns the Parameter list as defined by the member models """
        return self.parameter_list

    cpdef list get_constant_list(self):
        """ Returns the constant list as defined by the member models """
        return self.constant_list

    cpdef void update_parameters(self, double[:] param):
        """ Updates the Parameters for each of the sub-models """
        self.flow_stress.update_parameters(
                param[self.flow_stress_idx[0]   : self.specific_heat_idx[0]]
                )
        self.specific_heat.update_parameters(
                param[self.specific_heat_idx[0] : self.shear_modulus_idx[0]]
                )
        self.shear_modulus.update_parameters(
                param[self.shear_modulus_idx[0] : self.melt_model_idx[0]]
                )
        self.melt_model.update_parameters(
                param[self.melt_model_idx[0]    : self.density_model_idx[0]]
                )
        self.density_model.update_parameters(
                param[self.density_model_idx[0] : ]
                )
        return

    cpdef void initialize_constants(self, double[:] const):
        """ Sets initial values for the constants for each of the sub-models """
        self.flow_stress.initialize_constants(
                const[self.flow_stress_idx[1] : self.specific_heat_idx[1]]
                )
        self.specific_heat.initialize_constants(
                const[self.specific_heat_idx[1] : self.shear_modulus_idx[1]]
                )
        self.shear_modulus.initialize_constants(
                const[self.shear_modulus_idx[1] : self.melt_model_idx[1]]
                )
        self.melt_model.initialize_constants(
                const[self.melt_model_idx[1] : self.density_model_idx[1]]
                )
        self.density_model.initialize_constants(
                const[self.density_model_idx[1] : ]
                )
        return

    cpdef void initialize(
            self,
            dict parameters,
            dict constants,
            double T = 300.,
            double strain = 0.,
            double stress = 0.,
            ):
        try:
            self.initialize_constants(
                np.array([constants[key] for key in self.constant_list], dtype = np.float64)
                )
        except KeyError:
            print('Some constant value missing from input constants!')
            raise
        try:
            self.update_parameters(
                np.array([parameters[key] for key in self.parameter_list], dtype = np.float64)
                )
        except KeyError:
            print('Some Parameter Value Missing From Input parameters!')
            raise
        try:
            self.chi = constants['chi']
        except KeyError:
            print('chi value missing from input constants')
            raise
        self.initialize_state(T, strain, stress)
        return

    cpdef double get_chi(self):
        """ Get current value of chi """
        return self.chi

    cpdef bint check_constraints(self):
        if  self.flow_stress.check_constraints()   and \
            self.specific_heat.check_constraints() and \
            self.shear_modulus.check_constraints() and \
            self.melt_model.check_constraints()    and \
            self.density_model.check_constraints():
            return True
        else:
            return False

    def __init__(
            self,
            object state               = MaterialState,
            object flow_stress_model   = 'Constant',
            object specific_heat_model = 'Constant',
            object shear_modulus_model = 'Constant',
            object melt_model          = 'Constant',
            object density_model       = 'Constant',
            ):
        """ Builds a complete model to describe material reactions """

        self.models_used = {
            'flow_stress_model'   : flow_stress_model,
            'specific_heat_model' : specific_heat_model,
            'shear_modulus_model' : shear_modulus_model,
            'melt_model'          : melt_model,
            'density_model'       : density_model,
            }

        flow_stress   = YieldStress[flow_stress_model]
        specific_heat = SpecificHeat[specific_heat_model]
        shear_modulus = ShearModulus[shear_modulus_model]
        melt_temp     = MeltTemperature[melt_model]
        density       = Density[density_model]

        # Set model indices
        self.flow_stress_idx   = (0,0) # start with flow stress at 0
        self.specific_heat_idx = (
            len(flow_stress.parameter_list) + self.flow_stress_idx[0],
            len(flow_stress.constant_list)  + self.flow_stress_idx[1],
            )
        self.shear_modulus_idx = (
            len(specific_heat.parameter_list) + self.specific_heat_idx[0],
            len(specific_heat.constant_list)  + self.specific_heat_idx[1],
            )
        self.melt_model_idx    = (
            len(shear_modulus.parameter_list) + self.shear_modulus_idx[0],
            len(shear_modulus.constant_list)  + self.shear_modulus_idx[1],
            )
        self.density_model_idx = (
            len(melt_temp.parameter_list) + self.melt_model_idx[0],
            len(melt_temp.constant_list)  + self.melt_model_idx[1],
            )

        # Declare complete parameter list
        self.parameter_list = (
            flow_stress.parameter_list   +
            specific_heat.parameter_list +
            shear_modulus.parameter_list +
            melt_temp.parameter_list     +
            density.parameter_list
            )
        # Declare complete constant list
        self.constant_list = (
            flow_stress.constant_list    +
            specific_heat.constant_list  +
            shear_modulus.constant_list  +
            melt_temp.constant_list      +
            density.constant_list
            )

        # Model Initialization
        self.state         = state(self)
        self.flow_stress   = flow_stress(self, self.flow_stress_idx[0],
                                            self.flow_stress_idx[1])
        self.specific_heat = specific_heat(self, self.specific_heat_idx[0],
                                            self.specific_heat_idx[1])
        self.shear_modulus = shear_modulus(self, self.shear_modulus_idx[0],
                                            self.specific_heat_idx[1])
        self.melt_model    = melt_temp(self, self.melt_model_idx[0],
                                            self.melt_model_idx[1])
        self.density_model = density(self, self.density_model_idx[0],
                                            self.density_model_idx[1])
        return


cpdef dict models_available():
    d = {
        'yield_stress'  : list(YieldStress.keys()),
        'specific_heat' : list(SpecificHeat.keys()),
        'shear_modulus' : list(ShearModulus.keys()),
        'melt_temp'     : list(MeltTemperature.keys()),
        'density'       : list(Density.keys()),
        }
    return d

# EOF
