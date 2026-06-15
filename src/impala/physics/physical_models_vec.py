"""
physical_models_vec.py

    A module for material strength behavior to be imported into python scripts for
    optimizaton or training emulators.  Adapted from strength_models_add_ptw.py

    Authors:
        DJ Luscher,    djl@lanl.gov
        Peter Trubey,  ptrubey@lanl.gov
        Devin Francom, dfrancom@lanl.gov
        JeeYeon Plohr, jplohr@lanl.gov
        Sky Sjue, sjue@lanl.gov
        Lauren VanDervort, @lvandervort@lanl.gov
        Daniel N Blaschke, dblaschke@lanl.gov
"""

import numpy as np

from . import functions

PTW_goodparam = functions.PTW_goodparam

np.seterr(all="raise")

## Error Definitions


class ConstraintError(ValueError):
    pass


class PTWStressError(FloatingPointError):
    pass


## Model Definitions


class BaseModel:
    """
    Base Class for property Models (flow stress, specific heat, melt, density,
    etc.).  Must be instantiated as a child of MaterialModel
    """

    def value(self, *args):
        pass

    def update_parameters(self, x):
        self.parent.parameters.update_parameters(x, self.params)

    def __init__(self, parent):
        self.params = []
        self.consts = []
        self.parent = parent


########################
# Specific Heat Models
########################


class Constant_Specific_Heat(BaseModel):
    """
    Constant Specific Heat Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["Cv0"]

    def value(self, *args):
        return self.parent.parameters.Cv0 * np.ones(len(self.parent.state.T))


class Linear_Specific_Heat(BaseModel):
    """
    Linear Specific Heat Model
    calls Cubic Specific Heat Model with c2=0=c3 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["c0", "c1"]

    def value(self, *args):
        return functions.Cubic_Specific_Heat(
            c0=self.parent.parameters.c0,
            c1=self.parent.parameters.c1,
            c2=0,
            c3=0,
            T=self.parent.state.T,
        )


class Quadratic_Specific_Heat(BaseModel):
    """
    Quadratic Specific Heat Model
    calls Cubic Specific Heat Model with c3=0 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["c0", "c1", "c2"]

    def value(self, *args):
        return functions.Cubic_Specific_Heat(
            c0=self.parent.parameters.c0,
            c1=self.parent.parameters.c1,
            c2=self.parent.parameters.c2,
            c3=0,
            T=self.parent.state.T,
        )


class Cubic_Specific_Heat(BaseModel):
    """
    Cubic Specific Heat Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["c0", "c1", "c2", "c3"]

    def value(self, *args):
        return functions.Cubic_Specific_Heat(
            c0=self.parent.parameters.c0,
            c1=self.parent.parameters.c1,
            c2=self.parent.parameters.c2,
            c3=self.parent.parameters.c3,
            T=self.parent.state.T,
        )


class Piecewise_Linear_Specific_Heat(BaseModel):
    """
    Piecewise Linear Specific Heat Model
    calls Piecewise_Cubic_Specific_Heat Model with c20=0=c21 and c30=0=c31 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["T_t", "c0_0", "c1_0", "c0_1", "c1_1"]

    def value(self, *args):
        return functions.Piecewise_Cubic_Specific_Heat(
            Tt=self.parent.parameters.T_t,
            c00=self.parent.parameters.c0_0,
            c01=self.parent.parameters.c0_1,
            c10=self.parent.parameters.c1_0,
            c11=self.parent.parameters.c1_1,
            c20=0,
            c21=0,
            c30=0,
            c31=0,
            T=self.parent.state.T,
        )


class Piecewise_Quadratic_Specific_Heat(BaseModel):
    """
    Piecewise Quadratic Specific Heat Model
    calls Piecewise_Cubic_Specific_Heat Model with c30=0=c31 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["T_t", "c0_0", "c1_0", "c2_0", "c0_1", "c1_1", "c2_1"]

    def value(self, *args):
        return functions.Piecewise_Cubic_Specific_Heat(
            Tt=self.parent.parameters.T_t,
            c00=self.parent.parameters.c0_0,
            c01=self.parent.parameters.c0_1,
            c10=self.parent.parameters.c1_0,
            c11=self.parent.parameters.c1_1,
            c20=self.parent.parameters.c2_0,
            c21=self.parent.parameters.c2_1,
            c30=0,
            c31=0,
            T=self.parent.state.T,
        )


class Piecewise_Cubic_Specific_Heat(BaseModel):
    """
    Piecewise Cubic Specific Heat Model
    Cv (T) = c0_0 + c1_0 * T + c2_0 * T**2  + c3_0 * T**3 for T<=T_t
    Cv (T) = c0_1 + c1_1 * T + c2_1 * T**2  + c3_1 * T**3 for T>T_t
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = [
            "T_t",
            "c0_0",
            "c1_0",
            "c2_0",
            "c3_0",
            "c0_1",
            "c1_1",
            "c2_1",
            "c3_1",
        ]

    def value(self, *args):
        return functions.Piecewise_Cubic_Specific_Heat(
            Tt=self.parent.parameters.T_t,
            c00=self.parent.parameters.c0_0,
            c01=self.parent.parameters.c0_1,
            c10=self.parent.parameters.c1_0,
            c11=self.parent.parameters.c1_1,
            c20=self.parent.parameters.c2_0,
            c21=self.parent.parameters.c2_1,
            c30=self.parent.parameters.c3_0,
            c31=self.parent.parameters.c3_1,
            T=self.parent.state.T,
        )


########################
# Density Models
########################


class Constant_Density(BaseModel):
    """
    Constant Density Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["rho0"]

    def value(self, *args):
        return self.parent.parameters.rho0 * np.ones(len(self.parent.state.T))


class Linear_Density(BaseModel):
    """
    Linear Density Model
    calls Cubic_Density Model with r2=0=r3 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["r0", "r1"]

    def value(self, *args):
        return functions.Cubic_Density(
            r0=self.parent.parameters.r0,
            r1=self.parent.parameters.r1,
            r2=0,
            r3=0,
            T=self.parent.state.T,
        )


class Quadratic_Density(BaseModel):
    """
    Quadratic Density Model
    calls Cubic_Density Model with r3=0 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["r0", "r1", "r2"]

    def value(self, *args):
        return functions.Cubic_Density(
            r0=self.parent.parameters.r0,
            r1=self.parent.parameters.r1,
            r2=self.parent.parameters.r2,
            r3=0,
            T=self.parent.state.T,
        )


class Cubic_Density(BaseModel):
    """
    Cubic Density Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["r0", "r1", "r2", "r3"]

    def value(self, *args):
        return functions.Cubic_Density(
            r0=self.parent.parameters.r0,
            r1=self.parent.parameters.r1,
            r2=self.parent.parameters.r2,
            r3=self.parent.parameters.r3,
            T=self.parent.state.T,
        )


########################
# Melt Temperature Models
########################


class Constant_Melt_Temperature(BaseModel):
    """
    Constant Melt Temperature Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["Tmelt0"]

    def value(self, *args):
        return self.parent.parameters.Tmelt0 * np.ones(len(self.parent.state.T))


class Linear_Melt_Temperature(BaseModel):
    """
    Linear Melt Temperature Model
    calls Cubic_Melt_Temperature Model with tm2=0=tm3 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["tm0", "tm1"]

    def value(self, *args):
        return functions.Cubic_Melt_Temperature(
            tm0=self.parent.parameters.tm0,
            tm1=self.parent.parameters.tm1,
            tm2=0,
            tm3=0,
            rho=self.parent.state.rho,
        )


class Quadratic_Melt_Temperature(BaseModel):
    """
    Quadratic Melt Temperature Model
    calls Cubic_Melt_Temperature Model with tm3=0 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["tm0", "tm1", "tm2"]

    def value(self, *args):
        return functions.Cubic_Melt_Temperature(
            tm0=self.parent.parameters.tm0,
            tm1=self.parent.parameters.tm1,
            tm2=self.parent.parameters.tm2,
            tm3=0,
            rho=self.parent.state.rho,
        )


class Cubic_Melt_Temperature(BaseModel):
    """
    Cubic Melt Temperature Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["tm0", "tm1", "tm2", "tm3"]

    def value(self, *args):
        return functions.Cubic_Melt_Temperature(
            tm0=self.parent.parameters.tm0,
            tm1=self.parent.parameters.tm1,
            tm2=self.parent.parameters.tm2,
            tm3=self.parent.parameters.tm3,
            rho=self.parent.state.rho,
        )


class BGP_Melt_Temperature(BaseModel):
    """
    Burakovsky-Greeff-Preston Melt Temperature Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["Tm_0", "rho_m", "gamma_1", "gamma_3", "q3"]

    def value(self, *args):
        mp = self.parent.parameters
        return functions.BGP_Melt_Temperature(
            Tm0=mp.Tm_0,
            rhom=mp.rho_m,
            gamma1=mp.gamma_1,
            gamma3=mp.gamma_3,
            q3=mp.q3,
            rho=self.parent.state.rho,
        )


########################
# Shear Modulus Models
########################


class Constant_Shear_Modulus(BaseModel):
    """
    Constant Shear Modulus Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["G0"]

    def value(self, *args):
        return self.parent.parameters.G0 * np.ones(len(self.parent.state.T))


class Linear_Cold_PW_Shear_Modulus(BaseModel):
    """
    Pinear Cold PW Shear Modulus
    calls Quadratic_Cold_PW_Shear_Modulus with g2=0 under the hood
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["g0", "g1", "alpha"]

    def value(self, *args):
        mp = self.parent.parameters
        return functions.Quadratic_Cold_PW_Shear_Modulus(
            g0=mp.g0,
            g1=mp.g1,
            g2=0,
            alpha=mp.alpha,
            rho=self.parent.state.rho,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


class Quadratic_Cold_PW_Shear_Modulus(BaseModel):
    """
    Quadratic Cold PW Shear Modulus
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["g0", "g1", "g2", "alpha"]

    def value(self, *args):
        mp = self.parent.parameters
        return functions.Quadratic_Cold_PW_Shear_Modulus(
            g0=mp.g0,
            g1=mp.g1,
            g2=mp.g2,
            alpha=mp.alpha,
            rho=self.parent.state.rho,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


class Simple_Shear_Modulus(BaseModel):
    """
    Simple Shear Modulus
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["G0", "alpha"]

    def value(self, *args):
        return functions.Simple_Shear_Modulus(
            G0=self.parent.parameters.G0,
            alpha=self.parent.parameters.alpha,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


class BGP_PW_Shear_Modulus(BaseModel):
    """BPG model provides cold shear, i.e. shear modulus at zero temperature as a function of density.
    PW describes the (linear) temperature dependence of the shear modulus. (Same dependency as
    in Simple_Shear_modulus.)
    With these two models combined, we get the shear modulus as a function of density and temperature;
    see Burakovsky, Greeff, Preston, Phys. Rev. B67 (2003) 094107, DOI:10.1103/PhysRevB.67.094107"""

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["G0", "rho_0", "gamma_1", "gamma_2", "q2", "alpha"]

    def value(self, *args):
        mp = self.parent.parameters
        gnow = functions.BGP_PW_Shear_Modulus(
            G0=mp.G0,
            rho_0=mp.rho_0,
            gamma_1=mp.gamma_1,
            gamma_2=mp.gamma_2,
            q2=mp.q2,
            alpha=mp.alpha,
            rho=self.parent.state.rho,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )
        return gnow


class Stein_Shear_Modulus(BaseModel):
    """
    Steinberg-Guinan Shear Modulus assuming constant density and pressure
    """

    # consts = ['G0', 'sgA', 'sgB']
    # assuming constant density and pressure
    # so we only include the temperature dependence
    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["G0", "sgB"]
        self.eta = 1.0

    def value(self, *args):
        return functions.Stein_Shear_Modulus(
            G0=self.parent.parameters.G0,
            sgB=self.parent.parameters.sgB,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


########################
# Yield Stress Models
########################


class Constant_Yield_Stress(BaseModel):
    """
    Constant Yield Stress Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.consts = ["yield_stress", "chi"]

    def value(self, *args):
        return self.parent.parameters.yield_stress * np.ones(
            len(self.parent.state.T)
        )


class JC_Yield_Stress(BaseModel):
    """
    Johnson-Cook Yield Stress Model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.params = ["A", "B", "C", "n", "m"]
        self.consts = [
            "Tref",
            "edot0",
            "chi",
        ]  ## nothing here depends on chi, why is it here?

    def value(self, edot):
        mp = self.parent.parameters
        return functions.JC_Yield_Stress(
            edot=edot,
            A=mp.A,
            B=mp.B,
            C=mp.C,
            n=mp.n,
            m=mp.m,
            Tref=mp.Tref,
            edot0=mp.edot0,
            eps=self.parent.state.strain,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


class PTW_Yield_Stress(BaseModel):
    """This class implements the PTW flow stress model"""

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        self.params = [
            "theta",
            "p",
            "s0",
            "beta",
            "sInf",
            "kappa",
            "lgamma",
            "y0",
            "yInf",
            "y1",
            "y2",
        ]
        self.consts = ["rho0", "matomic", "chi"]

    # @profile
    def value(self, edot):
        """
        function used to define PTW flow stress model
        arguments are:
            - edot: scalar, strain rate
            - material: an instance of MaterialModel class
        returns the flow stress at the current material state
        and specified strain rate
        """
        mp = self.parent.parameters
        good = PTW_goodparam(
            mp.s0, mp.sInf, mp.y0, mp.yInf, mp.y1, mp.y2, mp.beta
        )
        if np.any(np.logical_not(good)):
            # return np.array([-999.]*len(good))
            raise ConstraintError("PTW bad val")

        out = functions.PTW_Yield_Stress(
            p=mp.p,
            kappa=mp.kappa,
            s0=mp.s0,
            sInf=mp.sInf,
            y0=mp.y0,
            yInf=mp.yInf,
            y1=mp.y1,
            y2=mp.y2,
            beta=mp.beta,
            theta=mp.theta,
            lgamma=mp.lgamma,
            edot=edot,
            rho0=mp.rho0,
            matomic=mp.matomic,
            shear=self.parent.state.G,
            eps=self.parent.state.strain,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
            small=1.0e-10,
        )
        out[np.where(np.logical_not(good))] = -999.0
        return out


class Stein_Flow_Stress(BaseModel):
    """
    This class implements the Steinberg-Guinan flow stress model
    """

    def __init__(self, parent):
        BaseModel.__init__(self, parent)
        # TODO: generalize this model to include strain-rate dependence
        # self.params = ["y0", "a", "b", "beta", "n", "ymax"]
        ## params a, b are never used, so drop them:
        self.params = ["y0", "beta", "n", "ymax"]
        self.consts = ["G0", "epsi", "chi"]

    def value(self, *args):
        mp = self.parent.parameters
        return functions.Stein_Flow_Stress(
            y0=mp.y0,
            beta=mp.beta,
            n=mp.n,
            ymax=mp.ymax,
            G0=mp.G0,
            epsi=mp.epsi,
            shear=self.parent.state.G,
            eps=self.parent.state.strain,
            T=self.parent.state.T,
            Tmelt=self.parent.state.Tmelt,
        )


## Parameters Definition


class ModelParameters:
    def update_parameters(self, x):
        if isinstance(x, np.ndarray):
            self.__dict__.update(dict(zip(self.params, x)))
        elif isinstance(x, dict):
            for key in self.params:
                self.__dict__[key] = x[key]
        elif isinstance(x, list):
            assert len(x) == len(self.params), "Incorrect number of parameters!"
            for i, xi in enumerate(self.params):
                self.__dict__[self.params[i]] = xi
        else:
            raise TypeError(f"Type {type(x)} is not supported.")

    def __init__(self, parent):
        self.params = []
        self.consts = []
        self.parent = parent


## State Definition


class MaterialState:
    def set_state(self, T=300.0, strain=0.0, stress=0.0):
        self.T = T
        self.strain = strain
        self.stress = stress

    def __init__(self, parent, T=300.0, strain=0.0, stress=0.0):
        self.T = T
        self.Tmelt = None
        self.stress = stress
        self.strain = strain
        self.G = None
        self.parent = parent


## Material Model Definition


class MaterialModel:
    def __init__(
        self,
        parameters=ModelParameters,
        initial_state=MaterialState,
        flow_stress_model=Constant_Yield_Stress,
        specific_heat_model=Constant_Specific_Heat,
        shear_modulus_model=Constant_Shear_Modulus,
        melt_model=Constant_Melt_Temperature,
        density_model=Constant_Density,
    ):
        """
        Initialization routine for Material Model.  All of the arguments
        supplied are classes, which are then instantiated within the function.

        The reason for doing this is that then we can pass the MaterialModel
        instance to the physical models so that the model's parent can be
        declared at instantiation.  then the model.value() function can reach
        into the parent class to find whatever it needs.
        """
        self.parameters = parameters(self)
        self.state = initial_state(self)

        self.flow_stress = flow_stress_model(self)
        self.specific_heat = specific_heat_model(self)
        self.shear_modulus = shear_modulus_model(self)
        self.melt_model = melt_model(self)
        self.density = density_model(self)

        params = (
            self.flow_stress.params
            + self.specific_heat.params
            + self.shear_modulus.params
            + self.melt_model.params
            + self.density.params
        )
        consts = set(
            self.flow_stress.consts
            + self.specific_heat.consts
            + self.shear_modulus.consts
            + self.melt_model.consts
            + self.density.consts
        )

        assert len(set(params)) == len(params), (
            "Some Duplicate Parameters between models"
        )
        assert len(set(params).intersection(set(consts))) == 0, (
            "Duplicate item in parameters and constants"
        )

        self.parameters.params = params
        self.parameters.consts = consts

        ## call self.set_history_variables() to set these:
        self.emax = None
        self.edot = None
        self.Nhist = None
        self.strain_history = None

    def get_parameter_list(
        self,
    ):
        """
        The list of parameters used in the model.
        This also describes the order of their appearance in the sampling results
        """
        return self.parameters.params

    def get_constants_list(
        self,
    ):
        """
        List of Constants used in the model
        """
        return self.parameters.consts

    def update_state(self, edot, dt):
        chi = self.parameters.chi
        self.state.Cv = self.specific_heat.value()
        self.state.rho = self.density.value()
        # if we are working with microseconds, then this is a reasonable value
        # if we work in seconds, it should be changed to ~1.
        edotcrit = 1.0e-6
        # if edot > edotcrit:
        #  self.state.T += chi * self.state.stress * edot * dt / (self.state.Cv * self.state.rho)
        cond = edot > edotcrit
        # if any(cond):
        self.state.T = (
            self.state.T
            + cond
            * chi
            * self.state.stress
            * edot
            * dt
            / (self.state.Cv * self.state.rho)
        )
        self.state.strain = self.state.strain + edot * dt

        self.state.Tmelt = self.melt_model.value()
        self.state.G = self.shear_modulus.value()
        self.state.stress = self.flow_stress.value(edot)

    def update_parameters(self, x):
        self.parameters.update_parameters(x)

    def initialize(self, parameters, constants):
        """
        Initialize the model at a given set of parameters, constants
        """
        ## if user assumed one or more parameters constant, they would be in the constants var instead;
        ## check for this first:
        if not isinstance(self.parameters.params, list):
            self.parameters.params = list(self.parameters.params)
        if not isinstance(self.parameters.consts, set):
            self.parameters.consts = set(self.parameters.consts)
        user_constants = set(self.parameters.params).difference(parameters)
        for usercnst in user_constants:
            self.parameters.consts |= {usercnst}
        try:
            self.parameters.__dict__.update(
                {
                    key: parameters[key]
                    for key in set(self.parameters.params).difference(
                        user_constants
                    )
                },
            )
            self.parameters.__dict__ |= {
                key: constants[key] for key in user_constants
            }
        except KeyError:
            print(
                "{} missing from list of supplied parameters".format(
                    set(self.parameters.params).difference(
                        set(parameters.keys())
                    )
                )
            )
            raise
        try:
            self.parameters.__dict__.update(
                {key: constants[key] for key in self.parameters.consts},
            )
        except KeyError:
            print(
                "{} missing from list of supplied constants".format(
                    set(self.parameters.consts).difference(
                        set(constants.keys())
                    )
                )
            )
            raise

    def initialize_state(self, T=300.0, stress=0.0, strain=0.0):
        self.state.set_state(T, stress, strain)

    def set_history_variables(self, emax, edot, nhist):
        """initializes attributes emax, edot, and Nhist, then calls
        generate_strain_history() for those values and stores the
        results as self.strain_history."""
        self.emax = emax
        self.edot = edot
        self.Nhist = nhist
        self.strain_history = generate_strain_history(emax, edot, nhist)

    def get_history_variables(self):
        return [self.emax, self.edot, self.Nhist]

    def compute_state_history(self, strain_history=None):
        if strain_history is None:
            strain_history = self.strain_history
        strains = strain_history["strains"]
        times = strain_history["times"]
        strain_rate = strain_history["strain_rate"]
        # Nhist = len(strains)
        # nrep = len(self.parameters.kappa)
        nrep, Nhist = strains.shape  # nexp * nhist array

        results = np.empty((Nhist, 6, nrep))

        state = self.state
        self.update_state(strain_rate[:, 0], 0.0)

        # import pdb
        # pdb.set_trace()

        results[0] = np.array([
            times[:, 0],
            state.strain,
            state.stress,
            state.T,
            state.G,
            state.rho,
        ])  # np.repeat(state.rho,nrep)])

        for i in range(1, Nhist):
            self.update_state(
                strain_rate[:, i - 1], times[:, i] - times[:, i - 1]
            )
            # self.update_state(strain_rate.T[i-1], times.T[i] - times.T[i-1])
            # results[i] = [times[i], state.strain, state.stress, state.T, state.G, state.rho]
            results[i] = np.array([
                times[:, i],
                state.strain,
                state.stress,
                state.T,
                state.G,
                state.rho,
            ])  # np.repeat(state.rho, nrep)])

        return results


def generate_strain_history(emax, edot, nhist):
    """function to generate strain history to calculate along;
    it is called by the method set_history_variables() within the MaterialModel class."""
    tmax = emax / edot
    if isinstance(emax, float):
        emax = [emax] * len(edot)
    strains = np.linspace(0, emax, nhist)  # nhist * nexp
    times = np.linspace(0, tmax, nhist)  # nhist * nexp
    rates = np.diff(strains, axis=0) / np.diff(
        times, axis=0
    )  # (nhist - 1) * nexp
    return {"times": times.T, "strains": strains.T, "strain_rate": rates.T}


# only for backwards compatibility, TODO: deprecate
generate_strain_history_new = generate_strain_history
