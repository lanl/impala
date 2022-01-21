import numpy as np
import pyBASS as pb
import physical_models_vec as pm_vec
from itertools import cycle
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve 

epsilon = 1e-5

#####################
# model classes should have eval method and stochastic attribute
#####################
class ModelBassPca:
    """ PCA Based Model Emulator """
    def __init__(self, bmod, input_names, pool=True):
        """
        bmod        : ~
        input_names : list of the names of the inputs to bmod
        pool        : Whether using the pooled model
        """
        self.mod = bmod
        self.stochastic = True
        self.nmcmc = len(bmod.bm_list[0].samples.s2)
        self.input_names = input_names
        self.pool = pool # I don't do anything with this below...in hierarchical case, requires one theta for each BassPCA model
        self.trunc_error = self.mod.trunc_error
        self.basis = self.mod.basis
        return

    def eval(self, parmat, pool = None):
        """
        parmat : ~
        """
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        ind = 0#np.random.choice(range(self.nmcmc), 1)
        return self.mod.predict(parmat_array, mcmc_use=np.array([ind]), nugget=True)[0, :, :]

    def llik(self, setup, curr):
        vec = setup.y - curr.pred 
        out = -.5*(curr.cov['ldet'] + vec.T @ curr.cov['inv'] @ vec)
        return out

    def lik_cov_inv(self, setup, curr):
        chol = cho_factor(setup.trunc_error_cov + curr.Sigma + setup.discrep_cov + self.basis @ np.diag(curr.emu_vars) @ self.basis.T)
        inv = cho_solve(chol)
        ldet = 2 * np.sum(np.diag(chol))
        out = {'inv' : inv, 'ldet' : ldet}
        return out


from collections import namedtuple
StatePoolMult = namedtuple(
    'StatePoolEmuMult', 'theta Sigma pred emu_vars cov',
    )

class ModelF:
    def __init__(self, f, input_names): # not sure if this is vectorized
        self.mod = f
        self.input_names = input_names

    def eval(self, parmat, pool = None):
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        return np.apply_along_axis(self.mod, 1, parmat_array)

class ModelPTW:
    """ PTW Model for Hoppy-Bar / Quasistatic Experiments """
    def __init__(self, temps, edots, consts, strain_histories, pool=True):
        """
        temps  : ~
        edots  : ~
        consts : dictionary of constants for PTW model
        strain_histories : List of strain histories for HB/Quasistatic Experiments
        """
        self.meas_strain_histories = strain_histories
        self.meas_strain_max = np.array([v.max() for v in strain_histories])
        self.strain_max = self.meas_strain_max.max()
        self.nhists = sum([len(v) for v in strain_histories])
        self.model = pm_vec.MaterialModel(
            flow_stress_model=pm_vec.PTW_Yield_Stress, shear_modulus_model=pm_vec.Stein_Shear_Modulus,
            )
        self.constants = consts
        self.temps = temps
        self.edots = edots
        self.nexp = len(strain_histories)
        self.Nhist = 100
        self.stochastic = False
        self.pool = pool
        return


    def eval(self, parmat, pool = None): # note: extra parameters ignored
        """ parmat:  dictionary of parameters """
        if (pool is True) or self.pool:  # Pooled Case
            nrep = parmat['p'].shape[0]  # number of temper temps
            parmat_big = {key : np.kron(np.ones(self.nexp), parm) for key, parm in parmat.items()}
        else: # hierarchical case
            nrep = parmat['p'].shape[0] // self.nexp # number of temper temps
            parmat_big = parmat

        edots = np.kron(np.ones(nrep), self.edots) # 1d vector, nexp * temper_temps
        temps = np.kron(np.ones(nrep), self.temps) # 1d vector, nexp * temper_temps
        strain_maxs = np.kron(np.ones(nrep), self.meas_strain_max) # 1d vector, nexp * temper_temps
        ntot = edots.shape[0]  # nexp * temper_temps
        sim_strain_histories = pm_vec.generate_strain_history_new(strain_maxs, edots, self.Nhist)
        self.model.initialize(parmat_big, self.constants)
        self.model.initialize_state(T = temps, stress = np.zeros(ntot), strain = np.zeros(ntot))
        sim_state_histories = self.model.compute_state_history(sim_strain_histories)
        sim_strains = sim_state_histories[:,1].T  # 2d array: ntot, Nhist
        sim_stresses = sim_state_histories[:,2].T # 2d array: ntot, Nhist

        # Expand/Flatten Simulated strains to single vector--ensure no overlap.
        strain_ends = np.hstack((0., np.cumsum(strain_maxs + epsilon)[:-1])) # 1d vector, ntot
        flattened_sim_strain = np.hstack( # 1d vector: ntot * Nhist
            [x + y for x, y in zip(sim_strains, strain_ends)]
            )
        # Expand/flatten simulated stress to single vector
        flattened_sim_stress = np.hstack(sim_stresses)
        # Expand/flatten measured strain to single vector, for each parameter.  Use same
        #  Computed strain ends to ensure no overlap
        flattened_strain = np.hstack( # cycle will repeat through measured strain histories
            [x + y for x, y in zip(cycle(self.meas_strain_histories), strain_ends)]
            )
        ifunc = interp1d(  # Generate the interpolation function.
            flattened_sim_strain, flattened_sim_stress, kind = 'linear', assume_sorted = True
            )
        ypred = ifunc(flattened_strain).reshape(nrep, -1)  # Interpolate, and output.
        return ypred

def interpolate_experiment(args):
    """ Interpolate and predict at x.  Args is tuple(x_observed, y_observed, x_new) """
    ifunc = interp1d(args[0], args[1], kind = 'cubic')
    return ifunc(args[2])

# EOF
