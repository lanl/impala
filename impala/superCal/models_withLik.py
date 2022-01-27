import numpy as np
import pyBASS as pb
#import physical_models_vec as pm_vec
from impala import physics as pm_vec
from itertools import cycle
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve, cholesky
import scipy.linalg.lapack as la
import abc


epsilon = 1e-5

def cor2cov(R, s): # R is correlation matrix, s is sd vector
    return(np.outer(s, s) * R)

def chol_sample(mean, cov):
    return mean + np.dot(np.linalg.cholesky(cov), np.random.standard_normal(mean.size))

#####################
# model classes should have eval method and stochastic attribute
#####################
class AbstractModel:
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def eval(self, parmat): # this must be implemented for each model type
        pass

    def llik(self, yobs, pred, cov): # assumes diagonal cov
        vec = yobs - pred 
        out = -.5 * cov['ldet'] - .5 * np.sum(vec**2 * np.diag(cov['inv']))
        return out

    def lik_cov_inv(self, s2vec): # default is diagonal covariance matrix
        inv = np.diag(1/s2vec)
        ldet = np.sum(np.log(s2vec))
        out = {'inv' : inv, 'ldet' : ldet}
        return out

    def step(self):
        return


class ModelBassPca_mult(AbstractModel):
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
        self.trunc_error_cov = np.cov(self.mod.trunc_error)
        self.basis = self.mod.basis
        self.meas_error_cor = np.eye(self.basis.shape[0])
        self.discrep_cov = np.eye(self.basis.shape[0])*1e-12
        self.ii = 0
        npc = self.mod.nbasis
        self.mod_s2 = np.empty([self.nmcmc, npc])
        for i in range(npc):
            self.mod_s2[:,i] = self.mod.bm_list[i].samples.s2
        self.emu_vars = self.mod_s2[self.ii]
        self.yobs = None
        self.marg_lik_cov = None
        self.nd = 0
        return

    def step(self):
        self.ii = np.random.choice(range(self.nmcmc), 1).item()
        self.emu_vars = self.mod_s2[self.ii]
        return

    def eval(self, parmat, pool = None):
        """
        parmat : ~
        """
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        return self.mod.predict(parmat_array, mcmc_use=np.array([self.ii]), nugget=False)[0, :, :]

    def llik(self, yobs, pred, cov):
        vec = yobs - pred 
        out = -.5*(cov['ldet'] + vec.T @ cov['inv'] @ vec)
        return out

    def lik_cov_inv(self, s2vec):
        n = len(s2vec)
        Sigma = cor2cov(self.meas_error_cor[:n,:n], np.sqrt(s2vec)) # :n is a hack for when ntheta>1 in heir...fix this sometime
        mat = Sigma + self.trunc_error_cov + self.discrep_cov + self.basis @ np.diag(self.emu_vars) @ self.basis.T
        chol = cholesky(mat)
        ldet = 2 * np.sum(np.log(np.diag(chol)))
        #la.dpotri(chol, overwrite_c=True) # overwrites chol with original matrix inverse
        inv=np.linalg.inv(mat)
        out = {'inv' : inv, 'ldet' : ldet}
        return out





class ModelBassPca_func(AbstractModel):
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
        self.trunc_error_var = np.diag(np.cov(self.mod.trunc_error))
        self.basis = self.mod.basis
        self.meas_error_cor = np.eye(self.basis.shape[0])
        self.discrep_cov = np.eye(self.basis.shape[0])*1e-12
        self.ii = 0
        npc = self.mod.nbasis
        self.mod_s2 = np.empty([self.nmcmc, npc])
        for i in range(npc):
            self.mod_s2[:,i] = self.mod.bm_list[i].samples.s2
        self.emu_vars = self.mod_s2[self.ii]
        self.yobs = None
        self.marg_lik_cov = None
        self.discrep_vars = None
        self.nd = 0
        self.discrep_tau = 1.
        self.D = None
        self.discrep = 0.
        return

    def step(self):
        self.ii = np.random.choice(range(self.nmcmc), 1).item()
        self.emu_vars = self.mod_s2[self.ii]
        return
    #@profile
    def discrep_sample(self, yobs, pred, cov, itemp):
        #if self.nd>0:
        S = np.linalg.inv(
            np.eye(self.nd) / self.discrep_tau 
            + self.D.T @ cov['inv'] @ self.D
            )
        m = self.D.T @ cov['inv'] @ (yobs - pred)
        discrep_vars = chol_sample(S @ m, S/itemp)
        #self.discrep = self.D @ self.discrep_vars
        return discrep_vars
    #@profile
    def eval(self, parmat, pool = None):
        """
        parmat : ~
        """
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        return self.mod.predict(parmat_array, mcmc_use=np.array([self.ii]), nugget=False)[0, :, :]
    #@profile
    def llik(self, yobs, pred, cov):
        vec = yobs - pred 
        out = -.5*(cov['ldet'] + vec.T @ cov['inv'] @ vec)
        return out
    #@profile
    def lik_cov_inv(self, s2vec):
        vec = self.trunc_error_var + s2vec
        #mat = np.diag(vec) + self.basis @ np.diag(self.emu_vars) @ self.basis.T
        #inv = np.linalg.inv(mat)
        #ldet = np.linalg.slogdet(mat)[1]
        #out = {'inv' : inv, 'ldet' : ldet}
        Ainv = np.diag(1/vec)
        Aldet = np.log(vec).sum()
        out = self.swm(Ainv, self.basis, np.diag(1/self.emu_vars), self.basis.T, Aldet, np.log(self.emu_vars).sum())
        return out
    #@profile
    def chol_solve(self, x):
        mat = cho_factor(x)
        ldet = 2 * np.sum(np.log(np.diag(mat[0])))
        ##la.dpotri(mat, overwrite_c=True) # overwrites mat with original matrix inverse, but not correct
        #inv = cho_solve(mat, np.eye(x.shape[0])) # correct, but slower for small dimension
        inv = np.linalg.inv(x)
        out = {'inv' : inv, 'ldet' : ldet}
        return out
    #@profile
    def swm(self, Ainv, U, Cinv, V, Aldet, Cldet): # sherman woodbury morrison (A+UCV)^-1 and |A+UCV|
        in_mat = self.chol_solve(Cinv + V @ Ainv @ U)
        inv = Ainv - Ainv @ U @ in_mat['inv'] @ V @ Ainv
        ldet = in_mat['ldet'] + Aldet + Cldet
        out = {'inv' : inv, 'ldet' : ldet}
        return out



class ModelF(AbstractModel):
    def __init__(self, f, input_names): # not sure if this is vectorized
        self.mod = f
        self.input_names = input_names
        self.stochastic = False
        self.yobs = None
        self.meas_error_cor = 1.#np.diag(self.basis.shape[0])
        return

    def eval(self, parmat, pool = None):
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        return np.apply_along_axis(self.mod, 1, parmat_array)





class ModelPTW(AbstractModel):
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
        self.yobs = None
        #self.meas_error_cor = np.diag(self.basis.shape[0])
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
