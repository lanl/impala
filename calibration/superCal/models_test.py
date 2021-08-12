import numpy as np
import pyBASS as pb
import physical_models_vec as pm_vec
from scipy.interpolate import interp1d

#####################
# model classes should have eval method and stochastic attribute
#####################
class ModelBassPca:
    def __init__(self, bmod, input_names, pool=True): # input_names is a list of the names of the inputs to bmod
        self.mod = bmod
        self.stochastic = True
        self.nmcmc = len(bmod.bm_list[0].samples.s2)
        self.input_names = input_names
        self.pool = pool # I don't do anything with this below...in hierarchical case, requires one theta for each BassPCA model

    def eval(self, parmat):
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        ind = 0#np.random.choice(range(self.nmcmc), 1)
        return self.mod.predict(parmat_array, mcmc_use=np.array([ind]), nugget=True)[0, :, :]

class ModelF:
    def __init__(self, f, input_names): # not sure if this is vectorized
        self.mod = f
        self.input_names = input_names

    def eval(self, parmat):
        parmat_array = np.vstack([parmat[v] for v in self.input_names]).T # get correct subset/ordering of inputs
        return np.apply_along_axis(self.mod, 1, parmat_array)

class ModelPTW:
    def __init__(self, temps, edots, consts, strain_histories, pool=True): # list of strain_histories, eval interpolates to these
        self.meas_strain_histories = strain_histories
        self.meas_strain_max = np.array([v.max() for v in strain_histories])
        self.strain_max = self.meas_strain_max.max()
        self.nhists = sum([len(v) for v in strain_histories])
        #self.sim_strain_histories = pm_vec.generate_strain_history(self.strain_max, edots, Nhist=100)
        self.model = pm_vec.MaterialModel(flow_stress_model=pm_vec.PTW_Yield_Stress, shear_modulus_model=pm_vec.Stein_Shear_Modulus)
        self.constants = consts
        self.temps = temps
        self.edots = edots
        self.nexp = len(strain_histories)
        #self.Nhist = 100
        self.Nhist = 50
        self.stochastic = False
        self.pool = pool

        self.xpred = np.hstack(strain_histories) # flatten for use with vectorized interpolator, strains for 1st experiment, then strains for 2nd experiment, etc.
        flatten_shiftx = []
        for i in range(self.nexp):
            flatten_shiftx.append(np.ones(len(self.meas_strain_histories[i]))*(self.strain_max+.0001)*i)
        self.flatten_shiftx = np.hstack(flatten_shiftx)

    def eval(self, parmat, expList = None): # note: extra parameters ignored
        """
        parmat:  dictionary of parameters
        explist: array of experiment indices that are under evaluation (dtype = int)
                if hierarchical, then len(parmat[n][key]) == len(explist)
        """
        if expList is None: # Pooled Case
            nrep = parmat['p'].shape[0]
            edots = np.kron(np.ones(nrep), self.edots)
            temps = np.kron(np.ones(nrep), self.temps)
            parmat_big = {key : np.kron(np.ones(self.nexp), parm) for key, parm in parmat.items()}
            strain_maxs = np.kron(np.ones(nrep), self.meas_strain_max)
            strain_hists = np.kron(np.ones(nrep), self.meas_strain_histories)
            # xpred_big = np.kron(np.ones(nrep), self.xpred) this isn't gonna work
        else:                    # hierarchical case
            edots = self.edots[explist]
            temps = self.temps[explist]
            parmat_big = parmat
            strain_maxs = self.meas_strain_max[explist]
            strain_hists = self.meas_strain_histories[explist]

        # Building xpred vector
        strain_ends = np.hstack(np.array((0.,)), np.cumsum(strain_maxs + 1e-4)[:-1])
        flattened_strain = np.hstack([x + y for x, y in zip(strain_hists, strain_ends)])
        ntot = edots.shape[0]
        sim_strain_histories = pm_vec.generate_strain_history_new(strain_maxs, edots, self.Nhist)
        self.model.initialize(parmat_big, self.constants)
        self.model.initialize_state(T = temps, stress = np.zeros(ntot), strain = np.zeros(ntot))
        state_hists = self.model.compute_state_history(sim_strain_histories)
        sim_strains = results[:,1]
        sim_stresses = results[:,2]

        flattened_sim_strain = np.hstack([x + y for x, y in zip(sim_strains, strain_ends])])
        flattened_sim_stress = np.hstack(sim_stresses)

        ifunc = interp1d(flattened_sim_strain, flattened_sim_stress, kind = linear,
                                            assume_sorted = True, fill_value = 'extrapolate')
        ypred = ifunc(flattened_strain).reshape((nrep, self.nhists)) # this can't be right...

        # if self.pool:
        #     nrep = len(parmat['p'])
        #     parmat_big = {k: np.kron(v, np.ones((self.nexp))) for k, v in parmat.items()} # parameter values constant while edot/temp loop
        # else:
        #     nrep = len(parmat[0]['p']) # hierarchical case should pass a list of dicts
        #     parmat_big = parmat[0]
        #     for i in range(1,self.nexp):
        #         for key in parmat[i]:
        #             parmat_big[key] = [parmat_big[key], np.kron(parmat[i].items(), np.ones((self.nexp)))]

        # ntot = nrep * self.nexp
        # edots_big = np.kron(np.ones(nrep), self.edots)
        # temps_big = np.kron(np.ones(nrep), self.temps)
        # xpred2 = np.kron(np.ones(nrep), self.xpred)
        #
        # flatten_shiftx = []
        # for i in range(nrep):
        #     flatten_shiftx.append(self.flatten_shiftx+i*(self.strain_max+.0001)*self.nexp)
        #
        # flatten_shiftx = np.hstack(flatten_shiftx)
        # xpred2 = xpred2 + flatten_shiftx
        #
        # sim_strain_histories = pm_vec.generate_strain_history(self.strain_max, edots_big, Nhist=self.Nhist)
        # # note: everything above here could go in __init__ if we knew nrep -> maybe make a method to change nrep
        #
        # self.model.initialize(parmat_big, self.constants)
        # self.model.initialize_state(T=temps_big, stress=np.repeat(0., ntot), strain=np.repeat(0., ntot))
        # state_hists = self.model.compute_state_history(sim_strain_histories)
        #
        # flatten_shiftx_sim = np.array([float(i) * (self.strain_max+.0001) for i in list(range(ntot))])
        # strain_hists2 = state_hists[:, 1, :] + np.kron(flatten_shiftx_sim,np.ones((self.Nhist,1)))
        #
        # x = strain_hists2.flatten("F") # ordered as 100 strains for first experiment, first param setting, then 100 strains for 2nd experiment, 1st param setting...
        # y = state_hists[:, 2, :].flatten("F")
        # ifunc = interp1d(x, y, kind='linear', assume_sorted=False, fill_value="extrapolate")
        # ypred = ifunc(xpred2).reshape([nrep,self.nhists])
        # flattening and then interpolating is much faster than looping over interpolating

        return ypred #[x,y,xpred2,state_hists,ypred]

def interpolate_experiment(args):
    """ Interpolate and predict at x.  Args is tuple(x_observed, y_observed, x_new) """
    ifunc = interp1d(args[0], args[1], kind = 'cubic')
    return ifunc(args[2])

# EOF
