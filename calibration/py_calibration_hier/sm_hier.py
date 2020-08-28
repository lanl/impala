from experiment import Experiment, Transformer
from physical_models_c import MaterialModel
import parallel_tempering as pt
from math import exp, log
import sqlite3 as sql
import numpy as np

class SubChain(Transformer):
    """ experiment-specific sampling of strength parameters """
    s_theta  = None
    s_sigma2 = None
    s_iter   = 0

    prior_sigma2_a = 100.
    prior_sigma2_b = 1.e-4

    def set_state(self, theta, sigma2):
        self.s_theta[self.s_iter] = theta
        self.s_sigma2[self.s_iter] = sigma2

    @property
    def curr_theta(self):
        return self.s_theta[self.s_iter]

    @property
    def curr_sigma2(self):
        return self.s_sigma2[self.s_iter]

    @property
    def curr_theta0(self):
        return self.parent.curr_theta0

    @property
    def curr_Sigma(self):
        return self.parent.curr_Sigma

    def probit_sse(self, theta):
        return self.experiment.sse(tuple(self.unnormalize(self.invprobit(theta))))

    def initialize_sampler(self, ns):
        self.s_iter = 0
        self.s_theta = np.array((ns, self.d))
        self.s_sigma2 = np.array(ns)
        self.s_theta[0] = 0
        return

    def sample_sigma2(self, theta):
        aa = 0.5 * self.N / self.temper_temp + self.priors_sigma2.a
        bb = 0.5 * self.probit_sse(theta) / self.temper_temp + self.priors_sigma2.b
        return invgamma.rvs(aa, scale = bb)

    def log_posterior_theta(self, theta, sigma2, theta0, SigmaInv):
        ssse = self.probit_sse(theta) / sigma2
        sssd = (theta - theta0).dot(SigmaInv).dot(theta - theta0)
        ldj  = self.invprobitlogjac(theta)
        ll   = 0.5 * (ssse + sssd) / self.temper_temp
        return ll + ldj

    def localcov(self, target):
        lc = localcov(self.s_theta[:(self.s_iter - 1)], target, **self.temper_args)
        return lc

    def sample_theta(self, curr_theta, sigma2):
        curr_cov = self.localcov(curr_theta)
        prop_theta = curr_theta + normal(size = self.d).dot(cholesky(curr_cov))
        if not self.check_constraints(prop_theta):
            return curr_theta
        prop_cov = self.localcov(prop_theta)
        curr_lp = self.log_posterior_theta(curr_theta,       sigma2,
                                           self.curr_theta0, self.curr_Sigma)
        prop_lp = self.log_posterior_theta(prop_theta,       sigma2,
                                           self.curr_theta0, self.curr_Sigma)
        cp_ld = mvnorm.logpdf(prop_theta, curr_theta, curr_cov)
        pc_ld = mvnorm.logpdf(curr_theta, prop_theta, prop_cov)

        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld

        if log(uniform()) < log_alpha:
            return prop_theta
        else:
            return curr_theta

    def iter_sample(self):
        curr_theta = self.curr_theta
        self.s_iter += 1

        self.s_sigma2[self.s_iter] = self.sample_sigma2(curr_theta)
        self.s_theta[self.s_iter] = self.sample_theta(curr_theta, self.curr_sigma2)
        return

    def sample(self, ns):
        for _ in range(ns):
            self.iter_sample()
        return

    def __init__(self, parent, conn, table_name, type, bounds, constants, model_args):
        cursor = conn.cursor()
        self.experiment = Experiment[type](cursor, table_name, model_args)
        # Exposing Model Properties
        self.parameter_list = self.experiment.parameter_list
        self.d = len(self.parameter_list)
        # Initializing Model
        self.bounds = np.array([bounds[key] for key in self.parameter_list])
        init_param = self.unnormalize(np.array([.5] * self.d))
        init_param_d = {x : y for x, y in zip(self.parameter_list, init_param)}
        self.experiment.initialize_model(init_param_d, constants)
        return

class Chain(Transformer, pt.PTSlave):
    """ hierarchical sampling of strength parameters """
    s_theta0 = None
    s_Sigma  = None

    def get_state(self):
        state = {
            'theta'    : self.curr_theta,
            'sigma2'   : self.curr_sigma2,
            'theta0'   : self.curr_theta0,
            'Sigma'    : self.curr_Sigma,
            'SigmaInv' : self.curr_Sigma_Inv,
            }
        return state

    def set_state(self, state):
        self.s_theta0[self.s_iter] = state['theta0']
        self.s_Sigma[self.s_iter] = state['Sigma']
        for i in range(N):
            self.subchains[i].set_state(state['theta'][i], state['sigma2'][i])
        return

    @property
    def curr_theta(self):
        return np.array([subchain.curr_theta for subchain in self.subchains])

    @property
    def curr_sigma2(self):
        return np.array([subchain.curr_sigma2 for subchain in self.subchains])

    @property
    def curr_theta0(self):
        return self.s_theta0[self.s_iter]

    @property
    def curr_Sigma(self):
        return self.s_Sigma[self.s_iter]

    @property
    def curr_Sigma_Inv(self):
        return self.pd_matrix_inversion(self.curr_Sigma)

    def initialize_sampler(self, ns):
        for subchain in self.subchains:
            subchain.initialize_sampler(ns)

        self.s_iter   = 0
        self.s_theta0 = np.array((ns, self.d))
        self.s_Sigma  = np.array((ns, self.d, self.d))
        self.s_theta0[0] = 0
        self.s_Sigma[0] = np.eye(self.d)
        return

    def iter_sample(self):
        for subchain in self.subchains:
            subchain.iter_sample()

        curr_Sigma = self.curr_Sigma

        self.s_iter += 1
        self.s_theta0[self.s_iter] = self.sample_theta0(self.curr_theta, curr_Sigma)
        self.s_Sigma[self.s_iter]  = self.sample_Sigma(self.curr_theta, self.curr_theta0)
        return

    def sample_Sigma(self, theta, theta0):
        tdiff = theta - theta0
        C = sum([np.outer(tdiff[i], tdiff[i]) for i in range(tdiff.shape[0])])
        psi0 = self.prior_Sigma_psi + C / self.temper_temp
        nu0  = self.prior_Sigma_nu  + self.N / self.temper_temp
        return invwishart.rvs(df = nu0, scale = psi0)

    def sample_theta0(self, theta, Sigma):
        theta_bar = theta.mean(axis = 0)
        SigInv = self.pd_matrix_inversion(Sigma)
        S0 = self.pd_matrix_inversion(
            (self.N / self.temper_temp) * SigInv + self.prior_theta0_Sinv
            )
        M0 = S0.dot((self.N / self.temper_temp) * SigInv.dot(theta_bar) +
                    self.prior_theta0_Sinv.dot(self.prior_theta0_mu))
        gen = mvnorm(mean = M0, cov = S0)
        theta0_try = gen.rvs()
        while not self.check_constraints(theta0_try):
            theta0_try = gen.rvs()
        return theta0_try

    def check_constraints(self, theta):
        param = self.unnormalize(self.invprobit(theta))
        self.model.update_parameters(param)
        return self.model.check_constraints()

    def __init__(self, temp, path, bounds, constants, model_args):
        self.model = MaterialModel(model_args)
        self.parameter_list = self.model.get_parameter_list()
        self.bounds = np.array([bounds[key] for key in self.parameter_list])
        self.d = len(self.parameter_list)
        self.set_temper_temp(temp)
        return

class PTMaster(pt.PTMaster):
    def initialize_sampler(self, ns):
        for chain in self.chains:
            chain.initialize_sampler(ns)

    def sample(self, nsamp, k):
        self.initialize_sampler(nsamp + 1)
        for _ in range(int(nsamp / k)):
            self.sample_k(k)
            self.try_swap_states()
        self.sample(nsamp % k)
        return

    def __init__(self, temperatures, **kwargs):
        self.chains = [Chain(temp, **kwargs) for temp in temperatures]
        return
# EOF
