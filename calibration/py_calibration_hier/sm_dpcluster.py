from experiment import Experiment, Transformer, cholesky_inversion
import parallel_tempering as pt
from physical_models_c import MaterialModel
from collections import namedtuple
from math import exp, log
import sqlite3 as sql
import numpy as np
from numpy.linalg import multi_dot
from functools import lru_cache

# Defining Sum-squared-error
#   Holding this outside of the SubChain model (or Experiment)
#   for the purpose of parallelizing it.  Cluster membership
#   can be farmed out to multiple processors without issue.
#   Not sure how much cost there is in *making* the material model
#   relative to going through the model updating.  I guess we'll see...

def sse_shpb(exp_tuple, parameters, constants, model_args):
    """
    Computes sum-squared-error for a given experiment.

    kappa      : Kappa (namedtuple, with theta and sigma2 as terms)
               :-> theta  = parameter vector (probit scale)
               :-> sigma2 = error vector
    constants  : np.array (d) (constants used)
    model_args : dict (what models are used)

    """
    # Build model, predict stress at given strain values
    model = MaterialModel(model_args)
    model.set_history_parameters(exptuple.emax, exptuple.edot, 100)
    model.initialize_constants(constants)
    model.update_parameters(parameters)
    model.initialize_state(exptuple.temp)
    res = (model.compute_state_history()[:, 1:3]).T
    est_curve = interp1d(res[0], res[1], kind = 'cubic')
    preds = est_curve(extuple.X)
    # Compute diffrences, return sum-squared-error
    diffs = (exptuple.Y - preds)
    return (diffs * diffs).sum()

sse = {
    'shpb' : sse_shpb
    }

# Defining SubChains

class SubChainBase(object):
    samples = None
    experiment = None
    N = None

    @property
    def curr_sigma2(self):
        raise NotImplementedError('Overwrite this!')

    def log_posterior_theta(self, sse):
        raise NotImplementedError('Overwrite this!')

    def sample_sigma2(self, sse):
        raise NotImplementedError('Overwrite this!')

    def iter_sample(self, sse):
        raise NotImplementedError('Overwrite this!')

    def initialize_sampler(self, ns):
        raise NotImplementedError('Overwrite this!')

    def __init__(self, **kwargs):
        raise NotImplementedError('Overwrite this!')

SamplesSHPB = namedtuple('SamplesSHPB', 'sigma2')
PriorsSHPB = namedtuple('PriorsSHPB', 'a b')


class SubChainSHPB(SubChainBase):
    samples = None
    experiment = None
    N = None

    @property
    def curr_sigma2(self):
        return self.samples.sigma2[self.curr_iter]

    def log_posterior_theta(self, sse):
        return -0.5 * sse / self.curr_sigma2

    def sample_sigma2(self, sse):
        aa = self.N * self.parent.inv_temper_temp + self.priors.a
        bb = sse * self.parent.inv_temper_temp + self.priors.b
        return invgamma.rvs(aa, scale = bb)

    def iter_sample(self, sse):
        self.curr_iter += 1
        self.samples_sigma2[self.curr_iter] = self.sample_sigma2(sse)
        return

    def initialize_sampler(self, ns):
        self.samples = SamplesSHPB(np.empty(ns + 1))
        self.samples[0] = gamma.rvs(self.priors.a, scale = 1/self.priors.b)
        return

    def __init__(self, experiment):
        self.experiment = experiment
        self.priors = PriorsSHPB(50, 1e-6)
        self.N = self.experiment.X.shape[0]
        return

SubChain = {
    'shpb' : SubChainSHPB
    }

Samples = namedtuple("Samples", "theta delta Sigma theta0")
PriorsChain = namedtuple('PriorsChain', 'psi nu mu Sinv')
class Chain(Transformer, pt.PTSlave):
    samples = None
    N = None

    @staticmethod
    def clean_delta_theta(delta, theta):
        nj = np.array(
            [(delta == j).sum() for j in range(delta.max() + 2)],
            dtype = np.int,
            )
        j = np.where(nj == 0)[0]
        while(delta.max() > j):
            delta[delta > j] = delta[delta > j] - 1
            theta = np.delete(theta, j, axis = 0)
            nj = np.array(
                [(delta == j).sum() for j in range(delta.max() + 2)],
                dtype = np.int,
                )
            j = np.where(nj == 0)[0]
        return delta, theta

    def sample_delta_i(self, delta, theta, theta0, Sigma, alpha, i):
        _delta, _theta = self.clean_delta_theta(np.delete(delta, i), theta)
        _dmax  = _delta.max()
        nj     = np.array([(delta == j).sum() for in in range(_dmax + 1 + self.m)])
        lj     = nj + (nj == 0) * alpha / self.m
        th_new = self.sample_theta_new(self.theta0, self.Sigma, self.m)
        thetas = np.vstack((_theta, th_new))
        phis   = self.unnormalize(self.invprobit(thetas))
        assert(thetas.shape[0] == lj.shape[0])
        args   = zip(
                repeat(self.subchains[i].tuple),    phis.tolist(),
                repeat(self.constants_vec),         repeat(self.model_args),
                )
        sses   = np.array(list(self.pool.map(args, lambda arg: sse[arg[0].type](*arg))))
        sigma2 = self.subchains[i].curr_sigma2
        lps    = np.array(self.subchains[i].log_posterior_theta(sse) for sse in sses)
        probs  = exp(lps) * nj / (exp(lps) * nj).sum()
        dnew   = choice(range(_dmax + 1 + self.m), 1, p = probs)
        if dnew > _dmax:
            theta = np.vstack((_theta,thetas[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
        return delta, theta

    def log_posterior_theta_j(self, theta_j, cluster_j, theta0, SigInv):
        phi_j = self.unnormalize(self.invprobit(theta_j))
        clust_exp_tuples = [self.subchains[i].experiment.tuple for i in cluster_j]
        args  = zip(
                clust_exp_tuples,           repeat(phi_j),
                repeat(self.constants_vec), repeat(self.model_args),
                )
        sses  = np.array(list(self.pool.map(args, lambda arg: sse[arg[0].type](*arg))))
        lliks = np.array([
                    self.subchains[i].log_posterior_theta(sse)
                    for i, sse in zip(cluster_j, sses)
                    ])
        llik  = lliks.sum()
        lpri  = - 0.5 * multi_dot((theta_j - theta0, SigInv, theta_j - theta0))
        ldj   = self.invprobitlogjac(theta_j)
        return llik + lpri + ldj

    def sample_theta_j(self, curr_theta_j, cluster_j, theta0, SigInv):
        """
        cluster_j is an array of indices on which cluster j holds sway.
        I.e., curr_theta_j is the current value of theta for cluster j.
        """
        curr_cov = self.localcov(np.vstack(self.samples.theta[:-1]), curr_theta_j)
        gen = mvnorm(mean = curr_theta_j, cov = curr_cov)
        prop_theta_j = gen.rvs()

        while not self.check_constraints(prop_theta_j):
            prop_theta_j = gen.rvs()

        prop_cov = self.localcov(np.vstack(self.samples.theta[:-1]), prop_theta_j)

        curr_lp = self.log_posterior_theta_j(curr_theta_j, cluster_j, theta0, Sigma)
        prop_lp = self.log_posterior_theta_j(prop_theta_j, cluster_j, theta0, Sigma)

        cp_ld   = gen.logpdf(prop_theta_j)
        pc_ld   = mvnorm(mean = prop_theta_j, cov = prop_cov).logpdf(curr_theta_j)

        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld
        if log(uniform()) < log_alpha:
            return prop_theta_j
        else:
            return curr_theta_j

    def sample_thetas(self, thetas, delta, theta0, Sigma):
        assert (thetas.shape[0] == delta.max() + 1)
        theta_new = np.empty((theta.shape))
        SigInv = self.pd_matrix_inversion(Sigma)
        for j in range(theta_new.shape[0]):
            theta_new[j] = self.sample_theta_j(thetas[j], np.where(delta == j), theta0, SigInv)
        return theta_new

    def sample_theta0(self, theta, Sigma):
        if len(theta.shape) == 1:
            theta_bar = theta
            N = 1
        else:
            theta_bar = theta.mean(axis = 0)
            N = theta.shape[0]
        SigInv = self.pd_matrix_inversion(Sigma)
        S0 = self.pd_matrix_inversion(N * self.inv_temper_temp * SigInv + self.priors.Sinv)
        M0 = S0.dot(N * self.inv_temper_temp * SigInv.dot(theta_bar) +
                    self.priors.Sinv.dot(self.priors.mu))
        gen = mvnorm(mean = M0, cov = S0)
        theta0_try = gen.rvs()
        while not self.check_constraints(theta0_try):
            theta0_try = gen.rvs()
        return theta0_try

    def sample_theta_new(self, theta0, Sigma, ns):
        theta_new = np.empty((ns, self.d))
        gen = mvnorm(mean = theta0, cov = Sigma)
        for i in range(ns):
            theta_try = gen.rvs()
            while not self.check_constraints(theta_try):
                theta_try = gen.rvs()
            theta_new[i] = theta_try
        return theta_new

    def sample_Sigma(self, theta, theta0):
        N = theta.shape[0]
        diff = theta - theta_0
        C = np.array([np.outer(diff[i], diff[i]) for i in range(N)]).sum(axis = 0)
        psi0 = self.priors.psi + C * self.inv_temper_temp
        nu0  = self.priors.nu  + N * self.inv_temper_temp
        return invwishart.rvs(df = nu0, scale = psi0)

    def check_constraints(self, theta):
        self.model.update_parameters(self.unnormalize(self.invprobit(theta)))
        return self.model.check_constraints()

    def localcov(self, history, target):
        return localcov(history, target, *self.cov_args)

    def initialize_sampler(self, ns):
        self.samples = Samples([], np.empty((ns + 1, self.N)),
                                np.empty((ns + 1, self.d, self.d)), np.empty((ns + 1, self.d)))
        self.samples.delta[0] = range(self.N)
        theta_start = np.empty((self.N, self.d))
        init_normal = normal(0, 0.5)
        for i in range(self.N):
            theta_try = init_normal.rvs(size = self.d)
            while not self.check_constraints(theta_try):
                theta_try = init_normal.rvs(size = self.d)
            theta_start[i] = theta_try
        self.samples.theta.append(theta_start)
        theta_try = init_normal.rvs(size = self.d)
        while not self.check_constraints(theta_try):
            theta_try = init_normal.rvs(size = self.d)
        self.samples.theta0[0] = theta_try
        self.samples.Sigma[0] = self.sample_Sigma(self.samples.theta0[0], self.samples.theta[0])
        return

    def __init__(self, path, constants, model_args):
        conn = sql.connect(path)
        cursor = conn.cursor()
        self.model = MaterialModel(model_args)
        self.parameter_list = self.model.get_parameter_list()
        self.constant_list  = self.model.get_constant_list()
        self.constant_vec   = np.array([constants[key] for key in self.constant_list])
        meta = list(cursor.execute(" SELECT type, table_name FROM meta; "))
        self.subchains = [
            SubChain[type](Experiment[type](cursor, table_name, model_args))
            for type, table_name in tables
            ]
        self.pool = Pool(8)
        return

# EOF
