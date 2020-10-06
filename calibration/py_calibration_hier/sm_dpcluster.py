"""
Base code for Chinese Restaurant Process clustering of Strength Model Parameters

Chain - main sampler.

User-Definable Constants: POOL_SIZE (size of multiprocessing pool that gets spawned)
"""
# Required Modules
from scipy.stats import invgamma, invwishart, uniform, beta, gamma
from scipy.stats import multivariate_normal as mvnormal, norm as normal
from scipy.interpolate import interp1d
from numpy.linalg import multi_dot, slogdet, cholesky
from numpy.random import choice
import numpy as np
np.seterr(under = 'ignore')
# Builtins
from collections import namedtuple
from itertools import repeat
from functools import lru_cache
from multiprocessing import Pool
from math import exp, log, prod
import matplotlib.pyplot as plt
import sqlite3 as sql
import os
# Custom Modules
from experiment import Experiment, Transformer, cholesky_inversion
from physical_models_c import MaterialModel
from pointcloud import localcov
import pickledBass as pb
import pt
import ipdb #ipdb.set_trace()

POOL_SIZE = 8

# Defining Sum-squared-error
#   Holding this outside of the SubChain model (or Experiment)
#   for the purpose of parallelizing it.  Cluster membership
#   can be farmed out to multiple processors without issue.
#   Not sure how much cost there is in *making* the material model
#   relative to going through the model updating.  I guess we'll see...

class Bunch(object):
    """
    Bunch object.  Object for holding arbitrary keywords
        (like a dictionary, but neater; like a namedtuple, but less restrictive.)
    """
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        return

def predict_shpb(exp_tuple, parameters, constants, model_args, substate):
    """ Form the prediction output for a SHPB experiment """
    model = MaterialModel(**model_args)
    model.set_history_variables(exp_tuple.emax, exp_tuple.edot, 100)
    model.initialize_constants(constants)
    model.update_parameters(np.array(parameters))
    model.initialize_state(exp_tuple.temp)
    return (model.compute_state_history()[:,1:3]).T

def predict_pca(exp_tuple, parameters, constants, model_args, substate):
    #ipdb.set_trace()
    phi_zeta = np.append(parameters, substate.zeta).reshape(1,-1)
    preds = pb.predictPCA(phi_zeta, exp_tuple.samples, exp_tuple.tbasis,
                            exp_tuple.ysd, exp_tuple.ymean, exp_tuple.bounds)
    return preds.reshape(-1)


def predict_wpca(exp_tuple, parameters, constants, model_args, substate):
    pass

def sse_shpb(exp_tuple, parameters, constants, model_args, substate):
    """
    Computes sum-squared-error for a given experiment, for quasi-statics and shpb's.
    exp_tuple  : relevant experiment data wrapped in a tuple. (x, y, emax, edot, temp)
    parameters : real-values of parameters used to calculate sse.
    constants  : np.array (d) (constants used)
    model_args : dict (defines what physical models are used)
    """
    # Build model, predict stress at given strain values
    res = predict_shpb(exp_tuple, parameters, constants, model_args, substate)
    if np.isnan(res).any():
        return 1.e9
    est_curve = interp1d(res[0], res[1], kind = 'cubic')
    preds = est_curve(exp_tuple.X)
    # Compute diffrences, return sum-squared-error
    diffs = exp_tuple.Y - preds
    return diffs @ diffs

def sse_pca(exp_tuple, parameters, constants, model_args, substate):
    #ipdb.set_trace()
    preds = predict_pca(exp_tuple, parameters, constants, model_args, substate)
    diffs = (exp_tuple.Y - preds).reshape(-1)
    return diffs @ diffs

def sse_wpca(exp_tuple, parameters, constants, model_args, substate):
    pass

sse = {
    'shpb' : sse_shpb,
    'pca'  : sse_pca,
    'wpca' : sse_wpca,
    }

predict = {
    'shpb' : predict_shpb,
    'pca'  : predict_pca,
    'wpca' : predict_wpca,
    }

# sse_wrapper
#  Provides a wrapper around the sse dictionary such that it can be called by the multiprocessing
#  pool against a row of a given iterator.  I.e., pool.map(sse_wrapper, arg_iterable)

def predict_wrapper(args):
    """ Provides a wrapper around the sse dictionary so that we can invoke the particular predict
    type relevant to the data, all at the same time. """
    return predict[args[0].type](*args)

def sse_wrapper(args):
    """ Provides a wrapper around the sse dictionary so that we can invoke the particular SSE
    type relevant to the data, all at the same time. """
    #ipdb.set_trace()
    return sse[args[0].type](*args)

# Defining SubChains
#  Each SubChain object is built towards sampling a particular type, and whatever error structure
#  is associated with that type.

class SubChainBase(Transformer):
    """ Base object for subchains to inherit from.  Method signatures set here MUST be overwritten,
    as the Chain object expects them. """
    samples = None
    experiment = None
    N = None

    create_stmt = """ CREATE TABLE {}({}); """
    insert_stmt = """ INSERT INTO {}({}) values ({}); """

    def set_temperature(self, temperature):
        self.temper_temp = temperature
        self.inv_temper_temp = 1. / temperature
        return

    def get_substate(self):
        raise NotImplementedError('overwrite this!')

    def set_substate(self, substate):
        raise NotImplementedError('overwrite this!')

    def log_posterior_theta(self, sse, substate):
        raise NotImplementedError('Overwrite this!')

    def log_posterior_substate(self, sse, substate):
        raise NotImplementedError('overwrite this!')

    def iter_sample(self, sse):
        raise NotImplementedError('Overwrite this!')

    def initialize_sampler(self, ns):
        raise NotImplementedError('Overwrite this!')

    def write_to_disk(self, cursor, prefix, nburn, thin):
        raise NotImplementedError('Overwrite this')

    def __init__(self, **kwargs):
        raise NotImplementedError('Overwrite this!')

PriorsSHPB = namedtuple('PriorsSHPB', 'a b')
SubstateSHPB = namedtuple('SubstateSHPB','sigma2')
class SamplesSHPB(object):
    sigma2 = None

    def __init__(self, ns):
        self.sigma2 = np.empty(ns + 1)
        return

class SubChainSHPB(SubChainBase):
    """ Subchain class for SHPB """
    samples = None
    experiment = None
    N = None

    @property
    def curr_sigma2(self):
        return self.samples.sigma2[self.curr_iter].copy()

    def get_substate(self):
        """ Get current state of subchain """
        return SubstateSHPB(self.curr_sigma2)

    def set_substate(self, substate):
        """ Set current state of subchain """
        self.samples.sigma2[self.curr_iter] = substate.sigma2

    def log_posterior_theta(self, sse, substate):
        """ Compute log-posterior for theta given SSE, sigma2 """
        return -0.5 * sse * self.inv_temper_temp / substate.sigma2

    def log_posterior_substate(self, sse, substate):
        """ Compute log-posterior for substate (that is, theta + sigma2) """
        lpt = self.log_posterior_theta(sse, substate)
        lpp = -((self.N * self.inv_temper_temp + self.priors.a + 1) * log(substate.sigma2)
                    + self.priors.b / substate.sigma2)
        return lpt + lpp

    def sample_sigma2(self, sse):
        """ Given sse, generate sigma2 """
        aa = self.N * self.inv_temper_temp + self.priors.a
        bb = sse * self.inv_temper_temp + self.priors.b
        return invgamma.rvs(aa, scale = bb)

    def iter_sample(self, sse):
        """ Advance the sampler one iteration """
        self.curr_iter += 1
        self.samples.sigma2[self.curr_iter] = self.sample_sigma2(sse)
        return

    def initialize_sampler(self, ns):
        """ Initialize the sampler. Set initial values """
        self.samples = SamplesSHPB(ns)
        self.samples.sigma2[0] = invgamma.rvs(self.priors.a, scale = self.priors.b)
        self.curr_iter = 0
        return

    def write_to_disk(self, cursor, prefix, nburn, thin):
        """ Write Subchain samples to disk """
        sigma2_create = self.create_stmt.format('{}_sigma2'.format(prefix), 'sigma2 REAL')
        sigma2_insert = self.insert_stmt.format('{}_sigma2'.format(prefix), 'sigma2', '?')
        cursor.execute(sigma2_create)
        cursor.executemany(sigma2_insert, [(x,) for x in self.samples.sigma2[nburn::thin].tolist()])
        return

    def __init__(self, parent, experiment, index):
        self.parent     = parent
        self.experiment = experiment
        self.index      = index
        self.table_name = self.experiment.table_name
        self.priors     = PriorsSHPB(25, 1e-6)
        self.N          = self.experiment.X.shape[0]
        return

class SamplesPCA(object):
    sigma2 = None
    eta = None

    def __init__(self, ns, d):
        self.sigma2 = np.empty(ns + 1)
        self.eta    = np.empty((ns + 1, d))
        return

PriorsPCA = namedtuple('PriorsPCA', 'a b')
SubstatePCA = namedtuple('SubstatePCA','eta zeta sigma2')
class SubChainPCA(SubChainBase):
    @property
    def curr_eta(self):
        """ eta is the additional variables on probit scale """
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_zeta(self):
        """ zeta is eta, transformed back to real scale """
        #ipdb.set_trace()
        return self.unnormalize(self.invprobit(self.curr_eta))

    @property
    def curr_sigma2(self):
        return self.samples.sigma2[self.curr_iter].copy()

    @property
    def curr_theta(self):
        return self.parent.curr_thetas[self.parent.curr_delta[self.index]].copy()

    @property
    def curr_phi(self):
        return self.parent.unnormalize(self.invprobit(self.curr_theta))

    def get_substate(self):
        return SubstatePCA(self.curr_eta, self.curr_zeta, self.curr_sigma2)

    def set_substate(self, substate):
        self.samples.sigma2[self.curr_iter] = substate.sigma2
        self.samples.eta[self.curr_iter] = substate.eta
        return

    def log_posterior_theta(self, sse, substate):
        return - 0.5 * sse * self.inv_temper_temp / substate.sigma2

    def log_posterior_eta(self, eta, theta, sigma2):
        phi  = self.parent.unnormalize(self.invprobit(theta))
        zeta = self.unnormalize(self.invprobit(eta))
        t    = self.experiment.tuple
        #ipdb.set_trace()
        #sse  = sse_pca(np.append(phi, zeta), t.samples, t.tbasis, t.ysd, t.ymean, t.bounds)
        sse = sse_pca(t, phi, None, None, self.get_substate()) # zeta gets pulled from substate
        ldj  = self.invprobitlogjac(eta)
        return - 0.5 * sse * self.inv_temper_temp / sigma2 + ldj

    def log_posterior_substate(self, sse, substate):
        lpt = self.log_posterior_theta(sse, substate)
        lpp = -((prod(self.experiment.Y.shape) * self.inv_temper_temp + self.priors.a + 1) * log(substate.sigma2)
                    + self.priors.b / substate.sigma2)
        ldj = self.invprobitlogjac(substate.eta)
        return lpt + lpp + ldj


    def sample_eta(self, curr_eta, theta, sigma2):
        curr_cov = self.parent.localcov(curr_eta)
        prop_eta = cholesky(curr_cov) @ normal.rvs(size = self.d) + curr_eta
        prop_cov = self.parent.localcov(prop_eta)

        curr_lp = self.log_posterior_eta(curr_eta, theta, sigma2)
        prop_lp = self.log_posterior_eta(prop_eta, theta, sigma2)

        if self.d > 1:
            cp_ld = mvnormal(mean = curr_eta, cov = curr_cov).logpdf(prop_eta)
            pc_ld = mvnormal(mean = prop_eta, cov = prop_cov).logpdf(curr_eta)
        elif self.d == 1:
            cp_ld = normal(loc = curr_eta, scale = curr_cov).logpdf(prop_eta)
            pc_ld = normal(loc = prop_eta, scale = prop_cov).logpdf(curr_eta)

        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld
        if log(uniform.rvs()) < log_alpha:
            return prop_eta
        else:
            return curr_eta
        pass

    def sample_sigma2(self, sse):
        aa = prod(self.experiment.Y.shape) * self.inv_temper_temp + self.priors.a
        bb = sse * self.inv_temper_temp + self.priors.b
        return invgamma.rvs(aa, scale = bb)

    def initialize_sampler(self, ns):
        self.samples = SamplesPCA(ns, self.d)
        self.samples.eta[0] = 0
        self.samples.sigma2[0] = invgamma.rvs(self.priors.a, scale = self.priors.b)
        self.curr_iter = 0
        return

    def iter_sample(self, sse):
        sigma2 = self.curr_sigma2
        eta    = self.curr_eta
        theta  = self.curr_phi
        exptp  = self.experiment.tuple

        self.curr_iter += 1

        #ipdb.set_trace()
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, theta, sigma2)
        
        #sse = sse_pca(exptp, phi, None, None, SubStatePCA(self.curr_eta, self.curr_zeta, sigma2))
        self.samples.sigma2[self.curr_iter] = self.sample_sigma2(sse)
        return

    def write_to_disk(self, cursor, prefix, nburn, thin):
        """ Write Subchain samples to disk """
        sigma2_create = self.create_stmt.format('{}_sigma2'.format(prefix), 'sigma2 REAL')
        sigma2_insert = self.insert_stmt.format('{}_sigma2'.format(prefix), 'sigma2', '?')
        cursor.execute(sigma2_create)
        cursor.executemany(sigma2_insert, [(x,) for x in self.samples.sigma2[nburn::thin].tolist()])
        return

    def __init__(self, parent, experiment, index):
        self.experiment = experiment
        self.parent = parent
        self.index = index
        self.check_constraints = self.experiment.check_constraints
        self.eta_cols = self.experiment.eta_cols
        self.d = len(self.eta_cols)
        #ipdb.set_trace()
        self.priors = PriorsPCA(0.1, 0.1)
        self.bounds = self.experiment.bounds
        pass

class SamplesWPCA(object):
    pass

class SubChainWPCA(object):
    pass

SubChain = {
    'shpb' : SubChainSHPB,
    'pca'  : SubChainPCA,
    'wpca' : SubChainWPCA,
    }

# Defining Chain
#  The Chain object defines the sampler; instantiates observed data and assigns it to the subchains.
#  at the end of a model run, it writes samples to disk.

PriorsChain = namedtuple('PriorsChain', 'psi nu mu Sinv eta_a eta_b')
StateChain = namedtuple('StateChain', 'alpha theta0 Sigma delta thetas substates')

class ChainSamples(object):
    """ Sample object, where chain samples are stored. """
    theta = None
    delta = None
    Sigma = None
    theta0 = None
    alpha = None
    accepted = None

    def __init__(self, N, d, ns):
        self.theta = []
        self.delta = np.empty((ns + 1, N), dtype = np.int)
        self.Sigma = np.empty((ns + 1, d, d))
        self.theta0 = np.empty((ns + 1, d))
        self.alpha = np.empty(ns + 1)
        self.accepted = np.empty(ns + 1)
        return

class Chain(Transformer, pt.PTChain):
    """ Chain object.  This object implements the parallel tempering  """
    samples = None
    N = None
    # Configuration regarding localcov:
    r0 = 1.     # base radius
    nu = 50     # prior degrees of freedom
    psi0 = 1e-4 # prior diagonal magnitude

    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter].copy()

    @property
    def curr_theta0(self):
        return self.samples.theta0[self.curr_iter].copy()

    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()

    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_thetas(self):
        return (self.samples.theta[self.curr_iter]).copy()

    @property
    def curr_substates(self):
        return tuple(subchain.get_substate() for subchain in self.subchains)

    @staticmethod
    def clean_delta_theta(deltas, thetas, i):
        """ drops index i from the delta list.  If index i was a singleton (only obsv in cluster),
        then shift all deltas higher down by 1.  keep doing this until there's no empty clusters
        in the stack. """
        assert(deltas.max() + 1 == thetas.shape[0])
        _delta = np.delete(deltas, i)
        _theta = thetas[np.array([j for j in range(_delta.max() + 1) if j in set(_delta)])]
        nj     = np.array([(_delta == j).sum() for j in range(_delta.max() + 2)])
        fz     = np.where(nj == 0)[0][0]
        while fz <= _delta.max():
            _delta[_delta > fz] = _delta[_delta > fz] - 1
            nj = np.array([(_delta == j).sum() for j in range(_delta.max() + 2)])
            fz = np.where(nj == 0)[0][0]
        return _delta, _theta

    def sample_delta_i(self, deltas, thetas, theta0, Sigma, alpha, i):
        """ Sample cluster membership index """
        assert(deltas.max() + 1 == thetas.shape[0])
        # Check cluster membership without observation i
        _delta, _theta = self.clean_delta_theta(deltas, thetas, i)
        _dmax  = _delta.max()
        # Compute number of observations within each cluster
        nj     = np.array([(_delta == j).sum() for j in range(_dmax + 1 + self.m)])
        # Compute prior probability of membership in each cluster (nj or alpha / m)
        lj     = nj + (nj == 0) * alpha / self.m
        # Generate new thetas for m possible new clusters
        th_new = self.sample_theta_new(theta0, Sigma, self.m)
        th_stack = np.vstack((_theta, th_new))
        phi_stack   = self.unnormalize(self.invprobit(th_stack))
        assert(th_stack.shape[0] == lj.shape[0])
        # compute SSE's under each cluster (extant or non-extant)
        args = zip(
                repeat(self.subchains[i].experiment.tuple),
                phi_stack.tolist(),
                repeat(self.constants_vec),
                repeat(self.model_args),
                repeat(self.subchains[i].get_substate())
                )
        sses   = np.array(list(self.pool.map(sse_wrapper, args)))
        # sses   = np.array(list(map(sse_wrapper, args)))
        substate = self.subchains[i].get_substate()
        # Get log-posterior given
        lps    = np.array([self.subchains[i].log_posterior_theta(sse, substate) for sse in sses])
        lps[np.isnan(lps)] = - np.inf
        lps -= lps.max() # normalizing the log-posteriors so at least some of the sse's
                                 # will produce non-zero posteriors when exponentiated.
        unnormalized = np.exp(lps) * lj
        try:
            normalized = unnormalized / unnormalized.sum()
            dnew   = choice(range(_dmax + self.m + 1), 1, p = normalized)
        except (ValueError, FloatingPointError):
            # nan_idx = np.where(np.isnan(unnormalized))[0][0]
            # print('nan idx: {}'.format(nan_idx))
            print('unnormalized: {}'.format(unnormalized))
            print('sses: {}'.format(sses))
            print('lps: {}'.format(lps))
            print('lj: {}'.format(lj))
            print('nj: {}'.format(nj))
            self.fail_state = {
                'thetas'       : thetas,
                'deltas'       : deltas,
                'theta0'       : theta0,
                'Sigma'        : Sigma,
                'alpha'        : alpha,
                'i'            : i,
                'th_stack'     : th_stack,
                'phi_stack'    : phi_stack,
                'sses'         : sses,
                'lps'          : lps,
                'nj'           : nj,
                'lj'           : lj,
                'unnormalized' : unnormalized,
                }
            raise
        if dnew > _dmax:
            theta = np.vstack((_theta, th_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            theta = _theta
        return delta, theta

    def log_posterior_theta_j(self, theta_j, cluster_j, theta0, SigInv):
        phi_j = self.unnormalize(self.invprobit(theta_j))
        clust_exp_tuples = [self.subchains[i].experiment.tuple for i in cluster_j]
        clust_substates = [self.subchains[i].get_substate() for i in cluster_j]
        args  = zip(
                clust_exp_tuples,           repeat(phi_j),
                repeat(self.constants_vec), repeat(self.model_args),
                clust_substates,
                )
        sses  = np.array(list(self.pool.map(sse_wrapper, args)))
        # sses   = np.array(list(map(sse_wrapper, args)))
        lliks = np.array([
                    self.subchains[i].log_posterior_theta(sse, self.subchains[i].get_substate())
                    for i, sse in zip(cluster_j, sses)
                    ])
        llik  = lliks.sum()
        lpri  = - 0.5 * multi_dot((theta_j - theta0, SigInv, theta_j - theta0)) * self.inv_temper_temp
        ldj   = self.invprobitlogjac(theta_j)
        return llik + lpri + ldj

    def sample_theta_j(self, curr_theta_j, cluster_j, theta0, SigInv):
        """
        cluster_j is an array of indices on which cluster j holds sway.
        I.e., curr_theta_j is the current value of theta for cluster j.
        """
        curr_cov = self.localcov(curr_theta_j)
        gen = mvnormal(mean = curr_theta_j, cov = curr_cov)
        prop_theta_j = gen.rvs()

        if not self.check_constraints(prop_theta_j):
            return curr_theta_j, False

        prop_cov = self.localcov(prop_theta_j)

        curr_lp = self.log_posterior_theta_j(curr_theta_j, cluster_j, theta0, SigInv)
        prop_lp = self.log_posterior_theta_j(prop_theta_j, cluster_j, theta0, SigInv)

        cp_ld   = gen.logpdf(prop_theta_j)
        pc_ld   = mvnormal(mean = prop_theta_j, cov = prop_cov).logpdf(curr_theta_j)

        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld
        if log(uniform.rvs()) < log_alpha:
            return prop_theta_j, True
        else:
            return curr_theta_j, False

    def sample_thetas(self, thetas, delta, theta0, Sigma):
        assert (thetas.shape[0] == delta.max() + 1)
        theta_new = np.empty((thetas.shape))
        SigInv = self.pd_matrix_inversion(Sigma)
        accept = np.empty(theta_new.shape[0])
        for j in range(theta_new.shape[0]):
            cluster_j = np.where(delta == j)[0]
            res = self.sample_theta_j(thetas[j], cluster_j, theta0, SigInv)
            assert type((res)) is tuple
            try:
                theta_new[j], accept[j] = res
            except ValueError:
                print(res)
                raise
        return theta_new, accept.mean()

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
        gen = mvnormal(mean = M0, cov = S0)
        theta0_try = gen.rvs()
        while not self.check_constraints(theta0_try):
            theta0_try = gen.rvs()
        return theta0_try

    def sample_theta_new(self, theta0, Sigma, ns):
        theta_new = np.empty((ns, self.d))
        gen = mvnormal(mean = theta0, cov = Sigma)
        for i in range(ns):
            theta_try = gen.rvs()
            while not self.check_constraints(theta_try):
                theta_try = gen.rvs()
            theta_new[i] = theta_try
        return theta_new

    def sample_Sigma(self, theta, theta0):
        N = theta.shape[0]
        diff = theta - theta0
        C = np.array([np.outer(diff[i], diff[i]) for i in range(N)]).sum(axis = 0)
        psi0 = self.priors.psi + C * self.inv_temper_temp
        nu0  = self.priors.nu  + N * self.inv_temper_temp
        return invwishart.rvs(df = nu0, scale = psi0)

    def sample_alpha(self, curr_alpha, delta):
        nclust = delta.max() + 1 # one-off error of array indexing... # of unique is max + 1
        g = beta.rvs(curr_alpha + 1, self.N)
        aa = self.priors.eta_a + nclust
        bb = self.priors.eta_b - log(g)
        eps = (aa - 1) / (self.N * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(aaa, bb)

    def sample_subchains(self, thetas, deltas):
        exp_tuples = [self.subchains[i].experiment.tuple for i in range(self.N)]
        substates  = [self.subchains[i].get_substate() for i in range(self.N)]
        phis = self.unnormalize(self.invprobit(thetas[deltas]))
        args = zip(exp_tuples, phis.tolist(), repeat(self.constants_vec),
                    repeat(self.model_args), substates)
        sses = np.array(list(self.pool.map(sse_wrapper, args)))
        # sses = np.array(list(map(sse_wrapper, args)))
        for sse, subchain in zip(sses, self.subchains):
            subchain.iter_sample(sse)
        # for i in range(self.N):
        #     self.subchains[i].iter_sample(sses[i])
        return

    def iter_sample(self):
        # Set the current values for all parameters
        theta0 = self.curr_theta0
        Sigma  = self.curr_Sigma
        thetas = self.curr_thetas
        deltas = self.curr_delta
        alpha  = self.curr_alpha

        # Advance the sampler 1 step
        self.curr_iter += 1

        # Sample new cluster assignments given current values of theta
        # (admittedly this won't mix all that well)
        assert(deltas.max() + 1 == thetas.shape[0])
        for i in range(self.N):
            deltas, thetas = self.sample_delta_i(deltas, thetas, theta0, Sigma, alpha, i)

        # Given the new cluster assignments, sample new values for theta
        # and the rest of the parameters
        self.samples.delta[self.curr_iter] = deltas
        new_thetas, accepted = self.sample_thetas(thetas, self.curr_delta, theta0, Sigma)
        self.samples.theta.append(new_thetas)
        self.curr_theta_stack = np.vstack(self.samples.theta)
        self.samples.accepted[self.curr_iter] = accepted
        self.samples.theta0[self.curr_iter] = self.sample_theta0(thetas, Sigma)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_thetas, self.curr_theta0)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(alpha, self.curr_delta)
        # Sample the error terms (and whatever other experiment-specific parameters there are)
        self.sample_subchains(thetas, deltas)
        return

    def sample_n(self, k):
        for _ in range(k):
            self.iter_sample()

    def sample(self, ns):
        self.initialize_sampler(ns)
        self.sample_k(ns)
        return

    def check_constraints(self, theta):
        self.model.update_parameters(self.unnormalize(self.invprobit(theta)))
        return self.model.check_constraints()

    def localcov(self, target):
        return localcov(self.curr_theta_stack, target, self.radius, self.nu, self.psi0)

    def initialize_sampler(self, ns):
        self.samples = ChainSamples(self.N, self.d, ns)
        self.samples.delta[0] = range(self.N)
        theta_start = np.empty((self.N, self.d))
        init_normal = normal(0, scale = 0.2)
        for i in range(self.N):
            theta_try = init_normal.rvs(size = self.d)
            # fail = 0
            while not self.check_constraints(theta_try):
                theta_try = init_normal.rvs(size = self.d)
                # fail += 1
                # print('\rfail: {}'.format(fail), end = '')
            theta_start[i] = theta_try
            self.subchains[i].initialize_sampler(ns)

        self.samples.theta.append(theta_start)
        self.curr_theta_stack = np.vstack(self.samples.theta)
        theta_try = init_normal.rvs(size = self.d)
        while not self.check_constraints(theta_try):
            theta_try = init_normal.rvs(size = self.d)
        self.samples.theta0[0] = theta_try
        self.samples.Sigma[0] = self.sample_Sigma(self.samples.theta0[0], self.samples.theta[0])
        self.samples.alpha[0] = 5.
        self.curr_iter = 0
        self.samples.accepted[0] = 0.
        return

    def set_temperature(self, temperature):
        self.temper_temp = temperature
        self.inv_temper_temp = 1. / temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        for subchain in self.subchains:
            subchain.set_temperature(temperature)
        return

    def get_state(self):
        """ Return current "State" of the Chain """
        state = StateChain(self.curr_alpha, self.curr_theta0, self.curr_Sigma,
                            self.curr_delta, self.curr_thetas, self.curr_substates)
        return state

    def set_state(self, state):
        """ Set current "State" of the Chain """
        self.samples.alpha[self.curr_iter]  = state.alpha
        self.samples.theta0[self.curr_iter] = state.theta0
        self.samples.theta[self.curr_iter]  = state.thetas
        self.samples.delta[self.curr_iter]  = state.delta
        self.samples.Sigma[self.curr_iter]  = state.Sigma
        for substate, subchain in zip(state.substates, self.subchains):
            subchain.set_substate(substate)
        return

    def log_posterior_state(self, state):
        """ For a supplied chain state, calculate the log-posterior. """
        SigInv = self.pd_matrix_inversion(state.Sigma)
        # Set up the arguments to sse_wrapper
        exp_tuples = [subchain.experiment.tuple for subchain in self.subchains]
        substates  = [subchain.get_substate() for subchain in self.subchains]
        thetas = state.thetas[state.delta]
        phis   = self.unnormalize(self.invprobit(thetas))
        args   = zip(exp_tuples, phis, repeat(self.constants_vec),
                        repeat(self.model_args), substates)
        # farm calculating sse's out to the pool
        sses   = np.array(list(self.pool.map(sse_wrapper, args)))
        # sses   = np.array(list(map(sse_wrapper, args)))
        # calculate log-posteriors given calculated sse's, substates
        lpss   = np.array([
            subchain.log_posterior_substate(sse, substate)
            for sse, subchain, substate in zip(sses, self.subchains, state.substates)
            ]).sum()
        # Log determinant of the jacobian for each theta
        ldjs   = sum([self.invprobitlogjac(state.thetas[i]) for i in range(state.thetas.shape[0])])
        # Log-posterior for theta (against theta0, that part that isn't covered by the subchains)
        thdiff = state.thetas - state.theta0
        lpts   = - 0.5 * self.inv_temper_temp * sum([
                thdiff[i] @ SigInv @ thdiff[i]
                for i in range(thdiff.shape[0])
                ])
        # Log posterior for theta0 (against mu, that part that isn't covered by lpts)
        t0diff = state.theta0 - self.priors.mu
        lpth0  = - 0.5 * (t0diff @ self.priors.Sinv @ t0diff)
        # Log Posterior for Sigma (that part that isn't covered by lpts)
        lpSig  = (
            - 0.5 * (self.N * self.inv_temper_temp + self.priors.nu + self.d + 1) * slogdet(state.Sigma)[1]
            - 0.5 * (self.priors.psi @ SigInv).trace()
            )
        # Log-posterior is the sum of these.  There's also alpha, but I haven't figured out how
        # to include that.
        return lpss + lpts + ldjs + lpth0 + lpSig

    def get_accept_probability(self, nburn = 0):
        return self.samples.accepted[nburn:].mean()

    def write_to_disk(self, path, nburn = 0, thin = 1):
        if os.path.exists(path):
            os.remove(path)

        conn   = sql.connect(path)
        curs   = conn.cursor()

        # strip and thin the resulting samples.  Add
        thetas = np.vstack([
            np.vstack((np.ones(theta.shape[0]) * i, list(range(theta.shape[0])), theta.T)).T
            for i, theta in enumerate(self.samples.theta[nburn::thin])
            ])
        # thetas_df = pd.DataFrame(thetas, cols =)
        phis   = np.vstack((thetas[:,:2].T,self.unnormalize(self.invprobit(thetas[:,2:])).T)).T
        deltas = self.samples.delta[nburn::thin]
        Sigma  = self.samples.Sigma[nburn::thin]
        theta0 = self.samples.theta0[nburn::thin]
        phi0   = self.unnormalize(self.invprobit(theta0))
        alpha  = self.samples.alpha[nburn::thin]
        models = list(self.model.report_models_used().items())
        constants = list(self.model.report_constants().items())

        # create_stmt = """ CREATE TABLE {}({}); """
        # insert_stmt = """ INSERT INTO {}({}) values ({}); """

        models_create = self.create_stmt.format('models', 'model_type TEXT, model_name TEXT')
        models_insert = self.insert_stmt.format('models', 'model_type, model_name', '?,?')
        curs.execute(models_create)
        curs.executemany(models_insert, models)

        param_create_list = [x + ' REAL' for x in self.parameter_list]
        thetas_create = self.create_stmt.format(
                'thetas', ','.join(['iteration INT', 'cluster INT'] +  param_create_list)
                )
        thetas_insert = self.insert_stmt.format(
                'thetas', ','.join(['iteration', 'cluster'] + self.parameter_list),
                ','.join(['?'] * (2 + self.d))
                )
        curs.execute(thetas_create)
        curs.executemany(thetas_insert, thetas.tolist())

        phis_create = self.create_stmt.format(
                'phis', ','.join(['iteration INT', 'cluster INT'] + param_create_list),
                )
        phis_insert = self.insert_stmt.format(
                'phis', ','.join(['iteration', 'cluster']  + self.parameter_list),
                ','.join(['?'] * (2 + self.d))
                )
        curs.execute(phis_create)
        curs.executemany(phis_insert, phis.tolist())

        theta0_create = self.create_stmt.format('theta0',','.join(param_create_list))
        theta0_insert = self.insert_stmt.format(
                'theta0', ','.join(self.parameter_list), ','.join(['?'] * self.d)
                )
        curs.execute(theta0_create)
        curs.executemany(theta0_insert, theta0.tolist())

        phi0_create = self.create_stmt.format('phi0', ','.join(param_create_list))
        phi0_insert = self.insert_stmt.format(
                'phi0', ','.join(self.parameter_list), ','.join(['?'] * self.d)
                )
        curs.execute(phi0_create)
        curs.executemany(phi0_insert, phi0.tolist())

        delta_list = ['delta_{:03d}'.format(i) for i in range(1, self.N + 1)]
        deltas_create = self.create_stmt.format('delta', ','.join([x + ' INT' for x in delta_list]))
        deltas_insert = self.insert_stmt.format('delta', ','.join(delta_list), ','.join(['?'] * self.N))
        curs.execute(deltas_create)
        curs.executemany(deltas_insert, deltas.tolist())

        meta_create = self.create_stmt.format('meta', 'table_name TEXT, cluster_id TEXT, prefix TEXT')
        meta_insert = self.insert_stmt.format('meta', 'table_name, cluster_id, prefix', '?,?,?')
        curs.execute(meta_create)
        meta_list = [
            (subchain.table_name, delta_id, prefix)
            for subchain, delta_id, prefix in zip(self.subchains, delta_list, self.subchain_prefix_list)
            ]
        curs.executemany(meta_insert, meta_list)

        Sigma_cols = [
            'Sigma_{}_{}'.format(i,j)
            for i in range(1, self.d + 1)
            for j in range(1, self.d + 1)
            ]
        Sigma_create = self.create_stmt.format('Sigma',','.join([x + ' REAL' for x in Sigma_cols]))
        Sigma_insert = self.insert_stmt.format('Sigma',','.join(Sigma_cols), ','.join(['?'] * self.d * self.d))
        curs.execute(Sigma_create)
        curs.executemany(Sigma_insert, Sigma.reshape(Sigma.shape[0], -1).tolist())

        constant_create = self.create_stmt.format('constants','constant TEXT, value REAL')
        constant_insert = self.insert_stmt.format('constants','constant, value', '?,?')
        curs.execute(constant_create)
        curs.executemany(constant_insert, constants)

        alpha_create = self.create_stmt.format('alpha', 'alpha REAL, nclust INT')
        alpha_insert = self.insert_stmt.format('alpha', 'alpha, nclust', '?,?')
        curs.execute(alpha_create)
        curs.executemany(alpha_insert, np.vstack((alpha, deltas.max(axis = 1) + 1)).T.tolist())

        bounds_create = self.create_stmt.format('bounds', 'parameter TEXT, lower REAL, upper REAL')
        bounds_insert = self.insert_stmt.format('bounds', 'parameter, lower, upper', '?,?,?')
        curs.execute(bounds_create)
        bounds = [
            (param, bound[0],bound[1])
            for param, bound in zip(self.parameter_list, self.bounds)
            ]
        curs.executemany(bounds_insert, bounds)

        for prefix, subchain in zip(self.subchain_prefix_list, self.subchains):
            subchain.write_to_disk(curs, prefix, nburn, thin)

        conn.commit()
        conn.close()
        return

    def complete(self):
        self.pool.close()
        return

    def __init__(self, path, bounds, constants, model_args, temperature = 1., m = 20):
        conn = sql.connect(path)
        cursor = conn.cursor()
        self.model = MaterialModel(**model_args)
        self.model_args = model_args
        self.parameter_list = self.model.get_parameter_list()
        self.constant_list = self.model.get_constant_list()
        self.bounds = np.array([bounds[key] for key in self.parameter_list])
        self.constants_vec = np.array([constants[key] for key in self.constant_list])
        self.model.initialize_constants(self.constants_vec)
        tables = list(cursor.execute(" SELECT type, table_name FROM meta; "))
        tables = [(i, table[0], table[1]) for i, table in enumerate(tables)]
        self.subchains = [
            SubChain[type](self, Experiment[type](conn, table_name, model_args), i)
            for i, type, table_name in tables
            ]
        self.set_temperature(temperature)
        self.N = len(self.subchains)
        self.d = len(self.parameter_list)
        self.subchain_prefix_list = ['subchain_{}'.format(i) for i in range(self.N)]
        self.pool = Pool(processes = POOL_SIZE)
        self.priors = PriorsChain(
            psi = np.eye(self.d) * 0.5,
            nu = self.d + 2,
            mu = np.zeros(self.d),
            Sinv = np.eye(self.d) * 1e-6,
            eta_a = 2.,
            eta_b = 5.,
            )
        self.m = m
        return

class ResultBase(object):
    def plot_calibrated(self, path):
        pass

    def load_data(self, cursor, prefix):
        pass

    def __init__(self, cursor, prefix, samples, experiment, constants, model_args, pool):
        """
        experiment       : model experiment (As defined by experiment.py)
        cluster_samples  : samples of phi (theta in real space) relevant to observation's cluster
        hier_samples     : samples of phi0 (theta0 in real space)
        subchain_samples : samples of relevant items from subchain
        constants        : vector of constants that were supplied when model was calibrated
        model_args       : dictionary of physical models used when model was calibrated
        """
        self.samples          = samples
        self.pool             = pool # expose multiprocessing.pool, only build once.
        self.experiment       = experiment
        self.constants        = constants
        self.model_args       = model_args
        self.load_data(cursor, prefix)
        return

class ResultSHPB(ResultBase):
    # def load_data(self, cursor, prefix):
    #     sigma2 = np.array(
    #         [x[0] for x in cursor.execute(' SELECT * FROM {}_sigma2; '.format(prefix))]
    #         )
    #     # self.samples = Bunch(
    #     #     cluster_phi = self.parent.samples.
    #     #     sigma2 = sigma2,
    #     # )
    #     return

    def calibrated_curve_summary(self, parameters, predictive = False, quantiles = (0.05, 0.5, 0.95)):
        # This should be fixed such as to provide the substate.  Still, not provided yet.
        args = zip(repeat(self.experiment.tuple), parameters.tolist(),
                    repeat(self.constants), repeat(self.model_args), repeat(None))
        res = self.pool.map(predict_wrapper, args)
        strain = res[0][0]
        stress = np.vstack([r[1] for r in res])
        stress_summary = np.quantile(stress, quantiles, axis = 0)
        return strain, stress_summary

    def plot_calibrated(self, path, smax = None, predictive = False):
        ul_alpha = 0.6; m_alpha = 0.95; linewidth = 0.4

        strain, stress_cluster_summary = self.calibrated_curve_summary(self.samples.phi, predictive)
        strain, stress_hier_summary    = self.calibrated_curve_summary(self.samples.phi0, predictive)

        ul_line_args = {'alpha' : 0.60, 'linestyle' : '-', 'linewidth' : 0.4}
        me_line_args = {'alpha' : 0.95, 'linestyle' : '-', 'linewidth' : 0.4}

        plt.plot(strain, stress_cluster_summary[0], color = 'blue', **ul_line_args)
        plt.plot(strain, stress_cluster_summary[1], color = 'blue', **me_line_args)
        plt.plot(strain, stress_cluster_summary[2], color = 'blue', **ul_line_args)

        plt.plot(strain, stress_hier_summary[0], color = 'green', **ul_line_args)
        plt.plot(strain, stress_hier_summary[1], color = 'green', **me_line_args)
        plt.plot(strain, stress_hier_summary[2], color = 'green', **ul_line_args)

        plt.plot(self.experiment.X, self.experiment.Y, 'bo')
        if smax:
            pass
        else:
            smax = max(
                stress_cluster_summary.max(),
                stress_hier_summary.max(),
                self.experiment.Y.max()
                )
        plt.ylim((0., smax))
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path)
        plt.clf()
        return

Result = {
    'shpb' : ResultSHPB,
    }

class ResultSummaryBase(Transformer):
    def plot_pairwise(self, path, title = None, figsize = (5,6), dpi = 300):
        def off_diag(x, y, **kwargs):
            plt.scatter(x, y, **kwargs)
            ax = plt.gca()
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            return

        def on_diag(x, **kwargs):
            sea.distplot(x, bins = 20, kde = False, rug = True, **kwargs)
            ax = plt.gca()
            ax.set_xlim(0,1)
            return

        def uniquify(df_columns):
            seen = set()
            for item in df_columns:
                fudge = 1
                newitem = item

                while newitem in seen:
                    fudge += 1
                    newitem = "{}_{}".format(item, fudge)

                yield newitem
                seen.add(newitem)

        d = sample_parameters.shape[1]
        df = pd.DataFrame(sample_parameters, columns = self.parameter_order)
        df.columns = list(uniquify(df.columns))
        g = sea.PairGrid(df)
        g = g.map_offdiag(off_diag, s = 1)
        g = g.map_ondiag(on_diag)

        for i in range(d):
            g.axes[i,i].annotate(self.parameter_order[i], xy = (0.05, 0.9))
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')

        plt.savefig(path)
        return

    def plot_calibrated(self, basepath):
        for result in self.results:
            result.plot_calibrated(
                os.path.join(basepath, '{}.png'.format(result.experiment.table_name))
                )
        return

    def __init__(self, path_inputs, path_results):
        pass

class ResultSummary(ResultSummaryBase):
    def plot_clustplot(self, path, ordering = None, title = None, figsize = (5,6), dpi = 300):
        delta = self.samples.delta
        nsamp, ndat = delta.shape
        clust_mat_sum = np.zeros((ndat, ndat))
        for i in range(nsamp):
            clust_mat = np.zeros((ndat, delta[i].max() + 1))
            for j in range(ndat):
                clust_mat[j, delta[i,j]] = 1
            clust_mat_sum += clust_mat @ clust_mat.T
        if ordering:
            clust_mat_sum = clust_mat_sum.T[ordering].T[ordering]
        clust_mat_prb = clust_mat_sum / delta.shape[0]
        fig = plt.figure(figsize = figsize)
        plt.matshow(clust_mat_prb)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.savefig(path, dpi = dpi)
        plt.close()
        return

    def cluster_by_strainrate(self, basepath):
        edots = [experiment.edot for experiment in self.experiments]
        idx = list(range(len(edots)))
        ordering = [x for _, x in sorted(zip(edots, idx))]
        self.plot_clustplot(os.path.join(basepath, 'cluster_strain.png'), ordering, 'Strain Rate')
        return

    def cluster_by_temperature(self, basepath):
        temps = [experiment.temp for experiment in self.experiments]
        idx = list(range(len(temps)))
        ordering = [x for _, x in sorted(zip(temps, idx))]
        self.plot_clustplot(os.path.join(basepath, 'cluster_temp.png'), ordering, 'Temperature')
        return

    def __init__(self, path_inputs, path_results):
        self.pool = Pool(POOL_SIZE)

        iconn = sql.connect(path_inputs)
        oconn = sql.connect(path_results)
        icursor = iconn.cursor()
        ocursor = oconn.cursor()

        tables = list(icursor.execute(' SELECT type, table_name FROM meta; '))
        self.models = dict(ocursor.execute(' select * FROM models; '))
        constants = dict(ocursor.execute(' SELECT * FROM constants; '))
        self.experiments = [
            Experiment[type](icursor, table_name, self.models)
            for type, table_name in tables
            ]
        constant_list = self.experiments[0].model.get_constant_list()
        self.constant_vec = np.array([constants[key] for key in constant_list])
        for experiment in self.experiments:
            experiment.model.initialize_constants(self.constant_vec)

        meta   = list(ocursor.execute(' SELECT table_name, cluster_id, prefix FROM meta; '))
        temp   = np.array(list(ocursor.execute(' SELECT alpha, nclust FROM alpha; ')))
        alpha  = temp[:,0]
        nclust = temp[:,1]
        delta  = np.array(list(ocursor.execute(' SELECT * FROM delta; ')))
        N      = delta.shape[1]
        constants = dict(list(ocursor.execute(' SELECT constant, value FROM constants; ')))
        theta0 = np.array(list(ocursor.execute(' SELECT * FROM theta0; ')))
        phi0   = np.array(list(ocursor.execute(' SELECT * FROM phi0; ')))
        ns, d  = theta0.shape
        Sigma  = np.array(list(ocursor.execute(' SELECT * FROM Sigma; '))).reshape(ns,d,d)
        temp   = np.array(list(ocursor.execute(' SELECT * FROM thetas; ')))
        thetas = [temp[np.where(temp.T[0] == i)[0], 2:] for i in range(ns)]
        temp   = np.array(list(ocursor.execute(' SELECT * FROM phis; ')))
        phis   = [temp[np.where(temp.T[0] == i)[0], 2:] for i in range(ns)]

        self.samples = Bunch(
            alpha  = alpha,  delta = delta,
            theta0 = theta0, phi0  = phi0,
            thetas = thetas, phis  = phis,
            Sigma  = Sigma,  meta  = meta,
            )
        theta_by_exp = [np.array(th[d] for th, d in zip(thetas, delta.T[i])) for i in range(N)]
        phi_by_exp   = [np.array(ph[d] for ph, d in zip(phis,   delta.T[i])) for i in range(N)]

        self.results = [
            Result[experiment.tuple.type](
                ocursor,
                None,
                Bunch(theta = theta, phi = phi, theta0 = theta0, phi0 = phi0, Sigma = Sigma),
                experiment,
                self.constant_vec,
                self.models,
                self.pool,
                )
            for experiment, theta, phi in zip(self.experiments, theta_by_exp, phi_by_exp)
            ]
        iconn.close()
        oconn.close()
        return



# EOF
