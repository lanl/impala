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
from math import exp, log
import sqlite3 as sql
import os
# Custom Modules
from experiment import Experiment, Transformer, cholesky_inversion
from physical_models_c import MaterialModel
from pointcloud import localcov
import pt

import sm_dpcluster as smdp


class SubChainHierBase(smdp.SubChainBase, Transformer):
    """ In the hierarchical model, sampling of theta_i (for observation i) has been moved into
    subchain i... therefore the covariance estimation for theta_i must happen in subchain_i """
    # parameters relevant to localcov
    r0 = 1.
    nu = 50
    psi0 = 1e-4

    def set_temperature(self, temperature):
        super().set_temperature(temperature)
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def localcov(self, target):
        # localized covariance matrix
        return localcov(self.samples.theta[:self.curr_iter], target, self.radius, self.nu, self.psi0)

PriorsSHPB = namedtuple('PriorsSHPB','a b')
SubstateSHPB = namedtuple('SubstateSHPB', 'theta sigma2')
class SamplesSHPB(object):
    sigma2 = None
    theta  = None
    accepted = None

    def __init__(self, d, ns):
        self.sigma2   = np.empty(ns + 1)
        self.theta    = np.empty((ns + 1, d))
        self.accepted = np.empty(ns + 1)
        return

class SubChainSHPB(SubChainHierBase):
    samples = None
    experiment = None
    N = None

    @property
    def curr_sigma2(self):
        return self.samples.sigma2[self.curr_iter].copy()

    @property
    def curr_theta(self):
        return self.samples.theta[self.curr_iter].copy()

    @property
    def curr_accepted(self):
        return self.samples.accepted[self.curr_iter].copy()

    def set_substate(self, substate):
        self.samples.sigma2[self.curr_iter] = substate.sigma2
        self.samples.theta[self.curr_iter] = substate.theta

    def sample_sigma2(self, sse):
        aa = self.N * self.inv_temper_temp + self.priors.a
        bb = sse * self.inv_temper_temp + self.priors.b
        return invgamma.rvs(aa, scale = bb)

    def log_posterior_theta(self, theta, sigma2, theta0, SigInv):
        phi = self.unnormalize(self.invprobit(theta))
        sse = smdp.sse_shpb(self.experiment.tuple, phi, self.constant_vec, self.model_args, self.get_substate())
        # (scaled) sum squared error
        ssse = sse / sigma2
        sssd = (theta - theta0).T @ SigInv @ (theta - theta0)
        ldj  = self.invprobitlogjac(theta)
        lp = (
            - 0.5 * self.inv_temper_temp * ssse
            - 0.5 * self.inv_temper_temp * sssd
            + ldj
            )
        return lp

    def log_posterior_substate(self, theta0, SigInv, substate):
        # Log posterior for theta_i -- includes the difference from prior theta0..
        lpt = self.log_posterior_theta(substate.theta, substate.sigma2, theta0, SigInv)
        # additional contribution from sigma2
        lpp = -((self.N * self.inv_temper_temp + self.priors.a + 1) * log(substate.sigma2)
                + self.priors.b / substate.sigma2)
        return lpt + lpp

    def check_constraints(self, theta):
        phi = self.unnormalize(self.invprobit(theta))
        self.model.update_parameters(phi)
        return self.model.check_constraints()

    def sample_theta(self, curr_theta, sigma2, theta0, SigInv):
        curr_cov   = self.localcov(curr_theta)
        prop_theta = curr_theta + cholesky(curr_cov) @ normal.rvs(size = self.d)
        if not self.check_constraints(prop_theta):
            return curr_theta, False
        prop_cov   = self.localcov(prop_theta)

        curr_logd  = self.log_posterior_theta(curr_theta, sigma2, theta0, SigInv)
        prop_logd  = self.log_posterior_theta(prop_theta, sigma2, theta0, SigInv)

        cp_ld = mvnormal(mean = curr_theta, cov = curr_cov).logpdf(prop_theta)
        pc_ld = mvnormal(mean = prop_theta, cov = prop_cov).logpdf(curr_theta)

        log_alpha = prop_logd + pc_ld - curr_logd - cp_ld
        if log(uniform.rvs()) < log_alpha:
            return prop_theta, True
        else:
            return curr_theta, False

    def iter_sample(self, theta0, SigInv):
        sigma2 = self.curr_sigma2
        theta  = self.curr_theta
        self.curr_iter += 1

        theta_new, accepted = self.sample_theta(theta, sigma2, theta0, SigInv)
        phi = self.unnormalize(self.invprobit(theta_new))
        sse = smdp.sse_shpb(self.experiment.tuple, phi, self.constant_vec, self.model_args, self.get_substate())
        self.samples.theta[self.curr_iter] = theta_new
        self.samples.accepted[self.curr_iter] = accepted
        self.samples.sigma2[self.curr_iter] = self.sample_sigma2(sse)
        return

    def get_substate(self):
        return SubstateSHPB(self.curr_theta, self.curr_sigma2)

    def set_substate(self, substate):
        self.samples.theta[self.curr_iter] = substate.theta
        self.samples.sigma2[self.curr_iter] = substate.sigma2
        return

    def initialize_sampler(self, ns):
        self.curr_iter = 0
        self.samples = SamplesSHPB(self.d, ns)
        gen = normal(scale = 0.2)
        theta_try = gen.rvs(size = self.d)
        while not self.check_constraints(theta_try):
            theta_try = gen.rvs(size = self.d)
        self.samples.theta[0] = theta_try
        self.samples.sigma2[0] = invgamma(self.priors.a, scale = self.priors.b).rvs()
        self.samples.accepted[0] = 0
        return

    def write_to_disk(self, cursor, prefix, nburn, thin):
        sigma2_create = self.create_stmt.format('{}_sigma2'.format(prefix), 'sigma2 REAL')
        sigma2_insert = self.insert_stmt.format('{}_sigma2'.format(prefix), 'sigma2', '?')
        cursor.execute(sigma2_create)
        cursor.executemany(sigma2_insert, [(x,) for x in self.samples.sigma2[nburn::thin].tolist()])

        theta  = self.samples.theta[nburn::thin]
        phi    = self.unnormalize(self.invprobit(theta))

        param_create_list = ','.join([x + ' REAL' for x in self.parameter_list])
        param_insert_tple = (','.join(self.parameter_list), ','.join(['?'] * self.d))

        theta_create = self.create_stmt.format('{}_theta'.format(prefix), param_create_list)
        theta_insert = self.insert_stmt.format('{}_theta'.format(prefix), *param_insert_tple)
        phi_create   = self.create_stmt.format('{}_phi'.format(prefix), param_create_list)
        phi_insert   = self.insert_stmt.format('{}_phi'.format(prefix), *param_insert_tple)
        cursor.execute(theta_create)
        cursor.executemany(theta_insert, theta.tolist())
        cursor.execute(phi_create)
        cursor.executemany(phi_insert, phi.tolist())
        return

    def __init__(self, parent, experiment, index, constant_vec, bounds):
        self.parent     = parent
        self.experiment = experiment
        self.index      = index
        self.bounds     = bounds
        self.table_name = self.experiment.table_name
        self.priors     = PriorsSHPB(25, 1.e-6)
        self.N          = self.experiment.X.shape[0]
        self.model      = self.experiment.model
        self.model.initialize_constants(constant_vec)
        self.model_args = self.model.report_models_used()
        self.constant_vec   = constant_vec
        self.parameter_list = self.model.get_parameter_list()
        self.d          = len(self.parameter_list)
        return

class SubChainEmulated(SubChainHierBase):
    pass

SubChain = {
    'shpb' : SubChainSHPB,
    }

PriorsChain = namedtuple('PriorsChain', 'psi nu mu Sinv')
StateChain = namedtuple('StateChain','theta0 Sigma SigInv substates')
class ChainSamples(object):
    theta0 = None
    Sigma  = None
    accepted = None

    def __init__(self, d, ns):
        self.theta0 = np.empty((ns + 1, d))
        self.Sigma = np.empty((ns + 1, d, d))
        self.accepted = np.empty(ns + 1)
        self.accepted[0] = 0
        return

class Chain(Transformer, pt.PTChain):
    samples = None
    N = None

    def set_temperature(self, temperature):
        self.inv_temper_temp = 1 / temperature
        self.temper_temp = temperature
        for subchain in self.subchains:
            subchain.set_temperature(temperature)
        return

    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter].copy()

    @property
    def curr_SigInv(self):
        return self.pd_matrix_inversion(self.curr_Sigma)

    @property
    def curr_theta0(self):
        return self.samples.theta0[self.curr_iter].copy()

    @property
    def curr_substates(self):
        return [subchain.get_substate() for subchain in self.subchains]

    @property
    def curr_thetas(self):
        return np.vstack([subchain.curr_theta for subchain in self.subchains])

    def sample_theta0(self, thetas, SigInv):
        theta_bar = thetas.mean(axis = 0)
        N = thetas.shape[0]
        S0 = self.pd_matrix_inversion(N * self.inv_temper_temp * SigInv + self.priors.Sinv)
        M0 = S0.dot(N * self.inv_temper_temp * SigInv.dot(theta_bar) +
                                        self.priors.Sinv.dot(self.priors.mu))
        gen = mvnormal(mean = M0, cov = S0)
        theta0_try = gen.rvs()
        while not self.check_constraints(theta0_try):
            theta0_try = gen.rvs()
        return theta0_try

    def sample_Sigma(self, thetas, theta0):
        diff = thetas - theta0
        C = np.array([np.outer(diff[i], diff[i]) for i in range(self.N)]).sum(axis = 0)
        psi0 = self.priors.psi + C * self.inv_temper_temp
        nu0  = self.priors.nu  + self.N * self.inv_temper_temp
        return invwishart.rvs(df = nu0, scale = psi0)

    def sample_subchains(self, theta0, SigInv):
        for subchain in self.subchains:
            subchain.iter_sample(theta0, SigInv)
        return

    def iter_sample(self):
        theta0, SigInv = self.curr_theta0, self.curr_SigInv
        self.sample_subchains(theta0, SigInv)
        self.curr_iter += 1
        self.samples.theta0[self.curr_iter] = self.sample_theta0(self.curr_thetas, SigInv)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_thetas, self.curr_theta0)
        self.samples.accepted[self.curr_iter] = np.array([
            subchain.curr_accepted
            for subchain in self.subchains
            ]).mean()
        return

    def sample_n(self, n):
        for _ in range(n):
            self.iter_sample()
        return

    def initialize_sampler(self, ns):
        self.curr_iter = 0

        for subchain in self.subchains:
            subchain.initialize_sampler(ns)

        self.samples = ChainSamples(self.d, ns)

        gen = normal(scale = 0.1)
        theta_try = gen.rvs(size = self.d)
        while not self.check_constraints(theta_try):
            theta_try = gen.rvs(size = self.d)

        self.samples.theta0[0] = theta_try
        self.samples.Sigma[0] = self.sample_Sigma(self.curr_thetas, self.curr_theta0)
        return

    def check_constraints(self, theta):
        phi = self.unnormalize(self.invprobit(theta))
        self.model.update_parameters(phi)
        return self.model.check_constraints()

    def log_posterior_state(self, state):
        lps = np.array([
            subchain.log_posterior_substate(state.theta0, state.SigInv, substate)
            for subchain, substate in zip(self.subchains, state.substates)
            ]).sum()
        tdiff = state.theta0 - self.priors.mu
        ldp1 = - 0.5 * tdiff @ self.priors.Sinv @ tdiff
        ldp2 = (
            - 0.5 * (self.N * self.inv_temper_temp + self.priors.nu + self.d + 1) * slogdet(state.Sigma)[1]
            - 0.5 * (self.priors.psi @ state.SigInv).trace()
            )
        return lps + ldp1 + ldp2

    def get_state(self):
        return StateChain(self.curr_theta0, self.curr_Sigma, self.curr_SigInv, self.curr_substates)

    def set_state(self, state):
        self.samples.theta0[self.curr_iter] = state.theta0
        self.samples.Sigma[self.curr_iter] = state.Sigma
        for subchain, substate in zip(self.subchains, state.substates):
            subchain.set_substate(substate)
        return

    def write_to_disk(self, path, nburn, thin):
        nburn += 1
        if os.path.exists(path):
            os.remove(path)

        conn = sql.connect(path)
        cursor = conn.cursor()

        theta0    = self.samples.theta0[nburn::thin]
        phi0      = self.unnormalize(self.invprobit(theta0))
        Sigma     = self.samples.Sigma[nburn::thin]
        models    = list(self.model.report_models_used().items())
        constants = list(self.model.report_constants().items())

        param_create_list = ','.join([x + ' REAL' for x in self.parameter_list])
        param_insert_tple = (','.join(self.parameter_list), ','.join(['?'] * self.d))
        theta0_create = self.create_stmt.format('theta0', param_create_list)
        theta0_insert = self.insert_stmt.format('theta0', *param_insert_tple)
        phi0_create   = self.create_stmt.format('phi0',   param_create_list)
        phi0_insert   = self.insert_stmt.format('phi0',   *param_insert_tple)
        cursor.execute(theta0_create)
        cursor.executemany(theta0_insert, theta0.tolist())
        cursor.execute(phi0_create)
        cursor.executemany(phi0_insert, phi0.tolist())

        models_create = self.create_stmt.format('models', 'model_type TEXT, model_name TEXT')
        models_insert = self.insert_stmt.format('models', 'model_type, model_name', '?,?')
        cursor.execute(models_create)
        cursor.executemany(models_insert, models)

        consts_create = self.create_stmt.format('constants', 'constant REAL, value REAL')
        consts_insert = self.insert_stmt.format('constants', 'constant, value', '?,?')
        cursor.execute(consts_create)
        cursor.executemany(consts_insert, constants)

        Sigma_cols = [
            'Sigma_{}_{}'.format(i,j)
            for i in range(1, self.d + 1)
            for j in range(1, self.d + 1)
            ]
        Sigma_create = self.create_stmt.format('Sigma',','.join([x + ' REAL' for x in Sigma_cols]))
        Sigma_insert = self.insert_stmt.format('Sigma',','.join(Sigma_cols), ','.join(['?'] * self.d * self.d))
        cursor.execute(Sigma_create)
        cursor.executemany(Sigma_insert, Sigma.reshape(Sigma.shape[0], -1).tolist())

        for prefix, subchain in zip(self.subchain_prefix_list, self.subchains):
            subchain.write_to_disk(cursor, prefix, nburn, thin)

        meta_list = list(zip(
            [subchain.experiment.table_name for subchain in self.subchains],
            self.subchain_prefix_list,
            ))
        meta_create = self.create_stmt.format('meta', 'table_name TEXT, prefix TEXT')
        meta_insert = self.insert_stmt.format('meta', 'table_name, prefix', '?,?')
        cursor.execute(meta_create)
        cursor.executemany(meta_insert, meta_list)

        conn.commit()
        conn.close()
        return

    def __init__(self, path, bounds, constants, model_args, temperature = 1.):
        self.model = MaterialModel(**model_args)
        self.model_args = model_args
        self.parameter_list = self.model.get_parameter_list()
        self.constant_list = self.model.get_constant_list()
        self.bounds = np.array([bounds[key] for key in self.parameter_list])
        self.constant_vec = np.array([constants[key] for key in self.constant_list])
        self.model.initialize_constants(self.constant_vec)
        conn = sql.connect(path)
        cursor = conn.cursor()
        tables = list(cursor.execute(' SELECT type, table_name FROM meta;'))
        tables = [(i, table[0], table[1]) for i, table in enumerate(tables)]
        self.subchains = [
            SubChain[type](
                self,
                Experiment[type](cursor, table_name, model_args),
                i,
                self.constant_vec,
                self.bounds,
                )
            for i, type, table_name in tables
            ]
        self.set_temperature(temperature)
        self.N = len(self.subchains)
        self.d = len(self.parameter_list)
        self.subchain_prefix_list = ['subchain_{}'.format(i) for i in range(self.N)]
        self.priors = PriorsChain(
            psi = np.eye(self.d) * 0.5,
            nu = self.d + 2,
            mu = np.zeros(self.d),
            Sinv = np.eye(self.d) * 1e-6,
            )
        return

# EOF
