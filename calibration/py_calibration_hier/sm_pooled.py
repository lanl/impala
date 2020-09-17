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
from sm_dpcluster import SubChain, sse_wrapper

POOL_SIZE = 8

StateChain = namedtuple('StateChain','theta substates')
class ChainSamples(object):
    theta    = None
    accepted = None

    def __init__(self, d, ns):
        self.theta = np.empty((ns + 1, d))
        self.accepted = np.empty(ns + 1)
        return

class Chain(Transformer, pt.PTChain):
    r0 = 1.
    nu = 50
    psi0 = 1e-4

    def set_temperature(self, temperature):
        self.inv_temper_temp = 1. / temperature
        self.temper_temp = temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        for subchain in self.subchains:
            subchain.set_temperature(temperature)
        return

    @property
    def curr_theta(self):
        return self.samples.theta[self.curr_iter].copy()

    @property
    def curr_substates(self):
        return [subchain.get_substate() for subchain in self.subchains]

    def log_posterior_theta(self, theta, substates):
        phi = self.unnormalize(self.invprobit(theta))
        args = zip(self.exp_tuples, repeat(phi), repeat(self.constant_vec), repeat(self.model_args))
        sses = self.pool.map(sse_wrapper, args)
        llks = np.array([
            subchain.log_posterior_theta(sse, substate)
            for sse, substate, subchain in zip(sses, substates, self.subchains)
            ])
        llks[np.isnan(llks)] = - np.inf
        llk = llks.sum()
        ldj = self.invprobitlogjac(theta)
        return llk + ldj

    def log_posterior_state(self, state):
        phi = self.unnormalize(self.invprobit(state.theta))
        args = zip(self.exp_tuples, repeat(phi), repeat(self.constant_vec), repeat(self.model_args))
        sses = self.pool.map(sse_wrapper, args)
        llks = np.array([
            subchain.log_posterior_substate(sse, substate)
            for sse, substate, subchain in zip(sses, state.substates, self.subchains)
            ])
        llks[np.isnan(llks)] = - np.inf
        llk = llks.sum()
        ldj = self.invprobitlogjac(state.theta)
        return llk + ldj

    def sample_theta(self, curr_theta, substates):
        curr_cov   = self.localcov(curr_theta)
        prop_theta = curr_theta + cholesky(curr_cov) @ normal.rvs(size = self.d)
        if not self.check_constraints(prop_theta):
            return curr_theta, False
        prop_cov   = self.localcov(prop_theta)
        # Log posterior under current and proposal
        curr_logd = self.log_posterior_theta(curr_theta, substates)
        prop_logd = self.log_posterior_theta(prop_theta, substates)
        # Log transition Density
        cp_ld = mvnormal(mean = curr_theta, cov = curr_cov).logpdf(prop_theta)
        pc_ld = mvnormal(mean = prop_theta, cov = prop_cov).logpdf(curr_theta)
        # Log Transition Probability
        log_alpha = prop_logd + pc_ld - curr_logd - cp_ld
        if log(uniform.rvs()) < log_alpha:
            return prop_theta, True
        else:
            return curr_theta, False

    def iter_sample(self):
        curr_theta = self.curr_theta
        self.curr_iter += 1

        theta_new, accepted = self.sample_theta(curr_theta, self.curr_substates)
        self.samples.theta[self.curr_iter] = theta_new
        self.samples.accepted[self.curr_iter] = accepted

        phi  = self.unnormalize(self.invprobit(self.curr_theta))
        args = zip(self.exp_tuples, repeat(phi), repeat(self.constant_vec), repeat(self.model_args))
        sses = self.pool.map(sse_wrapper, args)
        for sse, subchain in zip(sses, self.subchains):
            subchain.iter_sample(sse)
        return

    def get_state(self):
        return StateChain(self.curr_theta, self.curr_substates)

    def set_state(self, state):
        self.samples.theta[self.curr_iter] = state.theta
        for subchain, substate in zip(self.subchains, state.substates):
            subchain.set_substate(substate)
        return

    def initialize_sampler(self, ns):
        self.samples = ChainSamples(self.d, ns)
        self.samples.theta[0] = 0.
        for subchain in self.subchains:
            subchain.initialize_sampler(ns)
        self.curr_iter = 0
        return

    def localcov(self, target):
        lc = localcov(
            self.samples.theta[:self.curr_iter],
            target,
            self.radius,
            self.nu,
            self.psi0,
            )
        return lc

    def write_to_disk(self, path, nburn, thin):
        nburn += 1 # one-off
        if os.path.exists(path):
            os.remove(path)

        conn = sql.connect(path)
        cursor = conn.cursor()

        theta  = self.samples.theta[nburn::thin]
        phi    = self.unnormalize(self.invprobit(theta))
        models = list(self.model.report_models_used().items())
        constants = list(self.model.report_constants().items())

        param_create_list = ','.join([x + ' REAL' for x in self.parameter_list])
        param_insert_tple = (','.join(self.parameter_list), ','.join(['?'] * self.d))
        theta_create = self.create_stmt.format('theta', param_create_list)
        theta_insert = self.insert_stmt.format('theta', *param_insert_tple)
        phi_create   = self.create_stmt.format('phi',   param_create_list)
        phi_insert   = self.insert_stmt.format('phi',   *param_insert_tple)
        cursor.execute(theta_create)
        cursor.executemany(theta_insert, theta.tolist())
        cursor.execute(phi_create)
        cursor.executemany(phi_insert, phi.tolist())

        models_create = self.create_stmt.format('models', 'model_type TEXT, model_name TEXT')
        models_insert = self.insert_stmt.format('models', 'model_type, model_name', '?,?')
        cursor.execute(models_create)
        cursor.executemany(models_insert, models)

        consts_create = self.create_stmt.format('constants', 'constant REAL, value REAL')
        consts_insert = self.insert_stmt.format('constants', 'constant, value', '?,?')
        cursor.execute(consts_create)
        cursor.executemany(consts_insert, constants)

        for subchain in self.subchains:
            subchain.write_to_disk(cursor, nburn, thin)

        conn.commit()
        return

    def check_constraints(self, theta):
        phi = self.unnormalize(self.invprobit(theta))
        self.model.update_parameters(phi)
        return self.model.check_constraints()

    def __init__(self, path, bounds, constants, model_args, temperature = 1.):
        conn = sql.connect(path)
        cursor = conn.cursor()
        self.model = MaterialModel(**model_args)
        self.model_args = model_args
        self.parameter_list = self.model.get_parameter_list()
        self.constant_list = self.model.get_constant_list()
        self.bounds = np.array([bounds[key] for key in self.parameter_list])
        self.constant_vec = np.array([constants[key] for key in self.constant_list])
        self.model.initialize_constants(self.constant_vec)
        tables = list(cursor.execute(' SELECT type, table_name FROM meta; '))
        self.subchains = [
            SubChain[type](Experiment[type](cursor, table_name, model_args))
            for type, table_name in tables
            ]
        self.exp_tuples = [subchain.experiment.tuple for subchain in self.subchains]
        self.set_temperature(temperature)
        self.N = len(self.subchains)
        self.d = len(self.parameter_list)
        self.pool = Pool(processes = POOL_SIZE)
        return


# EOF
