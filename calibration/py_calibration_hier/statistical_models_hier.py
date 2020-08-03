"""
statistical_models_hier.py

Defines the Statistical models

Chain      - Defines the Chain object (Hierarchy)
SubChainHB - Defines the Subchain relating to Hopkinson Bar (1 Experiment)
ParallelTemperMaster - Defines the Parallel Tempering Class and methods

"""
import numpy as np
import pandas as pd
import sqlite3 as sql
import os

from scipy.stats import invgamma, invwishart, norm, multivariate_normal as mvnorm
from scipy.special import erf, erfinv
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import cholesky, slogdet
from numpy.random import normal, uniform
from random import sample, shuffle
from math import ceil, sqrt, pi, log, exp
from itertools import combinations

from pointcloud import localcov
from submodel import SubModelHB, SubModelTC, SubModelFP
from transport import TransportHB, TransportTC, TransportFP
from physical_models_c import MaterialModel

import matplotlib.pyplot as plt
import seaborn as sea
import time

sea.set(style = 'ticks')

class BreakException(Exception):
    pass

# Statistical Models

class Transformer(object):
    """ Class to hold transformations (logit, probit) """
    parameter_order = []
    bounds = None

    def normalize(self, x):
        """ Transform real scale to 0-1 scale """
        return (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def unnormalize(self, z):
        """ Transform 0-1 scale to real scale """
        return z * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]

    @staticmethod
    def probit(x):
        """
        Probit Transformation:
        For x bounded 0-1, y is unbounded; between -inf and inf
        """
        return sqrt(2.) * erfinv(2 * x - 1)

    @staticmethod
    def invprobit(y):
        """
        Inverse Probit Transformation
        For real-valued variable y, result x is bounded 0-1
        """
        return 0.5 * (1 + erf(y / sqrt(2.)))

    @staticmethod
    def invprobitlogjac(y):
        """ Computes the log jacobian for the inverse probit transformation """
        return (-0.5 * np.log(2 * pi) - y * y / 2.).sum()

    @staticmethod
    def parameter_trace_plot(sample_parameters, path):
        """ Plots the trace of the sampler """
        palette = plt.get_cmap('Set1')
        if len(sample_parameters.shape) == 1:
            n = sample_parameters.shape[0]
            plt.plot(range(n), sample_parameters, marker = '', linewidth = 1)
        else:
            n, d = sample_parameters.shape
            for i in range(d):
                plt.subplot(d, 1, i+1)
                plt.plot(range(n), sample_parameters[:,i], marker = '',
                        color = palette(i), linewidth = 1)
        #plt.show()
        plt.savefig(path)
        return

    def parameter_pairwise_plot(self, sample_parameters, path):
        """ """
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
        g = g.map_diag(on_diag)

        for i in range(d):
            g.axes[i,i].annotate(self.parameter_order[i], xy = (0.05, 0.9))
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')

        #plt.show()
        plt.savefig(path)
        return

class SubChainSHPB(Transformer):
    """ A sub-chain conducts sampling of PTW parameters and error terms for a
    particular experiment.  Each experiment gets its own sub-chain.

    SubChainHB is developed for use with Hopkinson Bars and quasistatics.
    """
    N = 0
    prior_sigma2_a = 100
    prior_sigma2_b = 5e-6
    temperature = 1.

    meta_query = " SELECT temperature, edot, emax FROM meta WHERE table_name = '{}'; "
    data_query = " SELECT strain, stress FROM {}; "

    @property
    def curr_theta(self):
        return self.s_theta[self.iter]

    @property
    def curr_sigma2(self):
        return self.s_sigma2[self.iter]

    def log_posterior_theta(self, theta0, Sigma, theta, sigma2):
        """ Computes the log-posterior / full conditional for theta.  Note that
            we are tempering the likelihood, and not the prior. Care is taken to
            maintain that. """
        ssse = self.probit_sse(theta) / sigma2 # scaled sum squared error
        tdiff = theta - theta0                 # diff (hier)
        sssd = tdiff.dot(Sigma).dot(tdiff)     # scaled sum squared diff (hier)
        ldj = self.invprobitlogjac(theta)      # Log determinant of jacobian
        tll = - 0.5 * (ssse + sssd) / self.temperature # tempered log Likelihood
        return ldj + tll

    def log_posterior_state(self, theta0, Sigma, theta, sigma2):
        """ Computes the log-posterior for a given sub-chain state at the
            chain's temperature.  This consists of the log-posterior for theta,
            additional contribution from posterior from sigma2.  Only Tempering
            the likelihood, not the prior. """
        # Contribution to total log posterior relevant to theta
        lld = self.log_posterior_theta(theta0, Sigma, theta, sigma2)
        # Additional Contributions to log posterior relevant to sigma2
        lpd = (
            - (0.5 * self.N / self.temperature + self.prior_sigma2_a + 1) * log(sigma2)
            - self.prior_sigma2_b / sigma2
            )
        return lld + lpd

    def initialize_submodel(self, params, consts):
        self.submodel.initialize_submodel(params, consts)
        return

    def sample_sigma2(self, theta):
        sse = self.probit_sse(theta)
        aa = 0.5 * self.N / self.temperature + self.prior_sigma2_a
        bb = 0.5 * sse / self.temperature + self.prior_sigma2_b
        return invgamma.rvs(aa, scale = bb)

    def sse(self, parameters):
        """ Calls submodel.sse for a given set of parameters. """
        # since submodel.sse is cached, the input needs to be hashable.
        # mutable objects (like numpy arrays) are not hashable, so translating
        # to tuple is necessary.
        return self.submodel.sse(tuple(parameters))

    def check_constraints(self, theta):
        """ Check the Material Model constraints """
        th = self.unnormalize(self.invprobit(theta))
        self.materialmodel.update_parameters(th)
        return self.materialmodel.check_constraints()

    def scaled_sse(self, normalized_parameters):
        """ de-scales the SSE from the 0-1 scale to the real-scale, passes
        on to sse """
        return self.sse(self.unnormalize(normalized_parameters))

    def probit_sse(self, probit_parameters):
        """ transforms parameters from probit scale, passes on to scaled_sse """
        return self.scaled_sse(self.invprobit(probit_parameters))

    def localcov(self, target):
        lc = localcov(self.s_theta[:(self.iter - 1)], target,
                      self.radius, self.nu, self.psi0)
        return lc

    def iter_sample(self):
        """ Performs the sample step for 1 iteration. """
        self.iter += 1

        # Generate new sigma^2
        curr_theta = self.s_theta[self.iter - 1]
        self.s_sigma2[self.iter] = self.sample_sigma2(curr_theta)

        # Setting up Covariance Matrix at current
        curr_cov = self.localcov(curr_theta)

        # Generating Proposal, check if meets constraints.  If it doesn't, skip
        prop_theta = curr_theta + normal(size = self.d).dot(cholesky(curr_cov))
        if not self.check_constraints(prop_theta):
            self.s_theta[self.iter] = curr_theta
            return

        # If meets constraints, compute local covariance at proposal.
        prop_cov = self.localcov(prop_theta)

        # Log-Posterior for theta at current and proposal
        curr_lp = self.log_posterior_theta(
            self.parent.curr_theta0,
            self.parent.curr_Sigma,
            curr_theta,
            self.curr_sigma2
            )
        prop_lp = self.log_posterior_theta(
            self.parent.curr_theta0,
            self.parent.curr_Sigma,
            curr_theta,
            self.curr_sigma2
            )

        # Log-density of proposal dist from current to proposal (and visa versa)
        cp_ld = mvnorm.logpdf(prop_theta, curr_theta, curr_cov)
        pc_ld = mvnorm.logpdf(curr_theta, prop_theta, prop_cov)

        # Compute alpha
        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld

        # Conduct Metropolis step:
        if log(uniform()) < log_alpha:
            self.accepted[self.iter] = 1
            self.s_theta[self.iter]  = prop_theta
        else:
            self.s_theta[self.iter]  = curr_theta
        return

    def set_state(self, theta, sigma2):
        self.s_theta[self.iter]  = theta
        self.s_sigma2[self.iter] = sigma2
        return

    def initialize_sampler(self, ns):
        self.s_theta  = np.empty((ns, self.d))
        self.s_sigma2 = np.empty(ns)
        self.accepted = np.zeros(ns)
        self.iter     = 0

        try_theta = normal(size = self.d)
        while not self.check_constraints(try_theta):
            try_theta = normal(size = self.d)

        self.s_theta[0]  = try_theta
        self.s_sigma2[0] = 1e-2
        return

    def set_temperature(self, temperature):
        """ Set the tempering temperature """
        self.temperature = temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def __init__(
            self,
            parent,
            cursor,
            table_name,
            bounds,
            constants,
            nu = 40,
            psi0 = 1e-4,
            r0 = 0.5,
            temperature = 1.,
            **kwargs
            ):
        """ Initialization of Hopkinson Bar Model.

        xp     - dictionary of inputs to transport class (HB)
        bounds - boundaries of parameters
        consts - dictionary of constant Values
        nu     - prior weighting for sampling covariance Matrix
        psi0   - prior diagonal for sampling covariance Matrix
        r0     - search radius for sampling covariance matrix (modified by temp)
        temp   - tempering temperature
        kwargs - Additional arguments to go to MaterialModel
        """
        self.parent = parent

        self.nu = nu
        self.psi0 = psi0
        self.r0 = r0
        self.set_temperature(temperature)

        temperature, edot, emax = list(cursor.execute(self.meta_query.format(table_name)))[0]
        data = np.array(list(cursor.execute(self.data_query.format(table_name))))

        self.submodel = SubModelHB(
            TransportHB(data = data, temp = temperature, emax = emax, edot = edot * 1e-6),
            **kwargs,
            )
        self.N = self.submodel.N
        self.parameter_order = self.submodel.parameter_order
        self.d = len(self.parameter_order)

        # Expose the submodel's materialmodel object and methods (check constraint)
        self.materialmodel = self.submodel.model

        self.bounds = np.array([bounds[key] for key in self.parameter_order])

        params = {
            x : y
            for x, y in
            zip(self.parameter_order,np.zeros(self.d))
            }
        self.initialize_submodel(params, constants)
        return

SubChain = {
    'shpb' : SubChainSHPB,
    }

class Chain(Transformer):
    N = 0
    temperature = 1.

    # Hyperparameter storage slots
    prior_Sigma_nu    = None
    prior_Sigma_psi   = None
    prior_theta0_mu   = None
    prior_theta0_Sinv = None

    rank = 0  # placeholder for MPI rank

    meta_query = """ SELECT type, table_name FROM meta; """
    create_statement = """ CREATE TABLE {}({}); """
    insert_statement = """ INSERT INTO {}({}) values ({}); """

    @property
    def curr_theta0(self):
        return self.s_theta0[self.iter]

    @property
    def curr_Sigma(self):
        return self.s_Sigma[self.iter]

    @property
    def curr_theta(self):
        theta = np.array([subchain.curr_theta for subchain in self.subchains])
        return theta

    @property
    def curr_sigma2(self):
        sigma2 = np.array([subchain.curr_sigma2 for subchain in self.subchains])
        return sigma2

    def set_temperature(self, temperature):
        self.temperature = temperature
        for subchain in self.subchains:
            subchain.set_temperature(temperature)
        return

    def sample_theta0(self, theta, Sigma):
        theta_bar = theta.mean(axis = 0)
        SigL = cho_factor(Sigma)
        Siginv = cho_solve(SigL, np.eye(self.d))
        S0 = cho_solve(
            cho_factor(self.N * Siginv + self.prior_theta0_Sinv),
            np.eye(self.d)
            )
        m0 = (
            self.N * theta_bar.T.dot(Siginv) +
            self.prior_theta0_mu.T.dot(self.prior_theta0_Sinv)
            ).dot(S0)
        S0L = cholesky(S0)
        theta0_try = m0 + S0L.dot(norm.rvs(size = self.d))
        while not self.check_constraints(theta0_try):
            theta0_try = m0 + S0L.dot(norm.rvs(size = self.d))
        return theta0_try

    def sample_Sigma(self, theta, theta0):
        """ Sample Sigma """
        tdiff = (theta - theta0)
        C = sum([np.outer(tdiff[i],tdiff[i]) for i in range(tdiff.shape[0])])
        # Compute parameters for Sigma
        psi0 = self.prior_Sigma_psi + C
        nu0  = self.prior_Sigma_nu + self.N
        # Compute Sigma
        Sigma = invwishart.rvs(df = nu0, scale = psi0)
        return Sigma

    def sample_subchains(self):
        pass

    def log_posterior_state(self, state):
        """ Computes the posterior for a given state, at the chain's temp. """
        # Contribution to log-posterior from subchains
        lps_sc = sum([
            self.subchains[i].log_posterior_state(
                state['theta0'],   state['Sigma'],
                state['theta'][i], state['sigma2'][i],
                )
            for i in range(self.N)
            ])
        # sum of squares between theta0 and prior mean
        t0_diff = state['theta0'] - self.prior_theta0_mu
        lp = (
            - (0.5 * self.N * slogdet(state['Sigma'])[1]) / self.temperature
            - (0.5 * t0_diff.T.dot(self.prior_theta0_Sinv).dot(t0_diff))
            )
        return lps_sc + lp

    def check_constraints(self, theta):
        """ Check the Material Model constraints """
        th = self.unnormalize(self.invprobit(theta))
        self.materialmodel.update_parameters(th)
        return self.materialmodel.check_constraints()

    def iter_sample(self):
        """ Pushes the sampler one iteration """
        # Push each subchain one iteration
        for subchain in self.subchains:
            subchain.iter_sample()

        # Start iteration for main chain
        self.iter += 1

        # Sample new theta0, Sigma
        self.s_Sigma[self.iter] = self.sample_Sigma(
            self.curr_theta, self.s_theta0[self.iter - 1]
            )
        self.s_theta0[self.iter] = self.sample_theta0(
            self.curr_theta, self.s_Sigma[self.iter]
            )
        return

    def sample(self, ns):
        for _ in range(ns):
            self.iter_sample()
        return

    def get_state(self):
        """ Extracts the current state of the chain, returns it as a dictionary.
        Expecting to send it as a pickled object back to the host process. """
        state = {
            'theta0' : self.curr_theta0,
            'Sigma'  : self.curr_Sigma,
            'theta'  : self.curr_theta,
            'sigma2' : self.curr_sigma2,
            'rank'   : self.rank,
            }
        return state

    def set_state(self, state):
        """ Sets the current state of the chain using supplied dictionary """
        self.s_theta0[self.iter] = state['theta0']
        self.s_Sigma[self.iter]  = state['Sigma']
        for i in range(self.N):
            self.subchains[i].set_state(state['theta'][i], state['sigma2'][i])
        return

    def get_history(self, nburn = 0, thin = 1):
        theta0 = self.invprobit(self.s_theta0[nburn::thin])
        return theta0

    def initialize_sampler(self, ns):
        self.s_theta0 = np.empty((ns, self.d))
        self.s_Sigma  = np.empty((ns, self.d, self.d))
        self.accepted = np.zeros(ns)
        self.iter     = 0

        for subchain in self.subchains:
            subchain.initialize_sampler(ns)

        try_theta = normal(size = self.d)
        while not self.check_constraints(try_theta):
            try_theta = normal(size = self.d)

        self.s_theta0[0] = try_theta
        self.s_Sigma[0]  = np.eye(self.d) * 1
        self.curr_ldj    = self.invprobitlogjac(self.curr_theta)
        return

    def write_to_disk(self, path):

        # If output previously exists, delete
        if os.path.exists(path):
            os.remove(path)

        # Create the SQL database
        conn = sql.connect(path)
        curs = conn.cursor()

        # Table Creation and insertion statements
        theta0_create_statement = self.create_statement.format(
            'theta0',
            ','.join([x + ' REAL' for x in self.parameter_order]),
            )
        theta0_insert_statement = self.insert_statement.format(
            'theta0',
            ','.join(self.parameter_order),
            ','.join(['?'] * len(self.parameter_order)),
            )
        thetai_create_statement = self.create_statement.format(
            'theta_{}',
            ','.join([x + ' REAL' for x in self.parameter_order]),
            )
        thetai_insert_statement = self.insert_statement.format(
            'theta_{}',
            ','.join(self.parameter_order),
            ','.join(['?'] * len(self.parameter_order)),
            )
        sigma2i_create_statement = self.create_statement.format(
            'sigma2_{}',
            'sigma2 REAL',
            )
        sigma2i_insert_statement = self.insert_statement.format(
            'sigma2_{}',
            'sigma2',
            '?',
            )

        # Insert hierarchical theta
        curs.execute(theta0_create_statement)
        curs.executemany(
            theta0_insert_statement,
            self.unnormalize(self.invprobit(self.s_theta0)).tolist(),
            )
        for i, subchain in zip(range(1, len(self.subchains) + 1), self.subchains):
            # Insert componenent thetas
            curs.execute(thetai_create_statement.format(i))
            curs.executemany(
                thetai_insert_statement.format(i),
                self.unnormalize(self.invprobit(subchain.s_theta)).tolist(),
                )
            # insert component sigma2's
            curs.execute(sigma2i_create_statement.format(i))
            curs.executemany(
                sigma2i_insert_statement.format(i),
                [(x,) for x in subchain.s_sigma2.tolist()]
                )

        # Write changes to disk, close connection.
        conn.commit()
        conn.close()
        return

    def __init__(self, path, bounds, constants, nu = 40,
                    psi0 = 1e-4, r0 = 0.5, temperature = 1., **kwargs):
        """
        Initialization of Hopkinson Bar Model.

        xp     - dictionary of inputs to transport class (HB)
        bounds - boundaries of parameters
        consts - dictionary of constant Values
        nu     - prior weighting for sampling covariance Matrix
        psi0   - prior diagonal for sampling covariance Matrix
        r0     - search radius for sampling covariance matrix (modified by temp)
        temp   - tempering temperature
        kwargs - Additional arguments to go to MaterialModel
        """
        self.materialmodel = MaterialModel(**kwargs)

        conn = sql.connect(path)
        cursor = conn.cursor()
        experiments = list(cursor.execute(self.meta_query))

        self.subchains = [
            SubChain[type](
                self, cursor, table_name, bounds = bounds, constants = constants,
                nu = nu, psi0 = psi0, r0 = r0, temperature = temperature, **kwargs,
                )
            for type, table_name
            in experiments
            ]
        self.nu = nu
        self.psi0 = psi0
        self.r0 = r0
        self.temperature = temperature

        self.N = len(self.subchains)
        self.n = np.array([subchain.N for subchain in self.subchains])

        self.parameter_order = self.materialmodel.get_parameter_list()
        self.d = len(self.parameter_order)

        self.prior_Sigma_nu    = self.d
        self.prior_Sigma_psi   = 0.5 * np.eye(self.d) / self.d
        self.prior_theta0_mu   = np.zeros(self.d)
        self.prior_theta0_Sinv = 0.5 * np.eye(self.d)

        self.bounds = np.array([bounds[key] for key in self.parameter_order])
        conn.close()
        return

class ParallelTemperMaster(Transformer):
    temperature_ladder = np.array([1.])
    chains = []

    swap_yes = []
    swap_no  = []

    def initialize_chains(self, kwargs):
        # Initialization
        self.chains = [
            Chain(**kwargs, temperature = t)
            for t in self.temperature_ladder
            ]
        # Approximation to MPI Rank (for tracking)
        for i in range(len(self.chains)):
            self.chains[i].rank = i + 1
        return

    def get_states(self):
        states = [chain.get_state() for chain in self.chains]
        return states

    def try_swap_states(self, chain1, chain2):
        state1 = chain1.get_state()
        state2 = chain2.get_state()

        lp11 = chain1.log_posterior_state(state1)
        lp12 = chain1.log_posterior_state(state2)
        lp21 = chain2.log_posterior_state(state1)
        lp22 = chain2.log_posterior_state(state2)

        log_alpha = lp12 + lp21 - lp11 - lp22

        if log(uniform()) < log_alpha:
            chain1.set_state(state2)
            chain2.set_state(state1)
            self.swap_yes.append((state1['rank'], state2['rank']))
        else:
            self.swap_no.append((state1['rank'], state2['rank']))
        return

    def initialize_sampler(self, tns):
        for chain in self.chains:
            chain.initialize_sampler(tns)
        return

    def temper_chains(self):
        chain_idx = list(range(self.size))
        shuffle(chain_idx)

        for cidx1, cidx2 in zip(chain_idx[::2], chain_idx[1::2]):
            self.try_swap_states(self.chains[cidx1], self.chains[cidx2])

        return

    def sample_chains(self, ns = 1):
        for chain in self.chains:
            chain.sample(ns)
        return

    def sample(self, nburn = 40000, nsamp = 60000, kswap = 5):
        sampled = 1
        tns = nsamp + nburn + 1
        print('Beginning sampling for {} total samples'.format(tns - 1))
        self.initialize_sampler(tns)
        print('\rSampling {:.1%} Complete'.format(sampled / tns), end = '')
        for _ in range(nburn // kswap):
            self.sample_chains(kswap)
            self.temper_chains()
            sampled += kswap
            print('\rSampling {:.1%} Complete'.format(sampled / tns), end = '')
        self.sample_chains(nburn % kswap)
        sampled += nburn % kswap
        for _ in range(nsamp // kswap):
            self.sample_chains(kswap)
            self.temper_chains()
            sampled += kswap
            print('\rSampling {:.1%} Complete'.format(sampled / tns), end = '')
        self.sample_chains(nsamp % kswap)
        sampled += nsamp % kswap
        print('\rSampling {:.1%} Complete'.format(sampled / tns))
        print('Sampling Complete for {} Samples'.format(nsamp))
        return

    def write_to_disk(self, path):
        self.chains[0].write_to_disk(path)
        return

    def get_history(self, *args):
        return self.chains[0].get_history(*args)

    def parameter_pairwise_plot(self, theta, path):
        self.chains[0].parameter_pairwise_plot(theta, path)
        return

    def complete(self):
        return

    def __init__(self, temperature_ladder, **kwargs):
        self.size = len(temperature_ladder)
        self.temperature_ladder = temperature_ladder
        self.initialize_chains(kwargs)
        return

if __name__ == '__main__':
    pass

# EOF
