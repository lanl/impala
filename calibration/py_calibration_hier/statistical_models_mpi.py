"""
statistical_models_mpi.py

Defines the Statistical models

RemoteChain - declared on the slave processes
ParallelTemperingMaster - declared on master process
Dispatcher  - declared on slave processes
"""
import numpy as np
import pandas as pd

from mpi4py import MPI
from scipy.stats import invgamma, multivariate_normal as mvnorm
from scipy.special import erf, erfinv
from numpy.linalg import cholesky
from numpy.random import normal, uniform
from random import sample
from math import ceil, sqrt, pi, log, exp
from itertools import combinations
from pointcloud import localcov
from random import shuffle

from submodel import SubModelHB, SubModelTC, SubModelFP
from transport import TransportHB, TransportTC, TransportFP

import matplotlib.pyplot as plt
import seaborn as sea
import time

sea.set(style = 'ticks')

class MPI_Error(Exception):
    pass

class BreakException(Exception):
    pass

# Statistical Models

class RemoteChain(object):
    """ RemoteChain objects are the sampler chains on each MPI process.
    Contained within each RemoteChain object is a complete Metropolis Hastings
    sampler. """
    N       = np.array((0,0,0))
    # PRIOR SETTINGS
    s2_hb_a = np.array((100.));         s2_hb_b = np.array((5e-6))
    s2_tc_a = np.array((2., 2.5, 2.5)); s2_tc_b = np.array((0.01, 0.2, 0.2))
    s2_fp_a = np.array((4., 6.));       s2_fp_b = np.array((0.00025, 200.))

    temper_temp = 1.;   inv_temp   = 1.

    def set_temperature(self, temperature):
        """ Sets the chain temperature, and related things. """
        self.temper_temp = temperature
        self.inv_temp = 1./temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def normalize(self, x):
        """ Scales x between 0 and 1 """
        return (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def unnormalize(self, z):
        """ Transforms x from 0-1 to original variable """
        return z * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]

    @staticmethod
    def logit(x):
        """
        Logit Transformation:
        For x bounded to 0-1, y is unbounded; between -inf and inf
        """
        return np.log(x / (1 - x))

    @staticmethod
    def invlogit(y):
        """
        Inverse Logit Transformation:
        For real-valued variable y, result x is bounded 0-1
        """
        return 1 / (1 + np.exp(-y))

    @staticmethod
    def invlogitlogjac(y):
        """
        Computes the log jacobian for the inverse logit transformation.
        """
        return (- y - 2 * np.log(1. + np.exp(-y))).sum()

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

    def sse(self, parameters):
        """ Computes SSE for each experiment type, returns a tuple of sse's """
        hb_sse = sum([sm.sse(parameters[sm.pidx]) for sm in self.submodels_hb])
        tc_sse = sum([sm.sse(parameters[sm.pidx]) for sm in self.submodels_tc])
        fp_sse = sum([sm.sse(parameters[sm.pidx]) for sm in self.submodels_fp])
        return hb_sse, tc_sse, fp_sse

    def scaled_sse(self, normalized_parameters):
        """ Unnormalizes the inputs, then computes the sum of squared errors """
        parameters = self.unnormalize(normalized_parameters)
        return self.sse(parameters)

    def probit_sse(self, probit_parameters):
        """ de-probits the inputs, then passes on to scaled_sse """
        normalized_parameters = self.invprobit(probit_parameters)
        return self.scaled_sse(normalized_parameters)

    def logit_sse(self, logit_parameters):
        """ de-logits the inputs, and passes on to scaled_sse """
        normalized_parameters = self.invlogit(logit_parameters)
        return self.scaled_sse(normalized_parameters)

    def get_lcd(self):
        """ Computes the Log-Determinant of the Covariance Matrix (after
        factoring out sigma2) unique to each submodel.  Currently only used for
        HB_AR1. """
        hb_lcd = sum([sm.curr_lcd for sm in self.submodels_hb])
        tc_lcd = sum([sm.curr_lcd for sm in self.submodels_tc])
        fp_lcd = sum([sm.curr_lcd for sm in self.submodels_fp])
        if len(self.submodels_hb) > 0:
            hb_lcd = hb_lcd / len(self.submodels_hb)
        if len(self.submodels_tc) > 0:
            tc_lcd = tc_lcd / len(self.submodels_tc)
        if len(self.submodels_fp) > 0:
            fp_lcd = fp_lcd / len(self.submodels_fp)
        return np.array((hb_lcd, tc_lcd, fp_lcd))

    def log_posterior(self, sse, s2hb, s2tc, s2fp, ldj, lcd, invtem = 1.):
        """ Computes a tempered log-posterior (where only the likelihood is
        being tempered).  Default temperature = 1, indicating untempered. """
        # Hopkinson Bar
        lp_hb = (
            - (0.5 * (self.N[0] * self.Wp[0] * invtem + lcd[0]) +
                    self.s2_hb_a + 1) * np.log(s2hb)
            - (0.5 * sse[0] * invtem + self.s2_hb_b) / s2hb
            ).sum()
        # Taylor Cylinder
        lp_tc = (
            - (0.5 * (self.N[1] / 3. * self.Wp[1] * invtem + lcd[1]) +
                    self.s2_tc_a + 1) * np.log(s2tc)
            - (0.5 * sse[1] * invtem + self.s2_tc_b) / s2tc
            ).sum()
        # Flyer Plate
        lp_fp = (
            - (0.5 * (self.N[2] / 2. * self.Wp[2] * invtem + lcd[2]) +
                    self.s2_fp_a + 1) * np.log(s2fp)
            - (0.5 * sse[2] * invtem + self.s2_fp_b) / s2fp
            ).sum()
        return lp_hb + lp_tc + lp_fp + ldj * invtem

    def get_N(self):
        """ Returns the number of observations per model type """
        hb_n = sum([hbm.N for hbm in self.submodels_hb])
        tc_n = sum([tcm.N for tcm in self.submodels_tc])
        fp_n = sum([fpm.N for fpm in self.submodels_fp])
        return np.array((hb_n, tc_n, fp_n))

    def get_accept_probability(self):
        """ Returns the MH Acceptance probability """
        return self.accepted.mean()

    def set_parameter_order(self):
        """
        Sets the parameter order, and the sub-model parameter indices.
        """
        # Finds the Material Model Parameter order -- These will be used for
        # each of the experiment types
        parameters = (
            self.submodels_hb + self.submodels_tc + self.submodels_fp
            )[0].parameter_order.copy()
        param_l = len(parameters)
        par_idx = list(range(param_l))

        mat_para = []
        exp_para = []

        # Finds each of the unique material parameter indices
        for sm in self.submodels_hb + self.submodels_tc + self.submodels_fp:
            mat_para += sm.mat_para
        mat_para = list(set(mat_para))

        # Finds the experiment parameter indices
        start = param_l + len(mat_para)
        for sm in self.submodels_hb + self.submodels_tc + self.submodels_fp:
            mat_idx = [
                param_l + mat_para.index(p)
                for p in sm.mat_para
                ]
            exp_para += sm.exp_para
            exp_idx = list(range(start, start + len(sm.exp_para)))
            # Assigns the parameter indices to the submodel
            sm.pidx = np.array(par_idx + exp_idx + mat_idx, dtype = int)
            start += len(sm.exp_para)

        # Specifies the complete sampling vector
        self.parameter_order = parameters + mat_para + exp_para

        # Remove this after confirming working
        if self.rank == 1:
            print(self.parameter_order)
            for sm in self.submodels_hb + self.submodels_tc + self.submodels_fp:
                print(sm.pidx)
        return

    def initialize_submodels(self, parameters, constants):
        """ Invokes the initialize_submodel method for each submodel """
        for sm in self.submodels_hb + self.submodels_tc + self.submodels_fp:
            sm.initialize_submodel(parameters, constants)
        return

    def initialize_sampler(self, ns, starting = None):
        """ Initializes the sampler for a given total number of samples.

        default value for starting starts the sampler in the middle of the range
        for each of the parameter values. """
        if starting is None:
            starting = np.array([0.] * self.d)

        # Initialize Sample Target arrays
        self.s_theta     = np.empty((ns, self.d))
        self.s_s2_hb     = np.empty(ns)
        self.s_s2_tc     = np.empty((ns, 3))
        self.s_s2_fp     = np.empty((ns, 2))
        self.accepted    = np.zeros(ns)

        # Pass Starting Values for theta
        self.s_theta[0]  = starting
        self.curr_sse    = self.probit_sse(self.s_theta[0])
        self.curr_ldj    = self.invprobitlogjac(self.s_theta[0])
        self.curr_lcd    = self.get_lcd()
        self.s_s2_hb[0]  = 1e-5
        self.s_s2_tc[0]  = (0.01, 0.01, 0.01)
        self.s_s2_fp[0]  = (0.01, 0.01)
        return

    def sample_sigma2_hb(self):
        """ Sample Sigma2 for Hopkinson Bar """
        alpha = 0.5 * self.inv_temp * self.N[0] * self.Wp[0] + self.s2_hb_a
        beta  = 0.5 * self.inv_temp * self.Wp[0] * self.curr_sse[0] + self.s2_hb_b
        return invgamma.rvs(alpha, scale = beta)

    def sample_sigma2_tc(self):
        """ Sample Sigma2 for Taylor Cylinder """
        alpha = 0.5 * self.inv_temp * self.N[1]/3. * self.Wp[1] + self.s2_tc_a
        beta  = 0.5 * self.inv_temp * self.Wp[1] * self.curr_sse[1] + self.s2_tc_b
        return invgamma.rvs(alpha, scale = beta)

    def sample_sigma2_fp(self):
        """ Sample Sigma2 for Flyer Plate """
        alpha = 0.5 * self.inv_temp * self.N[2]/2. * self.Wp[2] + self.s2_fp_a
        beta  = 0.5 * self.inv_temp * self.Wp[2] * self.curr_sse[2] + self.s2_fp_b
        return invgamma.rvs(alpha, scale = beta)

    def iter_sample(self):
        """
        Generates 1 draw from the posterior distribution -- Essentially a
        Metropolis-within-Gibbs Algorithm.

        Sigma2's are updated in a Gibbs step, sampling from full conditional.
        The Strength parameters, material parameters, and experiment specific
        parameters are all undated at once in a single Metropolis Hastings Step.

        The Metropolis Hastings step relies on a localized covariance structure.
        """
        self.iter += 1

        # Generating new sigma^2
        self.s_s2_hb[self.iter] = self.sample_sigma2_hb()
        self.s_s2_tc[self.iter] = self.sample_sigma2_tc()
        self.s_s2_fp[self.iter] = self.sample_sigma2_fp()

        # Setting up Covariance Matrix at current
        curr_theta = self.s_theta[self.iter - 1]
        curr_cov = localcov(
            self.s_theta[:self.iter-1], curr_theta, self.radius, self.nu, self.psi0,
            )

        # Generating Proposal, computing Covariance Matrix at proposal
        prop_theta = curr_theta + normal(size = self.d).dot(cholesky(curr_cov))
        prop_cov = localcov(
            self.s_theta[:self.iter-1], prop_theta, self.radius, self.nu, self.psi0,
            )
        prop_sse = self.probit_sse(prop_theta)
        prop_lcd = self.get_lcd()
        prop_ldj = self.invprobitlogjac(prop_theta)

        # Log-Posterior for theta at current and proposal
        curr_lp = self.log_posterior(
                self.curr_sse,
                self.s_s2_hb[self.iter],
                self.s_s2_tc[self.iter],
                self.s_s2_fp[self.iter],
                self.curr_ldj,
                self.curr_lcd,
                self.inv_temp,
                )
        prop_lp = self.log_posterior(
                prop_sse,
                self.s_s2_hb[self.iter],
                self.s_s2_tc[self.iter],
                self.s_s2_fp[self.iter],
                prop_ldj,
                prop_lcd,
                self.inv_temp,
                )

        # Log-density of proposal dist from current to proposal (and visa versa)
        cp_ld = mvnorm.logpdf(prop_theta, curr_theta, curr_cov)
        pc_ld = mvnorm.logpdf(curr_theta, prop_theta, prop_cov)

        # Compute alpha:
        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld

        # If uniform() < alpha: accept proposal; else repeat current
        if log(uniform()) < log_alpha:
            self.accepted[self.iter] = 1
            self.s_theta[self.iter] = prop_theta
            self.curr_sse = prop_sse
            self.curr_ldj = prop_ldj
            self.curr_lcd = prop_lcd
        else:
            self.s_theta[self.iter] = curr_theta
            # Re-compute SSE to ensure that we don't get stuck
            self.curr_sse = self.probit_sse(curr_theta)
        return

    def sample(self, n):
        """ iterates the sampler by n steps """
        for _ in range(n):
            self.iter_sample()
        return

    def get_state(self):
        """ Get the current state of the sampler, for passing via MPI """
        state = {
            'theta'  : self.s_theta[self.iter],
            's2hb'   : self.s_s2_hb[self.iter],
            's2tc'   : self.s_s2_tc[self.iter],
            's2fp'   : self.s_s2_fp[self.iter],
            'sse'    : self.curr_sse,
            'ldj'    : self.curr_ldj,
            'lcd'    : self.curr_lcd,
            'invtem' : self.inv_temp,
            'rank'   : self.rank,
            }
        return state

    def set_state(self, state):
        """ Sets the current state of the sampler, based on input dictionary """
        self.s_theta[self.iter] = state['theta']
        self.s_s2_hb[self.iter] = state['s2hb']
        self.s_s2_tc[self.iter] = state['s2tc']
        self.s_s2_fp[self.iter] = state['s2fp']
        self.curr_sse = state['sse']
        self.curr_ldj = state['ldj']
        self.curr_lcd = state['lcd']
        return

    def get_history(self, nburn = 0, thin = 1):
        """  """
        theta = self.invprobit(self.s_theta[(nburn+1)::thin])
        s2hb  = self.s_s2_hb[(nburn + 1)::thin]
        s2tc  = self.s_s2_tc[(nburn + 1)::thin]
        s2fp  = self.s_s2_fp[(nburn + 1)::thin]
        return (theta, s2hb, s2tc, s2fp)

    def write_to_disk(self):
        """ Writes the Sampler history to disk. """
        df_t = pd.DataFrame(self.invprobit(self.s_theta), columns = self.parameter_order)
        df_t.to_csv('~/samples/r{:02d}_theta.csv'.format(self.rank), index = False)
        df_s2_hb = pd.DataFrame({'s2_hb' : self.s_s2_hb})
        df_s2_tc = pd.DataFrame(self.s_s2_tc, columns = ('s2_tc_1','s2_tc_2','s2_tc_3'))
        df_s2_fp = pd.DataFrame(self.s_s2_fp, columns = ('s2_fp_1','s2_fp_2'))
        df_s2 = pd.concat([df_s2_hb, df_s2_tc, df_s2_fp], axis = 1)
        df_s2.to_csv('~/samples/r{:02d}_sigma2.csv'.format(self.rank), index = False)
        return

    def parameter_trace_plot(self, sample_parameters, path):
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

    def prediction_plot_hb(self, normalized_parameters, sigma2, path, ylim):
        """ Creates a plot of predicted Hopkinson Bar Stress Strain Curves,
        along with the observed data for each hoppy bar / quasistatic. """
        parameters = np.apply_along_axis(self.unnormalize, 1, normalized_parameters)
        plot_count = len(self.submodels_hb)
        d_across = ceil(sqrt(plot_count))
        d_down   = ceil(float(plot_count) / d_across)
        k = 0
        for sm in self.submodels_hb:
            k += 1
            plt.subplot(d_across, d_down, k)
            sm.prediction_plot(parameters[:,sm.pidx], sigma2, ylim)

        plt.savefig(path)
        return

    def prediction_plot_tc(self, normalized_parameters, sigma2, path):
        """ Creates plot of Taylor Cylinder predictions and posterior weighting """
        parameters = np.apply_along_axis(self.unnormalize, 1, normalized_parameters)
        row_count  = len(self.submodels_tc)
        row = 0
        for sm in self.submodels_tc:
            row += 1
            sm.prediction_plot(parameters[:,sm.pidx], sigma2, row_count, row)

        plt.savefig(path)
        return

    def prediction_plot_fp(self, normalized_parameters, sigma2, path):
        """ Creates plot of flyer plate predictions and posterior weighting """
        parameters = np.apply_along_axis(self.unnormalize, 1, normalized_parameters)
        row_count = len(self.submodels_fp) * 2
        smid = 1
        for sm in self.submodels_tc:
            print(row_count)
            print(smid)
            sm.prediction_plot(parameters[:, sm.pidx], sigma2, row_count, smid)
            smid += 2

        plt.savefig(path)
        return

    def __init__(
            self,
            rank,           # MPI Rank
            xp_hb,          # list of dictionaries of paths to files (Hopkinson Bar)
            xp_tc,          # list of dictionaries of paths to files (Taylor Cylinder)
            xp_fp,          # list of dictionaries of paths to files (Flyer Plate)
            bounds,         # Dictionary of parameter bounds
            constants,      # Dictionary of Constant values
            W,              # Initial Weighting by experiment type
            params = None,  # Dictionary of initial Parameter Values (Not required)
            nu     = 40,    # Prior weighting for sampling covariance matrix
            psi0   = 1e-4,  # Prior diagonal for sampling covariance matrix
            r0     = 0.5,   # Radius = r0 * log_{10}(Temperature + 1)
            temp   = 1.,    # Tempering Temperature
            **kwargs,       # MaterialModel arguments
            ):
        """  """
        self.rank = rank
        self.iter = 0
        self.nu   = nu
        self.psi0 = psi0
        self.r0   = r0
        self.set_temperature(temp)

        # Declare the Sub Models
        self.submodels_hb = [
            SubModelHB(TransportHB(**xp), **kwargs)
            for xp in xp_hb
            ]
        self.submodels_tc = [
            SubModelTC(TransportTC(**xp), **kwargs)
            for xp in xp_tc
            ]
        self.submodels_fp = [
            SubModelFP(TransportFP(**xp), **kwargs)
            for xp in xp_fp
            ]

        # Set up the likelihood weighting
        self.W = W
        self.N = self.get_N()
        with np.errstate(divide='ignore', invalid='ignore'):
            Wp = np.true_divide(self.W * self.N.sum(), self.N)
            Wp[Wp == np.inf] = 0.
            Wp = np.nan_to_num(Wp)
            self.Wp = Wp

        # Print Diagnostic Information
        if rank == 1:
            print("Number of Observations (Raw): {}".format(self.N))
            print("Assigned Weights: {}".format(self.W))
            print("Modified Weights: {}".format(self.Wp))
            print("Number of Observations (New): {}".format(self.N * self.Wp))

        # Set Sampling Parameter List and sub-model specific indices
        self.set_parameter_order()
        self.d = len(self.parameter_order)

        # Create numpy array of bounds in proper order
        self.bounds = np.array([bounds[key] for key in self.parameter_order])
        parameters = {
            x : y
            for x, y in zip(
                self.parameter_order,
                self.unnormalize(np.ones(len(self.parameter_order)) * 0.5)
                )
            }
        self.initialize_submodels(parameters, constants)
        return

class ParallelTemperingMaster(object):
    """ This is a Parallel Tempering Master Object.

    It exists on process rank 0 (master).  Methods within this class are meant
    to interact with the dispatcher function on processes with rank > 0.

    These methods implement a kind of Remote Procedure Call (RPC)
    infrastructure.  The RemoteChain objects exist in the global namespace of
    the slave processes, 1 per process, and are interacted with by the
    dispatcher function on those processes.

    PTM on main > MPI > dispatcher > RemoteChain """
    # Target list for Chain State Swap successes or failures
    swap_yes = []
    swap_no  = []

    # Expose Static Methods of RemoteChain class to PT class
    probit          = RemoteChain.probit
    invprobit       = RemoteChain.invprobit
    invprobitlogjac = RemoteChain.invprobitlogjac
    logit           = RemoteChain.logit
    invlogit        = RemoteChain.invlogit
    invlogitlogjac  = RemoteChain.invlogitlogjac

    def check_ready(self):
        """ Checks whether all the slave processes sent True (indicates they are
        ready for new instructions). """
        recv = [self.comm.irecv(source = i) for i in range(1,self.Size)]
        data = [r.wait() for r in recv]
        try:
            assert all(data)
        except AssertionError:
            raise MPI_Error('Some processes encountered errors, I guess.')
        return

    def initialize_chains(self, kwargs):
        """ Initializes chains on each of the slave processes.  Waits until all
        the processes are ready again. """
        sent = [
            self.comm.isend(['initialize_chain', {**kwargs, 'rank' : i}], dest = i)
            for i in range(1, self.Size)
            ]
        self.check_ready()
        return

    def set_chain_temperatures(self, temp_ladder):
        """ Set the temperature on each of the slave processes """
        sent = [
            self.comm.isend(['set_temperature', temp_ladder[i-1]], dest = i)
            for i in range(1, self.Size)
            ]
        self.check_ready()
        return

    def swap_chain_states(self, ranks):
        """ Swap States between chains.  Checks that states were successfully
        swapped.

        ranks is supplied as a tuple of process ranks indicating which chains
        are to be swapped.
        """
        sent = [self.comm.isend(['get_state'], dest = i) for i in ranks]
        recv = [self.comm.irecv(source = i) for i in ranks]
        data = [r.wait() for r in recv]
        sent = [
            self.comm.isend(['set_state', data[::-1][k]], dest = ranks[k])
            for k in range(2)
            ]
        return

    def get_states(self):
        """ Gets the current state from each of the chains. """
        sent = [
            self.comm.isend(['get_state'], dest = i)
            for i in range(1, self.Size)
            ]
        recv = [
            self.comm.irecv(source = i)
            for i in range(1, self.Size)
            ]
        data = [r.wait() for r in recv]
        return data

    def try_swap_states(self):
        """ Tries to swap states between chains.

        Generates list of dictionaries:
        [chain.state, chain2.state, ...]
        Each state is a dictionary describing the state of the chain.
        (sse, sigma2, ...) along with parameters of the chain (rank, inverse temp)

        Shuffles list of tuples, then for each 2 elements in list, attempts to
        swap states. """
        # Get list of current states, and shuffle the order
        states = self.get_states()
        shuffle(states)
        # Pull states, 2 at a time, from the list of states
        for x, y in list(zip(states[::2], states[1::2])):
            # Log posterior for chain x, using temp x
            lpxx = self.log_posterior(x['sse'], x['s2hb'], x['s2tc'], x['s2fp'],
                                        x['ldj'], x['lcd'], x['invtem'])
            # Log posterior for chain y, using temp y
            lpxy = self.log_posterior(x['sse'], x['s2hb'], x['s2tc'], x['s2fp'],
                                        x['ldj'], x['lcd'], y['invtem'])
            # Log posterior for chain y, using temp x
            lpyx = self.log_posterior(y['sse'], y['s2hb'], y['s2tc'], y['s2fp'],
                                        y['ldj'], y['lcd'], x['invtem'])
            # Log posterior for chain y, using temp y
            lpyy = self.log_posterior(y['sse'], y['s2hb'], y['s2tc'], y['s2fp'],
                                        y['ldj'], y['lcd'], y['invtem'])
            # Compute Chain Swap Probability
            log_alpha = lpxy + lpyx - lpxx - lpyy
            if log(uniform()) < log_alpha:
                # Swap States, report state swap success
                self.swap_yes.append((x['rank'], y['rank']))
                s1 = self.comm.isend(['set_state', x], dest = y['rank'])
                s2 = self.comm.isend(['set_state', y], dest = x['rank'])
            else:
                # Report state swap fail
                self.swap_no.append((x['rank'], y['rank']))
        return

    def sample(self, ns):
        """ Tells the chains to interate 'ns' samples.  Waits until all chains
        have done so. """
        sent = [
            self.comm.isend(['sample', ns], dest = i)
            for i in range(1, self.Size)
            ]
        self.check_ready()
        return

    def initialize_sampler(self, ns):
        """ Tells the chains to initialize the sampler for a given total sample
        size.  Waits until all chains have done so. """
        sent = [
            self.comm.isend(['initialize_sampler', ns], dest = i)
            for i in range(1, self.Size)
            ]
        self.check_ready()
        return

    def sampler(self, nsamp, nburn, tburn, kswap):
        """ Assembles total number of samples and initializes sampler.

        Then samples for tburn (number of burnin before tempering).
        Then samples for nburn (number of burnin after beginning tempering),
            attempting state swap every kswap iterations.
        Then samples for nsamp (number of output samples), continuing to attempt
            state swap every kswap iterations.
        """
        sampled = 1
        tns = nsamp + nburn + tburn + 1
        print('Beginning Sampling for {} Total samples'.format(tns - 1))
        self.initialize_sampler(tns)
        # Burnin before tempering
        self.sample(tburn)
        sampled += tburn
        print('\rSampling {:.1%} Complete'.format(sampled/tns), end = '')
        # Burnin after beginning tempering
        for _ in range(nburn // kswap):
            self.try_swap_states()
            self.sample(kswap)
            sampled += kswap
            print('\rSampling {:.1%} Complete'.format(sampled/tns), end = '')
        self.sample(nburn % kswap)
        sampled += (nburn % kswap)
        # Start actual sampling
        for _ in range(nsamp // kswap):
            self.try_swap_states()
            self.sample(kswap)
            sampled += kswap
            print('\rSampling {:.1%} Complete'.format(sampled/tns), end = '')
        self.sample(nsamp % kswap)
        sampled += nsamp % kswap
        print('\rSampling {:.1%} Complete'.format(sampled/tns))
        print('Completed Sampling for {} samples'.format(nsamp))
        return

    def get_accept_prob(self):
        """ Gets the MH acceptance probability for each slave process """
        sent = [
            self.comm.isend(['get_accept_prob'], dest = i)
            for i in range(1, self.Size)
            ]
        recv = [self.comm.irecv(source = i) for i in range(1, self.Size)]
        data = [r.wait() for r in recv]
        return data

    def get_swap_prob(self):
        """ Gets the swap probability between states, returns a matrix """
        swaps = np.zeros((self.Size - 1, self.Size - 1))
        fails = np.zeros((self.Size - 1, self.Size - 1))
        for a, b in self.swap_yes:
            swaps[a - 1, b - 1] += 1
            swaps[b - 1, a - 1] += 1
        for a, b in self.swap_no:
            fails[a - 1, b - 1] += 1
            fails[b - 1, a - 1] += 1
        swap_prob = swaps / (swaps + fails + np.eye(swaps.shape[0]))
        return swap_prob

    def get_history_theta(self, nsamp, d):
        """ Gets the sampling history from each of the chains. """
        sent = [
            self.comm.isend(['get_samples_theta'], dest = i)
            for i in range(1, self.Size)
            ]
        data = [np.empty((nsamp, d)).copy() for _ in range(1, self.Size)]
        for i in range(1, self.Size):
            self.comm.Recv([data[i-1], MPI.DOUBLE], source = i)
        return data

    def get_history_sigma2_hb(self, nsamp):
        """ Get the Hopkinson Bar sigma2 sample history """
        sent = [
            self.comm.isend(['get_samples_sigma2_hb'], dest = i)
            for i in range(1, self.Size)
            ]
        data = [np.empty(nsamp).copy() for _ in range(1, self.Size)]
        for i in range(1, self.Size):
            self.comm.Recv([data[i-1], MPI.DOUBLE], source = i)
        return data

    def get_history_sigma2_tc(self, nsamp):
        """ Get the Taylor Cylinder Sigma2 sample history """
        sent = [
            self.comm.isend(['get_samples_sigma2_tc'], dest = i)
            for i in range(1, self.Size)
            ]
        data = [np.empty((nsamp, 3)).copy() for _ in range(1, self.Size)]
        for i in range(1, self.Size):
            self.comm.Recv([data[i-1], MPI.DOUBLE], source = i)
        return data

    def get_history_sigma2_fp(self, nsamp):
        """ Get the Flyer Plate Sigma2 sample history """
        sent = [
            self.comm.isend(['get_samples_sigma2_fp'], dest = i)
            for i in range(1, self.Size)
            ]
        data = [np.empty((nsamp, 2)).copy() for _ in range(1, self.Size)]
        for i in range(1, self.Size):
            self.comm.Recv([data[i-1], MPI.DOUBLE], source = i)
        return data

    def write_to_disk(self):
        """ Tell each chain to write its sampling history to disk """
        sent = [
            self.comm.isend(['write_to_disk'], dest = i)
            for i in range(1, self.Size)
            ]
        self.check_ready()
        return

    def complete(self):
        """ Tells the slave processes that they're done. """
        sent = [
            self.comm.isend(['break'], dest = i)
            for i in range(1, self.Size)
            ]
        for s in sent:
            s.wait()
        return

    def swap_prob_plot(self, path, figsize = (5,6), dpi = 300):
        """ Create a Chain-swap-probability plot """
        swap_matrix = self.get_swap_prob()
        fig = plt.figure(figsize = figsize)
        plt.matshow(swap_matrix)
        plt.colorbar()
        plt.savefig(path, dpi = dpi)
        plt.close()
        return

    def accept_prob_plot(self, path, figsize = (4,5), dpi = 300):
        """ Create a within-chain acceptance probability plot """
        accept_prob = self.get_accept_prob()
        idx = range(self.Size - 1)
        temps = ['{:.2f}'.format(x) for x in list(self.temp_ladder)]
        fig = plt.figure(figsize = figsize)
        plt.bar(idx, height = accept_prob, tick_label = temps)
        plt.xticks(rotation = 60)
        plt.title('Acceptance Probability by Chain')
        plt.savefig(path, dpi = dpi)
        plt.close()
        return

    def __init__(self, comm, size, temp_ladder = np.array([1.,]), **kwargs):
        """ Initialization routine for MPI Parallel Tempering Master object.
        arguments:
        -          comm : mpi4py.MPI.comm
        -          size : mpi4py.MPI.comm.size
        -   temp_ladder : temperatures for parallel tempering--[1,...]
        -      **kwargs : arguments for RemoteChain class
        """
        try:
            assert len(temp_ladder) == (size - 1)
        except AssertionError:
            print('Length of Temperature Ladder must match MPI slave count')
        # MPI parameters
        self.comm            = comm
        self.Size            = size
        self.temp_ladder     = temp_ladder

        # Start the chains!
        self.initialize_chains(kwargs)
        self.set_chain_temperatures(temp_ladder)

        # Declare Reference Chain (to expose internal methods)
        self.ref_chain               = RemoteChain(**{'rank' : 0, **kwargs})
        self.parameter_order         = self.ref_chain.parameter_order
        self.log_posterior           = self.ref_chain.log_posterior
        self.parameter_trace_plot    = self.ref_chain.parameter_trace_plot
        self.parameter_pairwise_plot = self.ref_chain.parameter_pairwise_plot
        self.prediction_plot_hb      = self.ref_chain.prediction_plot_hb
        self.prediction_plot_tc      = self.ref_chain.prediction_plot_tc
        self.prediction_plot_fp      = self.ref_chain.prediction_plot_fp
        return

# Dispatcher class (spawned on the slave processes)

class Dispatcher(object):
    """ Class to control the RemoteChain Class via interaction through MPI with
    ParallelTemperingMaster class. """
    comm  = None
    rank  = None
    chain = None

    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        return

    def return_value(self, retv):
        """ Send a particular value back to root node. """
        sent = self.comm.isend(retv, dest = 0)
        sent.wait()
        return

    def return_true(self):
        """ send True back to root node (indicates ready for further
        instruction) """
        self.return_value(True)
        return

    def write_to_disk(self):
        """ Tells the chain to write sampling history to disk """
        self.chain.write_to_disk()
        return

    def watch(self):
        """ Watch for incoming instructions; when one comes in send it to
        dispatch. """
        try:
            while True:
                recv = self.comm.irecv(source = 0)
                parcel = recv.wait()
                self.dispatch(parcel)
        except BreakException:
            pass
        return

    def get_samples_theta(self):
        """ Send the Theta sampling history back to root node """
        theta = self.chain.invprobit(self.chain.s_theta[1:])
        sent = self.comm.Isend([theta, MPI.FLOAT], dest = 0)
        sent.wait()
        return

    def get_samples_sigma2_hb(self):
        """ Send the Hoppy Bar sigma2 back to root node """
        sigma2 = self.chain.s_s2_hb[1:]
        sent = self.comm.Isend([sigma2, MPI.FLOAT], dest = 0)
        sent.wait()
        return

    def get_samples_sigma2_tc(self):
        """ Send the Taylor Cylinder sigma2 back to root node """
        sigma2 = self.chain.s_s2_tc[1:]
        sent = self.comm.Isend([sigma2, MPI.FLOAT], dest = 0)
        sent.wait()
        return

    def get_samples_sigma2_fp(self):
        """ Send the Flyer plate sigma2 back to the root node """
        sigma2 = self.chain.s_s2_fp[1:]
        sent = self.comm.Isend([sigma2, MPI.FLOAT], dest = 0)
        sent.wait()
        return

    def dispatch(self, parcel):
        """ parcel is expected to be a list; the first element of the list is
        an instruction, and the second element is something related to that
        instruction. """
        if   parcel[0] == 'initialize_chain':
            self.chain = RemoteChain(**parcel[1])
            self.return_true()
            return
        elif parcel[0] == 'set_temperature':
            self.chain.set_temperature(parcel[1])
            self.return_true()
            return
        elif parcel[0] == 'initialize_sampler':
            self.chain.initialize_sampler(parcel[1])
            self.return_true()
            return
        elif parcel[0] == 'sample':
            self.chain.sample(parcel[1])
            self.return_true()
            return
        elif parcel[0] == 'get_state':
            self.return_value(self.chain.get_state())
            return
        elif parcel[0] == 'set_state':
            self.chain.set_state(parcel[1])
            return
        elif parcel[0] == 'get_accept_prob':
            self.return_value(self.chain.get_accept_probability())
            return
        elif parcel[0] == 'get_history':
            history = self.chain.get_history(*parcel[1])
            self.return_value(history)
            return
        elif parcel[0] == 'get_samples_theta':
            self.get_samples_theta()
            return
        elif parcel[0] == 'get_samples_sigma2_hb':
            self.get_samples_sigma2_hb()
            return
        elif parcel[0] == 'get_samples_sigma2_tc':
            self.get_samples_sigma2_tc()
            return
        elif parcel[0] == 'get_samples_sigma2_fp':
            self.get_samples_sigma2_fp()
            return
        elif parcel[0] == 'write_to_disk':
            self.chain.write_to_disk()
            self.return_true()
            return
        elif parcel[0] == 'break':
            raise BreakException
        else:
            raise ValueError('Must Provide one of the approved arguments!')

if __name__ == '__main__':
    pass

# EOF
