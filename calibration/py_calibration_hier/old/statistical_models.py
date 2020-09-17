#import cProfile
from physical_models import *
import numpy as np
import pandas as pd
#import ipdb
import time
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from scipy.stats import invgamma, multivariate_normal as mvnorm
from numpy.linalg import cholesky
from numpy.random import normal, uniform
from random import sample
from math import ceil, sqrt, pi, log
from itertools import combinations
from pointcloud import localssq
import warnings
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

#import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sea
sea.set(style = 'ticks')

class StatisticalModel(object):
    parameter_order = []

    def parameter_trace_plot(self, sample_parameters):
        palette = plt.get_cmap('Set1')
        if len(sample_parameters.shape) == 1:
            n = sample_parameters.shape[0]
            plt.plot(range(n), sample_parameters, marker = '', linewidth = 1)
        else:
            # df = pd.DataFrame(sample_parameters, self.parameter_order)
            n, d = sample_parameters.shape
            for i in range(d):
                plt.subplot(d, 1, i+1)
                plt.plot(range(n), sample_parameters[:,i], marker = '',
                        color = palette(i), linewidth = 1)
                ax = plt.gca()
                ax.set_ylim(0,1)
        plt.show()
        return

    def parameter_pairwise_plot(self, sample_parameters):
        def off_diag(x, y, **kwargs):
            plt.scatter(x, y, **kwargs)
            ax = plt.gca()
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            return

        #def on_diag(x, **kwargs):
        #    sea.distplot(x, bins = 20, **kwargs)
        #    ax = plt.gca()
        #    ax.set_xlim(0,1)
        #    return
        def on_diag(x, **kwargs):
            plt.hist(x, **kwargs)
            ax = plt.gca()
            ax.set_xlim(0,1)
            return

        d = sample_parameters.shape[1]
        df = pd.DataFrame(sample_parameters, columns = self.parameter_order)
        g = sea.PairGrid(df)
        g = g.map_offdiag(off_diag, s = 1)
        g = g.map_diag(on_diag)
        for i in range(d):
            g.axes[i,i].annotate(self.parameter_order[i], xy = (0.05, 0.9))
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')

        plt.show()
        return

class StatisticalSubModel(object):
    """
    The sub-model computes sum-squared-error (sse) for a given set of parameters.
    So, taking the set of parameters, it builds the stress/strain curve, computes
    an interpolation of the stress strain curve at the supplied data points, and
    then returns the sum of squared error from the supplied data values to the
    interpolated values.

    I'm expecting that this class can be sub-classed to add in functionality for
    emulators outside of Hopkinson-Bar data.   Everything should be opaque to the
    StatisticalModel class, which just expects to use the sse method, and the
    N value.
    """
    def load_data(self, data, temp):
        """
        Loads the data frame into the model
        """
        self.X = data[:,0]
        self.Y = data[:,1]
        self.N = len(self.X)
        self.temp = temp
        return

    def get_state_history(self, parameters, starting_temp):
        """
        Compute State History based on supplied parameters, starting temp.
        First, initializes the state with the supplied starting temp.
        Then supplies the given parameters to the model.
        Then generates the state history.  Returns columns [strain, stress].
        """
        self.model.initialize_state(T = starting_temp)
        self.model.update_parameters(parameters)
        
        try:
            res = self.model.compute_state_history(self.shist)
        except ConstraintError:
            res = np.append(self.shist[:,1,None],np.full([len(self.shist),1],-1000.),1) # -1000 if constraint, violated, will cause rejection in M-h
            return res
            
        return res[:,1:3]

    def prediction(self, parameters, x_new):
        """
        Generates the stress-strain curve at the supplied parameters, temp.\
        Interpolates the stress strain curve at the data values using a cubic
        spline interpolation.
        """
        model_curve = self.get_state_history(parameters, self.temp)
        est_curve = interp1d(model_curve[:,0], model_curve[:,1], kind = 'cubic')
        return est_curve(x_new)

    def prediction_plot(self, parameters, sigma2):
        n_curves = parameters.shape[0]
        emax, edot, Nhist = self.model.report_history_variables()
        result_y = np.empty((Nhist, n_curves))
        result_x = np.linspace(0., emax, Nhist)

        msigma = np.sqrt(sigma2.mean())

        for i in range(n_curves):
            if i == 0:
                temp = self.get_state_history(parameters[i,:], self.temp)
                result_y[:,i] = temp[:,1]
                result_x = temp[:,0]
            else:
                result_y[:,i] = self.get_state_history(parameters[i,:], self.temp)[:,1]

        for i in range(n_curves):
            plt.plot(result_x, result_y[:,i], color = 'gray',
                        linestyle = '-', linewidth = 0.25,)

        plt.plot(self.X, self.Y, 'bo')
        #plt.ylim(0.002,0.8)
        plt.errorbar(self.X, self.Y, fmt = 'none', yerr = msigma)
        ax = plt.gca()
        plot_text = '$T_0$ = {} °K\n ε̇ = {}/s'.format(int(self.temp),int(self.edot*1.e6))
        plt.text(0.65,0.12, plot_text, transform = ax.transAxes)
        return

    def prediction_plot2(self, parameters, sigma2):
        n_curves = parameters.shape[0]
        emax, edot, Nhist = self.model.get_history_variables()
        result_y = np.empty((Nhist, n_curves))
        result_x = np.linspace(0., emax, Nhist)

        msigma = np.sqrt(sigma2.mean())

        for i in range(n_curves):
            if i == 0:
                temp = self.get_state_history(parameters[i,:], self.temp)
                result_y[:,i] = temp[:,1]
                result_x = temp[:,0]
            else:
                result_y[:,i] = self.get_state_history(parameters[i,:], self.temp)[:,1]

        quantiles = np.linspace(0.05,0.95,19)
        qcurves = np.quantile(result_y, quantiles, axis = 1).T

        for i in range(9):
            alpha = 1. - 2*(quantiles[i] - 0.5)**2
            plt.plot(result_x, qcurves[:,i], alpha = alpha, color = 'gray',
                        linestyle = '-', linewidth = 0.25,)

        plt.plot(self.X, self.Y, 'bo')
        #plt.ylim(0.002,0.8)
        plt.errorbar(self.X, self.Y, fmt = 'none', yerr = msigma)
        ax = plt.gca()
        plot_text = '$T_0$ = {} °K\n ε̇ = {}/s'.format(int(self.temp),int(self.edot*1.e6))
        plt.text(0.65,0.12, plot_text, transform = ax.transAxes)
        return

    def sse(self, parameters):
        """
        Computes predictions for stress at supplied strain values, then sums
        the squared difference between measured stress and estimated stress.
        """
        preds = self.prediction(parameters, self.X)
        sse = (self.Y - preds).dot(self.Y - preds)
        #sum(np.power((self.Y - preds),2.))
        return sse

    def initialize_model(self, parameters, constants):
        self.model.initialize(parameters, constants)
        return

    def __init__(self, transport, **kwargs):
        """
        Already described what "StatisticalSubModel" is.
        Inputs:-----
        - transport: Object of Transport class.  this supplies data, starting
        temperature, etc.
        - **kwargs - Additional inputs concerning the MaterialModel.  E.g.,
            flow_stress_model = PTW_Yield_Stress
        """
        self.model = MaterialModel(**kwargs)
        self.parameter_order = self.model.get_parameter_list()
        self.model.set_history_variables(transport.emax, transport.edot, transport.Nhist)
        self.load_data(transport.data, transport.temp)
        self.edot = transport.edot
        self.shist = generate_strain_history(transport.emax, transport.edot, transport.Nhist)
        return

class MetropolisHastingsModel(StatisticalModel):
    """
    Structurally, StatisticalModel is the parent of StatisticalSubModel.

    All sampling, manipulation of parameters, etc. happens here.
    """
    s2_alpha = 50.    # 0 # 2
    s2_beta  = 5.e-6  # 0 # 0.001

    def normalize(self, x):
        """
        Normalize transforms the variable x to be bounded between 0 and 1,
        based on (x - min) / range(x)
        """
        return (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def unnormalize(self, z):
        """
        Unnormalize transforms variable (bounded between 0:1) back to its
        original scale.
        """
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
        return - y - 2 * np.log(1. + np.exp(-y))

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
        """
        Computes the log jacobian for the inverse probit transformation
        """
        return -0.5 * np.log(2 * pi) - y * y / 2.

    def sse(self, parameters):
        """
        Sums the sum squared errors from each of the component models.

        Inputs have been scaled to real values
        """
        return sum([submodel.sse(parameters) for submodel in self.submodels])

    def scaled_sse(self, normalized_parameters):
        """
        Unnormalizes the inputs, then computes the sum of squared errors
        """
        parameters = self.unnormalize(normalized_parameters)
        return self.sse(parameters)

    def probit_sse(self, probit_parameters):
        normalized_parameters = self.invprobit(probit_parameters)
        return(self.scaled_sse(normalized_parameters))

    def logit_sse(self, logit_parameters):
        normalized_parameters = self.invlogit(logit_parameters)
        return(self.scaled_sse(normalized_parameters))

    def log_posterior(self, sse, logdetjac, sigma2):
        lp = (
            - (0.5 * self.N + self.s2_alpha + 1) * np.log(sigma2)
            - (0.5 * sse + self.s2_beta) / sigma2
            + logdetjac
            )
        return lp

    def sampler(self, ns):
        """
        Metropolis within Gibbs MCMC sampler
        -- Physical parameters modelled with a metropolis hastings step
           using a multivariate normal proposal.  Parameters are real valued
           (as the parameters have been transformed to be on the real line)
        -- Sigma^2 follows a standard Gibbs step.
        """
        d = len(self.parameter_order)
        s_sigma2 = np.empty(ns)
        s_theta  = np.empty((ns, d))
        s_theta[0,:] = 0
        s_sigma2[0]  = 0.001

        log_sum_cov = np.empty(ns)
        eps = 1.e-9

        #Si = np.eye(d) * eps * 5.76 / d
        #S = cholesky(Si)
        log_sum_cov[0] = 0. #log(abs(S).sum() + 1.)


        print('Beginning Sampling\n---------------------')

        # finish this part for tempering

        curr_sse = self.probit_sse(s_theta[0,:])
        curr_ldj = sum(self.invprobitlogjac(s_theta[0,:]))

        accept = 0
        for i in range(1,ns):
            s_sigma2[i] = invgamma.rvs(
                self.N / 2. + self.s2_alpha,
                scale = curr_sse / 2. + self.s2_beta,
                )
            try:
                lssq, ns = localssq(s_theta[:i,], s_theta[i-1,:])
            except FloatingPointError:
                lssq, ns = np.zeros(d), 0.
            S = cholesky((lssq + diag(d) * 1.e-7)/(ns + 300 - d - 1))
                #S = cholesky((np.cov(s_theta[:i,:].T) + Si) * 5.76 / d)
            log_sum_cov[i] = log(abs(S).sum() + 1.)

            proposal = s_theta[i-1,:] + normal(size = d).dot(S)

            prop_sse = self.probit_sse(proposal)
            prop_ldj = self.invprobitlogjac(proposal).sum()
            prop_lp  = self.log_posterior(prop_sse, prop_ldj, s_sigma2[i])
            curr_lp  = self.log_posterior(curr_sse, curr_ldj, s_sigma2[i])

            if np.log(uniform()) < (prop_lp - curr_lp):
                accept += 1
                s_theta[i,:] = proposal
                curr_sse, curr_ldj = prop_sse, prop_ldj
            else:
                s_theta[i,:] = s_theta[i-1,:]

            if i % 50 == 0:
                print('\rSampling {:.1%} Done'.format(i/ns), end = '')
        print('\rSampling 100.0% Done\n---------------------\nSampling Completed')
        print('{:.4f} Sampling Efficiency Achieved'.format(accept/ns))
        return s_theta, s_sigma2, log_sum_cov

    def sample(self, nsamp, nburn, thin = 1):
        result_theta, result_sigma2, log_sum_cov = self.sampler(nsamp + nburn)
        result_param = self.invprobit(result_theta)
        return result_param[nburn::thin,:], result_sigma2[nburn::thin], log_sum_cov

    def initialize_model(self, parameters, constants):
        for submodel in self.submodels:
            submodel.initialize_model(parameters, constants)
        return

    def prediction_plot(self, normalized_parameters, sigma2, ylim = (0.,0.04)):
        parameters = np.apply_along_axis(self.unnormalize, 1, normalized_parameters)
        plot_count = len(self.submodels)
        d_across = ceil(sqrt(plot_count))
        d_down   = ceil(float(plot_count) / d_across)
        k = 0
        for i in range(d_across):
            for j in range(d_down):
                k += 1
                if k <= len(self.submodels):
                    plt.subplot(d_across, d_down, k)
                    self.submodels[k-1].prediction_plot2(parameters, sigma2)
        plt.show()
        return

    def __init__(self, transports, bounds, constants, params = None, **kwargs):
        self.submodels = [
                StatisticalSubModel(
                        transport,
                        **kwargs
                        )
                for transport in transports
                ]
        self.parameter_order = self.submodels[0].parameter_order
        self.N = sum([submodel.N for submodel in self.submodels])
        self.bounds = np.array([bounds[key] for key in self.parameter_order])

        if type(params) is dict:
            pass
        else:
            params = {
                x:y
                for x,y in zip(
                    self.parameter_order,
                    self.unnormalize(np.array([0.5] * len(self.parameter_order)))
                    )
                }
        self.initialize_model(params, constants)
        return

class Chain(MetropolisHastingsModel):
    n_accept        = 0
    n_try           = 0
    n_swaps         = 0
    n_swap_attempts = 0
    nu              = None
    psi             = None

    def log_posterior(self, sse, logdetjac, sigma2):
        lp = (
            - (0.5 * self.inv_temper_temp * self.N + self.s2_alpha + 1) * np.log(sigma2)
            - (0.5 * self.inv_temper_temp * sse + self.s2_beta) / sigma2
            + self.inv_temper_temp * logdetjac
            )
        return lp

    def sampler(self):
        raise NotImplementedError('Chain.sampler has been deprecated')

    def sample(self):
        raise NotImplementedError('Chain.sample has been deprecated')

    def draw(self, iter):
        self.n_try += 1
        alpha = 0.5 * self.inv_temper_temp * self.N + self.s2_alpha
        beta  = 0.5 * self.inv_temper_temp * self.curr_sse + self.s2_beta
        self.s_sigma2[iter] = invgamma.rvs(alpha, scale = beta)
        if iter > 1:
            res = localssq(self.s_theta[:iter-1], self.s_theta[iter-1], self.radius)
            lssq, ns = res.mat, res.n
        else:
            lssq, ns = np.zeros((self.d,self.d)), 0.
        curr_cov = (lssq + self.psi) / (ns + self.nu - self.d - 1)
        S = cholesky(curr_cov)
        self.log_sum_cov[iter] = log(abs(S).sum() + 1)

        proposal = self.s_theta[iter - 1] + normal(size = self.d).dot(S)
        if iter > 1:
            pres = localssq(self.s_theta[:iter-1], proposal, self.radius)
            plssq, pns = pres.mat, pres.n
        else:
            plssq, pns = np.zeros((self.d,self.d)), 0.
        prop_cov = (plssq + self.psi) / (pns + self.nu - self.d - 1)
        prop_sse = self.probit_sse(proposal)
        prop_ldj = self.invprobitlogjac(proposal).sum()
        self.curr_lp = self.log_posterior(
                self.curr_sse, self.curr_ldj, self.s_sigma2[iter],
                )
        prop_lp = self.log_posterior(
                prop_sse, prop_ldj, self.s_sigma2[iter],
                )

        cp_ld = mvnorm.logpdf(proposal, self.s_theta[iter-1], curr_cov)
        pc_ld = mvnorm.logpdf(self.s_theta[iter-1], proposal, prop_cov)

        if np.log(uniform()) < (prop_lp  + pc_ld - self.curr_lp - cp_ld):
            self.n_accept += 1
            self.s_theta[iter,:] = proposal
            self.curr_sse = prop_sse
            self.curr_ldj = prop_ldj
            self.curr_lp  = prop_lp
            self.accepted[iter] = 1.
        else:
            self.s_theta[iter,:] = self.s_theta[iter - 1,:]
        return

    def init_sampler(self, ns):
        temp_theta  = self.s_theta[-1,:]
        temp_sigma2 = self.s_sigma2[-1]

        self.s_theta  = np.empty((ns, self.d))
        self.s_sigma2 = np.empty(ns)
        self.accepted = np.zeros(ns)

        self.s_theta[0,:] = temp_theta
        self.s_sigma2[0]  = temp_sigma2

        self.log_sum_cov = np.empty(ns)
        self.log_sum_cov[0] = 0
        return

    def __init__(self, parent = None, temper_temp = 1, params = None,
                    nu = 200, psi_diag = 1.e-7, base_radius = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.inv_temper_temp = 1. / temper_temp
        self.d = len(self.parameter_order)
        self.nu = nu
        self.psi = np.eye(self.d) * psi_diag
        self.s_theta = np.empty((1, self.d))
        self.s_sigma2 = np.empty(1)

        if type(params) is dict:
            self.s_theta[0,:] = np.array([
                    params[key]
                    for key in self.parameter_order
                    ])
        elif type(params) is np.ndarray:
            self.s_theta[0,:] = params
        else:
            self.s_theta[0,:] = 0

        self.radius = base_radius * log(temper_temp + 1, 10)

        self.s_sigma2[0]  = 0.01
        self.curr_sse = self.probit_sse(self.s_theta[0,:])
        self.curr_ldj = self.invprobitlogjac(self.s_theta[0,:]).sum()
        self.curr_lp  = self.log_posterior(
                self.curr_sse, self.curr_ldj, self.s_sigma2[0],
                )
        return

class ParallelTemperingModel(StatisticalModel):
    def sampler(self, nsamp, nburn, tburn):
        """
        Implements the Parallel Tempering sampler.
        """
        ns = nsamp + nburn + tburn

        n_chains = len(self.chains)

        for chain in self.chains:
            chain.init_sampler(ns)

        tstart = time.time()

        n_swap_attempts = self.NChains - 1

        print('Beginning Sampling\n---------------------')
        for i in range(1, ns):
            if i % 50 == 0:
                print('\rSampling {:.1%} Done'.format(i/ns), end = '')

            for chain in self.chains:
                chain.draw(i)

            if (i > tburn):
                candidate_swaps = sample(self.possible_swaps, n_swap_attempts)
                for s1,s2 in candidate_swaps:
                    self.try_exchange_state(s1, s2, i)

        print('\rSampling 100.0% Done\n---------------------\nSampling Completed')
        tend = time.time()
        print('{:.2f} Hours Taken'.format((tend - tstart)/3600))
        return

    def log_posterior(self, sse, n, sigma2, ldj):
        lp = (
            - (0.5 * n + self.s2_alpha + 1) * log(sigma2)
            - (0.5 * sse + self.s2_beta) / sigma2
            + ldj
            )
        return lp

    def try_exchange_state(self, ca, cb, i):
        """

        """
        c1 = self.chains[ca]; c2 = self.chains[cb]

        temp_delta = c2.inv_temper_temp - c1.inv_temper_temp

        c1_lp = self.log_posterior(c1.curr_sse, c1.N, c1.s_sigma2[i], c1.curr_ldj)
        c2_lp = self.log_posterior(c2.curr_sse, c2.N, c2.s_sigma2[i], c2.curr_ldj)

        c1.n_swap_attempts += 1
        c2.n_swap_attempts += 1

        alpha = temp_delta * (c1_lp - c2_lp)

        if np.log(uniform()) < alpha:
            self.swapped.append((ca, cb, i))

            c1.s_theta[i,:], c2.s_theta[i,:] = \
                    c2.s_theta[i,:].copy(), c1.s_theta[i,:].copy()
            c1.s_sigma2[i],  c2.s_sigma2[i]  = \
                    c2.s_sigma2[i].copy(),  c1.s_sigma2[i].copy()

            # Exchange current SSE, logdetJac
            c1.curr_sse, c2.curr_sse = c2.curr_sse, c1.curr_sse
            c1.curr_ldj, c2.curr_ldj = c2.curr_ldj, c1.curr_ldj

            # Calculate new log posteriors given current state
            c1.curr_lp = c1.log_posterior(c1.curr_sse, c1.curr_ldj, c1.s_sigma2[i])
            c2.curr_lp = c2.log_posterior(c2.curr_sse, c2.curr_ldj, c2.s_sigma2[i])

            # iterate number of swaps
            c1.n_swaps += 1
            c2.n_swaps += 1
        else:
            self.failed_swap.append((ca,cb,i))
        return

    def plot_swap_matrix(self):
        swaps = np.zeros((self.NChains,self.NChains))
        failed_swaps = np.zeros((self.NChains,self.NChains))
        for a, b, i in self.swapped:
            swaps[a,b] += 1
            swaps[b,a] += 1

        for a, b, i in self.failed_swap:
            failed_swaps[a,b] += 1
            failed_swaps[b,a] += 1

        swap_prob =  swaps / (swaps + failed_swaps  + np.eye(swaps.shape[0]))
        plt.matshow(swap_prob)
        plt.colorbar()
        plt.show()
        return

    def plot_accept_prob(self):
        accept_prob = self.get_accept_prob()
        idx = range(len(accept_prob))
        temps = ['{:.4g}'.format(1/chain.inv_temper_temp) for chain in self.chains]
        plt.bar(idx, height = accept_prob, tick_label = temps)
        plt.xticks(rotation = 60)
        plt.title('Acceptance Probability by Chain')
        plt.show()
        return

    def get_swap_prob(self,):
        swaps = [chain.n_swaps for chain in self.chains]
        swap_attempts = self.chains[0].n_swap_attempts
        swaps_up = list()
        swaps_up.append(swaps[0])
        for i in range(1, len(swaps)):
            swaps_up.append(swaps[i] - swaps_up[i-1])
        return np.array(swaps_up[:-1]) / swap_attempts

    def get_accept_prob(self,):
        return [float(chain.n_accept) / chain.n_try for chain in self.chains]

    def sample(self,  nsamp,  nburn, tburn, thin = 1):
        self.sampler(nsamp, nburn, tburn)
        s_theta  = self.invprobit(self.chains[0].s_theta[(nburn + tburn)::thin,:])
        s_sigma2 = self.chains[0].s_sigma2[(nburn + tburn)::thin]
        history = [chain.s_theta for chain in self.chains]
        return s_theta, s_sigma2, history

    def __init__(self, temp_ladder = np.array([1.,]), cpus = 4, **kwargs):
        #self.pool   = mp.Pool(processes = cpus)
        self.chains = [
                Chain(self, temper_temp = temp, **kwargs)
                for temp in temp_ladder
                ]
        self.probit          = Chain.probit
        self.logit           = Chain.logit
        self.invprobit       = Chain.invprobit
        self.invlogit        = Chain.invlogit
        self.invlogitlogjac  = Chain.invlogitlogjac
        self.invprobitlogjac = Chain.invprobitlogjac

        self.NChains         = len(self.chains)
        self.possible_swaps  = list(combinations(range(self.NChains), 2))
        self.N               = self.chains[0].N
        self.s2_alpha        = self.chains[0].s2_alpha
        self.s2_beta         = self.chains[0].s2_beta
        self.parameter_order = self.chains[0].parameter_order
        self.prediction_plot = self.chains[0].prediction_plot
        self.swapped         = []
        self.failed_swap     = []
        return

class Transport(object):
    def __init__(self, data, temp, emax, edot, Nhist):
        self.data = data
        self.temp = temp
        self.emax = emax
        self.edot = edot
        self.Nhist = Nhist
        return

def import_strain_curve(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pd.read_csv(path, skiprows = range(0,17)).values
    data[:,1] = data[:,1] * 1.e-5
    return data[1:,:]

if __name__ == '__main__':
    pass

# EOF
