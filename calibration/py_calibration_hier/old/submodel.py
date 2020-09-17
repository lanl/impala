"""
submodel.py

Submodels create a structure around a particular experiment.  E.g., Hoppy bar
at a particular temperature and strain rate.

They define a few particular methods:

predict - provides a prediction based on input parameters
    sse - sum squared error based on input parameters
      N - number of observations
"""
import numpy as np
import pandas as pd

from physical_models_c import *
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from math import ceil, sqrt, pi, log
from numpy.linalg import svd, multi_dot
from scipy.linalg import cho_solve, cho_factor

import GPy
from GPy.models import GPCoregionalizedRegression
from GPy.util.multioutput import LCM, ICM
from GPy.models.gp_regression import GPRegression
from GPy.core.parameterization.priors import Gamma, Gaussian
from GPy.kern import Matern32, Matern52, RBF, Bias, Linear, Coregionalize
from GPy.inference.mcmc import HMC
from numpy.random import normal, choice

import pyBASS

from methodtools import lru_cache

import matplotlib.pyplot as plt
import os, sys

class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# SubModels

class SubModelBase(object):
    """
    SubModels hold the results of an experiment, and calculate sum-squared
    errors.  Each experiment gets its own submodel.

    mat_para - Material Parameters that are consistent across experiments.
    exp-para - Experiment Parameters that are unique to each experiment.
    pidx - The sampler samples a vector.  This specifies what indices of that
            vector pertain to this particular submodel.
    """
    mat_para = []
    exp_para = []
    pidx = []
    curr_lcd = 0.

    def initialize_submodel(self, parameters, constants):
        return

    def lcd(self, *args):
        return 0.

    def sse(self, *args):
        return 0.

    def check_constraints(self):
        return True

    def __init__(self, transport, **kwargs):
        return

class SubModelHB(SubModelBase):
    """
    The sub-model computes sum-squared-error (sse) for a given set of parameters.
    So, taking the set of parameters, it builds the stress/strain curve, computes
    an interpolation of the stress strain curve at the supplied data points, and
    then returns the sum of squared error from the supplied data values to the
    interpolated values.
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
        res = self.model.compute_state_history()
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

    def prediction_plot(self, parameters, sigma2, ylim):
        """
        Generates quantile-ed prediction curves for the parameter sets.

        Over the top of the assembled prediction curves, plots the raw data with
        an error bar of +- 1 (mean) standard error, as computed from sigma2.
        """
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

        quantiles = np.linspace(0.05,0.95,19)
        qcurves = np.quantile(result_y, quantiles, axis = 1).T

        for i in range(9):
            alpha = 1. - 2*(quantiles[i] - 0.5)**2
            plt.plot(result_x, qcurves[:,i], alpha = alpha, color = 'gray',
                        linestyle = '-', linewidth = 0.25,)

        plt.plot(self.X, self.Y, 'bo')
        plt.ylim(*ylim)
        plt.errorbar(self.X, self.Y, fmt = 'none', yerr = msigma, alpha = 0.1)
        ax = plt.gca()
        plot_text = '$T_0$ = {} °K\n ε̇ = {:.1e}'.format(int(self.temp),self.edot)
        plt.text(0.65,0.12, plot_text, transform = ax.transAxes)
        return

    def prediction_plot2(self, parameters, sigma2, ylim):
        pass

    @lru_cache(maxsize = 12)
    def sse(self, parameters):
        """
        Computes predictions for stress at supplied strain values, then sums
        the squared difference between measured stress and estimated stress.
        """
        preds = self.prediction(np.array(parameters), self.X)
        diffs = self.Y - preds
        sse   = (diffs * diffs).sum()
        return sse

    def initialize_submodel(self, parameters, constants):
        self.model.initialize(parameters, constants)
        return

    def check_constraints(self):
        return self.model.check_constraints()

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
        return

class SubModelHBAR1(SubModelHB):
    """
    Extends the Hopkinson Bar SubModel to have AR1 (autoregressive, order 1)
    structured errors.
    """
    mat_para = ['phi']

    def invcovmat(self, phi):
        """ Inverse Covariance Matrix for AR1 Error structure """
        n = self.Y.shape[0]
        D = np.eye(n) + np.diag(np.array([0.] + [phi*phi]*(n-2) + [0.]))
        return D + np.diag([-phi]*(n-1), 1) + np.diag([-phi]*(n-1), -1)

    def lcd(self, phi):
        """ Log Determinant of Covariance Matrix for AR1 Error structure """
        try:
            lcd = - log(1 - phi * phi)
        except ValueError:
            print(phi)
            raise
        return lcd

    def sse(self, parameters):
        """ Sum-squared errors (accounting for AR1 Error structure) """
        if parameters == stored_parameters:
            return self.stored_sse

        self.curr_lcd = self.lcd(parameters[-1])
        preds = self.prediction(parameters[:-1], self.X)
        diffs = self.Y - preds
        return diffs @ self.invcovmat(parameters[-1]) @ diffs

class SubModelGP(SubModelBase):
    """
    GPy based emulator of Taylor Cylinder Simulation Results.

    The addl_params list includes velocity, and shear modulus.

    Taylor Cylinder simulation inputs and outputs are standardized, then pushed
    to Gaussian process object.  Prediction generates a new output based on
    predicted mean and variance at the (new) unsimulated point.
    """
    mat_para = ['sm0']
    exp_para = ['vel']
    d1 = None       # input dimension
    d2 = None       # output dimension
    X  = None       # Raw source data
    Xmean = None; Xsd = None
    W  = None       # standardized source data
    Y  = None       # Raw output data
    Ymean = None; Ysd = None
    Z  = None       # Standardized Output Data
    Y_actual = None # Raw output data
    Z_actual = None # Standardized output data
    model = None    # GPy Model

    @property
    def N(self):
        """ Number of Observations """
        return np.prod(self.Y_actual.shape)

    def sse(self, param):
        """ Compute Sum Squared Error between actual data and predicted """
        predicted = self.predict(param)
        diff = (self.Y_actual - predicted).reshape(-1)
        return diff.dot(diff)

    def predict(self, param):
        """ Create a random vector centered around the prediction with the
        model variance at that prediction. """
        mu, sigma2 = self.model.predict(self.standardize_x(param).reshape(1,-1))
        mu = mu[0]; sd = sqrt(sigma2[0][0])
        return self.unstandardize(mu + normal(size = self.d2) * sd)

    def standardize(self, Y):
        """ scale the output """
        return (Y - self.Ymean) / self.Ysd

    def unstandardize(self, Z):
        """ unscale the output """
        return Z * self.Ysd + self.Ymean

    def standardize_x(self, X):
        """ scale the input """
        return (X - self.Xmean) / self.Xsd

    def unstandardize_x(self, W):
        """ unscale the input """
        return W * self.Xsd + self.Xmean

    def __init__(self, transport, **kwargs):
        model = MaterialModel(**kwargs)
        self.parameter_order = model.get_parameter_list()
        # Parse the inputs
        self.X = transport.X
        self.Y = transport.Y
        self.Xmean, self.Xsd = np.mean(self.X, axis = 0), np.std(self.X, axis = 0)
        self.Ymean, self.Ysd = np.mean(self.Y, axis = 0), np.std(self.Y, axis = 0)
        self.W = self.standardize_x(self.X)
        self.Z = self.standardize(self.Y)
        self.Y_actual = transport.Y_actual
        self.Z_actual = self.standardize(self.Y_actual)

        # Set dimensions
        self.d1 = self.X.shape[1]
        self.d2 = self.Y.shape[1]

        # Build the model
        kernel = RBF(input_dim = self.d1, ARD = True)

        # Build the model, and get MAP estimates
        self.model = GPRegression(self.W, self.Z, kernel = kernel)
        with HiddenPrint():
            self.model.kern.lengthscale[:].set_prior(Gamma.from_EV(1.,10.))
            self.model.kern.variance[:].set_prior(Gamma.from_EV(1.,10.))
            self.model.likelihood.variance[:].set_prior(Gamma.from_EV(1.,10))
            self.model.optimize()
        return

class SubModelBASS(SubModelBase):
    """
    BASS emulator of flyer plate or taylor cylinder simulation results, without warping.
    """
    def __init__(self, transport, **kwargs):
        model = MaterialModel(**kwargs)
        self.parameter_order = model.get_parameter_list() # not sure how to use this to change order of transport.X
        # Parse the inputs
        self.X        = transport.X
        self.Y        = transport.Y
        self.Y_actual = transport.Y_actual
        # use as many PCs as it takes to explain 99.9 percent of the variance
        self.model    = pyBASS.bassPCA(self.X, self.Y, percVar = 99.9,
                                        ncores = os.cpu_count())
        return


    def sse(self, param):
        """ Compute Sum Squared Error between actual data and predicted """
        predicted = self.model.predict(param, nugget = True,
                mcmc_use = choice(range(self.model.bm_list[0].nstore)))
                # randomly select a mcmc iteration
        diff = (self.Y_actual - predicted).reshape(-1)
        return diff.dot(diff)

class SubModelPE(SubModelBase):
    """
    GPy based emulator of simulator results

    Simulation inputs and outputs are standardized.  Standardized outputs (Z)
    are decomposed by SVD. Z = USV^t

    Then a (k-rank) approximation of Z is calculated as: Z_k = U_k S_k V_k^t
    where k = min_i (cumsum(S[1:i]^2) / sum(S^2) > 0.99), Then W_k = U_k x S_k

    This SubModel models each dimension of W_k with independent Gaussian process.
    """
    mat_para = ['sm0']
    exp_para = ['vel']

    lengthprior = (1., 10.)
    varprior    = (1., 10.)

    @property
    def N(self):
        return np.prod(self.Y_actual.shape)

    def sse(self, param):
        """ Squared error (keeps separate different parts for different error
        terms) """
        diff = self.Y_actual - self.predict(param).reshape(1,-1)
        return (diff * diff).reshape(-1)

    def predict(self, param):
        """ Creates a random predicted output given inputs """
        pred = [gp.predict(self.standardize_x(param).reshape(1,-1)) for gp in self.GPs]
        mean = np.array([x[0][0][0] for x in pred])
        sd = np.sqrt(np.array([x[1][0][0] for x in pred]))
        wstar = normal(mean, sd)
        return self.unstandardize_y(wstar @ self.Vh[:self.rank,:])

    def prediction_plot(self, parameters, sigma2, rowcount, row):
        """ Creates a plot of posterior predictions, simulation outputs, and
        observed values for each of the output dimensions. """
        plot_start = (row - 1) * 3 + 1
        preds = np.apply_along_axis(self.predict, 1, parameters)

        ub = np.vstack((preds, self.Y)).max(axis = 0)
        lb = np.vstack((preds, self.Y)).min(axis = 0)

        bins = [np.linspace(x, y, 100) for x, y in zip(lb, ub)]

        for i in range(preds.shape[1]):
            #print(rowcount)
            #print(plot_start)
            #print(i)
            plt.subplot(3,rowcount, plot_start + i)
            plt.hist(self.Y[:,i], bins[i], alpha = 0.5,
                        density = True, label = 'simulated')
            plt.hist(preds[:,i], bins[i], alpha = 0.5,
                        density = True, label = 'posterior')
            plt.axvline(x = (self.Y_actual.reshape(-1))[i], label = 'observed')
            plt.legend()

        return

    def standardize_y(self, Y):
        return (Y - self.Ymean) / self.Ysd

    def unstandardize_y(self, Z):
        return Z * self.Ysd + self.Ymean

    def standardize_x(self, X):
        return (X - self.Xmean) / self.Xsd

    def unstandardize_x(self, W):
        return W * self.Xsd + self.Xmean

    def __init__(self, transport, **kwargs):
        model = MaterialModel(**kwargs)
        self.parameter_order = model.get_parameter_list()

        # Setup, passing transport contents
        self.X = transport.X
        self.Y = transport.Y
        self.Xmean, self.Xsd = np.mean(self.X, axis = 0), np.std(self.X, axis = 0)
        self.Ymean, self.Ysd = np.mean(self.Y, axis = 0), np.std(self.Y, axis = 0)
        self.A = self.standardize_x(self.X)
        self.Z = self.standardize_y(self.Y)

        # Singular Value Decomposition
        self.U, self.S, self.Vh = svd(self.Z)
        # Find Truncation Rank
        iter = 0
        S2 = self.S**2
        while iter < len(S2):
            if (S2[:iter].sum() / S2.sum()) > 0.99:
                break
            iter += 1

        # Find truncated outputs
        self.W = self.U[:,:iter] @ np.diag(self.S[:iter])
        self.rank = iter

        # Declare Independent Gaussian Processes for each component
        self.GPs = []
        with HiddenPrint():
            for j in range(iter):
                kernel = Matern32(input_dim = self.X.shape[1])
                model = GPRegression(self.A, self.W[:,j].reshape(-1,1), kernel = kernel)
                model.kern.lengthscale[:].set_prior(Gamma.from_EV(*self.lengthprior))
                model.likelihood.variance.set_prior(Gamma.from_EV(*self.varprior))
                model.optimize()
                self.GPs.append(model)
        return

class SubModelTC(SubModelPE):
    pass

class SubModelFP(SubModelPE):
    """
    Extends SubModelPE to reflect Flyer Plate
    """
    def sse(self, param):
        diff = self.Y_actual - self.predict(param).reshape(1,-1)
        diff2 = (diff * diff).reshape(-1)
        return np.array((diff2[::2].sum(), diff2[1::2].sum()))

    def prediction_plot(self, parameters, sigma2, rowcount, row):
        preds = np.apply_along_axis(self.predict, 1, parameters)

        ub = np.vstack((preds, self.Y)).max(axis = 0)
        lb = np.vstack((preds, self.Y)).min(axis = 0)
        bins = [np.linspace(x, y, 100) for x, y in zip(lb, ub)]

        t_pred = preds[:,::2]
        t_sims = self.Y[:,::2]
        t_obsv = (self.Y_actual.reshape(-1))[::2]
        t_bins = bins[::2]

        v_pred = preds[:,1::2]
        v_sims = self.Y[:,1::2]
        v_obsv  = (self.Y_actual.reshape(-1))[1::2]
        v_bins = bins[1::2]

        plot_start = row * 10 + 1
        for i in range(5):
            plt.subplot(5, rowcount, plot_start + i)
            plt.hist(t_sims[:,i], t_bins[i], density = True,
                        alpha = 0.5, label = 'simulated')
            plt.hist(t_pred[:,i], t_bins[i], density = True,
                        alpha = 0.5, label = 'posterior')
            plt.axvline(x = t_obsv[i], label = 'observed')
            plt.legend()

        plot_start += 5
        for i in range(5):
            plt.subplot(5, rowcount, plot_start + i)
            plt.hist(v_sims[:,i], v_bins[i], density = True,
                        alpha = 0.5, label = 'simulated')
            plt.hist(v_pred[:,i], v_bins[i], density = True,
                        alpha = 0.5, label = 'posterior')
            plt.avxline(x = v_obsv[i], label = 'observed')
            plt.legend()

        return

# EOF
