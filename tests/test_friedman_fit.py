from scipy.stats import kstest
from scipy.stats import uniform as Uniform
import numpy as np
from numpy.typing import ArrayLike
from impala import superCal as sc

class Friedman:
    def __init__(self, grid: ArrayLike):
        """
        grid: Grid at which Friedman function is evaluated.
        """
        self.grid = grid

    def __call__(self, theta: ArrayLike):
        """
        theta: Friedman function parameters.
        """
        return (
            10 * np.sin(np.pi * self.grid * theta[0]) +
            20 * (theta[1] - .5) ** 2 +
            10 * theta[2] +
            5 * theta[3]
        )

def generate_data(n_features: int, gridsize: int):
    assert n_features >= 4, "Friedman function has 4 parameters, so n_feature must be at least 4."

    grid = np.linspace(0, 1, gridsize)
    friedman = Friedman(grid)

    # True calibration parameter value. Note that the Friedman function uses
    # only the first four elements. So the remaining elements should have
    # uniform posteriors.
    theta = np.random.rand(1, n_features)

    # True observation error standard deviation.
    sigma= 0.1

    mu = np.apply_along_axis(friedman, 1, theta).squeeze()
    yobs = np.random.normal(mu, sigma)
    param_truth = dict(theta = theta, sigma = sigma)

    return yobs, friedman, n_features, param_truth

def test_friedman_fit():
    np.random.seed(0)
    yobs, friedman, n_features, param_truth = generate_data(n_features=6, gridsize=50)

    # Create bounds for each theta. Names (keys) need to be strings. 
    bounds = {
        str(index): np.array([0, 1])
        for index in range(n_features)
    }

    # Initialize with the only constraints being the bounds.
    setup = sc.CalibSetup(bounds, constraint_func='bounds')

    # Put the Friedman function into the right structure; this could be replaced
    # with an emulator.
    model = sc.ModelF(friedman, input_names=bounds.keys())

    # NOTE: If you have multiple experiments, just call setup.addVecExperiments
    # multiple times.
    setup.addVecExperiments(
        # observation vector
        yobs=yobs,
        # model that predicts a vector
        model=model,
        # yobs error standard deviation estimate (possibly a vector of estimates for
        # different parts of yobs vector).
        sd_est=[1.],
        # yobs error degrees of freedom (larger means more confidence in sd_est),
        # same shape as sd_est.
        s2_df=[0],
        # if sd_est is a vector of length 3, this is a vector of length len(yobs)
        # with values (0, 1, 2) indicating which sd_est corresponds to which part of
        # yobs.
        s2_ind=[0] * len(yobs)
    )

    # temperature ladder, typically (1 + step) ** np.arange(ntemps) 
    setup.setTemperatureLadder(1.05 ** np.arange(40))

    # MCMC number of iterations, and how often to take a decorrelation step.
    setup.setMCMC(nmcmc=15000, decor=100)

    # Pooled calibration (takes less than a minute).
    out = sc.calibPool(setup)

    # out.theta is has shape (num_mcmc_samples, num_temperatures, n_features).
    # Use index zero to get the posterior for the coldest chain (temperatre=0).
    burn = 6000  # burn-in: initial samples to discard.
    thin = 2  # thinning factor: take every 2 samples after burn-in.
    theta_posterior = out.theta[burn::thin, 0]

    # Number of parameters utilized by the Friedman function.
    num_utilized_parameters = 4

    # Test that the posterior for the superfluous parameters in theta are
    # uniform.
    superfluous_theta_posterior = theta_posterior[:, num_utilized_parameters:]
    for i, theta in enumerate(superfluous_theta_posterior.T):
        assert kstest(theta, Uniform().cdf).pvalue > 0.10, f"theta_{i} is not Uniform!"

    # Test that the posterior for the utilized parameters in theta are
    # not uniform.
    utilized_theta_posterior = theta_posterior[:, :num_utilized_parameters]
    for i, theta in enumerate(utilized_theta_posterior.T):
        assert kstest(theta, Uniform().cdf).pvalue < 0.01, f"theta_{i} is Uniform!"

    # Test that the true values of theta are in the 95% credible interval.
    lower = np.quantile(theta_posterior, .025, 0)
    upper = np.quantile(theta_posterior, .975, 0)
    assert np.alltrue(lower < param_truth['theta'])
    assert np.alltrue(param_truth['theta'] < upper)
