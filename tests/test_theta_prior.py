import numpy as np
import pytest
from numpy import ndarray
from scipy.stats import beta as Beta
from scipy.stats import multivariate_normal
from scipy.stats import norm as Normal

from impala import superCal as sc
from impala.superCal.impala_noprobit_emu import (
    eval_theta_priors,
    theta_log_prior,
)

if np.version<'2':
    np.trapezoid = np.trapz


class Line:
    """y = t_0 * grid + t_1, a model with an analytically obvious posterior."""

    def __init__(self, grid: ndarray) -> None:
        self.grid = grid

    def __call__(self, theta: ndarray) -> ndarray:
        return theta[0] * self.grid + theta[1]


def make_setup(bounds=None):
    if bounds is None:
        bounds = {"t_0": np.array([0.0, 1.0]), "t_1": np.array([0.0, 1.0])}
    return sc.CalibSetup(bounds, constraint_func="bounds")


def test_no_prior_is_zero_and_default_empty():
    setup = make_setup()
    assert setup.theta_prior == []
    theta = np.random.rand(5, 2)
    lp = theta_log_prior(setup, theta)
    assert lp.shape == (5,)
    assert np.all(lp == 0.0)


def test_independent_priors_match_scipy():
    bounds = {"t_0": np.array([0.0, 4.0]), "t_1": np.array([0.0, 1.0])}
    setup = make_setup(bounds)
    setup.addThetaPrior(
        dist="normal", params={"mean": 2.0, "sd": 0.5}, pname="t_0"
    )
    setup.addThetaPrior(
        dist="beta", params={"shape1": 2.0, "shape2": 5.0}, pname="t_1"
    )
    assert len(setup.theta_prior) == 2

    # unit-scaled draws, as held by the sampler
    theta = np.array([[0.25, 0.10], [0.50, 0.90], [0.75, 0.50]])
    # native scale, per the bounds above
    t0 = theta[:, 0] * 4.0
    t1 = theta[:, 1]

    expected = Normal(2.0, 0.5).logpdf(t0) + Beta(2.0, 5.0).logpdf(t1)
    np.testing.assert_allclose(theta_log_prior(setup, theta), expected)


@pytest.mark.parametrize(
    ("dist", "params"),
    [
        ("normal", {"mean": 0.5, "sd": 1.0}),
        ("lognormal", {"meanlog": 0.0, "sdlog": 1.0}),
        ("beta", {"shape1": 2.0, "shape2": 2.0}),
        ("uniform", {"min": 0.0, "max": 1.0}),
        ("gamma", {"shape": 2.0, "rate": 3.0}),
        ("cauchy", {"location": 0.5, "scale": 1.0}),
    ],
)
def test_all_supported_dists_are_finite_and_normalized(dist, params):
    setup = make_setup()
    setup.addThetaPrior(dist=dist, params=params, pname="t_0")
    theta = np.linspace(0.01, 0.99, 25)[:, None]
    theta = np.column_stack((theta, np.full_like(theta, 0.5)))
    lp = theta_log_prior(setup, theta)
    assert lp.shape == (25,)
    assert np.all(np.isfinite(lp))
    # crude check that the density integrates to <= 1 over [0, 1]
    assert np.trapezoid(np.exp(lp), theta[:, 0]) <= 1.0 + 1e-8


def test_joint_prior():
    setup = make_setup()

    def joint_prior(params):
        x = np.column_stack((params["t_0"], params["t_1"]))
        return multivariate_normal.logpdf(
            x, mean=[0.5, 0.5], cov=[[1.0, 0.5], [0.5, 1.0]]
        )

    setup.addJointThetaPrior(["t_0", "t_1"], joint_prior)

    theta = np.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.1]])
    expected = multivariate_normal.logpdf(
        theta, mean=[0.5, 0.5], cov=[[1.0, 0.5], [0.5, 1.0]]
    )
    np.testing.assert_allclose(theta_log_prior(setup, theta), expected)


def test_joint_and_independent_priors_add():
    setup = make_setup()
    setup.addThetaPrior(
        dist="normal", params={"mean": 0.5, "sd": 0.2}, pname="t_0"
    )
    setup.addJointThetaPrior(["t_0", "t_1"], lambda params: -1.0)

    theta = np.array([[0.4, 0.6], [0.5, 0.5]])
    expected = Normal(0.5, 0.2).logpdf(theta[:, 0]) - 1.0
    np.testing.assert_allclose(theta_log_prior(setup, theta), expected)


def test_scalar_joint_prior_broadcasts():
    setup = make_setup()
    setup.addJointThetaPrior(["t_0", "t_1"], lambda params: -2.5)
    lp = theta_log_prior(setup, np.random.rand(4, 2))
    np.testing.assert_allclose(lp, np.full(4, -2.5))


def test_out_of_support_is_neg_inf():
    setup = make_setup()
    setup.addThetaPrior(
        dist="gamma", params={"shape": 2.0, "rate": 1.0}, pname="t_0"
    )
    theta = np.array([[0.0, 0.5]])
    assert theta_log_prior(setup, theta)[0] == -np.inf


def test_eval_theta_priors_accepts_names():
    priors = [
        {
            "name": "t_0",
            "dist": "normal",
            "params": {
                "mean": 0.0,
                "sd": 1.0,
            },
        }
    ]
    theta = np.array([[0.0, 9.0], [1.0, 9.0]])
    lp = eval_theta_priors(theta, priors, tnames=["t_0", "t_1"])
    np.testing.assert_allclose(lp, Normal(0.0, 1.0).logpdf([0.0, 1.0]))


def test_bad_input_raises():
    setup = make_setup()
    with pytest.raises(ValueError):
        setup.addThetaPrior(dist="normal", params={"mean": 0, "sd": 1})
    with pytest.raises(ValueError):
        setup.addThetaPrior(
            dist="normal", params={"mean": 0, "sd": 1}, pname="nope"
        )
    with pytest.raises(ValueError):
        setup.addThetaPrior(dist="weibull", params={}, pname="t_0")
    with pytest.raises(ValueError):
        setup.addThetaPrior(dist="normal", params={"mean": 0}, pname="t_0")
    with pytest.raises(ValueError):
        setup.addJointThetaPrior(["t_0", "nope"], lambda params: 0.0)
    with pytest.raises(TypeError):
        setup.addJointThetaPrior(["t_0"], "not a function")
    with pytest.raises(ValueError):
        eval_theta_priors(
            {"t_0": np.zeros(3)},
            [{"name": "t_0", "dist": "weibull", "params": {}}],
        )


def build_line_setup(yobs, model, nmcmc=6000):
    bounds = {"t_0": np.array([0.0, 1.0]), "t_1": np.array([0.0, 1.0])}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=yobs,
        model=model,
        sd_est=[0.1],
        s2_df=[0],
        s2_ind=[0] * len(yobs),
    )
    setup.setTemperatureLadder(1.05 ** np.arange(8))
    setup.setMCMC(nmcmc=nmcmc, decor=100)
    return setup


def test_calibPool_prior_shifts_posterior():
    """A tight prior away from the MLE should pull the posterior toward it."""
    np.random.seed(0)
    grid = np.linspace(0, 1, 40)
    line = Line(grid)
    model = sc.ModelF(line, input_names=["t_0", "t_1"])
    truth = np.array([0.8, 0.2])
    yobs = np.random.normal(line(truth), 0.1)

    np.random.seed(1)
    out_flat = sc.calibPool(build_line_setup(yobs, model))

    np.random.seed(1)
    setup = build_line_setup(yobs, model)
    setup.addThetaPrior(
        dist="normal", params={"mean": 0.2, "sd": 0.02}, pname="t_0"
    )
    out_prior = sc.calibPool(setup)

    burn = 3000
    flat_mean = out_flat.theta[burn:, 0, 0].mean()
    prior_mean = out_prior.theta[burn:, 0, 0].mean()

    # without a prior the posterior sits near the truth
    assert abs(flat_mean - truth[0]) < 0.1
    # with a tight prior at 0.2 the posterior is pulled well away from it
    assert prior_mean < flat_mean - 0.2


def test_calibPool_recovers_prior_with_no_information():
    """With a flat likelihood, the posterior should look like the prior."""
    np.random.seed(0)
    grid = np.linspace(0, 1, 20)

    # model ignores theta entirely, so the likelihood carries no information
    class Constant:
        def __call__(self, theta):
            return np.zeros_like(grid) + 0.0 * theta[0]

    model = sc.ModelF(Constant(), input_names=["t_0", "t_1"])
    yobs = np.random.normal(0.0, 0.1, 20)

    setup = build_line_setup(yobs, model, nmcmc=8000)
    setup.addThetaPrior(
        dist="beta", params={"shape1": 3.0, "shape2": 6.0}, pname="t_0"
    )
    np.random.seed(2)
    out = sc.calibPool(setup)

    draws = out.theta[4000::2, 0, 0]
    prior = Beta(3.0, 6.0)
    assert abs(draws.mean() - prior.mean()) < 0.05
    assert abs(draws.std() - prior.std()) < 0.05

    # t_1 has no prior, so it should still be uniform
    draws1 = out.theta[4000::2, 0, 1]
    assert abs(draws1.mean() - 0.5) < 0.05


def test_calibPool_v2_prior_shifts_posterior():
    np.random.seed(0)
    grid = np.linspace(0, 1, 30)
    line = Line(grid)
    model = sc.ModelF(line, input_names=["t_0", "t_1"])
    yobs = np.random.normal(line(np.array([0.8, 0.2])), 0.1)

    np.random.seed(1)
    out_flat = sc.calibPool_v2(build_line_setup(yobs, model, nmcmc=3000))

    setup = build_line_setup(yobs, model, nmcmc=3000)
    setup.addThetaPrior(
        dist="normal", params={"mean": 0.2, "sd": 0.02}, pname="t_0"
    )
    np.random.seed(1)
    out_prior = sc.calibPool_v2(setup)

    flat_mean = out_flat.theta[1500:, 0, 0].mean()
    prior_mean = out_prior.theta[1500:, 0, 0].mean()
    assert prior_mean < flat_mean - 0.2
    assert abs(prior_mean - 0.2) < 0.05


def test_calibPool_unchanged_without_prior():
    """No prior set => bitwise-identical results to the pre-prior sampler."""
    np.random.seed(0)
    grid = np.linspace(0, 1, 30)
    line = Line(grid)
    model = sc.ModelF(line, input_names=["t_0", "t_1"])
    yobs = np.random.normal(line(np.array([0.8, 0.2])), 0.1)

    np.random.seed(3)
    out_a = sc.calibPool(build_line_setup(yobs, model, nmcmc=2000))

    setup = build_line_setup(yobs, model, nmcmc=2000)
    setup.theta_prior = []
    np.random.seed(3)
    out_b = sc.calibPool(setup)

    np.testing.assert_array_equal(out_a.theta, out_b.theta)
