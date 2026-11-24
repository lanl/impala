"""Tests for the hierarchical truncation diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from impala import superCal as sc
from impala.superCal.post_process import (
    hier_truncation_weights,
    weighted_quantile,
)


def _setup(p=2, nexp=3, nobs=4):
    """A hierarchical setup; no MCMC is run, only setup fields are used."""
    bounds = {f"x{j}": np.array([0.0, 1.0]) for j in range(p)}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    ind = np.repeat(np.arange(nexp), nobs)
    model = sc.ModelF(
        lambda X: np.zeros((X.shape[0], nobs * nexp)),
        input_names=bounds.keys(),
        exp_ind=ind,
        s2="fix",
    )
    setup.addVecExperiments(
        yobs=np.zeros(nobs * nexp),
        model=model,
        sd_est=np.ones(nexp),
        s2_df=np.zeros(nexp),
        s2_ind=ind,
        theta_ind=ind,
    )
    return setup


def _out(th0, s0diag, ntemps=2):
    """calibHier-shaped output holding the given cold-chain draws."""
    nd, p = th0.shape
    theta0 = np.zeros((nd, ntemps, p))
    theta0[:, 0] = th0
    Sigma0 = np.zeros((nd, ntemps, p, p))
    for j in range(nd):
        Sigma0[j, 0] = np.diag(s0diag[j])
        Sigma0[j, 1] = np.eye(p)
    return SimpleNamespace(theta0=theta0, Sigma0=Sigma0)


def test_weights_match_closed_form():
    """For a diagonal Sigma0 the box probability is a product of normal CDFs,
    so the weights can be checked against the formula directly."""
    setup = _setup(p=2, nexp=3)
    n = sum(setup.ntheta)
    assert n == 3
    rng = np.random.default_rng(0)
    nd = 50
    th0 = rng.uniform(0.1, 0.9, size=(nd, 2))
    s0 = rng.uniform(0.02, 0.5, size=(nd, 2)) ** 2
    res = hier_truncation_weights(setup, _out(th0, s0), mcmc_use=np.arange(nd))

    sd = np.sqrt(s0)
    lp = np.log(norm.cdf((1 - th0) / sd) - norm.cdf(-th0 / sd)).sum(1)
    lw = -n * lp
    expect = np.exp(lw - lw.max())
    expect /= expect.sum()
    assert np.allclose(res["weights"], expect)
    assert np.allclose(res["log_box_prob"], lp)
    assert res["n_theta"] == n


def test_full_covariance_path_agrees_with_diagonal():
    """The Genz path and the diagonal shortcut must agree when the covariance
    is diagonal but stored with explicit (zero) off-diagonal entries."""
    from impala.superCal.post_process import _log_box_prob

    mean = np.array([0.3, 0.6])
    cov = np.diag([0.04, 0.09])
    diag_val = _log_box_prob(mean, cov)
    nudged = cov + np.array([[0.0, 1e-9], [1e-9, 0.0]])
    assert np.isclose(diag_val, _log_box_prob(mean, nudged), atol=1e-6)


def test_no_tilt_when_bounds_are_slack():
    """theta0 mid-box with a tiny Sigma0 keeps essentially all of the normal
    mass inside the bounds, so the weights are flat and nothing is corrected."""
    setup = _setup(p=2, nexp=5)
    rng = np.random.default_rng(1)
    nd = 200
    th0 = 0.5 + rng.normal(scale=0.01, size=(nd, 2))
    s0 = np.full((nd, 2), 0.01**2)
    res = hier_truncation_weights(setup, _out(th0, s0), mcmc_use=np.arange(nd))
    assert res["ess_frac"] > 0.99
    assert res["tilt_nats"] < 1e-6
    assert np.allclose(res["theta0_sd_raw"], res["theta0_sd_reweighted"],
                       rtol=1e-3)


def test_reweighting_widens_sigma0_when_bounds_bind():
    """With the bounds binding, the correction must push Sigma0 UP -- that is
    the contraction it exists to undo."""
    setup = _setup(p=1, nexp=6)
    rng = np.random.default_rng(2)
    nd = 4000
    th0 = rng.uniform(0.2, 0.8, size=(nd, 1))
    s0 = rng.uniform(0.05, 0.6, size=(nd, 1)) ** 2
    res = hier_truncation_weights(setup, _out(th0, s0), mcmc_use=np.arange(nd))
    assert res["tilt_nats"] > 0.1
    assert (
        res["sigma0_diag_median_reweighted"][0]
        > res["sigma0_diag_median_raw"][0]
    )


def test_defaults_to_second_half_of_chain():
    setup = _setup(p=1, nexp=2)
    nd = 100
    th0 = np.full((nd, 1), 0.5)
    s0 = np.full((nd, 1), 0.01)
    res = hier_truncation_weights(setup, _out(th0, s0))
    assert len(res["weights"]) == nd // 2


def test_custom_constraint_func_warns():
    bounds = {"a": np.array([0.0, 1.0])}
    setup = sc.CalibSetup(bounds, constraint_func=lambda x, b, c=None: True)
    ind = np.zeros(3, dtype=int)
    setup.addVecExperiments(
        yobs=np.zeros(3),
        model=sc.ModelF(
            lambda X: np.zeros((X.shape[0], 3)),
            input_names=bounds.keys(),
            exp_ind=ind,
            s2="fix",
        ),
        sd_est=np.ones(1),
        s2_df=np.zeros(1),
        s2_ind=ind,
        theta_ind=ind,
    )
    out = _out(np.full((10, 1), 0.5), np.full((10, 1), 0.04))
    with pytest.warns(UserWarning, match="bounds only"):
        hier_truncation_weights(setup, out, mcmc_use=np.arange(10))


def test_weighted_quantile_matches_unweighted():
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    w = np.full(5, 0.2)
    assert np.isclose(weighted_quantile(x, 0.5, w), np.median(x))
