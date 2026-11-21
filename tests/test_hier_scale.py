import numpy as np
import pytest

from impala import superCal as sc
from impala.superCal.impala_noprobit_emu import clip_unit, probit


def _flat_setup(scale, nmcmc, ntemps=1):
    """Hierarchical setup whose likelihood does not depend on theta, so the
    cold chain should sample the prior of the hierarchical model exactly."""
    p, nexp, nobs = 2, 6, 5
    bounds = {"a": np.array([0, 1]), "b": np.array([0, 1])}
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
    setup.setTemperatureLadder(1.5 ** np.arange(ntemps), start_temper=100)
    setup.setMCMC(nmcmc=nmcmc, decor=100)
    mean0 = 0.0 if scale == "probit" else 0.5
    setup.setHierPriors(
        theta0_prior_mean=np.repeat(mean0, p),
        theta0_prior_cov=np.eye(p) * 0.2**2,
        Sigma0_prior_df=p + 4,
        Sigma0_prior_scale=np.eye(p) * 0.3**2,
        scale=scale,
    )
    return setup


def test_bad_scale():
    setup = sc.CalibSetup({"a": np.array([0, 1])})
    with pytest.raises(ValueError):
        setup.setHierPriors(np.zeros(1), np.eye(1), 3, np.eye(1), scale="logit")


@pytest.mark.parametrize("ntemps", [1, 3])
def test_probit_hierarchy_recovers_prior(ntemps):
    """
    On the probit scale theta_i ~ N(theta0, Sigma0) is not truncated, so with a
    flat likelihood theta0 must follow its N(0, 0.2^2 I) prior.  (On the unit
    scale the untruncated-normal Gibbs update gives sd(theta0) ~ 0.16 here,
    because the bounds tilt the effective hyperprior.)
    """
    np.random.seed(11)
    setup = _flat_setup("probit", nmcmc=12000, ntemps=ntemps)
    out = sc.calibHier(setup)
    th0 = out.theta0[2000:, 0]
    assert np.all((th0 > 0) & (th0 < 1))  # returned on the [0, 1] scale
    z0 = probit(clip_unit(th0))
    # batch-means Monte Carlo standard error of the mean
    batches = z0[: len(z0) // 20 * 20].reshape(20, -1, z0.shape[1]).mean(axis=1)
    mcse = batches.std(axis=0, ddof=1) / np.sqrt(20)
    assert np.all(np.abs(z0.mean(0)) < 4 * mcse + 0.01)
    assert np.all(np.abs(z0.std(0) / 0.2 - 1) < 0.12)
    for th in out.theta:
        assert np.all((th[2000:, 0] > 0) & (th[2000:, 0] < 1))

    parent = setup.sample_hier_parent(
        out.theta0[2000:, 0], out.Sigma0[2000:, 0]
    )
    assert parent.shape == th0.shape
    assert np.all((parent > 0) & (parent < 1))


def test_unit_scale_parent_unchanged():
    """sample_hier_parent on the unit scale is the old constrained sampler."""
    setup = _flat_setup("unit", nmcmc=10)
    th0 = np.full((50, 2), 0.5)
    S0 = np.repeat(np.eye(2)[None] * 0.01, 50, axis=0)
    np.random.seed(5)
    a = setup.sample_hier_parent(th0, S0)
    np.random.seed(5)
    b = setup.chol_sample_nper_constraints(th0, S0, n=1)
    assert np.array_equal(a, b)
