import numpy as np
import scipy.stats as st

from impala import superCal as sc
from impala.superCal.impala_clust import sample_eta
from impala.superCal.impala_noprobit_emu import (
    ldhc_kern,
    ldig_kern,
    s2_swap_kern,
)


def test_sample_eta_preserves_prior():
    """
    Geweke-style check of the Escobar & West (1995) update: if
    eta ~ Gamma(a, b) and k | eta is the number of clusters of a CRP(eta)
    partition of n items, one update must return eta ~ Gamma(a, b) again.
    """
    rng = np.random.default_rng(0)
    np.random.seed(0)
    a, b, n, N = 2.0, 2.0, 8, 400_000
    eta = rng.gamma(a, 1 / b, size=N)
    # k = 1 + sum_{i=1}^{n-1} Bernoulli(eta / (eta + i))
    i = np.arange(1, n)
    k = 1 + (
        rng.uniform(size=(N, n - 1)) < eta[:, None] / (eta[:, None] + i)
    ).sum(1)
    eta_new = sample_eta(eta, k, n, a, b)
    assert abs(eta_new.mean() - a / b) < 0.01
    assert abs(eta_new.var() - a / b**2) < 0.02
    assert (
        st.kstest(eta_new[:20000], st.gamma(a, scale=1 / b).cdf).pvalue > 1e-3
    )


class _M:
    def __init__(self, s2):
        self.s2 = s2


class _Setup:
    def __init__(self, s2):
        self.models = [_M(s2)]
        self.s2_prior_kern = [ldhc_kern]


def test_s2_swap_kern_matches_update():
    assert s2_swap_kern(_Setup("gibbs"), 0) is ldig_kern
    assert s2_swap_kern(_Setup("gibbs_trunc"), 0) is ldig_kern
    assert s2_swap_kern(_Setup("MH"), 0) is ldhc_kern
    # with s2_df = 0 the Gibbs update targets p(s2) ~ 1/s2
    x = np.array([0.5, 2.0])
    assert np.allclose(ldig_kern(x, 0.0, 0.0), -np.log(x))


def test_pooled_gibbs_s2_with_tempering():
    """s2 Gibbs with s2_df = 0 and a model that does not depend on theta:
    s2 | y ~ InvGamma(n/2, SS/2) exactly, with or without tempering."""
    rs = np.random.RandomState(123)
    n = 20
    y = rs.normal(0, 0.5, n)
    a, b = n / 2, (y**2).sum() / 2
    np.random.seed(1)
    bounds = {"a": np.array([0, 1])}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=y,
        model=sc.ModelF(
            lambda th: np.zeros(n), input_names=bounds.keys(), s2="gibbs"
        ),
        sd_est=[1.0],
        s2_df=[0],
        s2_ind=[0] * n,
    )
    setup.setTemperatureLadder(2.0 ** np.arange(4), start_temper=100)
    setup.setMCMC(nmcmc=12000, decor=100)
    out = sc.calibPool(setup)
    s2 = out.s2[0][2000:, 0, 0]
    assert abs(s2.mean() / (b / (a - 1)) - 1) < 0.015
    assert abs(np.median(s2) / st.invgamma(a, scale=b).median() - 1) < 0.015


def test_clust_tempering_flat_likelihood():
    """With a flat likelihood every chain's target is the clustering prior, so
    the cold chain must reproduce the Gamma(2, 2) prior on eta (mean 1)."""
    np.random.seed(1)
    p, nobs, ntheta = 2, 4, 8
    L = nobs * ntheta
    bounds = {"a": np.array([0, 1]), "b": np.array([0, 1])}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    ind = np.repeat(np.arange(ntheta), nobs)

    def f(X):
        X = np.asarray(X)
        return np.zeros(L) if X.ndim == 1 else np.zeros((X.shape[0], L))

    setup.addVecExperiments(
        yobs=np.zeros(L),
        model=sc.ModelF(f, input_names=bounds.keys(), exp_ind=ind, s2="fix"),
        sd_est=np.ones(ntheta),
        s2_df=np.zeros(ntheta),
        s2_ind=ind,
        theta_ind=ind,
    )
    setup.setTemperatureLadder(1.5 ** np.arange(3), start_temper=100)
    setup.setMCMC(nmcmc=5000, decor=100)
    setup.setHierPriors(
        theta0_prior_mean=np.repeat(0.5, p),
        theta0_prior_cov=np.eye(p) * 0.2**2,
        Sigma0_prior_df=p + 4,
        Sigma0_prior_scale=np.eye(p) * 0.3**2,
    )
    setup.setClusterPriors(nclustmax=8, eta_prior_shape=2, eta_prior_rate=2)
    out = sc.calibClust(setup)
    eta = out.eta[1000:, 0]
    assert abs(eta.mean() - 1.0) < 0.12


def test_pooled_mh_s2_with_tempering():
    """
    s2 by Metropolis-Hastings on log(s2) with s2_df = 0 (kernel 1/(1 + s2)) and
    a model that does not depend on theta.  The Jacobian of the log transform
    must not be tempered, otherwise hot chains target a different family than
    the swap ratio assumes and the cold chain is biased (about -5.5% here).
    """
    from scipy import integrate

    rs = np.random.RandomState(123)
    n = 20
    y = rs.normal(0, 0.5, n)
    SS = (y**2).sum()

    def unnorm(s2):
        return np.exp(-n / 2 * np.log(s2) - SS / 2 / s2 - np.log1p(s2) + 20)

    Z = integrate.quad(unnorm, 1e-4, 50)[0]
    exact = integrate.quad(lambda s: s * unnorm(s), 1e-4, 50)[0] / Z

    np.random.seed(2)
    bounds = {"a": np.array([0, 1])}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=y,
        model=sc.ModelF(
            lambda th: np.zeros(n), input_names=bounds.keys(), s2="MH"
        ),
        sd_est=[0.5],
        s2_df=[0],
        s2_ind=[0] * n,
    )
    setup.setTemperatureLadder(2.0 ** np.arange(4), start_temper=200)
    setup.setMCMC(nmcmc=20000, decor=100, start_var_ls2=0.1)
    out = sc.calibPool(setup)
    s2 = out.s2[0][3000:, 0, 0]
    assert abs(s2.mean() / exact - 1) < 0.025
