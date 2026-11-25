"""Tests for the copula hierarchy used by calibHier.

The property that matters is the one a normal hierarchy truncated to the box
cannot deliver: a calibration parameter the data say nothing about must come
back with its prior margin, not pulled toward the middle of the box.
"""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import multivariate_normal as mvn

from impala import superCal as sc
from impala.superCal.impala_copula import (
    CopulaHyper,
    _latent_moments,
    draw_gamma0,
    from_theta,
    hier_logpdf,
    hyper_from_legacy,
    implied_theta_cov,
    to_theta,
)


def test_hier_logpdf_matches_scipy():
    """The hierarchy term must equal a diagonal multivariate normal."""
    rng = np.random.default_rng(0)
    ntemps, nth, p = 4, 6, 3
    v = rng.standard_normal((ntemps, nth, p))
    gam0 = rng.standard_normal((ntemps, p))
    rho = rng.uniform(0.2, 0.8, size=(ntemps, p))

    got = hier_logpdf(v, gam0, rho)
    assert got.shape == (ntemps, nth)
    for t in range(ntemps):
        for j in range(nth):
            assert got[t, j] == pytest.approx(
                mvn.logpdf(v[t, j], gam0[t], np.diag(1 - rho[t]))
            )


@pytest.mark.parametrize(("a", "b"), [(None, None), (2.0, 5.0), (8.0, 3.0)])
def test_theta_round_trip(a, b):
    """to_theta and from_theta must invert each other."""
    rng = np.random.default_rng(1)
    p = 4
    aa = None if a is None else np.full(p, a)
    bb = None if b is None else np.full(p, b)
    theta = rng.uniform(0.01, 0.99, size=(7, p))
    assert to_theta(from_theta(theta, aa, bb), aa, bb) == pytest.approx(theta)


def test_legacy_diffuse_prior_maps_to_uniform():
    """A vague legacy prior must become uniform margins, whatever its mean.

    theta0_prior_cov = eye(p) * 10**2 is a normal so diffuse that truncating it
    to the box leaves something indistinguishable from uniform, and that is the
    prior the truncated hierarchy was really using.  The nominal mean carries
    no information at that variance and must not be allowed to sharpen it.
    """
    p = 5
    off_centre = np.array([0.0017, 0.165, 0.5, 0.73, 0.98])
    a, b, _rho, clamped = hyper_from_legacy(
        off_centre, np.eye(p) * 10**2, p + 20, np.eye(p) * 0.1**2
    )
    assert np.all(clamped)
    assert a == pytest.approx(np.ones(p), abs=1e-2)
    assert b == pytest.approx(np.ones(p), abs=1e-2)


def test_legacy_informative_prior_round_trips():
    """An informative legacy prior must reproduce its mean and total spread."""
    p = 3
    mean, v_par, df, v_bet = 0.3, 0.05**2, p + 4, 0.05**2 * (p + 4 - p - 1)
    a, b, rho, clamped = hyper_from_legacy(
        np.full(p, mean), np.eye(p) * v_par, df, np.eye(p) * v_bet
    )
    assert not np.any(clamped)
    total = v_par + v_bet / (df - p - 1)
    assert beta_dist.mean(a[0], b[0]) == pytest.approx(mean, abs=1e-3)
    assert beta_dist.var(a[0], b[0]) == pytest.approx(total, rel=1e-3)
    # rho is the share of the marginal variance owned by the parent
    assert rho[0] == pytest.approx(v_par / total, abs=1e-2)


def test_margins_do_not_touch_the_updates():
    """F enters only the v -> theta map, so it must change no update.

    With a flat likelihood and a fixed seed, the latent chain must be
    bit-identical across different margins.
    """
    p, n, ntemps, nmcmc = 4, 8, 2, 300
    itl = np.ones(ntemps)
    chains = {}
    for a, b in [(1.0, 1.0), (2.0, 5.0), (0.7, 0.9)]:
        hy = CopulaHyper(p, a=np.full(p, a), b=np.full(p, b))
        np.random.seed(11)
        gam0 = np.zeros((ntemps, p))
        rho = np.full((ntemps, p), 0.5)
        v = [np.random.normal(size=(ntemps, n, p))]
        keep = np.empty((nmcmc, ntemps, p))
        for it in range(nmcmc):
            sd = np.sqrt(1.0 - rho)[:, None, :]
            v = [gam0[:, None, :] + sd * np.random.normal(size=v[0].shape)]
            gam0 = draw_gamma0(v, rho, itl)
            rho = hy.draw_rho(v, gam0, itl)
            keep[it] = rho
        chains[(a, b)] = (np.asarray(v[0]), keep)

    base_v, base_rho = chains[(1.0, 1.0)]
    for key, (vv, rr) in chains.items():
        assert np.array_equal(vv, base_v), key
        assert np.array_equal(rr, base_rho), key


@pytest.mark.parametrize(("a", "b"), [(1.0, 1.0), (2.0, 5.0)])
def test_margins_are_exact_under_flat_likelihood(a, b):
    """With no likelihood the cold-chain margin must be exactly Beta(a, b)."""
    p, n, ntemps, nmcmc = 3, 10, 3, 6000
    itl = np.array([1.0, 0.7, 0.5])
    hy = CopulaHyper(p, a=np.full(p, a), b=np.full(p, b))
    np.random.seed(3)
    gam0 = np.zeros((ntemps, p))
    rho = np.full((ntemps, p), 0.5)
    v = [np.random.normal(size=(ntemps, n, p))]
    keep = np.empty((nmcmc, n, p))
    keep_rho = np.empty((nmcmc, p))
    for it in range(nmcmc):
        sd = np.sqrt((1.0 - rho) / itl[:, None])[:, None, :]
        v = [gam0[:, None, :] + sd * np.random.normal(size=v[0].shape)]
        gam0 = draw_gamma0(v, rho, itl)
        rho = hy.draw_rho(v, gam0, itl)
        keep[it] = v[0][0]  # cold chain
        keep_rho[it] = rho[0]

    theta = hy.to_theta(keep[nmcmc // 2 :]).ravel()
    assert theta.mean() == pytest.approx(beta_dist.mean(a, b), abs=0.03)
    assert theta.std() == pytest.approx(np.sqrt(beta_dist.var(a, b)), abs=0.02)
    for q in (0.1, 0.5, 0.9):
        assert np.quantile(theta, q) == pytest.approx(
            beta_dist.ppf(q, a, b), abs=0.05
        )
    # rho keeps its Uniform(0,1) prior when the likelihood says nothing
    assert keep_rho[nmcmc // 2 :].mean() == pytest.approx(0.5, abs=0.06)


def test_implied_theta_cov_is_diagonal_and_sane():
    """The reported Sigma0 stand-in must be diagonal and of plausible size."""
    p = 3
    hy = CopulaHyper(p)
    gam0 = np.zeros((5, 2, p))
    rho = np.full((5, 2, p), 0.5)
    cov = implied_theta_cov(hy, gam0, rho, nsim=512)
    assert cov.shape == (5, 2, p, p)
    off = cov - np.einsum("mtk,kl->mtkl", np.diagonal(cov, 0, 2, 3), np.eye(p))
    assert np.allclose(off, 0.0)
    sd = np.sqrt(np.diagonal(cov, 0, 2, 3))
    assert np.all((sd > 0.05) & (sd < 0.5))


def test_unidentified_parameter_stays_uniform_end_to_end():
    """The whole point: calibHier must leave an unseen parameter uniform.

    Coordinate 3 is given a zero column in the design, so the likelihood says
    nothing about it at all.  Its posterior margin must be its prior margin.
    """
    np.random.seed(0)
    p, nexp, q, sd_obs = 4, 10, 8, 0.02
    rng = np.random.default_rng(5)
    design = rng.normal(size=(q, p))
    design[:, -1] = 0.0  # last coordinate unidentified
    truth = np.clip(0.5 + 0.12 * rng.standard_normal((nexp, p)), 0.02, 0.98)

    bounds = {f"x{k}": (0.0, 1.0) for k in range(p)}

    def f(parmat):
        return np.tile(parmat @ design.T, (1, nexp))

    model = sc.ModelF(
        f, list(bounds.keys()), exp_ind=np.repeat(np.arange(nexp), q)
    )
    yobs = np.hstack([
        truth[i] @ design.T + sd_obs * rng.normal(size=q) for i in range(nexp)
    ])
    ind = np.repeat(np.arange(nexp), q)

    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=yobs,
        model=model,
        sd_est=np.array([sd_obs] * nexp),
        s2_df=np.array([50] * nexp),
        s2_ind=ind,
        theta_ind=ind,
    )
    setup.setTemperatureLadder(1.05 ** np.arange(3), start_temper=500)
    setup.setMCMC(nmcmc=15000, decor=100)
    setup.setHierPriors(
        theta0_prior_mean=np.full(p, 0.5),
        theta0_prior_cov=np.eye(p) * 10**2,
        Sigma0_prior_df=p + 20,
        Sigma0_prior_scale=np.eye(p) * 0.1**2,
    )
    out = sc.calibHier(setup)

    assert np.allclose(out.copula.a, 1.0)  # diffuse prior -> uniform margins
    assert out.theta0.shape == (15000, 3, p)
    assert np.all((out.theta0 > 0.0) & (out.theta0 < 1.0))
    assert np.all((out.rho > 0.0) & (out.rho < 1.0))

    keep = np.arange(7500, 15000)
    theta = out.theta[0][keep, 0]

    # the unseen coordinate must be uniform
    unseen = theta[:, :, -1].ravel()
    assert unseen.mean() == pytest.approx(0.5, abs=0.06)
    assert unseen.std() == pytest.approx(0.2887, abs=0.035)
    assert ((unseen < 0.1) | (unseen > 0.9)).mean() == pytest.approx(
        0.2, abs=0.06
    )
    assert ((unseen > 0.4) & (unseen < 0.6)).mean() == pytest.approx(
        0.2, abs=0.06
    )

    # the identified coordinates must still be recovered
    mae = np.abs(theta[:, :, :-1].mean(0) - truth[:, :-1]).mean()
    assert mae < 0.02

    # and the unseen coordinate must pool less than the identified ones
    rho_mean = out.rho[keep, 0].mean(0)
    assert rho_mean[-1] < rho_mean[:-1].mean()


def test_masked_draws_honour_per_temperature_occupancy():
    """calibClust's hierarchy covers only occupied clusters.

    Which clusters are occupied differs by temperature and changes every
    iteration, so the gamma0 and rho draws take a mask and the cluster count
    they condition on is per-temperature, not a single number.
    """
    ntemps, nclust, p = 3, 6, 2
    rng = np.random.default_rng(0)
    v = [rng.standard_normal((ntemps, nclust, p))]
    mask = [
        np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=bool,
        )
    ]
    gam0 = np.zeros((ntemps, p))
    rho = np.full((ntemps, p), 0.5)
    itl = np.array([1.0, 0.7, 0.5])

    n, vsum, _sse = _latent_moments(v, gam0, mask_list=mask)
    assert np.array_equal(n, np.array([3, 2, 5]))
    # unoccupied clusters must not contribute
    assert vsum == pytest.approx((v[0] * mask[0][..., None]).sum(1))

    hy = CopulaHyper(p)
    assert draw_gamma0(v, rho, itl, mask_list=mask).shape == (ntemps, p)
    assert hy.draw_rho(v, gam0, itl, mask_list=mask).shape == (ntemps, p)

    # an all-True mask must reproduce the unmasked path exactly
    full = [np.ones((ntemps, nclust), dtype=bool)]
    np.random.seed(1)
    masked = hy.draw_rho(v, gam0, itl, mask_list=full)
    np.random.seed(1)
    plain = hy.draw_rho(v, gam0, itl)
    assert np.array_equal(masked, plain)

    np.random.seed(2)
    masked_g = draw_gamma0(v, rho, itl, mask_list=full)
    np.random.seed(2)
    plain_g = draw_gamma0(v, rho, itl)
    assert np.array_equal(masked_g, plain_g)


def test_parent_draw_latent_round_trips():
    """parent_draw(latent=True) must return the v behind the theta it would."""
    p = 3
    hy = CopulaHyper(p, a=np.full(p, 2.0), b=np.full(p, 5.0))
    gam0 = np.zeros((4, p))
    rho = np.full((4, p), 0.5)
    np.random.seed(7)
    v = hy.parent_draw(gam0, rho, n=5, latent=True)
    np.random.seed(7)
    theta = hy.parent_draw(gam0, rho, n=5)
    assert v.shape == (4, 5, p)
    assert hy.to_theta(v) == pytest.approx(theta)
    assert np.all((theta > 0.0) & (theta < 1.0))


def test_clust_unidentified_parameter_stays_uniform():
    """calibClust must leave an unseen parameter uniform too.

    Same criterion as the hierarchical case, but the cluster hierarchy covers
    only occupied clusters, so this exercises the masked gamma0/rho draws end
    to end.  Coordinate 3 has a zero column in the design.
    """
    np.random.seed(0)
    p, nsub, q, sd_obs = 4, 10, 8, 0.02
    rng = np.random.default_rng(11)
    design = rng.normal(size=(q, p))
    design[:, -1] = 0.0
    # two true groups, so the clustering has something to find
    truth = np.vstack([
        np.tile(
            np.clip(0.3 + 0.02 * rng.standard_normal(p), 0.02, 0.98),
            (nsub // 2, 1),
        ),
        np.tile(
            np.clip(0.7 + 0.02 * rng.standard_normal(p), 0.02, 0.98),
            (nsub // 2, 1),
        ),
    ])
    bounds = {f"x{k}": (0.0, 1.0) for k in range(p)}

    def f(theta):  # pool=True hands one theta row at a time
        return np.tile(np.asarray(theta) @ design.T, nsub)

    model = sc.ModelF(f, list(bounds.keys()))
    yobs = np.hstack([
        truth[i] @ design.T + sd_obs * rng.normal(size=q) for i in range(nsub)
    ])
    ind = np.repeat(np.arange(nsub), q)

    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=yobs,
        model=model,
        sd_est=np.array([sd_obs] * nsub),
        s2_df=np.array([50] * nsub),
        s2_ind=ind,
        theta_ind=ind,
    )
    setup.setTemperatureLadder(1.05 ** np.arange(3), start_temper=300)
    setup.setMCMC(nmcmc=12000, decor=100)
    setup.setClusterPriors(nclustmax=8)
    setup.setHierPriors(
        theta0_prior_mean=np.full(p, 0.5),
        theta0_prior_cov=np.eye(p) * 10**2,
        Sigma0_prior_df=p + 20,
        Sigma0_prior_scale=np.eye(p) * 0.1**2,
    )
    out = sc.calibClust(setup)

    assert np.allclose(out.copula.a, 1.0)
    assert np.all((out.theta0 > 0.0) & (out.theta0 < 1.0))
    assert np.all((out.rho > 0.0) & (out.rho < 1.0))

    keep = np.arange(6000, 12000)
    theta = out.theta_hist[0][keep, 0]

    # the clustering must find the two groups
    nclust = np.array([
        (np.bincount(out.delta[0][s, 0], minlength=out.nclustmax) > 0).sum()
        for s in keep
    ])
    assert 1.5 < nclust.mean() < 3.5

    unseen = theta[:, :, -1].ravel()
    assert unseen.mean() == pytest.approx(0.5, abs=0.07)
    assert unseen.std() == pytest.approx(0.2887, abs=0.04)
    assert ((unseen < 0.1) | (unseen > 0.9)).mean() == pytest.approx(
        0.2, abs=0.07
    )
    assert ((unseen > 0.4) & (unseen < 0.6)).mean() == pytest.approx(
        0.2, abs=0.07
    )

    mae = np.abs(theta[:, :, :-1].mean(0) - truth[:, :-1]).mean()
    assert mae < 0.02

    # the unseen coordinate's rho must stay at its prior mean while the
    # identified ones learn
    rho_mean = out.rho[keep, 0].mean(0)
    assert rho_mean[-1] == pytest.approx(0.5, abs=0.07)
