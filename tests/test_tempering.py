import numpy as np
import pytest

from impala import superCal as sc
from impala.superCal.tempering import (
    SwapScheduler,
    communication_barrier_ladder,
    even_odd_pairs,
)


def test_even_odd_pairs():
    assert even_odd_pairs(5, 0).tolist() == [[0, 1], [2, 3]]
    assert even_odd_pairs(5, 1).tolist() == [[1, 2], [3, 4]]
    assert even_odd_pairs(1, 0).shape == (0, 2)


def test_deo_alternates_parity():
    s = SwapScheduler(1 / 1.1 ** np.arange(6), scheme="deo")
    assert s.pairs()[:, 0].tolist() == [0, 2, 4]
    assert s.pairs()[:, 0].tolist() == [1, 3]
    assert s.pairs()[:, 0].tolist() == [0, 2, 4]


def test_random_scheme_pairs_are_disjoint():
    np.random.seed(1)
    s = SwapScheduler(1 / 1.1 ** np.arange(7), scheme="random")
    sw = s.pairs()
    assert sw.shape == (3, 2)
    assert len(np.unique(sw)) == 6
    assert np.all(sw[:, 0] < sw[:, 1])


def test_bad_options():
    with pytest.raises(ValueError):
        SwapScheduler(np.ones(3), scheme="nope")
    with pytest.raises(ValueError):
        SwapScheduler(np.ones(3), scheme="random", adapt=True)


def test_round_trip_rate_matches_theory():
    """
    With every swap accepted, DEO moves each replica cold -> hot -> cold in
    2(N - 1) rounds, so N replicas complete about rounds / 2 round trips
    (Syed et al. 2022, Theorem 3 with zero rejection).  The reversible
    stochastic even/odd scheme is diffusive and much slower.
    """
    np.random.seed(0)
    ntemps, rounds = 20, 20000
    trips = {}
    for scheme in ("deo", "seo"):
        s = SwapScheduler(1 / 1.1 ** np.arange(ntemps), scheme=scheme)
        for _ in range(rounds):
            sw = s.pairs()
            s.record(sw, np.zeros(len(sw)), np.ones(len(sw), dtype=bool))
        trips[scheme] = s.round_trips
    assert abs(trips["deo"] - rounds / 2) < 0.05 * rounds / 2
    assert trips["seo"] < trips["deo"] / 5


def test_barrier_ladder_equalises_rejection():
    """If rejection is proportional to the gap in inverse temperature, the
    re-spaced ladder must have equal gaps (and hence equal rejection)."""
    itl = 1 / 1.3 ** np.arange(10)
    rej = 0.5 * -np.diff(itl)
    new = communication_barrier_ladder(itl, rej)
    assert new[0] == itl[0] and new[-1] == itl[-1]
    assert np.all(np.diff(new) < 0)
    gaps = -np.diff(new)
    assert np.allclose(gaps, gaps.mean())


def test_scheduler_adaptation_schedule():
    s = SwapScheduler(
        1 / 1.3 ** np.arange(6), scheme="deo", adapt=True, adapt_until=1000
    )
    before = s.itl.copy()
    changed = []
    for m in range(1, 2000):
        sw = s.pairs()
        # rejection grows towards the cold end
        la = np.log(1 - 0.1 * (len(before) - sw[:, 0]) / len(before))
        s.record(sw, la, np.zeros(len(sw), dtype=bool))
        if s.maybe_adapt(m):
            changed.append(m)
    assert changed and max(changed) <= 1000
    assert s.itl[0] == before[0] and np.isclose(s.itl[-1], before[-1])
    assert not np.allclose(s.itl, before)


def _gaussian_setup(scheme, adapt):
    np.random.seed(3)
    nobs = 30
    truth = np.array([0.3, 0.6])
    y = np.random.normal(np.repeat(truth, nobs), 0.05)

    def f(th):
        return np.repeat(th, nobs)

    bounds = {"a": np.array([0, 1]), "b": np.array([0, 1])}
    setup = sc.CalibSetup(bounds, constraint_func="bounds")
    setup.addVecExperiments(
        yobs=y,
        model=sc.ModelF(f, input_names=bounds.keys(), s2="fix"),
        sd_est=[0.05],
        s2_df=[0],
        s2_ind=[0] * len(y),
    )
    setup.setTemperatureLadder(
        1.5 ** np.arange(8),
        start_temper=200,
        swap_scheme=scheme,
        adapt_ladder=adapt,
    )
    setup.setMCMC(nmcmc=4000, nburn=1500, decor=100)
    return setup, y.reshape(2, nobs).mean(axis=1), 0.05 / np.sqrt(nobs)


@pytest.mark.parametrize(
    "scheme,adapt", [("deo", False), ("deo", True), ("random", False)]
)
def test_pooled_gaussian_posterior(scheme, adapt):
    """Cold chain recovers the analytic posterior mean and sd."""
    setup, post_mean, post_sd = _gaussian_setup(scheme, adapt)
    out = sc.calibPool(setup)
    th = out.theta[1500:, 0]
    assert np.all(np.abs(th.mean(0) - post_mean) < 0.25 * post_sd)
    assert np.all(np.abs(th.std(0) / post_sd - 1) < 0.25)
    assert out.temperature_ladder.shape == (8,)
    assert out.temperature_ladder[0] == 1.0
    assert out.swap_rejection.shape == (7,)
    assert out.round_trips >= 0
    if adapt:
        assert not np.allclose(out.temperature_ladder, 1 / 1.5 ** np.arange(8))
    else:
        assert np.allclose(out.temperature_ladder, 1 / 1.5 ** np.arange(8))
