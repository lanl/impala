"""The batched Gibbs implementation must preserve draws and RNG state."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from cache_material_preparation import setup_for

from impala.superCal._pooled_gibbs import DiagonalGibbsBatch


def reference(
    model, itl, ny, a, b, indices, weights, observed, dev, pred, discrepancy
):
    logs, covs, ll = [], [], []
    for t in range(len(itl)):
        noise = np.log(
            1
            / np.random.gamma(
                itl[t] * (ny / 2 + a + 1) - 1,
                1 / (itl[t] * (b + dev[t].flatten() / 2)),
            )
        )
        logs.append(noise)
        covs.append(model.lik_cov_inv(np.exp(noise)[indices], weights, indices))
        ll.append(
            model.llik(observed - discrepancy[t], pred[t], covs[-1], weights)
        )
    return np.array(logs), covs, np.array(ll)


@pytest.mark.parametrize("ntemps", [1, 8, 50])
@pytest.mark.parametrize("seed", [42, 123, 456, 789])
def test_draws_likelihoods_and_rng_state(ntemps, seed):
    setup = setup_for("pooled", temperatures=ntemps)
    model = setup.models[0]
    itl = (np.ones((setup.ns2[0], ntemps)) * setup.itl).T
    args = (
        model,
        itl,
        setup.ny_s2[0],
        setup.ig_a[0],
        setup.ig_b[0],
        setup.s2_ind[0],
        setup.wt[0],
        setup.ys[0],
    )
    adapter = DiagonalGibbsBatch(*args)
    rng = np.random.default_rng(seed)
    pred = np.broadcast_to(setup.ys[0], (ntemps, len(setup.ys[0]))).copy()
    pred += rng.normal(0, 0.01, pred.shape)
    discrepancy = np.zeros_like(pred)
    membership = setup.s2_ind[0][:, None] == np.arange(setup.ns2[0])
    dev = (pred - setup.ys[0]) ** 2 @ membership
    np.random.seed(seed)
    expected = reference(*args, dev, pred, discrepancy)
    after_expected = np.random.get_state()
    np.random.seed(seed)
    actual = adapter.update(dev, pred, discrepancy)
    after_actual = np.random.get_state()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[2], expected[2])
    for a, b in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(a["inv"], b["inv"])
        np.testing.assert_array_equal(a["ldet"], b["ldet"])
    for a, b in zip(after_actual, after_expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("llik", lambda *args: 0),
        ("lik_cov_inv", lambda *args: {}),
        ("step", lambda: None),
        ("s2", "gibbs_trunc"),
        ("nd", 1),
        ("stochastic", True),
    ],
)
def test_reject_unsupported_likelihood(field, value):
    setup = setup_for("pooled")
    model = setup.models[0]
    setattr(model, field, value)
    with pytest.raises(ValueError, match="diagonal likelihoods"):
        DiagonalGibbsBatch(model, None, None, None, None, None, None, None)
