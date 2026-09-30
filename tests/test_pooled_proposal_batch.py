"""Proposal batching preserves scalar arithmetic, current swaps, and RNG state."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from cache_material_preparation import setup_for

from impala.superCal._pooled_gibbs import DiagonalGibbsBatch


@pytest.mark.parametrize("ntemps", [1, 8, 50])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_proposal_likelihood_after_covariance_swaps(ntemps, layout):
    setup = setup_for("pooled", temperatures=ntemps)
    model = setup.models[0]
    itl = (np.ones((setup.ns2[0], ntemps)) * setup.itl).T
    adapter = DiagonalGibbsBatch(
        model,
        itl,
        setup.ny_s2[0],
        setup.ig_a[0],
        setup.ig_b[0],
        setup.s2_ind[0],
        setup.wt[0],
        setup.ys[0],
    )
    rng = np.random.default_rng(42)
    shape = (ntemps, len(setup.ys[0]))
    pred = rng.normal(size=shape)
    if layout == "F":
        pred = np.asfortranarray(pred)
    elif layout == "strided":
        storage = np.empty((ntemps, shape[1] * 2))
        storage[:, ::2] = pred
        pred = storage[:, ::2]
    discrepancy = np.zeros(shape)
    covs = [
        model.lik_cov_inv(rng.uniform(0.01, 1, shape[1]), setup.wt[0])
        for _ in range(ntemps)
    ]
    state = np.random.get_state()
    for _ in range(3):
        # Swaps reorder the actual dictionaries; new Gibbs draws replace them.
        order = rng.permutation(ntemps)
        covs = [covs[t].copy() for t in order]
        covs[0] = model.lik_cov_inv(rng.uniform(0.01, 1, shape[1]), setup.wt[0])
        expected = np.array([
            model.llik(
                setup.ys[0] - discrepancy[t], pred[t], covs[t], setup.wt[0]
            )
            for t in range(ntemps)
        ])
        actual = adapter.proposal_likelihood(pred, discrepancy, covs)
        np.testing.assert_array_equal(actual, expected)
    for before, after in zip(state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)
