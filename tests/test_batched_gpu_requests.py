"""The batching boundary must preserve each request's prediction and likelihood."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
spec = importlib.util.spec_from_file_location(
    "batched_chains", ROOT / "benchmarks/gpu_batched_chains.py"
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_heterogeneous_requests_preserve_rows_and_likelihood(backend):
    if backend == "gpu" and os.environ.get("IMPALA_TEST_GPU") != "1":
        pytest.skip("opt-in CUDA hardware test")
    reference = batch.setup_for("pooled").models[0]
    reference.model.history_backend = "numba"
    combined = batch.setup_for("pooled").models[0]
    combined.model.history_backend = "numba"
    if backend == "gpu":
        from gpu.history import attach

        attach(combined.model)
    with np.load(ROOT / "tests/data/material_preparation_baseline.npz") as f:
        saved = {k: f[k] for k in f.files}
    native = batch.mc.sc.tran_unif(
        saved["normalized_parameters"],
        saved["bounds"],
        saved["parameter_names"],
    )
    requests = [
        {k: v[start : start + n] for k, v in native.items()}
        for start, n in [(10, 1), (50, 4), (100, 50), (0, 37)]
    ]
    expected = [reference.eval(r) for r in requests]
    actual, total = batch.evaluate_group(combined, requests)
    assert total == 92
    cov = reference.lik_cov_inv(saved["variance"], saved["weights"])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)
        for row, control in zip(a, b):
            assert (
                abs(
                    reference.llik(
                        saved["observed"], row, cov, saved["weights"]
                    )
                    - reference.llik(
                        saved["observed"], control, cov, saved["weights"]
                    )
                )
                <= 1e-6
            )
    # Input arrays must not be modified by concatenation/dispatch.
    np.testing.assert_array_equal(requests[0]["theta"], native["theta"][10:11])


def test_mismatched_requests_rejected_before_evaluation():
    with pytest.raises(ValueError, match="parameter names"):
        batch.evaluate_group(None, [{"a": np.ones(1)}, {"b": np.ones(2)}])
    with pytest.raises(ValueError, match="shapes"):
        batch.evaluate_group(None, [{"a": np.ones(0)}])
