"""Prediction-only resident adapter: numerical gates and cache invalidation."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gpu.predictions import GpuCuPredictions
from test_gpu_material_history import DATA, model_and_params

pytestmark = pytest.mark.skipif(
    os.environ.get("IMPALA_TEST_GPU") != "1", reason="opt-in CUDA hardware test"
)


@pytest.mark.parametrize("hoist", [False, True])
@pytest.mark.parametrize("block_size", [32, 64, 128, 256])
def test_fixture_and_likelihood(block_size, hoist):
    model, params = model_and_params()
    adapter = GpuCuPredictions(model, block_size=block_size, hoist=hoist)
    with np.load(DATA / "material_preparation_baseline.npz") as f:
        actual = adapter(params)
        np.testing.assert_allclose(
            actual, f["predictions"], rtol=1e-8, atol=1e-10
        )
        cov = model.lik_cov_inv(f["variance"], f["weights"])
        ll = [
            model.llik(f["observed"], row, cov, f["weights"]) for row in actual
        ]
        np.testing.assert_allclose(ll, f["log_likelihood"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("hoist", [False, True])
def test_live_conditions_parameters_and_constants(hoist):
    cpu, params = model_and_params(n=50)
    other = deepcopy(cpu)
    adapter = GpuCuPredictions(other, hoist=hoist)
    for i in range(5):
        if i == 1:
            params["theta"] *= 0.91
            cpu.constants["G0"] *= 0.97
            other.constants["G0"] *= 0.97
        if i == 2:
            cpu.temps[0] += 12
            other.temps[0] += 12
            cpu.edots[0] *= 0.8
            other.edots[0] *= 0.8
        if i == 3:
            cpu.Nhist = other.Nhist = 31
        if i == 4:
            cpu.meas_strain_histories[0][1] *= 0.99
            other.meas_strain_histories[0][1] *= 0.99
        np.testing.assert_allclose(
            adapter(params), cpu.eval(params), rtol=1e-8, atol=1e-10
        )
    assert adapter.hits == 1
    assert adapter.misses == 4


@pytest.mark.parametrize("hoist", [False, True])
def test_bounded_cache_unequal_batches_and_thresholds(hoist):
    cpu, params = model_and_params(n=200)
    params["p"][:4] = [0, 1e-11, 1e-10, 1e-7]
    cpu.edots[:] = [1e-9, 1e-6, 1.0001e-6, 1e-3, 0.01, 0.001]
    adapter = GpuCuPredictions(deepcopy(cpu), cache_bytes=100_000, hoist=hoist)
    for n in (1, 4, 50, 159, 200, 1):
        native = {k: v[:n] for k, v in params.items()}
        np.testing.assert_allclose(
            adapter(native), cpu.eval(native), rtol=1e-8, atol=1e-10
        )
        assert adapter.resident_bytes <= adapter.cache_bytes


def test_reject_invalid_inputs():
    from impala.physics import ConstraintError

    cpu, params = model_and_params(n=4)
    adapter = GpuCuPredictions(cpu)
    with pytest.raises(ValueError):
        adapter(params, pool=False)
    with pytest.raises(ValueError):
        adapter({k: v[:0] for k, v in params.items()})
    params["sInf"][:] = 1
    with pytest.raises(ConstraintError):
        adapter(params)


def test_timing_preserves_predictions_and_block_order():
    model, params = model_and_params(n=159)
    reference = GpuCuPredictions(model)(params)
    for size in (32, 64, 128, 256):
        adapter = GpuCuPredictions(
            deepcopy(model), block_size=size, timing=True
        )
        np.testing.assert_array_equal(adapter(params), reference)
        sample = adapter.stage_samples[-1]
        assert sample["history_device_s"] > 0
        assert sample["interpolation_device_s"] > 0
        assert sample["download_and_host_allocation_s"] >= 0
    with pytest.raises(ValueError, match="block size"):
        GpuCuPredictions(model, block_size=512)
