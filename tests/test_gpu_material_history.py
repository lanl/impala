"""Explicit hardware tests: IMPALA_TEST_GPU=1 python -m pytest this_file."""

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from impala import superCal as sc
from impala.physics import ConstraintError

pytestmark = pytest.mark.skipif(
    os.environ.get("IMPALA_TEST_GPU") != "1", reason="opt-in CUDA hardware test"
)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
spec = importlib.util.spec_from_file_location(
    "gpu_history", ROOT / "benchmarks/gpu/history.py"
)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def model_and_params(pool=True, n=256):
    cfg = json.loads((DATA / "material_preparation_config.json").read_text())
    cfg["strain_histories"] = [np.array(a) for a in cfg["strain_histories"]]
    cfg["pool"] = pool
    cfg["history_backend"] = "numba"
    model = sc.ModelMaterialStrength(**cfg)
    with np.load(DATA / "material_preparation_baseline.npz") as f:
        params = sc.tran_unif(
            f["normalized_parameters"][np.arange(n) % 256],
            f["bounds"],
            f["parameter_names"],
        )
    return model, params


@pytest.mark.parametrize("pool", [True, False])
def test_predictions_and_final_states(pool):
    for n in (1, 4, 50):
        cpu, params = model_and_params(pool, n if pool else n * 6)
        other = deepcopy(cpu)
        gpu.attach(other.model)
        expected = cpu.eval(params)
        actual = other.eval(params)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
        for key in ("T", "strain", "stress", "Cv", "rho", "Tmelt", "G"):
            np.testing.assert_allclose(
                getattr(other.model.state, key),
                getattr(cpu.model.state, key),
                rtol=1e-8,
                atol=1e-10,
            )


def test_original_fixture_and_likelihood():
    model, params = model_and_params()
    gpu.attach(model.model)
    with np.load(DATA / "material_preparation_baseline.npz") as f:
        predicted = model.eval(params)
        np.testing.assert_allclose(
            predicted, f["predictions"], rtol=1e-8, atol=1e-10
        )
        cov = model.lik_cov_inv(f["variance"], f["weights"])
        ll = np.array([
            model.llik(f["observed"], row, cov, f["weights"])
            for row in predicted
        ])
        np.testing.assert_allclose(ll, f["log_likelihood"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("nh", [2, 31, 127])
def test_entire_history_and_continuation(nh):
    model, params = model_and_params(n=4)
    model.Nhist = nh
    model.constants["chi"] = 0.7
    model.temps[0] += 17
    model.edots[0] *= 0.8
    model.eval(params)
    cpu = deepcopy(model.model)
    other = deepcopy(cpu)
    gpu.attach(other)
    h = deepcopy(cpu.strain_history)
    h["times"][:, 1:] += 0.001
    for _ in range(2):
        np.testing.assert_allclose(
            other.compute_state_history(h),
            cpu.compute_state_history(h),
            rtol=1e-8,
            atol=1e-10,
        )


def test_small_p_rate_threshold_and_invalid_parameters():
    cpu, params = model_and_params(n=4)
    params["p"][:] = [0, 1e-11, 1e-10, 1e-7]
    cpu.edots[:] = [1e-9, 1e-6, 1.0001e-6, 1e-3, 0.01, 0.001]
    other = deepcopy(cpu)
    gpu.attach(other.model)
    np.testing.assert_allclose(
        other.eval(params), cpu.eval(params), rtol=1e-8, atol=1e-10
    )
    params["sInf"][:] = 1
    with pytest.raises(ConstraintError, match="PTW bad val"):
        other.eval(params)


def test_mutated_inputs_are_transferred_again():
    cpu, params = model_and_params(n=3)
    other = deepcopy(cpu)
    gpu.attach(other.model)
    for i in range(3):
        cpu.constants["G0"] *= 0.95
        other.constants["G0"] *= 0.95
        cpu.temps[0] += 15
        other.temps[0] += 15
        params["theta"] *= 0.9
        np.testing.assert_allclose(
            other.eval(params), cpu.eval(params), rtol=1e-8, atol=1e-10
        )
