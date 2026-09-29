"""Accuracy gates for the optional compiled Cu history driver."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from impala import superCal as sc
from impala.physics import ConstraintError

pytest.importorskip("numba")
DATA = Path(__file__).parent / "data"


def fixture_model(pool=True, backend="python"):
    cfg = json.loads((DATA / "material_preparation_config.json").read_text())
    cfg["strain_histories"] = [np.array(x) for x in cfg["strain_histories"]]
    cfg["pool"] = pool
    cfg["history_backend"] = backend
    return sc.ModelMaterialStrength(**cfg)


def fixture_parameters(n):
    with np.load(DATA / "material_preparation_baseline.npz") as f:
        return sc.tran_unif(
            f["normalized_parameters"][
                np.arange(n) % len(f["normalized_parameters"])
            ],
            f["bounds"],
            f["parameter_names"],
        )


def compare(model, params):
    reference = deepcopy(model)
    reference.model.history_backend = "python"
    model.model.history_backend = "numba"
    expected = reference.eval(params)
    actual = model.eval(params)
    np.testing.assert_array_equal(actual, expected)
    for name in ("T", "strain", "stress", "Cv", "rho", "Tmelt", "G"):
        np.testing.assert_array_equal(
            getattr(model.model.state, name),
            getattr(reference.model.state, name),
        )


@pytest.mark.parametrize("pool", [True, False])
def test_batches_and_changed_parameters(pool):
    model = fixture_model(pool)
    for n in (1, 4, 50, 2):
        params = fixture_parameters(n if pool else n * model.nexp)
        compare(model, params)
        params["theta"] *= 0.95
        compare(model, params)


def test_independent_original_fixture():
    model = fixture_model(backend="numba")
    with np.load(DATA / "material_preparation_baseline.npz") as f:
        predicted = model.eval(fixture_parameters(256))
        np.testing.assert_allclose(
            predicted, f["predictions"], rtol=1e-8, atol=1e-10
        )
        cov = model.lik_cov_inv(f["variance"], f["weights"])
        likelihood = np.array([
            model.llik(f["observed"], row, cov, f["weights"])
            for row in predicted
        ])
        np.testing.assert_allclose(
            likelihood, f["log_likelihood"], rtol=0, atol=1e-6
        )


@pytest.mark.parametrize("nhist", [2, 31, 127])
def test_resolution_conditions_and_constants(nhist):
    model = fixture_model()
    model.Nhist = nhist
    model.temps[0] += 17
    model.edots[0] *= 0.8
    model.constants["chi"] = 0.75
    compare(model, fixture_parameters(3))


def test_full_history_and_continuation():
    model = fixture_model()
    model.eval(fixture_parameters(2))
    reference = deepcopy(model.model)
    candidate = deepcopy(reference)
    candidate.history_backend = "numba"
    # Nonzero initial state, nonuniform supplied history, and repeated calls.
    history = deepcopy(reference.strain_history)
    history["times"][:, 3:] += 0.123
    for _ in range(2):
        expected = reference.compute_state_history(history)
        actual = candidate.compute_state_history(history)
        np.testing.assert_array_equal(actual, expected)
        for name in ("T", "strain", "stress", "Cv", "rho", "Tmelt", "G"):
            np.testing.assert_array_equal(
                getattr(candidate.state, name), getattr(reference.state, name)
            )


def test_parameter_constraint_preserved():
    model = fixture_model()
    params = fixture_parameters(2)
    params["sInf"][:] = 1
    for backend in ("python", "numba"):
        model.model.history_backend = backend
        with pytest.raises(ConstraintError, match="PTW bad val"):
            model.eval(params)


def test_unsupported_models_and_method_override_fail_explicitly():
    model = fixture_model()
    model.model.history_backend = "numba"
    model.model.flow_stress.value = lambda edot: np.zeros_like(edot)
    with pytest.raises(ValueError, match="overridden model methods"):
        model.eval(fixture_parameters(2))
    del model.model.flow_stress.value
    from impala.physics import Constant_Density

    model.model.density = Constant_Density(model.model)
    with pytest.raises(ValueError, match="Cu PTW model combination"):
        model.eval(fixture_parameters(2))


def test_unknown_backend_and_old_instance():
    model = fixture_model()
    model.model.history_backend = "typo"
    with pytest.raises(ValueError, match="Unknown history backend"):
        model.eval(fixture_parameters(1))
    del model.model.history_backend
    model.eval(fixture_parameters(1))


def test_small_p_and_threshold_rates():
    model = fixture_model()
    params = fixture_parameters(4)
    params["p"][:] = [0, 1e-11, 1e-10, 1e-7]
    model.edots[:] = [1e-9, 1e-6, 1.0001e-6, 1e-3, 0.01, 0.001]
    compare(model, params)


@pytest.mark.parametrize(
    "kind",
    ["float32", "nonfinite", "state_shape", "constant_array", "history_shape"],
)
def test_unsupported_inputs_fail_explicitly(kind):
    model = fixture_model()
    model.eval(fixture_parameters(1))
    physics = model.model
    physics.history_backend = "numba"
    if kind == "float32":
        physics.state.T = physics.state.T.astype(np.float32)
    elif kind == "nonfinite":
        physics.state.T[0] = np.nan
    elif kind == "state_shape":
        physics.state.T = physics.state.T[:-1]
    elif kind == "constant_array":
        physics.parameters.chi = np.ones(6)
    else:
        physics.strain_history["strain_rate"] = np.zeros((6, 2))
    with pytest.raises(ValueError, match="numba history backend requires"):
        physics.compute_state_history()


def test_nonfinite_computation_raises():
    model = fixture_model(backend="numba")
    model.constants.update(c0=0.0, c1=0.0, c2=0.0)
    with pytest.raises(FloatingPointError):
        model.eval(fixture_parameters(1))
