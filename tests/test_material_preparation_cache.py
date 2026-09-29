"""Numerical comparisons against the frozen, pre-optimization implementation."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from impala import superCal as sc

spec = spec_from_file_location(
    "material_eval_reference",
    Path(__file__).parent / "reference/material_eval.py",
)
reference = module_from_spec(spec)
spec.loader.exec_module(reference)
original_eval = reference.original_material_eval


def make_model(pool=True):
    return sc.ModelMaterialStrength(
        temps=np.array([298.0, 1000.0]),
        edots=np.array([1e-9, 2.5e-3]),
        consts={
            "alpha": 0.2,
            "beta": 0.33,
            "matomic": 45.9,
            "Tmelt0": 2110.0,
            "rho0": 4.419,
            "Cv0": 0.525e-5,
            "G0": 0.4,
            "chi": 1.0,
            "sgB": 6.44e-4,
        },
        strain_histories=[np.linspace(0, 0.2, 9), np.linspace(0, 0.6, 13)],
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Constant_Melt_Temperature",
        shear_model="Simple_Shear_Modulus",
        specific_heat_model="Constant_Specific_Heat",
        density_model="Constant_Density",
        pool=pool,
    )


def parameters(n):
    values = {
        "theta": 0.1,
        "p": 2.0,
        "s0": 0.02,
        "sInf": 0.01,
        "kappa": 0.3,
        "lgamma": -12.0,
        "y0": 0.01,
        "yInf": 0.003,
        "y1": 0.09,
        "y2": 0.7,
    }
    result = {k: np.full(n, v) for k, v in values.items()}
    result["theta"] *= np.linspace(0.8, 1.2, n)
    return result


def assert_original(model, params, pool=None):
    old = deepcopy(model)
    expected = original_eval(old, params, pool=pool)
    actual = model.eval(params, pool=pool)
    # Stronger than a tolerance comparison: caching must preserve arithmetic.
    np.testing.assert_array_equal(actual, expected)
    variance = np.full(actual.shape[1], 1e-6)
    weights = np.ones(actual.shape[1])
    cov = model.lik_cov_inv(variance, weights)
    obs = expected[0] * 0.98
    for prediction, baseline in zip(actual, expected):
        assert model.llik(obs, prediction, cov, weights) == old.llik(
            obs, baseline, cov, weights
        )
    return actual


@pytest.mark.parametrize("pool", [True, False])
def test_changing_batches_and_parameters(pool):
    model = make_model(pool)
    for n in (1, 4, 2, 4, 7, 3, 1, 50, 2):
        params = parameters(n if pool else n * model.nexp)
        assert_original(model, params)
        params["theta"] *= 0.9
        assert_original(model, params)
        assert len(model._experiment_cache) <= 4


@pytest.mark.parametrize(
    "change", ["temps", "edots", "grid", "max", "Nhist", "constants"]
)
def test_mutable_inputs_do_not_return_stale_results(change):
    model = make_model()
    params = parameters(4)
    assert_original(model, params)
    if change == "temps":
        model.temps[0] += 150
    elif change == "edots":
        model.edots[0] = 1e-3
    elif change == "grid":
        model.meas_strain_histories[0][3] += 0.001
    elif change == "max":
        model.meas_strain_max[0] += 0.1
    elif change == "Nhist":
        model.Nhist = 31
    elif change == "constants":
        model.constants["G0"] *= 0.9
    assert_original(model, params)
    assert_original(model, params)


def test_pool_override_and_exposed_history_mutation():
    model = make_model(False)
    params = parameters(4)
    assert_original(model, params)
    assert_original(model, params, pool=True)
    model.model.strain_history["strains"][:] = -123
    model.model.strain_history["times"][:] = -123
    model.model.strain_history["strain_rate"][:] = -123
    model.model.edot[:] = -123
    model.model.emax[:] = -123
    assert_original(model, params, pool=True)
    assert_original(model, params)


def test_old_instance_without_cache():
    model = make_model()
    del model._experiment_cache
    assert_original(model, parameters(1))


def test_invalid_parameters_still_raise():
    model = make_model()
    params = parameters(2)
    assert_original(model, params)
    params["sInf"][:] = 0.3
    with pytest.raises(Exception) as baseline:
        original_eval(deepcopy(model), params)
    with pytest.raises(type(baseline.value), match="PTW bad val"):
        model.eval(params)


def test_observation_outside_history_still_raises():
    model = make_model()
    params = parameters(2)
    assert_original(model, params)
    model.meas_strain_histories[-1][-1] = 10
    with pytest.raises(ValueError):
        original_eval(deepcopy(model), params)
    with pytest.raises(ValueError):
        model.eval(params)


def test_saved_copper_predictions_and_likelihoods():
    import json

    data = Path(__file__).parent / "data"
    config = json.loads((data / "material_preparation_config.json").read_text())
    config["strain_histories"] = [
        np.array(x) for x in config["strain_histories"]
    ]
    model = sc.ModelMaterialStrength(**config)
    with np.load(data / "material_preparation_baseline.npz") as saved:
        params = sc.tran_unif(
            saved["normalized_parameters"],
            saved["bounds"],
            saved["parameter_names"],
        )
        for _ in range(2):  # Cold miss and warm hit.
            prediction = model.eval(params)
            np.testing.assert_allclose(
                prediction, saved["predictions"], rtol=1e-8, atol=1e-10
            )
            cov = model.lik_cov_inv(saved["variance"], saved["weights"])
            likelihood = np.array([
                model.llik(saved["observed"], row, cov, saved["weights"])
                for row in prediction
            ])
            np.testing.assert_allclose(
                likelihood, saved["log_likelihood"], rtol=0, atol=1e-6
            )
        bounds = dict(zip(saved["parameter_names"], saved["bounds"]))
        mask = sc.constraints_ptw(
            sc.tran_unif(
                saved["constraint_inputs"],
                saved["bounds"],
                saved["parameter_names"],
            ),
            bounds,
            model.constants,
        )
        np.testing.assert_array_equal(mask, saved["constraint_mask"])


def test_actual_simulated_grid_change_invalidates_interpolation_cache():
    from impala.physics import MaterialModel

    class ChangedGrid(MaterialModel):
        factor = 1.0

        def compute_state_history(self, strain_history=None):
            result = super().compute_state_history(strain_history)
            result[:, 1] *= self.factor
            return result

    model = make_model()
    model.model.__class__ = ChangedGrid
    params = parameters(2)
    assert_original(model, params)
    model.model.factor = 1.1
    assert_original(model, params)
    assert_original(model, params)
