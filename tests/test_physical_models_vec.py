from pathlib import Path

import numpy as np
import pandas as pd

import impala.physics.physical_models_vec as physics
from impala import superCal as sc


def test_physics():
    """Regression test for PTW"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "alpha": 0.2,
        "beta": 0.33,
        "matomic": 45.9,
        "Tmelt0": 2110.0,
        "rho0": 4.419,
        "Cv0": 0.525e-5,
        "G0": 0.4,
        "chi": 1.0,
        "sgB": 6.44e-4,
    }

    params = {
        "theta": np.array([0.1]),
        "p": np.array([2.0]),
        "s0": np.array([0.02]),
        "sInf": np.array([0.01]),
        "kappa": np.array([0.3]),
        "lgamma": np.array([-12.0]),
        "y0": np.array([0.01]),
        "yInf": np.array([0.003]),
        "y1": np.array([0.09]),
        "y2": np.array([0.7]),
    }

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.Stein_Shear_Modulus,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    sim_strain_histories = physics.generate_strain_history_new(
        emax=np.array([emax]), edot=np.array([edot]), nhist=nhist
    )
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history(sim_strain_histories)
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline.csv", index_col=0
    ).values

    # Test that the current model output matches the baseline.
    assert np.allclose(strainstress_old, strainstress_new)


def test_constparams():
    """tests the pooled calibration of the ptw model some of its parameters kept
    constant for artificial data. Despite the small number of MCMC iterations,
    we check that the "best" parameters are within 1 permille of the values
    used to generate the artifical data, as well as that the keys of "params"
    match the keys of the calibrated parameters."""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "alpha": 0.2,
        "beta": 0.33,
        "matomic": 45.9,
        "Tmelt0": 2110.0,
        "rho0": 4.419,
        "Cv0": 0.525e-5,
        "G0": 0.4,
        "chi": 1.0,
        "sgB": 6.44e-4,
        "y1": 0.09,
        "y2": 0.7,
        "kappa": 0.3,
        "lgamma": -12.0,
    }

    params = {
        "theta": np.array([0.1]),
        "p": np.array([2.0]),
        "s0": np.array([0.02]),
        "sInf": np.array([0.01]),
        "y0": np.array([0.01]),
        "yInf": np.array([0.003]),
    }

    bounds_ptw = {}
    for k, v in params.items():
        bounds_ptw[k] = (0.98 * v[0], 1.02 * v[0])
        if v < 0:
            bounds_ptw[k] = (1.02 * v[0], 0.98 * v[0])

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.Stein_Shear_Modulus,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    sim_strain_histories = physics.generate_strain_history_new(
        np.array([emax]), np.array([edot]), nhist
    )
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history(sim_strain_histories)
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline.csv", index_col=0
    ).values

    # Test that the current model output matches the baseline, this time using more constants
    assert np.allclose(strainstress_old, strainstress_new)

    setup_pool_ptw = sc.CalibSetup(bounds_ptw, sc.constraints_ptw)
    model_pool_ptw = sc.ModelMaterialStrength(
        temps=np.array(temp),
        edots=np.array(edot),
        consts=consts,
        strain_histories=[sim_strains],
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Constant_Melt_Temperature",
        shear_model="Stein_Shear_Modulus",
        specific_heat_model="Constant_Specific_Heat",
        density_model="Constant_Density",
        pool=True,
        s2="gibbs",
    )
    yobs = sim_stresses[:, 0]
    setup_pool_ptw.addVecExperiments(
        yobs=yobs,
        model=model_pool_ptw,
        sd_est=[1.0],
        s2_df=[0],
        s2_ind=[0] * len(yobs),
    )
    setup_pool_ptw.setTemperatureLadder(1.05 ** np.arange(20))
    setup_pool_ptw.setMCMC(nmcmc=2000, decor=100)
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    out = sc.calibPool(setup_pool_ptw)
    for k, v in out.theta_native.items():
        bestval = np.min(np.abs(v / params[k] - 1))
        # print(k,params[k],bestval)
        assert bestval < 1e-3
    assert sorted(list(out.theta_native.keys())) == sorted(list(params.keys()))
