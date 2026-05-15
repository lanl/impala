from pathlib import Path

import numpy as np
import pandas as pd

import impala.physics.physical_models_vec as physics
from impala import superCal as sc


def test_modelinit_1():
    """check if models initialize correctly and perform some regression tests"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "alpha": 0.84,
        "beta": 0.33,
        "matomic": 45.9,
        "chi": 1.0,
        "G0": 0.44,
        "rho0": 4.419,
        "rho_0": 4.45,
        "gamma_1": 2.2,
        "gamma_2": -4.7,
        "q2": 0.8,
        "c0": 4.730036e-05,
        "tm0": -3925.796,
        "tm1": 1448.2,
        "r0": 4.426741,
        "c1": 1.371e-8,
        "r1": -2.5965e-5,
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

    # bounds_ptw = {
    #     "theta": (0.0001, 0.2),
    #     "p": (0.0001, 5.0),
    #     "s0": (0.0001, 0.05),
    #     "sInf": (0.0001, 0.05),
    #     "kappa": (0.0001, 0.5),
    #     "lgamma": (np.log(1e-6), np.log(1e-4)),
    #     "y0": (0.0001, 0.05),
    #     "yInf": (0.0001, 0.01),
    #     "y1": (0.001, 0.1),
    #     "y2": (0.33, 1.0),
    #     "vel": (0.99, 1.01),
    # }

    bounds_ptw = {}
    for k, v in params.items():
        bounds_ptw[k] = (0.98 * v[0], 1.02 * v[0])
        if v < 0:
            bounds_ptw[k] = (1.02 * v[0], 0.98 * v[0])

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.BGP_PW_Shear_Modulus,
        melt_model=physics.Linear_Melt_Temperature,
        specific_heat_model=physics.Linear_Specific_Heat,
        density_model=physics.Linear_Density,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    ptw.set_history_variables(emax=emax, edot=np.array([edot]), nhist=nhist)
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history()
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    # pd.DataFrame(strainstress_new).to_csv(data_dir / "physics_strainstress_baseline_1.csv", index=False)
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline_1.csv"
    ).values

    # Test that the current model output matches the baseline.
    assert np.allclose(strainstress_old, strainstress_new)

    setup_pool_ptw = sc.CalibSetup(bounds_ptw, sc.constraints_ptw)
    model_pool_ptw = sc.ModelMaterialStrength(
        temps=np.array(temp),
        edots=np.array(edot),
        consts=consts,
        strain_histories=[sim_strains],
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Linear_Melt_Temperature",
        shear_model="BGP_PW_Shear_Modulus",
        specific_heat_model="Linear_Specific_Heat",
        density_model="Linear_Density",
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
        assert bestval < 1e-2
    assert sorted(out.theta_native.keys()) == sorted(params.keys())


def test_modelinit_2():
    """check if models initialize correctly and perform some regression tests"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "alpha": 0.84,
        "beta": 0.33,
        "matomic": 45.9,
        "chi": 1.0,
        # "G0": 0.44,
        "rho0": 4.419,
        "rho_0": 4.45,
        "g0": 0.44,
        "g1": 0.01,
        "c0": 4.730036e-05,
        "tm0": -3925.796,
        "tm1": 1448.2,
        "tm2": 5,
        "r0": 4.426741,
        "c1": 1.371e-8,
        "c2": 1e-10,
        "r1": -2.5965e-5,
        "r2": -1e-8,
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

    bounds_ptw = {}
    for k, v in params.items():
        bounds_ptw[k] = (0.98 * v[0], 1.02 * v[0])
        if v < 0:
            bounds_ptw[k] = (1.02 * v[0], 0.98 * v[0])

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.Linear_Cold_PW_Shear_Modulus,
        melt_model=physics.Quadratic_Melt_Temperature,
        specific_heat_model=physics.Quadratic_Specific_Heat,
        density_model=physics.Quadratic_Density,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    ptw.set_history_variables(emax=emax, edot=np.array([edot]), nhist=nhist)
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history()
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    # pd.DataFrame(strainstress_new).to_csv(data_dir / "physics_strainstress_baseline_2.csv", index=False)
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline_2.csv"
    ).values

    # Test that the current model output matches the baseline.
    assert np.allclose(strainstress_old, strainstress_new)

    setup_pool_ptw = sc.CalibSetup(bounds_ptw, sc.constraints_ptw)
    model_pool_ptw = sc.ModelMaterialStrength(
        temps=np.array(temp),
        edots=np.array(edot),
        consts=consts,
        strain_histories=[sim_strains],
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Quadratic_Melt_Temperature",
        shear_model="Linear_Cold_PW_Shear_Modulus",
        specific_heat_model="Quadratic_Specific_Heat",
        density_model="Quadratic_Density",
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
        assert bestval < 1e-2
    assert sorted(out.theta_native.keys()) == sorted(params.keys())


def test_modelinit_3():
    """check if models initialize correctly and perform some regression tests"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "alpha": 0.84,
        "beta": 0.33,
        "matomic": 45.9,
        "chi": 1.0,
        # "G0": 0.44,
        "rho0": 4.419,
        "rho_0": 4.45,
        "g0": 0.44,
        "g1": 0.01,
        "g2": 0.001,
        "c0": 4.730036e-05,
        "tm0": -3925.796,
        "tm1": 1448.2,
        "tm2": 5,
        "tm3": 5,
        "r0": 4.426741,
        "c1": 1.371e-8,
        "c2": 1e-10,
        "c3": 1e-12,
        "r1": -2.5965e-5,
        "r2": -1e-8,
        "r3": -1e-10,
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

    bounds_ptw = {}
    for k, v in params.items():
        bounds_ptw[k] = (0.98 * v[0], 1.02 * v[0])
        if v < 0:
            bounds_ptw[k] = (1.02 * v[0], 0.98 * v[0])

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.Quadratic_Cold_PW_Shear_Modulus,
        melt_model=physics.Cubic_Melt_Temperature,
        specific_heat_model=physics.Cubic_Specific_Heat,
        density_model=physics.Cubic_Density,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    ptw.set_history_variables(emax=emax, edot=np.array([edot]), nhist=nhist)
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history()
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    # pd.DataFrame(strainstress_new).to_csv(data_dir / "physics_strainstress_baseline_3.csv",index=False)
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline_3.csv"
    ).values

    # Test that the current model output matches the baseline.
    assert np.allclose(strainstress_old, strainstress_new)

    setup_pool_ptw = sc.CalibSetup(bounds_ptw, sc.constraints_ptw)
    model_pool_ptw = sc.ModelMaterialStrength(
        temps=np.array(temp),
        edots=np.array(edot),
        consts=consts,
        strain_histories=[sim_strains],
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Cubic_Melt_Temperature",
        shear_model="Quadratic_Cold_PW_Shear_Modulus",
        specific_heat_model="Cubic_Specific_Heat",
        density_model="Cubic_Density",
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
        assert bestval < 1e-2
    assert sorted(out.theta_native.keys()) == sorted(params.keys())


def test_modelinit_4():
    """check if models initialize correctly and perform some regression tests"""
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"

    consts = {
        "Tm_0": 1500,
        "alpha": 0.84,
        "beta": 0.33,
        "matomic": 45.9,
        "chi": 1.0,
        "G0": 0.44,
        "rho0": 4.419,
        "rho_0": 4.45,
        "g0": 0.44,
        "g1": 0.01,
        "g2": 0.001,
        "gamma_1": 2.2,
        "gamma_2": -4.7,
        "gamma_3": -4.7,
        "q2": 0.8,
        "q3": 0.8,
        "c0_0": 4.730036e-05,
        "c0_1": 4.730036e-05,
        "tm0": -3925.796,
        "tm1": 1448.2,
        "tm2": 5,
        "tm3": 5,
        "rho_m": 3.5,
        "r0": 4.426741,
        "c1_0": 1.371e-8,
        "c2_0": 1e-10,
        "c3_0": 1e-12,
        "c1_1": 1.371e-8,
        "c2_1": 1e-10,
        "c3_1": 1e-12,
        "T_t": 500,
        "r1": -2.5965e-5,
        "r2": -1e-8,
        "r3": -1e-10,
        "epsi": 1e2,
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
        "n": np.array([1.001]),
        "ymax": np.array([1e6]),
    }

    bounds = {}
    for k, v in params.items():
        bounds[k] = (0.98 * v[0], 1.02 * v[0])
        if v < 0:
            bounds[k] = (1.02 * v[0], 0.98 * v[0])

    stein = physics.MaterialModel(
        flow_stress_model=physics.Stein_Flow_Stress,
        shear_modulus_model=physics.Quadratic_Cold_PW_Shear_Modulus,
        melt_model=physics.BGP_Melt_Temperature,
        specific_heat_model=physics.Piecewise_Cubic_Specific_Heat,
        density_model=physics.Cubic_Density,
    )

    edot = 2500.0 * 1e-6  # 2500/s
    temp = 1000  # K
    emax = 0.6
    nhist = 100

    stein.set_history_variables(emax=emax, edot=np.array([edot]), nhist=nhist)
    stein.initialize(params, consts)
    stein.initialize_state(
        T=np.array([temp]), stress=np.zeros(1), strain=np.zeros(1)
    )
    sim_state_histories = stein.compute_state_history()
    sim_strains = sim_state_histories[:, 1]  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2]  # 2d array: ntot, Nhist

    strainstress_new = np.column_stack([sim_strains, sim_stresses])
    # pd.DataFrame(strainstress_new).to_csv(data_dir / "physics_strainstress_baseline_4.csv",index=False)
    strainstress_old = pd.read_csv(
        data_dir / "physics_strainstress_baseline_4.csv"
    ).values

    # Test that the current model output matches the baseline.
    assert np.allclose(strainstress_old, strainstress_new)

    setup_pool_stein = sc.CalibSetup(bounds)
    model_pool_stein = sc.ModelMaterialStrength(
        temps=np.array(temp),
        edots=np.array(edot),
        consts=consts,
        strain_histories=[sim_strains],
        flow_stress_model="Stein_Flow_Stress",
        melt_model="BGP_Melt_Temperature",
        shear_model="Quadratic_Cold_PW_Shear_Modulus",
        specific_heat_model="Piecewise_Cubic_Specific_Heat",
        density_model="Cubic_Density",
        pool=True,
        s2="gibbs",
    )
    yobs = sim_stresses[:, 0]
    setup_pool_stein.addVecExperiments(
        yobs=yobs,
        model=model_pool_stein,
        sd_est=[1.0],
        s2_df=[0],
        s2_ind=[0] * len(yobs),
    )
    setup_pool_stein.setTemperatureLadder(1.05 ** np.arange(20))
    setup_pool_stein.setMCMC(nmcmc=2000, decor=100)
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    out = sc.calibPool(setup_pool_stein)

    for k, v in out.theta_native.items():
        bestval = np.min(np.abs(v / params[k] - 1))
        # print(k,params[k],bestval)
        assert bestval < 1e-2
    assert sorted(out.theta_native.keys()) == sorted(params.keys())
