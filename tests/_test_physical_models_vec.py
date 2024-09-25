import numpy as np
import impala.physics.physical_models_vec as physics

# FIXME: (alui) Unsure what this test does. Used to be in
# impala/physics/test.py
def test_physics():
    consts = {
        'alpha'  : 0.2,
        'beta'   : 0.33,
        'matomic': 45.9,
        'Tmelt0' : 2110.,
        'rho0'   : 4.419,
        'Cv0'    : 0.525e-5,
        'G0'     : 0.4,
        'chi'    : 1.0,
        'sgB'    : 6.44e-4
    }

    params = {
        'theta' : np.array([0.1]),
        'p'     : np.array([2.]),
        's0'    : np.array([0.02]),
        'sInf'  : np.array([0.01]),
        'kappa' : np.array([0.3]),
        'lgamma' : np.array([-12.]),
        'y0'    : np.array([0.01]),
        'yInf'  : np.array([0.003]),
        'y1'    : np.array([0.09]),
        'y2'    : np.array([0.7]),
    }

    ptw = physics.MaterialModel(
        flow_stress_model=physics.PTW_Yield_Stress,
        shear_modulus_model=physics.Stein_Shear_Modulus,
    )

    edot = 2500. * 1e-6 # 2500/s
    temp = 1000 # K
    emax = 0.6
    nhist = 100

    sim_strain_histories = physics.generate_strain_history_new(
        emax=np.array([emax]),
        edot=np.array([edot]),
        nhist=nhist
    )
    ptw.initialize(params, consts)
    ptw.initialize_state(
        T = np.array([temp]),
        stress = np.zeros(1),
        strain = np.zeros(1)
    )
    sim_state_histories = ptw.compute_state_history(sim_strain_histories)
    sim_strains = sim_state_histories[:,1].T  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:,2].T # 2d array: ntot, Nhist

    # FIXME: (alui) What does this plot test? Can we convert this into a
    # numberic test?

    import matplotlib.pyplot as plt
    plt.plot(sim_strains.T, sim_stresses.T)
    plt.show()
