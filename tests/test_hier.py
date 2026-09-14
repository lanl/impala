import numpy as np

from impala import superCal as sc


def test_hier():
    """
    this regression test suite is based on jupyter notebook examples/ex_shpb_hierarchical.ipynb
    """
    np.random.seed(0)  ## make everything reproducible
    ### Read in Data for Three SHPB Experiments
    dat0 = np.array([
        [0.01, 0.0100583],
        [0.02, 0.010381],
        [0.03, 0.0106112],
        [0.04, 0.0108041],
        [0.05, 0.0109619],
        [0.06, 0.011082],
        [0.07, 0.0112103],
        [0.08, 0.0113348],
        [0.09, 0.0114415],
        [0.1, 0.0115094],
    ])
    dat1 = np.array([
        [0.01, 0.004462],
        [0.02, 0.0045834],
        [0.03, 0.0046879],
        [0.04, 0.0047613],
        [0.05, 0.004824],
        [0.06, 0.0048593],
        [0.07, 0.0049031],
        [0.08, 0.0049449],
        [0.09, 0.0049828],
        [0.1, 0.005015],
        [0.11, 0.0050471],
        [0.12, 0.0050752],
        [0.13, 0.0050964],
        [0.14, 0.0051326],
        [0.15, 0.0051495],
        [0.16, 0.0051803],
        [0.17, 0.0051816],
        [0.18, 0.0052018],
        [0.19, 0.0052262],
        [0.2, 0.0052345],
        [0.21, 0.0052651],
        [0.22, 0.0052795],
        [0.23, 0.005296],
        [0.24, 0.0053118],
        [0.25, 0.0053274],
        [0.26, 0.0053431],
        [0.27, 0.00536],
        [0.28, 0.0053702],
        [0.29, 0.0053762],
        [0.3, 0.0053926],
        [0.31, 0.00541],
        [0.32, 0.0054108],
    ])
    dat2 = np.array([
        [0.005682, 0.0075202],
        [0.012309, 0.0077424],
        [0.020317, 0.0078811],
        [0.02867, 0.0079921],
        [0.036673, 0.0080891],
        [0.045022, 0.0081583],
        [0.05337, 0.0082275],
        [0.061718, 0.0082967],
        [0.070064, 0.008352],
        [0.078411, 0.0084074],
        [0.087102, 0.0084349],
        [0.095447, 0.0084763],
        [0.103792, 0.0085177],
        [0.112137, 0.0085591],
        [0.120482, 0.0086005],
        [0.129173, 0.008628],
        [0.137517, 0.0086555],
        [0.145862, 0.0086969],
        [0.154206, 0.0087244],
        [0.162897, 0.0087519],
        [0.173673, 0.0087794],
        [0.187578, 0.0088206],
        [0.198008, 0.0088619],
        [0.209478, 0.0088754],
        [0.217127, 0.008903],
        [0.225816, 0.0089166],
        [0.234158, 0.0089302],
        [0.243543, 0.0089438],
        [0.250844, 0.0089713],
        [0.259188, 0.0089988],
        [0.26753, 0.0090124],
        [0.276219, 0.009026],
        [0.28491, 0.0090535],
        [0.293252, 0.0090671],
        [0.301594, 0.0090807],
        [0.310284, 0.0090943],
        [0.318973, 0.0091079],
        [0.327316, 0.0091216],
        [0.336005, 0.0091352],
        [0.346434, 0.0091626],
    ])

    # and here are the temperatures and strain rates for the three experiments:
    temp0 = 573.0  # units: Kelvin
    temp1 = 1373.0
    temp2 = 973.0

    edot0 = 800.0  # units: 1/s
    edot1 = 2500.0
    edot2 = 2500.0

    # put the three experiments together in a list
    dat_all = [dat0, dat1, dat2]
    temps = [temp0, temp1, temp2]
    edots = [edot0, edot1, edot2]

    stress_stacked = np.hstack([np.array(v)[:, 1] for v in dat_all])
    strain_hist_list = [np.array(v)[:, 0] for v in dat_all]

    # constants fixed for PTW calibration
    consts_ptw = {
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

    # bounds on PTW input parameters to calibrate
    bounds_ptw = {
        "theta": (0.0001, 0.2),
        "p": (0.0001, 5.0),
        "s0": (0.0001, 0.05),
        "sInf": (0.0001, 0.05),
        "kappa": (0.0001, 0.5),
        "lgamma": (-14.0, -9.0),
        "y0": (0.0001, 0.05),
        "yInf": (0.0001, 0.01),
        "y1": (0.001, 0.1),
        "y2": (0.33, 1.0),
    }

    ntemps = 20
    ## assume we already have a pooled calibration result, we can use it as a starting point for hierarchical:
    best_pool = np.array([
        0.165374,
        0.135012,
        0.71787,
        0.001748,
        0.315716,
        0.033456,
        0.478695,
        0.00509,
        0.73037,
        0.166709,
    ])

    model_ptw = sc.ModelMaterialStrength(
        temps=np.array(temps),
        edots=np.array(edots) * 1e-6,
        consts=consts_ptw,
        strain_histories=strain_hist_list,
        flow_stress_model="PTW_Yield_Stress",
        melt_model="Constant_Melt_Temperature",
        shear_model="Simple_Shear_Modulus",
        specific_heat_model="Constant_Specific_Heat",
        density_model="Constant_Density",
        pool=False,
    )
    s2_ind = np.hstack([
        [j] * len(np.array(xj)) for j, xj in enumerate(dat_all)
    ])  # this is a vector of length len(yobs) with values (0, 1, 2) indicating which experiment corresponds to which part of yobs.
    setup = sc.CalibSetup(bounds_ptw, sc.constraints_ptw)
    setup.addVecExperiments(
        yobs=stress_stacked,  # observation vector, here all experiments stacked into a long vector
        model=model_ptw,  # model that predicts a vector. Here, this is our PTW model, but this could be replaced with an emulator.
        sd_est=np.array(
            [0.0001] * len(dat_all)
        ),  # yobs error estimate (possibly a vector of estimates for different parts of yobs vector)
        s2_df=np.array(
            [15] * len(dat_all)
        ),  # yobs error degrees of freedom (larger means more confidence in sd_est), same shape as sd_est
        s2_ind=s2_ind,
        theta_ind=s2_ind,
    )
    setup.setTemperatureLadder(
        1.05 ** np.arange(ntemps), start_temper=2000
    )  # temperature ladder, typically (1 + step)**np.arange(ntemps)
    setup.setMCMC(
        nmcmc=5000, decor=100
    )  # MCMC number of iterations, and how often to take a decorrelation step
    setup.setHierPriors(
        theta0_prior_mean=best_pool,  # prior mean for theta_0 values
        theta0_prior_cov=np.eye(setup.p)
        * 10**2,  # prior covariances for theta_0 values
        Sigma0_prior_df=setup.p
        + 20,  # degrees of freedom for Inverse Wishart prior for Sigma_0. Generally, larger values indicate greater borrowing across experiments.
        Sigma0_prior_scale=np.eye(setup.p)
        * 0.1
        ** 2,  # scale for Inverse Wishart prior for Sigma_0. Generally, larger values indicate greater borrowing across experiments.
    )
    setup.theta0_start = np.repeat(best_pool.reshape(1, -1), ntemps, axis=0)
    out = sc.calibHier(setup)

    theta_parent = sc.chol_sample_1per_constraints(
        out.theta0[:, 0],
        out.Sigma0[:, 0],
        setup.checkConstraints,
        setup.bounds_mat,
        setup.bounds.keys(),
        setup.bounds,
        setup.constants,
    )
    mcmc_use = np.arange(2500, 5000, 2)  # burn and thin index
    mat = theta_parent[mcmc_use, :]
    pred = setup.models[0].eval(
        sc.tran_unif(np.array(mat), setup.bounds_mat, setup.bounds.keys()),
        pool=True,
    )

    pred_sse = np.sum(
        (pred - np.repeat(setup.ys[0].reshape(1, -1), len(mcmc_use), axis=0))
        ** 2,
        axis=1,
    )
    theta_minsse = mat[np.where(pred_sse == pred_sse.min())[0][0], :]
    theta_minsse_baseline = np.array([
        0.12125296,
        0.889518,
        0.7980942,
        0.00803078,
        0.31638529,
        0.04834673,
        0.49717085,
        0.00753855,
        0.87773007,
        0.03336257,
    ])
    assert np.allclose(theta_minsse, theta_minsse_baseline)
