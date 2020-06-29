from mpi4py import MPI
from physical_models_c import PTWYieldStress, SimpleShearModulus
from statistical_models_mpi import RemoteChain, ParallelTemperingMaster, \
    BreakException, Dispatcher
from numpy import array, float64

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank > 0:    # if not on root node
   dispatcher = Dispatcher(comm, rank)
   dispatcher.watch()

elif rank == 0: # if on root node
    if True: # Hopkinson Bar setup
        paths_hb = [
            './copper/CuRT203.txt',
            './copper/Cu20203.txt',
            './copper/Cu40203.txt',
            './copper/Cu60203.txt',
            './copper/CuRT10-1.SRC.txt',
            './copper/CuRT10-3.SRC.txt',
            ]
        temps_hb = array([298., 473., 673., 873., 298., 298.], dtype = float64)
        edots_hb = array([2000., 2000., 2000., 2000., 0.1, 0.001], dtype = float64) * 1.e-6
        xp_hb = [
            {'path' : x, 'temp' : y, 'edot' : z, 'emax' : 0.65, 'Nhist' : 100}
            for x,y,z in zip(paths_hb, temps_hb, edots_hb)
            ]
    if True: # Taylor Cylinder setup
        paths_tc = {
            'path_x' : './copper/inputs_sim_tc.csv',
            'path_y' : './copper/outputs_sim_tc.csv',
            'path_y_actual' : './copper/outputs_real_tc.csv',
            }
        xp_tc = [paths_tc]
    if True: # Flyer Plate setup
        paths_fp = {
            'path_x' : './copper/inputs_sim_fp.txt',
            'path_y' : './copper/outputs_sim_fp.csv',
            'path_y_actual' : './copper/outputs_real_fp.csv'
            }
        xp_fp = [paths_fp]

    # Constants and Parameter Bounds
    starting_consts = {
            'alpha'  : 0.2,   'matomic' : 63.546, 'Tref' : 298.,
            'Tmelt0' : 1358., 'rho0'    : 8.96,   'Cv0'  : 0.385e-5,
            'G0'     : 0.70,  'chi'     : 0.95,
            }
    parameter_bounds = {
            'theta' : (1e-3, 0.1),
            'p'     : (9e-3, 10.),
            's0'    : (3e-3, 0.05),
            'sInf'  : (1e-3, 0.05),
            'y0'    : (6.8e-6, 0.05),
            'yInf'  : (6.5e-3, 0.04),
            'beta'  : (1.e-1, 0.35),
            'kappa' : (1e-6, 1.0),
            'gamma' : (1e-6, 0.1),
            'vel'   : (3e-2, 0.03),
            }

    # Declare Model
    model = ParallelTemperingMaster(
        comm        = comm,
        size        = size,
        temp_ladder = 1.5 ** array(range(size - 1)),
        xp_hb       = xp_hb,
        xp_tc       = xp_tc,
        xp_fp       = xp_fp,
        W           = array([1., 1., 1.]) / 3,
        psi0        = 1e-6,
        bounds      = parameter_bounds,
        constants   = starting_consts,
        flow_stress_model   = PTWYieldStress,
        shear_modulus_model = SimpleShearModulus,
        )

    # Run the sampler
    nburn = int(4e4)
    tburn = 0
    nsamp = int(2e4)
    thin  = 1
    model.sampler(nsamp, nburn, tburn, thin)

    # Print within-chain Acceptance Probability
    accept_prob_list = ', '.join('{:.3f}'.format(p) for p in model.get_accept_prob())
    print('Accept Prob: {}'.format(accept_prob_list))

    # Print Swap Probability Matrix
    print('Swap Matrix')
    print(model.get_swap_prob())

    # Write all the chains sampling history to disk
    model.write_to_disk()

    # Get the sampling history from each of the chains
    thetas = model.get_history_theta(nsamp + nburn + tburn, len(model.parameter_order))
    sigma2_hb = model.get_history_sigma2_hb(nsamp + nburn + tburn)
    sigma2_tc = model.get_history_sigma2_tc(nsamp + nburn + tburn)
    sigma2_fp = model.get_history_sigma2_fp(nsamp + nburn + tburn)

    # Plot Posterior Pairwise plot
    model.parameter_pairwise_plot(
        thetas[0][-nsamp:],
        os.path.join(basepath, 'plots','pairwise.png')
        )
    # Plot Posterior Trace Plot
    model.parameter_trace_plot(
        thetas[0][-nsamp:],
        os.path.join(basepath, 'plots', 'trace.png')
        )
    # Plot Posterior Predictions for Hopkinson Bar
    model.prediction_plot_hb(
        thetas[0][-nsamp:],
        sigma2_hb[0][-nsamp:],
        os.path.join(basepath, 'plots', 'prediction_hb.png'),
        (0., 0.004)
        )
    # Plot Posterior Prediction for Taylor Cylinder
    model.prediction_plot_tc(
        thetas[0][-nsamp:],
        sigma2_tc[0][-nsamp:],
        os.path.join(basepath, 'plots', 'prediction_tc.png')
        )
    # Plot Posterior Prediction for Flyer Plate
    model.prediction_plot_fp(
        thetas[0][-nsamp:],
        sigma2_fp[0][-nsamp:],
        os.path.join(basepath, 'plots', 'prediction_fp.png')
        )
    # Plot Chain Swap Probability
    model.swap_prob_plot(os.path.join(basepath, 'plots', 'swap_matrix.png'))
    # Plot within-chain acceptance probability
    model.accept_prob_plot(os.path.join(basepath, 'plots', 'accept_probability.png'))
    model.complete()

# EOF
