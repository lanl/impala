"""
test_pt_mpi_cu.py

Run the MPI-parallelized parallel tempering program for Copper
"""
from mpi4py import MPI
from physical_models_c import JCYieldStress, SimpleShearModulus
from statistical_models_mpi import RemoteChain, ParallelTemperingMaster, \
    BreakException, Dispatcher
from numpy import array, float64
import os

basepath = os.path.expanduser('~')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank > 0:     # If Not on root node
   dispatcher = Dispatcher(comm, rank)
   dispatcher.watch()

elif rank == 0:  # If on root node
    if True: # Hopkinson Bar Inputs
        paths_hb = [
            './copper/CuRT203.txt',
            './copper/Cu20203.txt',
            './copper/Cu40203.txt',
            './copper/Cu60203.txt',
            './copper/CuRT10-1.SRC.txt',
            './copper/CuRT10-3.SRC.txt',
            ]
        temps_hb = array([298., 473., 673., 873., 298., 298.], dtype = float64)
        edots_hb = array(
            [2000., 2000., 2000., 2000., 0.1, 0.001],
            dtype = float64) * 1.e-6
        xp_hb = [
            {'path' : x, 'temp' : y, 'edot' : z, 'emax' : 0.65, 'Nhist' : 100}
            for x,y,z in zip(paths_hb, temps_hb, edots_hb)
            ]
    if True: # Taylor Cylinder Inputs
        paths_tc = {
            'path_x' : './copper/inputs_sim_tc.csv',
            'path_y' : './copper/outputs_sim_tc.csv',
            'path_y_actual' : './copper/outputs_real_tc.csv',
            }
        xp_tc = [paths_tc]
    if True: # Flyer Plate Inputs
        paths_fp = {
            'path_x' : './copper/inputs_sim_fp.txt',
            'path_y' : './copper/outputs_sim_fp.csv',
            'path_y_actual' : './copper/outputs_real_fp.csv'
            }
        xp_fp = [paths_fp]

    # Values for Constants supplied to sampler
    starting_consts = {
            'Tref' : 298.,  'Tmelt0' : 933.,   'edot0' : 1.e-6,
            'rho0' : 2.683, 'Cv0'    : 0.9e-5, 'G0'    : 0.7,
            'chi'  : 0.9,
            }
    # Sampling Bounds - prior Uniform
    parameter_bounds = {
            'A'   : (0.0001, 0.0100), 'B'   : (0.0001, 0.0100),
            'C'   : (0.0002, 0.0300), 'n'   : (0.0010, 1.5000),
            'm'   : (0.0050, 3.0000), 'vel' : (0.0300, 0.0315),
            'sm0' : (0.3000, 0.6000), 'phi' : (0.0000, 1.0000),
            }
    # Declare the model
    model = ParallelTemperingMaster(
        comm        = comm,
        size        = size,
        temp_ladder = 1.5 ** array(range(size - 1)),
        xp_hb       = xp_hb,
        xp_tc       = xp_tc,
        xp_fp       = xp_fp,
        W           = array([1., 1., 1.]) / 3,
        psi0        = 1e-3,
        bounds      = parameter_bounds,
        constants   = starting_consts,
        flow_stress_model   = JCYieldStress,
        )

    # Sampling length
    nburn = int(4e4)   # Burn-in after beginning tempering
    tburn = 0          # Burn-in before tempering
    nsamp = int(2e4)   # Number of posterior samples
    thin  = 1          # Thinning

    # Run the sampler
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
    thetas = model.get_history_theta(
        nsamp + nburn + tburn,
        len(model.parameter_order),
        )
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
