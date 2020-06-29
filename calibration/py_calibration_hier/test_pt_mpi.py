from mpi4py import MPI
from physical_models_c import PTWYieldStress, SimpleShearModulus
from statistical_models_mpi import RemoteChain, ParallelTemperingMaster, \
    BreakException, Dispatcher
from submodel import SubModelHB, SubModelTC, SubModelFP
from transport import TransportHB, TransportTC, TransportFP
from numpy import array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank > 0:
   dispatcher = Dispatcher(comm, rank)
   dispatcher.watch()

elif rank == 0:
    paths = [
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T-196C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T125C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_1T125C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S2000T-196C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S2500T25C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S3000T100C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S3500T200C.csv',
        '../Al-5083/Stress-Strain_Data/Gray94_Al5083_S7000T25C.csv',
        ]
    temps = array([ -196.,  125.,  125., -196., 25., 100., 200., 25.]) + 273.15
    edots = array([0.001, 0.001, 0.1, 2000, 2500, 3000, 3500, 7000,]) * 1.e-6
    transports_hb = [
            TransportHB(path = x, temp = y, emax = 0.5, edot = z, Nhist = 100)
            for x,y,z in zip(paths, temps, edots)
            ]
    starting_consts = {
            'y1'     : 0.094, 'y2'      : 0.575, 'beta' : 0.25,
            'alpha'  : 0.2,   'matomic' : 27.,   'Tref' : 298.,
            'Tmelt0' : 933.,  'rho0'    : 2.683, 'Cv0'  : 0.9e-5,
            'G0'     : 0.70,  'chi'     : 0.90,
            }
    parameter_bounds = {
            'theta' : (0.0001,   0.05),
            'p'     : (0.0001,   5.),
            's0'    : (0.0001,   0.05),
            'sInf'  : (0.0001,   0.005),
            'kappa' : (0.0001,   0.5),
            'gamma' : (0.000001, 0.0001),
            'y0'    : (0.0001,   0.005),
            'yInf'  : (0.0001,   0.005),
            }
    model = ParallelTemperingMaster(
        comm = comm,
        size = size,
        temp_ladder = 1.25 ** array(range(24)),
        xp_hb = transports_hb,
        xp_tc = [],
        xp_fp = [],
        bounds = parameter_bounds,
        constants = starting_consts,
        flow_stress_model = PTWYieldStress,
        shear_modulus_model = SimpleShearModulus,
        )

    nburn = int(4e4)
    nsamp = int(2e4)
    tburn = 0
    thin  = 1

    model.sampler(nsamp, nburn, tburn, 1)
    # history = model.get_history()
    # samples = model.get_samples(nburn, thin)

    print(
        'Acceptance Probabilities, {}'.format(
            ','.join('{:.3f}'.format(p) for p in model.get_accept_prob())
            )
        )
    print('Swap Matrix')
    print(model.get_swap_prob())

    model.write_to_disk()

    # thetas = model.get_history_theta(
    #         nsamp + nburn + tburn,
    #         len(model.parameter_order),
    #         )
    # sigma2s = model.get_history_sigma2(nsamp + nburn + tburn)

    # model.parameter_pairwise_plot(thetas[0], '~/plots/pairwise.png')
    # model.parameter_trace_plot(thetas[0], '~/plots/trace.png')
    # model.prediction_plot_hb(thetas[0], sigma2s[0], '~/plots/prediction.png')

    model.complete()
# EOF
