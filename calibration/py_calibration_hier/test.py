from physical_models_c import PTWYieldStress, SimpleShearModulus
from statistical_models_hier_mpi import ParallelTemperMaster, Dispatcher
#from statistical_models_hier import ParallelTemperMaster
from numpy import array, float64
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# rank = 0
# size = 6

if rank > 0:
    pass
    dispatcher = Dispatcher(comm, rank)
    dispatcher.watch()

elif rank == 0:
    # if True:
    #     paths_hb = [
    #        './copper/CuRT203.txt',
    #        './copper/Cu20203.txt',
    #        './copper/Cu40203.txt',
    #        './copper/Cu60203.txt',
    #        './copper/CuRT10-1.SRC.txt',
    #        './copper/CuRT10-3.SRC.txt',
    #        ]
    #     temps_hb = array([298., 473., 673., 873., 298., 298.], dtype = float64)
    #     edots_hb = array([2000., 2000., 2000., 2000., 0.1, 0.001], dtype = float64) * 1.e-6
    #     xps_hb = [
    #     {'path' : x, 'temp' : y, 'edot' : z, 'emax' : 0.65, 'Nhist' : 100}
    #         for x,y,z in zip(paths_hb, temps_hb, edots_hb)
    #        ]

    # parameter_bounds = {
    #         'theta' : (1e-3, 0.1),
    #         'p'     : (9e-3, 10.),
    #         's0'    : (3e-3, 0.05),
    #         'sInf'  : (1e-3, 0.05),
    #         'y0'    : (6.8e-6, 0.05),
    #         'yInf'  : (6.5e-3, 0.04),
    #         'beta'  : (1.e-1, 0.35),
    #         'kappa' : (1e-6, 1.0),
    #         'gamma' : (1e-6, 0.1),
    #         'vel'   : (3e-2, 0.03),
    #         }
    # starting_consts = {
    #         'alpha'  : 0.2,    'matomic' : 63.546, 'Tref' : 298.,
    #         'Tmelt0' : 1358.,  'rho0'    : 8.96,   'Cv0'  : 0.385e-5,
    #         'G0'     : 0.70,   'chi'     : 0.95,   'beta' : 0.33,
    #         'y1'     : 0.0245, 'y2'      : 0.33,
    #         }
    starting_consts = {
            'alpha'  : 0.2,
            'y1'     : 0.0245,
            'y2'     : 0.33,
            'beta'   : 0.33,
            'matomic': 45.9,
            'Tmelt0' : 2110.,
            'rho0'   : 4.419,
            'Cv0'    : 0.525e-5,
            'G0'     : 0.4,
            'chi'    : 1.0,
            'sgB'    : 6.44e-4
            }
    parameter_bounds = {
            'theta' : (0.0001,   0.2),
            'p'     : (0.0001,   5.),
            's0'    : (0.0001,   0.05),
            'sInf'  : (0.0001,   0.05),
            'kappa' : (0.0001,   0.5),
            'gamma' : (0.000001, 0.0001),
            'y0'    : (0.0001,   0.05),
            'yInf'  : (0.0001,   0.01),
            }

    # Define the Model
    model = ParallelTemperMaster(
        comm = comm,
        size = size,
        temperature_ladder = 1.3 ** array(range(size - 1)),
        path = './data_Ti64.db',
        bounds = parameter_bounds,
        constants = starting_consts,
        flow_stress_model = PTWYieldStress,
        shear_modulus_model = SimpleShearModulus,
        )
    model.sample(10000,10000)
    model.write_to_disk('Ti64_results.db')
    theta0 = model.invprobit(model.get_history(0,1))
    model.parameter_pairwise_plot(theta0, 'Ti64_pairwise.png')
    model.parameter_trace_plot(theta0, 'Ti64_trace.png')
    model.complete()

if __name__ == '__main__':
    pass

# EOF
