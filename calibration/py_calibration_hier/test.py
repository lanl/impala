import numpy as np
import time
from numpy import array, float64
np.seterr(under = 'ignore')
from mpi4py import MPI

#import sm_dpcluster as sm
import sm_pooled as sm
# import pt
import pt_mpi as pt
pt.MPI_MESSAGE_SIZE = 2**12
sm.POOL_SIZE = 8


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# rank = 0
# size = 4

material = 'Al5083'

# Defining Paths, Constants, Parameter Ranges
if True:
    if material == 'Al5083':
        path = './data_Al5083.db'
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
    if material == 'copper':
        path = './data_copper.db'
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
        starting_consts = {
            'alpha'  : 0.2,    'matomic' : 63.546, 'Tref' : 298.,
            'Tmelt0' : 1358.,  'rho0'    : 8.96,   'Cv0'  : 0.385e-5,
            'G0'     : 0.70,   'chi'     : 0.95,   'beta' : 0.33,
            'y1'     : 0.0245, 'y2'      : 0.33,
            }
    if material == 'Ti64':
        path = './data_Ti64.db'
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

if __name__ == '__main__':
    if rank > 0:
        pass
        chain = pt.PTSlave(comm = comm, statmodel = sm.Chain)
        chain.watch()

    elif rank == 0:
        model = pt.PTMaster(
            comm,
            # statmodel = sm.Chain,
            temperature_ladder = 1.3 ** array(range(size - 1)),
            path       = path,
            bounds     = parameter_bounds,
            constants  = starting_consts,
            # model_args = {'flow_stress_model'   : 'PTW', 'shear_modulus_model' : 'Stein'},
            model_args = {'flow_stress_model'   : 'PTW', 'shear_modulus_model' : 'Simple'},
            )
        model.sample(20000, 5)
        #model.write_to_disk('results_pool_Al5083.db', 20001, 5)
        # model.plot_accept_probability('results_cluster_ti64_accept.png')
        # model.plot_swap_probability('results_cluster_ti64_swapped.png')
        # model.complete()

# EOF
