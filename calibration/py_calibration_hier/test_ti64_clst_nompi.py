import numpy as np
import time
from numpy import array, float64
np.seterr(under = 'ignore')
#from mpi4py import MPI

import sm_dpcluster as sm
import pt
#import pt_mpi as pt
#pt.MPI_MESSAGE_SIZE = 2**13
sm.POOL_SIZE = 2


#comm = MPI.COMM_WORLD
rank = 0#comm.Get_rank()
size = 2#comm.Get_size()

material = 'Ti64'

# Defining Paths, Constants, Parameter Ranges
if True:
    if material == 'Al5083':
        path = './data/data_Al5083.db'
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
        path = './data/data_copper.db'
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
        path = './data/data_Ti64.db'
        starting_consts = {
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
        parameter_bounds = {
            'theta0' : (0.0001,   0.2),
            'p'     : (0.0001,   5.),
            's0'    : (0.0001,   0.05),
            'sInf'  : (0.0001,   0.05),
            'kappa' : (0.0001,   0.5),
            'gamma' : (0.000001, 0.0001),
            'y0'    : (0.0001,   0.05),
            'yInf'  : (0.0001,   0.01),
            'y1'    : (0.001,    0.1),
	    'y2'    : (0.3,      1.),
            'vel'   : (0.99,     1.01),
            }

if __name__ == '__main__':
    if rank > 0:
        chain = pt.PTSlave(sm.Chain)
        chain.watch()

    elif rank == 0:
        model = pt.PTMaster(
            statmodel=sm.Chain,
            temperature_ladder = 1.2 ** array(range(size - 1)),
            path       = path,
            bounds     = parameter_bounds,
            constants  = starting_consts,
            model_args = {'flow_stress_model'   : 'PTW', 'shear_modulus_model' : 'Stein'},
            )
        model.sample(400, 1)
        model.write_to_disk('./results/Ti64/res_ti64_clst.db', 200, 1)
        model.plot_accept_probability('./results/Ti64/res_ti64_clst_accept.png', 200)
        model.plot_swap_probability('./results/Ti64/res_ti64_clst_swapped.png', 200 // 5)
        model.complete()

# EOF
