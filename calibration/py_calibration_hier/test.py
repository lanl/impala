from physical_models_c import PTWYieldStress, SimpleShearModulus
from statistical_models_hier_redux import Chain, SubChainHB, ChainPlaceholder
from numpy import array, float64

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

if __name__ == '__main__':
    subchain = SubChainHB(
        xp = xp_hb[0],
        parent = ChainPlaceholder(),
        bounds = parameter_bounds,
        constants = starting_consts,
        flow_stress_model  = PTWYieldStress,
        shear_modulus_model = SimpleShearModulus,
        )
    subchain.initialize_sampler(10000, array([0.]*8))

# EOF
