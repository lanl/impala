from timeit import timeit
#from numpy import array, max as npmax
#from test_c import TestClassIndex, TestClassSlice, TestClassName

setupc = """import physical_models_c as pm
params = {
        'theta' : 0.0183,   'p'     : 3.,
        's0'    : 0.019497, 'sInf'  : 0.002902,
        'kappa' : 0.3276,   'gamma' : 1.e-5,
        'y0'    : 0.002948, 'yInf'  : 0.001730,
        }
consts = {
        'y1'    : 0.094, 'y2'      : 0.575,  'beta'   : 0.25,
        'alpha' : 0.2,   'matomic' : 27.,    'Tmelt0' : 933.,
        'rho0'  : 2.683, 'Cv0'     : 0.9e-5, 'G0'     : 0.70,
        'chi'   : 0.90,
        }
model = pm.MaterialModel(
        flow_stress_model   = pm.PTW_Yield_Stress,
        shear_modulus_model = pm.Simple_Shear_Modulus,
        )
model.initialize(params, consts, 298., 0., 0.)
model.set_history_variables(emax = 0.5, edot = 0.01, Nhist = 100)"""
testc = "model.initialize_state(298.); results = model.compute_state_history()"

setupp = """import physical_models as pm
params = {
    'theta' : 0.0183,   'p'     : 3.,
    's0'    : 0.019497, 'sInf'  : 0.002902,
    'kappa' : 0.3276,   'gamma' : 1.e-5,
    'y0'    : 0.002948, 'yInf'  : 0.001730,
    }
consts = {
    'y1'    : 0.094, 'y2'      : 0.575,  'beta'   : 0.25,
    'alpha' : 0.2,   'matomic' : 27.,    'Tmelt0' : 933.,
    'rho0'  : 2.683, 'Cv0'     : 0.9e-5, 'G0'     : 0.70,
    'chi'   : 0.90,
    }
model = pm.MaterialModel(
    flow_stress_model = pm.PTW_Yield_Stress,
    shear_modulus_model = pm.Simple_Shear_Modulus,
    )
model.initialize(params, consts)
model.initialize_state(298.)
shist = pm.generate_strain_history(emax = 0.5, edot = 0.01, Nhist = 100)"""
testp = "model.initialize_state(298.); results = model.compute_state_history(shist)"

setup_gpy = """from test_gpy import submodel, xtry2"""
test_gpy = """sse = submodel.sse(xtry2)"""


if __name__ == '__main__':
    # reference by name wins; assign by index is only slightly slower.  slice is much slower.
    #print(timeit("x = TestClassName(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)[0:8]); x.remath()",  setup = "from test_c import TestClassName; from numpy import array, float64", number =  100000))
    #print(timeit("x = TestClassIndex(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)[0:8]); x.remath()", setup = "from test_c import TestClassIndex; from numpy import array, float64", number = 100000))
    #print(timeit("x = TestClassSlice(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)[0:8]); x.remath()", setup = "from test_c import TestClassSlice; from numpy import array, float64", number = 100000))
    #print(timeit("x = TestClassIndex2(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)[0:8]); x.remath()", setup = "from test_c import TestClassIndex2; from numpy import array, float64", number = 100000))
    #print(timeit('x = max(a,b)',      setup = 'import numpy as np; a = np.float64(2); b = np.float64(1)'))
    #print(timeit('x = np.max([a,b])', setup = 'import numpy as np; a = np.float64(2); b = np.float64(1)'))

    #print(timeit("x = TestClassName(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)); x.remath()",  setup = "from test import TestClassName; from numpy import array, float64", number = 100000))
    #print(timeit("x = TestClassIndex(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)); x.remath()", setup = "from test import TestClassIndex; from numpy import array, float64", number = 100000))
    #print(timeit("x = TestClassSlice(array([1.,2.,3.,4.,5.,6.,7.,8.], dtype = float64)); x.remath()", setup = "from test import TestClassSlice; from numpy import array, float64", number = 100000))
    #setup = "from pointcloud import localssq1, localssq2, localssq3, localssq4; import numpy as np; from numpy.random import normal; x1 = normal(0,3, 100000); x2 = normal(0.7*x1 + 1, 3); x3 = normal(0.2*x2 + 0.3*x1, 3); x4 = normal(0.2*x3 + 0.3*x2, 3);  x5 = normal(0.1*x4 + 0.7*x3, 3);  x = np.vstack((x1,x2,x3,x4,x5)).T; target = np.mean(x, axis = 0)"
    #print(timeit("res = localssq1(x, target, 0.5)", setup = setup, number = 10000))
    #print(timeit("res = localssq2(x, target, 0.5)", setup = setup, number = 10000))
    #print(timeit("res = localssq3(x, target, 0.5)", setup = setup, number = 10000))
    #print(timeit("res = localssq4(x, target, 0.5)", setup = setup, number = 10000))
    #print(timeit(testc, setup = setupc, number = 10000))
    #print(timeit(testp, setup = setupp, number = 10000))

    print(timeit(test_gpy, setup = setup_gpy, number = 100000))


# EOF
