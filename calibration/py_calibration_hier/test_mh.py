from statistical_models import MetropolisHastingsModel, Transport, import_strain_curve
from physical_models import PTW_Yield_Stress
from numpy import array, apply_along_axis

if __name__ == '__main__':
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
    temps = array([-196.,  125.,  125., -196.,
                        25.,  100.,  200.,   25.,]) + 273.15
    edots = array([0.001, 0.001,   0.1,  2000,
                       2500,  3000,  3500,  7000,]) * 1.e-6
    datas = [import_strain_curve(path) for path in paths]

    transports = [
            Transport(data = x, temp = y, emax = 0.5, edot = z, Nhist = 100)
            for x,y,z in zip(datas, temps, edots)
            ]

    # build the material starting parameters
    starting_params = {
            'theta' : 0.0183,   'p'     : 3.,
            's0'    : 0.019497, 'sInf'  : 0.002902,
            'kappa' : 0.3276,   'gamma' : 1.e-5,
            'y0'    : 0.002948, 'yInf'  : 0.001730,
            }
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

    model = MetropolisHastingsModel(
            transports = transports,
            bounds     = parameter_bounds,
            params     = starting_params,
            constants  = starting_consts,
            flow_stress_model = PTW_Yield_Stress,
            )
    res = model.sample(10000, 5000, 10)
    model.prediction_plot(res[0],res[1])

# EOF
