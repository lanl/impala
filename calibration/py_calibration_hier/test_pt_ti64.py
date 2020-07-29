from statistical_models import ParallelTemperingModel, Transport, import_strain_curve
from physical_models import PTW_Yield_Stress, Stein_Shear_Modulus
from numpy import array
import glob
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
import itertools
import dill



def calib_use_list(i, use_list, lookup):
    use = use_list[i]
    temps = array(lookup['temp'][use]) # kelvin
    edots = array(lookup['edot'][use]) * 1.e-6 # centimeter-gram-microsecond units (FLAG default), inside parens is 1/s
    paths = []
    for i in use:
        paths.append(base_path + lookup['paper'][i] + '/' + lookup['file'][i])

    datas = []
    for path in paths:
        data = pd.read_csv(path, skiprows = range(0,17)).values
        data[:,1] = data[:,1] * 1.e-5 # this is going from MPa to Mbar
        datas.append(data[1:,:])

    transports = [
        Transport(data = x, temp = y, emax = 0.7, edot = z, Nhist = 100)
        for x,y,z in zip(datas, temps, edots)
        ]
    starting_consts = {
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

    model = ParallelTemperingModel(
            temp_ladder = 1.3 ** array(range(6)),
            transports  = transports,
            bounds      = parameter_bounds,
            constants   = starting_consts,
            flow_stress_model   = PTW_Yield_Stress,
            shear_modulus_model = Stein_Shear_Modulus,
            )
    res = model.sample(7000, 3000, 2000, 4) # nsamp, nburn, tburn, thin
    return [res, model]


if __name__ == '__main__':

    dill.settings['recurse'] = True

    base_path = '../../data/ti-6al-4v/Data/SHPB/'
    papers = glob.glob(base_path + '*')
    papers = [temp.split('/')[-1] for temp in papers]

    index = []
    temps = []
    edots = []
    fname = []
    pname = []
    k = 0
    for paper in papers:
        paths = glob.glob(base_path + paper + '/*.Plast.csv')
        for pp in paths:
            f=open(pp)
            lines=f.readlines()
            temps.append(float(lines[1].split(" ")[2])) # Kelvin
            edots.append(float(lines[2].split(" ")[3])) # 1/s
            temp = pp.split("/")
            fname.append(temp[-1])
            pname.append(temp[-2])
            index.append(k)
            k = k + 1

    lookup = pd.DataFrame(list(zip(index, temps, edots, fname, pname)),
                        columns = ['index', 'temp', 'edot', 'file', 'paper'])


    use_list = list(map(lambda el:[el], list(range(len(lookup))))) # list of combinations


    use = list(range(len(lookup))) # which of use_list to actually do

    # use = [0,2,37]
    # res_list = list(map(calib_use_list, use, itertools.repeat(use_list, len(use)),itertools.repeat(lookup, len(use))))

    pool = Pool()
    res_list = list(pool.map(calib_use_list, use, itertools.repeat(use_list, len(use)),itertools.repeat(lookup, len(use))))
    pool.close()
    pool.join()


    with open("store_results.dill", "wb") as dill_file:
        dill.dump(res_list, dill_file)

    #lookup.to_csv('file_index.csv', encoding='utf-8', index=False)
# =============================================================================
#     i = 1
#     res_list[i][1].prediction_plot(res_list[i][0][0], res_list[i][0][1])
#     res_list[i][1].plot_swap_matrix()
#     res_list[i][1].parameter_pairwise_plot(res_list[i][0][0])
#     res_list[i][1].parameter_trace_plot(res_list[i][0][0])
#     res_list[i][1].parameter_trace_plot(res_list[i][0][1])
# =============================================================================

    #with open("store_results.dill", "rb") as dill_file:
    #    res_list = dill.load(dill_file)

#EOF
