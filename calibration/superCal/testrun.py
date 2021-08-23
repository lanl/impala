# %%
from plots import PTW_Plotter
import numpy as np
import pandas as pd
import sqlite3 as sq
# import pyBASS as pb
import impala_test3 as impala
import models_test as models

nExp = 40

#%%
## get meta data
con = sq.connect('./../py_calibration_hier/data/data_Ti64.db')
meta = pd.read_sql('select * from meta;', con)
edots = meta.edot[:nExp]
temps = meta.temperature[:nExp]

## get SHPB experimental data
dat_all = []
for i in range(nExp):
    ## get first datset
    dat_all.append(pd.read_sql('select * from data_{};'.format(i + 1), con).values)

consts = {
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
bounds = {
    'theta' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'gamma' : (0.000001, 0.0001),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.33,      1.),
    'vel'   : (0.99,     1.01),
    }

# constraints: sInf < s0, yInf < y0, y0 < s0, yInf < sInf, s0 < y1, beta < y2
def cf(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

setup = impala.CalibSetup(bounds, cf)

#%%
yobs = np.hstack([v.T[1] for v in dat_all])
sh = [v.T[0] for v in dat_all]
model = models.ModelPTW(np.array(meta.temperature[0:nExp]), np.array(meta.edot[0:nExp]), consts, sh, False)
sd_est = np.array([.001]*nExp)
s2_df = np.array([5]*nExp)
s2_ind = np.hstack([[v]*len(dat_all[v]) for v in list(range(nExp))])

setup.addVecExperiments(yobs, model, sd_est, s2_df, s2_ind, theta_ind = s2_ind.copy())
setup.setTemperatureLadder(1.1**np.arange(20))
setup.setMCMC(30000,10000,1,100)

np.seterr(under='ignore')
#%%

out = impala.calibHier(setup)

plots = PTW_Plotter(setup, out)
plots.ptw_prediction_plots('./predictions.pdf')
plots.pairwise_theta_plot('./pairwise.pdf')

#%%



# EOF 