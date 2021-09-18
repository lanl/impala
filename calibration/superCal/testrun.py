# %%
import plots
import numpy as np
import pandas as pd
import sqlite3 as sq
import impala as impala
import models as models
np.seterr(under='ignore')
nExp = 40
## get meta data
con = sq.connect('./../py_calibration_hier/data/data_Ti64.db')
meta = pd.read_sql('select * from meta;', con)
edots = meta.edot[:nExp].values
temps = meta.temperature[:nExp].values
## get SHPB experimental data
dat_all = []
for i in range(nExp):
    ## get first datset
    dat_all.append(pd.read_sql('select * from data_{};'.format(i + 1), con).values)
yobs = np.hstack([v.T[1] for v in dat_all])
sh = [v.T[0] for v in dat_all]
## Constants and Bounds
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
    #'gamma' : (0.000001, 0.0001),
    'lgamma' : (-14., -9.),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.33,      1.),
    'vel'   : (0.99,     1.01),
    }
## constraints: sInf < s0, yInf < y0, y0 < s0, yInf < sInf, s0 < y1, beta < y2
def cf(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

#%% Common setup
sd_est = np.array([.001]*nExp)
s2_df = np.array([5]*nExp)
s2_ind = np.hstack([[v]*len(dat_all[v]) for v in list(range(nExp))])

#%%  Pooled Model 
# model = models.ModelPTW(temps, edots, consts, sh, False) # True)
# setup = impala.CalibSetup(bounds, cf)
# setup.addVecExperiments(yobs, model, sd_est, s2_df, s2_ind)
# setup.setTemperatureLadder(1.1**np.arange(20))
# setup.setMCMC(30000,10000,1,100)
# out = impala.calibPool(setup)
# plot1 = plots.PTW_Plotter(setup, out)
# plot1.ptw_prediction_plots('./pool_predictions.pdf')
# plot1.pairwise_theta_plot('./pool_pairwise.pdf')

#%%  Hierarchical Model
# model2 = models.ModelPTW(temps, edots, consts, sh, False)
# setup2 = impala.CalibSetup(bounds, cf)
# setup2.addVecExperiments(yobs, model2, sd_est, s2_df, s2_ind, theta_ind = s2_ind.copy())
# setup2.setTemperatureLadder(1.1**np.arange(20))
# setup2.setMCMC(30000, 10000, 1, 100)
# out2 = impala.calibHier(setup2)
# plot2 = plots.PTW_Plotter(setup2, out2)
# plot2.ptw_prediction_plots('./hier_predictions.pdf')
# plot2.pairwise_theta_plot('./hier_pairwise.pdf')

#%%  Clustered Model
if __name__ == '__main__':
    setup = impala.CalibSetup(bounds, cf)
    model = models.ModelPTW(temps, edots, consts, sh, False)
    setup.addVecExperiments(yobs, model, sd_est, s2_df, s2_ind, theta_ind = s2_ind.copy())
    # for i in range(nExp):
    #     model = models.ModelPTW(
    #          temps[i:(i+1)], edots[i:(i+1)], consts, dat_all[i].T[0].reshape(1,-1), False
    #           )
    #     setup.addVecExperiments(
    #           dat_all[i].T[1], model, np.array([0.001]), 
    #           np.array([10]), np.zeros(dat_all[i].T[1].shape[0]),
    #           )
    setup.setTemperatureLadder(1.05**np.arange(10))
    setup.setMCMC(30000, 10000, 1, 100)
    setup.set_max_clusters(50)
    out  = impala.calibClust(setup)
    plot = plots.PTW_Plotter(setup, out)
    plot.ptw_prediction_plots('./clust_predictions.pdf')
    plot.pairwise_theta_plot('./clust_pairwise.pdf')
    plot.cluster_matrix_plot('./clust_clusters.pdf')

# EOF 