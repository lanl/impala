##########################################################################################
## This script demonstrates how to do pooled impala calibration of PTW
## using SHPB/quasistatic data.
##########################################################################################

import numpy as np
from impala import superCal as sc
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3 as sq
np.seterr(under='ignore')

##########################################################################################
## Three Ti64 experiments, 1st column is plastic strain 2nd column is stress (units: mbar)


## get meta data
con = sq.connect('../../../Desktop/data_Ti64.db')
meta = pd.read_sql('select * from meta;', con)
n_tot = meta.shape[0]
n_shpb = np.sum(meta['type']=='shpb')
n_tc = np.sum(meta['type']=='tc')
edots = np.hstack(meta.edot[:n_shpb].values)
temps = np.hstack(meta.temperature[:n_shpb].values)
## get SHPB experimental data
dat_all = []
for i in range(n_shpb):
    ## get first datset
    dat_all.append(pd.read_sql('select * from data_{};'.format(i + 1), con).values)
stress_stacked = np.hstack([v.T[1] for v in dat_all])
strain_hist_list = [v.T[0] for v in dat_all]

for i in range(n_shpb):
    plt.plot(dat_all[i][:,0],dat_all[i][:,1])
plt.show()


import pyBASS as pb
emu_list = []
y_emu_list = []
for i in range(n_shpb, n_tot, 1):
    X = pd.read_sql('select * from sims_input_{};'.format(i + 1), con).values
    y = pd.read_sql('select * from sims_output_{};'.format(i + 1), con).values
    mod = pb.bassPCA(X, y, ncores=15, percVar=99.9)
    emu_list.append(mod)
    yobs = pd.read_sql('select * from data_{};'.format(i + 1), con).values
    y_emu_list.append(yobs)

emu_list[4].plot()
##########################################################################################
# set constants and parameter bounds
consts_ptw = {
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
bounds_ptw = {
    'theta' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'lgamma': (-14.,     -9.),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.33,      1.),
    }



##########################################################################################
# constraints: sInf < s0, yInf < y0, y0 < s0, yInf < sInf, s0 < y1, beta < y2 (but beta is fixed)
def constraints_ptw(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good



##########################################################################################
# these define measurement error estimates, I would leave these as is for most SHPB/quasistatic data
sd_est = np.array(([.0001]*n_shpb + [.001]*n_tc))
s2_df = np.array([15]*n_tot)
s2_ind = np.hstack([[v]*len(dat_all[v]) for v in list(range(n_shpb))] + [[v + n_shpb]*len(y_emu_list[v]) for v in list(range(n_tc))])

input_names_emu = list(bounds_ptw.keys()) + ['vel']
models_emu = []
for i in range(n_tc):
    models_emu.append(sc.ModelBassPca_func(emu_list[i], input_names_emu))

##########################################################################################
# define PTW model

model_ptw = sc.ModelMaterialStrength(temps=np.array(temps), 
    edots=np.array(edots), 
    consts=consts_ptw, 
    strain_histories=strain_hist_list, 
    flow_stress_model='PTW_Yield_Stress',
    melt_model='Constant_Melt_Temperature', 
    shear_model='Simple_Shear_Modulus', 
    specific_heat_model='Constant_Specific_Heat', 
    density_model='Constant_Density',
    pool=False)
# bring everything together into calibration structure
setup_pool_ptw = sc.CalibSetup(bounds_ptw, constraints_ptw)
setup_pool_ptw.addVecExperiments(yobs=stress_stacked, 
    model=model_ptw, 
    sd_est=sd_est, 
    s2_df=s2_df, 
    s2_ind=s2_ind)
setup_pool_ptw.setTemperatureLadder(1.05**np.arange(50), start_temper=2000)
setup_pool_ptw.setMCMC(nmcmc=10000, nburn=5000, thin=1, decor=100, start_tau_theta=-4.)

setup_pool_jc = sc.CalibSetup(bounds_jc, constraints_jc)
setup_pool_jc.addVecExperiments(yobs=stress_stacked, 
    model=model_jc, 
    sd_est=sd_est, 
    s2_df=s2_df, 
    s2_ind=s2_ind)
setup_pool_jc.setTemperatureLadder(1.05**np.arange(50), start_temper=2000)
setup_pool_jc.setMCMC(nmcmc=10000, nburn=5000, thin=1, decor=100, start_tau_theta=-4.)

##########################################################################################
# calibrate
out_pool_jc = sc.calibPool(setup_pool_jc)
out_pool_ptw = sc.calibPool(setup_pool_ptw)


out_pool = out_pool_ptw
setup_pool = setup_pool_ptw

# thetas trace plot, coolest temperature
plt.plot(out_pool.theta[:,0,0])
plt.show()

plt.plot(out_pool.llik)
plt.show()

# index of posterior samples I will use
uu = np.arange(5000, 10000, 5)
 
mle_idx = np.argmax(out_pool.llik[uu])
mat = np.vstack((out_pool.theta[uu,0,:], out_pool.theta[uu[mle_idx],0,:], out_pool.theta[uu,0,:].mean(0)))

##########################################################################################
# posterior predictions (without measurement error, which has standard deviation np.sqrt(out.s2)), disregarding the first 25000 MCMC samples
pred = setup_pool.models[0].eval( # evaluates the model passed to setup.  If more than one call to "addExperiment..." use the index
    sc.tran_unif( # transforms thetas (which are scaled to 0-1) to native scale, results in a dict
        mat, setup_pool.bounds_mat, setup_pool.bounds.keys()
        )
    ) # note, our PTW eval doesn't care about parameter ordering, just the names in the dict

plt.plot(pred.T,color='grey') # note, these don't include the additional error we are estimating (measurement error, plus everything else)
plt.plot(pred[len(uu)],color='red')
plt.plot(pred[len(uu)+1],color='green')
plt.plot(stress_stacked,color='black')
plt.show()


best_theta = sc.tran_unif( # transforms thetas (which are scaled to 0-1) to native scale, results in a dict
    out_pool.theta[uu[mle_idx],0,:], setup_pool.bounds_mat, setup_pool.bounds.keys()
    )


# pairs plot of parameter posterior samples
import pandas as pd
import seaborn as sns
dat = pd.DataFrame(sc.tran_unif(mat, setup_pool.bounds_mat, setup_pool.bounds.keys()))
dat['col'] = ['blue']*len(uu) + ['red'] + ['green']
g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [30]*2}, corner=True, diag_kind='hist', hue='col')
#g.set(xlim=(0,1), ylim = (0,1))
for i in range(out_pool.theta.shape[2]):
    g.axes[i,i].set_xlim(setup_pool.bounds[dat.keys()[i]])
    g.axes[i,i].set_ylim(setup_pool.bounds[dat.keys()[i]])
#g.axes[1,1].set_xlim((-20,20))
g.fig.set_size_inches(10,10)
g
plt.show()



