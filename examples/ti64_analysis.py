##########################################################################################
## This script demonstrates how to do pooled impala calibration of PTW
## using SHPB/quasistatic data.
##########################################################################################

import matplotlib
## Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from impala import superCal as sc
import matplotlib.pyplot as plt
import impala.superCal.post_process as pp
import numpy as np
#import dill
import pandas as pd
import sqlite3 as sq
import os
np.seterr(under='ignore')


#import os, psutil
#process = psutil.Process(os.getpid())
#print('start '+str(process.memory_info().rss/1e9))

name = 'hier_Ti64'
path = name + '_results/'
os.makedirs(path, exist_ok=True)

##########################################################################################
## Three Ti64 experiments, 1st column is plastic strain 2nd column is stress (units: mbar)

## get meta data
con = sq.connect('data/ti-6al-4v/data_Ti64.db')
meta = pd.read_sql('select * from meta;', con)
n_tot = meta.shape[0]
n_shpb = np.sum(meta['type']=='shpb')
n_tc = np.sum(meta['type']=='tc')

#notmst_idx = np.array([st[0:3]!='MST' for st in np.array(meta.pname)[:289]])
#use_idx = np.where(notmst_idx * (meta['type']=='shpb')[:289])[0]
edots = np.hstack(meta.edot[:n_shpb].values)
temps = np.hstack(meta.temperature[:n_shpb].values)

#n_shpb = use_idx.shape[0]
# edots = np.hstack(meta.edot[use_idx].values)
# temps = np.hstack(meta.temperature[use_idx].values)
## get SHPB experimental data
dat_all = []
#for i in use_idx:
for i in range(n_shpb):
    ## get first datset
    dat_all.append(pd.read_sql('select * from data_{};'.format(i + 1), con).values)
stress_stacked = np.hstack([v.T[1] for v in dat_all])
strain_hist_list = [v.T[0] for v in dat_all]

#print('get data  '+str(process.memory_info().rss/1e9))

emu = True

#for i in range(n_shpb):
#    plt.plot(dat_all[i][:,0],dat_all[i][:,1])
#plt.show()

#pd.read_sql("SELECT * FROM sqlite_master where type='table';", con)

if emu:
    import pyBASS as pb
    emu_list = []
    y_emu_list = []
    input_names_emu = ['theta','p','s0','sInf','y0','yInf','y1','y2','kappa','lgamma','vel'] # actually gamma, not lgamma, but we take log below
    for i in np.where(meta['type'] != 'shpb')[0][range(n_tc)]:
        X = pd.read_sql('select * from sims_input_{};'.format(i + 1), con).values
        X[:,9] = np.log(X[:,9]) # we use log gamma below
        y = pd.read_sql('select * from sims_output_{};'.format(i + 1), con).values
        mod = pb.bassPCA(X, y, ncores=15, percVar=99.9)
        emu_list.append(mod)
        yobs = pd.read_sql('select * from data_{};'.format(i + 1), con).values
        y_emu_list.append(yobs)
        #print('emu '+str(i)+' '+str(process.memory_info().rss/1e9))
    
    #emu_list[0].plot()

#print('emulators '+str(process.memory_info().rss/1e9))

con.close()
del con

##########################################################################################
# set constants and parameter bounds
consts_ptw = {
  'alpha'  : 0.84, 
  'beta'   : 0.33,
  'matomic': 45.9,
  'chi'    : 1.0,
  'G0'     : 0.44,
  'rho0'   : 4.419,
  'rho_0'  : 4.45,
  'gamma_1': 2.2,
  'gamma_2': -4.7,
  'q2'     : 0.8,
  'c0'     : 4.730036e-05,
  'tm0'    : -3925.796,
  'tm1'    : 1448.2,
  'r0'     : 4.426741,
  'c1'     : 1.371e-8,
  'r1'     : -2.5965e-5
    }

bounds_ptw = {
    'theta' : (0.0001,       0.2),
    'p'     : (0.0001,       5.),
    's0'    : (0.0001,       0.05),
    'sInf'  : (0.0001,       0.05),
    'kappa' : (0.0001,       0.5),
    'lgamma': (np.log(1e-6), np.log(1e-4)),
    'y0'    : (0.0001,       0.05),
    'yInf'  : (0.0001,       0.01), 
    'y1'    : (0.001,        0.1),
    'y2'    : (0.33,         1.),
    'vel'   : (0.99,         1.01),
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

sd_est_shpb = np.array([.00025]*n_shpb)
s2_df_shpb = np.array([5]*n_shpb)
s2_ind_shpb = np.hstack([[v]*len(dat_all[v]) for v in list(range(n_shpb))])

##########################################################################################
# define PTW model

#impala = reload(impala)
#models = reload(models)

flow_stress_model = 'PTW_Yield_Stress'
melt_model = 'Linear_Melt_Temperature'
shear_model = 'BGP_PW_Shear_Modulus'
specific_heat_model = 'Linear_Specific_Heat'
density_model = 'Linear_Density'

model_ptw_hier = sc.ModelMaterialStrength(temps=np.array(temps), 
    edots=np.array(edots), 
    consts=consts_ptw, 
    strain_histories=strain_hist_list, 
    flow_stress_model=flow_stress_model,
    melt_model=melt_model, 
    shear_model=shear_model, 
    specific_heat_model=specific_heat_model, 
    density_model=density_model,
    pool=False, s2='gibbs')

#print('before init '+str(process.memory_info().rss/1e9))
# bring everything together into calibration structure
setup_hier_ptw = sc.CalibSetup(bounds_ptw, constraints_ptw)
#print('init1 '+str(process.memory_info().rss/1e9))
setup_hier_ptw.addVecExperiments(yobs=stress_stacked, 
    model=model_ptw_hier, 
    sd_est=sd_est_shpb, 
    s2_df=s2_df_shpb, 
    s2_ind=s2_ind_shpb,
    theta_ind=s2_ind_shpb)

#print('init2 '+str(process.memory_info().rss/1e9))
if emu:
    models_emu = []
    for i in range(n_tc):
        models_emu.append(sc.ModelBassPca_func(emu_list[i], input_names_emu, exp_ind=np.array([0]*len(y_emu_list[i])), s2='fix'))
    for i in range(n_tc):
        setup_hier_ptw.addVecExperiments(
            yobs=y_emu_list[i].flatten(),
            model=models_emu[i],
            sd_est=np.array([0.001]), 
            s2_df=np.array([150]), 
            s2_ind=np.array([0]*len(y_emu_list[i])),
            theta_ind=np.array([0]*len(y_emu_list[i]))
        )
#print('init emu '+str(process.memory_info().rss/1e9))
setup_hier_ptw.setTemperatureLadder(1.05**np.arange(30), start_temper=2000)
setup_hier_ptw.setMCMC(nmcmc=30000, thin=1, decor=100, start_tau_theta=-4.)
setup_hier_ptw.setHierPriors(
    theta0_prior_mean=np.repeat(0.5, setup_hier_ptw.p), 
    theta0_prior_cov=np.eye(setup_hier_ptw.p)*10**2, 
    Sigma0_prior_df=setup_hier_ptw.p + 20, 
    #Sigma0_prior_df=setup_hier_ptw.p + 2, 
    Sigma0_prior_scale=np.eye(setup_hier_ptw.p)*.1**2 # used .1**2 before, .5 is what we used elsewhere, indicating low shrinkage (may want to use .5**2 or .25**2 instead), but that was probit space
    ) 

#print('before run '+str(process.memory_info().rss/1e9))

##########################################################################################
# calibrate

out_hier = sc.calibHier(setup_hier_ptw)

mcmc_use = np.arange(20000, 30000, 2) # burn and thin index
pp.save_parent_strength(setup_hier_ptw, setup_hier_ptw.models[0], out_hier, mcmc_use, path+'parent_'+name+'.csv') # saves parent distribution

# save output
#dill.dump_session(path + name + '.pkl')

# rank parent distribution samples by stress at particular strain, strain rate, temperature, save to file

pp.parameter_trace_plot(out_hier.theta0[:,0],ylim=[0,1]) # we want these to look like they converge, choose burn-in accordingly
plt.savefig(path+'traceTheta0_'+name+'.pdf')

pp.pairs(setup_hier_ptw, out_hier.theta0[mcmc_use, 0]) # pairs plot of posterior samples of theta0 (parent mean)
plt.savefig(path+'samplePairsTheta0_'+name+'.pdf')

pp.parameter_trace_plot(out_hier.theta[0][:,0,4],ylim=[0,1]) # trace plot of the 5th experiment parameters
plt.savefig(path+'traceTheta5_'+name+'.pdf')

pp.pairs(setup_hier_ptw, out_hier.theta[0][mcmc_use,0,4]) # pairs plot of the 5th experiment parameters
plt.savefig(path+'samplePairsTheta5_'+name+'.pdf')

pp.pairwise_theta_plot_hier(setup_hier_ptw, out_hier, path+'pairs_'+name+'.pdf', mcmc_use) # saves the combined pairs plot
pp.ptw_prediction_plots_hier(setup_hier_ptw, out_hier, path+'pred_'+name+'.pdf', mcmc_use, ylim=[0,.025]) # saves predictions

pp.get_best_sse(path+'parent_'+name+'.csv', path+'bestSSE_'+name+'.csv') # function to get best parameters (uses sum of squared error, sse)
pp.get_bounds(edot=1.1e6, strain=1., temp=0.2*2400, results_csv=path+'parent_'+name+'.csv', write_path=path+'bounds_'+name+'.csv') # function to get bounds
