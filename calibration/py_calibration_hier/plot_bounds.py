#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:46:47 2020

@author: dfrancom
"""


import physical_models_c as pm
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sql

# PTW parameters
params = {
    'theta' : 0.0375,   
    'p'     : 1.,
    's0'    : 0.0245, 
    'sInf'  : 0.009,
    'kappa' : 0.19,   
    'gamma' : 2.e-6,
    'y0'    : .019, 
    'yInf'  : 0.0076
    }
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


model_ptw = pm.MaterialModel(
    flow_stress_model = 'PTW',
    shear_modulus_model = 'Simple'
    )

model_ptw.initialize(params, starting_consts)
model_ptw.initialize_state(T=298.)

model_ptw.set_history_variables(.5,.01,100)
results_ptw = model_ptw.compute_state_history()


def sample_params(params, parameter_bounds):
    kk = list(params.keys())
    for k in kk:
        bb = parameter_bounds[k]
        params[k] = (np.random.rand()+bb[0])*(bb[1]-bb[0])
    return params


for i in range(10000):
    pp = sample_params(params,parameter_bounds)
    
    model_ptw.initialize(pp, starting_consts)
    model_ptw.initialize_state(T=800.)
    
    results_ptw = model_ptw.compute_state_history()
    if results_ptw[1,2]>0:
        plt.plot(results_ptw[:,1],results_ptw[:,2])

conn = sql.connect('./data_copper.db')
curs = conn.cursor()

for i in range(6):
    ll = np.array(list(curs.execute("SELECT * from data_" + str(i+1) + ";")))
    plt.plot(ll[:,0],ll[:,1])
    
axes = plt.gca()
axes.set_ylim([0,.006])




import physical_models as pm2

params = {
    'theta' : 0.0375,   
    'p'     : 1.,
    's0'    : 0.0245, 
    'sInf'  : 0.009,
    'kappa' : 0.19,   
    'gamma' : 2.e-6,
    'y0'    : .019, 
    'yInf'  : 0.0076
    }
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

model_stein = pm2.MaterialModel(
    flow_stress_model = pm2.PTW_Yield_Stress,
    shear_modulus_model = pm2.Simple_Shear_Modulus,
    )
model_stein.initialize(params, starting_consts)
model_stein.initialize_state(T=298.)

shist = pm2.generate_strain_history(emax = 0.5, edot = 0.01, Nhist = 100)
results_stein = model_stein.compute_state_history(shist)


model_ptw = pm.MaterialModel(
    flow_stress_model = 'PTW',
    shear_modulus_model = 'Simple'
    )

model_ptw.initialize(params, starting_consts)
model_ptw.initialize_state(T=298.)

model_ptw.set_history_variables(.5,.01,100)
results_ptw = model_ptw.compute_state_history()

plt.plot(results_ptw[:,1],results_ptw[:,2])
plt.plot(results_stein[:,1],results_stein[:,2])


#model.initialize_state(T=298.)
#model.update_parameters(params)
#results = model.compute_state_history(shist)








import physical_models as pm2

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
        'sgB'    : 6.44e-4,
        'alpha'  : 0.2
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

params = {
    'theta' : 0.128,   
    'p'     : 3.03,
    's0'    : 0.0259, 
    'sInf'  : 0.0219,
    'kappa' : 0.159,   
    'gamma' : 3.78e-5,
    'y0'    : 0.0237, 
    'yInf'  : 5.4e-3
    }
model_stein = pm2.MaterialModel(
    flow_stress_model = PTW_Yield_Stress,
    shear_modulus_model = pm2.Stein_Shear_Modulus,
    )
model_stein.initialize(params, starting_consts)
model_stein.initialize_state(T=573.)

shist = pm2.generate_strain_history(emax = 0.5, edot = 800*1e-6, Nhist = 100)
results_stein = model_stein.compute_state_history(shist)
plt.plot(results_stein[:,1],results_stein[:,2])



conn = sql.connect('./data_Ti64.db')
curs = conn.cursor()

ll = np.array(list(curs.execute("SELECT * from data_1;")))
plt.plot(ll[:,0],ll[:,1])

