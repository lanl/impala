#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:46:47 2020

@author: dfrancom
"""


import physical_models as pm
import matplotlib.pyplot as plt
shist = pm.generate_strain_history(emax = 0.5, edot = 0.01, Nhist = 100)




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
consts = {
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
model_ptw = pm.MaterialModel(
    flow_stress_model = pm.PTW_Yield_Stress,
    shear_modulus_model = pm.Stein_Shear_Modulus,
    )
model_ptw.initialize(params, consts)
model_ptw.initialize_state(T=298.)

results_ptw = model_ptw.compute_state_history(shist)






# Steinberg parameters
params = {
    'y0'    : 1.33e-2,
    'beta'  : 12.0,
    'n'     : 0.10,
    'ymax'  : 2.12e-2, # >y0
    'a'     : 1.15,
    'b'     : 6.44e-4
    }
consts = {
    'epsi'  : 0.0,
    'g0'    : 0.419,
    'sgB'   : 6.44e-4,
    'Tmelt0': 2110.0,
    'rho0'  : 4.419,
    'Cv0'   : 0.525e-5,
    'G0'    : 0.4,
    'chi'   : 1.0
    }

model_stein = pm.MaterialModel(
    flow_stress_model = pm.Stein_Flow_Stress,
    shear_modulus_model = pm.Stein_Shear_Modulus,
    )
model_stein.initialize(params, consts)
model_stein.initialize_state(T=298.)

results_stein = model_stein.compute_state_history(shist)





plt.plot(results_ptw[:,1],results_ptw[:,2])
plt.plot(results_stein[:,1],results_stein[:,2])
plt.legend(['PTW','Steinburg'],loc='lower right')

#model.initialize_state(T=298.)
#model.update_parameters(params)
#results = model.compute_state_history(shist)
