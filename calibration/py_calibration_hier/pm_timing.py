
import sqlite3 as sq
import numpy as np
#from physical_models_c import MaterialModel
#from physical_models_old import MaterialModel
from scipy.special import erf, erfinv
from math import ceil, sqrt, pi, log
#import ipdb #ipdb.set_trace()

## settings of interest
edot = 40250. * 1e-6 # first term is per second
strain = 0.6
temp = 694. # Kelvin
res_path = './results/Ti64/ti64_hier_temperTest2.db'
dat_path = './data/data_Ti64.db'
name = 'ti64_hier_temperTest2'
out_path = './results/Ti64/'
nexp = 197
plot = True
write = False

## connect to calibration output
con = sq.connect(res_path)
cursor = con.cursor()

## get posterior samples of overall mean parameters
cursor.execute("SELECT * FROM 'phi0';")
phi0 = cursor.fetchall()
phi0_names = list(map(lambda x: x[0], cursor.description))

## get constants used
cursor.execute("SELECT * FROM 'constants';")
constants = dict(cursor.fetchall())

## get models used
cursor.execute("SELECT * FROM 'models';")
models = dict(cursor.fetchall())

parameter_bounds = {
    'theta' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'gamma' : (0.000001, 0.0001),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.3,      1.),
    'vel'   : (0.99,     1.01),
    }

nmcmc = len(phi0)
nparams = len(phi0[0])



def getStrength(edot, strain, temp, params, model_args, consts):
    # get stress at given strain, edot, temp, and params
    model = MaterialModel(flow_stress_model=model_args['flow_stress_model'],shear_modulus_model=model_args['shear_modulus_model'])
    model.set_history_variables(strain, edot, 100)

    # ensure correct ordering
    constant_list = model.get_constant_list()
    param_list = model.get_parameter_list()
    constant_vec = np.array([consts[key] for key in constant_list])
    param_vec = np.array([params[key] for key in param_list])

    model.initialize_constants(constant_vec)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temp)

    #if not model.check_constraints():
    #    ipdb.set_trace()

    return model.compute_state_history()[99, 2]




# temporary model setup
#import physical_models_old as pm
import physical_models_vec as pm

sh = pm.generate_strain_history(.6, np.repeat(edot,2000), 100)
model= pm.MaterialModel(flow_stress_model=pm.PTW_Yield_Stress)

phi0_names[0]='theta'
# ensure correct ordering
constant_list = model.parameters.consts
param_list = model.parameters.params

import time
t0 = time.time()
parmat = np.array(phi0)#np.vstack((np.array(phi0),np.array(phi0),np.array(phi0)))
parameter_matrix = dict((phi0_names[i], parmat[:,i]) for i in range(parmat.shape[1]))
model.initialize(parameter_matrix,constants)
model.initialize_state(T=np.repeat(temp,parmat.shape[0]),stress=np.repeat(0.,parmat.shape[0]),strain=np.repeat(0.,parmat.shape[0]))
out = model.compute_state_history(sh)
t1 = time.time()
t1-t0

out[:,2,100]

def getStress(params):
    parameters = dict(zip(phi0_names,params))
    param_vec = np.array([parameters[key] for key in param_list])

    model.set_history_variables(strain, edot, 100)
    model.initialize(parameters,constants)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temp)
    out = model.compute_state_history(sh)
    return out

aa = getStress(phi0[0])
from matplotlib import pyplot as plt
plt.plot(aa[:,1],aa[:,2])

bounds = np.array([parameter_bounds[key] for key in param_list]) # order bounds correctly

import time

t0 = time.time()
for i in range(2000):
    aa = getStress(phi0[0])
t1 = time.time()
t1-t0

t0 = time.time()
for i in range(2000):
    aa = getStrength(edot, strain, temp, dict(zip(phi0_names,ph)), models, constants)
t1 = time.time()

total = t1-t0




import numpy as np
import pandas as pd
dat = pd.read_csv('./results/Ti64/res_ti64_hier2_postSamplesPTW.csv',header=0,skiprows=1)
parmat = np.array(dat)[range(0,2000,100),0:10]

import sqlite3 as sq
## get meta data
con = sq.connect('./data/data_Ti64.db')
cursor = con.cursor()
cursor.execute("SELECT * FROM meta;")
meta_names = list(map(lambda x: x[0], cursor.description))
meta = pd.DataFrame(cursor.fetchall(),columns=meta_names)
edots = meta.edot[range(196)]
temps = meta.temperature[range(196)]

nrep = parmat.shape[0]
nexp = len(edots)

parmat_big = np.kron(np.ones((nexp,1)),parmat)
edots_big = np.kron(edots,np.ones((nrep)))
temps_big = np.kron(temps,np.ones((nrep)))


## connect to calibration output
con = sq.connect('./results/Ti64/ti64_hier_temperTest2.db')
cursor = con.cursor()
## get constants used
cursor.execute("SELECT * FROM 'constants';")
constants = dict(cursor.fetchall())


import physical_models_vec as pm_vec

sh = pm_vec.generate_strain_history(.6, edots_big, 100)
model= pm_vec.MaterialModel(flow_stress_model=pm_vec.PTW_Yield_Stress)

param_names = list(dat.columns)
param_names[0]='theta'
# ensure correct ordering
constant_list = model.parameters.consts
param_list = model.parameters.params
parameter_matrix = dict((param_names[i], parmat_big[:,i]) for i in range(parmat_big.shape[1]))

import time
t0 = time.time()
model.initialize(parameter_matrix,constants)
model.initialize_state(T=temps_big,stress=np.repeat(0.,parmat_big.shape[0]),strain=np.repeat(0.,parmat_big.shape[0]))
out = model.compute_state_history(sh)
t1 = time.time()
time_vec = t1-t0



import physical_models_old as pm_old
out_old = np.empty(out.shape)

t0 = time.time()
for i in range(nexp*nrep):
    sh = pm_old.generate_strain_history(.6, edots_big[i], 100)
    model= pm_old.MaterialModel(flow_stress_model=pm_old.PTW_Yield_Stress)
    parameters = dict(zip(param_names, parmat_big[i]))
    model.initialize(parameters,constants)
    model.initialize_state(T=temps_big[i],stress=0.,strain=0.)
    out_old[:,:,i] = model.compute_state_history(sh)
t1 = time.time()
time_old = t1-t0

np.max(np.abs(out_old-out))


time_old/time_vec

import physical_models_c as pm_c
out_c = np.empty(out.shape)
param_names[0]='theta0'
model = pm_c.MaterialModel(flow_stress_model="PTW")
t0 = time.time()
for i in range(nexp*nrep):
    model.set_history_variables(0.6, edots_big[i], 100)

    constant_list = model.get_constant_list()
    param_list = model.get_parameter_list()
    parameters = dict(zip(param_names, parmat_big[i]))
    constant_vec = np.array([constants[key] for key in constant_list])
    param_vec = np.array([parameters[key] for key in param_list])

    model.initialize_constants(constant_vec)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temps_big[i])
    out_c[:,:,i] = model.compute_state_history()
t1 = time.time()
time_c = t1-t0

np.max(np.abs(out-out_c))

time_c/time_vec