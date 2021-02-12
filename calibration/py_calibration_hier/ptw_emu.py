import sqlite3 as sq
import numpy as np
from physical_models_c import MaterialModel
from scipy.special import erf, erfinv
from math import ceil, sqrt, pi, log
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

dat_path = './data/data_Ti64.db'

## get meta data
con = sq.connect(dat_path)
cursor = con.cursor()
cursor.execute("SELECT * FROM meta;")
meta_names = list(map(lambda x: x[0], cursor.description))
meta = pd.DataFrame(cursor.fetchall(),columns=meta_names)




meta.temperature
meta.edot

nexp = 196



parameter_bounds = {
    'theta0' : (0.0001,   0.2),
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
constants = {
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

param_names =     ['theta0','p','s0' ,'sInf' ,    'kappa' ,    'gamma' ,    'y0'   ,    'yInf' ,'y1'   ,'y2']

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

def getStress(edot, temp, params, model_args, consts):
    model = MaterialModel(flow_stress_model=model_args['flow_stress_model'],shear_modulus_model=model_args['shear_modulus_model'])
    model.set_history_variables(0.6, edot, 100)

    constant_list = model.get_constant_list()
    param_list = model.get_parameter_list()
    constant_vec = np.array([consts[key] for key in constant_list])
    param_vec = np.array([params[key] for key in param_list])

    model.initialize_constants(constant_vec)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temp)
    return model.compute_state_history()[:, 1:3]

def unnormalize(z,bounds):
    """ Transform 0-1 scale to real scale """
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


models = {'flow_stress_model'   : 'PTW', 'shear_modulus_model' : 'Stein'}
model_temp = MaterialModel(flow_stress_model=models['flow_stress_model'],shear_modulus_model=models['shear_modulus_model'])
param_list = model_temp.get_parameter_list()
bounds = np.array([parameter_bounds[key] for key in param_list]) # order bounds correctly


ntrain = 10000
nx = 10
nexp = 196
train_stress = np.empty([ntrain,100,nexp])
#train_stress_flat = np.empty([ntrain,100*nexp])
train_inputs = np.empty([ntrain,nx])
edots = meta.edot
temps = meta.temperature

for i in range(ntrain):
    cst = False
    while not cst:  # do until sample meets constraints
        ph = unnormalize(np.random.uniform(size=nx),bounds)
        params = dict(zip(param_names, ph))
        stress_temp = getStress(edots[0], temps[0], params, models, constants)
        cst = stress_temp[1,1] > 0.0

    train_inputs[i,:] = ph
    for j in range(nexp):
        train_stress[i,:,j] = getStress(edots[j], temps[j], params, models, constants)[:,1]
        #train_stress_flat[i,range(100*(j-1),100*j)] = train_stress[i,:,j]

    print(i)

xx = getStress(meta.edot[0], meta.temperature[0], params, models, constants)[:,0]

X = train_inputs
Y = train_stress.reshape([ntrain,100*nexp],order='F')

import pyBASS as pb

mod = pb.bassPCA(X,Y,ncores=2,percVar=99.99)

pred = mod.predict(X,mcmc_use=np.array([1,100]),nugget=False)

import matplotlib.pyplot as plt

plt.scatter(Y[:,50],pred[0,:,50])


plt.plot(Y[0,0:99])
plt.plot(pred[0,0,0:99])

plt.plot(Y[0,: ])
plt.plot(pred[0,0,:])

plt.plot(Y[0,0:599 ])
plt.plot(pred[0,0,0:599])
