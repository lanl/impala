
import sqlite3 as sq
import numpy as np
from physical_models_c import MaterialModel
from scipy.special import erf, erfinv
from math import ceil, sqrt, pi, log
import ipdb #ipdb.set_trace()

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
model_temp = MaterialModel(flow_stress_model=models['flow_stress_model'],shear_modulus_model=models['shear_modulus_model'])
param_list = model_temp.get_parameter_list()
bounds = np.array([parameter_bounds[key] for key in param_list]) # order bounds correctly

ph = phi0[0]
stress_temp = getStrength(edot, strain, temp, dict(zip(phi0_names,ph)), models, constants)

import time

t0 = time.time()
for i in range(2000):
    aa = getStrength(edot, strain, temp, dict(zip(phi0_names,ph)), models, constants)
t1 = time.time()

total = t1-t0


















import numpy as np
from scipy import special
y0 = ph[6]
yInf = ph[7]
s0 = ph[2]
sInf = ph[3]
kappa = ph[4]
gamma = ph[5]
theta0 = ph[0]
p = ph[1]
y1 = ph[8]
y2 = ph[9]
beta = .33
G0 = .4
sgB = 0.000644




Tmelt0 = 2110
rho0 = 4.419
chi = 0
matomic = 63.54
T = 300
psidot = 10**-2
M = matomic/6.025*10**23

That = T/Tmelt0
G = G0*(1-alpha*That)
xidot = .5*4*(np.pi*rho0/3/M)**(1/3)*np.sqrt(G/rho0)

psi = .5

tauhaty_low = y0 - (y0-yInf)*special.erf(kappa*That*np.log(gamma*xidot/psidot))
tauhats_low = s0 - (s0-sInf)*special.erf(kappa*That*np.log(gamma*xidot/psidot))

tauhats_high = s0*(psidot/gamma/xidot)**beta
tauhats = max(tauhats_low, tauhats_high)

tauhaty_med = y1*(psidot/gamma/xidot)**y2
tauhaty = max(tauhaty_low, min(tauhaty_med,tauhats_high)) # since tauhaty_high = tauhats_high

tauhat = tauhats + 1/p * (s0-tauhaty)*np.log( 1 - (1-np.exp(-p*(tauhats-tauhaty)/(s0-tauhaty))) * np.exp(-p*theta0*psi / ( (s0-tauhaty)*(np.exp(p*(tauhats-tauhaty)/(s0-tauhaty))-1) )) )

taus = tauhats*G
tauy = tauhaty*G
tau = tauhat * G


import time

t0 = time.time()
for i in range(20000):
    1+1
t1 = time.time()

total = t1-t0
