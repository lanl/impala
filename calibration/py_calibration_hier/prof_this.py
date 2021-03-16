import numpy as np
import pandas as pd
import sqlite3 as sq
## get constants used
con = sq.connect('./results/Ti64/ti64_hier_temperTest2.db')
cursor = con.cursor()
cursor.execute("SELECT * FROM 'constants';")
constants = dict(cursor.fetchall())
## get meta data
con = sq.connect('./data/data_Ti64.db')
cursor = con.cursor()
cursor.execute("SELECT * FROM meta;")
meta_names = list(map(lambda x: x[0], cursor.description))
meta = pd.DataFrame(cursor.fetchall(),columns=meta_names)
edots = meta.edot[range(196)]
temps = meta.temperature[range(196)]
## get some parameter values
dat = pd.read_csv('./results/Ti64/res_ti64_hier2_postSamplesPTW.csv',header=0,skiprows=1)




parmat = np.array(dat)[range(0,2000,10),0:10]
nrep = parmat.shape[0]
nexp = len(edots)

parmat_big = np.kron(np.ones((nexp,1)),parmat)
edots_big = np.kron(edots,np.ones((nrep)))
temps_big = np.kron(temps,np.ones((nrep)))


import physical_models_vec as pm_vec

sh = pm_vec.generate_strain_history(.6, edots_big, 100)
model = pm_vec.MaterialModel(flow_stress_model=pm_vec.PTW_Yield_Stress,shear_modulus_model=pm_vec.Stein_Shear_Modulus)

param_names = list(dat.columns)
param_names[0]='theta'
# ensure correct ordering
constant_list = model.parameters.consts
param_list = model.parameters.params
parameter_matrix = dict((param_names[i], parmat_big[:,i]) for i in range(parmat_big.shape[1]))

np.seterr(under='ignore')

model.initialize(parameter_matrix,constants)
model.initialize_state(T=temps_big,stress=np.repeat(0.,parmat_big.shape[0]),strain=np.repeat(0.,parmat_big.shape[0]))
out = model.compute_state_history(sh)





