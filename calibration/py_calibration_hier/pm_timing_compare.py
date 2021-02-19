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




subi = [1000,500,100,50,10]
time_vec = np.empty(len(subi))
time_old = np.empty(len(subi))
time_c = np.empty(len(subi))
neval = np.empty(len(subi))
for j in range(len(subi)):
    parmat = np.array(dat)[range(0,2000,subi[j]),0:10]
    nrep = parmat.shape[0]
    nexp = len(edots)
    neval[j] = nrep*nexp

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
    import time
    t0 = time.time()
    model.initialize(parameter_matrix,constants)
    model.initialize_state(T=temps_big,stress=np.repeat(0.,parmat_big.shape[0]),strain=np.repeat(0.,parmat_big.shape[0]))
    out = model.compute_state_history(sh)
    t1 = time.time()
    time_vec[j] = t1-t0

    #import cProfile
    #cProfile.run('out = model.compute_state_history(sh)')


    import physical_models_old as pm_old
    out_old = np.empty((100,6,nexp*nrep))

    t0 = time.time()
    for i in range(nexp*nrep):
        sh = pm_old.generate_strain_history(.6, edots_big[i], 100)
        model= pm_old.MaterialModel(flow_stress_model=pm_old.PTW_Yield_Stress,shear_modulus_model=pm_old.Stein_Shear_Modulus)
        parameters = dict(zip(param_names, parmat_big[i]))
        model.initialize(parameters,constants)
        model.initialize_state(T=temps_big[i],stress=0.,strain=0.)
        out_old[:,:,i] = model.compute_state_history(sh)
    t1 = time.time()
    time_old[j] = t1-t0

    np.max(np.abs(out_old-out))


    time_old/time_vec

    import physical_models_c as pm_c
    out_c = np.empty((100,6,nexp*nrep))
    param_names[0]='theta0'
    model = pm_c.MaterialModel(flow_stress_model="PTW",shear_modulus_model='Stein')
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
    time_c[j] = t1-t0

    np.max(np.abs(out-out_c))

time_c/time_vec
time_old/time_c
time_old/time_vec

import matplotlib.pyplot as plt
plt.plot(neval,np.log10(time_old),label='python')
plt.plot(neval,np.log10(time_c),label='cython')
plt.plot(neval,np.log10(time_vec),label='vectorized')
plt.legend(loc="upper left")
plt.xlabel('number of PTW evaluations')
plt.ylabel('log10 seconds')
plt.show()

