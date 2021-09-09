# %%
import plots
import numpy as np
import pandas as pd
import sqlite3 as sq
import impala as impala
import models as models
np.seterr(under='ignore')





def f1(x):
    oo = x[0] #+ x[1]
    out = np.array([oo,oo])
    return out
yobs1 = np.array([.5,.5])

def f2(x):
    oo = x[1] #+ x[1]
    out = np.array([oo,oo])
    return out
yobs2 = np.array([.7,.7])


p = 4
input_names = [str(v) for v in list(range(p))]
bounds = dict(zip(input_names,np.concatenate((np.zeros((p,1)),np.ones((p,1))),1)))



def cf(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

#%%


#impala = reload(impala)

setup = impala.CalibSetup(bounds, cf)
model1 = models.ModelF(f1, input_names)
model2 = models.ModelF(f2, input_names)
model3 = models.ModelF(f1, input_names)
setup.addVecExperiments(yobs1, model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
setup.addVecExperiments(yobs2, model2, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
setup.addVecExperiments(yobs1, model3, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
setup.setTemperatureLadder(1.05**np.arange(2))
setup.setMCMC(30000,8000,1,100)

# there is a problem when you turn on decorrelation (no problem with tempering)


#out = impala.calibHier(setup)
#out2 = impala.calibPool(setup)
out3 = impala.calibClust(setup)





