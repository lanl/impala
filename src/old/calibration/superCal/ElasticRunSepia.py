#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from sepia.SepiaData import SepiaData
#import SepiaData
from sepia.SepiaModel import SepiaModel
#import SepiaModel
import sepia.SepiaPlot
import dill


# In[2]:

design = np.loadtxt('./../../../../git/tucker/x_sepia.csv',delimiter=',',  skiprows=1)

# Preprocess data
y_sim =np.loadtxt('./../../../../git/tucker/y_sepia.csv',delimiter=',',  skiprows=1).T
y_obs =np.loadtxt('./../../../../git/tucker/yobs_sepia.csv',delimiter=',',  skiprows=1)
y_obs = np.reshape(y_obs, (-1, len(y_obs))) #reshape to 2D;

 

n_features = y_obs.shape[1]  # number of feature to be calibrated; 
y_ind = np.arange(1, n_features+1)

    
#%% Gatt: calibrate and plot data, responses, etc.

from sepia.SepiaPredict import SepiaFullPrediction
import scipy as sp

data = SepiaData(t_sim=design[:,[0,1,2,3,4,5,8]], y_sim=y_sim, y_ind_sim=y_ind, 
                 y_obs=y_obs, y_ind_obs=y_ind ,
                 Sigy=np.eye(y_sim.shape[1])*.1 )
data.standardize_y()
data.transform_xt()
pca = 0.97
data.create_K_basis(n_pc=pca)
Dobs = np.zeros(400)
Dobs[200:] += 1.
D = Dobs.reshape(1,400)
data.create_D_basis(D_obs=D, D_sim=D)
print(data)


model = SepiaModel( data )
# model.set_param('lamOs',fix=1)   # <---- make Sigy not adjustable
model.do_mcmc(3000)

dill.dump_session('./../../../../git/tucker/elastiCal2.pkl')
#dill.dump_session('./../../../../git/tucker/elastiCal.pkl')

#dill.load(open('./../../../../git/tucker/elastiCal.pkl', 'rb'))

samps=model.get_samples()
sepia.SepiaPlot.theta_pairs(samps)
sepia.SepiaPlot.mcmc_trace(samps)

# show data - observation and simulations
plt.figure()
plt.subplot(2,2,1)
plt.plot(y_sim.T)
plt.plot(y_obs.T,'k',linewidth=2)
plt.title('Data: sims, obs')

# show obs and reconstructed obs alone
plt.subplot(2,2,2)
plt.plot(y_obs.T,'k',linewidth=2)
plt.plot(((sp.linalg.lstsq(data.obs_data.K.T,data.obs_data.y_std.T)[0].T@model.data.obs_data.K)*
        data.obs_data.orig_y_sd + data.obs_data.orig_y_mean).T,'r',linewidth=2)
plt.legend(['observation','PCA modeled observation'])
plt.title('PCA truncation effect on observation')

# show data projected and reconstructed through K basis
# (this is the problem being solved given PCA truncation)
plt.subplot(2,2,3)
# add the obs projected and reconstructed through the K basis
plt.plot(((sp.linalg.lstsq(data.sim_data.K.T,data.sim_data.y_std.T)[0].T@model.data.sim_data.K)*
        data.sim_data.orig_y_sd + data.sim_data.orig_y_mean).T)
plt.plot(((sp.linalg.lstsq(data.obs_data.K.T,data.obs_data.y_std.T)[0].T@model.data.obs_data.K)*
        data.obs_data.orig_y_sd + data.obs_data.orig_y_mean).T,'k',linewidth=2)
plt.title('K projected: sims, obs')
plt.show()


# prior and calibrated predictions
# calibrated predictions
psamps=model.get_samples(sampleset=np.arange(1000,3000,2))
sepia.SepiaPlot.theta_pairs(psamps)
pred=SepiaFullPrediction(model=model, samples=psamps )
ypred=pred.get_ysim()
ypred=pred.get_yobs(as_obs=True)
predd=pred.get_discrepancy(as_obs=True)
plt.plot(predd[:,0,:].T)
plt.show()

plt.plot(y_sim.T,color='grey')
plt.plot(np.quantile(ypred,0.5,axis=(0,1)).T,'k',linewidth=2)
plt.plot(np.quantile(ypred,[0.05,0.95],axis=(0,1)).T,'k:',linewidth=2)
plt.plot(y_obs.T)
plt.title('Posterior [0.025,0.975] prediction')
plt.show()

np.savetxt('./../../../../git/tucker/ypred_sepia.csv',ypred[:,0,:],delimiter=',')