#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:30:15 2023

@author: Dr. Lauren J Beesley VanDervort, lvandervort@lanl.gov
"""

### Imports
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import median_absolute_error, make_scorer
import gc
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
from scipy.stats import invgamma
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from impala import superCal as sc
from impala import physics as pm_vec
from itertools import cycle
from scipy.optimize import NonlinearConstraint
from scipy.stats import halfcauchy
import sklearn
import pyswarm
from scipy.stats import qmc
import time
import os
import multiprocess as mp


### Function for obtaining the MAP estimator associated
### with the pooled Impala model
def get_map_impalapool(setup, n_samples = 1000, theta_init = None, disc_init = None, optmethod = 'bh', niter = None, T = 1, n_cores = 10): 
    
    if str(type(niter)) == "<class 'NoneType'>":
        niter = setup.p*10

    ### Draw s2
    s2 = [np.empty([n_samples,len(setup.ig_a[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        if setup.models[i] == 'fix':
            s2[i] = setup.sd_est[i]
        else:
            for j in range(len(setup.ig_a[i])):
                if setup.s2_df[i][j] == 0:
                    s2[i][:,j] = halfcauchy.rvs(loc = 0, scale = 1, size = n_samples).reshape(1,n_samples)    
                else:
                    s2[i][:,j] = invgamma.rvs(a=setup.ig_a[i][j], scale = setup.ig_b[i][j], size = n_samples).reshape(1,n_samples)
    s2_expand = [np.empty([n_samples,len(setup.ys[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        for j in range(len(setup.ig_a[i])):
            s2_expand[i][:,setup.s2_ind[i] == j] = s2[i][:,j].reshape(n_samples,1)
    
    ### Handle constraints
    if optmethod == 'bh':
        def my_constraint(y):
            y = np.reshape(y, (1, -1))
            A = int(setup.checkConstraints(sc.tran_unif(y[0,0:setup.p].reshape(1,-1),setup.bounds_mat, setup.bounds.keys()), setup.bounds)[0])
            return A    
    else:
        def my_constraint(y):
            y = np.reshape(y, (1, -1))
            A = int(setup.checkConstraints(sc.tran_unif(y[0,0:setup.p].reshape(1,-1),setup.bounds_mat, setup.bounds.keys()), setup.bounds)[0])
            return (A-0.5)
    non_linear_constr = NonlinearConstraint(my_constraint, 0.5, np.inf)
    ### Optimize
    def neg_log_lik(y):
        y = np.reshape(y, (1, -1))
        if y.shape[1] > setup.p:
            v_cur = y[0,setup.p:y.shape[1]]
            disc_dims = [setup.models[i].nd for i in range(setup.nexp)]
            v_cur_list = [v_cur[0:disc_dims[i]] if i == 0 else v_cur[int(np.sum(disc_dims[0:(i-1)])):int(np.sum(disc_dims[0:i]))] for i in range(setup.nexp)]
            theta_cur = y[0,0:setup.p].reshape(1,-1)
        else:
            theta_cur = y[0,:].reshape(1,-1)
        CONSTRAINTS = setup.checkConstraints(sc.tran_unif(theta_cur,setup.bounds_mat, setup.bounds.keys()), setup.bounds)
        if CONSTRAINTS[0] == True and (theta_cur<=0).sum() == 0 and (theta_cur>=1).sum() == 0:
            disc_y = [(setup.models[i].D @ v_cur_list[i]).reshape(1,-1) if setup.models[i].nd > 0 else np.repeat(0,len(setup.ys[i])).reshape(1,-1) for i in range(setup.nexp)]     
            pred_y = [setup.models[i].eval(sc.tran_unif(theta_cur,setup.bounds_mat, setup.bounds.keys()), pool=True) for i in range(setup.nexp)]
            loglik_y = [-0.5* (((((setup.ys[i] - pred_y[i] - disc_y[i])**2)/s2_expand[i]) +np.log(s2_expand[i])).sum(axis = 1)).reshape(n_samples,1) for i in range(setup.nexp)]
            loglik_y_disc = [-0.5* (((v_cur_list[i]**2)/setup.models[i].discrep_tau) +np.log(setup.models[i].discrep_tau)).sum() if setup.models[i].nd>0 else 0 for i in range(setup.nexp)]
            llik = np.nansum(loglik_y,axis=0)+np.nansum(np.hstack(loglik_y_disc))
            llik_max = llik.max()
            llik = llik - llik_max  
            output = np.log(np.exp(llik).mean()) + llik_max
            output = np.array(output)
            output = output.reshape(1,1)
            output = -1*output[0][0]
        else:
            dist_func = getattr(setup, 'distConstraints', None)
            if str(type(dist_func)) != "<class 'NoneType'>":
                VAL = setup.distConstraints(sc.tran_unif(y,setup.bounds_mat, setup.bounds.keys()), setup.bounds)
                output = (10 ** 6) + np.sum(VAL)
            else:
                output = 10 ** 6
        return output
        
        
    if str(type(disc_init)) != "<class 'NoneType'>":
        param_cur = np.append(sc.normalize(theta_init.flatten(),setup.bounds_mat),np.hstack(disc_init))
    else:
        param_cur = sc.normalize(theta_init.flatten(),setup.bounds_mat)

    if optmethod == 'bh':
        res = basinhopping(neg_log_lik, param_cur, minimizer_kwargs={"method":"COBYLA","constraints":non_linear_constr}, niter=niter, disp = True, stepsize = 0.1, T = T)
    elif optmethod == 'pso':
        res = pyswarm.pso(neg_log_lik, lb = np.repeat(0,len(param_cur.flatten())), ub = np.repeat(1,len(param_cur.flatten())), ieqcons=[my_constraint], swarmsize=niter, debug = True, omega = 0.9, maxiter = niter)
    elif optmethod == 'grid':
        from scipy.stats import qmc
        import time
        import os
        import multiprocess as mp
        from scipy.optimize import minimize
        sampler = qmc.LatinHypercube(d=setup.p)
        sample = sampler.random(n=10000)  
        sample = np.append(sc.normalize(theta_init.flatten(),setup.bounds_mat).reshape(1,-1), sample,axis=0)
        CONSTRAINTS = setup.checkConstraints(sc.tran_unif(sample,setup.bounds_mat, setup.bounds.keys()), setup.bounds)
        sample = sample[CONSTRAINTS,:]
        niter = np.min((niter,sample.shape[0]))
        sample = sample[0:niter,:]
        
        if str(type(disc_init)) != "<class 'NoneType'>":
            def run_mcmc_in_parallel(i):
                disc_rand = np.hstack(disc_init) + np.random.normal(loc = np.repeat(0,len(disc_init.flatten())),scale =np.sqrt(np.repeat(setup.models[0].discrep_tau,len(disc_init.flatten()))), size = len(disc_init.flatten()))
                res = minimize(neg_log_lik, np.append(sample[i,:].reshape(1,-1), disc_rand.reshape(1,-1),axis=1).flatten(), method = 'COBYLA', constraints = non_linear_constr)
                return (res.x)
        else:
            def run_mcmc_in_parallel(i):
                res = minimize(neg_log_lik, sample[i,:], method = 'COBYLA', constraints = non_linear_constr)
                return (res.x)            
        with mp.Pool(n_cores) as pool:
            A = pool.map(run_mcmc_in_parallel, range(niter))
    else:
        print('Invalid optmethod')
        return
    
    import copy
    if optmethod == 'bh':
        res_trans0 = copy.deepcopy(res.x)
    elif optmethod == 'pso':
        res_trans0 = copy.deepcopy(res[0])
    elif optmethod == 'grid':
        B = np.apply_along_axis(neg_log_lik, 1, A)
        res_trans0 = A[np.where(B == B.min())[0][0]]
    res_trans = sc.tran_unif(res_trans0[0:setup.p],setup.bounds_mat, setup.bounds.keys())
    
    if len(res_trans0) > setup.p:
        res_trans['v'] = res_trans0[setup.p:len(res_trans0)]
    return res_trans



### Monte Carlo integrated posterior for pooled impala model, integrating out everything but v (basis coefficients for discrepancy)
def eval_partialintlogposterior_impalapool(setup, n_samples = 1000, theta = None, disc_v = None): 
    ### Draw s2
    s2 = [np.empty([n_samples,len(setup.ig_a[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        if setup.models[i].s2 == 'fix':
            for j in range(len(setup.ig_a[i])):
                s2[i][:,j] = np.repeat(setup.sd_est[i],s2[i].shape[0])
        else:
            for j in range(len(setup.ig_a[i])):
                if setup.s2_df[i][j] == 0:
                    s2[i][:,j] = halfcauchy.rvs(loc = 0, scale = 1, size = n_samples).reshape(1,n_samples)    
                else:
                    s2[i][:,j] = invgamma.rvs(a=setup.ig_a[i][j], scale = setup.ig_b[i][j], size = n_samples).reshape(1,n_samples)
    s2_expand = [np.empty([n_samples,len(setup.ys[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        for j in range(len(setup.ig_a[i])):
            s2_expand[i][:,setup.s2_ind[i] == j] = s2[i][:,j].reshape(n_samples,1)
    
    ### Optimize
    def neg_log_lik(y):
        y = np.reshape(y, (1, -1))
        if y.shape[1] > setup.p:
            v_cur = y[0,setup.p:y.shape[1]]
            disc_dims = [setup.models[i].nd for i in range(setup.nexp)]
            v_cur_list = [v_cur[0:disc_dims[i]] if i == 0 else v_cur[int(np.sum(disc_dims[0:(i-1)])):int(np.sum(disc_dims[0:i]))] for i in range(setup.nexp)]
            theta_cur = y[0,0:setup.p].reshape(1,-1)
        else:
            theta_cur = y[0,:].reshape(1,-1)
        pred_y = [np.empty([n_samples,len(setup.ys[i])]) for i in range(setup.nexp)]
        for i in range(setup.nexp):
            pred_y[i] = setup.models[i].eval(sc.tran_unif(theta_cur,setup.bounds_mat, setup.bounds.keys()), pool=True)
        disc_y = [(setup.models[i].D @ v_cur_list[i]).reshape(1,-1) if setup.models[i].nd > 0 else np.repeat(0,len(setup.ys[i])).reshape(1,-1) for i in range(setup.nexp)]     
        loglik_y = [np.empty([n_samples,1]) for i in range(setup.nexp)]
        for i in range(setup.nexp):
            loglik_y[i] = -0.5* (((((setup.ys[i] - pred_y[i] - disc_y[i])**2)/s2_expand[i]) +np.log(s2_expand[i])).sum(axis = 1)).reshape(n_samples,1)
        llik = np.nansum(loglik_y,axis = 0) #loglik_y[0]
        #for i in range(setup.nexp-1):
        #    llik = llik + loglik_y[i+1]
        llik_max = llik.max()
        llik = llik - llik_max  
        llik[llik< -10**2]=-10**2 #preventing exp underflow issue
        output = np.log(np.exp(llik).mean()) + llik_max
        output = np.array(output)
        output = output.reshape(1,1)
        output = -1*output[0][0]
        return output
    if str(type(disc_v)) != "<class 'NoneType'>":
        loglik=-1*np.apply_along_axis(neg_log_lik, 1, np.append(sc.normalize(theta,setup.bounds_mat),disc_v,axis=1))
    else:
        loglik=-1*np.apply_along_axis(neg_log_lik, 1, sc.normalize(theta,setup.bounds_mat))
    return loglik


### Monte Carlo integrated posterior for pooled impala model, integrating out everything including v (basis coefficients for discrepancy)
def eval_fullintlogposterior_impalapool(setup, n_samples = 1000, theta = None): 
    ### Draw s2
    s2 = [np.empty([n_samples,len(setup.ig_a[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        if setup.models[i] == 'fix':
            for j in range(len(setup.ig_a[i])):
                s2[i][:,j] = np.repeat(setup.sd_est[i],s2[i].shape[0])
        else:
            for j in range(len(setup.ig_a[i])):
                if setup.s2_df[i][j] == 0:
                    s2[i][:,j] = halfcauchy.rvs(loc = 0, scale = 1, size = n_samples).reshape(1,n_samples)    
                else:
                    s2[i][:,j] = invgamma.rvs(a=setup.ig_a[i][j], scale = setup.ig_b[i][j], size = n_samples).reshape(1,n_samples)
    s2_expand = [np.empty([n_samples,len(setup.ys[i])]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        for j in range(len(setup.ig_a[i])):
            s2_expand[i][:,setup.s2_ind[i] == j] = s2[i][:,j].reshape(n_samples,1)
    
    disc_dims = [setup.models[i].nd for i in range(setup.nexp)]
    
    ### Draw v_cur
    v_cur = np.empty([n_samples,np.sum(disc_dims)])
    for i in range(setup.nexp):
        if 'discrep_tau' not in dir(setup.models[i]):
            setup.models[i].discrep_tau = 1e-10
        if i == 0:
            v_cur[:,0:disc_dims[i]] = np.random.normal(size = n_samples * disc_dims[i], loc = 0, scale = np.sqrt(setup.models[i].discrep_tau)).reshape(n_samples, disc_dims[i])
        else:
            v_cur[:,np.sum(disc_dims[0:i]):np.sum(disc_dims[0:(i+1)])] = np.random.normal(size = n_samples * disc_dims[i], loc = 0, scale = np.sqrt(setup.models[i].discrep_tau)).reshape(n_samples, disc_dims[i])
    v_cur_list = [v_cur[:,0:disc_dims[i]] if i == 0 else v_cur[:,int(np.sum(disc_dims[0:(i-1)])):int(np.sum(disc_dims[0:i]))] for i in range(setup.nexp)]
  
    ### Optimize
    def neg_log_lik(y):
        y = np.reshape(y, (1, -1))
        pred_y = [np.empty([n_samples,len(setup.ys[i])]) for i in range(setup.nexp)]
        for i in range(setup.nexp):
            pred_y[i] = setup.models[i].eval(sc.tran_unif(y,setup.bounds_mat, setup.bounds.keys()), pool=True)
        disc_y = [(setup.models[i].D @ v_cur_list[i].T).T if setup.models[i].nd > 0 else np.repeat(0,len(setup.ys[i])).reshape(n_samples,-1) for i in range(setup.nexp)]     
        loglik_y = [np.empty([n_samples,1]) for i in range(setup.nexp)]
        for i in range(setup.nexp):
            loglik_y[i] = -0.5* (((((setup.ys[i] - pred_y[i] - disc_y[i])**2)/s2_expand[i]) +np.log(s2_expand[i])).sum(axis = 1)).reshape(n_samples,1)
        llik = loglik_y[0]
        for i in range(setup.nexp-1):
            llik = llik + loglik_y[i+1]
        llik_max = llik.max()
        llik = llik - llik_max  
        llik[llik < -10**2] = -10**2
        output = np.log(np.exp(llik).mean()) + llik_max
        output = np.array(output)
        output = output.reshape(1,1)
        output = -1*output[0][0]
        return output
    loglik=-1*np.apply_along_axis(neg_log_lik, 1, sc.normalize(theta,setup.bounds_mat))
    return loglik

