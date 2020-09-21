#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:37:20 2020

@author: dfrancom
"""
import pickle
import numpy as np
from collections import namedtuple
from pyBASS import bassPCA

####################################################
## generate emulator

def f2(x):
  out = 10. * np.sin(3.14 * tt * x[1]) + 20. * (x[2] - .5)**2 + 10 * x[3] + 5. * x[4]
  return out


tt = np.linspace(0,1,50)
n = 500
p = 9
x = np.random.rand(n,p)-.5
xx = np.random.rand(1000,p)-.5
e = np.random.normal(size=n*len(tt))
y = np.apply_along_axis(f2, 1, x) #+ e.reshape(n,len(tt))

modf = bassPCA(x,y,ncores=2,percVar=99.99)


####################################################
#### functions

def pos(a):
    return (abs(a)+a)/2

def const(signs,knots):
  cc = np.prod(((signs+1)/2 - signs*knots))
  if cc==0:
    return 1
  return cc

def makeBasis(signs,vs,knots,xdata):
  cc = const(signs,knots)
  temp1 = pos(signs * (xdata[:,vs]-knots))
  if len(signs) == 1:
      return temp1/cc
  temp2 = np.prod(temp1,axis=1)/cc
  return temp2

def normalize(x, bounds):
    return (x - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

def makeBasisMatrix(samples, X): # make basis matrix for model
    nb = samples.nbasis
    n = len(X)
    mat = np.zeros([n,nb+1])
    mat[:,0] = 1
    for m in range(nb):
        ind = list(range(samples.n_int[m]))
        mat[:,m+1] = makeBasis(samples.signs[m,ind],samples.vs[m,ind],samples.knots[m,ind],X).reshape(n)
    return mat

def predict(X, ssi, bounds, nugget=False):
    Xs = normalize(X, bounds)
    out = np.zeros([len(Xs)])

    out = np.dot(ssi.beta,makeBasisMatrix(ssi,Xs).T)
    if nugget:
        out = out + np.random.normal(size=len(Xs),scale=np.sqrt(ssi.s2)).T
    return out
                                     
def predictPCA(X, samples_list, basist, ysd, ymean, bounds, nugget=True):
    pred_coefs = list(map(lambda ii: predict(X, samples_list[ii], bounds, nugget), list(range(len(ss)))))
    out = np.dot(np.dstack(pred_coefs), basist)
    return out*ysd + ymean


Sample = namedtuple('Sample', 's2 nbasis n_int signs vs knots beta')



####################################################
#### constant quantities for a given emulator

ymean = modf.y_mean
ysd = modf.y_sd
tbasis = modf.basis.T
bounds = modf.bm_list[0].data.bounds


####################################################
#### particular MCMC sample (mcmc_use) quanitities, to be pickled

modf_nmcmc = len(modf.bm_list[0].samples.nbasis)
mcmc_use = np.random.choice(range(modf_nmcmc))
ss = []
for ipc in range(modf.nbasis):
    model_use = modf.bm_list[ipc].model_lookup[mcmc_use]
    nbasis = modf.bm_list[ipc].samples.nbasis[mcmc_use]
    ss.append(Sample(modf.bm_list[ipc].samples.s2[mcmc_use], 
           nbasis, 
           modf.bm_list[ipc].samples.n_int[model_use,:],
           modf.bm_list[ipc].samples.signs[model_use,0:nbasis,:],
           modf.bm_list[ipc].samples.vs[model_use,0:nbasis,:],
           modf.bm_list[ipc].samples.knots[model_use,0:nbasis,:],
           modf.bm_list[ipc].samples.beta[mcmc_use,0:(nbasis+1)]))


pickling_on = open("samples.pickle","wb")
pickle.dump(ss, pickling_on)
pickling_on.close()



####################################################
#### unpickle and predict at theta

pickle_off = open("samples.pickle","rb")
ss2 = pickle.load(pickle_off)

theta = xx[1,:].reshape([1,9])

predictPCA(theta,ss2,tbasis,ysd,ymean,bounds)
#modf.predict(theta,mcmc_use=np.array([mcmc_use]),nugget=True)











