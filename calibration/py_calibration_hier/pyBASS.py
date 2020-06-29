#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:10:54 2020

@author: dfrancom
"""

import numpy as np
import scipy as sp
from math import pi, sqrt, log, erf, exp, sin
import matplotlib.pyplot as plt
from itertools import combinations, chain
from scipy.special import comb
from datetime import datetime
from collections import namedtuple

pos = lambda a: (abs(a)+a)/2

def const(signs,knots):
  cc = np.prod(((signs+1)/2 - signs*knots))
  if cc==0:
    return 1
  return cc

## make basis function (from continuous variables)
def makeBasis(signs,vs,knots,data):
  cc = const(signs,knots)
  temp1 = pos(signs * (data[:,vs]-knots)) 
  temp2 = np.prod(temp1,axis=1)/cc
  return temp2


def normalize(x, bounds):
    return (x - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

def unnormalize(z, bounds):
    return z * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)


def dmwnchBass(z_vec, vars_use):
    alpha = z_vec[vars_use] / sum(np.delete(z_vec, vars_use))
    j = len(alpha)
    ss = 1 + (-1)**j * 1 / (sum(alpha) + 1)
    for i in range(j-1):
        idx = comb_index(j, i + 1)
        temp = alpha[idx]
        ss = ss + (-1)**(i + 1) * sum(1 / (temp.sum(axis = 1) + 1))
    return ss
      
      
def getQf(XtX, Xty):
    try:
        R = np.linalg.cholesky(XtX)
    except LinAlgError:
        return None
    dr = np.diag(R)
    if len(dr) > 1:
        if max(dr[1:]) / min(dr) > 1e3:
            return None
    bhat = sp.linalg.solve_triangular(R, sp.linalg.solve_triangular(R, Xty, trans=1))
    qf = np.dot(bhat, Xty)
    out = namedtuple('out', 'R bhat qf')
    return out(R, bhat, qf)


def logProbChangeMod(n_int, vars_use, I_vec, z_vec, p, vars_len, maxInt):
    if n_int == 1:
        out = log(I_vec[n_int - 1) - log(2 * p * vars_len[vars_use]) + #proposal
      log(2 * p * vars_len[vars_use]) + log(maxInt)
    else:
        x = np.zeros(p)
        x[vars_use] = 1
        lprob_vars_noReplace = log(dmwnchBass(z_vec, vars_use))
        out = log(I_vec[n_int]) + lprob_vars_noReplace - n_int * log(2) - sum(log(vars_len[vars_use])) + # proposal
      n_int * log(2) + sum(log(vars_len[vars_use])) + log(comb(p,n.int)) + log(maxInt) # prior
    return out

def genCandBasis():
    pass

class BassPrior:
    def __init__(self, maxInt, maxBasis, npart, g1, g2, s2_lower, h1, h2, a_tau, b_tau):
        self.maxInt = maxInt
        self.maxBasis = maxBasis
        self.npart = npart
        self.g1 = g1
        self.g2 = g2
        self.s2_lower = s2_lower
        self.h1 = h1
        self.h2 = h2
        self.a_tau = a_tau
        self.b_tau = b_tau
        return
        


class BassData:
    def __init__(self, xx, y):
        self.xx_orig = xx
        self.y = y
        self.n = len(xx)
        self.p = len(xx[0])
        self.bounds = np.zeros([p, 2])
        for i in range(p):
            self.bounds[i, 0] = np.min(xx[:, i])
            self.bounds[i, 1] = np.max(xx[:, i])
        self.xx = normalize(self.xx_orig, self.bounds)
        return

class BassState:
    def log_post(self):
        lp = (
            - (self.s2_rate + self.prior.g2) / self.s2
            - (self.data.n/2 + 1 + (self.nbasis + 1) / 2 + self.prior.g1) * log(self.s2)
            + np.sum(log(abs(np.diag(self.R)))) # .5*determinant of XtX
            + (self.prior.a_tau + (self.nbasis + 1) / 2 - 1) * log(self.tau) - self.prior.a_tau * self.tau
            - (self.nbasis + 1) / 2 * log(2 * pi)
            + (self.prior.h1 + self.nbasis - 1) * log(self.lam) - self.lam * (self.prior.h2 + 1)
            )# curr$nbasis-1 because poisson prior is excluding intercept (for curr$nbasis instead of curr$nbasis+1)
    #-lfactorial(curr$nbasis) # added, but maybe cancels with prior
        self.log_post = lp
        return
        
    # REMEMBER - these function (methods) cannot be called inside the class definition
    def propose(self): # create XtX_cand, etc., including alpha, add them all to self
        pass
    
    def update(self): # make XtX = XtX_cand, etc.
        pass
    
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.s2 = 1.
        self.nbasis = 0
        self.tau = 1.
        self.s2_rate = 1.
        self.R = 1
        self.lam = 1
        self.I_star = np.ones(prior.maxInt) * prior.w1
        self.I_vec = self.I_star/np.sum(self.I_star)
        self.z_star = np.ones(data.p) * prior.w2
        self.z_vec = self.z_star/np.sum(self.z_star)
        self.basis = np.ones([data.n, 1])
        self.nc = 1
        self.knots = np.zeros([0, prior.maxInt])
        self.knotInd = np.zeros([0, prior.maxInt], dtype = int)
        self.signs = np.zeros([0, prior.maxInt], dtype = int)
        self.vars = np.zeros([0, prior.maxInt], dtype = "int8") # could do "bool_", but would have to transform 0 to -1
        self.n_int = 0
        self.Xty = np.zeros(prior.maxBasis + 2)
        self.Xty[0] = np.sum(data.y)
        self.XtX = np.zeros([prior.maxBasis + 2, prior.maxBasis + 2])
        self.XtX[0, 0] = data.n
        self.R = np.array([[sqrt(data.n)]])#np.linalg.cholesky(self.XtX[0, 0])
        self.R_inv_t = np.array([[1/sqrt(data.n)]])
        self.bhat = np.mean(data.y)
        self.qf = pow(sqrt(data.n) * np.mean(data.y), 2)
        self.count = np.zeros(3)
        return

class BassModel:

    def __init__(self, data, prior, nstore):
        self.data = data
        self.prior = prior
        self.state = BassState(self.data, self.prior)
        self.s2 =  np.zeros(nstore)# add all the other relevant parameters to store
        return
    
    def writeState(self): # take relevant parts of state and write to storage (only manipulates storage vectors)
        pass
    
    def plot(self):
        pass
    
    def predict(self):
        pass
    
   
def bass(xx, y, nmcmc = 10000, nburn = 9000, thin = 1, w1 = 5, w2 = 5, maxInt = 3, maxBasis = 1000, npart = None, g1 = 0, g2 = 0, s2_lower = 0, h1 = 10, h2 = 10, a_tau = 0.5, b_tau = None):
    bd = BassData(xx, y)
    bp = BassPrior(maxInt, maxBasis, npart, g1, g2, s2_lower, h1, h2, a_tau, b_tau, w1, w2)
    nstore = (nmcmc - nburn) / thin
    bm = BassModel(bd, bp, nstore) # if we add tempering, bm should have as many states as temperatures
    for i in range(nmcmc): # rjmcmc loop
        bm.state.propose()
        if(bm.state.proposal_alpha < log(np.random.rand())):
            bm.state.update()
        if i > nburn and ((i - nburn) % thin) == 0:
            bm.writeState()
        if i%%1000==0:
            print(datetime.now())
    del bm.writeState # the user should have access to this
    return bm

# how do others (random forest) do predict?  Make it a method for BassModel, or a standalone function?