#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:01:16 2020

@author: dfrancom
"""

import numpy as np
from math import pi, sqrt, log, erf, exp, sin
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# generate data
def f(x):
  out = 10. * np.sin(pi * x[:,0] * x[:,1]) + 20. * (x[:,2] - .5)**2 + 10 * x[:,3] + 5. * x[:,4]
  return out

n = 5000
p = 6
x = np.random.rand(n,p)
xx = np.random.rand(1000,p)
y = f(x)


# Plot
plt.scatter(x[:,0], y)
df = pd.DataFrame(x)
df['y'] = y
sns.pairplot(df)

# source pyBASS.py

knots = np.array([.2,.1,.3,.5])
data = x
signs = np.array([-1,1,1,-1])
vs = np.array([1,3,4,5])

knots = np.array([.2])
data = x
signs = np.array([-1])
vs = np.array([1])

const(signs,knots)

## make basis function (from continuous variables)



    
plt.scatter(data[:,vs], makeBasis(signs,vs,knots,x))

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='red')


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x, y);
y_hat = rf.predict(xx)

plt.scatter(f(xx), y_hat)
abline(1,0)

path = '~/git/immpala/data/Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T-196C.csv'
data = pd.read_csv(path, sep = '\t|\s+|,|;', engine = 'python').values


path = '~/git/immpala/data/ti-6al-4v/Data/SHPB/Babu2013/Babu2013_Fig3-293K-1e-2.csv'
data = pd.read_csv(path, skiprows = range(0,17))
