
import plots
import numpy as np
import pandas as pd
import sqlite3 as sq
import impala_noProbitHC as impala
import models as models
import matplotlib.pyplot as plt
from importlib import reload
np.seterr(under='ignore')

#import scipy as sp




def f1(x):
    oo = x[0] #+ x[1]
    out = np.array([oo,oo,oo,oo,oo,oo])
    return out



p = 4
input_names = [str(v) for v in list(range(p))]
bounds = dict(zip(input_names,np.concatenate((np.zeros((p,1)),np.ones((p,1))),1)))



def cf(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good




impala = reload(impala)

setup = impala.CalibSetup(bounds, cf)
model1 = models.ModelF(f1, input_names)
#setup.addVecExperiments(np.array([.7,.7]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(np.array([.8,.8]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(np.array([.8,.8]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
setup.addVecExperiments(np.array([.7,.7,.8,.8,.8,.8]), model1, sd_est=np.array([0.01,0.01,0.01]), s2_df=np.array([50.,50.,50.]), s2_ind=np.array([0,0,1,1,2,2]))

#setup.addVecExperiments(np.array([.9,.9]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(np.array([.75,.75]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(np.array([.85,.85]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(np.array([.81,.81]), model1, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))

#setup.addVecExperiments(yobs3, model2, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(yobs3, model3, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
#setup.addVecExperiments(yobs4, model4, sd_est=np.array([0.01]), s2_df=np.array([50.]), s2_ind=np.array([0,0]))
setup.set_max_clusters(50)
setup.setTemperatureLadder(1.05**np.arange(10))
setup.setMCMC(15000,2000,1,100)

# there is a problem when you turn on decorrelation (no problem with tempering)


out = impala.calibHier(setup)
out2 = impala.calibPool(setup)

# hier mixing gets bad when prior for Sigma0 is small (why? prior for theta0 is uniform, so there is flexibility, but maybe we learn more about theta0), but results in crazy values when Sigma0 is big (in probit space, get tail bunching because prior is that way)
# maybe put prior for theta0 (and Sigma0) in native space, but propose from probit space (and include Jacobian)

plt.plot(impala.invprobit(out2.theta[:,0,0]))
plt.axhline(.7)
plt.show()


plt.plot(out.theta0[:,0,0])
plt.plot(out.theta[0][:,0,0,0])
plt.axhline(.7)
plt.show()

plt.plot(impala.invprobit(out2.theta[:,0,0]))
plt.plot((out.theta[0][:,0,0,0]))
plt.axhline(.7)
plt.show()

plt.plot(out.s2[0][10000:,0,0])
plt.plot(out2.s2[0][10000:,0,0])
plt.show()

out3 = impala.calibClust(setup)


out3.delta[0][-10:,0,0]
out3.delta[1][-10:,0,0]
out3.delta[2][-10:,0,0]

(out3.delta[2][-1000:,0,0]==out3.delta[0][-1000:,0,0]).mean()
(out3.delta[1][-1000:,0,0]==out3.delta[0][-1000:,0,0]).mean()
(out3.delta[1][-1000:,0,0]==out3.delta[2][-1000:,0,0]).mean()

out3.delta[3][-10:,0,0]

impala.invprobit(out3.theta_hist[0][-10:-1,0,0,0])
impala.invprobit(out3.theta_hist[1][-10:-1,0,0,0])
impala.invprobit(out3.theta_hist[2][-10:-1,0,0,0])

pp = plots.PTW_Plotter(setup, out3)
pp.cluster_matrix_plot(None)

import matplotlib.pyplot as plt
plt.plot((out3.theta_hist[0][:,0,0,0]))
plt.plot((out3.theta_hist[1][:,0,0,0]))
plt.plot((out3.theta_hist[2][:,0,0,0]))
plt.show()

plt.plot(impala.invprobit(out.theta[0][:,0,0,0]))
plt.plot(impala.invprobit(out.theta[1][:,0,0,0]))
plt.plot(impala.invprobit(out.theta[2][:,0,0,0]))
plt.show()

plt.plot(impala.invprobit(out.theta0[:,0,0]))

plt.plot(impala.invprobit(out3.theta0[:,0,0]))
plt.show()

plt.plot(impala.invprobit(out2.theta[:,0,0]))
plt.axhline(.5)
plt.axhline(.7)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


plt.plot(out.theta0[:,0,0])
plt.plot(out.theta[0][:,0,0,0])
plt.plot(out.theta[1][:,0,0,0])
plt.plot(out.theta[2][:,0,0,0])
plt.plot(out.theta[3][:,0,0,0])
plt.plot(out.theta[4][:,0,0,0])
plt.plot(out.theta[5][:,0,0,0])

plt.axhline((.7),color='r')
plt.axhline((.8),color='r')
plt.axhline((.9),color='r')
plt.axhline((.75),color='r')
plt.axhline((.85),color='r')
plt.axhline((.81),color='r')

plt.show()

# this is with relaxed updating of tau.  But it still seems to be going towards low acceptance.
# also has odd correlation.
# try this all with pooled version, see if (without decor and temper, for a not-bad model) the AM alone gets good mixing


plt.plot(np.sqrt(out.s2[0][5:,0]))
plt.plot(np.sqrt(out.s2[1][5:,0]))
plt.plot(np.sqrt(out.s2[2][5:,0]))
plt.plot(np.sqrt(out.s2[3][5:,0]))
plt.plot(np.sqrt(out.s2[4][5:,0]))
plt.plot(np.sqrt(out.s2[5][5:,0]))
plt.show()

plt.plot(np.sqrt(out.Sigma0[5:,0,0,0]))
plt.plot(np.sqrt(out.Sigma0[5:,0,1,1]))
plt.plot(np.sqrt(out.Sigma0[5:,0,2,2]))
plt.plot(np.sqrt(out.Sigma0[5:,0,3,3]))
plt.show()

dat = pd.DataFrame(impala.invprobit(out.theta[5000:,0,:]))
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

dat = pd.DataFrame(out2.theta[5000:,0,:])
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g
plt.show()


import seaborn as sns
dat = pd.DataFrame((out.theta[2][5000:,0,0,:]))
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

dat = pd.DataFrame((out.theta0[5000:,0,:]))
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

out.Sigma0[29999]

plt.plot(out.theta0[:,0,3])
plt.show()


import seaborn as sns
dat = pd.DataFrame(out3.theta_hist[2][5000:,0,0,:])
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

dat = pd.DataFrame(impala.invprobit(out3.theta_hist[1][5000:10000,0,0,:]))
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

dat = pd.DataFrame((out3.theta0[5000:,0,:]))
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()

dat = pd.DataFrame(out3.theta0[5000:10000,0,:])
g = sns.pairplot(dat, plot_kws={"s": 3}, corner=True, diag_kind='hist')
#g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()


# here are all the problems I know of right now:
# 1. Edge stacking (without tempering and decor) in theta that should be uniform (though not in theta0, so put a breakpoint when theta > 5, maybe a jacobian thing) - number of bugs in pooled version
# 2. Weird that theta0 left tail extends all the way to 0 when I only have experiments of .5 and .7.  Could it be priors??
# 3. Possibly related to 1, the adaptive metropolis seems to have trouble mixing









# current status (11-2-21)
# changed hier and clust to not use probit (to improve priors).
# added a better decor step to hier.
# could add new decor to clust (with tau)
# study priors in clust, figure out how to control