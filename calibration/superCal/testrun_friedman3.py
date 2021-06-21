# %%

import numpy as np
import pandas as pd
import sqlite3 as sq
import pyBASS as pb
import impala
import models

#%%

#%%

def f1(x):
    out = 10. * np.sin(2*np.pi * tt * x[0])*x[7]
    return out

def f2(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out


def f1vec(x):
    out = 10. * np.sin(2*np.pi * tt * x[0])*x[7]
    return out

def f2vec(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out

tt = np.linspace(0, 1, 50)
p = 9

def d1(tt):
    return 0.#3. * tt

def d2(tt):
    return 0.#3. * tt**2 + 10.

#%%
xx_true = np.random.rand(1, p)
yobs1 = np.apply_along_axis(f1, 1, xx_true).reshape(len(tt)) + d1(tt) + np.random.normal(size=len(tt)) *.1
yobs2 = np.apply_along_axis(f2, 1, xx_true).reshape(len(tt)) + d2(tt) + np.random.normal(size=len(tt)) *.1



import matplotlib.pyplot as plt
plt.plot(yobs1[0])
plt.plot(yobs2[0])
plt.plot(np.apply_along_axis(f1, 1, xx_true)[0])
plt.plot(np.apply_along_axis(f2, 1, xx_true)[0])
#plt.plot(pred[0:300,:].T)
plt.show()


#%%
input_names = [str(v) for v in list(range(p))]
bounds = dict(zip(input_names,np.concatenate((np.zeros((p,1)),np.ones((p,1))),1)))



def cf(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

#%%

setup = impala.CalibSetup(bounds, cf)
model1 = models.ModelF(f1, input_names)
model2 = models.ModelF(f2, input_names)
setup.addVecExperiments(yobs1, model1, sd_est=np.array([0.1]), s2_df=np.array([10.]), s2_ind=np.array([0]*len(tt)))
setup.addVecExperiments(yobs2, model2, sd_est=np.array([0.1]), s2_df=np.array([10.]), s2_ind=np.array([0]*len(tt)))
setup.setTemperatureLadder(1.1**np.arange(10))
setup.setMCMC(30000,8000,1,100000)



out = impala.calibHier(setup)

## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## NOTE: sampling the same banana shaped posterior in both experiments, theta0 should follow the banana and Sigma0 should be small.
## since a tempering swap is for all thetas and theta0, it should be able to handle this.
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# here is what I've tried:
# - theta0_prior_cov 1 and Sigma0_prior_scale 1 -> basically uniform theta0 and good theta
# - theta0_prior_cov 1 and Sigma0_prior_scale .01 -> bad, probably too much shrinkage, but should be able to move around with theta0? Thats not happening for some reason.
# - theta0_prior_cov 1 and Sigma0_prior_scale 0.1 -> thetas get good prediction, but not full range (not uniform when it should be). Theta0 is odd.

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# need to deep dive into sampling of theta0 and Sigma0, make sure that is happening correctly
# timing note: the largest chunks are the distribution part that uses scipy (invwishart), should write my own density function (random sample function would be harder)
# probably need to move to probit space
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#out = impala.calibPool(setup)

if False:

    #%%
    import matplotlib.pyplot as plt
    uu = range(25000, 30000, 5)
    pred1 = setup.models[0].eval(impala.tran(out.theta[0][uu,0,0,:], setup.bounds_mat, setup.bounds.keys()))
    pred2 = setup.models[1].eval(impala.tran(out.theta[1][uu,0,0,:], setup.bounds_mat, setup.bounds.keys()))
    plt.plot(pred1.T,color='grey')
    plt.plot(pred2.T,color='blue')
    plt.plot(yobs1,color='yellow')
    plt.plot(yobs2,color='orange')
    plt.plot(out.pred_curr[0][0,0,:], '--')
    plt.plot(out.pred_curr[1][0,0,:], '--')
    #plt.plot(pred[0:300,:].T)
    plt.show()

    pred1[1000,:] - out.pred_curr[0][0,0,:]
    pred2[9,:] - out.pred_curr[1][0,0,:]

    impala.tran(out.theta[0][uu[0],0,0,:], setup.bounds_mat, setup.bounds.keys())
    out.theta[0][uu[0],0,0,:]
    # todo: tests to see when this doesnt work right, profiling, what to do with emulator error

    #%%
    import seaborn as sns
    dat = pd.DataFrame(out.theta0[uu,0,:])
    dat = dat.append(pd.DataFrame(xx_true))
    dat['col'] = ['blue']*len(uu) + ['red']
    g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [50]}, corner=True, hue='col', diag_kind='hist')
    g.set(xlim=(0,1), ylim = (0,1))
    g
    plt.show()

    dat = pd.DataFrame(out.theta[0][uu,0,0,:])
    dat = dat.append(pd.DataFrame(xx_true))
    dat['col'] = ['blue']*len(uu) + ['red']
    g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [50]}, corner=True, hue='col', diag_kind='hist')
    g.set(xlim=(0,1), ylim = (0,1))
    g
    plt.show()

    dat = pd.DataFrame(out.theta[1][uu,0,0,:])
    dat = dat.append(pd.DataFrame(xx_true))
    dat['col'] = ['blue']*len(uu) + ['red']
    g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [50]}, corner=True, hue='col', diag_kind='hist')
    g.set(xlim=(0,1), ylim = (0,1))
    g
    plt.show()

##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Shrinking too much????

    #%%
    out.count
    #%%
    out.count_decor
    #%%
    out.tau # note: if these are getting too small (which happens if s2 gets really small), things degenerate (since adaptation is harder as we go further in the chain)
    #%%
    np.sqrt(out.s2)

    #%%

    import matplotlib.pyplot as plt
    plt.plot(out.theta[1][:,0,0,3])
    plt.show()

    plt.plot(np.log(out.s2[1][:,0,0]))
    plt.show()

    #%%
    #%matplotlib qt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter



    fig = plt.figure()
    ax = fig.gca(projection='3d')


    # Plot the surface.
    surf = ax.scatter(out.theta[25000:30000,0,2],out.theta[25000:30000,0,3],out.theta[25000:30000,0,4], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(0, 1)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # rotate the axes and update
    #for angle in range(0, 360):
    #   ax.view_init(30, 30)

    ax.view_init(30, -90)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # %%
