# %%

import numpy as np
import impala_noProbit_emu as impala
import models_withLik as models
import matplotlib.pyplot as plt
from importlib import reload
import pyBASS as pb

#%%


def f1(x):
    out = x[2]*x[0]*tt + x[4]
    return out

def f2(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out

nt = 50
tt = np.linspace(0, 1, nt)
n = 500
p = 9
x = np.random.rand(n, p)
xx = np.random.rand(1000, p)
e = np.random.normal(size=n * nt)
y = np.apply_along_axis(f2, 1, x) + e.reshape(n, nt)*1

modf = pb.bassPCA(x, y, ncores=5, npc=5)

#%%
xx_true = np.random.rand(1, p)
yobs1 = np.apply_along_axis(f1, 1, xx_true).reshape(nt) + np.concatenate(
    (np.random.normal(size=20)*.001, np.random.normal(size=30)*.03))
yobs2 = np.apply_along_axis(f2, 1, xx_true).reshape(nt) + np.random.normal(size=nt)*.01

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

impala = reload(impala)
models = reload(models)

setup = impala.CalibSetup(bounds, cf)
model1 = models.ModelF(f1, input_names)
#model2 = models.ModelF(f2, input_names)
model2 = models.ModelBassPca(modf, input_names)
setup.addVecExperiments(yobs1, model1, sd_est=np.ones(50)*.01, s2_df=np.ones(50)*20, s2_ind=np.arange(0,50,1), theta_ind=np.arange(0,50,1))
setup.addVecExperiments(yobs2, model2, sd_est=np.array([0.01]), s2_df=np.array([15.]), s2_ind=np.array([0] * nt), theta_ind=np.array([0] * nt))
setup.setTemperatureLadder(1.05**np.arange(30))
setup.setMCMC(15000,1000,1,100)
out = impala.calibHier(setup)
#out = impala.calibPool(setup)
uu = np.arange(10000, 15000, 2)

# CURRENTLY: if emulator error is large, you want to have informative prior for measurement error


#%%
import matplotlib.pyplot as plt
pred3 = setup.models[0].eval(impala.tran_unif(out.theta0[uu,0,:], setup.bounds_mat, setup.bounds.keys()))
pred1 = setup.models[0].eval(impala.tran_unif(out.theta[0][uu,0,1,:], setup.bounds_mat, setup.bounds.keys()))
plt.plot(pred3.T,color='grey')
plt.plot(pred1.T,color='blue')
plt.plot(setup.ys[0])
plt.show()

#pred2 = setup.models[1].eval(impala.tran_unif(out.theta[uu,0,:], setup.bounds_mat, setup.bounds.keys()))
pred3 = setup.models[1].eval(impala.tran_unif(out.theta0[uu,0,:], setup.bounds_mat, setup.bounds.keys()))
pred2 = setup.models[1].eval(impala.tran_unif(out.theta[1][uu,0,0,:], setup.bounds_mat, setup.bounds.keys()))
plt.plot(pred3.T,color='grey')
plt.plot(pred2.T,color='blue')
plt.plot(setup.ys[1])
plt.show()

# todo: tests to see when this doesnt work right, profiling, what to do with emulator error

#%%
import seaborn as sns
import pandas as pd
dat = pd.DataFrame(out.theta[uu,0,:])
dat = pd.DataFrame(out.theta[0][uu,0,0,:])
dat = pd.DataFrame(out.theta[1][uu,0,0,:])
dat = pd.DataFrame(out.theta0[uu,0,:])
dat = dat.append(pd.DataFrame(xx_true))
dat['col'] = ['blue']*len(uu) + ['red']
g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [50]}, corner=True, hue='col', diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()
#%%
out.count
#%%
out.count_decor
#%%
out.cov_theta_cand.tau
out.cov_ls2_cand[1].tau
#%%
np.sqrt(out.s2[0])

#%%

import matplotlib.pyplot as plt
plt.plot(out.theta[:,0,1])
plt.show()

plt.plot(np.sqrt(out.s2[0][5000:,0,0]))
plt.plot(np.sqrt(out.s2[0][5000:,0,1]))
plt.plot(np.sqrt(out.s2[1][5000:,0,0]))
plt.axhline(.001)
plt.axhline(.03)
plt.axhline(.01)
plt.show()

# this still has problems


#%%
#%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



fig = plt.figure()
ax = fig.gca(projection='3d')


# Plot the surface.
surf = ax.scatter(out.theta[uu,0,2],out.theta[uu,0,3],out.theta[uu,0,4], cmap=cm.coolwarm,
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
