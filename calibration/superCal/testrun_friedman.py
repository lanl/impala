# %%

import numpy as np
import pandas as pd
import sqlite3 as sq
import pyBASS as pb
import impala
import models

#%%

def f2(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out


tt = np.linspace(0, 1, 50)
n = 500
p = 9
x = np.random.rand(n, p)
xx = np.random.rand(1000, p)
e = np.random.normal(size=n * len(tt))
y = np.apply_along_axis(f2, 1, x)  # + e.reshape(n,len(tt))

modf = pb.bassPCA(x, y, ncores=3, percVar=99.99)

#%%
xx_true = np.random.rand(1, p)
yobs = np.apply_along_axis(f2, 1, xx_true)

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
model = models.ModelBassPca(modf, input_names)
setup.addVecExperiments(yobs, model, sd_est=np.array([0.01]), s2_df=np.array([10.]), s2_ind=np.array([0]*len(tt)))
setup.setTemperatureLadder(1.15**np.arange(100))
setup.setMCMC(30000,10000,1,100)
out = impala.calibPool(setup)

#%%
import matplotlib.pyplot as plt
pred = setup.models[0].eval(impala.tran(out.theta[:,0,:], setup.bounds_mat, setup.bounds.keys()))
plt.plot(pred[25000:30000,:].T,color='grey')
plt.plot(yobs[0])
#plt.plot(pred[0:300,:].T)
plt.show()

# todo: tests to see when this doesnt work right, profiling, what to do with emulator error

#%%
import seaborn as sns
dat = pd.DataFrame(out.theta[25000:30000,0,:])
dat.append(pd.DataFrame(xx_true))
dat['col'] = ['blue']*4999 + ['red']
g = sns.pairplot(dat, plot_kws={"s": [3]*4999 + [50]}, corner=True, hue='col', diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
#%%
out.count
#%%
out.count_decor
#%%
out.tau
#%%
np.sqrt(out.s2)

#%%

import matplotlib.pyplot as plt
plt.plot(out.theta[:,0,4])
plt.show()

plt.plot(np.log(out.s2[0][:,0,0]))
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
