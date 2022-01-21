import numpy as np
import impala_noProbit_emu as impala
import models_withLik as models
import matplotlib.pyplot as plt
from importlib import reload
import pyBASS as pb
np.seterr(under='ignore')


def f1(x):
    out = np.repeat(x[0], 50)
    return out

def f2(x):
    out = np.repeat(x[1], 10)
    return out



p = 2
input_names = [str(v) for v in list(range(p))]
bounds = dict(zip(input_names,np.concatenate((np.zeros((p,1)),np.ones((p,1))),1)))



def cf(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good


x = np.random.rand(100, 2)
y = np.kron(x[:,1].reshape([100,1]), np.ones([1,10]))
mod = pb.bassPCA(x,y,npc=1, maxInt=1)
#mod.plot()

impala = reload(impala)
models = reload(models)


setup = impala.CalibSetup(bounds, cf)
model1 = models.ModelF(f1, input_names)
model2 = models.ModelBassPca(mod, input_names)
#model2 = models.ModelF(f2, input_names)

#D2 =  

setup.addVecExperiments(np.repeat(0.8, 50)+np.concatenate((np.random.normal(size=20)*.01, np.random.normal(size=30)*.03)), model1, sd_est=np.array([0.01,0.03]), s2_df=np.array([5.,5.]), s2_ind=np.array([0]*20 + [1]*30), theta_ind=np.array([0]*20 + [1]*30))
setup.addVecExperiments(np.repeat(0.3, 10)+np.random.normal(size=10)*.001, model2, sd_est=np.array([0.001]), s2_df=np.array([5.]), s2_ind=np.array([0]*10), theta_ind=np.array([0]*10), D=D2)


setup.setTemperatureLadder(1.05**np.arange(10))
setup.setMCMC(15000,2000,1,100)

out = impala.calibHier(setup)

# left off fixing bugs in emulator calibration part - first, correlation being too big

#out = impala.calibPool(setup)

out.theta0[13990,0,:]
np.sqrt(out.s2[0][14999][0])

plt.plot(out.theta[0][:,0,0,0])
plt.plot(out.theta[0][:,0,1,0])
plt.plot(out.theta[1][:,0,0,1])
plt.axhline(.82)
plt.axhline(.78)
plt.show()
plt.plot(out.theta0[:,0,:])
plt.show()



plt.plot(np.sqrt(out.s2[0][5000:,0,0]))
plt.plot(np.sqrt(out.s2[0][5000:,0,1]))
plt.axhline(.01)
plt.axhline(.03)
plt.plot(np.sqrt(out.s2[1][5000:,0,0]))
plt.axhline(.001)
plt.show()


# TODO:
# play with priors for shrinkage?