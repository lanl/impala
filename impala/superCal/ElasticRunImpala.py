import numpy as np
from scipy.interpolate.interpolate import interp1d
#import impala_noProbit_emu as impala
#import models_withLik as models
from impala import superCal as sc
import matplotlib.pyplot as plt
from importlib import reload
import pyBASS as pb
import fdasrsf as fs
import pandas as pd
from scipy.interpolate import UnivariateSpline

path = './../../../../Documents/move/al_flyers_dat/'

np.random.seed(120121)

M = 200
sim_output = np.genfromtxt(path + 'sim_output_y.csv', delimiter=',')
sim_time = np.genfromtxt(path + 'sim_output_x.csv', delimiter=',')
sim_time2 = np.linspace(sim_time.min(), sim_time.max(), M)
dat = sim_output.T

f = np.empty((M, dat.shape[1]))
M1 = dat.shape[0]
x = np.linspace(0, 1, M)
for i in range(0, dat.shape[1]):
    obj = UnivariateSpline(np.linspace(0, 1, M1), dat[:,i], s=0)
    f[:, i] = obj(x)

ho = np.random.choice(1000, 20, replace=False)
y_train = np.delete(f, ho, axis=1)  # training data
y_test = f[:, ho]  # test data

sim_inputs = np.genfromtxt(path + 'sim_input.csv', delimiter=',', skip_header=1)
X_train = np.delete(sim_inputs, ho, axis=0)
X_test = sim_inputs[ho,:]

# warp training curves to the first simulation
out = fs.fdawarp(y_train, x)
out.multiple_align_functions(y_train[:, 0], parallel=True, lam=.01)
gam_train = out.gam
vv_train = fs.geometry.gam_to_v(gam_train)
ftilde_train = out.fn
qtilde_train = out.qn

# normalize inputs
ftilde_sd = np.std(ftilde_train)
vv_train_sd = np.std(vv_train)
y_sim = np.vstack((ftilde_train/ftilde_sd,vv_train/vv_train_sd))

y_obs_dat = np.genfromtxt(path + 'Data_S104S.txt',skip_header=2)
y_obs_dat[:, 0] = y_obs_dat[:, 0]*1e-4
y_obs2 = np.interp(sim_time2, y_obs_dat[:, 1], y_obs_dat[:, 0])
f2tilde, gam, q2tilde = fs.pairwise_align_functions(y_train[:, 0], y_obs2, x, lam=.01)
v_obs = fs.geometry.gam_to_v(gam)
y_obs = np.concatenate((f2tilde/ftilde_sd, v_obs/vv_train_sd))



# rescale inputs, for plotting later
X = pb.normalize(sim_inputs, np.concatenate((sim_inputs.min(axis=0)[np.newaxis, ...],sim_inputs.max(axis=0)[np.newaxis, ...])).T)
Xtrain = np.delete(X, ho, axis=0)

# emulator
#mod = pb.bassPCA(X, y_sim, ncores=5, npc=5)
emu_ftilde = pb.bassPCA(Xtrain, y_sim[0:200].T, ncores=15, npc=15)
emu_vv = pb.bassPCA(Xtrain, y_sim[200:400].T, ncores=10, npc=10)
# emu_vv.plot()

#emu_both = pb.bassPCA(Xtrain, y_sim.T, ncores=10, npc=20)

p = X.shape[1]
input_names = [str(v) for v in list(range(p))]
bounds = dict(zip(input_names, np.concatenate((np.zeros((p, 1)),np.ones((p, 1))), 1)))



def constraint_funcion(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good


sc = reload(sc)


# initialize setup
setup = sc.CalibSetup(bounds, constraint_funcion)

# create appropriate model objects
model_ftilde = sc.ModelBassPca_func(emu_ftilde, input_names) # for aligned data
model_vv = sc.ModelBassPca_func(emu_vv, input_names) # for warping functions
# now add the experiments to the setup
# First add the aligned data
setup.addVecExperiments(y_obs[0:200], model_ftilde, sd_est=np.array([0.01]), s2_df=np.array([20.]), s2_ind=np.array([0]*200))
# then the warping function
setup.addVecExperiments(y_obs[200:400], model_vv, sd_est=np.array([0.01]), s2_df=np.array([20.]), s2_ind=np.array([0]*200), D=np.ones([200,1]))
# note: alternatively could use a joint emulator, but have a separate calibration error variance for the aligned data and warping functions
#setup.addVecExperiments(y_obs, model1, sd_est=np.array([0.1, 0.7]), s2_df=np.array([20., 20.]), s2_ind=np.array([0]*200 + [1]*200))
# if you want to use tempering, specify a temperature ladder (typically geometrically spaced)
setup.setTemperatureLadder(1.05**np.arange(20))
# set number of MCMC iterations, burn, thin, space between decorrelation steps
setup.setMCMC(15000,1000,1,100)
# run the calibration!
out = sc.calibPool(setup)

# I'll use this to burn/thin below
uu = np.arange(10000, 15000, 2)


ftilde_pred_obs = setup.models[0].eval(sc.tran_unif(out.theta[uu,0,:], setup.bounds_mat, setup.bounds.keys())) * ftilde_sd
plt.plot(ftilde_train,color='lightgrey')
plt.plot(ftilde_pred_obs.T,color='lightblue')
plt.plot(f2tilde, color='black')
plt.show()


vpred_obs = setup.models[1].eval(sc.tran_unif(out.theta[uu,0,:], setup.bounds_mat, setup.bounds.keys())) * vv_train_sd
gampred_obs = fs.geometry.v_to_gam(vpred_obs.T).T
plt.plot(vv_train,color='lightgrey')
plt.plot(vpred_obs.T,color='lightblue')
plt.plot(v_obs, color='black')
plt.show()

plt.plot(gam_train,color='lightgrey')
plt.plot(gampred_obs.T,color='lightblue')
plt.plot(gam, color='black')
plt.show()

# trace plot for one theta
plt.plot(out.theta[:,0,0])
plt.show()

# trace plot for measurement error variance for ftilde
plt.plot(np.sqrt(out.s2[0][1000:,0,0]))
plt.show()

# trace plot for measurement error variance for vv
plt.plot(np.sqrt(out.s2[1][1000:,0,0]))
plt.show()

# trace plot of discrepancy coefficient
plt.plot(out.discrep_vars[1][:,0,:])
plt.show()

# pairs plot of parameters
import seaborn as sns
import pandas as pd
dat = pd.DataFrame(out.theta[uu,0,:])
g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu)}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g
plt.show()




# unwarp<-function(gg,ywarp,x){
#   seq01<-seq(0,1,length.out=length(gg))
#   x.unwarped<-BASS:::unscale.range(approx(seq01,gg,seq01)$y,range(x))
#   approx(x.unwarped,ywarp,x)$y
# }

def unnormalize(z, x_min, x_max):
    return z * (x_max - x_min) + x_min

from scipy.interpolate import interp1d
def unwarp(gam, ftilde, x):
    seq01 = np.linspace(0, 1, len(gam))
    x_unwarped = sc.unnormalize(gam, min(x), max(x))
    ifunc = interp1d( x_unwarped, ftilde, kind = 'linear')
    return ifunc(x)