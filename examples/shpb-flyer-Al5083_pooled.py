import numpy as np
from impala import superCal as sc
import matplotlib.pyplot as plt
import pyBASS as pb
import fdasrsf as fs
import pandas as pd
np.seterr(under='ignore')

#############################################################################
# TODO:
# Add SHPB experiments

#############################################################################
## get data

sim_inputs = np.genfromtxt('~/impala/data/Al-5083/flyer_data/sim_input.csv', delimiter=',', skip_header=1)
input_names = ['A', 'B', 'C', 'n', 'm', 'v1', 'v2', 'v3', 'G1', 'del2', 'del3']

xx104 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/xsims104.csv', delimiter=',')
xx105 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/xsims105.csv', delimiter=',')
xx106 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/xsims106.csv', delimiter=',')
xx_all_list = [xx104, xx105, xx106]

sims104 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/sims104.csv', delimiter=',')
sims105 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/sims105.csv', delimiter=',')
sims106 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/sims106.csv', delimiter=',')
sims_all_list = [sims104, sims105, sims106]
#sims_all = np.hstack(sims_all_list)

obs104 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/obs104.csv', delimiter=',')
obs105 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/obs105.csv', delimiter=',')
obs106 = np.genfromtxt('~/impala/data/Al-5083/flyer_data/obs106.csv', delimiter=',')
obs_all_list = [obs104, obs105, obs106]
#obs_all = np.hstack(obs_all_list)

nexp = len(sims_all_list)
n, M = sims_all_list[0].shape
p = len(input_names)

# what shape of discrepancy to have in v space for a shift in time
x1 = sims_all_list[0][0]
sp = np.linspace(-.07,.1,10)
ll = []
for ii in sp:
    ll.append(np.interp(xx104, xx104+ii, x1))

x3 = np.vstack(ll)
plt.plot(x1)
plt.plot(x3.T)
plt.show()

out = fs.fdawarp(x3.T, xx104)
out.multiple_align_functions(x1, parallel=True, lam=.01)

plt.plot(out.fn)
plt.show()

plt.plot(out.gam)
plt.show()

plt.plot(fs.geometry.gam_to_v(out.gam))
plt.show()




# warp curves to the first simulation # may want to choose a different one
# particularly, to get ones without elastic plateau, choose one with high elastic plateau?
gam_sim_list = [None] * nexp
vv_sim_list = [None] * nexp
ftilde_sim_list = [None] * nexp
gam_obs_list = [None] * nexp
vv_obs_list = [None] * nexp
ftilde_obs_list = [None] * nexp
for j in range(nexp):
    out = fs.fdawarp(sims_all_list[j].T, xx_all_list[j])
    out.multiple_align_functions(sims_all_list[j][0], parallel=True, lam=.01)
    #out.multiple_align_functions(obs_all_list[j], parallel=True, lam=.001)
    gam_sim_list[j] = out.gam
    vv_sim_list[j] = fs.geometry.gam_to_v(out.gam)
    ftilde_sim_list[j] = out.fn
    out2 = fs.pairwise_align_functions(sims_all_list[j][0], obs_all_list[j], xx_all_list[j], lam=.01)
    #out2 = fs.pairwise_align_functions(obs_all_list[j], obs_all_list[j], xx_all_list[j], lam=.01)
    gam_obs_list[j] = out2[1]
    vv_obs_list[j] = fs.geometry.gam_to_v(out2[1])
    ftilde_obs_list[j] = out2[0]
ftilde_obs_all = np.hstack(ftilde_obs_list)
vv_obs_all = np.hstack(vv_obs_list)

figure, axis = plt.subplots(nexp, 4)
xx = np.arange(200)/199
for j in range(nexp):
    axis[j, 0].plot(xx_all_list[j], sims_all_list[j].T, color='lightgrey')
    axis[j, 0].plot(xx_all_list[j], obs_all_list[j])
    axis[j, 1].plot(xx, ftilde_sim_list[j], color='lightgrey')
    axis[j, 1].plot(xx, ftilde_obs_list[j])
    axis[j, 2].plot(xx, gam_sim_list[j]*(np.max(xx_all_list[j])-np.min(xx_all_list[j]))+np.min(xx_all_list[j]), color='lightgrey')
    axis[j, 2].plot(xx, gam_obs_list[j]*(np.max(xx_all_list[j])-np.min(xx_all_list[j]))+np.min(xx_all_list[j]))
    axis[j, 3].plot(vv_sim_list[j], color='lightgrey')
    axis[j, 3].plot(vv_obs_list[j])
figure.set_size_inches(15,10)

axis[0,0].title.set_text('Misaligned')
axis[0,1].title.set_text('Aligned')
axis[0,2].title.set_text('Warping functions')
axis[0,3].title.set_text('Shooting vectors')

axis[0,0].set_ylabel('velocity (experiment 1)')
axis[1,0].set_ylabel('velocity (experiment 2)')
axis[2,0].set_ylabel('velocity (experiment 3)')

axis[0,2].set_ylabel('time')
axis[1,2].set_ylabel('time')
axis[2,2].set_ylabel('time')

axis[2,0].set_xlabel('time')
axis[2,1].set_xlabel('warped time')
axis[2,2].set_xlabel('warped time')

plt.tight_layout()

plt.savefig('./../../Desktop/foo.png', dpi=300)
#plt.savefig('foo.pdf')

plt.show()

starts = [np.where(sims_all_list[j][0]>1e-7)[0][0] for j in range(nexp)]

# emulator
# training data
ho = np.random.choice(n, 20, replace=False)
# ftilde_train = np.delete(np.vstack(ftilde_sim_list), ho, axis=1)
# ftilde_test = np.vstack(ftilde_sim_list)[:,ho]
# vv_train = np.delete(np.vstack(vv_sim_list), ho, axis=1)
# vv_test = np.vstack(vv_sim_list)[:,ho]
# gam_train = np.delete(np.vstack(gam_sim_list), ho, axis=1)
# gam_test = np.vstack(gam_sim_list)[:,ho]
Xtrain = np.delete(sim_inputs, ho, axis=0)
Xtest = sim_inputs[ho]

ftilde_train_list = [np.delete(ftilde_sim_list[j], ho, axis=1) for j in range(nexp)]
ftilde_test_list = [ftilde_sim_list[j][:,ho] for j in range(nexp)]
vv_train_list = [np.delete(vv_sim_list[j], ho, axis=1) for j in range(nexp)]
vv_test_list = [vv_sim_list[j][:,ho] for j in range(nexp)]
gam_train_list = [np.delete(gam_sim_list[j], ho, axis=1) for j in range(nexp)]
gam_test_list = [gam_sim_list[j][:,ho] for j in range(nexp)]

#mod = pb.bassPCA(X, y_sim, ncores=5, npc=5)
emu_ftilde_list = [pb.bassPCA(Xtrain, ftilde_train_list[j].T, ncores=15, npc=15, nmcmc=50000, nburn=10000, thin=40) for j in range(nexp)]
emu_vv_list = [pb.bassPCA(Xtrain, vv_train_list[j].T, ncores=9, npc=9, nmcmc=50000, nburn=10000, thin=40) for j in range(nexp)]
# emu_ftilde[0].plot()
# emu_vv[0].plot()

pred_ftilde = emu_ftilde_list[0].predict(Xtrain)
plt.plot(pred_ftilde.mean(0).T)
plt.show()

plt.plot(pred_ftilde.mean(0).T - ftilde_train_list[0])
plt.show()

plt.plot(ftilde_train_list[0])
plt.show()

pred_vv = emu_vv_list[0].predict(Xtrain)
plt.plot(pred_vv.mean(0).T - vv_train_list[0])
plt.show()

plt.plot(vv_train_list[2])
plt.plot(vv_obs_list[2])
plt.show()

plt.plot(sims_all_list[0][0])
plt.plot(sims_all_list[1][0])
plt.plot(sims_all_list[2][0])
plt.show()

plt.plot(emu_vv_list[0].trunc_error)
plt.show()

pred_gam = fs.geometry.v_to_gam(pred_vv.mean(0).T)
plt.plot(pred_gam - gam_train_list[0])
plt.show()

plt.plot(pred_gam)
plt.show()

plt.plot(emu_vv_list[0].trunc_error)
plt.show()

plt.plot(emu_ftilde_list[0].trunc_error)
plt.show()

bounds_mat = np.array([[sim_inputs[:,i].min(), sim_inputs[:,i].max()] for i in range(p)])
bounds = dict(zip(input_names, bounds_mat))

def constraint_funcion(x, bounds):
    k = list(bounds.keys())[0]
    good = x[k] < bounds[k][1]
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

ends = M-125
#sc = reload(sc)

#ind_true = 599 # change to one that is close to truth (get distance of aligned and v to experiment)
# initialize setup
setup = sc.CalibSetup(bounds, constraint_funcion)

# create appropriate model objects
for j in range(nexp):
    model_ftilde = sc.ModelBassPca_func(emu_ftilde_list[j], input_names) # for aligned data
    model_vv = sc.ModelBassPca_func(emu_vv_list[j], input_names) # for warping functions
    setup.addVecExperiments(ftilde_obs_list[j], 
        model_ftilde, 
        sd_est=np.array([0.001]), 
        s2_df=np.array([20.]), 
        s2_ind=np.array([0]*M))
    setup.addVecExperiments(vv_obs_list[j], 
        model_vv, 
        sd_est=np.array([0.1]), 
        s2_df=np.array([20.]), 
        s2_ind=np.array([0]*M), 
        D=np.hstack((
            np.vstack((np.ones([starts[j],1]), np.zeros([M-starts[j],1]))),
            np.vstack((np.zeros([starts[j],1]), np.ones([M-starts[j]-ends,1]), np.zeros([ends,1]))),
            np.vstack((np.zeros([starts[j],1]), np.linspace(-1,1,M-starts[j]-ends)[:,None], np.zeros([ends,1]))),
            np.vstack((np.zeros([M-ends,1]), np.ones([ends,1])))
        )),
        discrep_tau=1**2)

setup.setTemperatureLadder(1.05**np.arange(20))
setup.setMCMC(25000,1000,1,100)
out = sc.calibPool(setup)

import dill
#dill.dump_session('./../../Desktop/al-al-flyer_pool.pkl')
#dill.load_session('./../../Desktop/al-al-flyer_pool.pkl')


# I'll use this to burn/thin below
uu = np.arange(15000, 25000, 2)

plt.plot(out.theta[:,0,0])
plt.show()

ends2 = 78
D2=np.hstack((
    np.vstack((np.ones([starts[j],1]), np.zeros([M-starts[j],1]))),
    np.vstack((np.zeros([starts[j],1]), np.ones([M-starts[j]-ends2,1]), np.zeros([ends2,1]))),
    np.vstack((np.zeros([starts[j],1]), np.linspace(-1,1,M-starts[j]-ends)[:,None], np.zeros([ends,1]))),
    np.vstack((np.zeros([M-ends2,1]), np.ones([ends2,1])))
))

theta_dict = sc.tran_unif(out.theta[uu,0,:], setup.bounds_mat, setup.bounds.keys())
ftilde_pred_obs = [setup.models[j].eval(theta_dict) for j in range(0,nexp*2,2)] 
vv_pred_obs = [setup.models[j].eval(theta_dict) for j in range(1,nexp*2,2)] # note, it seems like adding discrepancy here doesn't really change anything
gam_pred_obs = [fs.geometry.v_to_gam(x.T) for x in vv_pred_obs]
gamd_pred_obs = [fs.geometry.v_to_gam(vv_pred_obs[j].T + setup.models[j*2 + 1].D @ out.discrep_vars[j*2 + 1][uu,0].T) for j in range(nexp)]
#gamd_pred_obs = [fs.geometry.v_to_gam(vv_pred_obs[j].T + D2 @ out.discrep_vars[j*2 + 1][uu,0].T) for j in range(nexp)]

plt.plot(setup.models[j*2 + 1].D)
plt.show()

figure, axis = plt.subplots(1,nexp)
for j in range(nexp):
    axis[j].plot(xx, ftilde_train_list[j],color='lightgrey')
    axis[j].plot(xx, ftilde_pred_obs[j].T,color='lightblue')
    #plt.plot(ftilde_train[:,ind_true],color='black')
    axis[j].plot(xx, ftilde_obs_list[j], color='black')
    axis[j].set_xlabel('warped time')
    axis[j].set_ylabel('velocity')
    plt.legend(['model runs','calibrated predictions','experiment'], loc='upper left', prop={'size': 6})
ax = plt.gca()
leg = axis[j].get_legend()
leg.legendHandles[0].set_color('lightgrey')
leg.legendHandles[1].set_color('lightblue')
leg.legendHandles[2].set_color('black')
figure.set_size_inches(10,5)
plt.tight_layout()
plt.savefig('./../../Desktop/pred_aligned.png', dpi=300)
plt.show()

figure, axis = plt.subplots(1,nexp)
for j in range(nexp):
    axis[j].plot(vv_train_list[j],color='lightgrey')
    #axis[j].plot(vv_pred_obs[j].T,color='blue')
    axis[j].plot(vv_pred_obs[j].T + setup.models[j*2 + 1].D @ out.discrep_vars[j*2 + 1][uu,0].T,color='lightblue')
    #plt.plot(ftilde_train[:,ind_true],color='black')
    axis[j].plot(vv_obs_list[j], color='black')
    #axis[j].set_xlabel('warped time')
    #axis[j].set_ylabel('velocity')
    plt.legend(['model runs','calibrated predictions','experiment'], loc='upper left', prop={'size': 6})
ax = plt.gca()
leg = axis[j].get_legend()
leg.legendHandles[0].set_color('lightgrey')
leg.legendHandles[1].set_color('lightblue')
leg.legendHandles[2].set_color('black')
figure.set_size_inches(10,5)
plt.tight_layout()
plt.savefig('~/impala_runs/Al5083/plots/pred_vv.png', dpi=300)
plt.show()

figure, axis = plt.subplots(1,nexp)
for j in range(nexp):
    axis[j].plot(xx, gam_train_list[j]*(np.max(xx_all_list[j])-np.min(xx_all_list[j]))+np.min(xx_all_list[j]),color='lightgrey')
    #plt.plot(gam_pred_obs[j],color='lightblue')
    axis[j].plot(xx, gamd_pred_obs[j]*(np.max(xx_all_list[j])-np.min(xx_all_list[j]))+np.min(xx_all_list[j]),color='lightblue')
    #plt.plot(ftilde_train[:,ind_true],color='black')
    axis[j].plot(xx, gam_obs_list[j]*(np.max(xx_all_list[j])-np.min(xx_all_list[j]))+np.min(xx_all_list[j]), color='black')
    axis[j].set_xlabel('warped time')
    axis[j].set_ylabel('time')
    plt.legend(['model runs','calibrated predictions','experiment'], loc='upper left', prop={'size': 6})
ax = plt.gca()
leg = axis[j].get_legend()
leg.legendHandles[0].set_color('lightgrey')
leg.legendHandles[1].set_color('lightblue')
leg.legendHandles[2].set_color('black')
figure.set_size_inches(10,5)
plt.tight_layout()
plt.savefig('~/impala_runs/Al5083/plots/pred_warping.png', dpi=300)
plt.show()


# trace plot for one theta
plt.plot(out.theta[:,0,4])
plt.show()

# trace plot for measurement error variance for ftilde
plt.plot(np.sqrt(out.s2[0][1000:,0,0]))
plt.show()

# trace plot for measurement error variance for vv
plt.plot(np.sqrt(out.s2[1][1000:,0,0]))
plt.show()

# trace plot of discrepancy coefficient

plt.plot(setup.models[1].D)
plt.show()

plt.plot(out.discrep_vars[1][:,0,:])
plt.show()

dat = pd.DataFrame(out.discrep_vars[1][uu,0,:])
g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu)}, corner=True, diag_kind='hist')
g
plt.show()

figure = plt.figure()
#xx_true = Xtrain[ind_true][None, :]
# pairs plot of parameters
import seaborn as sns
import pandas as pd
dat = pd.DataFrame(out.theta[uu,0,:])
#dat = dat.append(pd.DataFrame(xx_true))
#dat['col'] = ['blue']*len(uu) + ['red']
#g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu) + [50]}, corner=True, hue='col', diag_kind='hist')
g = sns.pairplot(dat, plot_kws={"s": [3]*len(uu)}, corner=True, diag_kind='hist')
g.set(xlim=(0,1), ylim = (0,1))
g.fig.set_size_inches(10,10)
g
plt.tight_layout()
plt.savefig('~/impala_runs/Al5083/plots/params.png', dpi=300)
plt.show()
plt.show()

# misaligned prediction
obspred_all_list = [np.zeros([len(uu), M]) for _ in range(nexp)]
for j in range(nexp):
    for i in range(len(uu)):
        obspred_all_list[j][i] = fs.warp_f_gamma(xx_all_list[j],ftilde_pred_obs[j][i,:],fs.invertGamma(gamd_pred_obs[j][:,i]))


figure, axis = plt.subplots(1,nexp)
for j in range(nexp):
    axis[j].plot(xx_all_list[j], sims_all_list[j].T, color="lightgrey")
    axis[j].plot(xx_all_list[j], obspred_all_list[j].T, color="lightblue")
    axis[j].plot(xx_all_list[j], obs_all_list[j], color="black")
    #axis[j].title.set_text('Misaligned')
    axis[j].set_xlabel('time')
    axis[j].set_ylabel('velocity')
    plt.legend(['model runs','calibrated predictions','experiment'], loc='upper left', prop={'size': 6})
ax = plt.gca()
leg = axis[j].get_legend()
leg.legendHandles[0].set_color('lightgrey')
leg.legendHandles[1].set_color('lightblue')
leg.legendHandles[2].set_color('black')
figure.set_size_inches(10,5)
plt.tight_layout()
plt.savefig('~/impala_runs/Al5083/plots/pred_misaligned.png', dpi=300)
plt.show()

axis[0,0].title.set_text('Misaligned')
axis[0,1].title.set_text('Aligned')
axis[0,2].title.set_text('Warping functions')
axis[0,3].title.set_text('Shooting vectors')

axis[0,0].set_ylabel('velocity (experiment 1)')
axis[1,0].set_ylabel('velocity (experiment 2)')
axis[2,0].set_ylabel('velocity (experiment 3)')

axis[0,2].set_ylabel('time')
axis[1,2].set_ylabel('time')
axis[2,2].set_ylabel('time')

axis[2,0].set_xlabel('time')
axis[2,1].set_xlabel('warped time')
axis[2,2].set_xlabel('warped time')

plt.tight_layout()

plt.savefig('./../../Desktop/pred_misaligned.png', dpi=300)

