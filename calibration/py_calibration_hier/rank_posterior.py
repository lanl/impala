import sqlite3 as sq
import numpy as np
from physical_models_c import MaterialModel
from scipy.special import erf, erfinv
from math import ceil, sqrt, pi, log
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

## settings of interest
edot = 10**5 * 1e-6 #10^5/s
strain = 1.
temp = 700. # Kelvin
res_path = './results/Ti64/res_ti64_hier-test.db'
dat_path = './data/data_Ti64.db'
name = 'res_ti64_hier3'
out_path = './results/Ti64/'
nexp = 197

## connect to calibration output
con = sq.connect(res_path)
cursor = con.cursor()

## get posterior samples of overall mean parameters
cursor.execute("SELECT * FROM 'phi0';")
phi0 = cursor.fetchall()
phi0_names = list(map(lambda x: x[0], cursor.description))

## get posterior samples of overall mean parameters, standardized
cursor.execute("SELECT * FROM 'theta0';")
theta0 = cursor.fetchall()
theta0_names = list(map(lambda x: x[0], cursor.description))

## get posterior samples of covariance matrix for standardized parameters
cursor.execute("SELECT * FROM 'Sigma';")
Sigma = cursor.fetchall()
Sigma_names = list(map(lambda x: x[0], cursor.description))

## get constants used
cursor.execute("SELECT * FROM 'constants';")
constants = dict(cursor.fetchall())

## get models used
cursor.execute("SELECT * FROM 'models';")
models = dict(cursor.fetchall())

## get bounds, make into a dict like when they are input
cursor.execute("SELECT * FROM 'bounds';")
bounds = cursor.fetchall()
bounds_names = list(map(lambda x: x[0], cursor.description))
parameter_bounds = {bounds_names[idx] : [bounds[0][idx], bounds[1][idx]] for idx in range(len(bounds_names))}

nmcmc = len(phi0)
nparams = len(phi0[0])



def getStrength(edot, strain, temp, params, model_args, consts):
    # get stress at given strain, edot, temp, and params
    model = MaterialModel(flow_stress_model=model_args['flow_stress_model'],shear_modulus_model=model_args['shear_modulus_model'])
    model.set_history_variables(strain, edot, 100)

    # ensure correct ordering
    constant_list = model.get_constant_list()
    param_list = model.get_parameter_list()
    constant_vec = np.array([consts[key] for key in constant_list])
    param_vec = np.array([params[key] for key in param_list])

    model.initialize_constants(constant_vec)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temp)
    return model.compute_state_history()[99, 2]



def unnormalize(z,bounds):
    """ Transform 0-1 scale to real scale """
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

def invprobit(y):
    """
    Inverse Probit Transformation
    For real-valued variable y, result x is bounded 0-1
    """
    return 0.5 * (1 + erf(y / sqrt(2.)))




# temporary model setup
model_temp = MaterialModel(flow_stress_model=models['flow_stress_model'],shear_modulus_model=models['shear_modulus_model'])
param_list = model_temp.get_parameter_list()
bounds = np.array([parameter_bounds[key] for key in param_list]) # order bounds correctly

## get phi, stress
phi = np.empty([nmcmc,nparams]) # store parameter samples here (including Sigma uncertainty)
stress = np.zeros(nmcmc) # store stress here
for i in range(nmcmc):
    th0 = theta0[i]
    S = np.array(Sigma[i]).reshape(nparams,nparams)
    cst = False
    while not cst: # do until sample meets constraints
        th = np.random.multivariate_normal(th0, S, 1) # get a sample
        ph = unnormalize(invprobit(th), bounds)[0] # unstandardize the sample
        model_temp.update_parameters(ph) # update model parameters so we can check constraints
        cst = model_temp.check_constraints() # check constraints

    params = dict(zip(theta0_names,ph)) # make into a dict

    phi[i] = ph
    stress[i] = getStrength(edot, strain, temp, params, models, constants)


out = pd.DataFrame(phi,columns=param_list) # make into a dataframe
out['stress'] = stress # append
out['rank'] = ss.rankdata(stress) # append


## write to file
template = "edot(1/s)=" + str(edot/1e-6) + ", strain=" + str(strain) + ", temp(K)=" + str(temp) + "\n{}"

with open(out_path + name + '_postSamplesPTW.csv', 'w') as fp:
    fp.write(template.format(out.to_csv(index=False)))



def getStress(edot, temp, params, model_args, consts):
    model = MaterialModel(flow_stress_model=model_args['flow_stress_model'],shear_modulus_model=model_args['shear_modulus_model'])
    model.set_history_variables(0.6, edot, 100)

    constant_list = model.get_constant_list()
    param_list = model.get_parameter_list()
    constant_vec = np.array([consts[key] for key in constant_list])
    param_vec = np.array([params[key] for key in param_list])

    model.initialize_constants(constant_vec)
    model.update_parameters(np.array(param_vec))
    model.initialize_state(temp)
    return model.compute_state_history()[:, 1:3]

## get meta data
con = sq.connect(dat_path)
cursor = con.cursor()
cursor.execute("SELECT * FROM meta;")
meta_names = list(map(lambda x: x[0], cursor.description))
meta = pd.DataFrame(cursor.fetchall(),columns=meta_names)

## get SHPB experimental data
dat_all = []
for i in range(nexp):
    ## get first datset
    cursor.execute("SELECT * FROM data_" + str(i+1) + ";")
    dat_all.append(cursor.fetchall())



## get posterior predictive distributions
con = sq.connect(res_path)
cursor = con.cursor()

phii = np.empty([nmcmc,nparams,nexp])
phi_stress = np.empty([nmcmc,100,nexp])
phi0_stress = np.empty([nmcmc,100,nexp])
phii_stress = np.empty([nmcmc,100,nexp])
for j in range(nexp):
    cursor.execute("SELECT * FROM 'subchain_" + str(j) + "_phi';")
    phii[:,:,j] = cursor.fetchall()
    for i in range(nmcmc):
        params = dict(zip(theta0_names, phi0[i]))  # make into a dict
        phi0_stress[i,:,j] = getStress(meta.edot[j], meta.temperature[j], params, models, constants)[:,1]

        params = dict(zip(theta0_names, phi[i]))  # make into a dict
        phi_stress[i,:,j] = getStress(meta.edot[j], meta.temperature[j], params, models, constants)[:,1]

        params = dict(zip(theta0_names, phii[i,:,j]))  # make into a dict
        phii_stress[i, :, j] = getStress(meta.edot[j], meta.temperature[j], params, models, constants)[:, 1]
    print(j)

xx = getStress(meta.edot[0], meta.temperature[0], params, models, constants)[:,0]

## get quantiles
phi_quant = np.quantile(phi_stress, [.025,.975], 0)
phi0_quant = np.quantile(phi0_stress, [.025,.975], 0)
phii_quant = np.quantile(phii_stress, [.025,.975], 0)

## get standard deviations
stdev = np.empty([nmcmc,197])
for j in range(197):
    cursor.execute("SELECT * FROM 'subchain_" + str(j) + "_sigma2';")
    stdev[:,j] = np.sqrt(np.array(cursor.fetchall()).reshape(nmcmc))
stdev_mean = stdev.mean(0)

## plot SHPB predictions
ind = np.where(meta.type == 'shpb')[0].tolist()

nx = 9
ny = 8
ip = 0
k = 0
for i in ind:
    if ip % (nx*ny) == 0:
        ip = 0
        if i>0:
            plt.savefig(out_path + name + '_postpredSHPB'+str(k)+'.png', bbox_inches='tight')
        k+=1
        plt.figure(k, figsize=(20, 15))
#for i in range(nx*ny):
    ax1=plt.subplot(ny,nx,ip+1)
    plt.fill_between(xx, phi_quant[0,:,i], phi_quant[1,:,i],color='lightgrey',label=r'$\theta^*$')
    plt.fill_between(xx, phi0_quant[0,:,i], phi0_quant[1,:,i],color='lightblue',label=r'$\theta_0$')
    plt.fill_between(xx, phii_quant[0,:,i], phii_quant[1,:,i],color='lightgreen',label=r'$\theta_i$')
    plt.scatter(np.array(dat_all[i])[:,0], np.array(dat_all[i])[:,1],color='blue',s=.5,label='y')
    plt.vlines(x=np.array(dat_all[i])[:,0], ymin=np.array(dat_all[i])[:,1] - 2*stdev_mean[i], ymax=np.array(dat_all[i])[:,1] + 2*stdev_mean[i],color='blue',label='')
    plt.ylim(0,.027)
    plt.xlim(0, .6)
    ax = plt.gca()
    if ip < ny*nx-nx:
        ax.axes.xaxis.set_visible(False)
    if (ip+1) % nx != 1:
        ax.axes.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.annotate(str(round(meta.edot[i]/1e-6,5)) + "/s  " + str(int(meta.temperature[i])) + "K",xy=(0.01, 0.9), xycoords='axes fraction')
    if (ip+1)==(nx*ny):
        ax.legend()
    ip+=1
    if i == ind[-1]:
        plt.savefig(out_path + name + '_postpredSHPB'+str(k)+'.png', bbox_inches='tight')

## TODO
# try different shrinkage levels
# plots for FP & TC




#sns.set_theme(style="ticks")
#df = pd.DataFrame(phi0arr)
#sns.pairplot(df)
#
# i=3
# j=4
# n=12
# plt.scatter(phii[:,i,n],phii[:,j,n])
# oo = contx(phii[:, i,n], phii[:, j,n])
# plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'], colors='blue')
#
# phii[:,:,n].mean(axis=0)
#
# from sklearn.mixture import GaussianMixture
# cl = GaussianMixture(2).fit(phii[:,:,n])
#
# d1 = (((phii[:,:,n] - cl.means_[0])/bounds[:,1])**2).sum(1)
# d2 = (((phii[:,:,n] - cl.means_[1])/bounds[:,1])**2).sum(1)
#
# bounds
# c1 = d1 < d2
#
# df = pd.DataFrame(phii[:,:,n])
# df['cat'] = c1
# sns.pairplot(df,hue = 'cat')
#
# df = pd.DataFrame(phii[:,:,n])
# df.loc[2000] = cl.means_.mean(0)#phii[:,:,n].mean(axis=0)
# df['cat'] = 1
# df['cat'][2000] = 2
# sns.pairplot(df,hue = 'cat')
#
# params = dict(zip(theta0_names, cl.means_.mean(0)))  # make into a dict
# plt.plot(xx,getStress(meta.edot[n], meta.temperature[n], params, models, constants)[:, 1])
#
# plt.fill_between(xx, phi_quant[0,:,n], phi_quant[1,:,n],color='lightgrey',label=r'$\theta^*$')
# plt.fill_between(xx, phi0_quant[0,:,n], phi0_quant[1,:,n],color='lightblue',label=r'$\theta_0$')
# plt.fill_between(xx, phii_quant[0,:,n], phii_quant[1,:,n],color='lightgreen',label=r'$\theta_i$')
#
# cols = ['red','purple']
# for i in range(2000):
#     plt.plot(xx,phii_stress[i,:,n],color=cols[int(c1[i])],zorder=1)
#
# plt.scatter(np.array(dat_all[n])[:,0], np.array(dat_all[n])[:,1],color='blue',s=1,label='y',zorder=2)

# multimodality is justified with this small of s2.  If we relax the s2 prior, may get fewer modes.  This is an interesting phenomena, where likelihood modes get more peaked as s2 gets small
# note: also want to fiddle with shrinkage priors...all priors really



## plot parameter pairs plots, 90% contours

def contx(x1,x2,perc=.9): # get contour for percecntile using kde
    dd = ss.gaussian_kde([x1,x2],bw_method='silverman')
    X, Y = np.mgrid[min(x1):max(x1):100j, min(x2):max(x2):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    z = dd(positions)
    z = z/z.sum()

    t = np.linspace(0, z.max(), 1000)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))

    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([perc]))
    return {'X':X, 'Y':Y, 'Z':z.reshape([100,100]), 'conts':t_contours }

phi0arr = np.array(phi0)
#nexp = 3
plt.figure(1, figsize=(15, 15))

for i in range(nparams):
    for j in range(nparams):
        if i == j:
            plt.subplot2grid((nparams, nparams), (i, j))

            for k in range(nexp):
                sns.distplot(phii[:, i, k], hist=False, kde=True,color='lightgreen')

            sns.distplot(phi0arr[:,i], hist=False, kde=True,color='blue')

            sns.distplot(phi[:, i], hist=False, kde=True,color='grey')

            plt.xlim(bounds[i,0], bounds[i,1])
            ax = plt.gca()
            ax.axes.yaxis.set_visible(False)
            plt.xlabel(theta0_names[i])
            ax.tick_params(axis='x', which='major', labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        if i < j:
            plt.subplot2grid((nparams, nparams), (i, j))

            for k in range(nexp):
                oo = contx(phii[:, j, k], phii[:, i, k])
                plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'], colors='lightgreen')

            oo = contx(phi0arr[:,j], phi0arr[:,i])
            plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'],colors = 'blue')

            oo = contx(phi[:,j], phi[:,i])
            plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'],colors='grey')

            plt.xlim(bounds[j, 0], bounds[j, 1])
            plt.ylim(bounds[i, 0], bounds[i, 1])
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            print(i)
plt.subplots_adjust(wspace=.05, hspace=.05)
plt.subplot2grid((nparams, nparams), (2, 0))
from matplotlib.lines import Line2D
colors = ['lightgreen','blue','grey']
lines = [Line2D([0],[0],color=c,linewidth=2) for c in colors]
labels = [r'$\theta_i$',r'$\theta_0$',r'$\theta^*$']
plt.legend(lines,labels)
plt.axis('off')

plt.savefig(out_path + name + '_postThetas.png', bbox_inches='tight')