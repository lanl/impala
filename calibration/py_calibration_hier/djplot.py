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
#import ipdb #ipdb.set_trace()

## settings of interest
edot = 40250. * 1e-6 # first term is per second
strain = 0.6
temp = 694. # Kelvin
res_path = './results/Ti64/res_ti64_hier.db'
dat_path = './data/data_Ti64.db'
name = 'djplot'
out_path = './results/Ti64/'
nexp = 197
plot = True
write = False

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
#cursor.execute("SELECT * FROM 'bounds';")
#bounds = cursor.fetchall()
#parameter_bounds = {bounds[idx][0] : [bounds[idx][1], bounds[idx][2]] for idx in range(len(bounds))}

parameter_bounds = {
    'theta0' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'gamma' : (0.000001, 0.0001),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.3,      1.),
    'vel'   : (0.99,     1.01),
    }

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

    #if not model.check_constraints():
    #    ipdb.set_trace()

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
        #model_temp.update_parameters(ph) # update model parameters so we can check constraints
        stress_temp = getStrength(edot, strain, temp, dict(zip(theta0_names,ph)), models, constants)
        cst = stress_temp > 0.0
        #cst = model_temp.check_constraints() # check constraints

    params = dict(zip(theta0_names,ph)) # make into a dict

    phi[i] = ph
    stress[i] = getStrength(edot, strain, temp, params, models, constants)
    if stress[i] < 0:
        ipdb.set_trace()


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



    plt.figure(1, figsize=(15, 15))
    for i in range(nparams):
        for j in range(nparams):
            if i == j:
                plt.subplot2grid((nparams, nparams), (i, j))

                #for k in range(nexp):
                #    sns.distplot(phii[:, i, k], hist=False, kde=True, color='lightgreen')

                #sns.distplot(phi0arr[:, i], hist=False, kde=True, color='blue')

                sns.distplot(phi[:, i], hist=False, kde=True, color='darkblue')

                plt.xlim(bounds[i, 0], bounds[i, 1])
                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel(theta0_names[i])
                ax.tick_params(axis='x', which='major', labelsize=8)
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            if i < j:
                plt.subplot2grid((nparams, nparams), (i, j))

                #for k in range(nexp):
                #    oo = contx(phii[:, j, k], phii[:, i, k])
                #    plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'], colors='lightgreen')

                #oo = contx(phi0arr[:, j], phi0arr[:, i])
                #plt.contour(oo['X'], oo['Y'], oo['Z'], oo['conts'], colors='blue')

                oo = contx(phi[:, j], phi[:, i])
                plt.contour(oo['X'], oo['Y'], oo['Z'], colors='darkblue')


                plt.xlim(bounds[j, 0], bounds[j, 1])
                plt.ylim(bounds[i, 0], bounds[i, 1])
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                print(i)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.subplot2grid((nparams, nparams), (2, 0))
    from matplotlib.lines import Line2D

    #colors = ['lightgreen', 'blue', 'grey']
    #lines = [Line2D([0], [0], color=c, linewidth=2) for c in colors]
    #labels = [r'$\theta_i$', r'$\theta_0$', r'$\theta^*$']
    #plt.legend(lines, labels)
    plt.axis('off')

    plt.savefig(out_path + name + '_postThetas.png', bbox_inches='tight',dpi=300)