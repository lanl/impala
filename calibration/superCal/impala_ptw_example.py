##########################################################################################
## This script demonstrates how to do pooled impala calibration of PTW
## using SHPB/quasistatic data.
##########################################################################################

# %%
import numpy as np
import impala
import models
import matplotlib.pyplot as plt
np.seterr(under='ignore')

#%%
##########################################################################################
## Three Ti64 experiments, 1st column is plastic strain 2nd column is stress (units: mbar)
dat0 = np.array([[0.01     , 0.0100583],
       [0.02     , 0.010381 ],
       [0.03     , 0.0106112],
       [0.04     , 0.0108041],
       [0.05     , 0.0109619],
       [0.06     , 0.011082 ],
       [0.07     , 0.0112103],
       [0.08     , 0.0113348],
       [0.09     , 0.0114415],
       [0.1      , 0.0115094]])
dat1 = np.array([[0.01     , 0.004462 ],
       [0.02     , 0.0045834],
       [0.03     , 0.0046879],
       [0.04     , 0.0047613],
       [0.05     , 0.004824 ],
       [0.06     , 0.0048593],
       [0.07     , 0.0049031],
       [0.08     , 0.0049449],
       [0.09     , 0.0049828],
       [0.1      , 0.005015 ],
       [0.11     , 0.0050471],
       [0.12     , 0.0050752],
       [0.13     , 0.0050964],
       [0.14     , 0.0051326],
       [0.15     , 0.0051495],
       [0.16     , 0.0051803],
       [0.17     , 0.0051816],
       [0.18     , 0.0052018],
       [0.19     , 0.0052262],
       [0.2      , 0.0052345],
       [0.21     , 0.0052651],
       [0.22     , 0.0052795],
       [0.23     , 0.005296 ],
       [0.24     , 0.0053118],
       [0.25     , 0.0053274],
       [0.26     , 0.0053431],
       [0.27     , 0.00536  ],
       [0.28     , 0.0053702],
       [0.29     , 0.0053762],
       [0.3      , 0.0053926],
       [0.31     , 0.00541  ],
       [0.32     , 0.0054108]])
dat2 = np.array([[0.005682 , 0.0075202],
       [0.012309 , 0.0077424],
       [0.020317 , 0.0078811],
       [0.02867  , 0.0079921],
       [0.036673 , 0.0080891],
       [0.045022 , 0.0081583],
       [0.05337  , 0.0082275],
       [0.061718 , 0.0082967],
       [0.070064 , 0.008352 ],
       [0.078411 , 0.0084074],
       [0.087102 , 0.0084349],
       [0.095447 , 0.0084763],
       [0.103792 , 0.0085177],
       [0.112137 , 0.0085591],
       [0.120482 , 0.0086005],
       [0.129173 , 0.008628 ],
       [0.137517 , 0.0086555],
       [0.145862 , 0.0086969],
       [0.154206 , 0.0087244],
       [0.162897 , 0.0087519],
       [0.173673 , 0.0087794],
       [0.187578 , 0.0088206],
       [0.198008 , 0.0088619],
       [0.209478 , 0.0088754],
       [0.217127 , 0.008903 ],
       [0.225816 , 0.0089166],
       [0.234158 , 0.0089302],
       [0.243543 , 0.0089438],
       [0.250844 , 0.0089713],
       [0.259188 , 0.0089988],
       [0.26753  , 0.0090124],
       [0.276219 , 0.009026 ],
       [0.28491  , 0.0090535],
       [0.293252 , 0.0090671],
       [0.301594 , 0.0090807],
       [0.310284 , 0.0090943],
       [0.318973 , 0.0091079],
       [0.327316 , 0.0091216],
       [0.336005 , 0.0091352],
       [0.346434 , 0.0091626]])

# plot the three stress-strain curves
plt.plot(dat0[:,0], dat0[:,1])
plt.plot(dat1[:,0], dat1[:,1])
plt.plot(dat2[:,0], dat2[:,1])
plt.show()

# and here are the temperatures and strain rates for the three experiments:
temp0 = 573.0 # units: Kelvin
temp1 = 1373.0
temp2 = 973.0

edot0 = 800.0 # units: 1/s
edot1 = 2500.0
edot2 = 2500.0

# put the three experiments together in a list
dat_all = [dat0, dat1, dat2]
temps = [temp0, temp1, temp2]
edots = [edot0, edot1, edot2]
nexp = len(dat_all) # number of experiments

stress_stacked = np.hstack([np.array(v)[:,1] for v in dat_all])
strain_hist_list = [np.array(v)[:,0] for v in dat_all]

##########################################################################################
# set constants and parameter bounds
consts = {
    'alpha'  : 0.2,
    'beta'   : 0.33,
    'matomic': 45.9,
    'Tmelt0' : 2110.,
    'rho0'   : 4.419,
    'Cv0'    : 0.525e-5,
    'G0'     : 0.4,
    'chi'    : 1.0,
    'sgB'    : 6.44e-4
    }
bounds = {
    'theta' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'gamma' : (0.000001, 0.0001),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
    'y2'    : (0.33,      1.),
    }

##########################################################################################
# constraints: sInf < s0, yInf < y0, y0 < s0, yInf < sInf, s0 < y1, beta < y2 (but beta is fixed)
def constraints_ptw(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

##########################################################################################
# these define measurement error estimates, I would leave these as is for most SHPB/quasistatic data
sd_est = np.array([.001]*nexp)
s2_df = np.array([5]*nexp)
s2_ind = np.hstack([[v]*len(dat_all[v]) for v in list(range(nexp))])

##########################################################################################
# define PTW model
model = models.ModelPTW(temps=np.array(temps), edots=np.array(edots)*1e-6, consts=consts, strain_histories=strain_hist_list)

# bring everything together into calibration structure
setup = impala.CalibSetup(bounds, constraints_ptw)
setup.addVecExperiments(stress_stacked, model, sd_est, s2_df, s2_ind)
setup.setTemperatureLadder(1.1**np.arange(50))
setup.setMCMC(nmcmc=30000, nburn=10000, thin=1, decor=100)

##########################################################################################
# calibrate
out = impala.calibPool(setup)


##########################################################################################
# posterior predictions (without measurement error, which has standard deviation np.sqrt(out.s2)), disregarding the first 25000 MCMC samples
pred = setup.models[0].eval(impala.tran(out.theta[25000:30000,0,:], setup.bounds_mat, setup.bounds.keys()))

plt.plot(pred.T,color='grey')
plt.plot(stress_stacked,color='black')
plt.show()

# pairs plot of parameter posterior samples
import pandas as pd
import seaborn as sns
dat = pd.DataFrame(impala.tran(out.theta[25000:30000,0,:], setup.bounds_mat, setup.bounds.keys()))
g = sns.pairplot(dat, plot_kws={"s": [3]*5000}, corner=True, diag_kind='hist')
g
plt.show()



