#########################################
#########################################
### Cu Pooled Calibration ###
#########################################
#########################################


import matplotlib
from impala import superCal as sc
import matplotlib.pyplot as plt
import impala.superCal.post_process as pp
import numpy as np
import pandas as pd
import sqlite3 as sq
import os
from scipy.optimize import minimize
np.seterr(under='ignore')
from sklearn.metrics import r2_score


path_to_dir = '/Users/lbeesley/Desktop/tame_impala/'
path_to_data = '/Users/lbeesley/Desktop/tame_impala/Cu/data/Cu-annealed/'
path_to_results = '/Users/lbeesley/Desktop/tame_impala/Cu/pooled_flyer/results/'
path_to_plots = '/Users/lbeesley/Desktop/tame_impala/Cu/pooled_flyer/plots/'

 


with open(path_to_dir+'Helper_Functions/MCMC_Diagnostics.py') as fd:
    exec(fd.read())    
       
with open(path_to_dir+'Helper_Functions/Posterior_Mode.py') as fd:
    exec(fd.read())
    
         
    
################
### Comments ###
################

### This script implements pooled Impala calibration for Ta, including 
### - subsampling to roughly 100 per experiment
### - isothermal heat transfer model

####################
### Read in Data ###
####################

### New New Data from JeeYeon: 
data_CuOFE1 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__77K_0.001s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE2 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__295K_0.1s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE3 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__295K_0.001s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE4 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__298K_2000s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE5 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__473K_0.1s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE6 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__473K_2000s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE7 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__673K_0.1s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE8 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__673K_2000s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE9 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__873K_0.1s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]
data_CuOFE10 = pd.read_csv(path_to_data + 'Cu_OFHC__annealed_600C_1hr__873K_2000s_1_clean.txt', sep= ",", engine = 'python').to_numpy()[:,0:2]



### Samanta Data ###
data_Samanta1 = pd.read_csv(path_to_data + 'Samanta/Samanta_0.066_1173_MPa', sep= ",", engine = 'python').to_numpy()
data_Samanta2 = pd.read_csv(path_to_data + 'Samanta/Samanta_960_1173_MPa', sep= ",", engine = 'python').to_numpy()
data_Samanta3 = pd.read_csv(path_to_data + 'Samanta/Samanta_1800_1023MPa', sep= ",", engine = 'python').to_numpy()
data_Samanta4 = pd.read_csv(path_to_data + 'Samanta/Samanta_2300_873_MPa', sep= ",", engine = 'python').to_numpy()


### Nemat-Nasser Data ###
data_NN1 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_0.1_296_MPa', sep= ",", engine = 'python').to_numpy()
data_NN1 = data_NN1[1:,:]
data_NN2 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_4000_77_MPa', sep= ",", engine = 'python').to_numpy()
data_NN3 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_4000_496_MPa', sep= ",", engine = 'python').to_numpy()
data_NN4 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_4000_696', sep= ",", engine = 'python').to_numpy()
data_NN5 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_4000_896', sep= ",", engine = 'python').to_numpy()
data_NN6 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_4000_1096_MPa', sep= ",", engine = 'python').to_numpy()
data_NN7 = pd.read_csv(path_to_data + 'Nemat-Nasser/Nemat-Nasser_8000_296_MPa', sep= ",", engine = 'python').to_numpy()


# Adjusting for units of mBar (case specific)
data_CuOFE1[:,1] = data_CuOFE1[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_CuOFE2[:,1] = data_CuOFE2[:,1]/1E5
data_CuOFE3[:,1] = data_CuOFE3[:,1]/1E5
data_CuOFE4[:,1] = data_CuOFE4[:,1]/1E5
data_CuOFE5[:,1] = data_CuOFE5[:,1]/1E5
data_CuOFE6[:,1] = data_CuOFE6[:,1]/1E5
data_CuOFE7[:,1] = data_CuOFE7[:,1]/1E5
data_CuOFE8[:,1] = data_CuOFE8[:,1]/1E5
data_CuOFE9[:,1] = data_CuOFE9[:,1]/1E5
data_CuOFE10[:,1] = data_CuOFE10[:,1]/1E5


data_Samanta1[:,1] = data_Samanta1[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_Samanta2[:,1] = data_Samanta2[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_Samanta3[:,1] = data_Samanta3[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_Samanta4[:,1] = data_Samanta4[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN1[:,1] = data_NN1[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN2[:,1] = data_NN2[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN3[:,1] = data_NN3[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN4[:,1] = data_NN4[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN5[:,1] = data_NN5[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN6[:,1] = data_NN6[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 
data_NN7[:,1] = data_NN7[:,1]/1E5 #Check with JeeYeon whether this scaling is right. I think it is, but not positive. 


# strain rates for the experiments# units: 1/s
edot_CuOFE1       = 0.001
edot_CuOFE2       = 0.1
edot_CuOFE3       = 0.001
edot_CuOFE4       = 2000
edot_CuOFE5       = 0.1
edot_CuOFE6       = 2000
edot_CuOFE7       = 0.1
edot_CuOFE8       = 2000
edot_CuOFE9       = 0.1
edot_CuOFE10       = 2000


edot_Samanta1 = 0.066
edot_Samanta2 = 960
edot_Samanta3 = 1800
edot_Samanta4 = 2300

edot_NN1 = 0.1
edot_NN2 = 4000
edot_NN3 = 4000
edot_NN4 = 4000
edot_NN5 = 4000
edot_NN6 = 4000
edot_NN7 = 8000




# Annealing temperatures (in Kelvin) for each experiment
# Max value of the temperature linear model would be 900 Kelvin
temp_CuOFE1       = 77
temp_CuOFE2       = 295
temp_CuOFE3       = 295
temp_CuOFE4       = 298
temp_CuOFE5      = 473
temp_CuOFE6      = 473
temp_CuOFE7       = 673
temp_CuOFE8       = 673
temp_CuOFE9       = 873
temp_CuOFE10       = 873


temp_Samanta1 = 1173
temp_Samanta2 = 1173
temp_Samanta3 = 1023
temp_Samanta4 = 873

temp_NN1 = 296
temp_NN2 = 77
temp_NN3 = 496
temp_NN4 = 696
temp_NN5 = 896
temp_NN6 = 1096
temp_NN7 = 296

dat_all = [data_CuOFE1, data_CuOFE2, data_CuOFE3, data_CuOFE4, data_CuOFE5, data_CuOFE6, data_CuOFE7, data_CuOFE8, data_CuOFE9, data_CuOFE10,
           data_Samanta1, data_Samanta2, data_Samanta3, data_Samanta4,
           data_NN1, data_NN2, data_NN3, data_NN4, data_NN5, data_NN6, data_NN7]
temps = [temp_CuOFE1, temp_CuOFE2, temp_CuOFE3, temp_CuOFE4, temp_CuOFE5, temp_CuOFE6, temp_CuOFE7, temp_CuOFE8, temp_CuOFE9, temp_CuOFE10,
         temp_Samanta1, temp_Samanta2, temp_Samanta3, temp_Samanta4,
         temp_NN1, temp_NN2, temp_NN3, temp_NN4, temp_NN5, temp_NN6, temp_NN7]
edots = [edot_CuOFE1, edot_CuOFE2, edot_CuOFE3, edot_CuOFE4, edot_CuOFE5, edot_CuOFE6, edot_CuOFE7, edot_CuOFE8, edot_CuOFE9, edot_CuOFE10,
         edot_Samanta1, edot_Samanta2, edot_Samanta3, edot_Samanta4,
         edot_NN1, edot_NN2, edot_NN3, edot_NN4, edot_NN5, edot_NN6, edot_NN7]

nexp = len(dat_all) # number of experiments



stress_stacked = np.hstack([np.array(v)[:,1] for v in dat_all])
strain_hist_list = [np.array(v)[:,0] for v in dat_all]

### Implement Subsampling 
#[len(strain_hist_list[j]) for j in range(len(dat_all))]
NSAMP = 50
inds_subsampled = [np.linspace(1,len(strain_hist_list[j]),len(strain_hist_list[j])) % np.floor(len(strain_hist_list[j])/NSAMP) == 0 if len(strain_hist_list[j]) > NSAMP else np.repeat(True,len(strain_hist_list[j])) for j in range(len(dat_all))]
#[len(strain_hist_list[j][inds_subsampled[j]]) for j in range(len(dat_all))]


#import cm
n = len(dat_all)
from_list = matplotlib.colors.LinearSegmentedColormap.from_list
cm = from_list('Set15', plt.cm.Set1(range(0,n)), n)
plt.cm.register_cmap(None, cm)
plt.set_cmap(cm)


cmap = plt.get_cmap('jet', 14)
cmap.set_under('gray')

plt.close('all')

fig,ax=plt.subplots(1,2,figsize=(12,5), sharey = False)
fig.suptitle('Figure 1: Observed Stress-Strain Curves \n and Experimental Strain Rates/Temperatures')
for j in range(len(dat_all)):
    ax[0].plot(dat_all[j][:,0], dat_all[j][:,1], color = cmap(j))
    ax[1].scatter(edots[j], temps[j], color = cmap(j))

ax[0].set_ylabel('Stress')
ax[0].set_xlabel('Strain')
#ax[0].set_title('Figure 1: Observed Stress-Strain Curves')
ax[1].set_xlabel('Strain Rate (1/s)')
ax[1].set_ylabel('Temperature (K)')
#ax[1].set_title('Figure 2: Experimental Strain Rates and Temperatures')
plt.show()

# for i in range(len(dat_all)):
#     print(i)
#     print(dat_all[i][dat_all[i].shape[0]-1,:])




#######################
### Read in TC Data ###
#######################
data_CuTC = pd.read_csv(path_to_data + '../taylor_cylinder/allen-ofhc-cu-taylor.txt', sep= "  ", engine = 'python', comment='#').to_numpy()
data_CuTC = data_CuTC[:,0:2]

# plt.plot(data_CuTC[:,0],data_CuTC[:,1])
# plt.show()

### code from Skye to extract stress/strain information from TC experiment
from math import *
from scipy.optimize import fmin
from numpy import interp

#initial cylinder radius and length.  only length is actually used
r0=3.935     #mm #updated for cu
l0=39.35    #mm #updated for cu
vel0=214*100.0              #m/s converted to cm/s #updated for cu
rho=8.592                     #density in g/cc #updated for cu
rhovsq=rho*(vel0*1.0e-6)**2   #convert cm/s to cm/us, gives pressure Mbar unit
lfdata=27.19                  #final length of deformed cylinder in mm #updated for cu



def epsEq12(x0, *args):
 """solves Eq 12 from Simple and Sophisticated Models of Taylor's Cylinder Impact"""
 rhovsq=args[0]
 sig=args[1]
 epsguess=x0[0]
 if epsguess < 0.0:  epsguess=0.0
 if epsguess > 0.99: epsguess=0.99
 lhs=rhovsq/2.0/sigma
 rhs=-log(1.0-epsguess)-epsguess
 badnow=abs(lhs-rhs)
 return badnow

#might need to adjust sigma bounds for a different material. 0.001 Mbar corresponds to 100 MPa
sigma=0.001
sigs=[]
lf=[]
hf=[]
xf=[]
eps0=[]
while sigma <= 0.015:
 x0=[0.5]
 xout=fmin(epsEq12, x0, xtol=1.0e-8, args=(rhovsq, sigma), disp=False)
 testeps=xout[0]
 x=l0*(1.0-testeps)
 h=-l0*(1.0-testeps)*log(1.0-testeps)
 eps0.append(testeps)
 hf.append(h)
 xf.append(x)
 ltot=l0*(1.0-testeps)*(1.0-log(1.0-testeps))
 sigout=1.0e5*sigma    #convert from Mbar to MPa
 sigs.append(sigout)
 lf.append(ltot)
 #if unsure of results, might be useful to plot length as a function of stress using following output
 #print(str(round(1.0e5*sigma,6))+"   "+str(round(ltot,6)))
 sigma+=0.0005

hawkstress=interp(lfdata, lf, sigs)
print("hawkyard stress estimate "+str(hawkstress)+" MPa")
hawkhf=interp(hawkstress, sigs, hf)
print("final deformed portion length "+str(hawkhf))
hawkxf=interp(hawkstress, sigs, xf)
print("final undeformed portion length "+str(hawkxf))
hawkeps0=interp(hawkstress, sigs, eps0)
print("maximum engineering strain at foot of cylinder "+str(hawkeps0))
print("maximum true strain at foot of cylinder "+str(-log(1.0-hawkeps0)))
#Equation 10 strain rate estimate
tdeformation=(2.0*(l0-lfdata))/(10.0*vel0)
avtruestrain=-log((hawkhf)/(l0-hawkxf))
print("average true strain "+str(avtruestrain))
print("average strain rate "+str(avtruestrain/tdeformation)+"/s")

hawkstress_mbar = hawkstress/1E5 #convert to mbar
hawkedot = avtruestrain/tdeformation
hawktemp = 298.15

# plt.close('all')
# fig,ax=plt.subplots(1,2,figsize=(12,5), sharey = False)
# fig.suptitle('Figure 1: Observed Stress-Strain Curves \n and Experimental Strain Rates/Temperatures')
# for j in range(len(dat_all)):
#     if j in range(10):
#         ax[0].plot(dat_all[j][:,0], dat_all[j][:,1], c = 'green')
#     else:
#         ax[0].plot(dat_all[j][:,0], dat_all[j][:,1], c = 'blue')

# ax[0].scatter(avtruestrain, hawkstress_mbar, c = 'red')
# ax[0].set_ylabel('Stress')
# ax[0].set_xlabel('Strain')
# #ax[0].set_title('Figure 1: Observed Stress-Strain Curves')
# ax[1].scatter(edots[0:10], temps[0:10], c = 'green')
# ax[1].scatter(edots[10:], temps[10:], c ='blue')
# ax[1].scatter(hawkedot, hawktemp, c = 'red')
# ax[1].set_xlabel('Strain Rate (1/s)')
# ax[1].set_ylabel('Temperature (K)')
# #ax[1].set_title('Figure 2: Experimental Strain Rates and Temperatures')
# plt.show()


dat_all.append(np.asarray([avtruestrain, hawkstress_mbar]).astype('float').reshape(1,-1))
temps.append(hawktemp)
edots.append(hawkedot)
nexp = len(dat_all) # number of experiments

stress_stacked = np.hstack([np.array(v)[:,1] for v in dat_all])
strain_hist_list = [np.array(v)[:,0] for v in dat_all]

### Implement Subsampling 
#[len(strain_hist_list[j]) for j in range(len(dat_all))]
NSAMP = 50
inds_subsampled = [np.linspace(1,len(strain_hist_list[j]),len(strain_hist_list[j])) % np.floor(len(strain_hist_list[j])/NSAMP) == 0 if len(strain_hist_list[j]) > NSAMP else np.repeat(True,len(strain_hist_list[j])) for j in range(len(dat_all))]
#[len(strain_hist_list[j][inds_subsampled[j]]) for j in range(len(dat_all))]



##########################################
### Read in Flyer Simulations and Data ###
##########################################
import os
import re

FILES = np.asarray(os.listdir(path_to_data + '/../cu_flyer_training/'))
FILES = FILES[np.where(np.asarray(FILES) != '.DS_Store')[0]]
temp = np.hstack([re.sub("cu_", "", FILES[i]) for i in range(len(FILES))])
temp = temp.astype('int32')
FILES_REORDER = [ FILES[np.argsort(temp)[i]] for i in range(len(temp))]

flyer_sims = []
for i in range(len(FILES_REORDER)):
    temp = re.sub("cu_", "", FILES_REORDER[i])
    results = pd.read_csv(path_to_data + '../cu_flyer_training/' + FILES_REORDER[i] + '/flyer_cu_ses_ptw_'+ str(temp) +'.mat.ult', sep= "\s+", engine = 'python', skiprows = 2, header = None)
    results = pd.DataFrame(np.asarray(results), columns = ['Time','velocity'])
    results['velocity'] = 10000*results['velocity']
    results[results<0] = 0
    results['run_num'] = temp
    if i == 0:
        flyer_sims = [results]
    else:
        flyer_sims.append(results)


params = pd.read_csv(path_to_data + '/../flyer_lhs.csv', sep = ",", engine = 'python')

flyer_sims_long = pd.concat(flyer_sims)

flyer_obs = pd.read_csv(path_to_data + '/../Cu-Flyer/Cu-Cu_Thomas_PTW_PlateImpact/' + 'ofhc-cu-symmetric-impact.txt', sep= "\s+", engine = 'python', skiprows = 5, header = None)
flyer_obs = pd.DataFrame(np.asarray(flyer_obs), columns = ['Time','velocity'])
flyer_obs_raw = flyer_obs
flyer_obs = flyer_obs[flyer_obs['Time'] > 0.83]

from scipy.interpolate import interp1d


# ### Identify choppy ones
# probs = np.asarray([np.abs(np.diff(flyer_sims[i]['velocity'])).max() for i in range(len(flyer_sims))]) > 6.67
# probs2 = np.asarray([np.quantile(np.abs(np.diff(flyer_sims[i]['velocity'][(flyer_sims[i]['Time'] < 0.8)&(flyer_sims[i]['Time'] < 0.82) ])), 0.99) for i in range(len(flyer_sims))]) > 6.4




# params.iloc[probs].min(axis=0)
# params.iloc[probs].max(axis=0)


# params.min(axis=0)
# params.max(axis=0)

# import copy
# params_wide = params.copy()
# params_wide['probs'] = probs.astype('int')

# i=67
# plt.plot(flyer_sims[i]['Time'], flyer_sims[i]['velocity'])
# plt.show()
 
# import seaborn as sns
# sns.pairplot(params_wide, hue='probs', plot_kws={'alpha': 0.5})
# plt.show()
 

# plt.close('all')
# for i in np.arange(0,1000,1)[probs]:
#     #inds = np.where(flyer_sims['run_num'] == runs[i])[0]
#     plt.plot(flyer_sims[i]['Time'], flyer_sims[i]['velocity'], zorder = 0)#, c = 'gray', zorder = 0)
# plt.scatter(flyer_obs_raw['Time'], flyer_obs_raw['velocity'])
# plt.xlim(0.7,1.4)
# plt.xlabel('Time')
# plt.ylabel('Velocity (m/s)')
# plt.savefig(path_to_plots + 'temp.png')
# plt.show()




time_grid = np.arange(0.5,1.4,(1.4-0.7)/500)
flyer_sims_interp_wide = np.zeros([len(flyer_sims),len(time_grid)])
flyer_sims_interp = []
for i in range(len(flyer_sims)):
    results = pd.DataFrame(np.append(time_grid.reshape(-1,1),np.exp(interp1d(flyer_sims[i]['Time'], np.log(flyer_sims[i]['velocity']+1e-9), kind="linear", fill_value='extrapolate')(time_grid)).reshape(-1,1), axis = 1), 
                           columns = ['Time', 'velocity'])
    results[results<0] = 0
    flyer_sims_interp_wide[i,:] = np.asarray(results['velocity'])
    if i == 0:
        flyer_sims_interp = [results]
    else:
        flyer_sims_interp.append(results)
  
        
flyer_obs_interp = pd.DataFrame(np.append(time_grid.reshape(-1,1),np.exp(interp1d(flyer_obs['Time'], np.log(flyer_obs['velocity']+1e-9), kind="linear", fill_value='extrapolate')(time_grid)).reshape(-1,1), axis = 1), 
                           columns = ['Time', 'velocity'])
flyer_obs_interp[flyer_obs_interp<0]=0





# flyer_diff_interp_wide = np.hstack([np.diff(flyer_sims_interp_wide[i,:]) for i in range(flyer_sims_interp_wide.shape[0])])


plt.close('all')
# runs = np.unique(flyer_sims_long['run_num'])
# runs = runs[np.arange(0,1000,100)]
for i in np.arange(0,1000,1):
    #inds = np.where(flyer_sims['run_num'] == runs[i])[0]
    plt.plot(flyer_sims[i]['Time'], flyer_sims[i]['velocity'], zorder = 0)#, c = 'gray', zorder = 0)
plt.scatter(flyer_obs_raw['Time'], flyer_obs_raw['velocity'])
plt.xlim(0.7,1.4)
plt.xlabel('Time')
plt.ylabel('Velocity (m/s)')
plt.savefig(path_to_plots + 'flyer_sims_raw.png')
plt.show()



# plt.close('all')
# # runs = np.unique(flyer_sims_long['run_num'])
# # runs = runs[np.arange(0,1000,100)]
# for i in np.arange(0,1000,1):
#     #inds = np.where(flyer_sims['run_num'] == runs[i])[0]
#     plt.plot(flyer_sims_interp[i]['Time'], flyer_sims_interp[i]['velocity'], zorder = 0)#, c = 'gray', zorder = 0)
# plt.scatter(flyer_obs_interp['Time'], flyer_obs_interp['velocity'])
# plt.xlim(0.5,1.4)
# plt.xlabel('Time')
# plt.ylabel('Velocity (m/s)')
# plt.savefig(path_to_plots + 'flyer_sims.png')
# plt.show()






###############################
### Warp Functions to Obs ###
###############################
import fdasrsf as fs

out = fs.fdawarp(flyer_sims_interp_wide.T, time_grid)
out.multiple_align_functions(np.asarray(flyer_obs_interp['velocity']), parallel=True, lam=.01)
gam_sim_list = out.gam
vv_sim_list = fs.geometry.gam_to_v(out.gam)
ftilde_sim_list = out.fn
out2 = fs.pairwise_align_functions(np.asarray(flyer_obs_interp['velocity']), np.asarray(flyer_obs_interp['velocity']), time_grid, lam=.01)
gam_obs_list = out2[1]
vv_obs_list = fs.geometry.gam_to_v(out2[1])
ftilde_obs_list = out2[0]
ftilde_obs_all = np.hstack(ftilde_obs_list)
vv_obs_all = np.hstack(vv_obs_list)

figure, axis = plt.subplots(1, 4)
xx = np.arange(len(time_grid))/(len(time_grid)-1)
axis[0].plot(time_grid, flyer_sims_interp_wide.T, color='lightgrey')
axis[0].plot(time_grid, np.asarray(flyer_obs_interp['velocity']))
axis[1].plot(xx, ftilde_sim_list, color='lightgrey')
axis[1].plot(xx, ftilde_obs_list)
axis[2].plot(xx, gam_sim_list*(np.max(time_grid)-np.min(time_grid))+np.min(time_grid), color='lightgrey')
axis[2].plot(xx, gam_obs_list*(np.max(time_grid)-np.min(time_grid))+np.min(time_grid))
axis[3].plot(vv_sim_list, color='lightgrey')
axis[3].plot(vv_obs_list)
figure.set_size_inches(15,4)
axis[0].title.set_text('Misaligned')
axis[1].title.set_text('Aligned')
axis[2].title.set_text('Warping functions')
axis[3].title.set_text('Shooting vectors')
plt.tight_layout()
plt.savefig(path_to_plots + 'foo.png', dpi=300)




plt.close('all')
# runs = np.unique(flyer_sims_long['run_num'])
# runs = runs[np.arange(0,1000,100)]
for i in np.arange(0,1000,1):
    #inds = np.where(flyer_sims['run_num'] == runs[i])[0]
    plt.plot(time_grid, ftilde_train_list[:,i], zorder = 0)#, c = 'gray', zorder = 0)
plt.scatter(time_grid, ftilde_obs_all)
plt.xlim(0.7,1.4)
plt.xlabel('Time')
plt.ylabel('Velocity (m/s)')
plt.savefig(path_to_plots + 'flyer_sims_warp.png')
plt.show()


####################
### Fit Emulator ###
####################


input_names = ['theta', 'p','s0','sInf','kappa','lgamma','y0','yInf','y1','y2']

n = params.shape[0]
M = len(time_grid)
p = len(input_names)

# ho = np.random.choice(n, 5, replace=False)
# sim_inputs = np.asarray(params[input_names])
# Xtrain = np.delete(sim_inputs, ho, axis=0)
# Xtest = sim_inputs[ho]
ftilde_train_list = ftilde_sim_list
# ftilde_test_list = ftilde_sim_list[:,ho]
# vv_train_list = np.delete(vv_sim_list, ho, axis=1)
# vv_test_list = vv_sim_list[:,ho]


flyer_sims_long_emu = []
for i in range(ftilde_train_list.shape[1]):
    results = pd.DataFrame(np.append(np.append(time_grid.reshape(-1,1),ftilde_train_list[:,i].reshape(-1,1), axis = 1), np.repeat(np.asarray(params)[i,:].reshape(1,-1),len(time_grid), axis = 0),axis=1), 
                           columns = ['Time', 'velocity']+input_names)
    if i == 0:
        flyer_sims_long_emu = [results]
    else:
        flyer_sims_long_emu.append(results)
    
flyer_sims_long_emu = pd.concat(flyer_sims_long_emu)


from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


np.random.seed(0)

# This is the emulator; just a neural net.
nn_emulator = Pipeline(
    [
        # First scale input; this is crucial.
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor((128, 64, 32, 16, 8, 4, 2), batch_size=128, activation="tanh"),
        ),
    ]
)

emu_X = flyer_sims_long_emu[["Time"]+ input_names]
emu_y = flyer_sims_long_emu['velocity']
# Fit emulator.
nn_emulator.fit(emu_X.values, emu_y.values)

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
pickle.dump(nn_emulator, open(path_to_results + "nn_emulator.pkl", 'wb'))


PREDS = nn_emulator.predict(emu_X.values)
plt.close('all')
plt.scatter(emu_y.values, PREDS, s=1, alpha = 0.05)
#plt.savefig(path_to_plots + 'ysim_emu_train.png')
plt.show()


# weird = np.where(np.abs(emu_y.values - PREDS) > 15)


# emu_X_weird = emu_X.iloc[weird[0]]

# import pickle
# def save_object(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# save_object(emu_ysim_list, path_to_results + 'emu_ysim_list_shift.pkl')


# emu_ysim_list.plot()



# emu_ysim_list.plot()


# preds = emu_ysim_list.predict(Xtrain)
# plt.scatter(ysim_train_list.flatten(), preds.mean(axis=0).flatten(), s=1, alpha = 0.1)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# #plt.savefig(path_to_plots + 'ysim_emu_train.png')
# plt.show()


# preds_valid = emu_ysim_list.predict(Xtest)
# plt.scatter(ysim_test_list.flatten(), preds_valid.mean(axis=0).flatten(), s=1, alpha = 0.1)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# # plt.savefig(path_to_plots + 'ysim_emu_test.png')
# plt.show()



#####################
### Get Constants ###
#####################

# set constants and parameter bounds
# constants fixed for PTW calibration
# set constants and parameter bounds
consts_ptw = {                 #UNITS                  Name 
  'matomic': 63.546,          # [u]                   # Atomic Mass
  'rho0'   : 8.98,            # [g/cm^3]              # Density
  'beta'   : 0.25,            # PTW parameter. The Overdriven Shock Exponent
  
  'chi'    : 1.0,             # []                    # Taylor-Quinney Coeffecient (Ratio of plastic work converted to heat)
  'r0'     : 9.09  ,          # Y-intercept for a Linear density model
  'r1'     : -5.02E-4 ,         # Slope for a Linear density model

# specific heat for Be is about 1.88 J/(g K) at room temperature.
# In sesame 3337, it is 1.74 J/(g K) at 300 K.

  'c0'     : 3.48E-6,         
  'c1'     : 1.23E-9,         
  'c2'     : -6.77E-13,        
  
  'Tmelt0'  : 1833.77,
  'tm0'    : -4579,           # Y-intercept for Temp melt linear model
  'tm1'    : 743.6,           # slope for Temp melt linear model
  
  'alpha'  : 0.28,            # Thermal Softening of Shear Modulus

  'rho_0'  : 9.02,            # [g/cm^3]              # Density  according to bgp model
  'G0'     : 0.521,           # [MBar]                 # Shear Modulus
  'gamma_1': 1.27,            # 
  'gamma_2': 2.82,            # 
  'q2'     : 0.7,             # 
  'parent' : True
   }


# bounds on PTW input parameters to calibrate
# bounds_ptw = {
#   'theta' : (3E-4,      1.5),
#   'p'     : (3E-4,      12.),
#   's0'    : (8.4E-3,    9.1E-2),
#   'sInf'  : (1.1E-4,    4.5E-3),
#   'kappa' : (1E-4,      75E-2),
#   'lgamma': (-5,         -2),
#   'y0'    : (1E-4,      8.3E-3),
#   'yInf'  : (3E-5,      1E-4),
#   'y1'    : (9.3E-2,    1.5E-1),
#   'y2'    : (2.9E-1,    4.5),
#   'vel'   : (params['vel'].min(),  params['vel'].max()),
#    }

# bounds on PTW input parameters to calibrate
bounds_ptw = {    
    'theta' : (0.001,   0.2),    
    'p'     : (0,   5.),   
    's0'    : (0.0001,   0.1),   
    'sInf'  : (0.0001,   0.1),    
    'kappa' : (0.01,   0.5),    
    'lgamma': (-18.,     -1.),    
    'y0'    : (0.00001,   0.1),    
    'yInf'  : (0.00001,   0.1),    
    'y1'    : (0.001,    0.5), 
    'y2'    : (0.281,      5.),  #y2 has to be at least beta  
 }

def constraints_ptw(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good



########################
### Define Model Run ###
########################

import impala


class BGP_PW_Shear_Modulus(impala.physics.BaseModel):
    #BPG model provides cold shear, i.e. shear modulus at zero temperature as a function of density.
    #PW describes the (lienar) temperature dependence of the shear modulus. (Same dependency as
    #in Simple_Shear_modulus.)
    #With these two models combined, we get the shear modulus as a function of density and temperature.
    consts = ['G0', 'rho_0', 'gamma_1', 'gamma_2', 'q2', 'alpha']
    def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        cold_shear  = mp.G0*np.exp(6.*mp.gamma_1*(np.power(mp.rho_0,-1./3.)-np.power(rho,-1./3.))+ 2*mp.gamma_2/mp.q2*(np.power(mp.rho_0,-mp.q2)-np.power(rho,-mp.q2)))
        gnow = cold_shear*(1.- mp.alpha* (temp/tmelt))
        gnow[temp > tmelt] = (cold_shear*(1.- mp.alpha))[temp > tmelt]
        gnow[np.where(gnow < 0)] = 0.
        #if temp >= tmelt: gnow = 0.0
        #if gnow < 0.0:    gnow = 0.0
        return gnow
         
impala.physics.BGP_PW_Shear_Modulus = BGP_PW_Shear_Modulus

model_pool_ptw = sc.ModelMaterialStrength(temps=np.array(temps), 
        edots=np.array(edots)*1e-6, 
        consts=consts_ptw, 
        strain_histories=[strain_hist_list[j][inds_subsampled[j]].astype('float') for j in range(len(dat_all))],  #Note: implementing subsampling!!!!
        flow_stress_model='PTW_Yield_Stress',
        melt_model= 'Linear_Melt_Temperature',
        shear_model= 'BGP_PW_Shear_Modulus',
        specific_heat_model='Quadratic_Specific_Heat', 
        density_model='Linear_Density',
        pool=True, s2='gibbs')

# import pickle
# with open('/Users/lbeesley/Desktop/tame_impala/Cu/pooled_flyer/results/' + 'emu_ysim_list.pkl', 'rb') as f: 
#     emu_ysim_list = pickle.load(f)

# input_names = ['theta', 'p','s0','sInf','kappa','lgamma','y0','yInf','y1','y2']

# model_pool_emu = sc.ModelBassPca_func(emu_ysim_list, input_names = input_names, s2='MH')



import pickle
with open('/Users/lbeesley/Desktop/tame_impala/Cu/pooled_flyer/results/' + 'nn_emulator.pkl', 'rb') as f: 
    nn_emulator = pickle.load(f)

input_names = ['theta', 'p','s0','sInf','kappa','lgamma','y0','yInf','y1','y2']

sub_inds0 = np.arange(0,len(time_grid),10)
sub_inds = sub_inds0[(time_grid[sub_inds0] > 0.8) * (time_grid[sub_inds0] < 1.1) ]
print(len(sub_inds))
time_grid2 = time_grid[sub_inds]

# plt.plot(time_grid, np.asarray(ftilde_obs_list).flatten())
# plt.plot(time_grid, np.asarray(flyer_obs_interp)[:,1].flatten())
# plt.scatter(time_grid2, np.asarray(ftilde_obs_list[sub_inds]).flatten(), c = 'red')
# plt.show()

def NN_emu_pooled(parmat):
    parmat_array = np.vstack(parmat).T
    res = [nn_emulator.predict(np.append(np.repeat(time_grid2[i], parmat_array.shape[0]).reshape(-1,1),parmat_array,axis=1)) for i in range(len(time_grid2))]
    res = np.hstack(res)
    return res

model_pool_emu = sc.ModelF_bigdata(NN_emu_pooled, input_names = input_names)



#####################
### Set up Models ###
#####################

s2_ind = np.hstack([[j]*len(np.array(dat_all[j])[inds_subsampled[j],1]) for j in range(len(dat_all))])
setup_pool_ptw = sc.CalibSetup(bounds_ptw, constraints_ptw)
setup_pool_ptw.addVecExperiments(yobs=np.hstack([np.array(dat_all[j])[inds_subsampled[j],1] for j in range(len(dat_all))]).astype('float'), #Note: implementing subsampling!!!!
        model=model_pool_ptw, 
        sd_est=np.array([0.0001]*len(dat_all)),  
        s2_df=np.array([15]*len(dat_all)), 
        s2_ind=s2_ind,  #Note: implementing subsampling!!!!
        theta_ind=s2_ind)  #Note: implementing subsampling!!!!
setup_pool_ptw.addVecExperiments(yobs=np.asarray(flyer_obs_interp)[:,1].flatten()[sub_inds], #Note: implementing subsampling!!!!
        model=model_pool_emu, 
        sd_est=np.array([50]),  
        s2_df=np.array([15]), 
        s2_ind=[0]*len(sub_inds),  #Note: implementing subsampling!!!!
        theta_ind=[0]*len(sub_inds))  #Note: implementing subsampling!!!!
setup_pool_ptw.setTemperatureLadder(1.02**np.arange(50), start_temper=2000) 
setup_pool_ptw.setMCMC(nmcmc=25000, decor=100)#, start_tau_theta=-4.)


###########################
### Perform Calibration ### (Takes several hours to run)
###########################

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



np.seterr(divide = 'ignore')  #FloatingPointError: divide by zero encountered in log
np.seterr(under = 'ignore')  #FloatingPointError: divide by zero encountered in log

### Run Calibration 
out_pool = sc.calibPool(setup_pool_ptw)

### Save Parameter Distribution (theta) Draws
df = pd.DataFrame(out_pool.theta[:,0,:], columns = setup_pool_ptw.bounds.keys())
df.to_csv(path_to_results + 'theta_draws.csv',index=False)

### Save Tempering Diagnostic
total_temperature_swaps(out_pool,setup_pool_ptw)
plt.savefig(path_to_plots + 'tempering.png')



###################################
### Read in Calibration Results ###
###################################

df = pd.read_csv(path_to_results + 'theta_draws.csv')

###################
### Perform MAP ###
###################

# uu = np.arange(10000, 40000, 10)
# from scipy.stats import qmc
# l_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,0]
# u_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,1]
# parent_scaled = qmc.scale(np.array(df)[uu,:], l_bounds, u_bounds)

# def distConstraints(x,bounds):
#     const_dist = int(x['sInf'] >= x['s0']) * ((x['sInf'] - x['s0'])**2)
#     const_dist += int(x['yInf'] >= x['y0']) * ((x['yInf'] - x['y0'])**2)
#     const_dist += int(x['y0'] >= x['s0']) * ((x['y0'] - x['s0'])**2)
#     const_dist += int(x['yInf'] >= x['sInf']) * ((x['yInf'] - x['sInf'])**2)
#     const_dist += int(x['s0'] >= x['y1']) * ((x['s0'] - x['y1'])**2)
    
#     for k in list(bounds.keys()):
#         const_dist += int(x[k] >= bounds[k][1]) * ((x[k] - bounds[k][1])**2)
#         const_dist += int(x[k] <= bounds[k][0]) * ((x[k] - bounds[k][0])**2)
#     return const_dist
# setup_pool_ptw.distConstraints = distConstraints

# POST_pool = eval_partialintlogposterior_impalapool(setup_pool_ptw, n_samples = 100, theta = parent_scaled)
# MAP_pool = get_map_impalapool(setup_pool_ptw, n_samples = 100, theta_init = parent_scaled[np.argmax(POST_pool),:], niter = 100)
# save_object(MAP_pool, path_to_results + 'map.pkl')

# with open(path_to_results + 'map.pkl', 'rb') as f: 
#    MAP_pool = pickle.load(f)



#########################
### Visualize Outputs ###
#########################

uu = np.arange(10000, 25000, 10)
n_exp0 = len(np.unique(setup_pool_ptw.s2_ind[0]))
n_exp1 = len(np.unique(setup_pool_ptw.s2_ind[1]))
s2_inds = setup_pool_ptw.s2_ind[0]

### Trace Plot
plt.close('all')
pp.parameter_trace_plot(np.asarray(df),ylim=[0,1])
plt.savefig(path_to_plots + 'trace.png')


# ### KDE Pairs
# pairs_kde(df.loc[uu])
# plt.savefig(path_to_plots + 'pairs.png')

# pp.pairs(setup_pool_ptw, mat_st = np.asarray(df.loc[uu]))
# plt.savefig(path_to_plots + 'pairs2.png')


### Prediction Plots 
THETA_Y = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df)[uu,:],setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T)['outcome_draws']
QUANTS_THETA_Y = [np.quantile(THETA_Y[j],[0.025,0.5,0.975],axis=0) for j in range(len(THETA_Y))]
for exp_ind in range(n_exp0):
    fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
    ax.fill_between(setup_pool_ptw.models[0].meas_strain_histories[exp_ind], QUANTS_THETA_Y[0][0,np.where(s2_inds == exp_ind)].flatten(), QUANTS_THETA_Y[0][2,np.where(s2_inds == exp_ind)].flatten(), color = 'pink', zorder = 1)            
    ax.plot(setup_pool_ptw.models[0].meas_strain_histories[exp_ind],QUANTS_THETA_Y[0][1,np.where(s2_inds == exp_ind)].flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'Pooled Prediction')
    ax.scatter(setup_pool_ptw.models[0].meas_strain_histories[exp_ind],setup_pool_ptw.ys[0][np.where(s2_inds == exp_ind)], color = 'black', zorder = 3, label = 'Data')
    ax.set_ylabel('Stress')
    ax.set_xlabel('Strain')
    ax.title.set_text('Prediction for Experiment '+str(exp_ind))
    ax.legend()
    plt.savefig(path_to_plots + 'experiment_' +str(exp_ind)+'.png')

matplotlib.pyplot.close()


fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
ax.fill_between(time_grid[sub_inds], QUANTS_THETA_Y[1][0,:].flatten(), QUANTS_THETA_Y[0][2,:].flatten(), color = 'pink', zorder = 1)            
ax.plot(time_grid[sub_inds],QUANTS_THETA_Y[1][1,:].flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'Pooled Prediction')
ax.scatter(time_grid[sub_inds],setup_pool_ptw.ys[1], color = 'black', zorder = 3, label = 'Data')
plt.xlabel('Velocity')
plt.ylabel('Time')
ax.title.set_text('Prediction for Flyer Experiment')
ax.legend()
plt.savefig(path_to_plots + 'experiment_flyer.png')



################################
### Output "Best" Parameters ###
################################

from scipy.stats import qmc
l_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,0]
u_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,1]
parent_scaled = qmc.scale(np.array(df)[uu,:], l_bounds, u_bounds)

BEST = dict()
SSE = dict()
MAPE = dict()

### Parent_Median
pred_median = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.median(parent_scaled,axis = 0).reshape(1,-1))['outcome_draws']
median_sse = sum([((pred_median[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp0)]) + sum([((pred_median[1].flatten()-setup_pool_ptw.ys[1].flatten())**2).mean() for i in range(1)])
median_mape = 100*np.mean([(np.abs(pred_median[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/(setup_pool_ptw.ys[0][np.where(s2_inds == i)]+0.00000001)).mean(axis=1) for i in range(n_exp0)])
BEST['parent_median'] = np.median(parent_scaled,axis = 0)
SSE['parent_median'] = median_sse
MAPE['parent_median'] = median_mape

### Best Draw 
parent_sse = sum([((THETA_Y[0][:,np.where(s2_inds == i)[0]]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp0)]) + sum([((THETA_Y[1]-setup_pool_ptw.ys[1])**2).mean(axis=1) for i in range(1)]) 
pred_minsse = np.hstack([THETA_Y[0][np.where(parent_sse == parent_sse.min())[0][0],np.where(s2_inds == i)[0]]for i in range(n_exp0)]).reshape(1,-1)
pred_minsse = np.append(pred_minsse,np.hstack([THETA_Y[1][np.where(parent_sse == parent_sse.min())[0][0],:]for i in range(1)]).reshape(1,-1), axis = 1 )
A = [(np.abs(pred_minsse[0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/(setup_pool_ptw.ys[0][np.where(s2_inds == i)]+0.00000001)).mean(axis=1) for i in range(n_exp0)]
B = [(np.abs(pred_minsse[0,np.arange(len(s2_inds),len(pred_minsse.flatten()),1)]-setup_pool_ptw.ys[1])/setup_pool_ptw.ys[1]).mean() for i in range(1)]
A.append(np.asarray(B))
parent_mape = 100*np.mean(A)
BEST['parent_minsse'] = parent_scaled[np.where(parent_sse == parent_sse.min())[0],:].flatten()
SSE['parent_minsse'] = np.array([parent_sse.min()])
MAPE['parent_minsse'] = parent_mape

# ### MAP
# pred_map = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.array(pd.DataFrame(MAP_pool.values())).reshape(1,-1))['outcome_draws']
# map_sse = sum([((pred_map[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp)])
# map_mape = 100*np.mean([(np.abs(pred_map[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(nexp)])
# BEST['map'] = np.array(pd.DataFrame(MAP_pool.values())).flatten()
# SSE['map'] = map_sse
# MAPE['map'] = map_mape

BEST_df = pd.DataFrame(BEST.values(), columns = np.array(pd.DataFrame(setup_pool_ptw.bounds.keys())).flatten())
BEST_df['method'] = np.array(pd.DataFrame(BEST.keys())).flatten()
BEST_df['sse'] = np.array(pd.DataFrame(SSE.values())).flatten()
BEST_df['mape'] = np.array(pd.DataFrame(MAPE.values())).flatten()
BEST_df.to_csv(path_to_results + 'best.csv',index=False)


### Visualize Best-Fitting Values
fig,ax=plt.subplots(1,1,figsize=(16,6), sharey = False)   
OBS = setup_pool_ptw.ys[0]
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_median).flatten()[np.arange(0,len(s2_inds),1)], color = 'purple', zorder = 2, linewidth = 2, label = 'Posterior Median')
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_minsse).flatten()[np.arange(0,len(s2_inds),1)], color = 'red', zorder = 2, linewidth = 2, label = 'Best SSE')
# ax.plot(np.arange(0,len(OBS)),np.hstack(pred_map).flatten(), color = 'green', zorder = 3, linewidth = 2, label = 'MAP')
ax.scatter(np.arange(0,len(OBS)),OBS, color = 'black', zorder = 4, label = 'Data')
ax.set_ylabel('Stress')
ax.set_xlabel('Observation Index')
ax.title.set_text('Prediction for All Experiments')
ax.legend()
plt.savefig(path_to_plots + 'best_all.png')


### Visualize Best-Fitting Values
fig,ax=plt.subplots(1,1,figsize=(16,6), sharey = False)   
OBS = setup_pool_ptw.ys[1]
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_median).flatten()[np.arange(len(s2_inds),len(pred_minsse.flatten()),1)], color = 'purple', zorder = 2, linewidth = 2, label = 'Posterior Median')
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_minsse).flatten()[np.arange(len(s2_inds),len(pred_minsse.flatten()),1)], color = 'red', zorder = 2, linewidth = 2, label = 'Best SSE')
# ax.plot(np.arange(0,len(OBS)),np.hstack(pred_map).flatten(), color = 'green', zorder = 3, linewidth = 2, label = 'MAP')
ax.scatter(np.arange(0,len(OBS)),OBS, color = 'black', zorder = 4, label = 'Data')
ax.set_ylabel('Stress')
ax.set_xlabel('Observation Index')
ax.title.set_text('Prediction for All Experiments')
ax.legend()
plt.savefig(path_to_plots + 'best_all_flyer.png')



###################################
### Summarize Prediction Errors ###
###################################

# ### Get Error and Coverage Rates
# THETA_COVERAGE = [np.quantile(THETA_Y[0][:,np.where(s2_inds == j)],[0.025,0.975],axis=0) for j in range(n_exp)]
# THETA_COVERAGE_BIN = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp)]
# for i in range(len(THETA_COVERAGE)): 
#     THETA_COVERAGE_BIN[i] = (setup_pool_ptw.ys[0][np.where(s2_inds == i)]<= THETA_COVERAGE[i][1,:]).astype(int) + (setup_pool_ptw.ys[0][np.where(s2_inds == i)]>= THETA_COVERAGE[i][0,:]).astype(int)
#     THETA_COVERAGE_BIN[i][THETA_COVERAGE_BIN[i] == 1] = 0
#     THETA_COVERAGE_BIN[i][THETA_COVERAGE_BIN[i] == 2] = 1

# THETA_COVERAGE_AGG = []
# for j in range(len(THETA_COVERAGE)):
#     THETA_COVERAGE_AGG = np.append(THETA_COVERAGE_AGG, THETA_COVERAGE_BIN[j]) 

# THETA_ERRORS = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp)]
# for i in range(n_exp): 
#     THETA_ERRORS[i] = np.abs(setup_pool_ptw.ys[0][np.where(s2_inds == i)] -np.median(THETA_Y[0][:,np.where(s2_inds == i)],axis=0))/setup_pool_ptw.ys[0][np.where(s2_inds == i)]

# THETA_ERRORS_AGG = [np.median(THETA_ERRORS[i]) for i in range(n_exp)]
# THETA_ERRORS_AGG=np.vstack(THETA_ERRORS_AGG)


# MEDIAN_ERRORS = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp)]
# for i in range(n_exp): 
#     MEDIAN_ERRORS[i] = np.abs(setup_pool_ptw.ys[0][np.where(s2_inds == i)] -np.hstack(pred_median).flatten()[np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]

# MEDIAN_ERRORS_AGG = [np.median(MEDIAN_ERRORS[i]) for i in range(n_exp)]
# MEDIAN_ERRORS_AGG=np.vstack(MEDIAN_ERRORS_AGG)


# MINSSE_ERRORS = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp)]
# for i in range(n_exp): 
#     MINSSE_ERRORS[i] = np.abs(setup_pool_ptw.ys[0][np.where(s2_inds == i)] -np.hstack(pred_minsse).flatten()[np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]

# MINSSE_ERRORS_AGG = [np.median(MINSSE_ERRORS[i]) for i in range(n_exp)]
# MINSSE_ERRORS_AGG=np.vstack(MINSSE_ERRORS_AGG)

# MAP_ERRORS = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp)]
# for i in range(n_exp): 
#     MAP_ERRORS[i] = np.abs(setup_pool_ptw.ys[0][np.where(s2_inds == i)] -np.hstack(pred_map).flatten()[np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]
# MAP_ERRORS_AGG = [np.median(MAP_ERRORS[i]) for i in range(n_exp)]
# MAP_ERRORS_AGG=np.vstack(MAP_ERRORS_AGG)


# fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
# ax.scatter(np.linspace(0,nexp-1,nexp),100*THETA_ERRORS_AGG, label = 'Entire Distribution (Overall = ' + str(round(100*THETA_ERRORS_AGG.mean(),1)) + '%)')
# ax.scatter(np.linspace(0,nexp-1,nexp),100*MEDIAN_ERRORS_AGG, label = 'Posterior Median (Overall = ' + str(round(100*MEDIAN_ERRORS_AGG.mean(),1)) + '%)', color = 'red')
# ax.scatter(np.linspace(0,nexp-1,nexp),100*MINSSE_ERRORS_AGG, label = 'Best SSE (Overall = ' + str(round(100*MINSSE_ERRORS_AGG.mean(),1)) + '%)', color = 'purple')
# #ax.scatter(np.linspace(0,nexp-1,nexp),100*MAP_ERRORS_AGG, label = 'MAP (Overall = ' + #str(round(100*MAP_ERRORS_AGG.mean(),1)) + '%)', color = 'blue')
# ax.set_ylabel('Percent Error')
# ax.title.set_text('Mean Absolute Percent Error (Coverage = ' + str(round(THETA_COVERAGE_AGG.mean(),3)) + ')')
# ax.set_xlabel('Experiment')
# ax.legend()
# plt.savefig(path_to_plots + 'error.png')





#############################################
### Compare to existing parameterizations ###
#############################################
#theta, p, s0, sinf, kappa, loggamma, y0, yinf, y1, y2, 
#https://www.osti.gov/servlets/purl/1237423


ptw = [0.025, 2.0, 0.0085, 0.00055, 0.11, np.log(0.00001), 0.0001, 0.00009, 0.094, 0.575]  #made yInf < y0
#andrews = [3.952e-2, 5.456e-1, 1.066e-2, 3.910e-4, 8.580e-2, np.log(1.007e-5), 8.262e-5, 3.948e-5, 7.811e-2, 4.569e-1] 


### Best Draw 
proposed = [0.020344643704575414,	3.2206804019598385	,0.012137366246955468	,0.001743369,	0.33325130012872756,	-14.75212483	,0.000176498	,0.000154564,	0.4944635910851172,	1.0357233285565326]

n_exp0 = len(np.unique(setup_pool_ptw.s2_ind[0]))
n_exp1 = len(np.unique(setup_pool_ptw.s2_ind[1]))
s2_inds = setup_pool_ptw.s2_ind[0]

import copy
extended_bounds = np.copy(setup_pool_ptw.bounds_mat)
extended_bounds[:,0] = -100
extended_bounds[:,1] = 100
def get_errors(param):
    theta_input = np.asarray(param).reshape(1,-1)
    THETA_Y = model_pool_ptw.eval(sc.tran_unif(sc.normalize(theta_input,extended_bounds),extended_bounds, setup_pool_ptw.bounds.keys()))
    THETA_ERRORS = [np.empty([len(np.where(s2_inds == i)[0])]) for i in range(n_exp0)]
    for i in range(n_exp0): 
        THETA_ERRORS[i] = np.abs(setup_pool_ptw.ys[0][np.where(s2_inds == i)] -THETA_Y[:,np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]
    THETA_ERRORS_AGG = [np.mean(THETA_ERRORS[i]) for i in range(n_exp0)]
    THETA_ERRORS_AGG=np.vstack(THETA_ERRORS_AGG)
    
    THETA_Y_flyer = [model_pool_emu.eval(sc.tran_unif(sc.normalize(theta_input,extended_bounds),extended_bounds, setup_pool_ptw.bounds.keys())) for j in range(1)]
    THETA_ERRORS_AGG = np.append(THETA_ERRORS_AGG, np.vstack([(np.abs(setup_pool_ptw.ys[j+1] -THETA_Y_flyer[j])/setup_pool_ptw.ys[j+1]).mean() for j in range(1)]).flatten())
    return 100*THETA_ERRORS_AGG

errors_sse =get_errors(np.asarray(proposed).flatten())

errors_ptw = get_errors(np.asarray(ptw).flatten())
# errors_andrews = get_errors(np.asarray(andrews).flatten())

16.60016036
17.05888923





cmap = plt.get_cmap('jet', 7)
cmap.set_under('gray')

inds = np.arange(0,len(edots),1)
reorder = np.argsort(errors_sse[inds].flatten())
x = np.arange(0,len(inds),1)
plt.close('all')
plt.plot(x+1,errors_sse[inds].flatten()[reorder], label = 'Proposed', color = 'purple')
plt.scatter(x+1,errors_sse[inds].flatten()[reorder], color = 'purple')
# plt.plot(x,errors_sse_high[inds], label = 'Proposed (High Strain Rate)', color = cmap(3))
# plt.scatter(x,errors_sse_high[inds], color = cmap(3))
plt.plot(x+1,errors_ptw[inds].flatten()[reorder], label = 'PTW Paper', color = 'red')
plt.scatter(x+1,errors_ptw[inds].flatten()[reorder], color = 'red')
# plt.plot(x+1,errors_andrews[inds].flatten()[reorder], label = 'Andrews and Wilson', color = cmap(5))
# plt.scatter(x+1,errors_andrews[inds].flatten()[reorder], color = cmap(5))
plt.xlabel('Experiment Number')
plt.ylabel('Stress Percent Error')
plt.xticks(x+1)
plt.legend()
plt.savefig(path_to_plots + 'error_shpb.png')
plt.show()






cmap = plt.get_cmap('jet', 7)
cmap.set_under('gray')

inds = np.arange(0,len(edots),1)
x = np.arange(0,len(inds),1)
reorder = x
plt.close('all')
plt.plot(x+1,errors_sse[inds].flatten()[reorder], label = 'Proposed', color = 'purple')
plt.scatter(x+1,errors_sse[inds].flatten()[reorder], color = 'purple')
# plt.plot(x,errors_sse_high[inds], label = 'Proposed (High Strain Rate)', color = cmap(3))
# plt.scatter(x,errors_sse_high[inds], color = cmap(3))
plt.plot(x+1,errors_ptw[inds].flatten()[reorder], label = 'PTW Paper', color = 'red')
plt.scatter(x+1,errors_ptw[inds].flatten()[reorder], color = 'red')
# plt.plot(x+1,errors_andrews[inds].flatten()[reorder], label = 'Andrews and Wilson', color = cmap(5))
# plt.scatter(x+1,errors_andrews[inds].flatten()[reorder], color = cmap(5))
plt.xlabel('Experiment Number')
plt.ylabel('Stress Percent Error')
plt.xticks(x+1)
plt.legend()
plt.savefig(path_to_plots + 'error_shpb_notreordered.png')
plt.show()




def get_preds(param, j):
    theta_input = np.asarray(param).reshape(1,-1)
    if j == 0:
        THETA_Y = model_pool_ptw.eval(sc.tran_unif(sc.normalize(theta_input,extended_bounds),extended_bounds, setup_pool_ptw.bounds.keys()))
    else:
        THETA_Y = model_pool_emu.eval(sc.tran_unif(sc.normalize(theta_input,extended_bounds),extended_bounds, setup_pool_ptw.bounds.keys()))
    return THETA_Y

preds_sse =[get_preds(np.asarray(proposed).flatten(),j) for j in range(2)]
preds_ptw =[get_preds(np.asarray(ptw).flatten(),j) for j in range(2)]



path_to_clust_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer_cluster/results/'
df_dirich = pd.read_csv(path_to_clust_results + 'parentdirich_draws.csv')
Results_PTW = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_dirich), setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T)['outcome_draws']
QUANTS_PTW = [np.quantile(Results_PTW[j], [0.025,0.5,0.975],axis=0) for j in range(len(Results_PTW))]

cmap = plt.get_cmap('jet', 7)
cmap.set_under('gray')

plt.close('all')

### Prediction Plots 
for exp_ind in range(n_exp0):
    fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
    ax.fill_between(setup_pool_ptw.models[0].meas_strain_histories[exp_ind], QUANTS_PTW[0][0,np.where(s2_inds == exp_ind)].flatten(), QUANTS_PTW[0][2,np.where(s2_inds == exp_ind)].flatten(), color = 'gray', zorder = 3, label = 'Clustered 95% Intervals', alpha = 0.1)    
    ax.plot(setup_pool_ptw.models[0].meas_strain_histories[exp_ind],preds_sse[0][0,np.where(s2_inds == exp_ind)].flatten(), color = 'purple', zorder = 2, linewidth = 2, label = 'Proposed', alpha = 0.9)
    ax.plot(setup_pool_ptw.models[0].meas_strain_histories[exp_ind],preds_ptw[0][0,np.where(s2_inds == exp_ind)].flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'PTW Paper', alpha = 0.9)
    ax.scatter(setup_pool_ptw.models[0].meas_strain_histories[exp_ind],setup_pool_ptw.ys[0][np.where(s2_inds == exp_ind)], color = 'black', zorder = 0, label = 'Data')
    ax.set_ylabel('Stress')
    ax.set_xlabel('Strain')
    ax.title.set_text( ' Strain Rate = ' + str(edots[exp_ind]) + '/s   Temperature = ' + str(temps[exp_ind]) + 'K')
    ax.legend()
    plt.savefig(path_to_plots + 'compare_' +str(exp_ind)+'.png')
matplotlib.pyplot.close()




### Visualize Best-Fitting Values
fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)   
OBS = setup_pool_ptw.ys[1]
ax.plot(np.arange(0,len(OBS)),np.hstack(preds_sse[1]).flatten(), color = 'purple', zorder = 2, linewidth = 2, label = 'Proposed')
ax.plot(np.arange(0,len(OBS)),np.hstack(preds_ptw[1]).flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'PTW Paper')
# ax.plot(np.arange(0,len(OBS)),np.hstack(pred_map).flatten(), color = 'green', zorder = 3, linewidth = 2, label = 'MAP')
ax.scatter(np.arange(0,len(OBS)),OBS, color = 'black', zorder = 4, label = 'Data')
ax.set_ylabel('Velocity')
ax.set_xlabel('Time')
ax.title.set_text('Prediction for All Experiments')
ax.legend()
plt.savefig(path_to_plots + 'best_all_flyer.png')











# errors_sse.mean()
# errors_sjue.mean()

# x = pd.DataFrame(np.asarray(ptw).reshape(1,-1), columns = bounds_ptw.keys())
# bounds = bounds_ptw
# def constraints_ptw(x, bounds):
#     good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
#     for k in list(bounds.keys()):
#         good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
#     return good
# constraints_ptw(x,bounds)

# (x['sInf'] < x['s0']) 
# (x['yInf'] < x['y0']) 
# (x['y0'] < x['s0']) 
# (x['yInf'] < x['sInf']) 
# (x['s0'] < x['y1'])
# for k in list(bounds.keys()):
#        print((x[k] < bounds[k][1]) * (x[k] > bounds[k][0]))




################################
### Compare Predicted Stress ###
################################

STRAINS = 0.05+np.arange(0,0.5,0.05)
LOG10EDOTS = np.arange(-4,6,0.5)
TEMPRATIOS = np.arange(0.05,0.5,0.025)


KEYS = ['theta', 'p', 's0', 'sInf', 'kappa', 'lgamma', 'y0', 'yInf', 'y1','y2']
sample_scaled = np.array([(x,y) for x in LOG10EDOTS for y in TEMPRATIOS])
model_info = setup_pool_ptw.models[0].model_info
model_temp = sc.ModelMaterialStrength(temps=np.array(sample_scaled[:,1])*consts_ptw['Tmelt0'], 
        edots=(10**np.array(sample_scaled[:,0]))*1e-6, 
        consts=consts_ptw, 
        strain_histories=[STRAINS for j in range(sample_scaled.shape[0])],  #Note: implementing subsampling!!!!
        flow_stress_model= model_info[0],
        melt_model= model_info[3],
        shear_model= model_info[1],
        specific_heat_model= model_info[2], 
        density_model= model_info[4],
        pool=True, s2='gibbs')
s2_ind = np.hstack([[j]*len(STRAINS) for j in range(sample_scaled.shape[0])])
setup_temp = sc.CalibSetup(bounds_ptw, constraints_ptw)
setup_temp.addVecExperiments(yobs=np.hstack([np.repeat(0.1, len(STRAINS)) for j in range(sample_scaled.shape[0])]), #Note: implementing subsampling!!!!
        model=model_temp, 
        sd_est=np.array([setup_pool_ptw.sd_est[0][0]]*sample_scaled.shape[0]),  
        s2_df=np.array([setup_pool_ptw.s2_df[0][0]]*sample_scaled.shape[0]), 
        s2_ind=s2_ind,  #Note: implementing subsampling!!!!
        theta_ind=s2_ind)  #Note: implementing subsampling!!!!


Results_proposed = get_outcome_predictions_impala(setup_temp, theta_input = np.asarray(proposed).flatten().reshape(1,-1))['outcome_draws'][0]
Results_ptw = get_outcome_predictions_impala(setup_temp, theta_input = np.asarray(ptw).flatten().reshape(1,-1))['outcome_draws'][0]

v_min = np.min([np.min(Results_proposed), np.min(Results_proposed), np.min(Results_ptw)])
v_max = np.max([np.max(Results_proposed), np.max(Results_proposed), np.max(Results_ptw)])


plt.close('all')
Results_Agg = pd.DataFrame(np.append(np.append((Results_proposed).reshape(-1,1), np.repeat(sample_scaled,len(STRAINS),axis=0), axis = 1), np.repeat(STRAINS.reshape(-1,1),sample_scaled.shape[0] ,axis = 1).T.flatten().reshape(-1,1), axis = 1), columns = ['value','log10edot','temp_prop','strain'])
TO_PLOT = Results_Agg[Results_Agg['strain']==0.1].reset_index(drop = True).drop('strain',axis=1).pivot(index='log10edot', columns='temp_prop', values='value')
ax = sns.heatmap(TO_PLOT, linewidths =0, xticklabels= TO_PLOT.columns.values.round(3), yticklabels= TO_PLOT.index.values.round(3), vmin = v_min, vmax  = v_max)
ax.set_xlabel('Proportion of Melt Temp. (1071K)')
ax.set_ylabel('log10(Strain Rate)')
ax.set_title('Proposed (overall) parameterization')
ax.scatter(len(np.unique(sample_scaled[:,1]))*((np.array(temps)/consts_ptw['Tmelt0']) - sample_scaled[:,1].min())/(sample_scaled[:,1].max()-sample_scaled[:,1].min()), len(np.unique(sample_scaled[:,0]))*(np.array(np.log10(edots)) - sample_scaled[:,0].min())/(sample_scaled[:,0].max()-sample_scaled[:,0].min()), marker='*', s=100, color='yellow') 
plt.tight_layout()
plt.savefig(path_to_plots + 'stress_overall.png')
#plt.show()


plt.close('all')
Results_Agg = pd.DataFrame(np.append(np.append((Results_ptw).reshape(-1,1), np.repeat(sample_scaled,len(STRAINS),axis=0), axis = 1), np.repeat(STRAINS.reshape(-1,1),sample_scaled.shape[0] ,axis = 1).T.flatten().reshape(-1,1), axis = 1), columns = ['value','log10edot','temp_prop','strain'])
TO_PLOT = Results_Agg[Results_Agg['strain']==0.1].reset_index(drop = True).drop('strain',axis=1).pivot(index='log10edot', columns='temp_prop', values='value')
ax = sns.heatmap(TO_PLOT, linewidths =0, xticklabels= TO_PLOT.columns.values.round(3), yticklabels= TO_PLOT.index.values.round(3), vmin = v_min, vmax  = v_max)
ax.set_xlabel('Proportion of Melt Temp. (1071K)')
ax.set_ylabel('log10(Strain Rate)')
ax.set_title('Proposed (high strain rate) parameterization')
ax.scatter(len(np.unique(sample_scaled[:,1]))*((np.array(temps)/consts_ptw['Tmelt0']) - sample_scaled[:,1].min())/(sample_scaled[:,1].max()-sample_scaled[:,1].min()), len(np.unique(sample_scaled[:,0]))*(np.array(np.log10(edots)) - sample_scaled[:,0].min())/(sample_scaled[:,0].max()-sample_scaled[:,0].min()), marker='*', s=100, color='yellow') 
plt.tight_layout()
plt.savefig(path_to_plots + 'stress_ptw.png')
#plt.show()




path_to_clust_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer_cluster/results/'

df_dirich = pd.read_csv(path_to_clust_results + 'parentdirich_draws.csv')

Results_PTW = get_outcome_predictions_impala(setup_temp, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_dirich), setup_temp.bounds_mat, setup_temp.bounds.keys()).values())).T)['outcome_draws'][0]
QUANTS_PTW = np.quantile(Results_PTW, [0.025,0.5,0.975],axis=0)
WIDTH = QUANTS_PTW[2,:]-QUANTS_PTW[0,:]
MEDIAN = QUANTS_PTW[1,:]
SEs = np.std(Results_PTW, axis = 0)

plt.close('all')
Results_Agg = pd.DataFrame(np.append(np.append((SEs/MEDIAN).reshape(-1,1), np.repeat(sample_scaled,len(STRAINS),axis=0), axis = 1), np.repeat(STRAINS.reshape(-1,1),sample_scaled.shape[0] ,axis = 1).T.flatten().reshape(-1,1), axis = 1), columns = ['value','log10edot','temp_prop','strain'])
TO_PLOT = Results_Agg[Results_Agg['strain']==0.1].reset_index(drop = True).drop('strain',axis=1).pivot(index='log10edot', columns='temp_prop', values='value')
ax = sns.heatmap(100*TO_PLOT, linewidths =0, xticklabels= TO_PLOT.columns.values.round(3), yticklabels= TO_PLOT.index.values.round(3), annot=True)
ax.set_xlabel('Proportion of Melt Temp. (1833.77K)')
ax.set_ylabel('log10(Strain Rate)')
ax.scatter(len(np.unique(sample_scaled[:,1]))*((np.array(temps)/consts_ptw['Tmelt0']) - sample_scaled[:,1].min())/(sample_scaled[:,1].max()-sample_scaled[:,1].min()), len(np.unique(sample_scaled[:,0]))*(np.array(np.log10(edots)) - sample_scaled[:,0].min())/(sample_scaled[:,0].max()-sample_scaled[:,0].min()), marker='*', s=100, color='yellow')
plt.tight_layout()
plt.savefig(path_to_plots + 'stress_unc.png')
plt.show()





path_to_clust_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer/results/'

df_hier = pd.read_csv(path_to_clust_results + 'parent_draws.csv')

Results_PTW = get_outcome_predictions_impala(setup_temp, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_hier)[np.arange(30000, 40000, 1),:], setup_temp.bounds_mat, setup_temp.bounds.keys()).values())).T)['outcome_draws'][0]
QUANTS_PTW = np.quantile(Results_PTW, [0.025,0.5,0.975],axis=0)
WIDTH = QUANTS_PTW[2,:]-QUANTS_PTW[0,:]
MEDIAN = QUANTS_PTW[1,:]
SEs = np.std(Results_PTW, axis = 0)

plt.close('all')
Results_Agg = pd.DataFrame(np.append(np.append((SEs/MEDIAN).reshape(-1,1), np.repeat(sample_scaled,len(STRAINS),axis=0), axis = 1), np.repeat(STRAINS.reshape(-1,1),sample_scaled.shape[0] ,axis = 1).T.flatten().reshape(-1,1), axis = 1), columns = ['value','log10edot','temp_prop','strain'])
TO_PLOT = Results_Agg[Results_Agg['strain']==0.1].reset_index(drop = True).drop('strain',axis=1).pivot(index='log10edot', columns='temp_prop', values='value')
ax = sns.heatmap(100*TO_PLOT, linewidths =0, xticklabels= TO_PLOT.columns.values.round(3), yticklabels= TO_PLOT.index.values.round(3), annot=True)
ax.set_xlabel('Proportion of Melt Temp. (1833.77K)')
ax.set_ylabel('log10(Strain Rate)')
ax.scatter(len(np.unique(sample_scaled[:,1]))*((np.array(temps)/consts_ptw['Tmelt0']) - sample_scaled[:,1].min())/(sample_scaled[:,1].max()-sample_scaled[:,1].min()), len(np.unique(sample_scaled[:,0]))*(np.array(np.log10(edots)) - sample_scaled[:,0].min())/(sample_scaled[:,0].max()-sample_scaled[:,0].min()), marker='*', s=100, color='yellow')
plt.tight_layout()
plt.savefig(path_to_plots + 'stress_unc_hier.png')
plt.show()




###################
### Pairs Plots ###
###################


path_to_clust_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer_cluster/results/'
df_dirich = pd.read_csv(path_to_clust_results + 'parent_draws.csv')
l_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,0]
u_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,1]
df_dirich_scaled = qmc.scale(np.array(df_dirich), l_bounds, u_bounds)



path_to_hier_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer/results/'
df_hier = pd.read_csv(path_to_hier_results + 'parent_draws.csv')
l_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,0]
u_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,1]
df_hier_scaled = qmc.scale(np.array(df_hier)[np.arange(30000, 40000, 1),:], l_bounds, u_bounds)






input_names = ['theta', 'p', 's0', 'sInf', 'kappa', 'lgamma', 'y0', 'yInf', 'y1', 'y2']
A = pd.DataFrame(df_dirich_scaled, columns =input_names)
A['type'] = 'Clustered'

D = pd.DataFrame(df_hier_scaled, columns =input_names)
D['type'] = 'Hierarchical'
to_plot = pd.concat([A,D])



g = sns.PairGrid(to_plot, hue = 'type', corner = True, diag_sharey=False)
g.map_lower(sns.scatterplot, alpha = 0.05, size=None, sizes=(5, 5), legend="full")
#g.map_lower(sns.kdeplot, fill = True, alpha = 0.5)
g.map_diag(sns.histplot, common_norm = False, stat = 'density', kde = True, alpha = 0.5)
for ax in g.axes.flatten():
    if ax:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)  # Rotate labels by 45 degrees

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05) # Adjust values as needed     
g.add_legend()
g.fig.set_figheight(15)
g.fig.set_figwidth(15)
plt.savefig(path_to_plots + 'pairs_dirich_both2.png')





######################################
### Summarize all the calibrations ###
######################################

path_to_pooled_results = '/Users/lbeesley/Desktop/tame_impala/Cu/pooled_flyer/results/'
path_to_hier_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer/results/'
path_to_clust_results = '/Users/lbeesley/Desktop/tame_impala/Cu/hierarchical_flyer_cluster/results/'


alphas = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01])
probs = np.append(np.flip(alphas/2),1-(alphas/2))

df_pool = pd.read_csv(path_to_pooled_results + 'theta_draws.csv')
df_hier = pd.read_csv(path_to_hier_results + 'parent_draws.csv')
df_clust = pd.read_csv(path_to_clust_results + 'parentdirich_draws.csv')

# Get parent draws
dist_clust = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_clust),setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T
dist_hier = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_hier)[np.arange(30000, 40000, 1),:],setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T
dist_pool = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_pool)[np.arange(10000, 25000, 10),:],setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T
ptw = [
  'theta',
  'p',
  's0', 
  'sInf',
  'kappa', 
  'lgamma',
  'y0', 
  'yInf',
  'y1',
  'y2',
]


theta_i = []
for j in range(len(ptw)):
    mat = pd.DataFrame(dist_pool[:,j], columns = ['value'])
    mat['param'] = ptw[j]
    mat['Analysis'] = 'Pooled'
    if j==0:
        theta_i = [mat]
    else:
        theta_i.append(mat)
for j in range(len(ptw)):
    mat = pd.DataFrame(dist_hier[:,j], columns = ['value'])
    mat['param'] = ptw[j]
    mat['Analysis'] = 'Hierarchical'
    theta_i.append(mat)
for j in range(len(ptw)):
    mat = pd.DataFrame(dist_clust[:,j], columns = ['value'])
    mat['param'] = ptw[j]
    mat['Analysis'] = 'Clustered'
    theta_i.append(mat)    

theta_i_long = pd.concat(theta_i)


fig,ax=plt.subplots(2,5,figsize=(18,10))
fig.tight_layout(pad=3.0)
for i in range(2):
    for j in range(5):
        #ax[i,j].set_xlabel('Value')
        #ax[i,j].set_ylabel('Frequency')
        ax[i,j].set_title(ptw[5*i + j]) 
        A = sns.boxplot(data = theta_i_long.loc[theta_i_long['param'] == ptw[5*i+j]], x = 'Analysis', y = 'value', hue = 'Analysis',  ax=ax[i,j], showfliers=False, width = 1, dodge = False)
        A.set(xlabel='',ylabel='Value')
        plt.setp(ax[i,j].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        #ax[i,j].tick_params(axis='x', rotation=45, rotation_mode = 'anchor')#, ha="right", va = "center")
        if i == 0 or j != 2:
            ax[i,j].legend_.remove()
        ax[i,j].set_ylim(bounds_ptw[ptw[5*i+j]])
        if i == 0:
            ax[i,j].set_xticks([])
        #plt.xticks(rotation=45, ha='right', rotation_mode='anchor')         
plt.subplots_adjust(bottom=0.3)
plt.savefig(path_to_plots + 'densities_2levels.png')

#fig.suptitle('Cluster-Specific Posteriors from Clustered Analysis', fontsize=12)
