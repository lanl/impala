#########################################
#########################################
### Beryllium Clustered Calibration ###
#########################################
#########################################

import impala
import matplotlib
from impala import superCal as sc
import matplotlib.pyplot as plt
import impala.superCal.post_process as pp
import numpy as np
#import dill
import pandas as pd
import sqlite3 as sq
import os
np.seterr(under='ignore')
#from sklearn.metrics import r2_score

# path_to_dir = '/Users/lbeesley/Desktop/tame_impala/'
# path_to_data = '/Users/lbeesley/Desktop/tame_impala/Be/data/QS_SHPB_Z_2017/'
# path_to_results = '/Users/lbeesley/Desktop/tame_impala/Be/hierarchical_zmachine_cluster_highedot/results/'
# path_to_plots = '/Users/lbeesley/Desktop/tame_impala/Be/hierarchical_zmachine_cluster_highedot/plots/'


 
path_to_dir = '/vast/home/lvandervort/impala_runs/'
path_to_data = '/vast/home/lvandervort/impala_runs/Be/data/QS_SHPB_Z_2017/'
path_to_results = '/vast/home/lvandervort/impala_runs/Be/hierarchical_zmachine_cluster_highedot/results/'
path_to_plots = '/vast/home/lvandervort/impala_runs/Be/hierarchical_zmachine_cluster_highedot/plots/'

with open(path_to_dir+'Helper_Functions/MCMC_Diagnostics.py') as fd:
    exec(fd.read())    

with open(path_to_dir+'Helper_Functions/Posterior_Mode.py') as fd:
    exec(fd.read())

# with open(path_to_dir+'Helper_Functions/Piecewise_Linear.py') as fd:
#     exec(fd.read())


impa = sc
with open(path_to_dir+'Helper_Functions/impala_clust.py') as fd:
    exec(fd.read())    
       
################
### Comments ###
################

### This script implements clustered Impala calibration for Be, including 
### - subsampling to produce about 100 points per experiment
### - strong shrinkage toward parent distribution mean
### - adiabatic heat transfer model


####################
### Read in Data ###

dat0 = np.loadtxt(path_to_data + "Bef200c1e-3_Mbar.txt")
dat1 = np.loadtxt(path_to_data + "Bef300c1_Mbar.txt")
dat2 = np.loadtxt(path_to_data + "Bef400c1_Mbar.txt")
dat3 = np.loadtxt(path_to_data + "Bef500c1_Mbar.txt")
dat4 = np.loadtxt(path_to_data + "Bef600c1_Mbar.txt")
dat5 = np.loadtxt(path_to_data + "Bed50433_Mbar.txt")
dat6 = np.loadtxt(path_to_data + "Bed30393_Mbar.txt")
dat7 = np.loadtxt(path_to_data + "Bed20373_Mbar.txt")
dat8 = np.loadtxt(path_to_data + "Bef0203500_Mbar.x.txt")
dat9 = np.loadtxt(path_to_data + "Bed-50353_Mbar.txt")


s200F_1 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_293K_1850s_clean_Mbar.txt")
s200F_2 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_295K_0.001s_clean_Mbar.txt")
s200F_3 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_295K_1s_clean_Mbar.txt")
s200F_4 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_473K_0.001s_clean_Mbar.txt")
s200F_5 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_473K_1s_clean_Mbar.txt")
s200F_6 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_473K_2000s_clean_Mbar.txt")
s200F_7 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_673K_0.001s_clean_Mbar.txt")
s200F_8 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_673K_1s_clean_Mbar.txt")
s200F_9 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_new_673K_3000s_clean_Mbar.txt")
s200F_10 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_293K_1900s_clean_Mbar.txt")
s200F_11 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_296K_0.001s_clean_Mbar.txt")
s200F_12 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_473K_0.001s_clean_Mbar.txt")
s200F_13 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_473K_2200s_clean_Mbar.txt")
s200F_14 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_673K_0.001s_clean_Mbar.txt")
s200F_15 = np.loadtxt(path_to_data + "../S200F_2018/Be_S200F_old_673K_2700s_clean_Mbar.txt")


z_1 = np.loadtxt(path_to_data + "../Z-machine/Be_Brown17GPa.txt")
z_2 = np.loadtxt(path_to_data + "../Z-machine/Be_Brown25GPa.txt")
z_3 = np.loadtxt(path_to_data + "../Z-machine/Be_Brown33GPa.txt")
z_4 = np.loadtxt(path_to_data + "../Z-machine/Be_Brown43GPa.txt")
z_5 = np.loadtxt(path_to_data + "../Z-machine/Be_Brown90GPa.txt")

z_1[:,1] = z_1[:,1] / 1e5 #convert to Mbar
z_2[:,1] = z_2[:,1] / 1e5 #convert to Mbar
z_3[:,1] = z_3[:,1] / 1e5 #convert to Mbar
z_4[:,1] = z_4[:,1] / 1e5 #convert to Mbar
z_5[:,1] = z_5[:,1] / 1e5 #convert to Mbar

# pressure ρ1 G0 Tm(ρ1) G
# (Mbar) (g/cm3) (Mbar) (K) (Mbar)
# 0.17 2.0829 1.877 2287 1.804
# 0.25 2.174 2.022 2431 1.948
# 0.33 2.259 2.161 2563 2.085
# 0.43 2.357 2.325 2710 2.248
# 0.90 2.745 3.009 3256 2.899


temp0 =   473.0
temp1 =   573.0  
temp2 =   673.0  
temp3 =   773.0  
temp4 =   873.0  
temp5 =   773.0  
temp6 =   573.0  
temp7 =   473.0  
temp8 =   298.0  
temp9 =   223.0  

temps200F_1 = 293
temps200F_2 = 295
temps200F_3 = 295
temps200F_4 = 473
temps200F_5 = 473
temps200F_6 = 473
temps200F_7 = 673
temps200F_8 = 673
temps200F_9 = 673
temps200F_10 = 293
temps200F_11 = 296
temps200F_12 = 473
temps200F_13 = 473
temps200F_14 = 673
temps200F_15 = 673

tempz_1 = 450
tempz_2 = 450
tempz_3 = 450
tempz_4 = 450
tempz_5 = 600



edot0 =  0.001
edot1 =  1.0
edot2 =  1.0
edot3 =  1.0
edot4 =  1.0
edot5 =  4300.0
edot6 =  3900.0
edot7 =  3700.0
edot8 =  3500.0
edot9 =  3500.0

edots200F_1 = 1850
edots200F_2 = 0.001
edots200F_3 = 1
edots200F_4 = 0.001
edots200F_5 = 1
edots200F_6 = 2000
edots200F_7 = 0.001
edots200F_8 = 1
edots200F_9 = 3000
edots200F_10 = 1900
edots200F_11 = 0.001
edots200F_12 = 0.001
edots200F_13 = 2200
edots200F_14 = 0.001
edots200F_15 = 2700

edotz_1 = 1.5 * 10**5
edotz_2 = 1.5 * 10**5
edotz_3 = 1.5 * 10**5
edotz_4 = 1.5 * 10**5
edotz_5 = 3 * 10**5

# dat_all = [dat0, dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9]
# temps = [temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9]
# edots = [edot0, edot1, edot2, edot3, edot4, edot5, edot6, edot7, edot8, edot9]




dat_all = [dat5, dat6, dat7, dat8, dat9,
           s200F_1, s200F_6, s200F_9, s200F_10, s200F_13, s200F_15]
temps = [temp5, temp6, temp7, temp8, temp9,
         temps200F_1, temps200F_6, temps200F_9, temps200F_10, temps200F_13, temps200F_15]
edots = [edot5, edot6, edot7, edot8, edot9,
         edots200F_1, edots200F_6, edots200F_9, edots200F_10, edots200F_13, edots200F_15]



#dat_all_z = [z_1, z_2, z_3, z_4, z_5]
NREP = 1
dat_all_z = [np.repeat(z_1[0,:].reshape(1,-1), NREP, axis = 0), 
             np.repeat(z_2[0,:].reshape(1,-1), NREP, axis = 0), 
             np.repeat(z_3[0,:].reshape(1,-1), NREP, axis = 0), 
             np.repeat(z_4[0,:].reshape(1,-1), NREP, axis = 0), 
             np.repeat(z_5[0,:].reshape(1,-1), NREP, axis = 0)]
temps_z = [tempz_1, tempz_2, tempz_3, tempz_4, tempz_5]
edots_z = [edotz_1, edotz_2, edotz_3, edotz_4, edotz_5]



nexp = len(dat_all) # number of experiments
nexp_z = len(dat_all_z) # number of experiments

stress_stacked = np.hstack([np.array(v)[:,1] for v in dat_all])
strain_hist_list = [np.array(v)[:,0] for v in dat_all]


stress_stacked_z = np.hstack([np.array(v)[:,1] for v in dat_all_z])
strain_hist_list_z = [np.array(v)[:,0] for v in dat_all_z]

### Implement Subsampling 
#[len(strain_hist_list[j]) for j in range(len(dat_all))]
NSAMP = 100
inds_subsampled = [np.linspace(1,len(strain_hist_list[j]),len(strain_hist_list[j])) % np.floor(len(strain_hist_list[j])/NSAMP) == 0 if len(strain_hist_list[j]) > NSAMP else np.repeat(True,len(strain_hist_list[j])) for j in range(len(dat_all))]
#[len(strain_hist_list[j][inds_subsampled[j]]) for j in range(len(dat_all))]


#####################
### Get Constants ###
#####################
#### should be chi = 0 for quasistatics....


consts_ptw = {
    'beta'   : 0.25,
    'matomic': 9.0122, #u
    'Tmelt0' : 1910 , #melt temp at ambient pressure, Constant_Melt_Temperature
    'rho0'   : 1.85, #at amnient P and T

    #Cubic density
    'r0'     :  1.85839609e+00,
    'r1'     : -7.23098711e-06,
    'r2'     : -1.06942297e-07,
    'r3'     : 4.47660023e-11,

    #Cubic Melt
   
    'tm0': 16.34174022,
    'tm1': 124.56295767,
    'tm2': 700.02491043,
    'tm3': -102.09747419,

    #Cubic specific heat
    
    'c0': -9.61847335e-06,
    'c1': 1.36125051e-07,
    'c2': -1.82037960e-10,
    'c3': 8.44960145e-14,

#BGP shear modulus and melt curve

    'chi'    : 1.0, 
    'G0'     : 1.523, #Mbar, reference cold shear for BGP
    'rho_0'  : 1.855, #reference density for BGP cold shear
    'Tm_0'   : 1560., #reference melt temperature for BGP melt curve
    'rho_m'  : 1.794,  #reference density for BGP melt
    'gamma_1': 0.184,
    'gamma_2': 0.0,
    'gamma_3': 3.0,
    'q3'     : 1.8,
    'q2'     : 1.0, #gamma_2 is zero and q2 is arbitary nonzero constant
    'alpha'  : 0.15, #0.199
    
    }


bounds_ptw = {

    'theta' : (0.0001,   0.2),
    'p'     : (0.0001,   5.),
    's0'    : (0.0001,   0.05),
    'sInf'  : (0.0001,   0.05),
    'kappa' : (0.0001,   0.5),
    'lgamma': (-8.,     -2.),
    'y0'    : (0.0001,   0.05),
    'yInf'  : (0.0001,   0.01),
    'y1'    : (0.001,    0.1),
#    'y2'    : (0.25,    1.3),
    'y2'    : (0.257,      1.3),
    }

def constraints_ptw(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

########################
### Define Model Run ###
########################

class BGP_Melt_Temperature_Fixed(impala.physics.BaseModel):
    consts = ['Tm_0', 'rho_m', 'gamma_1', 'gamma_3', 'q3']
    def value(self, *args):
        mp    = self.parent.parameters
        rho   = self.parent.state.rho
        melt_temp = mp.Tm_0*np.power(rho/mp.rho_m, 1./3.)*np.exp(6*mp.gamma_1*(np.power(mp.rho_m,-1./3.)-np.power(rho,-1./3.))\
                    +2.*mp.gamma_3/mp.q3*(np.power(mp.rho_m,-mp.q3)-np.power(rho,-mp.q3)))
        return melt_temp

class BGP_PW_Shear_Modulus_Fixed_Unprotected(impala.physics.BaseModel):
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
        cold_shear  = mp.G0*np.power(rho/mp.rho_0, 4./3.)*np.exp(6.*mp.gamma_1*(np.power(mp.rho_0,-1./3.)-np.power(rho,-1./3.))\
                    + 2*mp.gamma_2/mp.q2*(np.power(mp.rho_0,-mp.q2)-np.power(rho,-mp.q2)))
        gnow = cold_shear*(1.- mp.alpha* (temp/tmelt))
        gnow[np.where(temp >= tmelt)] = cold_shear[np.where(temp >= tmelt)]*(1.- mp.alpha* 1) #limit to shear at melt temperature
        gnow[np.where(gnow < 0)] = 0.
        #if temp >= tmelt: gnow = 0.0
        #if gnow < 0.0:    gnow = 0.0
        return gnow

impala.physics.BGP_Melt_Temperature_Fixed = BGP_Melt_Temperature_Fixed
impala.physics.BGP_PW_Shear_Modulus_Fixed_Unprotected = BGP_PW_Shear_Modulus_Fixed_Unprotected


class Cubic_Specific_Heat(impala.physics.BaseModel):
    """
    Cubic Specific Heat Model
    """
    consts = ['c0', 'c1', 'c2','c3']

    def value(self, *args):

        tnow=self.parent.state.T
        cnow=self.parent.parameters.c0+self.parent.parameters.c1*tnow+self.parent.parameters.c2*tnow**2+self.parent.parameters.c3*tnow**3
        return cnow
impala.physics.Cubic_Specific_Heat=Cubic_Specific_Heat

class Cubic_Melt_Temperature(impala.physics.BaseModel):
    """
    Cubic Melt Temperature Model
    """
    consts=['tm0', 'tm1', 'tm2', 'tm3']
    def value(self, *args):
        rnow=self.parent.state.rho
        tmeltnow=self.parent.parameters.tm0+self.parent.parameters.tm1*rnow+self.parent.parameters.tm2*rnow**2+self.parent.parameters.tm3*rnow**3
        return tmeltnow    
impala.physics.Cubic_Melt_Temperature=Cubic_Melt_Temperature



model_clust_ptw = sc.ModelMaterialStrength(temps=np.array(temps), 
        edots=np.array(edots)*1e-6, 
        consts=consts_ptw, 
        strain_histories=[strain_hist_list[j][inds_subsampled[j]] for j in range(len(dat_all))],  #Note: implementing subsampling!!!!
        flow_stress_model='PTW_Yield_Stress',
        melt_model= 'Cubic_Melt_Temperature',
        shear_model= 'BGP_PW_Shear_Modulus_Fixed_Unprotected',
        specific_heat_model='Cubic_Specific_Heat', 
        density_model='Cubic_Density',
        pool=False, s2='gibbs')


#Sourced from JeeYeon's table in write-up
ColdShear = [1.877, 2.022, 2.161, 2.325, 3.009]
Density = [2.0829, 2.174, 2.259, 2.357, 2.745]
consts_ptw_z = [{
    'beta'   : 0.25,
    'matomic': 9.0122, 
    'Tmelt0' : 1910 , #melt temp at ambient pressure, 
    'rho0'   : 1.85, #at amnient P and T
    'rho_fixed'     : Density[j],
    'Cv0'  : 0.0001, #should set specific heat = 0, according to JeeYeon
    'chi'    : 0, #z machine should be isothermal, according to JeeYeon
    'G0'     : 1.523, #Mbar, reference cold shear for BGP
    'rho_0'  : 1.844, #reference density for BGP cold shear
    'Tm_0'   : 1560., #reference melt temperature for BGP melt curve
    'rho_m'  : 1.794,  #reference density for BGP melt
    'gamma_1': 0.184,
    'gamma_2': 0.0,
    'gamma_3': 3.0,
    'q3'     : 1.8,
    'q2'     : 1.0, #gamma_2 is zero and q2 is arbitary nonzero constant
    'alpha'  : 0.15,
    'tm0': 16.34174022,
    'tm1': 124.56295767,
    'tm2': 700.02491043,
    'tm3': -102.09747419
    } for j in range(len(dat_all_z))]


class Constant_Density_Custom(impala.physics.BaseModel):
    """
    Constant Density Model
    """
    consts = ['rho_fixed']
    def value(self, *args):
        tnow = self.parent.state.T
        return self.parent.parameters.rho_fixed + tnow*0
    
impala.physics.Constant_Density_Custom = Constant_Density_Custom

model_z = [sc.ModelMaterialStrength(temps=np.array(temps_z)[j], 
        edots=np.array(edots_z)[j]*1e-6, 
        consts=consts_ptw_z[j], 
        strain_histories=[strain_hist_list_z[j]],  #Note: implementing subsampling!!!!
        flow_stress_model='PTW_Yield_Stress',
        melt_model= 'Cubic_Melt_Temperature', #same as main
        shear_model= 'BGP_PW_Shear_Modulus_Fixed_Unprotected',
        specific_heat_model='Constant_Specific_Heat', #different from main
        density_model='Constant_Density_Custom', #different from main
        pool=False, s2='gibbs') for j in range(len(dat_all_z))]



#####################
### Set up Models ###
#####################

s2_ind = np.hstack([[j]*len(np.array(dat_all[j])[inds_subsampled[j],1]) for j in range(len(dat_all))])

setup_clust_ptw = sc.CalibSetup(bounds_ptw, constraints_ptw)
setup_clust_ptw.addVecExperiments(yobs=np.hstack([np.array(dat_all[j])[inds_subsampled[j],1] for j in range(len(dat_all))]), #Note: implementing subsampling!!!!
        model=model_clust_ptw, 
        sd_est=np.array([0.0005]*len(dat_all)),  
        s2_df=np.array([50]*len(dat_all)), 
        s2_ind=s2_ind,  #Note: implementing subsampling!!!!
        theta_ind=s2_ind)  #Note: implementing subsampling!!!!
for j in range(len(dat_all_z)):
    setup_clust_ptw.addVecExperiments(yobs=np.array(dat_all_z[j])[:,1], #Note: implementing subsampling!!!!
        model=model_z[j], 
        sd_est=np.array([0.0005]),  
        s2_df=np.array([50]), 
        s2_ind=[0]*len(np.array(dat_all_z[j])[:,1]),  #Note: implementing subsampling!!!!
        theta_ind=[0]*len(np.array(dat_all_z[j])[:,1]))  #Note: implementing subsampling!!!!
setup_clust_ptw.setTemperatureLadder(1.05**np.arange(50), start_temper=2000) 
setup_clust_ptw.setMCMC(nmcmc=40000, nburn=5000, thin=1, decor=100, start_tau_theta=-4.)
setup_clust_ptw.setHierPriors(
        theta0_prior_mean=np.repeat(0.5, setup_clust_ptw.p), 
        theta0_prior_cov=np.eye(setup_clust_ptw.p)*10**2, 
        Sigma0_prior_df=setup_clust_ptw.p + 20, 
        Sigma0_prior_scale=np.eye(setup_clust_ptw.p)*.1**2 
        )
setup_clust_ptw.setClusterPriors(
        nclustmax=10, 
        eta_prior_shape=1, 
        eta_prior_rate=2 #changed from before, adding more shrinkage
        )

###########################
### Perform Calibration ### (Takes several hours to run)
###########################

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

### Run Calibration 
out_clust = calibClust(setup_clust_ptw)


# with open(path_to_results + 'out_clust.pkl', 'rb') as f: 
#     out_clust = pickle.load(f)

### Save Parent Distribution Mean (theta0) Draws
df0 = pd.DataFrame(out_clust.theta0[:,0,:], columns = setup_clust_ptw.bounds.keys())
df0.to_csv(path_to_results + 'theta0_draws.csv',index=False)

### Save All Parent Distribution Draws (~N(theta0,Sigma0))
theta_parent = sc.chol_sample_1per_constraints(
        out_clust.theta0[:,0], out_clust.Sigma0[:,0], setup_clust_ptw.checkConstraints,
        setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys(), setup_clust_ptw.bounds,
        )
df = pd.DataFrame(theta_parent, columns = setup_clust_ptw.bounds.keys())
df.to_csv(path_to_results + 'parent_draws.csv',index=False)

### Save All Experiment-Specific Theta (thetai) Draws
theta_exp = [out_clust.theta_hist[0][:,0,j,:] for j in range(nexp)]
save_object(theta_exp, path_to_results + 'thetai_draws.pkl')

theta_exp_z1 = out_clust.theta_hist[1][:,0,0,:]
save_object(theta_exp_z1, path_to_results + 'thetai_draws_z1.pkl')

theta_exp_z2 = out_clust.theta_hist[2][:,0,0,:]
save_object(theta_exp_z2, path_to_results + 'thetai_draws_z2.pkl')

theta_exp_z3 = out_clust.theta_hist[3][:,0,0,:]
save_object(theta_exp_z3, path_to_results + 'thetai_draws_z3.pkl')

theta_exp_z4 = out_clust.theta_hist[4][:,0,0,:]
save_object(theta_exp_z4, path_to_results + 'thetai_draws_z4.pkl')

theta_exp_z5 = out_clust.theta_hist[5][:,0,0,:]
save_object(theta_exp_z5, path_to_results + 'thetai_draws_z5.pkl')


delta_exp = [out_clust.delta[0][:,0,j] for j in range(nexp)]
save_object(delta_exp, path_to_results + 'deltai_draws.pkl')

delta_exp_z1 = out_clust.delta[1][:,0,0]
save_object(delta_exp_z1, path_to_results + 'deltai_draws_z1.pkl')

delta_exp_z2 = out_clust.delta[2][:,0,0]
save_object(delta_exp_z2, path_to_results + 'deltai_draws_z2.pkl')

delta_exp_z3 = out_clust.delta[3][:,0,0]
save_object(delta_exp_z3, path_to_results + 'deltai_draws_z3.pkl')

delta_exp_z4 = out_clust.delta[4][:,0,0]
save_object(delta_exp_z4, path_to_results + 'deltai_draws_z4.pkl')

delta_exp_z5 = out_clust.delta[5][:,0,0]
save_object(delta_exp_z5, path_to_results + 'deltai_draws_z5.pkl')


eta = pd.DataFrame(out_clust.eta[:,0].reshape(-1,1), columns = ['eta'])
eta.to_csv(path_to_results + 'eta_draws.csv',index=False)

theta_g = [out_clust.theta[j][0,:,:] for j in range(len(out_clust.theta))]
theta_g = np.asarray(theta_g)
save_object(theta_g, path_to_results + 'thetag_draws.pkl')


# ### Save Tempering Diagnostic
# total_temperature_swaps(out_clust,setup_clust_ptw)
# plt.savefig(path_to_plots + 'tempering.png')

# ### Save Big Pairs Plot
# pp.pairwise_theta_plot_hier(setup_clust_ptw, out_clust, path_to_plots + 'pairs_hier.png', np.arange(10000, 40000, 10)) # saves the combined pairs plot



###################################
### Read in Calibration Results ###
###################################

df0 = pd.read_csv(path_to_results + 'theta0_draws.csv')
df = pd.read_csv(path_to_results + 'parent_draws.csv')
with open(path_to_results + 'thetai_draws.pkl', 'rb') as f: 
    theta_exp = pickle.load(f)

with open(path_to_results + 'thetai_draws_z1.pkl', 'rb') as f: 
    theta_exp_z1 = pickle.load(f)

with open(path_to_results + 'thetai_draws_z2.pkl', 'rb') as f: 
    theta_exp_z2 = pickle.load(f)

with open(path_to_results + 'thetai_draws_z3.pkl', 'rb') as f: 
    theta_exp_z3 = pickle.load(f)

with open(path_to_results + 'thetai_draws_z4.pkl', 'rb') as f: 
    theta_exp_z4 = pickle.load(f)

with open(path_to_results + 'thetai_draws_z5.pkl', 'rb') as f: 
    theta_exp_z5 = pickle.load(f)


eta = pd.read_csv(path_to_results + 'eta_draws.csv')


with open(path_to_results + 'deltai_draws.pkl', 'rb') as f: 
    delta_exp = pickle.load(f)

with open(path_to_results + 'deltai_draws_z1.pkl', 'rb') as f: 
    delta_exp_z1 = pickle.load(f)

with open(path_to_results + 'deltai_draws_z2.pkl', 'rb') as f: 
    delta_exp_z2 = pickle.load(f)

with open(path_to_results + 'deltai_draws_z3.pkl', 'rb') as f: 
    delta_exp_z3 = pickle.load(f)

with open(path_to_results + 'deltai_draws_z4.pkl', 'rb') as f: 
    delta_exp_z4 = pickle.load(f)

with open(path_to_results + 'deltai_draws_z5.pkl', 'rb') as f: 
    delta_exp_z5 = pickle.load(f)

with open(path_to_results + 'thetag_draws.pkl', 'rb') as f: 
    theta_g = pickle.load(f)


### Option 3: Sampling with appropriate proportions conditional on draws from 10 clusters and option to add new parent draw
### Arthur Lui says this is right :) 
# uu = np.arange(10000, 40000, 10)
# A = theta_g[uu,:,:] #theta_g (niter x nclust x p)
# B = np.zeros((len(uu),setup_clust_ptw.nclustmax))  #membership numbers (niter x nclust)
# for i in range(len(delta_exp)):
#     for j in range(len(uu)):
#         B[j,delta_exp[i][uu][j]]+=1
# N = B[0,:].sum() #number of experiments
# new_parent_ind = [np.random.choice(a=np.arange(0,2), size = 1, p = np.append(N/(np.asarray(eta)[uu][j] + N), np.asarray(eta)[uu][j]/(np.asarray(eta)[uu][j] + N)))[0] for j in range(len(uu))]
# new_member_probs = [ (B[j,:]/(B[j,:].sum() + np.asarray(eta)[uu][j])) +  (np.asarray(eta)[uu][j]/(B[j,:].sum() + np.asarray(eta)[uu][j])*(1/((B[j,:]==0).sum()+1e-10))*(B[j,:] == 0))   for j in range(len(uu))]
# new_member_probs = np.asarray(new_member_probs)
# df_dirich = [ A[j,np.random.choice(a=np.arange(0,setup_clust_ptw.nclustmax), size = 1, p = B[j,:]/(B[j,:].sum()))[0]] if new_parent_ind[j] == 0 else np.asarray(df)[j] for j in range(len(uu))]
# df_dirich = np.asarray(df_dirich)
# df_dirich = pd.DataFrame(df_dirich,columns = setup_clust_ptw.bounds.keys())
# df_dirich.to_csv(path_to_results + 'parentdirich_draws.csv',index=False)



df_dirich = pd.read_csv(path_to_results + 'parentdirich_draws.csv')


#########################
### Visualize Outputs ###
#########################


uu = np.arange(10000, 40000, 10)
n_exp = len(np.unique(setup_clust_ptw.s2_ind[0]))
n_exp_z = len(dat_all_z)
s2_inds = setup_clust_ptw.s2_ind[0]

### Trace Plot
plt.close('all')
pp.parameter_trace_plot(np.asarray(df_dirich),ylim=[0,1])
plt.savefig(path_to_plots + 'trace_parent.png')

plt.close('all')
pp.parameter_trace_plot(np.asarray(eta))
plt.savefig(path_to_plots + 'trace_eta.png')

### Trace Plot
# pp.parameter_trace_plot(np.asarray(df_dirich),ylim=[0,1])
# plt.show()




### KDE Pairs
# pairs_kde(np.asarray(df)[uu,:])
# plt.savefig(path_to_plots + 'pairs.png')


### Prediction Plots 
PARENT_Y = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_dirich),setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETA0_Y = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df0)[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETAi_Y = [get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp[j][uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws'] for j in range(nexp)]

QUANTS_PARENT_Y = [np.quantile(PARENT_Y[j],[0.025,0.5,0.975],axis=0) for j in range(len(PARENT_Y))]
QUANTS_THETA0_Y = [np.quantile(THETA0_Y[j],[0.025,0.5,0.975],axis=0) for j in range(len(THETA0_Y))]
QUANTS_THETAi_Y = [np.quantile(THETAi_Y[j][0],[0.025,0.5,0.975],axis=0) for j in range(len(THETAi_Y))]
for exp_ind in range(len(np.unique(setup_clust_ptw.s2_ind[0]))):
    fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
    ax.fill_between(setup_clust_ptw.models[0].meas_strain_histories[exp_ind], QUANTS_PARENT_Y[0][0,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], QUANTS_PARENT_Y[0][2,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], color = 'lightgray', zorder = 1)            
    ax.plot(setup_clust_ptw.models[0].meas_strain_histories[exp_ind],QUANTS_PARENT_Y[0][1,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], color = 'darkgray', zorder = 2, linewidth = 2, label = 'Parent')
    ax.fill_between(setup_clust_ptw.models[0].meas_strain_histories[exp_ind], QUANTS_THETAi_Y[exp_ind][0,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], QUANTS_THETAi_Y[exp_ind][2,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], color = 'pink', zorder = 3)            
    ax.plot(setup_clust_ptw.models[0].meas_strain_histories[exp_ind],QUANTS_THETAi_Y[exp_ind][1,np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], color = 'red', zorder = 4, linewidth = 2, label = 'Experiment-Specific')
    ax.scatter(setup_clust_ptw.models[0].meas_strain_histories[exp_ind],setup_clust_ptw.ys[0][np.where(setup_clust_ptw.s2_ind[0] == exp_ind)[0]], color = 'black', zorder = 3, label = 'Data')
    ax.set_ylabel('Stress')
    ax.set_xlabel('Strain')
    ax.title.set_text('Prediction for Experiment '+str(exp_ind))
    ax.legend()
    plt.savefig(path_to_plots + 'experiment_' +str(exp_ind)+'.png')
matplotlib.pyplot.close()


THETAi_Y_z1 = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp_z1[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETAi_Y_z2 = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp_z2[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETAi_Y_z3 = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp_z3[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETAi_Y_z4 = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp_z4[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
THETAi_Y_z5 = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(theta_exp_z5[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']


purity = np.append(np.repeat('Lower',5), np.repeat('Higher',6))

### Plot Differences between Experiments 
KEYS = np.array(pd.DataFrame(setup_clust_ptw.bounds.keys())).flatten()
theta_i = []
for j in range(len(theta_exp)):
    mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp[j][uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
    mat['exp'] = j
    mat['strain_ind'] = np.argsort(np.argsort(np.asarray(edots)))[j]
    mat['purity'] = purity[j]
    if(edots[j] <= 1):
        mat['Category'] = 'QS/SHPB, Low Strain Rate'
    else:
        mat['Category'] = 'QS/SHPB, Higher Strain Rate'
    if j == 0:
        theta_i = [mat]
    else:
        theta_i.append(mat)
 
mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp_z1[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
mat['exp'] = n_exp + 1   
mat['strain_ind'] = n_exp + 1 
mat['Category'] = 'Z Machine' 
mat['purity'] = 'Lower'  
theta_i.append(mat)     
mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp_z2[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
mat['exp'] = n_exp + 2 
mat['strain_ind'] = n_exp + 2
mat['Category'] = 'Z Machine'    
mat['purity'] = 'Lower'  
theta_i.append(mat)
mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp_z3[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
mat['exp'] = n_exp + 3      
mat['strain_ind'] = n_exp + 3
mat['Category'] = 'Z Machine'   
mat['purity'] = 'Lower'   
theta_i.append(mat)
mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp_z4[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
mat['exp'] = n_exp + 4       
mat['strain_ind'] = n_exp + 4  
mat['Category'] = 'Z Machine'  
mat['purity'] = 'Lower'      
theta_i.append(mat)
mat = pd.DataFrame(np.array(pd.DataFrame(sc.tran_unif(theta_exp_z5[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T, columns = KEYS)
mat['exp'] = n_exp + 5       
mat['strain_ind'] = n_exp + 5
mat['Category'] = 'Z Machine' 
mat['purity'] = 'Lower'    
theta_i.append(mat)
        
theta_i_long = pd.concat(theta_i)

# palette = {
#     'QS/SHPB, Low Strain Rate': 'tab:blue',
#     'QS/SHPB, Higher Strain Rate': 'tab:purple',
#     'Z Machine': 'tab:red',
# }

palette = ['tab:blue','tab:purple','tab:red']
fig,ax=plt.subplots(2,5,figsize=(20,8))
fig.tight_layout(pad=2.0)
for i in range(2):
    for j in range(5):
        ax[i,j].set_xlabel('Experiment')
        ax[i,j].set_ylabel('')
        ax[i,j].set_title(KEYS[5*i+j]) 
        A = sns.boxplot(data = theta_i_long, x = theta_i_long['exp'], y = theta_i_long[KEYS[5*i+j]], hue = theta_i_long['Category'],ax=ax[i,j],showfliers=False, palette = palette, width = 1, dodge = False)
        A.set(xlabel='Experiment',ylabel='')
        A.set(xticklabels=[]) 
        if i > 0 or j > 0:
            ax[i,j].get_legend().set_visible(False)
            #A._legend.remove()
plt.savefig(path_to_plots + 'thetai.png')



palette = ['tab:blue','tab:purple','tab:red']
fig,ax=plt.subplots(2,5,figsize=(20,8))
fig.tight_layout(pad=2.0)
for i in range(2):
    for j in range(5):
        ax[i,j].set_xlabel('Experiment')
        ax[i,j].set_ylabel('')
        ax[i,j].set_title(KEYS[5*i+j]) 
        A = sns.boxplot(data = theta_i_long, x = theta_i_long['strain_ind'], y = theta_i_long[KEYS[5*i+j]], hue = theta_i_long['Category'], ax=ax[i,j],showfliers=False, palette = palette, width = 1, dodge = False)
        A.set(xlabel='Experiment',ylabel='')
        A.set(xticklabels=[]) 
        if i > 0 or j > 0:
            ax[i,j].get_legend().set_visible(False)
            #A._legend.remove()
plt.savefig(path_to_plots + 'thetai_sorted.png')




### Plot Associations with Strain Rate, Temp
# MEDS = theta_i_long.groupby('exp').mean()
# MEDS = np.array(MEDS)
# KEYS = np.array(pd.DataFrame(setup_clust_ptw.bounds.keys())).flatten()

### Plot Associations with Strain Rate, Temp
MEDS = np.hstack([np.asarray(theta_i_long[KEYS[i]].groupby(theta_i_long['exp']).mean()).reshape(-1,1) for i in range(len(KEYS))])
MEDS = np.array(MEDS)
KEYS = np.array(pd.DataFrame(setup_clust_ptw.bounds.keys())).flatten()

# fig,ax=plt.subplots(1,setup_clust_ptw.p,figsize=(30,4))
# fig.tight_layout(pad=2.0)
# for j in range(setup_clust_ptw.p):
#     ax[j].set_xlabel('log10(Strain Rate)')
#     ax[j].set_ylabel('Post. Mean')
#     ax[j].set_title(KEYS[j]) 
#     sns.scatterplot(x=np.log10(edots), y=MEDS[:,j], ax = ax[j])
# plt.savefig(path_to_plots + 'edots.png')

# fig,ax=plt.subplots(1,setup_clust_ptw.p,figsize=(30,4))
# fig.tight_layout(pad=2.0)
# for j in range(setup_clust_ptw.p):
#     ax[j].set_xlabel('Temperature')
#     ax[j].set_ylabel('Post. Mean')
#     ax[j].set_title(KEYS[j]) 
#     sns.scatterplot(x=temps, y=MEDS[:,j], ax = ax[j])
# plt.savefig(path_to_plots + 'temp.png')



###############################
### Get Bounding Parameters ###
###############################

save_parent_strength2(setup_clust_ptw, setup_clust_ptw.models[0], df_dirich.reset_index(drop=True), path_to_results+'parent_strength.csv') # saves parent distribution
pp.get_bounds(edot = 10**5, strain = 0.5, temp = 1000, 
               results_csv = path_to_results+'parent_strength.csv', 
               write_path = path_to_results+'bounding_sets.csv', percentiles=[0.05, 0.5, 0.95])



################################
### Output "Best" Parameters ###
################################


from scipy.stats import qmc
l_bounds = np.array(pd.DataFrame(setup_clust_ptw.bounds.values()))[:,0]
u_bounds = np.array(pd.DataFrame(setup_clust_ptw.bounds.values()))[:,1]
#parent_scaled = qmc.scale(np.array(df)[uu,:], l_bounds, u_bounds)
parent_scaled = qmc.scale(np.array(df_dirich), l_bounds, u_bounds)


parent_scaled_native = pd.DataFrame(parent_scaled, columns = setup_clust_ptw.bounds.keys())
parent_scaled_native.to_csv(path_to_results + 'parent_draws_native.csv',index=False)


BEST = dict()
SSE = dict()
MAPE = dict()

### Parent_Median
pred_median = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.median(parent_scaled,axis = 0).reshape(1,-1))['outcome_draws']
median_sse = sum([((pred_median[0][0,np.where(s2_inds == i)]-setup_clust_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp)]) + sum([((pred_median[i]-setup_clust_ptw.ys[i])**2).mean(axis=1) for i in range(1,6)])
median_mape = np.sum([(np.abs(pred_median[0][0,np.where(s2_inds == i)]-setup_clust_ptw.ys[0][np.where(s2_inds == i)])/setup_clust_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(n_exp)])
median_mape = median_mape + np.sum([(np.abs(pred_median[j]-setup_clust_ptw.ys[j])/setup_clust_ptw.ys[j]).mean(axis=1) for j in range(1,6)])
median_mape = 100*(median_mape/(n_exp + n_exp_z))
BEST['parent_median'] = np.median(parent_scaled,axis = 0)
SSE['parent_median'] = median_sse
MAPE['parent_median'] = median_mape

### Best Draw 
parent_sse = sum([((PARENT_Y[0][:,np.where(s2_inds == i)[0]]-setup_clust_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp)]) +  np.sum(np.hstack([((PARENT_Y[i]-setup_clust_ptw.ys[i])**2).mean(axis=1).reshape(-1,1) for i in range(1,6)]), axis = 1)
pred_minsse = np.hstack([PARENT_Y[0][np.where(parent_sse == parent_sse.min())[0][0],np.where(s2_inds == i)[0]] for i in range(n_exp)]).reshape(1,-1)
parent_mape = np.sum([(np.abs(pred_minsse[0,np.where(s2_inds == i)]-setup_clust_ptw.ys[0][np.where(s2_inds == i)])/setup_clust_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(n_exp)])
parent_mape = parent_mape + np.sum(np.hstack([(np.abs(PARENT_Y[j][np.where(parent_sse == parent_sse.min())[0][0]]-setup_clust_ptw.ys[j])/setup_clust_ptw.ys[j]).mean() for j in range(1,6)]))
parent_mape = 100*(parent_mape/(n_exp + n_exp_z))
BEST['parent_minsse'] = parent_scaled[np.where(parent_sse == parent_sse.min())[0][0],:].flatten()
SSE['parent_minsse'] = np.array([parent_sse.min()])
MAPE['parent_minsse'] = parent_mape



BEST_df = pd.DataFrame(BEST.values(), columns = np.array(pd.DataFrame(setup_clust_ptw.bounds.keys())).flatten())
BEST_df['method'] = np.array(pd.DataFrame(BEST.keys())).flatten()
BEST_df['sse'] = np.array(pd.DataFrame(SSE.values())).flatten()
BEST_df['mape'] = np.array(pd.DataFrame(MAPE.values())).flatten()
BEST_df.to_csv(path_to_results + 'best.csv',index=False)


### Visualize Best-Fitting Values
fig,ax=plt.subplots(1,1,figsize=(16,6), sharey = False)   
OBS = setup_clust_ptw.ys[0]
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_median[0][0]).flatten(), color = 'purple', zorder = 2, linewidth = 2, label = 'Posterior Median')
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_minsse).flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'Best SSE')
ax.scatter(np.arange(0,len(OBS)),OBS, color = 'black', zorder = 4, label = 'Data')
ax.set_ylabel('Stress')
ax.set_xlabel('Observation Index')
ax.title.set_text('Prediction for All Experiments')
ax.legend()
plt.savefig(path_to_plots + 'best_all.png')


###################################
### Summarize Prediction Errors ###
###################################

# ### Get Parent Distribution Error and Coverage Rates
# PARENT_COVERAGE = [np.quantile(PARENT_Y[j],[0.025,0.975],axis=0) for j in range(len(PARENT_Y))]
# PARENT_COVERAGE_BIN = [np.empty([setup_clust_ptw.ys[i].shape[0]]) for i in range(len(PARENT_COVERAGE))]
# for i in range(len(PARENT_COVERAGE)): 
#     PARENT_COVERAGE_BIN[i] = (setup_clust_ptw.ys[i]<= PARENT_COVERAGE[i][1,:]).astype(int) + (setup_clust_ptw.ys[i]>= PARENT_COVERAGE[i][0,:]).astype(int)
#     PARENT_COVERAGE_BIN[i][PARENT_COVERAGE_BIN[i] == 1] = 0
#     PARENT_COVERAGE_BIN[i][PARENT_COVERAGE_BIN[i] == 2] = 1
# PARENT_COVERAGE_AGG = []
# for j in range(len(PARENT_COVERAGE)):
#     PARENT_COVERAGE_AGG = np.append(PARENT_COVERAGE_AGG, PARENT_COVERAGE_BIN[j]) 

# PARENT_ERRORS = [np.empty([setup_clust_ptw.ys[i].shape[0]]) for i in range(len(PARENT_Y))]
# for i in range(len(PARENT_Y)): 
#     PARENT_ERRORS[i] = np.abs(setup_clust_ptw.ys[i] -np.mean(PARENT_Y[i],axis=0))/setup_clust_ptw.ys[i]
# PARENT_ERRORS_AGG = [np.empty([setup_clust_ptw.ns2[i]]) for i in range(len(PARENT_Y))]
# for j in range(len(PARENT_Y)):
#     for i in range(setup_clust_ptw.ns2[j]):
#         PARENT_ERRORS_AGG[j][i] = np.mean(PARENT_ERRORS[j][setup_clust_ptw.s2_ind[j] == i])
# PARENT_ERRORS_AGG=np.vstack(PARENT_ERRORS_AGG)

# MEDIAN_ERRORS = [np.empty([setup_clust_ptw.ys[i].shape[0]]) for i in range(len(PARENT_Y))]
# for i in range(len(PARENT_Y)): 
#     MEDIAN_ERRORS[i] = np.abs(setup_clust_ptw.ys[i] -np.hstack(pred_median).flatten())/setup_clust_ptw.ys[i]
# MEDIAN_ERRORS_AGG = [np.empty([setup_clust_ptw.ns2[i]]) for i in range(len(PARENT_Y))]
# for j in range(len(PARENT_Y)):
#     for i in range(setup_clust_ptw.ns2[j]):
#         MEDIAN_ERRORS_AGG[j][i] = np.mean(MEDIAN_ERRORS[j][setup_clust_ptw.s2_ind[j] == i])
# MEDIAN_ERRORS_AGG=np.vstack(MEDIAN_ERRORS_AGG)

# MINSSE_ERRORS = [np.empty([setup_clust_ptw.ys[i].shape[0]]) for i in range(len(PARENT_Y))]
# for i in range(len(PARENT_Y)): 
#     MINSSE_ERRORS[i] = np.abs(setup_clust_ptw.ys[i] -np.hstack(pred_minsse).flatten())/setup_clust_ptw.ys[i]
# MINSSE_ERRORS_AGG = [np.empty([setup_clust_ptw.ns2[i]]) for i in range(len(PARENT_Y))]
# for j in range(len(PARENT_Y)):
#     for i in range(setup_clust_ptw.ns2[j]):
#         MINSSE_ERRORS_AGG[j][i] = np.mean(MINSSE_ERRORS[j][setup_clust_ptw.s2_ind[j] == i])
# MINSSE_ERRORS_AGG=np.vstack(MINSSE_ERRORS_AGG)

# fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
# #ax.scatter(np.linspace(0,nexp-1,nexp),100*PARENT_ERRORS_AGG, label = 'Entire Distribution (Overall = ' + str(round(100*PARENT_ERRORS_AGG.mean(),1)) + '%)')
# ax.scatter(np.linspace(0,nexp-1,nexp),100*MEDIAN_ERRORS_AGG, label = 'Posterior Median (Overall = ' + str(round(100*MEDIAN_ERRORS_AGG.mean(),1)) + '%)', color = 'red')
# ax.scatter(np.linspace(0,nexp-1,nexp),100*MINSSE_ERRORS_AGG, label = 'Best SSE (Overall = ' + str(round(100*MINSSE_ERRORS_AGG.mean(),1)) + '%)', color = 'purple')
# ax.set_ylabel('Percent Error')
# ax.title.set_text('Mean Absolute Percent Error (Parent Coverage = ' + str(round(PARENT_COVERAGE_AGG.mean(),3)) + ')')
# ax.set_xlabel('Experiment')
# ax.legend()
# plt.savefig(path_to_plots + 'error.png')


################################
### Cross-Experiment Heatmap ###
################################


CROSS_ERRORS = np.empty([nexp+5,nexp+5])
for i in range(nexp): 
    for j in range(nexp):
        CROSS_ERRORS[i,j] = np.mean(np.abs(setup_clust_ptw.ys[0][np.where(setup_clust_ptw.s2_ind[0] == j)[0]] -np.mean(THETAi_Y[i][0][:,np.where(setup_clust_ptw.s2_ind[0] == j)[0]],axis=0))/setup_clust_ptw.ys[0][np.where(setup_clust_ptw.s2_ind[0] == j)[0]])

for i in range(nexp): 
    for j in range(nexp,nexp+5):
        CROSS_ERRORS[i,j] = np.mean(np.abs(setup_clust_ptw.ys[j-nexp+1] -np.mean(THETAi_Y[i][j-nexp+1],axis=0))/setup_clust_ptw.ys[j-nexp+1])

THETAi_Y_z = [THETAi_Y_z1,THETAi_Y_z2,THETAi_Y_z3,THETAi_Y_z4,THETAi_Y_z5]
for i in range(nexp,nexp+5): 
    for j in range(nexp):
        CROSS_ERRORS[i,j] = np.mean(np.abs(setup_clust_ptw.ys[0][np.where(setup_clust_ptw.s2_ind[0] == j)[0]] -np.mean(THETAi_Y_z[i-nexp][0][:,np.where(setup_clust_ptw.s2_ind[0] == j)[0]],axis=0))/setup_clust_ptw.ys[0][np.where(setup_clust_ptw.s2_ind[0] == j)[0]])

for i in range(nexp,nexp+5): 
    for j in range(nexp,nexp+5):
        CROSS_ERRORS[i,j] = np.mean(np.abs(setup_clust_ptw.ys[j-nexp+1] -np.mean(THETAi_Y_z[i-nexp][j-nexp+1],axis=0))/setup_clust_ptw.ys[j-nexp+1])


CROSS_ERRORS[CROSS_ERRORS>1.2]=1.2
# i = rows = theta_i index
# j = columns = prediction index
# reverse for plotting
plt.close('all')
ax = sns.heatmap(CROSS_ERRORS.T*100, linewidths =0)
ax.set_xlabel('Theta Index')
ax.set_ylabel('Prediction Index')
ax.set_title('Prediction Errors (%)')
plt.savefig(path_to_plots + 'cross_errors.png')

import copy
CROSS_ERRORS_SORTED = np.copy(CROSS_ERRORS)
CROSS_ERRORS_SORTED = CROSS_ERRORS_SORTED[np.argsort(np.append(np.asarray(edots).flatten(),np.asarray(edots_z))),:]
CROSS_ERRORS_SORTED = CROSS_ERRORS_SORTED[:,np.argsort(np.append(np.asarray(edots).flatten(),np.asarray(edots_z)))]


plt.close('all')
ax = sns.heatmap(CROSS_ERRORS.T*100, linewidths =0)
ax.set_xlabel('Theta Index')
ax.set_ylabel('Prediction Index')
ax.set_title('Prediction Errors (%)')
plt.savefig(path_to_plots + 'cross_errors.png')



plt.close('all')
ax = sns.heatmap(CROSS_ERRORS_SORTED.T*100, linewidths =0)
ax.set_xlabel('Theta Index (by edot)')
ax.set_ylabel('Prediction Index (by edot)')
ax.set_title('Prediction Errors (%)')
plt.savefig(path_to_plots + 'cross_errors_sorted.png')


#######################
### Scoring Metrics ###
#######################
#https://github.com/adrian-lison/interval-scoring

# with open(path_to_dir+'Helper_Functions/scoring.py') as fd:
#     exec(fd.read())    

# alphas = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01])
# probs = np.append(np.flip(alphas/2),1-(alphas/2))
# #PARENT_Y = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df)[uu,:],setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
# PARENT_Y = get_outcome_predictions_impala(setup_clust_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df_dirich),setup_clust_ptw.bounds_mat, setup_clust_ptw.bounds.keys()).values())).T)['outcome_draws']
# QUANTS_PARENT_Y = np.quantile(PARENT_Y[0],probs,axis=0)
# MEDIANS_PARENT_Y = np.quantile(PARENT_Y[0],0.5,axis=0)

# ### WIS
# quantile_dict_test = dict(zip(list(probs), QUANTS_PARENT_Y))
# point_dict_test = dict(zip(list(probs), np.repeat(MEDIANS_PARENT_Y.reshape(1,-1),len(probs),axis=0))) 
# WIS = weighted_interval_score(setup_clust_ptw.ys[0],alphas=alphas,weights=alphas/2,q_dict=quantile_dict_test)[0]
# WIS_AGG = np.empty([setup_clust_ptw.ns2[0]])
# for i in range(n_exp):
#     WIS_AGG[i] = np.mean(WIS[setup_clust_ptw.s2_ind[0] == i])

# fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
# ax.scatter(np.linspace(0,nexp-1,nexp),WIS_AGG, label = 'Overall WIS = ' + str(round(WIS_AGG.mean(),4)))
# ax.set_ylabel('Percent Error')
# ax.title.set_text('Weighted Interval Score')
# ax.set_xlabel('Experiment')
# ax.legend()
# plt.savefig(path_to_plots + 'wis.png')
# plt.show()

# ### COVERAGES
# COVERAGES = [np.mean(1-outside_interval(observations =setup_clust_ptw.ys[0], lower = np.quantile(PARENT_Y[0],alphas[j]/2,axis=0), upper = np.quantile(PARENT_Y[0],1-(alphas[j]/2),axis=0) )) for j in range(len(alphas))]

# n = PARENT_Y[0].shape[1]
# p = np.array(COVERAGES)
# plt.plot(100*(1-alphas), 100*p, color = 'blue')
# plt.fill_between(x=100*(1-alphas), y1=100*(p-1.96*np.sqrt(p*(1-p)/n)), y2 = 100*(p+1.96*np.sqrt(p*(1-p)/n)), alpha = 0.1, color = 'blue')
# plt.scatter(100*(1-alphas), 100*p, color = 'blue')
# plt.axline((0, 0), slope=1, color = 'black', linestyle = 'dashed')
# plt.ylabel('Empirical Coverage (%)')
# plt.xlabel('Confidence Level (%)')
# plt.title('Coverage by Level')
# plt.xlim(0,105)
# plt.ylim(0,105)
# plt.savefig(path_to_plots + 'coverage.png')
# plt.show()

# ### PIT VALUES
# PITS = [np.mean(PARENT_Y[0][:,u] <= setup_clust_ptw.ys[0][u]) for u in range(len(setup_clust_ptw.ys[0]))]

# fig, ax = plt.subplots(figsize=(6, 5))
# A = sns.histplot(data=pd.DataFrame(PITS), x=np.asarray(PITS), stat='probability', ax=ax)
# ax.set_ylabel('Frequency')
# ax.title.set_text('PIT Histogram')
# ax.set_xlabel('PIT Probability')
# plt.savefig(path_to_plots + 'pit.png')
# plt.show()



# ### MEAN INTERVAL WIDTH
# WIDTHS = [np.mean(np.quantile(PARENT_Y[0],1-(alphas[j]/2),axis=0) - np.quantile(PARENT_Y[0],alphas[j]/2,axis=0)) for j in range(len(alphas))]

# plt.plot(100*(1-alphas), WIDTHS, color = 'blue')
# plt.scatter(100*(1-alphas), WIDTHS, color = 'blue')
# plt.ylabel('Interval Widths')
# plt.xlabel('Confidence Level (%)')
# plt.title('Average Interval Width by Level')
# plt.savefig(path_to_plots + 'widths.png')
# plt.show()


################################
### Experiment Co-Clustering ###
################################

import itertools
cluster_draws = np.vstack(delta_exp).T
cluster_draws = np.append(cluster_draws, delta_exp_z1.reshape(-1,1), axis = 1)
cluster_draws = np.append(cluster_draws, delta_exp_z2.reshape(-1,1), axis = 1)
cluster_draws = np.append(cluster_draws, delta_exp_z3.reshape(-1,1), axis = 1)
cluster_draws = np.append(cluster_draws, delta_exp_z4.reshape(-1,1), axis = 1)
cluster_draws = np.append(cluster_draws, delta_exp_z5.reshape(-1,1), axis = 1)
cluster_draws = cluster_draws[uu,:]
D = []
for i in range(setup_clust_ptw.nclustmax):
    A = [np.where(cluster_draws[j,:]==i)[0] for j in range(cluster_draws.shape[0]) if len(np.where(cluster_draws[j,:]==i)[0]) > 1]
    B = [np.asarray(list(itertools.combinations(A[j], 2)))for j in range(len(A))]
    D.append(np.concatenate(B,axis=0))
cluster_pairs = np.concatenate(D,axis=0)

co_cluster = np.zeros((nexp + n_exp_z,nexp + n_exp_z),dtype = 'int32')
for i in range(cluster_pairs.shape[0]):
    co_cluster[cluster_pairs[i,0],cluster_pairs[i,1]] = co_cluster[cluster_pairs[i,0],cluster_pairs[i,1]] + 1
    co_cluster[cluster_pairs[i,1],cluster_pairs[i,0]] = co_cluster[cluster_pairs[i,0],cluster_pairs[i,1]] + 1
co_cluster = co_cluster/cluster_draws.shape[0]
for i in range(nexp + n_exp_z):
    co_cluster[i,i] = 1

plt.close('all')
ax = sns.heatmap(co_cluster, linewidths =0)
ax.set_xlabel('Experiment Index')
ax.set_ylabel('Experiment Index')
ax.set_title('Percent of Iterations in Same Cluster (%)')
plt.savefig(path_to_plots + 'cross_clusters.png')
plt.show()


import copy

co_cluster_sorted = np.copy(co_cluster)
co_cluster_sorted = co_cluster_sorted[np.argsort(np.append(np.asarray(edots).flatten(), np.asarray(edots_z).flatten())),:]
co_cluster_sorted = co_cluster_sorted[:,np.argsort(np.append(np.asarray(edots).flatten(), np.asarray(edots_z).flatten()))]

# ### twoway sorting
# edots_long = np.append(np.asarray(edots).flatten(), np.asarray(edots_z).flatten())
# temps_long = np.append(np.asarray(temps).flatten(), np.asarray(temps_z).flatten())
# custom_ordering = []
# for i in range(len(np.unique(edots_long))):
#     mat = np.where(edots_long == np.unique(edots_long)[i])[0]
#     mat = mat[np.argsort(temps_long[mat])]
#     if i == 0:
#         custom_ordering = [mat]
#     else:
#         custom_ordering.append(mat)
# custom_ordering = np.hstack(custom_ordering)

# co_cluster_sorted = np.copy(co_cluster)
# co_cluster_sorted = co_cluster_sorted[custom_ordering,:]
# co_cluster_sorted = co_cluster_sorted[:,custom_ordering]


plt.close('all')
ax = sns.heatmap(co_cluster_sorted*100, linewidths =0)
ax.set_xlabel('Experiment Index (by edot)')
ax.set_ylabel('Experiment Index (by edot)')
ax.set_title('Percent of Iterations in Same Cluster (%)')
plt.savefig(path_to_plots + 'cross_clusters_sorted.png')



num_clust = [ len(np.unique(cluster_draws[i,:])) for i in range(cluster_draws.shape[0])]

plt.close('all')
ax=sns.histplot(pd.DataFrame(num_clust))
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Frequency')
ax.set_title('Number of Clusters')
plt.savefig(path_to_plots + 'num_clusters.png')
plt.show()