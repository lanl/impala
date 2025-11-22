#########################################
#########################################
### Beryllium Pooled  PTW-BP Calibration ###
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
from sklearn.metrics import r2_score

 
path_to_dir = '/Users/jyplohr/impala_runs/'
path_to_data = '/Users/jyplohr/impala_runs/Be/data/'
path_to_results = '/Users/jyplohr/impala_runs/Be/pooled_Z_RMI_PTW-BP/results/'
path_to_plots = '/Users/jyplohr/impala_runs/Be/pooled_Z_RMI_PTW-BP/plots/'


with open(path_to_dir+'Helper_Functions/MCMC_Diagnostics.py') as fd:
    exec(fd.read())    
       
with open(path_to_dir+'Helper_Functions/Posterior_Mode.py') as fd:
    exec(fd.read())


# with open(path_to_dir+'Helper_Functions/Piecewise_Linear.py') as fd:
#     exec(fd.read())

################
### Comments ###
################

### This script implements pooled Impala calibration for Au, including 
### - subsampling to produce about 50 points per experiment
### - strong shrinkage toward parent distribution mean
### - adiabatic heat transfer model

####################
### Read in Data ###
####################
## 1st column is plastic strain 2nd column is stress (units: mbar)
## Be experimental data (0-9: QS, SHPB, 10-14:Z)
## NOTE: dat10- dat14 are Z-machine data and at non-ambient pressure. Need to fix density and shear modulus for these sets


dat0 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef200c1e-3_Mbar.txt")
dat1 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef300c1_Mbar.txt")
dat2 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef400c1_Mbar.txt")
dat3 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef500c1_Mbar.txt")
dat4 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef600c1_Mbar.txt")
dat5 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bed50433_Mbar.txt")
dat6 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bed30393_Mbar.txt")
dat7 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bed20373_Mbar.txt")
dat8 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bef0203500_Mbar.x.txt")
dat9 = np.loadtxt(path_to_data + "QS_SHPB_Z_2017/Bed-50353_Mbar.txt")


s200F_1 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_293K_1850s_clean_Mbar.txt")
s200F_2 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_295K_0.001s_clean_Mbar.txt")
s200F_3 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_295K_1s_clean_Mbar.txt")
s200F_4 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_473K_0.001s_clean_Mbar.txt")
s200F_5 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_473K_1s_clean_Mbar.txt")
s200F_6 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_473K_2000s_clean_Mbar.txt")
s200F_7 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_673K_0.001s_clean_Mbar.txt")
s200F_8 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_673K_1s_clean_Mbar.txt")
s200F_9 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_new_673K_3000s_clean_Mbar.txt")
s200F_10 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_293K_1900s_clean_Mbar.txt")
s200F_11 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_296K_0.001s_clean_Mbar.txt")
s200F_12 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_473K_0.001s_clean_Mbar.txt")
s200F_13 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_473K_2200s_clean_Mbar.txt")
s200F_14 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_673K_0.001s_clean_Mbar.txt")
s200F_15 = np.loadtxt(path_to_data + "S200F_2018/Be_S200F_old_673K_2700s_clean_Mbar.txt")


z_1 = np.loadtxt(path_to_data + "Z-machine/Be_Brown17GPa.txt")
z_2 = np.loadtxt(path_to_data + "Z-machine/Be_Brown25GPa.txt")
z_3 = np.loadtxt(path_to_data + "Z-machine/Be_Brown33GPa.txt")
z_4 = np.loadtxt(path_to_data + "Z-machine/Be_Brown43GPa.txt")
z_5 = np.loadtxt(path_to_data + "Z-machine/Be_Brown90GPa.txt")



z_1[:,1] = z_1[:,1] / 1e5 #convert to Mbar
z_2[:,1] = z_2[:,1] / 1e5 #convert to Mbar
z_3[:,1] = z_3[:,1] / 1e5 #convert to Mbar
z_4[:,1] = z_4[:,1] / 1e5 #convert to Mbar
z_5[:,1] = z_5[:,1] / 1e5 #convert to Mbar


rmi = np.loadtxt(path_to_data + "CUI_Be_RMI") ###not in tame_impala
rmi[:,1] = rmi[:,1] / 1e5 #convert to Mbar
edot_rmi = 10**7
temp_rmi = 298

# plt.plot(rmi[:,0], rmi[:,1])
# plt.show()

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



dat_all = [dat0, dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9,
           s200F_1, s200F_2, s200F_3, s200F_4, s200F_5,
           s200F_6, s200F_7, s200F_8, s200F_9, s200F_10,
           s200F_11, s200F_12, s200F_13, s200F_14, s200F_15,
           rmi, rmi, rmi, rmi, rmi]
temps = [temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9,
         temps200F_1, temps200F_2, temps200F_3, temps200F_4, temps200F_5,
         temps200F_6, temps200F_7, temps200F_8, temps200F_9, temps200F_10,
         temps200F_11, temps200F_12, temps200F_13, temps200F_14, temps200F_15,
         temp_rmi, temp_rmi, temp_rmi, temp_rmi, temp_rmi] 
edots = [edot0, edot1, edot2, edot3, edot4, edot5, edot6, edot7, edot8, edot9,
        edots200F_1, edots200F_2, edots200F_3, edots200F_4, edots200F_5,
        edots200F_6, edots200F_7, edots200F_8, edots200F_9, edots200F_10,
        edots200F_11, edots200F_12, edots200F_13, edots200F_14, edots200F_15,
        edot_rmi, edot_rmi, edot_rmi, edot_rmi, edot_rmi]

#dat_all_z = [z_1, z_2, z_3, z_4, z_5]
dat_all_z = [np.repeat(z_1[0,:].reshape(1,-1), 100, axis = 0), 
             np.repeat(z_2[0,:].reshape(1,-1), 100, axis = 0), 
             np.repeat(z_3[0,:].reshape(1,-1), 100, axis = 0), 
             np.repeat(z_4[0,:].reshape(1,-1), 100, axis = 0), 
             np.repeat(z_5[0,:].reshape(1,-1), 100, axis = 0)]
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


plt.close('all')

fig,ax=plt.subplots(1,2,figsize=(12,5), sharex = False, sharey = False)
fig.suptitle('Figure 1: Observed Stress-Strain Curves \n and Experimental Strain Rates/Temperatures')

for j in range(len(dat_all)):
    if edots[j] <= 1:
        ax[0].plot(dat_all[j][:,0], dat_all[j][:,1], c = 'blue')
    else:
        ax[0].plot(dat_all[j][:,0], dat_all[j][:,1], c = 'purple')
    
for j in range(len(dat_all_z)):
    ax[0].scatter(dat_all_z[j][0,0], dat_all_z[j][0,1], c = 'red')
ax[0].set_ylabel('Stress')
ax[0].set_xlabel('Strain')
#ax[0].set_title('Figure 1: Observed Stress-Strain Curves')

ax[1].scatter(np.log10(np.asarray(edots)[np.asarray(edots)<=1]), np.asarray(temps)[np.asarray(edots)<=1], c = 'blue', label = 'QS, Low Strain Rate')
ax[1].scatter(np.log10(np.asarray(edots)[np.asarray(edots)>1]), np.asarray(temps)[np.asarray(edots)>1], c = 'purple', label = 'SHPB/RMI, Higher Strain Rate')
ax[1].scatter(np.log10(np.asarray(edots_z)), np.asarray(temps_z), c = 'red', label = 'Z Machine')
ax[1].set_xlabel('log10(Strain Rate) (1/s)')
ax[1].set_ylabel('Temperature (K)')
ax[1].legend()
#ax[1].set_title('Figure 2: Experimental Strain Rates and Temperatures')
plt.savefig(path_to_plots + 'observed_data_rmi.png')
plt.show()



#####################
### Get Constants ###
#####################
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
    'm'      : 2 #ADDED
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
    'lgammas' : (-2,17), #ADDED
    }

def constraints_ptw(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good

########################
### Define Model Run ###
########################

class BGP_PW_Shear_Modulus_Fixed(impala.physics.BaseModel):
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

        gnow[np.where(temp >= tmelt)] = 0.
        gnow[np.where(gnow < 0)] = 0.

        #if temp >= tmelt: gnow = 0.0
        #if gnow < 0.0:    gnow = 0.0
        return gnow

impala.physics.BGP_PW_Shear_Modulus_Fixed = BGP_PW_Shear_Modulus_Fixed




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
        #print(tmeltnow)
        return tmeltnow    
impala.physics.Cubic_Melt_Temperature=Cubic_Melt_Temperature

from scipy.special import erf


class PTWbp_Yield_Stress(impala.physics.BaseModel):
    params = ['theta','p','s0','sInf','kappa','lgamma','y0','yInf','y1', 'y2', 'lgammas']#, 'm']
    consts = ['rho0', 'beta', 'matomic', 'chi','m']
    #@profile
    def value(self, edot):
        """
        function used to define PTWb flow stress model
        arguments are:
            - edot: scalar, strain rate
            - material: an instance of MaterialModel class
        returns the flow stress at the current material state
        and specified strain rate
        """
        mp    = self.parent.parameters
        eps   = self.parent.state.strain
        temp  = self.parent.state.T
        tmelt = self.parent.state.Tmelt
        shear = self.parent.state.G

        good = (
            (mp.sInf < mp.s0) * (mp.yInf < mp.y0) * (mp.y0 < mp.s0)   *
            (mp.yInf < mp.sInf) * (mp.m > 0) * (mp.y1 > mp.s0) * (mp.y2 > mp.beta)
        )

        # lgammas=log(gammas)>-inf because gammas>0
        if np.any(~good):
            raise print('PTWb bad val')

        #convert to 1/s strain rate since PTW rate is in that unit
        edot = edot * 1.0E6

        t_hom = temp / tmelt

        afact = (4.0 / 3.0) * np.pi * mp.rho0 / mp.matomic ##### LAUREN EDITED!!!!!!!!!
        #ainv is 1/a where 4/3 pi a^3 is the atomic volume
        ainv  = afact ** (1.0 / 3.0)

        #transverse wave velocity up to units
        xfact = np.sqrt ( shear / mp.rho0 )
        #PTW characteristic strain rate [ 1/s ]
        xiDot = 0.5 * ainv * xfact * pow(6.022E29, 1.0 / 3.0) * 1.0E4
        # print('here')

        argErf = mp.kappa * t_hom * (mp.lgamma + np.log( xiDot / edot ))
        # print(mp.lgammas + np.log( edot / xiDot ))
        # print((mp.lgammas + np.log( edot / xiDot )).min())
        # print((mp.lgammas + np.log( edot / xiDot )).max())

        #relativistic = (1-np.exp((mp.lgammas + np.log( edot / xiDot )))**2)**mp.m
        relativistic = (1-((np.exp(mp.lgammas) * edot / xiDot )**2))**mp.m
        # print(relativistic.min())
        # print(relativistic.max())
        relativistic[relativistic<0.1] = 0.1
        relativistic[relativistic>10] = 10

        saturation1 = mp.s0 - ( mp.s0 - mp.sInf ) * erf( argErf )
        saturation2 = mp.s0 * np.exp(mp.beta * (-mp.lgamma + np.log(edot / xiDot))) / relativistic

        sat_cond = saturation1 > saturation2
        tau_s = np.copy(saturation2)
        tau_s[np.where(sat_cond)] = saturation1[sat_cond]


        ### JeeYeon's version
        ayield = mp.y0 - ( mp.y0 - mp.yInf ) * erf( argErf )
        byield = mp.y1 * np.exp( -mp.y2*(mp.lgamma + np.log( xiDot / edot ))) / relativistic
        y_cond = ayield > byield
        tau_y = np.copy(byield)
        tau_y[np.where(y_cond)] = ayield[y_cond]


        ### Lauren added this!!!!!
        small = 1.0e-10
        tau_y[tau_y > tau_s] = tau_s[tau_y > tau_s] - small
        

        scaled_stress = tau_s
        ind = np.where((mp.p > small) * (np.abs(tau_s - tau_y) > small))
        eArg1 = (mp.p * (tau_s - tau_y) / (mp.s0 - tau_y))[ind]
        eArg2 = (eps * mp.p * mp.theta)[ind] / (mp.s0 - tau_y)[ind] / (np.exp(eArg1) - 1.0) # eArg1 already subsetted by ind
        if (np.any((1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2)) <= 0) or \
                np.any(np.isinf(1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2)))):
            print('bad')
        # term_in_log = 1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2)
        # term_in_log[np.isinf(term_in_log)] = np.max(term_in_log[~np.isinf(term_in_log)])
        # term_in_log[term_in_log <= 0] = 1e-10
        theLog = np.log(1.0 - (1.0 - np.exp(- eArg1)) * np.exp(-eArg2))
        scaled_stress[ind] = (tau_s[ind] + ( mp.s0[ind] - tau_y[ind] ) * theLog / mp.p[ind] )
        ind2 = np.where((mp.p <= small) * (tau_s>tau_y))
        scaled_stress[ind2] = (
            + tau_s[ind2]
            - (tau_s - tau_y)[ind2] * np.exp(- eps[ind2] * mp.theta[ind2] / (tau_s - tau_y)[ind2])
            )
        # should be flow stress in units of Mbar
        out = scaled_stress * shear * 2.0
        out[np.where(~good)] = -999.
        return out
    
impala.physics.PTWbp_Yield_Stress = PTWbp_Yield_Stress


model_pool_ptw = sc.ModelMaterialStrength(temps=np.array(temps), 
        edots=np.array(edots)*1e-6, 
        consts=consts_ptw, 
        strain_histories=[strain_hist_list[j][inds_subsampled[j]] for j in range(len(dat_all))],  #Note: implementing subsampling!!!!
        flow_stress_model='PTWbp_Yield_Stress',
        melt_model= 'Cubic_Melt_Temperature',
        shear_model= 'BGP_PW_Shear_Modulus_Fixed',
        specific_heat_model='Cubic_Specific_Heat', 
        density_model='Cubic_Density',
        pool=True, s2='gibbs')


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
    'tm3': -102.09747419,
    'm'      : 2 #ADDED
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
        flow_stress_model='PTWbp_Yield_Stress',
        melt_model= 'Cubic_Melt_Temperature', #same as main
        shear_model= 'BGP_PW_Shear_Modulus_Fixed',
        specific_heat_model='Constant_Specific_Heat', #different from main
        density_model='Constant_Density_Custom', #different from main
        pool=True, s2='gibbs') for j in range(len(dat_all_z))]



#####################
### Set up Models ###
#####################

s2_ind = np.hstack([[j]*len(np.array(dat_all[j])[inds_subsampled[j],1]) for j in range(len(dat_all))])

setup_pool_ptw = sc.CalibSetup(bounds_ptw, constraints_ptw)
setup_pool_ptw.addVecExperiments(yobs=np.hstack([np.array(dat_all[j])[inds_subsampled[j],1] for j in range(len(dat_all))]), #Note: implementing subsampling!!!!
        model=model_pool_ptw, 
        sd_est=np.array([0.0005]*len(dat_all)),  
        s2_df=np.array([50]*len(dat_all)), 
        s2_ind=s2_ind,  #Note: implementing subsampling!!!!
        theta_ind=s2_ind)  #Note: implementing subsampling!!!!
for j in range(len(dat_all_z)):
    setup_pool_ptw.addVecExperiments(yobs=np.array(dat_all_z[j])[:,1], #Note: implementing subsampling!!!!
        model=model_z[j], 
        sd_est=np.array([0.0005]),  
        s2_df=np.array([50]), 
        s2_ind=[0]*len(np.array(dat_all_z[j])[:,1]),  #Note: implementing subsampling!!!!
        theta_ind=[0]*len(np.array(dat_all_z[j])[:,1]))  #Note: implementing subsampling!!!!
setup_pool_ptw.setTemperatureLadder(1.05**np.arange(50), start_temper=2000) 
setup_pool_ptw.setMCMC(nmcmc=40000, nburn=5000, thin=1, decor=100, start_tau_theta=-4.)



###########################
### Perform Calibration ### 
###########################

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

### Run Calibration 
np.seterr(divide = 'ignore')
np.seterr(invalid = 'ignore')
np.seterr(under = 'ignore')  #FloatingPointError: divide by zero encountered in log
np.seterr(over = 'ignore')  #FloatingPointError: divide by zero encountered in log
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

uu = np.arange(25000, 40000, 10)
n_exp = len(np.unique(setup_pool_ptw.s2_ind[0]))
n_exp_z = len(dat_all_z)
s2_inds = setup_pool_ptw.s2_ind[0]

### Trace Plot
pp.parameter_trace_plot(np.asarray(df),ylim=[0,1])
plt.savefig(path_to_plots + 'trace.png')


### KDE Pairs
# pairs_kde(df.loc[uu])
# plt.savefig(path_to_plots + 'pairs.png')


### Prediction Plots 
THETA_Y = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.array(pd.DataFrame(sc.tran_unif(np.asarray(df)[uu,:],setup_pool_ptw.bounds_mat, setup_pool_ptw.bounds.keys()).values())).T)['outcome_draws']
QUANTS_THETA_Y = [np.quantile(THETA_Y[j],[0.025,0.5,0.975],axis=0) for j in range(len(THETA_Y))]
for exp_ind in range(n_exp):
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

for j in range(1,6):
    fig,ax=plt.subplots(1,1,figsize=(6,4), sharey = False)    
    ax.fill_between(setup_pool_ptw.models[j].meas_strain_histories[0], QUANTS_THETA_Y[j][0,:].flatten(), QUANTS_THETA_Y[j][2,:].flatten(), color = 'pink', zorder = 1)            
    ax.plot(setup_pool_ptw.models[j].meas_strain_histories[0],QUANTS_THETA_Y[j][1,:].flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'Pooled Prediction')
    ax.scatter(setup_pool_ptw.models[j].meas_strain_histories[0],setup_pool_ptw.ys[j], color = 'black', zorder = 3, label = 'Data')
    ax.set_ylabel('Stress')
    ax.set_xlabel('Strain')
    ax.title.set_text('Prediction for Experiment '+str(j))
    ax.legend()
    plt.savefig(path_to_plots + 'experiment_zmachine_' +str(j)+'.png')



################################
### Output "Best" Parameters ###
################################

from scipy.stats import qmc
l_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,0]
u_bounds = np.array(pd.DataFrame(setup_pool_ptw.bounds.values()))[:,1]
parent_scaled = qmc.scale(np.array(df)[uu,:], l_bounds, u_bounds)

parent_scaled_native = pd.DataFrame(parent_scaled, columns = setup_pool_ptw.bounds.keys())
parent_scaled_native.to_csv(path_to_results + 'theta_draws_native.csv',index=False)


# g = sns.PairGrid(pd.DataFrame(parent_scaled_native), corner = True)
# g.map_lower(sns.histplot)
# #g.map_lower(sns.kdeplot, fill=True)
# g.map_diag(sns.histplot, kde=True)
# for ax in g.axes.flatten():
#     if ax:
#         for tick in ax.get_xticklabels():
#             tick.set_rotation(45)  # Rotate labels by 45 degrees
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05) # Adjust values as needed     
# plt.savefig(path_to_plots + 'pairs.png')


BEST = dict()
SSE = dict()
MAPE = dict()

### Parent_Median
pred_median = get_outcome_predictions_impala(setup_pool_ptw, theta_input = np.median(parent_scaled,axis = 0).reshape(1,-1))['outcome_draws']
median_sse = sum([((pred_median[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp)]) + sum([((pred_median[i]-setup_pool_ptw.ys[i])**2).mean(axis=1) for i in range(1,6)])
median_mape = np.sum([(np.abs(pred_median[0][0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(n_exp)])
median_mape = median_mape + np.sum([(np.abs(pred_median[j]-setup_pool_ptw.ys[j])/setup_pool_ptw.ys[j]).mean(axis=1) for j in range(1,6)])
median_mape = 100*(median_mape/(n_exp + n_exp_z))
BEST['parent_median'] = np.median(parent_scaled,axis = 0)
SSE['parent_median'] = median_sse
MAPE['parent_median'] = median_mape

### Best Draw 
parent_sse = sum([((THETA_Y[0][:,np.where(s2_inds == i)[0]]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])**2).mean(axis=1) for i in range(n_exp)]) +  np.sum(np.hstack([((THETA_Y[i]-setup_pool_ptw.ys[i])**2).mean(axis=1).reshape(-1,1) for i in range(1,6)]), axis = 1)
pred_minsse = np.hstack([THETA_Y[0][np.where(parent_sse == parent_sse.min()),np.where(s2_inds == i)[0]] for i in range(n_exp)]).reshape(1,-1)
parent_mape = np.sum([(np.abs(pred_minsse[0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(n_exp)])
parent_mape = parent_mape + np.sum(np.hstack([(np.abs(THETA_Y[j][np.where(parent_sse == parent_sse.min())]-setup_pool_ptw.ys[j])/setup_pool_ptw.ys[j]).mean(axis=1) for j in range(1,6)]))
parent_mape = 100*(parent_mape/(n_exp + n_exp_z))
BEST['parent_minsse'] = parent_scaled[np.where(parent_sse == parent_sse.min())[0],:].flatten()
SSE['parent_minsse'] = np.array([parent_sse.min()])
MAPE['parent_minsse'] = parent_mape


parent_mape_excludez = np.sum([(np.abs(pred_minsse[0,np.where(s2_inds == i)]-setup_pool_ptw.ys[0][np.where(s2_inds == i)])/setup_pool_ptw.ys[0][np.where(s2_inds == i)]).mean(axis=1) for i in range(n_exp)])
parent_mape_excludez = 100*(parent_mape_excludez/(n_exp ))

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
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_median).flatten(), color = 'purple', zorder = 2, linewidth = 2, label = 'Posterior Median')
ax.plot(np.arange(0,len(OBS)),np.hstack(pred_minsse).flatten(), color = 'red', zorder = 2, linewidth = 2, label = 'Best SSE')
#ax.plot(np.arange(0,len(OBS)),np.hstack(pred_map).flatten(), color = 'green', zorder = 3, linewidth = 2, label = 'MAP')
ax.scatter(np.arange(0,len(OBS)),OBS, color = 'black', zorder = 4, label = 'Data')
ax.set_ylabel('Stress')
ax.set_xlabel('Observation Index')
ax.title.set_text('Prediction for All Experiments')
ax.legend()
plt.savefig(path_to_plots + 'best_all.png')


