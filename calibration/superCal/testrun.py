# %%
import numpy as np
import pandas as pd
import sqlite3 as sq
# import pyBASS as pb
import impala
import models

#%%
## get meta data
con = sq.connect('./../py_calibration_hier/data/data_Ti64.db')
cursor = con.cursor()
cursor.execute("SELECT * FROM meta;")
meta_names = list(map(lambda x: x[0], cursor.description))
meta = pd.DataFrame(cursor.fetchall(),columns=meta_names)
edots = meta.edot[:196]
temps = meta.temperature[:196]

## get SHPB experimental data
dat_all = []
for i in range(196):
    ## get first datset
    cursor.execute("SELECT * FROM data_" + str(i + 1) + ";")
    dat_all.append(cursor.fetchall())


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
    'vel'   : (0.99,     1.01),
    }


# constraints: sInf < s0, yInf < y0, y0 < s0, yInf < sInf, s0 < y1, beta < y2
def cf(x, bounds):
    good = (x['sInf'] < x['s0']) * (x['yInf'] < x['y0']) * (x['y0'] < x['s0']) * (x['yInf'] < x['sInf']) * (x['s0'] < x['y1'])
    for k in list(bounds.keys()):
        good = good * (x[k] < bounds[k][1]) * (x[k] > bounds[k][0])
    return good



setup = impala.CalibSetup(bounds, cf)

yobs = np.hstack([np.array(v)[:,1] for v in dat_all])
#yobs = np.concatenate((np.array(dat_all[0])[:,1],np.array(dat_all[1])[:,1]))
#sh = [np.array(dat_all[0])[:,0], np.array(dat_all[1])[:,0]]
sh = [np.array(v)[:,0] for v in dat_all]
model = models.ModelPTW(np.array(meta.temperature[0:196]), np.array(meta.edot[0:196]), consts, sh)
sd_est = np.array([.001]*196)
s2_df = np.array([5]*196)
#s2_ind = np.array([0]*len(dat_all[0]) + [1]*len(dat_all[1]))

s2_ind = np.hstack([[v]*len(dat_all[v]) for v in list(range(196))])

setup.addVecExperiments(yobs, model, sd_est, s2_df, s2_ind)
setup.setTemperatureLadder(1.1**np.arange(100))
setup.setMCMC(30000,10000,1,100)

np.seterr(under='ignore')

#np.random.seed(11)

out = impala.calibPool(setup)


cf(impala.tran(out.theta[:,0,:], setup.bounds_mat, setup.bounds.keys()), setup.bounds)

out.count
out.count_decor
out.tau
np.sqrt(out.s2)

import matplotlib.pyplot as plt
plt.plot(out.theta[:,0,9])

plt.plot(np.sqrt(out.s2)[0,range(1,30000),0,1])

plt.plot(out.pred_curr[0][0,:])
plt.plot(yobs)


pred = setup.models[0].eval(impala.tran(out.theta[:,0,:], setup.bounds_mat, setup.bounds.keys()))
np.any(pred[:,0] == -999.)

plt.plot(pred[25000:30000,:].T,color='grey')
plt.plot(yobs,color='black')
plt.plot(pred[0:300,:].T)

plt.plot(pred[25010:25020,:].T)



plt.plot(pred[25000:30000,s2_ind==120].T,color='grey')
plt.plot(yobs[s2_ind==120],color='black')



