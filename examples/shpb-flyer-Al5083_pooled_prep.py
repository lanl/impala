import numpy as np
from scipy.interpolate.interpolate import interp1d
import matplotlib.pyplot as plt


import os
path_104 = os.path.abspath('../../../Downloads/realTime-master-AlAl-data_trial5/AlAl/data_trial5/104S/')
files_104_unsorted = os.listdir(path_104)
order = [int(str.split(ff, "_")[1]) for ff in files_104_unsorted]
files_104 = [x for _, x in sorted(zip(order, files_104_unsorted))]
nfiles_104 = len(files_104)
files2_104 = [path_104 + '/' + files_104[i]  for i in range(len(files_104))]
dat_104 = [None] * nfiles_104
for i in range(nfiles_104):
    with open(files2_104[i]) as file: 
        temp = file.readlines()
        dat_104[i] = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])

path_105 = os.path.abspath('../../../Downloads/realTime-master-AlAl-data_trial5/AlAl/data_trial5/105S/')
files_105_unsorted = os.listdir(path_105)
order = [int(str.split(ff, "_")[1]) for ff in files_105_unsorted]
files_105 = [x for _, x in sorted(zip(order, files_105_unsorted))]
nfiles_105 = len(files_105)
files2_105 = [path_105 + '/' + files_105[i]  for i in range(len(files_105))]
dat_105 = [None] * nfiles_105
for i in range(nfiles_105):
    with open(files2_105[i]) as file: 
        temp = file.readlines()
        dat_105[i] = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])

path_106 = os.path.abspath('../../../Downloads/realTime-master-AlAl-data_trial5/AlAl/data_trial5/106S/')
files_106_unsorted = os.listdir(path_106)
order = [int(str.split(ff, "_")[1]) for ff in files_106_unsorted]
files_106 = [x for _, x in sorted(zip(order, files_106_unsorted))]
nfiles_106 = len(files_106)
files2_106 = [path_106 + '/' + files_106[i]  for i in range(len(files_106))]
dat_106 = [None] * nfiles_106
for i in range(nfiles_106):
    with open(files2_106[i]) as file: 
        temp = file.readlines()
        dat_106[i] = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])




xrange_104 = [0.75, 1.3]
xrange_105 = [1.2, 2.2]
xrange_106 = [0.65, 1.3]
M = 200
n = 1000
xx104 = np.linspace(xrange_104[0], xrange_104[1], M)
xx105 = np.linspace(xrange_105[0], xrange_105[1], M)
xx106 = np.linspace(xrange_106[0], xrange_106[1], M)
xx_all = [xx104, xx105, xx106]

sims_all = np.empty([3, n, M]) # 3 datasets, 1000 samples, 200 points on curve

for i in range(n):
    ifunc = interp1d(dat_104[i][:,1], dat_104[i][:,3], kind = 'cubic')
    sims_all[0, i, :] = ifunc(xx104)

    ifunc = interp1d(dat_105[i][:,1], dat_105[i][:,3], kind = 'cubic')
    sims_all[1, i, :] = ifunc(xx105)

    ifunc = interp1d(dat_106[i][:,1], dat_106[i][:,3], kind = 'cubic')
    sims_all[2, i, :] = ifunc(xx106)



with open('./../data/Al-5083/flyer_data/Data_S104S.txt') as file: 
    temp = file.readlines()
    obs1 = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])

with open('./../data/Al-5083/flyer_data/Data_S105S.txt') as file: 
    temp = file.readlines()
    obs2 = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])

with open('./../data/Al-5083/flyer_data/Data_S106S.txt') as file: 
    temp = file.readlines()
    obs3 = np.vstack([np.float_(str.split(temp[i])) for i in range(2,len(temp))])

obs_all = np.empty([3, M])
ifunc = interp1d(obs1[:,1], obs1[:,0]*1e-4, kind = 'cubic')
obs_all[0] = ifunc(xx104)

ifunc = interp1d(obs2[:,1]-.4, obs2[:,0]*1e-4, kind = 'cubic')
obs_all[1] = ifunc(xx105)

ifunc = interp1d(obs3[:,0]-2.75, obs3[:,1]*1e-4, kind = 'cubic')
obs_all[2] = ifunc(xx106)

plt.plot(sims_all[0].T, color='lightgrey')
plt.plot(obs_all[0])
plt.show()

plt.plot(sims_all[1].T, color='lightgrey')
plt.plot(obs_all[1])
plt.show()

plt.plot(sims_all[2].T, color='lightgrey')
plt.plot(obs_all[2])
plt.show()

# let the obs have large time shift discrepancy

np.savetxt("./../data/Al-5083/flyer_data/sims104.csv", sims_all[0], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/sims105.csv", sims_all[1], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/sims106.csv", sims_all[2], delimiter=",")

np.savetxt("./../data/Al-5083/flyer_data/obs104.csv", obs_all[0], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/obs105.csv", obs_all[1], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/obs106.csv", obs_all[2], delimiter=",")

np.savetxt("./../data/Al-5083/flyer_data/xsims104.csv", xx_all[0], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/xsims105.csv", xx_all[1], delimiter=",")
np.savetxt("./../data/Al-5083/flyer_data/xsims106.csv", xx_all[2], delimiter=",")

#sim_inputs = np.genfromtxt('./../data/Al-5083/flyer_data/sim_input.csv', delimiter=',', skip_header=1)


