import numpy as np
import time
import sqlite3 as sql
from numpy import array, float64
np.seterr(under = 'ignore')

import sm_dpcluster as sm
sm.POOL_SIZE = 8

ipath = './data/data_Ti64.db'
opath = './results/temp_Ti64/res_ti64_clst.db'
ppath = './plots'

if __name__ == '__main__':
    results = sm.ResultSummary(ipath, opath)
    results.plot_calibrated(ppath)
    results.cluster_by_strainrate(ppath)
    results.cluster_by_temperature(ppath)




# EOF
