import numpy as np
import time
from numpy import array, float64
from experiment import Experiment_PCA
import sqlite3 as sql
import pickledBass as pb


if __name__ == '__main__':
    path = './data/data_copper.db'
    conn = sql.connect(path)
    cursor = conn.cursor()
    table_name = 'data_7'
    model_args = {'flow_stress_model'   : 'PTW', 'shear_modulus_model' : 'Simple'}

    exp = Experiment_PCA(conn, cursor, table_name, model_args)
    xmean = exp.Xemu.mean(axis = 0) # trial input for predict function
    t     = exp.tuple # generates a random MCMC sample, declares a tuple as per instruction
    preds = pb.predictPCA(xmean, t.samples, t.tbasis, t.ysd, t.ymean, t.bounds)


    pass

# EOF
