#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:11:24 2020

@author: dfrancom
"""
import numpy as np
import dill

with open("store_results.dill", "rb") as dill_file:
    res_list = dill.load(dill_file)


def unnormalize(z, bounds):
    return z * (bounds[:,1] - bounds[:,0]) + bounds[:,0]


bounds = res_list[0][1].chains[0].bounds
pnames = ",".join(res_list[0][1].parameter_order)
path = "/home/dfrancom/git/immpala/data/ti-6al-4v/results/posterior_sep/cond_"
uu = list(range(0, 1750, 10))
for i in list(range(174)):
    temp = res_list[i][0][0][uu,:]
    parameters = np.apply_along_axis(unnormalize, 1, temp, bounds = bounds)
    np.savetxt(path + str(i) + '.csv', parameters, delimiter=",", header=pnames)
