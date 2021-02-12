#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:36:41 2020

@author: dfrancom
"""

import sqlite3 as sq

con = sq.connect('./data/data_Ti64.db')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

cursor.execute("SELECT * FROM meta;")
dat = cursor.fetchall()

import pandas as pd

datdf = pd.DataFrame(dat)
datdf.to_csv("datti64.csv")


con = sq.connect('/Users/dfrancom/Desktop/Ti64/res_ti64_clst.db')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


cursor.execute(" SELECT sim_input, sim_output FROM meta where table_name = 'data_197';")
dat = cursor.fetchall()


cursor.execute(" SELECT * FROM 'pca_output_197';")
dat = cursor.fetchall()

con = sq.connect('/Users/dfrancom/Desktop/Ti64/res_ti64_clst.db')
cursor = con.cursor()
cursor.execute("SELECT * FROM delta;")
aa = cursor.fetchall()

aadf = pd.DataFrame(aa)
aadf.to_csv("deltati64.csv")

data_query = " SELECT * FROM {}; "
meta_query = " SELECT sim_input, sim_output FROM meta where table_name = '{}'; "
emui_query = " SELECT * FROM {}; "
emuo_query = " SELECT * FROM {}; "
emu_inputs, emu_outputs = list(cursor.execute(meta_query.format("data_198")))[0]