"""
Loads the data into the sqlite file
"""

import sqlite3 as sql
import numpy as np
import pandas as pd
import glob
import os
import string
import random


# Defining strings for creating and updating tables

# Meta Table Creation String;
#  Add additional meta information necessary to other experiment types to this
#  table.  Also need to add chemistry to this table.
meta_create = """CREATE TABLE meta (
    type TEXT,
    temperature REAL,
    edot REAL,
    emax REAL,
    pname TEXT,
    fname TEXT,
    table_name TEXT
    );
"""
shpb_data_create = """ CREATE TABLE {}(strain REAL, stress REAL); """
shpb_meta_insert = """ INSERT INTO
        meta(type, temperature, edot, emax, pname, fname, table_name)
        values (?,?,?,?,?,?,?);
        """
shpb_data_insert = """ INSERT INTO {}(strain, stress) values (?,?); """

def load_Ti64_shpb(curs):
    global ti64_curr_id
    shpb_emax = 0

    base_path = '../../data/ti-6al-4v/Data/SHPB'
    papers = [os.path.split(f)[1] for f in glob.glob(base_path + '/*')]

    xps = []

    # Load the transports
    for paper in papers:
        files = glob.glob(os.path.join(base_path, paper, '*.Plast.csv'))
        for file in files:
            f = open(file)
            lines = f.readlines()
            temp  = float(lines[1].split(" ")[2]) # Kelvin
            edot  = float(lines[2].split(" ")[3]) * 1e-6 # 1/s
            pname = os.path.split(os.path.dirname(file))[1]
            fname = os.path.splitext(os.path.basename(file))[0]
            data  = pd.read_csv(file, skiprows = range(0,17)).values
            data[:,1] = data[:,1] * 1e-5 # Transform from MPa to Mbar
            xps.append({'data' : data[1:], 'temp' : temp, 'edot' : edot,
                        'pname' : pname, 'fname' : fname})
            shpb_emax = max(shpb_emax, data[:,0].max())

    # For each experiment:
    for xp in xps:
        # Set the ID
        ti64_curr_id += 1
        table_name = 'data_{}'.format(ti64_curr_id)
        # Create the relevant data table
        curs.execute(shpb_data_create.format(table_name))
        # Create the corresponding line in the meta table
        curs.execute(
            shpb_meta_insert,
            ('shpb', xp['temp'], xp['edot'], shpb_emax,
                xp['pname'], xp['fname'], table_name)
            )
        # Fill the data tables
        curs.executemany(
            shpb_data_insert.format(table_name),
            xp['data'].tolist()
            )
    return

def load_Ti64_tc(curs):
    global ti64_curr_id
    return

def load_Ti64_fp(curs):
    global ti64_curr_id
    return

def load_Ti64():
    global ti64_curr_id, ti64_shpb_emax
    ti64_curr_id = 0
    ti64_shpb_emax = 0

    # Clear the old database
    if os.path.exists('./data_Ti64.db'):
        os.remove('./data_Ti64.db')

    # Start the SQLite connection
    connection = sql.connect('./data_Ti64.db')
    cursor = connection.cursor()
    cursor.execute(meta_create)
    connection.commit()
    # Load the data
    current_id = 0
    load_Ti64_shpb(cursor)
    load_Ti64_tc(cursor)
    load_Ti64_fp(cursor)
    # close the connection
    connection.commit()
    connection.close()
    return

def load_copper_shpb(curs):
    global copper_curr_id
    shpb_emax = 0

    paths_hb = [
        './copper/CuRT203.txt',
        './copper/Cu20203.txt',
        './copper/Cu40203.txt',
        './copper/Cu60203.txt',
        './copper/CuRT10-1.SRC.txt',
        './copper/CuRT10-3.SRC.txt',
        ]
    temps_hb = np.array([298., 473., 673., 873., 298., 298.])
    edots_hb = np.array([2000., 2000., 2000., 2000., 0.1, 0.001]) * 1e-6
    fnames_hb = [os.path.basename(path) for path in paths_hb]

    xps = []

    for i in range(len(paths_hb)):
        data = (pd.read_csv(paths_hb[i], sep = '\s+').values)
        data[:,1] = data[:,1] * 1e-5
        temp = temps_hb[i]
        edot = edots_hb[i]
        fname = fnames_hb[i]
        pname = ''
        shpb_emax = max(shpb_emax, data[:,0].max())
        xps.append({'data' : data[1:], 'temp' : temp, 'edot' : edot,
                    'pname' : pname, 'fname' : fname})

    for xp in xps:
        copper_curr_id += 1
        table_name = 'data_{}'.format(copper_curr_id)
        curs.execute(shpb_data_create.format(table_name))
        curs.execute(
            shpb_meta_insert,
            ('shpb', xp['temp'], xp['edot'], shpb_emax,
                xp['pname'], xp['fname'], table_name)
            )
        curs.executemany(
            shpb_data_insert.format(table_name),
            xp['data'].tolist()
            )
    return

def load_copper_tc(curs):
    global copper_curr_id
    return

def load_copper_fp(curs):
    global copper_curr_id
    return

def load_copper():
    global copper_curr_id
    copper_curr_id = 0
    shpb_emax = 0

    if os.path.exists('./data_copper.db'):
        os.remove('./data_copper.db')

    connection = sql.connect('./data_copper.db')
    cursor = connection.cursor()
    cursor.execute(meta_create)
    connection.commit()

    load_copper_shpb(cursor)
    load_copper_tc(cursor)
    load_copper_fp(cursor)

    connection.commit()
    connection.close()
    return

if __name__ == '__main__':
    load_Ti64()
    load_copper()

# EOF
