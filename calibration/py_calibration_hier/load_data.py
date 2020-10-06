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
    sim_input REAL,
    sim_output REAL,
    table_name TEXT
    );
"""
shpb_data_create = """ CREATE TABLE {}(strain REAL, stress REAL); """
shpb_meta_insert = """ INSERT INTO
        meta(type, temperature, edot, emax, pname, fname, table_name)
        values (?,?,?,?,?,?,?);
        """
shpb_data_insert = """ INSERT INTO {}(strain, stress) values (?,?); """
pca_meta_insert = """ INSERT INTO meta(type, table_name, sim_input, sim_output) values (?,?,?,?); """

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
            # shpb_emax = max(shpb_emax, data[:,0].max())

    # truncate to first 40 for runtime
    xps = xps[:40]

    for xp in xps:
        shpb_emax = max(shpb_emax, xp['data'][:,0].max())

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
            ('shpb', xp['temp'], xp['edot'], shpb_emax, xp['pname'], xp['fname'], table_name)
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
    if os.path.exists('./data/data_Ti64.db'):
        os.remove('./data/data_Ti64.db')

    # Start the SQLite connection
    connection = sql.connect('./data/data_Ti64.db')
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

def load_copper_shpb(curs, conn):
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

def load_copper_pca(curs, conn):
    global copper_curr_id

    xps = []
    xps.append({
        'realY' : './copper/outputs_real_fp.csv',
        'simX' : './copper/inputs_sim_fp_ptw.txt',
        'simY' : './copper/outputs_sim_fp_ptw.csv'
        })
    for xp in xps:
        copper_curr_id += 1
        table_name = 'data_{}'.format(copper_curr_id)
        emu_iname = 'pca_input_{}'.format(copper_curr_id)
        emu_oname = 'pca_output_{}'.format(copper_curr_id)

        real = pd.read_csv(xp['realY']) # first row
        simi = pd.read_csv(xp['simX'],sep=" ") # simulated x
        simo = pd.read_csv(xp['simY']) # simulated Y

        real.to_sql(table_name, conn)
        simi.to_sql(emu_iname, conn)
        simo.to_sql(emu_oname, conn)
        curs.execute(pca_meta_insert, ('pca', table_name, emu_iname, emu_oname))

    conn.commit()
    return

def load_copper_wpca(curs, conn):
    global copper_curr_id
    return

def load_copper():
    global copper_curr_id
    copper_curr_id = 0

    if os.path.exists('./data/data_copper.db'):
        os.remove('./data/data_copper.db')

    connection = sql.connect('./data/data_copper.db')
    cursor = connection.cursor()
    cursor.execute(meta_create)
    connection.commit()

    load_copper_shpb(cursor, connection)
    load_copper_pca(cursor, connection)
    load_copper_wpca(cursor, connection)

    connection.commit()
    connection.close()
    return

def load_Al5083_shpb(curs):
    global al5083_curr_id
    shpb_emax = 0
    base_path = '../../data/Al-5083/Stress-Strain_Data'
    files = [
        'Gray94_Al5083_S0_001T-196C.csv',
        'Gray94_Al5083_S0_001T125C.csv',
        'Gray94_Al5083_S0_1T125C.csv',
        'Gray94_Al5083_S2000T-196C.csv',
        'Gray94_Al5083_S2500T25C.csv',
        'Gray94_Al5083_S3000T100C.csv',
        'Gray94_Al5083_S3500T200C.csv',
        'Gray94_Al5083_S7000T25C.csv',
        ]
    paths = [os.path.join(base_path, file) for file in files]
    temps = np.array([ -196.,  125.,  125., -196., 25., 100., 200., 25.]) + 273.15
    edots = np.array([0.001, 0.001, 0.1, 2000, 2500, 3000, 3500, 7000,]) * 1.e-6

    xps = []
    for i in range(len(paths)):
        data = pd.read_csv(paths[i], skiprows = 2).values
        data[:,1] = data[:,1] * 1e-5
        temp = temps[i]
        edot = edots[i]
        fname = files[i]
        pname = ''
        shpb_emax = max(shpb_emax, data[:,0].max())
        xps.append({'data' : data, 'temp' : temp, 'edot' : edot,
                    'pname' : pname, 'fname' : fname})

    for xp in xps:
        al5083_curr_id += 1
        table_name = 'data_{}'.format(al5083_curr_id)
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

def load_Al5083_tc(curs):
    pass

def load_Al5083_fp(curs):
    pass

def load_Al5083():
    global al5083_curr_id
    al5083_curr_id = 0

    if os.path.exists('./data/data_Al5083.db'):
        os.remove('./data/data_Al5083.db')
    connection = sql.connect('./data/data_Al5083.db')
    cursor = connection.cursor()
    cursor.execute(meta_create)

    load_Al5083_shpb(cursor)
    load_Al5083_tc(cursor)
    load_Al5083_fp(cursor)

    connection.commit()
    connection.close()
    return

if __name__ == '__main__':
    # load_Ti64()
    load_copper()
    # load_Al5083()

# EOF
