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
    #xps = xps[:40]

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



def load_Ti64_pca(curs, conn):
    global ti64_curr_id

    sim_path_base = '~/impala_data/'
    obs_path_base = '~/git/immpala/data/ti-6al-4v/Data/'

    xps = []

    ########################################################################
    ## read in flyers
    simi = pd.read_csv(sim_path_base + 'Ti64_Flyer_TrainingSets/Ti64.design.ptw.1000.txt', sep=" ")  # simulated x, same for all flyers


    # # Ti64_Dandekar2000_VISAR_847_398
    # real = pd.read_csv(obs_path_base + 'FlyerPlate/Dandekar2000/Dandekar2000_Fig2-300K-398m_s-Ti64.csv',skiprows = range(0,25))
    #
    # sim_path = sim_path_base + 'Ti64_Flyer_TrainingSets/Ti64_Dandekar2000_VISAR_847_398/Results/'
    # files = glob.glob(os.path.join(os.path.expanduser(sim_path), 'Ti64*'))
    #
    # xx = real['Time (us)'] #np.linspace(1.8, 3., 200)
    # simo = np.zeros([simi.shape[0], len(xx)]) # simulated Y matrix
    # k = 0
    # for file in files:
    #     data = pd.read_csv(file)
    #     simo[k,:] = np.interp(xx,data['Time'],data['velocity'])
    #     k += 1
    #
    # xps.append({'real': real[' velocity (m/s)']/1000, # divide by 1000 for unit conversion (m/s -> ?)
    #             'simi': simi,
    #             'simo': pd.DataFrame(simo),
    #             'fname': sim_path})


    ##

    ########################################################################
    ## read in taylors
    simi = pd.read_csv(sim_path_base + 'TaylorTi64_Full/Ti64.design.ptw.1000.txt', sep=" ")

    # Test                     |      Deformed length (y-coordinate, cm)
    # Yu2011_T1        |      2.4493
    # Yu2011_T2        |      2.4095
    # Yu2011_T3        |      2.3901
    # Yu2011_T4        |      2.3702

    # taylor_yu2011_T1
    real = np.array([2.4493, 2.4095, 2.3901, 2.3702])
    simo = np.zeros([simi.shape[0], len(real)])
    jj = 0
    for exp in range(4):
        sim_path = sim_path_base + 'TaylorTi64_Full/taylor_yu2011_T' + str(exp+1) + '/Results/'
        files = glob.glob(os.path.join(os.path.expanduser(sim_path), 'taylor*'))
        for file in files:
            k = int((file.split('_')[-1]).split('.')[0]) - 1
            data = pd.read_csv(file)
            simo[k,jj] = data['y(cm)'].iloc[-1]
        jj += 1


    xps.append({'real': pd.DataFrame(real),
                'simi': simi,
                'simo': pd.DataFrame(simo),
                'fname': sim_path})

    ##



    for xp in xps:
        ti64_curr_id += 1
        table_name = 'data_{}'.format(ti64_curr_id)
        emu_iname = 'pca_input_{}'.format(ti64_curr_id)
        emu_oname = 'pca_output_{}'.format(ti64_curr_id)

        xp['real'].to_sql(table_name, conn)
        xp['simi'].to_sql(emu_iname, conn)
        xp['simo'].to_sql(emu_oname, conn)
        curs.execute(pca_meta_insert, ('pca', table_name, emu_iname, emu_oname))

    conn.commit()

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
    load_Ti64_pca(cursor, connection)
    # close the connection
    connection.commit()
    connection.close()
    return


if __name__ == '__main__':
    load_Ti64()


# EOF
