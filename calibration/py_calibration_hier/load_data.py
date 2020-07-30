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
current_id = 0

def load_Ti64_shpb(conn):
    global current_id
    if True:
        meta_table_create = """CREATE TABLE {} (
            type TEXT,
            temperature REAL,
            edot REAL,
            pname TEXT,
            fname TEXT,
            table_name TEXT
            );
        """
        data_table_create = """ CREATE TABLE {}(strain REAL, stress REAL); """
        meta_table_insert = """ INSERT INTO
            {}(type, temperature, edot, pname, fname, table_name)
            values (?,?,?,?,?,?);
            """
        data_table_insert = """ INSERT INTO {}(strain, stress) values (?,?); """

    curs = conn.cursor()

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
            edot  = float(lines[2].split(" ")[3]) # 1/s
            pname = os.path.split(os.path.dirname(file))[1]
            fname = os.path.splitext(os.path.basename(file))[0]
            data  = pd.read_csv(file, skiprows = range(0,17)).values
            data[:,1] = data[:,1] * 1e-5 # Transform from MPa to Mbar
            xps.append({'data' : data[1:,:], 'temp' : temp, 'edot' : edot,
                        'pname' : pname, 'fname' : fname})

    # Create the meta table
    curs.execute(meta_table_create.format('meta'))
    conn.commit()

    # For each experiment:
    for xp in xps:
        # Set the ID
        current_id += 1
        table_name = 'shpb_{}'.format(current_id)
        # Create the relevant data table
        curs.execute(data_table_create.format(table_name))
        # Create the corresponding line in the meta table
        curs.execute(
            meta_table_insert.format('meta'),
            ('shpb', xp['temp'], xp['edot'],
                xp['pname'], xp['fname'], table_name)
            )
        # Fill the data tables
        curs.executemany(
            data_table_insert.format(table_name),
            xp['data'].tolist()
            )

    # Commit the writes
    conn.commit()
    return

def load_Ti64_tc(conn):
    global current_id
    return

def load_Ti64_fp(conn):
    global current_id
    return

if __name__ == '__main__':
    # Clear the old database
    if os.path.exists('./data_Ti64.db'):
        os.remove('./data_Ti64.db')

    # Start the SQLite connection
    connection = sql.connect('data_ti64.db')
    load_Ti64_shpb(connection)
    load_Ti64_tc(connection)
    load_Ti64_fp(connection)
    connection.close()


# EOF
