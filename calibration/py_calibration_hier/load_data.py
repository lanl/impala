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
if True:
    meta_table_create = """CREATE TABLE {} (
        TEMPERATURE REAL,
        edot REAL,
        pname TEXT,
        fname TEXT,
        table_name TEXT
        );
    """
    data_table_create = """ CREATE TABLE {}(strain REAL, stress REAL); """
    meta_table_insert = """ INSERT INTO
        {}(temperature, edot, pname, fname, table_name)
        values (?,?,?,?,?);
        """
    data_table_insert = """ INSERT INTO {}(strain, stress) values (?,?); """

def random_string(length):
    s = ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(length)
        )
    return s

def load_ti64_shpb():
    conn = sql.connect('./data_Ti64.db')
    curs = conn.cursor()
    seen = set()

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

    # Delete extant relevant tables
    tables = list(curs.execute("select name from sqlite_master WHERE type = 'table';"))
    tables = [table[0] for table in tables if table[0].startswith('shpb_')]
    for table in tables:
        curs.execute('DROP TABLE {};'.format(table))

    conn.commit()

    # Create the meta table
    curs.execute(meta_table_create.format('shpb_meta'))
    conn.commit()

    # For each experiment:
    for xp in xps:
        # Randomize an experiment name
        table_name = 'shpb_' + random_string(8)
        while table_name in seen:
            table_name = 'shpb_' + random_string(8)

        seen.add(table_name)

        # Create the relevant data table
        curs.execute(data_table_create.format(table_name))
        # Create the corresponding line in the meta table
        curs.execute(
            meta_table_insert.format('shpb_meta'),
            (xp['temp'], xp['edot'], xp['pname'], xp['fname'], table_name)
            )
        # Fill the data tables
        curs.executemany(
            data_table_insert.format(table_name),
            xp['data'].tolist()
        )

    conn.commit()
    conn.close()
    return

def load_aluminium_shpb():
    pass

def load_copper_shpb():
    pass

if __name__ == '__main__':
    # Start the SQLite connection
    load_ti64_shpb()
    # pass



# EOF
