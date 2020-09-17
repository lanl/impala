import sqlite3 as sql
import pandas as pd
import numpy as np
from random import sample

def read_posterior(path, nSamp = 500):
    # Declare SQL connection and cursor
    conn = sql.connect(path)
    cursor = conn.cursor()
    # get the experiment table names
    tables = [x[0] for x in cursor.execute("SELECT calib_name FROM meta;")]
    # Get the parameter names
    param = [
        col[1]
        for col
        in cursor.execute("PRAGMA TABLE_INFO({});".format(tables[0]))
        ]
    nParm = len(param) # number of parameters
    # Length of output tables (number of posterior samples)
    nPost = list(cursor.execute("SELECT COUNT() FROM {};".format(tables[0])))[0][0]
    # Fix which indices of posterior samples we'll be looking at
    keep_idx = sample(list(range(nPost)), nSamp)
    # Declare output array
    output = np.empty((len(tables), nSamp, nParm))

    for i, table in enumerate(tables):
        # gather records from the data
        temp = np.array(list(cursor.execute("SELECT * FROM {};".format(table))))
        # subset the records to just the previosly selected indices
        output[i] = temp[keep_idx]

    conn.close()
    return output

if __name__ == '__main__':
    pass

# EOF
