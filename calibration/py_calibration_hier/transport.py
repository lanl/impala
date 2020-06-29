""" Transport Class Definitions

Transports assemble input variables using input paths.
Transports are directly supplied to a SubModel class at initialization.  E.g.,

    SubModelFP(TransportFP(**{...}))

For Hopkinson Bar, this needs to include the path to the empirical stress strain
curves, and starting temperatures, maximum strain, strain rate, and resolution

For Taylor Cylinder/Flyer plate, this needs to include the path to the design
matrix for simulations, the path to the simulated output matrix, annd the path
to actual observed values.

The order of columns in the design matrix needs to match the order of variables
specified by physical_models_c.MaterialModel
"""

import numpy as np
import pandas as pd
import warnings

# Transport Classes

class TransportHB(object):
    """ Hopkinson Bar Transport """
    data = None
    temp = None
    emax = None
    edot = None
    Nhist = None

    @staticmethod
    def import_strain_curve(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('asdf')
            data = pd.read_csv(path, skiprows = range(0,17)).values
        data[:,1] = data[:,1] * 1.e-5 # this is going from MPa to Mbar?
        return data[1:,:]

    def __init__(self, path, temp, emax, edot, Nhist):
        self.data = self.import_strain_curve(path)
        self.temp = temp
        self.emax = emax
        self.edot = edot
        self.Nhist = Nhist
        return

class TransportTC(object):
    """ Taylor Cylinder Transport """
    X = None
    Y = None
    Y_actual = None

    def __init__(self, path_x, path_y, path_y_actual):
        self.X = pd.read_csv(path_x, sep = '\s+|\t|,|;',
                                engine = 'python').values
        self.Y = pd.read_csv(path_y, sep = '\s+|\t|,|;',
                                engine = 'python').values
        self.Y_actual = pd.read_table(path_y_actual, sep = '\s+|\t|,|;',
                                        engine = 'python').values
        return

class TransportFP(object):
    """ Flyer Plate Transport """
    X = None
    Y = None
    Y_actual = None

    def __init__(self, path_x, path_y, path_y_actual):
        self.X = pd.read_csv(path_x, sep = '\s+|\t|,|;',
                                engine = 'python').values
        self.Y = pd.read_csv(path_y, sep = '\s+|\t|,|;',
                                engine = 'python').values
        self.Y[:,1::2] = self.Y[:,1::2] * 1e4 # Fixing unit discrepancy between
                                              # simulation outputs and observations
        self.Y_actual = pd.read_table(path_y_actual, sep = '\s+|\t|,|;',
                                        engine = 'python').values
        return

# EOF
