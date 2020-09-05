from physical_models_c import MaterialModel
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf, erfinv
from collections import namedtuple
from math import ceil, sqrt, pi, log
import methodtools as mt
import functools as ft
import numpy as np
import pandas as pd
import sqlite3 as sql
import os

@ft.lru_cache(maxsize = 128)
def cholesky_inversion(Sigma_as_tuple):
    Sigma = np.array(Sigma_as_tuple)
    return cho_solve(cho_factor(Sigma), np.eye(Sigma.shape[0]))

class Transformer(object):
    """ Class to hold transformations (logit, probit) """
    parameter_order = []
    bounds = None

    def normalize(self, x):
        """ Transform real scale to 0-1 scale """
        return (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])

    def unnormalize(self, z):
        """ Transform 0-1 scale to real scale """
        return z * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]

    @staticmethod
    def probit(x):
        """
        Probit Transformation:
        For x bounded 0-1, y is unbounded; between -inf and inf
        """
        return sqrt(2.) * erfinv(2 * x - 1)

    @staticmethod
    def invprobit(y):
        """
        Inverse Probit Transformation
        For real-valued variable y, result x is bounded 0-1
        """
        return 0.5 * (1 + erf(y / sqrt(2.)))

    @staticmethod
    def invprobitlogjac(y):
        """ Computes the log jacobian for the inverse probit transformation """
        return (-0.5 * np.log(2 * pi) - y * y / 2.).sum()

    @staticmethod
    def pd_matrix_inversion(mat):
        return cholesky_inversion(tuple(map(tuple, mat)))

SHPBTuple = namedtuple('SHPBTuple', 'X Y temp edot emax type')
class Experiment_SHPB(object):
    X     = None  # Observed Strains
    Y     = None  # Observed Flow Stresses
    model = None  # materialModel
    temp  = None  # Mateiral Starting Temperature
    edot  = None  # Strain Rate of experiment
    emax  = None  # Maximum strain that we're predicting at
    tuple = None  # namedtuple, describing the experiment
    table_name = None # what did this data come from?

    data_query = "SELECT strain,stress FROM {};"
    meta_query = "SELECT temperature, edot, emax FROM meta WHERE table_name = '{}';"

    @property
    def parameter_list(self):
        return self.model.get_parameter_list()

    def predict(self, param, x_new):
        self.model.initialize_state(T = self.temp)
        self.model.update_parameters(theta)
        res  = self.model.compute_state_history()[:,1:3]
        est_curve = interp1d(model_curve[:,0], model_curve[:,1], kind = 'cubic')
        yhat = est_curve(x_new)
        return yhat

    @mt.lru_cache()
    def sse(self, param):
        ydiff = self.Y - self.predict(np.array(param), x_new = self.X)
        return (ydiff * ydiff).sum()

    def check_constraints(self):
        return self.model.check_constraints()

    def load_data(self, cursor, table_name):
        data = np.array(list(cursor.execute(self.data_query.format(table_name))))
        self.X, self.Y = data[:,0], data[:,1]
        meta = list(cursor.execute(self.meta_query.format(table_name)))[0]
        self.temp, self.edot, self.emax = meta
        self.source_name = table_name
        return

    def initialize_model(self, params, consts):
        self.model.initialize(params, consts, T = self.temp)
        return

    def __init__(self, cursor, table_name, model_args):
        self.load_data(cursor, table_name)
        self.table_name = table_name
        self.model = MaterialModel(**model_args)
        self.tuple = SHPBTuple(self.X, self.Y, self.temp, self.edot, self.emax, 'shpb')
        return

# TCTuple = namedtuple('TCTuple', fill this in)
class Experiment_TC(object):
    pass

# FPTuple = namedtuple('FPTuple', fill this in)
class Experiment_FP(object):
    pass

Experiment = {
    'shpb' : Experiment_SHPB,
    'tc'   : Experiment_TC,
    'fp'   : Experiment_FP,
    }

# EOF
