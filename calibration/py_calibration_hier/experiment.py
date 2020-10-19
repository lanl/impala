from physical_models_c import MaterialModel
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf, erfinv
from collections import namedtuple
from math import ceil, sqrt, pi, log
import methodtools as mt
import functools as ft
import numpy as np
np.seterr(under = 'ignore')
import pandas as pd
import sqlite3 as sql
import os
import pickledBass as pb
#import ipdb #ipdb.set_trace()

POOL_SIZE = 8

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
        self.model.update_parameters(param)
        res  = self.model.compute_state_history()[:,1:3]
        est_curve = interp1d(model_curve[:,0], model_curve[:,1], kind = 'cubic')
        yhat = est_curve(x_new)
        return yhat

    @mt.lru_cache()
    def sse(self, param):
        ydiff = self.Y - self.predict(np.array(param), x_new = self.X)
        return (ydiff * ydiff).sum()

    def check_constraints(self, param):
        self.model.update_parameters(param)
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

    def initialize_constants(self, constant_vec):
        self.model.initialize_constants(constant_vec)
        return

    def __init__(self, conn, table_name, model_args):
        cursor = conn.cursor()
        self.load_data(cursor, table_name)
        self.table_name = table_name
        self.model = MaterialModel(**model_args)
        self.tuple = SHPBTuple(self.X, self.Y, self.temp, self.edot, self.emax, 'shpb')
        return

PCATuple = namedtuple('PCATuple', 'Y ymean ysd tbasis bounds samples type') # fill this in)
class Experiment_PCA(object):
    """ Experiment involving the BASS PCA based emulator """
    X     = None  # Observed Strains
    Y     = None  # Observed Flow Stresses
    Xemu  = None
    Yemu  = None
    model = None  # materialModel
    tuple = None  # namedtuple, describing the experiment
    table_name = None # what did this data come from?
    parameter_list = None

    data_query = " SELECT * FROM {}; "
    meta_query = " SELECT sim_input, sim_output FROM meta where table_name = '{}'; "
    emui_query = " SELECT * FROM {}; "
    emuo_query = " SELECT * FROM {}; "

    @property
    def tuple(self):
        mcmc_use = np.random.choice(range(self.n_emcmc))
        samples = []
        for ipc in range(self.emodel.nbasis):
            bm = self.emodel.bm_list[ipc]
            model_use = bm.model_lookup[mcmc_use]
            nbasis    = bm.samples.nbasis[mcmc_use]
            samples.append(
                pb.Sample(
                    bm.samples.s2[mcmc_use],
                    nbasis,
                    bm.samples.n_int[model_use, :],
                    bm.samples.signs[model_use, 0 : nbasis, :],
                    bm.samples.vs[model_use, 0 : nbasis, :],
                    bm.samples.knots[model_use, 0 : nbasis, :],
                    bm.samples.beta[mcmc_use, 0 : (nbasis + 1)],
                    )
                )
        return PCATuple(self.Y, self.emodel.y_mean, self.emodel.y_sd, self.emodel.basis.T,
                            self.emodel.bm_list[0].data.bounds, samples, 'pca')

    @property
    def parameter_list(self):
        return self.model.get_parameter_list()

    def predict(self, param, x_new):
        tuple = self.tuple
        preds = pb.predictPCA(param, tuple.samples. tuple.tbasis, tuple.ysd, tuple.ymean, tuple.bounds)
        return

    def sse(self, param):
        pass

    def check_constraints(self, param):
        self.model.update_parameters(param)
        return self.model.check_constraints()

    def initialize_constants(self, constant_vec):
        self.model.initialize_constants(constant_vec)
        return

    def load_data(self, conn, table_name):
        cursor = conn.cursor()
        emu_inputs, emu_outputs = list(cursor.execute(self.meta_query.format(table_name)))[0]
        # temp = np.array(list(cursor.execute(self.data_query.format(table_name))))
        self.Y = pd.read_sql(self.data_query.format(table_name), conn).values
        Xe = pd.read_sql(self.emui_query.format(emu_inputs), conn)
        Ye = pd.read_sql(self.emuo_query.format(emu_outputs), conn)
        cols = set([x for x in Xe.columns.values.tolist() if x != 'index'])
        param_list_lower = [x.lower() for x in self.parameter_list]
        self.eta_cols = list(cols.difference(set(param_list_lower)))
        self.Xemu = Xe.reindex(columns = param_list_lower + self.eta_cols).values
        self.bounds = np.vstack((
            Xe[self.eta_cols].values.min(axis = 0),
            Xe[self.eta_cols].values.max(axis = 0)
            )).T
        buffer = (self.bounds.T[1] - self.bounds.T[0]) * 0.05
        self.bounds.T[1] += buffer
        self.bounds.T[0] -= buffer
        self.Yemu = Ye.values
        return

    def __init__(self, conn, table_name, model_args):
        self.table_name = table_name
        self.model = MaterialModel(**model_args)
        self.load_data(conn, table_name)
        #ipdb.set_trace()
        self.emodel = pb.bassPCA(self.Xemu, self.Yemu, ncores = POOL_SIZE, percVar = 99.99)
        self.n_emcmc = len(self.emodel.bm_list[0].samples.nbasis)
        return

WPCATuple = namedtuple('WPCATuple', 'X Y')
class Experiment_WPCA(object):
    """ Experiment involving warping the BASS PCA Emulator """
    pass

Experiment = {
    'shpb' : Experiment_SHPB,
    'pca'  : Experiment_PCA,
    'wpca' : Experiment_WPCA,
    }

# EOF
