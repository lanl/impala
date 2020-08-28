from experiment import Experiment, Transformer, cholesky_inversion

from physical_models_c import MaterialModel
from math import exp, log
import sqlite3 as sql
import numpy as np

class Chain(Transformer):
    experiments = []
    temper_temp = 1.
    iter = 0
    s_theta = None
    s_sigma2 = None

    prior_sigma2_a = 100
    prior_sigma2_b = 1e-4

    @property
    def curr_theta(self):
        return self.s_theta[self.iter]

    @property
    def curr_sigma2(self):
        return self.s_sigma2[self.iter]

    def load_experiments(self, kwargs):
        with self.conn.cursor() as cursor:
            tables = list(cursor.execute('SELECT table_name, type FROM meta;'))
        self.experiments = [
            Experiment[type](table_name, **kwargs)
            for table_name, type in tables
            ]
        return

    def probit_sse(theta):
        param = self.unnormalize(self.invprobit(theta))
        sse = sum([experiment.sse(param) for experiment in self.experiments])
        return sse

    def log_posterior_theta(self, theta, sigma2):
        ldj = self.invprobitlogjac(theta)
        lp = - 0.5 * self.probit_sse(theta) / (sigma2 * self.temper_temp)
        return lp + ldj

    def localcov(self, target):
        lc = localcov(self.s_theta[:(self.iter - 1)], target,
                        self.radius, self.nu, self.psi0)
        return lc

    def sample_theta(self, curr_theta, past_thetas, sigma2):
        curr_cov   = self.localcov(curr_theta)
        prop_theta = curr_theta + normal(size = self.d).dot(cholesky(curr_cov))
        if not self.check_constraints(prop_theta):
            return curr_theta
        prop_cov   = self.localcov(prop_theta)
        # Log Posteriors
        curr_lp = self.log_posterior_theta(curr_theta, sigma2)
        prop_lp = self.log_posterior_theta(prop_theta, sigma2)
        # Transition log-Densities
        cp_ld = mvnorm.logpdf(prop_theta, curr_theta, curr_cov)
        pc_ld = mvnorm.logpdf(curr_theta, prop_theta, prop_cov)
        # compute alpha
        log_alpha = prop_lp + pc_ld - curr_lp - cp_ld
        # Metropolis step
        if log(uniform()) < log_alpha:
            self.accepted[self.iter] = 1
            self.s_theta[self.iter] = prop_theta
        else:
            self.s_theta[self.iter] = curr_theta
        return

    def sample_sigma2(self, theta, sigma2):
        aa = 0.5 * self.N / self.temper_temp + self.prior_sigma2_a
        bb = 0.5 * self.sse / self.temper_temp + self.prior_sigma2_b
        return invgamma.rvs(aa, scale = bb)

    def iter_sample(self):
        self.iter += 1l

    def __init__(self, path, bounds, **kwargs):
        self.materialmodel = MaterialModel(**kwargs)
        self.conn = sql.connect(path)
        return

# EOF
