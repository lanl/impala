import numpy as np
import time
import scipy
from scipy import stats
from scipy.special import multigammaln
from numpy.random import uniform, normal, beta, choice, gamma
from math import sqrt, floor, log
from scipy.special import erf, erfinv, gammaln
from scipy.stats import invwishart
from numpy.linalg import cholesky, slogdet
from itertools import repeat
import multiprocessing as mp
<<<<<<< HEAD
from impala.superCal.pbar import pbar
=======
import impala.superCal.pbar as pbar
>>>>>>> fa9ba43679c159ebdd99411c59fe8c967b25939a
#np.seterr(under='ignore')

# no probit tranform for hierarchical and DP versions

#####################
# class for setting everything up
#####################

class CalibSetup:
    """Structure for storing calibration experimental data, likelihood, discrepancy, etc."""
    def __init__(self, bounds, constraint_func=None):
        self.nexp = 0 # Number of independent emulators
        self.ys = []
        self.y_lens = []
        self.models = []
        self.tl = np.array(1.)
        self.itl = 1/self.tl
        self.bounds = bounds # should be a dict so we can use parameter names
        self.bounds_mat =np.array([v for v in bounds.values()])
        self.p = bounds.__len__()
        if constraint_func is None:
            constraint_func = lambda *x: True
        self.checkConstraints = constraint_func
        self.nmcmc = 10000
        self.nburn = 5000
        self.thin = 5
        self.decor = 100
        self.ntemps = 1
        self.sd_est = []
        self.s2_df = []
        self.ig_a = []
        self.ig_b = []
        self.s2_ind = []
        self.s2_exp_ind = []
        self.ns2 = []
        self.ny_s2 = []
        self.ntheta = []
        self.theta_ind = []
        self.nswap = 5
        self.s2_prior_kern = []
        return
    def addVecExperiments(self, yobs, model, sd_est, s2_df, s2_ind, meas_error_cor=None, theta_ind=None, D=None, discrep_tau=1):
        # if theta_ind specified, s2_ind is?
        self.ys.append(np.array(yobs))
        self.y_lens.append(len(yobs))
        if theta_ind is None:
            theta_ind = [0]*len(yobs)
        
        self.theta_ind.append(theta_ind)
        self.ntheta.append(len(set(theta_ind)))
        model.yobs = np.array(yobs)
        
        model.meas_error_cor = np.eye(len(yobs)) # this doesn't work when ntheta>1
        if meas_error_cor is not None:
            model.meas_error_cor = meas_error_cor
        
        if D is not None:
            model.D = D
            model.nd = D.shape[1]
            model.discrep_tau = discrep_tau

        self.models.append(model)
        self.nexp += 1
        self.sd_est.append(sd_est)
        self.s2_df.append(s2_df)
        self.ig_a.append(s2_df / 2)
        self.ig_b.append(s2_df/2 * sd_est ** 2)
        self.s2_ind.append(s2_ind)
        self.s2_exp_ind.append(list(range(sd_est.size)))
        self.ns2.append(sd_est.size)
        vec = np.empty(sd_est.size)
        for i in range(len(vec)):
            vec[i] = np.sum(s2_ind==i)
        self.ny_s2.append(vec)
        self.nclustmax = max(sum(self.ntheta), 10)
        if np.any(s2_df == 0):
            self.s2_prior_kern.append(ldhc_kern)
        else:
            self.s2_prior_kern.append(ldig_kern)
        return
    def setTemperatureLadder(self, temperature_ladder, start_temper=1000):
        self.tl = temperature_ladder
        self.itl = 1/self.tl
        self.ntemps = len(self.tl)
        self.nswap_per = floor(self.ntemps // 2)
        self.start_temper = start_temper
        return
    def setMCMC(self, nmcmc, nburn=0, thin=1, decor=100, start_var_theta=1e-8, start_tau_theta = 0., start_var_ls2=1e-5, start_tau_ls2=0., start_adapt_iter=300):
        self.nmcmc = nmcmc
        self.nburn = nburn
        self.thin = thin
        self.decor = decor
        self.start_var_theta = start_var_theta
        self.start_tau_theta = start_tau_theta
        self.start_var_ls2 = start_var_ls2
        self.start_tau_ls2 = start_tau_ls2
        self.start_adapt_iter = start_adapt_iter
        return
    def setHierPriors(self, theta0_prior_mean, theta0_prior_cov, Sigma0_prior_df, Sigma0_prior_scale):
        self.theta0_prior_mean = theta0_prior_mean
        self.theta0_prior_cov = theta0_prior_cov
        self.Sigma0_prior_df = Sigma0_prior_df
        self.Sigma0_prior_scale = Sigma0_prior_scale
        return
    def setClusterPriors(self, nclustmax):
        self.nclustmax = nclustmax
    pass


def normalize(x, bounds):
    """Normalize to 0-1 scale"""
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def unnormalize(z, bounds):
    """Inverse of normalize"""
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

def probit(x):
    """ Probit Transformation: For x in (0,1), y in (-inf,inf) """
    return np.sqrt(2.) * erfinv(2 * x - 1)

def invprobit(y):
    """ Inverse Probit Transformation: For y in (-inf,inf), x in (0,1) """
    return 0.5 * (1 + erf(y / np.sqrt(2.)))

initfunc_probit = np.random.normal # if probit, then normal--if uniform, then uniform
initfunc_unif = np.random.uniform

def tran_probit(th, bounds, names):
    return dict(zip(names, unnormalize(invprobit(th),bounds).T)) # If probit
    # return dict(zip(names, unnormalize(th, bounds).T)) # If uniform
    pass

def tran_unif(th, bounds, names):
    return dict(zip(names, unnormalize(th, bounds).T)) # If uniform

def chol_sample(mean, cov):
    return mean + np.dot(np.linalg.cholesky(cov), np.random.standard_normal(mean.size))

def chol_sample_1per(means, covs):
    return means + np.einsum('tnpq,tnq->tnp', cholesky(covs), normal(size = means.shape))

def chol_sample_nper(means, covs, n):
    return means + np.einsum('ijk,ilk->ilj', cholesky(covs), normal(size = (*means.shape, n)))

def chol_sample_1per_constraints(means, covs, cf, bounds_mat, bounds_keys, bounds):
    """ Sample with constraints.  If fail constraints, resample. """
    chols = cholesky(covs)
    cand = means + np.einsum('ijk,ik->ij', chols, normal(size = means.shape))
    good = cf(tran_unif(cand, bounds_mat, bounds_keys), bounds)
    while np.any(~good):
        cand[np.where(~good)] = (
            + means[~good]
            + np.einsum('ijk,ik->ij', chols[~good], normal(size = ((~good).sum(), means.shape[1])))
            )
        good[~good] = cf(tran_unif(cand[~good], bounds_mat, bounds_keys), bounds)
    return cand

def chol_sample_nper_constraints(means, covs, n, cf, bounds_mat, bounds_keys, bounds):
    """ Sample with constraints.  If fail constraints, resample. """
    chols = cholesky(covs)
    cand = means.reshape(means.shape[0], 1, means.shape[1]) + \
            np.einsum('ijk,ink->inj', chols, normal(size = (means.shape[0], n, means.shape[1])))
    for i in range(cand.shape[0]):
        goodi = cf(tran_unif(cand[i], bounds_mat, bounds_keys),bounds)
        while np.any(~goodi):
            cand[i, np.where(~goodi)[0]] = (
                + means[i]
                + np.einsum('ik,nk->ni', chols[i], normal(size = ((~goodi).sum(), means.shape[1])))
                )
            goodi[np.where(~goodi)[0]] = (
                cf(tran_unif(cand[i,np.where(~goodi)[0]], bounds_mat, bounds_keys), bounds)
                )
    return cand

def cov_3d_pcm(arr, mean):
    """ Covariance array from 3d Array (with pre-computed mean):
    arr = 3d Array (nSamp x nTemp x nCol)
    mean = 2d Array (nTemp x nCol)
    out = 3d Array (nTemp x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum('kij,kil->ijl', arr - mean, arr - mean) / (N - 1)

def cov_4d_pcm(arr, mean):
    """ Covariance Array from 4d Array (With pre-computed mean):
    arr = 4d array (nSamp x nTemp x nTheta x nCol)
    mean = 3d Array (nTemp x nCol)
    out = 4d Array (nTemp x nTheta x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum('ktij,ktil->tijl', arr - mean, arr - mean) / (N - 1)

def cov_anyd_pcm(arr, mean):
    """ Covariance Array from p dimensional Array (With pre-computed mean):
    arr = p-dim array (e.g., nSamp x nTemp x nTheta x nCol)
    mean = (p-1)-dim Array (e.g., nTemp x nCol)
    out = p-dim Array (nTemp x nTheta x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum('...ij,...il->...ijl', arr - mean, arr - mean) / (N - 1)

def mvnorm_logpdf(x, mean, Prec, ldet): # VALIDATED
    """
    # k = x.shape[-1]
    # part1 = -k * 0.5 * np.log(2 * np.pi) - 0.5 * ldet
    # x = x - mu
    # return part1 + np.squeeze(-x[..., None, :] @ Prec @ x[..., None] / 2)
    """
    ld = (
        - 0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet
        - 0.5 * np.einsum('tm,mn,tn->t', x - mean, Prec, x - mean)
        )
    return ld

def mvnorm_logpdf_(x, mean, prec, ldet): # VALIDATED
    """
    x = (ntemps, n_theta[i], k)
    mu = (ntemps[i])
    prec = (ntemps x k x k)
    ldet = (ntemps)
    """
    # m = np.repeat(mean.reshape(mean.shape[0], 1, mean.shape[1]), x.shape[1], 1)
    mean_reshape = (mean.shape[0], 1, mean.shape[1])
    ld = (
        - 0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet.reshape(-1,1)
        - 0.5 * np.einsum(
                    'tsm,tmn,tsn->ts', 
                    x - mean.reshape(mean_reshape), 
                    prec, 
                    x - mean.reshape(mean_reshape),
                    )
        )
    return ld

def invwishart_logpdf(w, df, scale): # VALIDATED
    """ unnormalized logpdf of inverse wishart w given df and scale """
    ld = (
        + 0.5 * df * slogdet(scale)[1]
        - multigammaln(df / 2, scale.shape[-1])
        - 0.5 * df * scale.shape[-1] * log(2.)
        - 0.5 * (df + w.shape[-1] + 1) * slogdet(w)[1]
        - 0.5 * np.einsum('...ii->...', np.einsum('ji,...ij->...ij', scale, np.linalg.inv(w)))
        )
    return ld

def invgamma_logpdf(s, alpha, beta):
    """ log pdf of inverse gamma distribution -- Assume s = (n x p); alpha, beta = (p) """
    ld = (
        + alpha * np.log(beta)
        - gammaln(alpha)
        - (alpha - 1) * np.log(s)
        - beta / s
        ).sum(axis = 1)
    return ld

def gamma_logpdf(s, alpha, beta):
    """ logpdf pf gamma distribution -- assume s = (n); alpha, beta  = 1"""
    ld = (
        + alpha * np.log(beta) 
        - gammaln(alpha)
        + (alpha - 1) * np.log(s)
        - beta * s
        )
    return ld

def bincount2D_vectorized(a, max_count):
    """
    Applies np.bincount across a 2d array (row-wise).

    Adapted From: https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    a_offs = a + np.arange(a.shape[0])[:,None]*max_count
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*max_count).reshape(-1,max_count)

def ldig_kern(x, a, b): # ig
  return (-a - 1) * np.log(x) - b / x

def ldhc_kern(x, a, b): # half cauchy
  return -np.log(x + 1)

from collections import namedtuple
OutCalibPool = namedtuple(
    'OutCalibPool', 'theta s2 count count_s2 count_decor cov_theta_cand cov_ls2_cand pred_curr discrep_vars llik',
    )
OutCalibHier = namedtuple(
    'OutCalibHier', 'theta s2 count count_s2 count_decor2 cov_theta_cand cov_ls2_cand count_temper pred_curr theta0 Sigma0',
    )
OutCalibClust = namedtuple(
    'OutCalibClust', 'theta theta_hist s2 count count_temper pred_curr theta0 Sigma0 delta eta nclustmax'
    )


class AMcov_pool:
    def __init__(self, ntemps, p, start_var=1e-4, start_adapt_iter=300, tau_start=0.):
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau  = np.repeat(tau_start, ntemps)
        self.S    = np.empty([ntemps, p, p])
        self.S[:] = np.eye(p) * start_var
        self.cov  = np.empty([ntemps, p, p])
        self.mu   = np.empty([ntemps, p])
        self.ntemps = ntemps
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = np.zeros(ntemps, dtype = int)

    def update(self, x, m):
        if m > self.start_adapt_iter:
            self.mu += (x[m-1] - self.mu) / m
            self.cov = (
                + ((m - 1) / m) * self.cov
                + ((m - 1) / (m * m)) * np.einsum('ti,tj->tij', x[m-1] - self.mu, x[m-1] - self.mu)
                )
            self.S   = self.AM_SCALAR * np.einsum('ijk,i->ijk', self.cov + np.eye(self.p) * self.eps, np.exp(self.tau))
            #S   = cc * np.einsum('ijk,i->ijk', cov_3d_pcm(theta[:m], theta[:m].mean(axis = 0)) + np.eye(setup.p) * eps, np.exp(tau))

        elif m == self.start_adapt_iter:
            self.mu  = x[:m].mean(axis = 0)
            self.cov = cov_3d_pcm(x[:m], self.mu)
            self.S   = self.AM_SCALAR * np.einsum('ijk,i->ijk', self.cov + np.eye(self.p) * self.eps, np.exp(self.tau))

    def update_tau(self, m):
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > self.start_adapt_iter):
            delta = min(0.5, 5 / sqrt(m + 1))
            self.tau[np.where(self.count_100 < 23)] = self.tau[np.where(self.count_100 < 23)] - delta
            self.tau[np.where(self.count_100 > 23)] = self.tau[np.where(self.count_100 > 23)] + delta
            self.count_100 *= 0
            # note, e^tau scales whole covariance matrix, so it shrinks covariance for inert inputs too much...need decor for those.

    def gen_cand(self, x, m):
            x_cand  = (
            + x[m-1]
            + np.einsum('ijk,ik->ij', cholesky(self.S), normal(size = (self.ntemps, self.p)))
            )
            return x_cand


class AMcov_hier:
    def __init__(self, nexp, ntheta, ntemps, p, start_var=1e-4, start_adapt_iter=300, tau_start=0.): # ntheta is a vector of length nexp
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = [tau_start * np.ones((ntemps, ntheta[i])) for i in range(nexp)]
        self.S   = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        for i in range(nexp):
            self.S[i][:] = np.eye(p) * start_var
        self.cov = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        self.mu  = [np.empty((ntemps, ntheta[i], p)) for i in range(nexp)]
        self.nexp = nexp
        self.ntemps = ntemps
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = [np.zeros((ntemps, ntheta[i])) for i in range(nexp)]

    def update(self, x, m): # called in mth iteration, so latest value is x[i][m-1]
        if m > self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i] += (x[i][m-1] - self.mu[i]) / m
                self.cov[i][:] = (
                    + ((m-1) / m) * self.cov[i]
                    + ((m-1) / (m * m)) * np.einsum(
                        'tej,tel->tejl', x[i][m-1] - self.mu[i], x[i][m-1] - self.mu[i],
                        )
                    )
                self.S[i] = self.AM_SCALAR * np.einsum(
                    'tejl,te->tejl', self.cov[i] + np.eye(self.p) * self.eps, np.exp(self.tau[i]),
                    )

        elif m == self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i][:]  = x[i][:m].mean(axis = 0)
                #self.mu[i][:]  = x[i].mean(axis = 0)
                self.cov[i][:] = cov_4d_pcm(x[i][:m], self.mu[i])
                self.S[i][:]   = self.AM_SCALAR * np.einsum('tejl,te->tejl', self.cov[i] + np.eye(self.p) * self.eps, np.exp(self.tau[i]))

    def update_tau(self, m):
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > self.start_adapt_iter):
            delta = min(0.5, 5 / np.sqrt(m + 1))
            for i in range(self.nexp):
                self.tau[i][self.count_100[i] < 23] -= delta
                self.tau[i][self.count_100[i] > 23] += delta
                self.count_100[i] *= 0

    def gen_cand(self, x, m):
        x_cand = [chol_sample_1per(x[i][m-1], self.S[i]) for i in range(self.nexp)]
        return x_cand

class AMcov_clust:
    def __init__(self):
        pass

    def update(self, x, m):
        pass

    def update_tau(self, m):
        pass

##############################################################################################################################################################################
## Hierarchical Calibration
#@profile
def calibHier(setup):
    t0 = time.time()
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    ntheta = np.sum(setup.ntheta)
    log_s2 = [np.zeros([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        log_s2[i][0] = np.log(setup.sd_est[i]**2)

    #sse    = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    theta  = [
        np.empty([setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p])
        for i in range(setup.nexp)
        ]
    theta_ind_mat = [
        (setup.theta_ind[i][:,None] == range(setup.ntheta[i]))
        for i in range(setup.nexp)
        ]
    s2_ind_mat = [
        (setup.s2_ind[i][:,None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
        ]
    s2_which_mat = [
        [np.where(s2_ind_mat[i][:,j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
        ]
    theta_which_mat = [
        [np.where(theta_ind_mat[i][:,j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
        ]
    theta0_start = initfunc_unif(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(
        tran_unif(theta0_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
        )
    while np.any(~good):
        theta0_start[np.where(~good)] = initfunc_unif(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran_unif(theta0_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta0[0] = theta0_start
    Sigma0[0] = np.eye(setup.p) * 0.25**2

    pred_curr = [None] * setup.nexp # [i], ntemps x ylens[i]
    pred_cand = [None] * setup.nexp # [i], ntemps x ylens[i]
    llik_curr  = [None] * setup.nexp # [i], ntheta[i] x ntemps
    llik_cand  = [None] * setup.nexp # [i], ntheta[i] x ntemps
    #dev_sq    = [None] * setup.nexp # [i], ntheta[i] x ntemps
    itl_mat   = [ # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ntheta[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
        ]

    marg_lik_cov_curr = [None] * setup.nexp

    for i in range(setup.nexp):
        theta[i][0] = chol_sample_nper_constraints(
                theta0[0], Sigma0[0], setup.ntheta[i], setup.checkConstraints, 
                setup.bounds_mat, setup.bounds.keys(), setup.bounds,
                )
        pred_curr[i] = setup.models[i].eval(
                tran_unif(theta[i][0].reshape(setup.ntemps * setup.ntheta[i], setup.p),
                    setup.bounds_mat, setup.bounds.keys()), pool=False
                )#.reshape(setup.ntemps, setup.y_lens[i])
        pred_cand[i] = pred_curr[i].copy()

        marg_lik_cov_curr[i] = [None] * setup.ntemps
        llik_curr[i] = np.empty([setup.ntemps, setup.ntheta[i]])
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = [None] * setup.ntheta[i]
            s2_stretched = log_s2[i][0][t,setup.theta_ind[i]]
            for j in range(setup.ntheta[i]):
                #marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][0, t, setup.s2_ind[i]])[setup.s2_ind[i]==j])
                marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(s2_stretched[s2_which_mat[i][j]]))
                # right now, assuming for vectorized models that new theta means new s2.
                # if you wanted to have multiple s2 for one theta, you would have to update thetas 
                # jointly or sequentially (not independently), unless working with diagonal
                # many possible cases, for now make it work for strength project, generalize later
                #llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_curr[i][t][setup.theta_ind[i]==j], marg_lik_cov_curr[i][t][j]) 
                llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_curr[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 
                #this isnt getting nthetas correct, probably need to change models script...
                # there should be a separate likelihood evaluation (with separate covariance) 
                # for every i, t, ntheta. In diagonal case, we could vectorize over t ntheta...
                # for now, break into separate calls.  Later may be worthwhile to try to vectorize more.
        llik_cand[i]  = llik_curr[i].copy()
# requirements: pooled, anything goes; hier, must have theta_ind matching s2_ind
    # tau = [-0 * np.ones((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    # S   = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    # cov = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    # mu  = [np.empty((setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    # for i in range(setup.nexp):
    #     S[i][:] = np.eye(setup.p) * 1e-4

    cov_theta_cand = AMcov_hier(
        setup.nexp, 
        np.array([setup.ntheta[i] for i in range(setup.nexp)]), 
        setup.ntemps, 
        setup.p, 
        start_var=setup.start_var_theta, 
        start_adapt_iter=setup.start_adapt_iter, 
        tau_start=setup.start_tau_theta)
    cov_ls2_cand = [AMcov_pool(
        setup.ntemps, 
        setup.ns2[i], 
        start_var=setup.start_var_ls2, 
        start_adapt_iter=setup.start_adapt_iter, 
        tau_start=setup.start_tau_ls2) 
        for i in range(setup.nexp)]

    theta0_prior_mean = setup.theta0_prior_mean#np.repeat(0.5, setup.p)
    theta0_prior_cov = setup.theta0_prior_cov#np.eye(setup.p)*1**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]

    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))

    Sigma0_prior_df = setup.Sigma0_prior_df#setup.p
    Sigma0_prior_scale = setup.Sigma0_prior_scale#np.eye(setup.p)*1**2#/setup.p
    Sigma0_dfs = Sigma0_prior_df + ntheta * setup.itl

    Sigma0_ldet_curr = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr  = np.linalg.inv(Sigma0[0])

    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = [np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    count_decor = [np.zeros((setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    count_decor2 = np.zeros((setup.ntemps, setup.p))
    #count_100 = [np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    count_s2 = np.zeros([setup.nexp, setup.ntemps], dtype = int)

    theta_cand = [np.empty([setup.ntemps, setup.ntheta[i], setup.p]) for i in range(setup.nexp)]
    theta_cand_mat = [np.empty([setup.ntemps * setup.ntheta[i], setup.p]) for i in range(setup.nexp)]
    theta_eval_mat = [np.empty(theta_cand_mat[i].shape) for i in range(setup.nexp)]

    alpha  = [np.ones((setup.ntemps, setup.ntheta[i])) * -np.inf for i in range(setup.nexp)]
    alpha_s2 = np.ones([setup.nexp, setup.ntemps]) * (-np.inf)
    accept = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    sw_alpha = np.zeros(setup.nswap_per)
    good_values = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    good_values_mat = [
        good_values[i].reshape(setup.ntheta[i] * setup.ntemps) for i in range(setup.nexp)
        ]
    ## start MCMC
    for m in pbar.pbar(range(1, setup.nmcmc)):

        for i in range(setup.nexp):
            theta[i][m] = theta[i][m-1].copy() # current set to previous, will change if accepted
            log_s2[i][m] = log_s2[i][m-1].copy()
            setup.models[i].step()
            if setup.models[i].stochastic: # update emulator
                pred_curr[i] = setup.models[i].eval(
                    tran_unif(theta[i][m].reshape(setup.ntemps * setup.ntheta[i], setup.p),
                        setup.bounds_mat, setup.bounds.keys()), pool=False
                    )
                for t in range(setup.ntemps):
                    for j in range(setup.ntheta[i]):
                        llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_curr[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 
           # No discrepancy for now...update here if added later

        #------------------------------------------------------------------------------------------
        ## adaptive Metropolis for each temperature / experiment
        # if m > 300:
        #     for i in range(setup.nexp):
        #         mu[i] += (theta[i][m-1] - mu[i]) / m
        #         cov[i][:] = (
        #             + ((m-1) / m) * cov[i]
        #             + ((m-1) / (m * m)) * np.einsum(
        #                 'tej,tel->tejl', theta[i][m-1] - mu[i], theta[i][m-1] - mu[i],
        #                 )
        #             )
        #         S[i] = AM_SCALAR * np.einsum(
        #             'tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]),
        #             )
        # elif m == 300:
        #     for i in range(setup.nexp):
        #         mu[i][:]  = theta[i][:m].mean(axis = 0)
        #         cov[i][:] = cov_4d_pcm(theta[i][:m], mu[i])
        #         S[i][:]   = AM_SCALAR * np.einsum('tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))
        # else:
        #     pass
        
        cov_theta_cand.update(theta, m)
        #------------------------------------------------------------------------------------------
        # MCMC update for thetas
        theta_cand = cov_theta_cand.gen_cand(theta, m)
        
        for i in range(setup.nexp):
            # Find new candidate values for theta
            theta_eval_mat[i][:] = theta[i][m-1].reshape(setup.ntemps * setup.ntheta[i], setup.p)
            #theta_cand[i][:] = chol_sample_1per(theta[i][m-1], S[i])
            theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntemps * setup.ntheta[i], setup.p)
            # Check constraints
            good_values_mat[i][:] = setup.checkConstraints(
                tran_unif(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                )
            good_values[i][:] = good_values_mat[i].reshape(setup.ntemps, setup.ntheta[i])
            # Generate Predictions at new Theta values
            theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][good_values_mat[i]]
            pred_cand[i][:] = setup.models[i].eval(
                    tran_unif(theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys()), pool=False
                    )#.reshape(setup.ntemps, setup.y_lens[i])



            #marg_lik_cov_curr[i] = [None] * setup.ntemps
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    #marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][0, t, setup.s2_ind[i]])[setup.s2_ind[i]==j])
                    llik_cand[i][t][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_cand[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 


            #sse_cand[i][:] = ((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]
            # Calculate log-probability of MCMC accept
            alpha[i][:] = - np.inf
            alpha[i][good_values[i]] = itl_mat[i][good_values[i]] * (
                #- 0.5 * (sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]])
                llik_cand[i][good_values[i]] - llik_curr[i][good_values[i]]
                + mvnorm_logpdf_(theta_cand[i], theta0[m-1],
                                    Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                - mvnorm_logpdf_(theta[i][m-1], theta0[m-1],
                                    Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                )
            # MCMC Accept
            accept[i][:] = np.log(uniform(size = alpha[i].shape)) < alpha[i]
            # Where accept, make changes
            theta[i][m][accept[i]]  = theta_cand[i][accept[i]].copy()
            #ind = accept[i] @ theta_ind_mat[i].T
            #pred_curr[i][ind] = pred_cand[i][ind].copy()

            for t in range(setup.ntemps):
                accept_t = np.where(accept[i][t])[0]
                if accept_t.shape[0] > 0:
                    ind = np.hstack([theta_which_mat[i][j] for j in accept_t])
                    pred_curr[i][t][ind] = pred_cand[i][t][ind].copy()
                #for j in np.where(accept[i][t])[0]:
                #    pred_curr[i][t][theta_which_mat[i][j]] = pred_cand[i][t][theta_which_mat[i][j]]
            llik_curr[i][accept[i]] = llik_cand[i][accept[i]].copy()
            count[i][accept[i]] += 1
            cov_theta_cand.count_100[i][accept[i]] += 1
            #count_100[i][accept[i]] += 1

        cov_theta_cand.update_tau(m)

        #if m>10000:
        #    print('help')

        # # Adaptive Metropolis Update
        # if m % 100 == 0 and m > 300:
        #     delta = min(0.1, 1/np.sqrt(m+1)*5)
        #     for i in range(setup.nexp):
        #         tau[i][count_100[i] < 23] -= delta
        #         tau[i][count_100[i] > 23] += delta
        #         count_100[i] *= 0

        ## Decorrelation Step
        if False:#m % setup.decor == 0:
            for i in range(setup.nexp):
                for k in range(setup.p):
                    # Find new candidate values for theta
                    theta_cand[i][:]     = theta[i][m].copy()
                    theta_eval_mat[i][:] = theta[i][m].reshape(setup.ntheta[i] * setup.ntemps, setup.p)
                    theta_cand[i][:,:,k] = initfunc(size = (setup.ntemps, setup.ntheta[i]))
                    theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntheta[i]*setup.ntemps, setup.p)
                    # Compute constraint flags
                    good_values_mat[i][:] = setup.checkConstraints(
                        tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                        )
                    # Generate predictions at "good" candidate values
                    theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][good_values_mat[i]]
                    good_values[i][:] = good_values_mat[i].reshape(setup.ntemps, setup.ntheta[i])
                    pred_cand[i][:]   = setup.models[i].eval(
                            tran(theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys())
                            )
                    sse_cand[i][:] = ((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1] ## check the [:] here !!!!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Calculate log-probability of MCMC Accept
                    alpha[i][:] = - np.inf
                    alpha[i][good_values[i]] = (
                        - 0.5 * itl_mat[i][good_values[i]] * (
                            sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]]
                            )
                        + itl_mat[i][good_values[i]] * (
                            + mvnorm_logpdf_(theta_cand[i], theta0[m-1],
                                                Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            - mvnorm_logpdf_(theta[i][m], theta0[m-1],
                                                Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            )
                        ) ## THIS NEEDS SOMETHING FOR THE PROPOSAL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # MCMC Accept
                    accept[i][:] = (np.log(uniform(size = alpha[i].shape)) < alpha[i])
                    # Where accept, make changes
                    theta[i][m][accept[i]] = theta_cand[i][accept[i]].copy()
                    pred_curr[i][accept[i] @ s2_ind_mat[i].T] = \
                                        pred_cand[i][accept[i] @ s2_ind_mat[i].T].copy()
                    sse_curr[i][accept[i]] = sse_cand[i][accept[i]].copy()
                    count_decor[i][accept[i], k] = count_decor[i][accept[i], k] + 1




        #------------------------------------------------------------------------------------------
        ## update s2
        for i in range(setup.nexp):

            if setup.models[i].s2=='gibbs':
                ## gibbs update s2       
                dev_sq = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] # squared deviations
                log_s2[i][m] = np.log(1 / np.random.gamma(
                        itl_mat[i] * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                        1 / (itl_mat[i] * (setup.ig_b[i] + dev_sq / 2)),
                        ))
                for t in range(setup.ntemps):
                    s2_stretched = log_s2[i][m][t,setup.theta_ind[i]]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(s2_stretched[s2_which_mat[i][j]]))
                        llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][s2_which_mat[i][j]], pred_curr[i][t][s2_which_mat[i][j]], marg_lik_cov_curr[i][t][j])
                
            elif setup.models[i].s2=='fix':
                    log_s2[i][m] = np.log(setup.sd_est[i]**2)

                    #for t in range(setup.ntemps):
                    #    for j in range(setup.ntheta[i]):
                    #        marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][m][t, setup.s2_ind[i]])[setup.s2_ind[i]==j])
                    #        llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_curr[i][t][setup.theta_ind[i]==j], marg_lik_cov_curr[i][t][j])

            else: # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                ## M-H update s2
                    # NOTE: there is something wrong with this...with no tempering, 10 kolski experiments, 
                    # reasonable priors, s2 can diverge for some experiments (not a random walk, has weird patterns).  
                    # This seems to be because of the joint update, but is strange.  Could be that individual updates 
                    # would make it go away, but it shouldn't be there anyway.

                cov_ls2_cand[i].update(log_s2[i], m)
                ls2_candi = cov_ls2_cand[i].gen_cand(log_s2[i], m)
                
                llik_candi = np.zeros([setup.ntemps, setup.ntheta[i]])
                marg_lik_cov_candi = [None] * setup.ntemps
                for t in range(setup.ntemps):
                    marg_lik_cov_candi[t] = [None] * setup.ntheta[i]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_candi[t][j] = setup.models[i].lik_cov_inv(np.exp(ls2_candi[t, setup.s2_ind[i]])[setup.s2_ind[i]==j])#s2[i][0, t, setup.s2_ind[i]])
                        llik_candi[t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_curr[i][t][setup.theta_ind[i]==j], marg_lik_cov_candi[t][j])
                        # something wrong still, getting way too large of variance
                    #marg_lik_cov_candi[t] = setup.models[i].lik_cov_inv(np.exp(ls2_candi[t])[setup.s2_ind[i]])#s2[i][0, t, setup.s2_ind[i]])
                    #llik_candi[t] = setup.models[i].llik(setup.ys[i], pred_curr[i][t], marg_lik_cov_candi[t])
                
                llik_diffi = (llik_candi - llik_curr[i])
                alpha_s2 = setup.itl * (llik_diffi)       
                alpha_s2 += setup.itl * setup.s2_prior_kern[i](np.exp(ls2_candi), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)#ldhc_kern(np.exp(ls2_cand[i])).sum(axis=1)#ldig_kern(np.exp(ls2_cand[i]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 += setup.itl * ls2_candi.sum(axis=1)
                alpha_s2 -= setup.itl * setup.s2_prior_kern[i](np.exp(log_s2[i][m-1]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)#ldhc_kern(np.exp(log_s2[i][m-1])).sum(axis=1)#ldig_kern(np.exp(log_s2[i][m-1]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 -= setup.itl * log_s2[i][m-1].sum(axis=1)

                runif = np.log(uniform(size=setup.ntemps))
                for t in np.where(runif < alpha_s2)[0]:
                    count_s2[i, t] += 1
                    llik_curr[i][t] = llik_candi[t].copy()
                    log_s2[i][m][t] = ls2_candi[t].copy()
                    marg_lik_cov_curr[i][t] = marg_lik_cov_candi[t].copy()
                    cov_ls2_cand[i].count_100[t] += 1

                cov_ls2_cand[i].update_tau(m)


        if False:
            ## MH update s2
            for i in range(setup.nexp):
                cov_ls2_cand[i].update(log_s2[i], m)
                llik_cand[i][:] = 0.

            ls2_cand = [cov_ls2_cand[i].gen_cand(log_s2[i], m) for i in range(setup.nexp)]
            

            
            marg_lik_cov_cand = [None] * setup.nexp
            for i in range(setup.nexp):
                marg_lik_cov_cand[i] = [None] * setup.ntemps
                for t in range(setup.ntemps):
                    marg_lik_cov_cand[i][t] = [None] * setup.ntheta[i]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_cand[i][t][j] = setup.models[i].lik_cov_inv(np.exp(ls2_cand[i][t, setup.s2_ind[i]])[setup.s2_ind[i]==j])#s2[i][0, t, setup.s2_ind[i]])
                        llik_cand[i][t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_curr[i][t][setup.theta_ind[i]==j], marg_lik_cov_cand[i][t][j])
            
            ## joint update for ntheta[i] s2s
            #llik_diff = (llik_cand.sum(axis=2) - llik_curr.sum(axis=2)) # should be summing over the nthera axis
            alpha_s2[:] = - np.inf
            #alpha_s2 = setup.itl * (llik_diff)       
            for i in range(setup.nexp): # this needs help...sum over ntheta axis
                alpha_s2[i,:] = setup.itl * (llik_cand[i].sum(axis=1) - llik_curr[i].sum(axis=1))
                alpha_s2[i,:] += setup.itl * setup.s2_prior_kern[i](np.exp(ls2_cand[i]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                alpha_s2[i,:] += setup.itl * ls2_cand[i].sum(axis=1)
                alpha_s2[i,:] -= setup.itl * setup.s2_prior_kern[i](np.exp(log_s2[i][m-1]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                alpha_s2[i,:] -= setup.itl * log_s2[i][m-1].sum(axis=1)

            runif = np.log(uniform(size=[setup.nexp, setup.ntemps]))
            for i in range(setup.nexp):
                for t in np.where(runif[i] < alpha_s2[i])[0]:
                    if np.any(ls2_cand[0][0] > np.log(100)) and t==0:
                        print('bad')
                    count_s2[i, t] += 1
                    llik_curr[i][t] = llik_cand[i][t].copy()
                    log_s2[i][m][t] = ls2_cand[i][t].copy()
                    marg_lik_cov_curr[i][t] = marg_lik_cov_cand[i][t].copy()
                    cov_ls2_cand[i].count_100[t] += 1

            for i in range(setup.nexp):
                cov_ls2_cand[i].update_tau(m)



            # dev_sq[i][:] = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i]
            # s2[i][m] = 1 / np.random.gamma(
            #     (itl_mat[i] * setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
            #     1 / (itl_mat[i] * (setup.ig_b[i] +  dev_sq[i] / 2)),
            #     )
            # sse_curr[i][:] = dev_sq[i] / s2[i][m]

        ## Gibbs update theta0
        cc = np.linalg.inv(
            np.einsum('t,tpq->tpq', ntheta * setup.itl, Sigma0_inv_curr) + theta0_prior_prec,
            )
        tbar *= 0.
        for i in range(setup.nexp):
            tbar += theta[i][m].sum(axis = 1)
        tbar /= ntheta
        dd = (
            + np.einsum('t,tl->tl', setup.itl, np.einsum('tlk,tk->tl', ntheta * Sigma0_inv_curr, tbar))
            + np.dot(theta0_prior_prec, theta0_prior_mean)
            )
        theta0[m][:] = chol_sample_1per_constraints(
            np.einsum('tlk,tk->tl', cc, dd), cc,
            setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds,
            )

        ## Gibbs update Sigma0
        mat *= 0.
        for i in range(setup.nexp):
            mat += np.einsum(
                    'tnp,tnq->tpq', 
                    theta[i][m] - theta0[m].reshape(setup.ntemps, 1, setup.p), 
                    theta[i][m] - theta0[m].reshape(setup.ntemps, 1, setup.p),
                    )
        Sigma0_scales = Sigma0_prior_scale + np.einsum('t,tml->tml',setup.itl,mat)
        for t in range(setup.ntemps):
            Sigma0[m,t] = invwishart.rvs(df = Sigma0_dfs[t], scale = Sigma0_scales[t])
        Sigma0_ldet_curr[:] = np.linalg.slogdet(Sigma0[m])[1]
        Sigma0_inv_curr[:] = np.linalg.inv(Sigma0[m])



        # better decorrelation step, joint
        if m % setup.decor == 0:
            for k in range(setup.p):
                z = np.random.normal()*.1
                theta0_cand = theta0[m].copy()
                theta0_cand[:,k] += z
                good_values_theta0 = setup.checkConstraints(
                    tran_unif(theta0_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                    )
                
                for i in range(setup.nexp):
                    # Find new candidate values for theta
                    theta_cand[i][:]     = theta[i][m].copy()
                    theta_eval_mat[i][:] = theta[i][m].reshape(setup.ntheta[i] * setup.ntemps, setup.p)
                    theta_cand[i][:,:,k] += z
                    theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntheta[i]*setup.ntemps, setup.p)
                    # Compute constraint flags
                    good_values_mat[i][:] = setup.checkConstraints(
                        tran_unif(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                        )
                    # Generate predictions at "good" candidate values
                    theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][good_values_mat[i]]
                    good_values[i][:] = (good_values_mat[i].reshape(setup.ntemps, setup.ntheta[i]).T * good_values_theta0).T
                    pred_cand[i][:]   = setup.models[i].eval(
                            tran_unif(theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys()), pool=False
                            )#.reshape(setup.ntemps, setup.ntheta[i], setup.y_lens[i])
                    #sse_cand[i][:] = ((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m]
                    for t in range(setup.ntemps):
                        for j in range(setup.ntheta[i]):
                            #llik_cand[i][t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_cand[i][t][setup.theta_ind[i]==j], marg_lik_cov_curr[i][t][j]) 
                            llik_cand[i][t][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_cand[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 
                    
                    alpha[i][:] = - np.inf
                    alpha[i][good_values[i]] = (
                        itl_mat[i][good_values[i]] * (
                            llik_cand[i][good_values[i]] - llik_curr[i][good_values[i]]
                            )
                        + itl_mat[i][good_values[i]] * (
                            + mvnorm_logpdf_(theta_cand[i], theta0_cand,
                                                Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            - mvnorm_logpdf_(theta[i][m], theta0[m],
                                                Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            )
                        )
                # now sum over alpha (for each temperature), add alpha for theta0 to prior, accept or reject
                #alpha_tot = mvnorm_logpdf_(theta0_cand, theta0_prior_mean.reshape(setup.ntemps,setup.p), theta0_prior_prec, theta0_prior_ldet)*itl + sum(alpha)
                alpha_tot = sum(alpha).T -0.5 * setup.itl * np.diag((theta0_cand - theta0_prior_mean) @ theta0_prior_prec @ (theta0_cand - theta0_prior_mean).T) + 0.5 * setup.itl * np.diag((theta0[m] - theta0_prior_mean) @ theta0_prior_prec @ (theta0[m] - theta0_prior_mean).T)
                
                accept_tot = (np.log(uniform(size = setup.ntemps)) < alpha_tot.sum(axis=0))
                #accept[i][:] = (np.log(uniform(size = alpha[i].shape)) < alpha[i])
                # Where accept, make changes
                theta0[m][accept_tot,:] = theta0_cand[accept_tot,:]
                for i in range(setup.nexp):
                    theta[i][m][accept_tot] = theta_cand[i][accept_tot].copy()
                    pred_curr[i][accept_tot,:] = pred_cand[i][accept_tot,:].copy()
                    llik_curr[i][accept_tot] = llik_cand[i][accept_tot].copy()
                    
                count_decor2[accept_tot, k] = count_decor2[accept_tot, k] + 1

        ## tempering swaps
        if m > setup.start_temper and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0. # reset swap probability
                sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                    + mvnorm_logpdf(theta0[m][sw.T[0]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    - mvnorm_logpdf(theta0[m][sw.T[1]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    + invwishart_logpdf(Sigma0[m][sw.T[0]], Sigma0_prior_df, Sigma0_prior_scale)
                    - invwishart_logpdf(Sigma0[m][sw.T[1]], Sigma0_prior_df, Sigma0_prior_scale)
                    )
                for i in range(setup.nexp):
                    sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                        # for t_0
                        + setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[0]]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                        + mvnorm_logpdf_(theta[i][m][sw.T[0]], theta0[m, sw.T[0]],
                                Sigma0_inv_curr[sw.T[0]], Sigma0_ldet_curr[sw.T[0]]).sum(axis = 1)
                        #- 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[0]]
                        #- 0.5 * sse_curr[i][sw.T[0]].sum(axis = 1)
                        + llik_curr[i][sw.T[0]].sum(axis=1)
                        # for t_1
                        - setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[1]]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                        - mvnorm_logpdf_(theta[i][m][sw.T[1]], theta0[m,sw.T[1]],
                                Sigma0_inv_curr[sw.T[1]], Sigma0_ldet_curr[sw.T[1]]).sum(axis = 1)
                        #+ 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[1]]
                        #+ 0.5 * sse_curr[i][sw.T[1]].sum(axis = 1)
                        - llik_curr[i][sw.T[1]].sum(axis=1)
                        )
                for tt in sw[np.where(np.log(uniform(size = setup.nswap_per)) < sw_alpha)]:
                    count_temper[tt[0], tt[1]] = count_temper[tt[0], tt[1]] + 1
                    for i in range(setup.nexp):
                        theta[i][m,tt[0]], theta[i][m,tt[1]]     = \
                                            theta[i][m,tt[1]].copy(), theta[i][m,tt[0]].copy()
                        log_s2[i][m][tt[0]], log_s2[i][m][tt[1]]               = \
                                            log_s2[i][m][tt[1]].copy(), log_s2[i][m][tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = \
                                            pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                        llik_curr[i][tt[0]], llik_curr[i][tt[1]]   = \
                                            llik_curr[i][tt[1]].copy(), llik_curr[i][tt[0]].copy()
                    theta0[m,tt[0]], theta0[m,tt[1]] = theta0[m,tt[1]].copy(), theta0[m,tt[0]].copy()
                    Sigma0[m,tt[0]], Sigma0[m,tt[1]] = Sigma0[m,tt[1]].copy(), Sigma0[m,tt[0]].copy()
                    Sigma0_inv_curr[tt[0]], Sigma0_inv_curr[tt[1]]   = \
                                            Sigma0_inv_curr[tt[1]].copy(), Sigma0_inv_curr[tt[0]].copy()
                    Sigma0_ldet_curr[tt[0]], Sigma0_ldet_curr[tt[1]] = \
                                            Sigma0_ldet_curr[tt[1]].copy(), Sigma0_ldet_curr[tt[0]].copy()
                #if np.exp(log_s2[i][m,0,0])>1:
                #    print('a')
        #print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))

    s2 = log_s2.copy()
    for i in range(setup.nexp):
        s2[i] = np.exp(log_s2[i])

    count_temper = count_temper + count_temper.T - np.diag(np.diag(count_temper))
    # theta_reshape = [np.swapaxes(t,1,2) for t in theta]
    out = OutCalibHier(theta, s2, count, count_s2, count_decor2, cov_theta_cand, cov_ls2_cand,
                            count_temper, pred_curr, theta0, Sigma0)
    return(out)






##############################################################################################################################################################################
## Pooled Calibration



#@profile
def calibPool(setup):
    t0 = time.time()
    theta = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    n_s2 = np.sum(setup.ns2)
    log_s2 = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    #s2_vec_curr = [s2[i][0,:,setup.s2_ind[i]] for i in range(setup.nexp)]
    s2_ind_mat = [
        (setup.s2_ind[i][:,None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
        ]
    theta_start = initfunc_unif(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(tran_unif(theta_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    while np.any(~good):
        theta_start[np.where(~good)] = initfunc_unif(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran_unif(theta_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta[0] = theta_start

    itl_mat   = [ # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ns2[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
        ]

    pred_curr = [None] * setup.nexp
    # sse_curr = np.empty([setup.ntemps, setup.nexp])
    llik_curr = np.empty([setup.nexp, setup.ntemps])
    #dev_sq = [np.empty((setup.ntemps, setup.ns2[i])) for i in range(setup.nexp)]
    marg_lik_cov_curr = [None] * setup.nexp
    for i in range(setup.nexp):
        marg_lik_cov_curr[i] = [None] * setup.ntemps
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][0, t, setup.s2_ind[i]])[setup.s2_ind[i]])
            # ask around: is list of lists lookup slow?? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    llik_curr[:] = 0.
    for i in range(setup.nexp):
        pred_curr[i] = setup.models[i].eval(
            tran_unif(theta[0], setup.bounds_mat, setup.bounds.keys()), pool=True
            )
        # sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)
        #((pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] / s2[i][0]).sum(axis = 1)
        for t in range(setup.ntemps):
            llik_curr[i, t] = setup.models[i].llik(setup.ys[i], pred_curr[i][t], marg_lik_cov_curr[i][t])

    #eps  = 1.0e-13
    #tau  = np.repeat(-4.0, setup.ntemps)
    #AM_const   = 2.4**2/setup.p
    #S    = np.empty([setup.ntemps, setup.p, setup.p])
    #S[:] = np.eye(setup.p)*1e-6
    #cov  = np.empty([setup.ntemps, setup.p, setup.p])
    #mu   = np.empty([setup.ntemps, setup.p])

    cov_theta_cand = AMcov_pool(ntemps = setup.ntemps, p = setup.p, start_var=setup.start_var_theta, tau_start=setup.start_tau_theta, start_adapt_iter=setup.start_adapt_iter)
    cov_ls2_cand = [AMcov_pool(ntemps = setup.ntemps, p = setup.ns2[i], start_var=setup.start_var_ls2, tau_start=setup.start_tau_ls2, start_adapt_iter=setup.start_adapt_iter) for i in range(setup.nexp)]

    count = np.zeros([setup.ntemps, setup.ntemps], dtype = int)
    count_s2 = np.zeros([setup.nexp, setup.ntemps], dtype = int)
    count_decor = np.zeros([setup.p, setup.ntemps], dtype = int)
    #count_100 = np.zeros(setup.ntemps, dtype = int)

    pred_cand = [_.copy() for _ in pred_curr]
    discrep_curr = [_*0. for _ in pred_curr]
    discrep_vars = [np.zeros([setup.nmcmc, setup.ntemps, setup.models[i].nd]) for i in range(setup.nexp)]

    llik_cand = llik_curr.copy()

    alpha    = np.ones(setup.ntemps) * (-np.inf)
    alpha_s2 = np.ones([setup.nexp, setup.ntemps]) * (-np.inf)
    sw_alpha = np.zeros(setup.nswap_per)

    llik = np.empty(setup.nmcmc)

    ## start MCMC
    for m in pbar.pbar(range(1, setup.nmcmc)):
        

        theta[m] = theta[m-1].copy() # current set to previous, will change if accepted
        for i in range(setup.nexp):
            log_s2[i][m] = log_s2[i][m-1].copy()
            if setup.models[i].nd>0: # update discrepancy
                for t in range(setup.ntemps):
                    discrep_vars[i][m][t] = setup.models[i].discrep_sample(setup.ys[i], pred_curr[i][t], marg_lik_cov_curr[i][t], setup.itl[t])
                    discrep_curr[i][t] = setup.models[i].D @ discrep_vars[i][m][t]
            
            setup.models[i].step()
            if setup.models[i].stochastic: # update emulator
                pred_curr[i] = setup.models[i].eval(
                    tran_unif(
                        theta[m], 
                        setup.bounds_mat, setup.bounds.keys()
                        ), pool=True
                    )
            if setup.models[i].nd>0 or setup.models[i].stochastic:
                for t in range(setup.ntemps):
                    llik_curr[i, t] = setup.models[i].llik(setup.ys[i] - discrep_curr[i][t], pred_curr[i][t], marg_lik_cov_curr[i][t])



        #----------------------------------------------------------
        ## adaptive Metropolis for each temperature

        cov_theta_cand.update(theta, m)

        # if m > 300:
        #     mu += (theta[m-1] - mu) / m
        #     cov = (
        #         + (m - 1) / m * cov
        #         + (m - 1) / m**2 * np.einsum('ti,tj->tij', theta[m-1] - mu, theta[m-1] - mu)
        #         )
        #     if m>10000:
        #         1+1
            #S   = AM_const * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))
            #S   = cc * np.einsum('ijk,i->ijk', cov_3d_pcm(theta[:m], theta[:m].mean(axis = 0)) + np.eye(setup.p) * eps, np.exp(tau))

        # elif m == 300:
        #     mu  = theta[:m].mean(axis = 0)
        #     cov = cov_3d_pcm(theta[:m], mu)
            #S   = AM_const * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))
        
        # else:
        #     pass

        #------------------------------------------------------------------------------------------
        # generate proposal
        theta_cand = cov_theta_cand.gen_cand(theta, m)
        # theta_cand  = (
        #     + theta[m-1]
        #     + np.einsum('ijk,ik->ij', cholesky(cov_theta_cand.S), normal(size = (setup.ntemps, setup.p)))
        #     )
        good_values = setup.checkConstraints(
            tran_unif(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
            )
        #------------------------------------------------------------------------------------------
        # get predictions and SSE
        pred_cand = [_.copy() for _ in pred_curr]
        llik_cand[:] = llik_curr.copy()
        if np.any(good_values):
            llik_cand[:, good_values] = 0.
            for i in range(setup.nexp):
                pred_cand[i][good_values] = setup.models[i].eval(
                    tran_unif(
                        theta_cand[good_values],#.repeat(setup.ns2[i], axis = 0), 
                        setup.bounds_mat, setup.bounds.keys()
                        ), pool=True
                    )
                for t in range(setup.ntemps):
                    llik_cand[i, t] = setup.models[i].llik(setup.ys[i] - discrep_curr[i][t], pred_cand[i][t], marg_lik_cov_curr[i][t])#(((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]).sum(axis = 1)

        #tsq_diff = 0.#((theta_cand * theta_cand).sum(axis = 1) - (theta[m-1] * theta[m-1]).sum(axis = 1))[good_values]
        llik_diff = (llik_cand.sum(axis=0) - llik_curr.sum(axis=0))[good_values] # sum over experiments
        #------------------------------------------------------------------------------------------
        # for each temperature, accept or reject
        alpha[:] = - np.inf
        alpha[good_values] = setup.itl[good_values] * (llik_diff)
        for t in np.where(np.log(uniform(size=setup.ntemps)) < alpha)[0]:
            theta[m,t] = theta_cand[t].copy()
            count[t,t] += 1
            for i in range(setup.nexp):
                llik_curr[i, t] = llik_cand[i, t].copy()
                pred_curr[i][t] = pred_cand[i][t].copy()
            cov_theta_cand.count_100[t] += 1
        #------------------------------------------------------------------------------------------
        # diminishing adaptation based on acceptance rate for each temperature
        #if m>2000:
        #    print('a')


        cov_theta_cand.update_tau(m)

        # if (m % 100 == 0) and (m > 300):
        #     delta = min(0.1, 5 / sqrt(m + 1))
        #     tau[np.where(count_100 < 23)] = tau[np.where(count_100 < 23)] - delta
        #     tau[np.where(count_100 > 23)] = tau[np.where(count_100 > 23)] + delta
        #     count_100 *= 0
        #------------------------------------------------------------------------------------------
        # decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = theta[m].copy()
                theta_cand[:,k] = initfunc_unif(size = setup.ntemps) # independence proposal, will vectorize of columns
                good_values = setup.checkConstraints(
                    tran_unif(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                    )
                pred_cand = [_.copy() for _ in pred_curr]
                llik_cand[:] = llik_curr.copy()

                if np.any(good_values):
                    llik_cand[:, good_values] = 0.
                    for i in range(setup.nexp):
                        pred_cand[i][good_values] = setup.models[i].eval(
                            tran_unif(theta_cand[good_values],#.repeat(setup.ns2[i], axis = 0), 
                            setup.bounds_mat, setup.bounds.keys()), pool=True
                            )
                        for t in range(setup.ntemps):
                            llik_cand[i, t] = setup.models[i].llik(setup.ys[i] - discrep_curr[i][t], pred_cand[i][t], marg_lik_cov_curr[i][t])#(((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]).sum(axis = 1)

                alpha[:] = -np.inf
                #tsq_diff = 0.#((theta_cand * theta_cand).sum(axis = 1) - (theta[m] * theta[m]).sum(axis = 1))[good_values]
                llik_diff = (llik_cand.sum(axis=0) - llik_curr.sum(axis=0))[good_values]
                alpha[good_values] = setup.itl[good_values] * (llik_diff)# + tsq_diff) + 0.5 * tsq_diff # last is for proposal, since this is an independence sampler step
                for t in np.where(np.log(uniform(size = setup.ntemps)) < alpha)[0]:
                    theta[m,t,k] = theta_cand[t,k].copy()
                    count_decor[k,t] += 1
                    for i in range(setup.nexp):
                        pred_curr[i][t] = pred_cand[i][t].copy()
                        llik_curr[i, t] = llik_cand[i, t].copy()


        #------------------------------------------------------------------------------------------
        ## update s2
        for i in range(setup.nexp):

            if setup.models[i].s2=='gibbs':
                ## gibbs update s2       
                
                dev_sq = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] # squared deviations
                log_s2[i][m] = np.log(1 / np.random.gamma(
                        itl_mat[i] * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                        1 / (itl_mat[i] * (setup.ig_b[i] + dev_sq / 2)),
                        ))
                for t in range(setup.ntemps):
                    marg_lik_cov_curr[i][t] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][m][t])[setup.s2_ind[i]])
                    llik_curr[i, t] = setup.models[i].llik(setup.ys[i] - discrep_curr[i][t], pred_curr[i][t], marg_lik_cov_curr[i][t])
                

            else:        
                ## M-H update s2
                    # NOTE: there is something wrong with this...with no tempering, 10 kolski experiments, 
                    # reasonable priors, s2 can diverge for some experiments (not a random walk, has weird patterns).  
                    # This seems to be because of the joint update, but is strange.  Could be that individual updates 
                    # would make it go away, but it shouldn't be there anyway.

                cov_ls2_cand[i].update(log_s2[i], m)
                ls2_candi = cov_ls2_cand[i].gen_cand(log_s2[i], m)
                
                llik_candi = np.zeros(setup.ntemps)
                marg_lik_cov_candi = [None] * setup.ntemps
                for t in range(setup.ntemps):
                    marg_lik_cov_candi[t] = setup.models[i].lik_cov_inv(np.exp(ls2_candi[t])[setup.s2_ind[i]])#s2[i][0, t, setup.s2_ind[i]])
                    llik_candi[t] = setup.models[i].llik(setup.ys[i] - discrep_curr[i][t], pred_curr[i][t], marg_lik_cov_candi[t])
                
                llik_diffi = (llik_candi - llik_curr[i])
                alpha_s2 = setup.itl * (llik_diffi)       
                alpha_s2 += setup.itl * setup.s2_prior_kern[i](np.exp(ls2_candi), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)#ldhc_kern(np.exp(ls2_cand[i])).sum(axis=1)#ldig_kern(np.exp(ls2_cand[i]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 += setup.itl * ls2_candi.sum(axis=1)
                alpha_s2 -= setup.itl * setup.s2_prior_kern[i](np.exp(log_s2[i][m-1]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)#ldhc_kern(np.exp(log_s2[i][m-1])).sum(axis=1)#ldig_kern(np.exp(log_s2[i][m-1]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 -= setup.itl * log_s2[i][m-1].sum(axis=1)

                runif = np.log(uniform(size=setup.ntemps))
                for t in np.where(runif < alpha_s2)[0]:
                    count_s2[i, t] += 1
                    llik_curr[i, t] = llik_candi[t].copy()
                    log_s2[i][m][t] = ls2_candi[t].copy()
                    marg_lik_cov_curr[i][t] = marg_lik_cov_candi[t].copy()
                    cov_ls2_cand[i].count_100[t] += 1

                cov_ls2_cand[i].update_tau(m)
        
        ## tempering swaps
        if m > setup.start_temper and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0.  # Log Probability of Swap
                sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(llik_curr[:, sw.T[0]].sum(axis=0) - llik_curr[:, sw.T[1]].sum(axis=0))
                for i in range(setup.nexp):
                    sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(
                        setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[0]]), setup.ig_a[i], setup.ig_b[i]).sum(axis = 1)
                        -setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[1]]), setup.ig_a[i], setup.ig_b[i]).sum(axis = 1)
                        )
                    if setup.models[i].nd > 0:
                        sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(
                            -.5 * (discrep_vars[i][m][sw.T[0]]**2).sum(axis=1) / setup.models[i].discrep_tau
                            +.5 * (discrep_vars[i][m][sw.T[1]]**2).sum(axis=1) / setup.models[i].discrep_tau
                        )
                for tt in sw[np.where(np.log(uniform(size = setup.nswap_per)) < sw_alpha)[0]]:
                    for i in range(setup.nexp):
                        log_s2[i][m][tt[0]], log_s2[i][m][tt[1]] = log_s2[i][m][tt[1]].copy(), log_s2[i][m][tt[0]].copy()
                        marg_lik_cov_curr[i][tt[0]], marg_lik_cov_curr[i][tt[1]] = marg_lik_cov_curr[i][tt[1]].copy(), marg_lik_cov_curr[i][tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                        discrep_curr[i][tt[0]], discrep_curr[i][tt[1]] = discrep_curr[i][tt[1]].copy(), discrep_curr[i][tt[0]].copy()
                        discrep_vars[i][m][tt[0]], discrep_vars[i][m][tt[1]] = discrep_vars[i][m][tt[1]].copy(), discrep_vars[i][m][tt[0]].copy()
                        llik_curr[i, tt[0]], llik_curr[i, tt[1]] = llik_curr[i, tt[1]].copy(), llik_curr[i, tt[0]].copy()
                        #if np.any(np.exp(log_s2[i][m][0]) > 10*np.exp(log_s2[i][m-1][0])):
                        #    print('bummer2')
                    count[tt[0],tt[1]] += 1
                    theta[m][tt[0]], theta[m][tt[1]] = theta[m][tt[1]].copy(), theta[m][tt[0]].copy()
   
        llik[m] = llik_curr[:, 0].sum()
        #print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    s2 = log_s2.copy()
    for i in range(setup.nexp):
        s2[i] = np.exp(log_s2[i])

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count = count + count.T - np.diag(np.diag(count))
    out = OutCalibPool(theta, s2, count, count_s2, count_decor, cov_theta_cand, cov_ls2_cand, pred_curr, discrep_vars, llik)
    return(out)


##############################################################################################################################################################################


if __name__ == '__main__':
    pass

# EOF
