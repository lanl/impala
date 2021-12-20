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
        return
    def addVecExperiments(self, yobs, model, sd_est, s2_df, s2_ind, theta_ind=None):
        self.ys.append(np.array(yobs))
        self.y_lens.append(len(yobs))
        if theta_ind is None:
            theta_ind = [0]*len(yobs)
        self.theta_ind.append(theta_ind)
        self.ntheta.append(len(set(theta_ind)))
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
        return
    def setTemperatureLadder(self, temperature_ladder):
        self.tl = temperature_ladder
        self.itl = 1/self.tl
        self.ntemps = len(self.tl)
        self.nswap_per = floor(self.ntemps // 2)
        return
    def setMCMC(self, nmcmc, nburn, thin, decor):
        self.nmcmc = nmcmc
        self.nburn = nburn
        self.thin = thin
        self.decor = decor
        return
    def set_max_clusters(self, nclustmax):
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

from collections import namedtuple
OutCalibPool = namedtuple(
    'OutCalibPool', 'theta s2 count count_decor count_100 tau pred_curr',
    )
OutCalibHier = namedtuple(
    'OutCalibHier', 'theta s2 count count_decor count_100 count_decor2 count_temper tau pred_curr theta0 Sigma0',
    )
OutCalibClust = namedtuple(
    'OutCalibClust', 'theta theta_hist s2 count count_temper pred_curr theta0 Sigma0 delta eta nclustmax'
    )

## Hierarchical Calibration

def calibHier(setup):
    t0 = time.time()
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    ntheta = np.sum(setup.ntheta)
    s2     = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    sse    = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    theta  = [
        np.empty([setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p])
        for i in range(setup.nexp)
        ]
    s2_ind_mat = [
        (setup.s2_ind[i][:,None] == range(setup.ntheta[i]))
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
    sse_curr  = [None] * setup.nexp # [i], ntheta[i] x ntemps
    sse_cand  = [None] * setup.nexp # [i], ntheta[i] x ntemps
    dev_sq    = [None] * setup.nexp # [i], ntheta[i] x ntemps
    itl_mat   = [ # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ntheta[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
        ]

    for i in range(setup.nexp):
        theta[i][0] = chol_sample_nper_constraints(
                theta0[0], Sigma0[0], setup.ntheta[i], setup.checkConstraints, 
                setup.bounds_mat, setup.bounds.keys(), setup.bounds,
                )
        pred_curr[i] = setup.models[i].eval(
                tran_unif(theta[i][0].reshape(setup.ntemps * setup.ntheta[i], setup.p),
                    setup.bounds_mat, setup.bounds.keys()),
                )
        pred_cand[i] = pred_curr[i].copy()
        sse_curr[i]  = ((pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][0]
        sse_cand[i]  = sse_curr[i].copy()
        dev_sq[i]    = sse_curr[i].copy()

    eps = 1.0e-12
    AM_SCALAR = 2.4**2/setup.p

    tau = [-0 * np.ones((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    S   = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    cov = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    mu  = [np.empty((setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        S[i][:] = np.eye(setup.p) * 1e-4

    theta0_prior_cov = np.eye(setup.p)*1**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = np.repeat(0.5, setup.p)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]

    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))

    Sigma0_prior_df = setup.p
    Sigma0_prior_scale = np.eye(setup.p)*0.1**2#/setup.p
    Sigma0_dfs = Sigma0_prior_df + ntheta * setup.itl

    Sigma0_ldet_curr = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr  = np.linalg.inv(Sigma0[0])

    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = [np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    count_decor = [np.zeros((setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    count_decor2 = np.zeros((setup.ntemps, setup.p))
    count_100 = [np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]

    theta_cand = [np.empty([setup.ntemps, setup.ntheta[i], setup.p]) for i in range(setup.nexp)]
    theta_cand_mat = [np.empty([setup.ntemps * setup.ntheta[i], setup.p]) for i in range(setup.nexp)]
    theta_eval_mat = [np.empty(theta_cand_mat[i].shape) for i in range(setup.nexp)]

    alpha  = [np.ones((setup.ntemps, setup.ntheta[i])) * -np.inf for i in range(setup.nexp)]
    accept = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    sw_alpha = np.zeros(setup.nswap_per)
    good_values = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    good_values_mat = [
        good_values[i].reshape(setup.ntheta[i] * setup.ntemps) for i in range(setup.nexp)
        ]
    ## start MCMC
    for m in range(1,setup.nmcmc):
        for i in range(setup.nexp):
            theta[i][m] = theta[i][m-1].copy() # current set to previous, will change if accepted
        #------------------------------------------------------------------------------------------
        ## adaptive Metropolis for each temperature / experiment
        if m > 300:
            for i in range(setup.nexp):
                mu[i] += (theta[i][m-1] - mu[i]) / m
                cov[i][:] = (
                    + ((m-1) / m) * cov[i]
                    + ((m-1) / (m * m)) * np.einsum(
                        'tej,tel->tejl', theta[i][m-1] - mu[i], theta[i][m-1] - mu[i],
                        )
                    )
                S[i] = AM_SCALAR * np.einsum(
                    'tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]),
                    )
        elif m == 300:
            for i in range(setup.nexp):
                mu[i][:]  = theta[i][:m].mean(axis = 0)
                cov[i][:] = cov_4d_pcm(theta[i][:m], mu[i])
                S[i][:]   = AM_SCALAR * np.einsum('tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))
        else:
            pass
        #------------------------------------------------------------------------------------------
        # MCMC update for thetas
        for i in range(setup.nexp):
            # Find new candidate values for theta
            theta_eval_mat[i][:] = theta[i][m-1].reshape(setup.ntemps * setup.ntheta[i], setup.p)
            theta_cand[i][:] = chol_sample_1per(theta[i][m-1], S[i])
            theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntemps * setup.ntheta[i], setup.p)
            # Check constraints
            good_values_mat[i][:] = setup.checkConstraints(
                tran_unif(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                )
            good_values[i][:] = good_values_mat[i].reshape(setup.ntemps, setup.ntheta[i])
            # Generate Predictions at new Theta values
            theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][good_values_mat[i]]
            pred_cand[i][:] = setup.models[i].eval(
                    tran_unif(theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys())
                    )
            sse_cand[i][:] = ((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]
            # Calculate log-probability of MCMC accept
            alpha[i][:] = - np.inf
            alpha[i][good_values[i]] = itl_mat[i][good_values[i]] * (
                - 0.5 * (sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]])
                + mvnorm_logpdf_(theta_cand[i], theta0[m-1],
                                    Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                - mvnorm_logpdf_(theta[i][m-1], theta0[m-1],
                                    Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                )
            # MCMC Accept
            accept[i][:] = np.log(uniform(size = alpha[i].shape)) < alpha[i]
            # Where accept, make changes
            theta[i][m][accept[i]]  = theta_cand[i][accept[i]].copy()
            pred_curr[i][accept[i] @ s2_ind_mat[i].T] = \
                             pred_cand[i][accept[i] @ s2_ind_mat[i].T].copy()
            sse_curr[i][accept[i]] = sse_cand[i][accept[i]].copy()
            count[i][accept[i]] += 1
            count_100[i][accept[i]] += 1

        #if m>10000:
        #    print('help')

        # Adaptive Metropolis Update
        if m % 100 == 0 and m > 300:
            delta = min(0.1, 1/np.sqrt(m+1)*5)
            for i in range(setup.nexp):
                tau[i][count_100[i] < 23] -= delta
                tau[i][count_100[i] > 23] += delta
                count_100[i] *= 0

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

        ## Gibbs update s2
        for i in range(setup.nexp):
            dev_sq[i][:] = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i]
            s2[i][m] = 1 / np.random.gamma(
                (itl_mat[i] * setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                1 / (itl_mat[i] * (setup.ig_b[i] +  dev_sq[i] / 2)),
                )
            sse_curr[i][:] = dev_sq[i] / s2[i][m]

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



        # better decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                z = np.random.normal()*.1
                theta0_cand = theta0[m].copy()
                theta0_cand[:,k] += z
                
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
                    good_values[i][:] = good_values_mat[i].reshape(setup.ntemps, setup.ntheta[i])
                    pred_cand[i][:]   = setup.models[i].eval(
                            tran_unif(theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys())
                            )
                    sse_cand[i][:] = ((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m]
                    
                    alpha[i][:] = - np.inf
                    alpha[i][good_values[i]] = (
                        - 0.5 * itl_mat[i][good_values[i]] * (
                            sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]]
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
                
                accept_tot = (np.log(uniform(size = setup.ntemps)) < alpha_tot.sum(axis=1))
                #accept[i][:] = (np.log(uniform(size = alpha[i].shape)) < alpha[i])
                # Where accept, make changes
                theta0[m][accept_tot,:] = theta0_cand[accept_tot,:]
                for i in range(setup.nexp):
                    theta[i][m][accept_tot] = theta_cand[i][accept_tot].copy()
                    pred_curr[i][accept_tot,:] = pred_cand[i][accept_tot,:].copy()
                    sse_curr[i][accept_tot] = sse_cand[i][accept_tot].copy()
                    
                count_decor2[accept_tot, k] = count_decor2[accept_tot, k] + 1


        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0. # reset swap probability
                sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                    - mvnorm_logpdf(theta0[m][sw.T[1]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    + mvnorm_logpdf(theta0[m][sw.T[0]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    - invwishart_logpdf(Sigma0[m][sw.T[1]], Sigma0_prior_df, Sigma0_prior_scale)
                    + invwishart_logpdf(Sigma0[m][sw.T[0]], Sigma0_prior_df, Sigma0_prior_scale)
                    )
                for i in range(setup.nexp):
                    sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                        # for t_0
                        + invgamma_logpdf(s2[i][m][sw.T[0]], setup.ig_a[i], setup.ig_b[i])
                        + mvnorm_logpdf_(theta[i][m][sw.T[0]], theta0[m, sw.T[0]],
                                Sigma0_inv_curr[sw.T[0]], Sigma0_ldet_curr[sw.T[0]]).sum(axis = 1)
                        - 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[0]]
                        - 0.5 * sse_curr[i][sw.T[0]].sum(axis = 1)
                        # for t_1
                        - invgamma_logpdf(s2[i][m][sw.T[1]], setup.ig_a[i], setup.ig_b[i])
                        - mvnorm_logpdf_(theta[i][m][sw.T[1]], theta0[m,sw.T[1]],
                                Sigma0_inv_curr[sw.T[1]], Sigma0_ldet_curr[sw.T[1]]).sum(axis = 1)
                        + 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[1]]
                        + 0.5 * sse_curr[i][sw.T[1]].sum(axis = 1)
                        )
                for tt in sw[np.where(sw[np.log(uniform(size = setup.nswap_per)) < sw_alpha])[0]]:
                    count_temper[tt[0], tt[1]] = count_temper[tt[0], tt[1]] + 1
                    for i in range(setup.nexp):
                        theta[i][m,tt[0]], theta[i][m,tt[1]]     = \
                                            theta[i][m,tt[1]].copy(), theta[i][m,tt[0]].copy()
                        s2[i][m,tt[0]], s2[i][m,tt[1]]               = \
                                            s2[i][m,tt[1]].copy(), s2[i][m,tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = \
                                            pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                        sse_curr[i][tt[0]], sse_curr[i][tt[1]]   = \
                                            sse_curr[i][tt[1]].copy(), sse_curr[i][tt[0]].copy()
                    theta0[m,tt[0]], theta0[m,tt[1]] = theta0[m,tt[1]].copy(), theta0[m,tt[0]].copy()
                    Sigma0[m,tt[0]], Sigma0[m,tt[1]] = Sigma0[m,tt[1]].copy(), Sigma0[m,tt[0]].copy()
                    Sigma0_inv_curr[tt[0]], Sigma0_inv_curr[tt[1]]   = \
                                            Sigma0_inv_curr[tt[1]].copy(), Sigma0_inv_curr[tt[0]].copy()
                    Sigma0_ldet_curr[tt[0]], Sigma0_ldet_curr[tt[1]] = \
                                            Sigma0_ldet_curr[tt[1]].copy(), Sigma0_ldet_curr[tt[0]].copy()

        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count_temper = count_temper + count_temper.T - np.diag(np.diag(count_temper))
    # theta_reshape = [np.swapaxes(t,1,2) for t in theta]
    out = OutCalibHier(theta, s2, count, count_decor, count_100, count_decor2,
                            count_temper, tau, pred_curr, theta0, Sigma0)
    return(out)

## Pooled Calibration
#@profile
def calibPool(setup):
    t0 = time.time()
    theta = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    n_s2 = np.sum(setup.ns2)
    s2 = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    # s2_vec_curr = [s2[i][0,:,setup.s2_ind[i]] for i in range(setup.nexp)]
    s2_ind_mat = [
        (setup.s2_ind[i][:,None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
        ]
    theta_start = initfunc_probit(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(tran_probit(theta_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    while np.any(~good):
        theta_start[np.where(~good)] = initfunc_probit(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran_probit(theta_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta[0] = theta_start

    itl_mat   = [ # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ns2[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
        ]

    pred_curr = [None] * setup.nexp
    # sse_curr = np.empty([setup.ntemps, setup.nexp])
    sse_curr = np.empty([setup.ntemps])
    dev_sq = [np.empty((setup.ntemps, setup.ns2[i])) for i in range(setup.nexp)]

    sse_curr[:] = 0.
    for i in range(setup.nexp):
        pred_curr[i] = setup.models[i].eval(
            tran_probit(theta[0], setup.bounds_mat, setup.bounds.keys()),
            )
        # sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)
        sse_curr += ((pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] / s2[i][0]).sum(axis = 1)

    eps  = 1.0e-13
    tau  = np.repeat(-4.0, setup.ntemps)
    AM_const   = 2.4**2/setup.p
    S    = np.empty([setup.ntemps, setup.p, setup.p])
    S[:] = np.eye(setup.p)*1e-6
    cov  = np.empty([setup.ntemps, setup.p, setup.p])
    mu   = np.empty([setup.ntemps, setup.p])

    count = np.zeros([setup.ntemps, setup.ntemps], dtype = int)
    count_decor = np.zeros([setup.p, setup.ntemps], dtype = int)
    count_100 = np.zeros(setup.ntemps, dtype = int)

    pred_cand = [_.copy() for _ in pred_curr]
    sse_cand = sse_curr.copy()

    alpha    = np.ones(setup.ntemps) * (-np.inf)
    sw_alpha = np.zeros(setup.nswap_per)

    ## start MCMC
    for m in range(1,setup.nmcmc):
        theta[m] = theta[m-1].copy() # current set to previous, will change if accepted
        #----------------------------------------------------------
        ## adaptive Metropolis for each temperature

        if m > 300:
            mu += (theta[m-1] - mu) / m
            cov = (
                + ((m - 1) / m) * cov
                + ((m - 1) / (m * m)) * np.einsum('ti,tj->tij', theta[m-1] - mu, theta[m-1] - mu)
                )
            S   = AM_const * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))
            #S   = cc * np.einsum('ijk,i->ijk', cov_3d_pcm(theta[:m], theta[:m].mean(axis = 0)) + np.eye(setup.p) * eps, np.exp(tau))

        elif m == 300:
            mu  = theta[:m].mean(axis = 0)
            cov = cov_3d_pcm(theta[:m], mu)
            S   = AM_const * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))
        
        else:
            pass
        
        #------------------------------------------------------------------------------------------
        # generate proposal
        theta_cand  = (
            + theta[m-1]
            + np.einsum('ijk,ik->ij', cholesky(S), normal(size = (setup.ntemps, setup.p)))
            )
        good_values = setup.checkConstraints(
            tran_probit(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
            )
        #------------------------------------------------------------------------------------------
        # get predictions and SSE
        pred_cand = [_.copy() for _ in pred_curr]
        sse_cand[:] = sse_curr.copy()
        if np.any(good_values):
            sse_cand[good_values] = 0.
            for i in range(setup.nexp):
                pred_cand[i][good_values] = setup.models[i].eval(
                    tran_probit(
                        theta_cand[good_values],#.repeat(setup.ns2[i], axis = 0), 
                        setup.bounds_mat, setup.bounds.keys()
                        )
                    )
                sse_cand += (((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]).sum(axis = 1)

        tsq_diff = ((theta_cand * theta_cand).sum(axis = 1) - (theta[m-1] * theta[m-1]).sum(axis = 1))[good_values]
        sse_diff = (sse_cand - sse_curr)[good_values] # sum over experiments
        #------------------------------------------------------------------------------------------
        # for each temperature, accept or reject
        alpha[:] = - np.inf
        alpha[good_values] = - 0.5 * setup.itl[good_values] * (sse_diff + tsq_diff)
        for t in np.where(np.log(uniform(size=setup.ntemps)) < alpha)[0]:
            theta[m,t] = theta_cand[t]
            count[t,t] += 1
            sse_curr[t] = sse_cand[t]
            for i in range(setup.nexp):
                pred_curr[i][t] = pred_cand[i][t]
            count_100[t] += 1
        #------------------------------------------------------------------------------------------
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > 300):
            delta = min(0.1, 5 / sqrt(m + 1))
            tau[np.where(count_100 < 23)] = tau[np.where(count_100 < 23)] - delta
            tau[np.where(count_100 > 23)] = tau[np.where(count_100 > 23)] + delta
            count_100 *= 0
        #------------------------------------------------------------------------------------------
        # decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = theta[m].copy()
                theta_cand[:,k] = initfunc_probit(size = setup.ntemps) # independence proposal, will vectorize of columns
                good_values = setup.checkConstraints(
                    tran_probit(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                    )
                pred_cand = [_.copy() for _ in pred_curr]
                sse_cand[:] = sse_curr.copy()

                if np.any(good_values):
                    sse_cand[good_values] = 0.
                    for i in range(setup.nexp):
                        pred_cand[i][good_values] = setup.models[i].eval(
                            tran_probit(theta_cand[good_values],#.repeat(setup.ns2[i], axis = 0), 
                            setup.bounds_mat, setup.bounds.keys()),
                            )
                        sse_cand += (((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]).sum(axis = 1)

                alpha[:] = -np.inf
                tsq_diff = ((theta_cand * theta_cand).sum(axis = 1) - (theta[m] * theta[m]).sum(axis = 1))[good_values]
                sse_diff = (sse_cand - sse_curr)[good_values]
                alpha[good_values] = -0.5 * setup.itl[good_values] * (sse_diff + tsq_diff) + 0.5 * tsq_diff # last is for proposal, since this is an independence sampler step
                for t in np.where(np.log(uniform(size = setup.ntemps)) < alpha)[0]:
                    theta[m,t,k] = theta_cand[t,k]
                    count_decor[k,t] += 1
                    for i in range(setup.nexp):
                        pred_curr[i][t] = pred_cand[i][t]
                    sse_curr[t] = sse_cand[t]
        #------------------------------------------------------------------------------------------
        ## Gibbs update s2
        sse_curr[:] = 0.
        for i in range(setup.nexp):
            dev_sq[i][:] = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] # squared deviations
            s2[i][m] = 1 / np.random.gamma(
                    itl_mat[i] * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                    1 / (itl_mat[i] * (setup.ig_b[i] + dev_sq[i] / 2)),
                    )
            sse_curr += (dev_sq[i] / s2[i][m]).sum(axis = 1)
        #------------------------------------------------------------------------------------------
        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0.  # Log Probability of Swap
                sw_alpha += 0.5 * (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(sse_curr[sw.T[1]] - sse_curr[sw.T[0]])
                for i in range(setup.nexp):
                    sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(
                        - 0.5 * (setup.ny_s2[i] * np.log(s2[i][m][sw.T[0]])).sum(axis = 1)
                        - ((setup.ig_a[i] + 1) * np.log(s2[i][m][sw.T[0]])).sum(axis = 1)
                        - (setup.ig_b[i] / s2[i][m][sw.T[0]]).sum(axis = 1)
                        + 0.5 * (setup.ny_s2[i] * np.log(s2[i][m][sw.T[1]])).sum(axis = 1)
                        + ((setup.ig_a[i] + 1) * np.log(s2[i][m][sw.T[1]])).sum(axis = 1)
                        + (setup.ig_b[i] / s2[i][m][sw.T[1]]).sum(axis = 1)
                        )
                sw_alpha = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * ( # probit transform jacobian
                     - 0.5 * (theta[m][sw.T[0]]**2).sum(axis = 1)
                     + 0.5 * (theta[m][sw.T[1]]**2).sum(axis = 1)
                     )
                for tt in sw[np.where(np.log(uniform(size = setup.nswap_per)) < sw_alpha)[0]]:
                    count[tt[0],tt[1]] += 1
                    theta[m][tt[0]], theta[m][tt[1]] = theta[m][tt[1]].copy(), theta[m][tt[0]].copy()
                    sse_curr[tt[0]], sse_curr[tt[1]] = sse_curr[tt[1]].copy(), sse_curr[tt[0]].copy()
                    for i in range(setup.nexp):
                        s2[i][m][tt[0]], s2[i][m][tt[1]] = s2[i][m][tt[1]].copy(), s2[i][m][tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                    # s2_vec_curr = [s2[i][m,:,setup.s2_ind[i]] for i in range(setup.nexp)]
            
        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count = count + count.T - np.diag(np.diag(count))
    out = OutCalibPool(theta, s2, count, count_decor, count_100, tau, pred_curr)
    return(out)

def sample_delta_per_temperature(curr_delta_t, njs_t, log_posts_t, inv_temp_t, eta_t):
    """ Given log-posteriors, current delta -- 
        iterates through delta assignments, reassigning probabilistically. """
    djs = (njs_t == 0) * eta_t / (sum(njs_t == 0) + 1e-9)
    temp = np.empty(log_posts_t.shape[0])
    unis = uniform(size = curr_delta_t.shape[0])
    for s in range(curr_delta_t.shape[0]):
        njs_t[curr_delta_t[s]] -= 1
        temp[:] = - np.inf
        temp[(njs_t + djs) > 0] = ( # log-posterior under each cluster
            inv_temp_t * (np.log((njs_t + djs)[njs_t + djs > 0]) + log_posts_t.T[s, njs_t + djs > 0])
            )
        temp[:] -= temp.max() # 
        # temp[:] = np.exp(temp) / np.exp(temp).sum()
        temp[:] = np.cumsum(np.exp(temp))
        temp[:] /= temp[-1] # normalized cumulative sum of probability of cluster choice
        curr_delta_t[s] = (unis[s] >= temp).sum()
        # curr_delta_t[s] = choice(log_posts_t.shape[0], p = temp)
        njs_t[curr_delta_t[s]] += 1
        djs[njs_t > 0] = 0.
    return curr_delta_t

def sample_delta_per_temperature_wrapper(args):
    return sample_delta_per_temperature(*args)

def sample_delta(curr_delta, njs, log_posts, inv_temps, eta, pool = None):
    args = zip(curr_delta, njs, log_posts, inv_temps, eta)
    if pool is None:
        delta = np.array(list(map(sample_delta_per_temperature_wrapper, args)))
    else:  
        delta = np.array(list(pool.map(sample_delta_per_temperature_wrapper,args)))
    return delta

def one_step_in_cluster_covariance_update(Sj, mSj, nSj, n, mus, covs):
    """
    Online update of covariance matrix -- as stepping through cluster assignments
    S, m, n -- modified in place

    Sj: view of covariance matrix
    Adapted from: https://tinyurl.com/onlinecovariance
    Sj   is [ntemps x p x p]
    nSj  is [ntemps]
    mSj  is [ntemps]
    mus  is [ntemps x p]
    covs is [ntemps x p x p]
    """
    nC  = nSj + n # ntemps
    mC  = (nSj.reshape(-1,1) * mSj + n * mus) / nC.reshape(-1,1) # ntemps x p
    Sj[:] = 1 / nC.reshape(-1,1,1) * (
        + nSj.reshape(-1,1,1) * Sj
        + n * covs
        + np.einsum('t,tp,tq->tpq', nSj, mSj - mC, mSj - mC)
        + n * np.einsum('tp,tq->tpq', mus - mC, mus - mC)
        )
    mSj[:] = mC
    nSj[:] = nC
    return Sj, mSj, nSj

def cluster_covariance_update_old(S, mS, nS, n, delta, covs, mus, nexp, nclustmax, temps):
    """
    S, m, n -- modified in place
    Goal is cluster-level covariance matrix
    input:
        means [i]x[ntemps, ns2[i], p], 
        covs  [i]x[ntemps, ns2[i], p, p], 
        delta [i]x[ntemps, ns2[i]]
    output: 
        S     [ntemps, nclustmax, p, p]
    """
    S[:]  = np.eye(S.shape[-1]) * 1e-9
    mS[:] = 0.
    nS[:] = 0.
    for i in range(nexp):
        for j in range(delta[i].shape[1]):
            S[temps, delta[i].T[j]], mS[temps, delta[i].T[j]], nS[temps, delta[i].T[j]] = \
                one_step_in_cluster_covariance_update(
                    S[temps,delta[i].T[j]], mS[temps, delta[i].T[j]], nS[temps, delta[i].T[j]],
                    n, mus[i][temps, j], covs[i][temps, j]
                    )
    return

def cluster_covariance_update(S, mS, nS, n, delta, covs, mus, nexp, nclustmax, temps):
    S[:]  = 0. 
    mS[:] = 0.
    nS[:] = 0.
    mC = np.empty((temps.shape[0], S.shape[-1]))
    nC = np.zeros((temps.shape[0],1))
    for i in range(nexp):
        for j in range(delta[i].shape[1]):
            nC[:] = nS[temps, delta[i].T[j], None] + n
            mC[:] = (nS[temps, delta[i].T[j],None] * mS[temps, delta[i].T[j]] + n * mus[i][temps, j]) / nC
            S[temps, delta[i].T[j]] =  1 / nC[:,:,None] * (
                + nS[temps, delta[i].T[j],None,None] * S[temps, delta[i].T[j]]
                + n * covs[i][temps, j]
                + np.einsum('t,tp,tq->tpq', nS[temps, delta[i].T[j]], 
                                mS[temps, delta[i].T[j]] - mC, mS[temps, delta[i].T[j]] - mC)
                + n * np.einsum('tp,tq->tpq', mus[i][temps, j] - mC, mus[i][temps, j] - mC)
                )
            mS[temps, delta[i].T[j]] = mC
            nS[temps, delta[i].T[j]] = nC.ravel()
    S[:] += np.eye(S.shape[-1]) * 1e-9
    return

def sample_eta(curr_eta, nclust, ndat):
    g = beta(curr_eta + 1, ndat)
    aa = 2 + nclust
    bb = 0.1 - np.log(g)
    eps = (aa - 1) / (ndat * bb + aa - 1)
    sel = (uniform(size = aa.shape) < eps) * 1
    aaa = np.vstack((aa,aa - 1)).T[np.arange(aa.shape[0]), sel]
    return gamma(shape = aaa, scale = 1 / bb)

## DP Cluster Calibration

def calibClust(setup, parallel = False):
    t0 = time.time()
    if parallel:
        pool = mp.Pool(processes = mp.cpu_count())
    else:
        pool = None
    ## Constants Declaration    
    s2_ind_mat = [(setup.s2_ind[i][:,None] == range(setup.ns2[i])) for i in range(setup.nexp)]
    ntheta_cand = 10
    ntotexp = sum(setup.ns2)
    temps = np.arange(setup.ntemps)
    itl_mat_theta = (np.ones((setup.ntemps, setup.nclustmax)).T * setup.itl).T
    itl_mat_s2 = [
        np.ones((setup.ntemps, setup.ns2[i])) * setup.itl.reshape(-1,1)
        for i in range(setup.nexp)
        ]
    eps = 1.0e-12
    AM_SCALAR = 2.4**2/setup.p
    ## Parameter Declaration
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    theta0[0] = chol_sample_1per_constraints(
            np.zeros((setup.ntemps, setup.p)), np.array([np.eye(setup.p)] * setup.ntemps),
            setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds,
            )
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    Sigma0[0] = np.eye(setup.p) * 0.25**2
    # initialize delta
    delta  = [np.empty([setup.nmcmc, setup.ntemps, setup.ns2[i]], dtype = int) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        delta[i][0] = choice(setup.nclustmax, size = (setup.ntemps, setup.ns2[i]))
    delta_ind_mat = [(delta[i][0,:,:,None] == range(setup.nclustmax)) for i in range(setup.nexp)]
    njs = np.zeros((setup.ntemps, setup.nclustmax), dtype = int) # (extant) cluster weight for ijth experiment
    not_njs = np.zeros(njs.shape, dtype = bool)                  # (njs > 0) -> false
    djs = np.empty(njs.shape)                                    # (candidate) cluster weight for ijth experiment
    cluster_weights = np.empty(njs.shape)
    for i in range(setup.nexp):
        njs[:] += bincount2D_vectorized(delta[i][0], setup.nclustmax)
    curr_delta    = [delta[i][0].copy() for i in range(setup.nexp)]
    theta_unravel = [np.repeat(np.arange(setup.ntemps), setup.ns2[i]) for i in range(setup.nexp)]
    # initialize Theta
    theta_long_shape = (setup.ntemps * setup.nclustmax, setup.p)
    theta_wide_shape = (setup.ntemps, setup.nclustmax, setup.p)
    theta      = np.empty((setup.nmcmc, setup.ntemps, setup.nclustmax, setup.p))
    theta[0]   = chol_sample_nper_constraints(
           theta0[0], Sigma0[0], setup.nclustmax, setup.checkConstraints,
           setup.bounds_mat, setup.bounds.keys(), setup.bounds,
           )
    theta_hist = [np.empty((setup.nmcmc, setup.ntemps, setup.ns2[i], setup.p)) for i in range(setup.nexp)]
    theta_cand = np.empty(theta_wide_shape)
    for i in range(setup.nexp):
        theta_hist[i][0] = (
            theta[0, theta_unravel[i], delta[i][0].ravel()].reshape(setup.ntemps, setup.ns2[i], setup.p)
            )
    theta_eval = np.empty(theta_wide_shape)
    theta_ext = np.zeros((setup.ntemps, setup.nclustmax), dtype = bool) # array of (current) extant theta locs
    for i in range(setup.nexp):
        theta_ext[theta_unravel, delta[0][0].ravel()] += True
    ntheta = theta_ext.sum(axis = 1) # count extant thetas
    # initialize sigma2 and eta
    s2 = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    eta = np.empty((setup.nmcmc, setup.ntemps))
    eta[0] = 5.  
    ## Initialize *Current* Variables
    pred_curr       = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    sse_curr        = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    dev_sq          = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    pred_cand_delta = [None] * setup.nexp # [i] x [ntemps, nclustmax, ylens[i]]
    sse_cand_delta  = [None] * setup.nexp # [i] x [ntemps, nclustmax, ns2[i]]
    pred_cand_theta = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    pred_curr_theta = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    sse_cand_theta  = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    sse_curr_theta  = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    sse_cand_theta_ = np.zeros((setup.ntemps, setup.nclustmax)) # above, Summarized by cluster assignment
    sse_curr_theta_ = np.zeros((setup.ntemps, setup.nclustmax)) #
    for i in range(setup.nexp):
        pred_curr[i] = setup.models[i].eval(
            tran_unif(theta_hist[i][0].reshape(-1, setup.p), 
                    setup.bounds_mat, setup.bounds.keys()), 
            False,
            )
        dev_sq[i] = ((pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][0]
        sse_curr[i]  = dev_sq[i] / s2[i][0]
        pred_cand_delta[i] = setup.models[i].eval(
            tran_unif(theta[0].reshape(-1, setup.p), setup.bounds_mat, setup.bounds.keys()), True,
            )
        sse_cand_delta[i] = (
            (pred_cand_delta[i] @ s2_ind_mat[i]).reshape(setup.ntemps, setup.nclustmax, setup.ns2[i])
            / s2[i][0].reshape(setup.ntemps, 1, setup.ns2[i])
            )
        pred_cand_theta[i] = setup.models[i].eval(
            tran_unif(theta_hist[i][0].reshape(-1,setup.p), setup.bounds_mat, setup.bounds.keys()), False,
            )
        pred_curr_theta[i] = pred_cand_theta[i].copy()
        sse_cand_theta[i] = (pred_cand_theta[i] @ s2_ind_mat[i]) / s2[i][0]
        sse_curr_theta[i] = sse_cand_theta[i].copy()
    ## Initialize Adaptive Metropolis related Variables
    S   = np.empty((setup.ntemps, setup.nclustmax, setup.p, setup.p))
    S[:] = np.eye(setup.p) * 1e-4
    nS  = np.zeros((setup.ntemps, setup.nclustmax))
    mS  = np.zeros((setup.ntemps, setup.nclustmax, setup.p))
    cov = [np.empty((setup.ntemps, setup.ns2[i], setup.p, setup.p)) for i in range(setup.nexp)]
    mu  = [np.empty((setup.ntemps, setup.ns2[i], setup.p)) for i in range(setup.nexp)]
    ## Initialize Theta0 and Sigma0 related variables 
    theta0_prior_cov = np.eye(setup.p)*1.**2 # was 10^2, but that is far from uniform when back transforming
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = np.repeat(0.5, setup.p)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]
    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))
    Sigma0_prior_df    = setup.p
    Sigma0_prior_scale = np.eye(setup.p)*.1**2#/setup.p
    Sigma0_dfs         = Sigma0_prior_df + setup.nclustmax * setup.itl
    Sigma0_scales      = np.empty((setup.ntemps, setup.p, setup.p))
    Sigma0_scales[:]   = Sigma0_prior_scale
    Sigma0_ldet_curr   = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr    = np.linalg.inv(Sigma0[0])
    cc = np.empty((setup.ntemps, setup.p, setup.p))
    dd = np.empty((setup.ntemps, setup.p))    
    ## Initialize Counters
    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = np.zeros(setup.ntemps)
    ## Initialize Alphas
    alpha_long_shape = (setup.ntemps * setup.nclustmax,)
    alpha_wide_shape = (setup.ntemps, setup.nclustmax,)
    alpha = np.ones(alpha_wide_shape) * -np.inf
    accept = np.zeros(alpha_wide_shape, dtype = bool)
    sw_alpha = np.zeros(setup.nswap_per)
    good_values = np.zeros(alpha_wide_shape, dtype = bool)

    delta_size = [(setup.ntemps, setup.nclustmax, setup.ns2[i]) for i in range(setup.nexp)]
    cluster_cum_prob = np.empty((setup.ntemps, setup.nclustmax))
    cluster_sample_unif = [np.empty((setup.ntemps, setup.ns2[i])) for i in range(setup.nexp)]

    tau  = np.repeat(-0.0, setup.ntemps)
    count_100 = np.zeros(setup.ntemps, dtype = int)

    ## start MCMC
    for m in range(1,setup.nmcmc):
        #------------------------------------------------------------------------------------------
        ## Gibbs Update for delta (cluster identifier)
        for i in range(setup.nexp):
            delta[i][m] = delta[i][m-1]
        
        #   Establish initial cluster weighting
        njs[:] = 0.
        for l in range(setup.nexp):
            njs[:] += bincount2D_vectorized(delta[l][m], setup.nclustmax)
        not_njs[:] = (njs == 0) # fixed at start of iteration
                                # if (non-extant) cluster j becomes extant, then set this to False.

        for i in range(setup.nexp):
            pred_cand_delta[i][:] = setup.models[i].eval(
                tran_unif(
                    theta[m-1].reshape(-1, setup.p).repeat(setup.ns2[i], axis = 0),
                    setup.bounds_mat, 
                    setup.bounds.keys(),
                    ),
                )
            sse_cand_delta[i][:] = (
                ((pred_cand_delta[i] - setup.ys[i])**2 @ s2_ind_mat[i]).reshape(delta_size[i])
                / s2[i][m-1,:,None,:]
                )
            # delta[i][m] = delta[i][m-1]
            cluster_sample_unif[i][:] = uniform(size = (setup.ntemps, setup.ns2[i]))
            for j in range(setup.ns2[i]):
                # weighting assigned to extant clusters for ij'th within-vectorized experiment
                njs[temps, delta[i][m,:,j]] -= 1
                # weighting assigned to candidate (non-extant) clusters
                djs[:] = not_njs * (eta[m-1] / (not_njs.sum(axis = 1) + 1e-9)).reshape(-1,1)
                # djs[:] = (njs == 0) * (eta[m-1] / (njs == 0).sum(axis = 1)).reshape(-1,1)
                # unnormalized log-probability of cluster membership
                with np.errstate(divide='ignore', invalid = 'ignore'):
                    cluster_cum_prob[:] = (
                        + np.log(njs + djs) 
                        - 0.5 * sse_cand_delta[i][:,:,j] * setup.itl.reshape(-1,1)
                        )
                # np.nan_to_num(cluster_cum_prob, False, nan = -np.inf, neginf = -np.inf)
                #---fix for numerical stability in np.exp
                cluster_cum_prob   -= cluster_cum_prob.max(axis = 1).reshape(-1,1)
                # Un-normalized cumulative probability of cluster membership
                cluster_cum_prob[:] = np.exp(cluster_cum_prob).cumsum(axis = 1)
                # Normalized cumulative probability of cluster membership
                cluster_cum_prob /= cluster_cum_prob[:,-1].reshape(-1,1)
                delta[i][m,:,j] = (
                    (cluster_sample_unif[i][:,j].reshape(-1,1) > cluster_cum_prob).sum(axis = 1)
                    )
                # Fix weights using new cluster assignments for ij'th within-vectorized experiment
                njs[temps, delta[i][m,:,j]] += 1
                # If a candidate cluster became extant--remove candidate flag.
                not_njs[njs > 0] = False
            
            delta_ind_mat[i][:] = delta[i][m,:,:,None] == range(setup.nclustmax)
            curr_delta[i][:] = delta[i][m]
        #------------------------------------------------------------------------------------------
        ## adaptive Metropolis per Cluster
        if m > 300:
            for i in range(setup.nexp):
                mu[i] += (theta_hist[i][m-1] - mu[i]) / m
                cov[i][:] = (
                    + ((m-1) / m) * cov[i]
                    + ((m-1) / (m * m)) * np.einsum(
                        'tej,tel->tejl', theta_hist[i][m-1] - mu[i], theta_hist[i][m-1] - mu[i],
                        )
                    )
            cluster_covariance_update(
                S, mS, nS, m, curr_delta, cov, mu, 
                setup.nexp, setup.nclustmax, temps,
                )
            S   = AM_SCALAR * np.einsum('ijkl,i->ijkl', S, np.exp(tau))
        elif m == 300:
            for i in range(setup.nexp):
                mu[i][:]  = theta_hist[i][:m].mean(axis = 0)
                cov[i][:] = cov_4d_pcm(theta_hist[i][:m], mu[i])
            cluster_covariance_update(
                S, mS, nS, m, curr_delta, cov, mu, 
                setup.nexp, setup.nclustmax, temps,
                )
            S   = AM_SCALAR * np.einsum('ijkl,i->ijkl', S, np.exp(tau))
        else:
            pass
        #------------------------------------------------------------------------------------------
        # MCMC within Gibbs update for thetas (in cluster)
        sse_cand_theta_[:] = 0.
        sse_curr_theta_[:] = 0.
        theta[m]      = theta[m-1]
        theta_eval[:] = theta[m-1]
        theta_cand[:] = chol_sample_1per(theta[m-1], S)
        good_values[:] = setup.checkConstraints(
            tran_unif(theta_cand.reshape(theta_long_shape), setup.bounds_mat, setup.bounds.keys()), 
            setup.bounds,
            ).reshape(setup.ntemps, setup.nclustmax)
        theta_eval[good_values] = theta_cand[good_values]
        for i in range(setup.nexp):
            pred_cand_theta[i][:] = setup.models[i].eval(
                tran_unif(theta_eval[theta_unravel[i], delta[i][m].ravel()], 
                        setup.bounds_mat, setup.bounds.keys())
                )
            pred_curr_theta[i][:] = setup.models[i].eval(
                tran_unif(theta[m-1, theta_unravel[i], delta[i][m].ravel()],
                        setup.bounds_mat, setup.bounds.keys())
                )
            sse_cand_theta[i][:] = ((pred_cand_theta[i] - setup.ys[i])**2 @ s2_ind_mat[i] / s2[i][m-1])
            sse_curr_theta[i][:] = ((pred_curr_theta[i] - setup.ys[i])**2 @ s2_ind_mat[i] / s2[i][m-1])
            sse_cand_theta_[:] += np.einsum('te,tek->tk', sse_cand_theta[i], delta_ind_mat[i])
            sse_curr_theta_[:] += np.einsum('te,tek->tk', sse_curr_theta[i], delta_ind_mat[i])
        
        alpha[:] = - np.inf
        alpha[good_values] = (itl_mat_theta * (
            - 0.5 * (sse_cand_theta_ - sse_curr_theta_)
            + mvnorm_logpdf_(theta_cand, theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)
            - mvnorm_logpdf_(theta[m-1], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)
            ))[good_values]
        accept[:] = np.log(uniform(size = alpha.shape)) < alpha
        theta[m,accept] = theta_cand[accept]

        for i in range(setup.nexp):
            theta_hist[i][m] = (
                theta[m, theta_unravel[i], delta[i][m].ravel()
                    ].reshape(setup.ntemps, setup.ns2[i], setup.p)
                )
            pred_curr[i][:] = setup.models[i].eval(
                tran_unif(theta_hist[i][m].reshape(-1,setup.p),setup.bounds_mat, setup.bounds.keys()),
                False,
                )
        count += accept.sum(axis = 1)
        theta_ext[:] = False
        for i in range(setup.nexp):
            theta_ext[theta_unravel, delta[i][m].ravel()] += True
        ntheta[:] = theta_ext.sum(axis = 1)

        if np.abs(theta_hist[0][m,0,0,2])>5:
            print('as')

        #------------------------------------------------------------------------------------------
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > 300):
            ddelta = min(0.1, 5 / sqrt(m + 1))
            tau[np.where(count_100 < 23)] = tau[np.where(count_100 < 23)] - ddelta
            tau[np.where(count_100 > 23)] = tau[np.where(count_100 > 23)] + ddelta
            count_100 *= 0
        #------------------------------------------------------------------------------------------
        ## Gibbs update s2
        for i in range(setup.nexp):
            dev_sq[i][:] = (pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i]
            s2[i][m] = 1 / np.random.gamma(
                (itl_mat_s2[i] * setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                1 / (itl_mat_s2[i] * (setup.ig_b[i] +  dev_sq[i] / 2)),
                )
            sse_curr[i][:] = dev_sq[i] / s2[i][m]
        ## Gibbs update theta0
        cc[:] = np.linalg.inv(
            np.einsum('t,tpq->tpq', ntheta * setup.itl, Sigma0_inv_curr) + theta0_prior_prec,
            )
        tbar[:] = (theta[m] * theta_ext.reshape(setup.ntemps, setup.nclustmax, 1)).sum(axis = 1)
        tbar[:] /= ntheta.reshape(setup.ntemps, 1)
        dd[:] = (
            + np.einsum('t,tl->tl', setup.itl, np.einsum('t,tlk,tk->tl', ntheta, Sigma0_inv_curr, tbar))
            + np.dot(theta0_prior_prec, theta0_prior_mean)
            )
        theta0[m][:] = chol_sample_1per_constraints(
            np.einsum('tlk,tk->tl', cc, dd), cc,
            setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds,
            )
        if m>2000 and theta0[m,0,0]<probit(.2):
            print('asdf')
        #------------------------------------------------------------------------------------------
        ## Gibbs update Sigma0
        mat[:] *= 0.
        #for i in range(setup.nexp): # This loop appears to be extra (just inflating) - devin
        
        #mat3 = mat*0
        #mat3 += np.einsum(
        #        'tnp,tnq->tpq', 
        #        (theta[m] - theta0[m].reshape(setup.ntemps, 1, setup.p))* theta_ext.reshape(setup.ntemps, setup.nclustmax, 1), 
        #        (theta[m] - theta0[m].reshape(setup.ntemps, 1, setup.p))* theta_ext.reshape(setup.ntemps, setup.nclustmax, 1),
        #        )

        mat += np.einsum(
                't,tnp,tnq->tpq',
                setup.itl,
                ((theta[m] - theta0[m].reshape(setup.ntemps, 1, setup.p)) 
                    * theta_ext.reshape(setup.ntemps, setup.nclustmax, 1)),
                ((theta[m] - theta0[m].reshape(setup.ntemps, 1, setup.p)) 
                    * theta_ext.reshape(setup.ntemps, setup.nclustmax, 1)),
                )
        #mat2 = mat*0.
        #for ii in np.where(theta_ext.reshape(setup.ntemps, setup.nclustmax, 1))[1]:
        #    mat2 += (theta[m,0,ii,:]-theta0[m,0,:]).reshape(setup.p,1) @ (theta[m,0,ii,:]-theta0[m,0,:]).reshape(setup.p,1).T
            #theta[m,0,np.where(theta_ext.reshape(setup.ntemps, setup.nclustmax, 1))[1],:].T @ theta[m,0,np.where(theta_ext.reshape(setup.ntemps, setup.nclustmax, 1))[1],:]

        Sigma0_scales[:] = Sigma0_prior_scale + mat
        Sigma0_dfs[:]    = Sigma0_prior_df + theta_ext.sum(axis = 1) * setup.itl
        for t in range(setup.ntemps):
            Sigma0[m,t] = invwishart.rvs(df = Sigma0_dfs[t], scale = Sigma0_scales[t])
        Sigma0_ldet_curr[:] = np.linalg.slogdet(Sigma0[m])[1]
        Sigma0_inv_curr[:] = np.linalg.inv(Sigma0[m])


        #if m>5000 :
        #    print('ss')

        #------------------------------------------------------------------------------------------
        ## Gibbs update for theta (not in cluster)
        theta_cand[:] = chol_sample_nper_constraints(
                theta0[m], Sigma0[m], setup.nclustmax, setup.checkConstraints,
                setup.bounds_mat, setup.bounds.keys(), setup.bounds,
                )
        theta[m,~theta_ext] = theta_cand[~theta_ext]    
        #------------------------------------------------------------------------------------------
        ## Gibbs Update eta
        eta[m] = sample_eta(eta[m-1], ntheta, sum(setup.ns2[i] for i in range(setup.nexp)))
        #------------------------------------------------------------------------------------------
        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0. # reset swap probability
                sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                    - mvnorm_logpdf(theta0[m][sw.T[1]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    + mvnorm_logpdf(theta0[m][sw.T[0]], theta0_prior_mean,
                                        theta0_prior_prec, theta0_prior_ldet)
                    - invwishart_logpdf(Sigma0[m][sw.T[1]], Sigma0_prior_df, Sigma0_prior_scale)
                    + invwishart_logpdf(Sigma0[m][sw.T[0]], Sigma0_prior_df, Sigma0_prior_scale)
                    - gamma_logpdf(eta[m][sw.T[1]], 2, 0.1)
                    + gamma_logpdf(eta[m][sw.T[0]], 2, 0.1)
                    )
                for i in range(setup.nexp):
                    sw_alpha[:] = sw_alpha + (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                        # for t_0
                        + invgamma_logpdf(s2[i][m][sw.T[0]], setup.ig_a[i], setup.ig_b[i])
                        + (mvnorm_logpdf_(theta[m,sw.T[0]], theta0[m,sw.T[0]],
                                Sigma0_inv_curr[sw.T[0]], Sigma0_ldet_curr[sw.T[0]],
                                ) * theta_ext[sw.T[0]]).sum(axis = 1)
                        - 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[0]]
                        - 0.5 * sse_curr[i][sw.T[0]].sum(axis = 1)
                        # for t_1
                        - invgamma_logpdf(s2[i][m][sw.T[1]], setup.ig_a[i], setup.ig_b[i])
                        - (mvnorm_logpdf_(theta[m,sw.T[1]], theta0[m,sw.T[1]], 
                                Sigma0_inv_curr[sw.T[1]], Sigma0_ldet_curr[sw.T[1]]
                                ) * theta_ext[sw.T[1]]).sum(axis = 1)
                        + 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[1]]
                        + 0.5 * sse_curr[i][sw.T[1]].sum(axis = 1)
                        )
                for tt in sw[np.where(sw[np.log(uniform(size = setup.nswap_per)) < sw_alpha])[0]]:
                    count_temper[tt[0], tt[1]] = count_temper[tt[0], tt[1]] + 1
                    theta[m,tt[0]], theta[m,tt[1]] = theta[m,tt[1]].copy(), theta[m,tt[0]].copy()
                    theta0[m,tt[0]], theta0[m,tt[1]] = theta0[m,tt[1]].copy(), theta0[m,tt[0]].copy()
                    Sigma0[m,tt[0]], Sigma0[m,tt[1]] = Sigma0[m,tt[1]].copy(), Sigma0[m,tt[0]].copy()
                    Sigma0_inv_curr[tt[0]], Sigma0_inv_curr[tt[1]]     = \
                                            Sigma0_inv_curr[tt[1]].copy(), Sigma0_inv_curr[tt[0]].copy()
                    Sigma0_ldet_curr[tt[0]], Sigma0_ldet_curr[tt[1]]   = \
                                            Sigma0_ldet_curr[tt[1]].copy(), Sigma0_ldet_curr[tt[0]].copy()
                    for i in range(setup.nexp):
                        theta_hist[i][m,tt[0]], theta_hist[i][m,tt[1]] = \
                                            theta_hist[i][m,tt[1]].copy(), theta_hist[i][m,tt[0]].copy()
                        delta[i][m,tt[0]], delta[i][m,tt[1]]           = \
                                            delta[i][m,tt[1]].copy(), delta[i][m,tt[0]].copy()
                        s2[i][m,tt[0]], s2[i][m,tt[1]] = s2[i][m,tt[1]].copy(), s2[i][m,tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]]       = \
                                            pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                        sse_curr[i][tt[0]], sse_curr[i][tt[1]]         = \
                                            sse_curr[i][tt[1]].copy(), sse_curr[i][tt[0]].copy()
                    theta_ext[tt[0]], theta_ext[tt[1]] = theta_ext[tt[1]].copy(), theta_ext[tt[0]].copy()
        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')
    #------------------------------------------------------------------------------------------
    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count_temper = count_temper + count_temper.T - np.diag(np.diag(count_temper))
    out = OutCalibClust(theta, theta_hist, s2, count, count_temper, 
                            pred_curr, theta0, Sigma0, delta, eta, setup.nclustmax)
    return(out)

if __name__ == '__main__':
    pass

# EOF
