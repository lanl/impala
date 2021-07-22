import numpy as np
#import pyBASS as pb
#import physical_models_vec as pm_vec
#from scipy.interpolate import interp1d
import time
import scipy
from scipy import stats
from numpy.random import uniform, normal
from math import sqrt, floor
from scipy.special import erf, erfinv, gammaln
from scipy.stats import invwishart
from numpy.linalg import cholesky, slogdet

#####################
# class for setting everything up
#####################

class CalibSetup:
    """Structure for storing calibration experimental data, likelihood, discrepancy, etc."""
    def __init__(self, bounds, constraint_func=None):
        self.nexp = 0 # not the true number of experiments, but the number of separate vectorized pieces (number of independent emulators + 1 for PTW)
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
    def addVecExperiments(self, yobs, model, sd_est, s2_df, s2_ind, theta_ind=None): # yobs should be a vector
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

# initfunc = np.random.normal # if probit, then normal--if uniform, then uniform
initfunc = np.random.uniform

def tran(th, bounds, names):
    return dict(zip(names, unnormalize(th, bounds).T))

def tran2(th, bounds, names):
    return dict(zip(names, unnormalize(invprobit(th),bounds).T))

def chol_sample(mean, cov):
    return mean + np.dot(np.linalg.cholesky(cov), np.random.standard_normal(mean.size))

def chol_sample_1per(means, covs):
    return means + np.einsum('tnpq,ntq->ntp', cholesky(covs), normal(size = means.shape))

def chol_sample_nper(means, covs, n):
    return means + np.einsum('ijk,ilk->ilj', cholesky(covs), normal(size = (*means.shape, n)))

def chol_sample_1per_constraints(means, covs, cf, bounds_mat, bounds_keys, bounds):
    """ Sample with constraints.  If fail constraints, resample. """
    chols = cholesky(covs)
    cand = means + np.einsum('ijk,ik->ij', chols, normal(size = means.shape))
    good = cf(tran(cand, bounds_mat, bounds_keys), bounds)
    while np.any(~good):
        cand[np.where(~good)] = (
            + means[~good]
            + np.einsum('ijk,ik->ij', chols[~good], normal(size = ((~good).sum(), means.shape[1])))
            )
        good[~good] = cf(tran(cand[~good], bounds_mat, bounds_keys), bounds)
    return cand

def chol_sample_nper_constraints(means, covs, n, cf, bounds_mat, bounds_keys, bounds):
    """ Sample with constraints.  If fail constraints, resample. """
    chols = cholesky(covs)
    cand = means + np.einsum('ijk,nik->nij', chols, normal(size = (n, *means.shape)))
    for i in range(cand.shape[1]):
        goodi = cf(tran(cand[:,i], bounds_mat, bounds_keys),bounds)
        while np.any(~goodi):
            cand[np.where(~goodi)[0],i] = (
                + means[i]
                + np.einsum('ik,nk->ni', chols[i], normal(size = ((~goodi).sum(), means.shape[1])))
                )
            goodi[np.where(~goodi)[0]] = cf(tran(cand[~goodi,i], bounds_mat, bounds_keys), bounds)
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
    return np.einsum('kitj,kitl->tijl', arr - mean, arr - mean) / (N - 1)

def mvnorm_logpdf(x, mean, Prec, ldet):
    # k = x.shape[-1]
    # part1 = -k * 0.5 * np.log(2 * np.pi) - 0.5 * ldet
    # x = x - mu
    # return part1 + np.squeeze(-x[..., None, :] @ Prec @ x[..., None] / 2)
    ld = (
        - 0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet
        - 0.5 * np.einsum('tm,mn,tn->t', x - mean, Prec, x - mean)
        )
    return ld

def mvnorm_logpdf_(x, mean, prec, ldet):     # x = (ntheta[i] * nTemps[i] * k)
    """
    x = (n_theta[i], ntemps, k)
    mu = (ntemps[i])
    prec = (ntemps x k x k)
    ldet = (ntemps)
    """
    ld = (
        - 0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet
        - 0.5 * np.einsum('stm,tmn,stn->st', x - mean, prec, x - mean)
        )
    return ld

def invwishart_logpdf(w, df, scale):
    """ unnormalized logpdf of inverse wishart w given df and scale """
    ld = (
        - 0.5 * (df + w.shape[-1] + 1) * slogdet(w)[1]
        - 0.5 * np.einsum('...ii->...', np.einsum('ji,...ij->...ij', scale, np.linalg.inv(w)))
        )
    return ld

def invgamma_logpdf(s, alpha, beta):
    """ log pdf of inverse gamma distribution -- Assume s = (n x p) """
    ld = (
        + alpha * np.log(beta)
        - gammaln(alpha)
        - (alpha - 1) * np.log(s)
        - beta / s
        ).sum(axis = 1)
    return ld

from collections import namedtuple
OutCalibPool = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 tau pred_curr')
OutCalibHier = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 count_temper tau pred_curr theta0 Sigma0')

def calibHier(setup):
    t0 = time.time()
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    ntheta = np.sum(setup.ntheta)
    s2     = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    sse    = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)] # for calc of s2
    theta  = [np.empty([setup.nmcmc, setup.ntheta[i], setup.ntemps, setup.p]) for i in range(setup.nexp)]
    s2_vec_curr = [s2[i][0,:, setup.s2_ind[i]] for i in range(setup.nexp)]

    theta0_start = initfunc(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(tran(theta0_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    while np.any(~good):
        theta0_start[np.where(~good)] = initfunc(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran(theta0_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta0[0] = theta0_start
    Sigma0[0] = np.eye(setup.p) * 0.25**2

    pred_curr = [None] * setup.nexp
    pred_cand = [None] * setup.nexp
    pred_cand_mat = [None] * setup.nexp
    sse_curr = [None] * setup.nexp # [i] * ntheta[i] * ntemps
    sse_cand = [None] * setup.nexp
    itl_mat  = [  # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        np.ones((setup.ntheta[i], setup.ntemps)) * setup.itl
        for i in range(setup.nexp)
        ]

    for i in range(setup.nexp):
        theta[i][0] = chol_sample_nper_constraints(theta0[0], Sigma0[0], setup.ntheta[i],
                        setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds)
        pred_curr[i] = setup.models[i].eval(
                tran(theta[i][0].reshape(setup.ntemps * setup.ntheta[i], setup.p),
                setup.bounds_mat, setup.bounds.keys())
                ).reshape((setup.ntheta[i], setup.ntemps, setup.y_lens[i]))
        pred_cand[i] = pred_curr[i].copy()
        pred_cand_mat[i] = pred_cand[i].reshape(setup.ntheta[i] * setup.ntemps, setup.y_lens[i])
        sse_curr[i] = ((pred_curr[i] - setup.ys[i]) * pred_curr[i] - setup.ys[i] / s2_vec_curr[i].T).sum(axis = 2)
        sse_cand[i] = sse_curr[i].copy()

    dev_sq = [np.empty(pred_curr[i].shape) for i in range(setup.nexp)]

    eps = 1.0e-13
    AM_SCALAR = 2.4**2/setup.p

    tau = [ -4 * np.ones((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    S   = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    cov = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    mu  = [np.empty((setup.ntheta[i], setup.ntemps, setup.p)) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        S[i][:] = np.eye(setup.p) * 1e-6

    theta0_prior_cov = np.eye(setup.p)*1.**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = np.repeat(.5, setup.p)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]

    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))

    Sigma0_prior_df = setup.p
    Sigma0_prior_scale = np.eye(setup.p)*.1**2
    Sigma0_dfs = Sigma0_prior_df + ntheta * setup.itl

    Sigma0_ldet_curr = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr  = np.linalg.inv(Sigma0[0])

    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = [np.zeros((setup.ntheta[i], setup.ntemps)) for i in range(setup.nexp)]
    count_decor = [np.zeros((setup.ntheta[i], setup.ntemps, setup.p)) for i in range(setup.nexp)]
    count_100 = [np.zeros((setup.ntheta[i], setup.ntemps)) for i in range(setup.nexp)]

    theta_cand = [np.empty([setup.ntheta[i], setup.ntemps, setup.p]) for i in range(setup.nexp)]
    theta_cand_mat = [np.empty([setup.ntemps * setup.ntheta[i], setup.p]) for i in range(setup.nexp)]
    pred_cand = pred_curr.copy()
    sse_cand  = sse_curr.copy()

    alpha  = [np.ones((setup.ntheta[i], setup.ntemps)) * -np.inf for i in range(setup.nexp)]
    accept = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    sw_alpha = np.zeros(setup.nswap_per)
    tau_up = [np.zeros(t.shape, dtype = bool) for t in tau]
    good_values = [np.zeros(alpha[i].shape, dtype = bool) for i in range(setup.nexp)]
    good_values_mat = [good_values[i].reshape(setup.ntheta[i] * setup.ntemps) for i in range(setup.nexp)]

    ## start MCMC
    for m in range(1,setup.nmcmc):
        for i in range(setup.nexp):
            theta[i][m] = theta[i][m-1].copy() # current set to previous, will change if accepted
        ## adaptive Metropolis for each temperature
        if m == 300:
            for i in range(setup.nexp):
                mu[i][:]  = theta[i][:(m-1)].mean(axis = 0)
                cov[i][:] = cov_4d_pcm(theta[i][:(m-1)], mu[i])
                S[i][:]   = AM_SCALAR * np.einsum('tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))

        if m > 300:
            for i in range(setup.nexp):
                mu[i] += (theta[i][m-1] - mu[i]) / (m - 1)
                cov[i][:] = (
                    + (m-1) / (m-2) * cov[i]
                    + (m-2) / (m-1) / (m-1) * np.einsum('etj,etl->tejl', theta[i][m-1] - mu[i], theta[i][m-1] - mu[i])
                    )
                S[i] = AM_SCALAR * np.einsum('tejl,te->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))

        # MCMC update for thetas
        for i in range(setup.nexp):
            # Find new candidate values for theta
            theta_cand[i][:] = chol_sample_1per(theta[i][m-1], S[i])
            theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntemps * setup.ntheta[i], setup.p)
            # Check constraints
            good_values_mat[i][:] = setup.checkConstraints(
                tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                )
            good_values[i][:] = good_values_mat[i].reshape(setup.ntheta[i], setup.ntemps)
            # Generate Predictions at new Theta values
            if np.any(good_values_mat[i]):
                pred_cand_mat[i][good_values_mat[i]] = setup.models[i].eval(
                    tran(theta_cand_mat[i][good_values_mat[i]], setup.bounds_mat, setup.bounds.keys())
                    )
            pred_cand[i][:] = pred_cand_mat[i].reshape(setup.ntheta[i], setup.ntemps, setup.y_lens[i])
            sse_cand[i][:] = ((pred_cand[i] - setup.ys[i]) * (pred_cand[i] - setup.ys[i]) / s2_vec_curr[i].T).sum(axis = 2)
            # Calculate log-probability of MCMC accept
            alpha[i][:] = - np.inf
            alpha[i][good_values[i]] = (
                -0.5 * itl_mat[i][good_values[i]] * (
                    + sse_cand[i][good_values[i]]
                    - sse_curr[i][good_values[i]]
                    + mvnorm_logpdf_(theta_cand[i], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                    - mvnorm_logpdf_(theta[i][m-1], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                    )
                )
            # MCMC Accept
            accept[i][:] = np.log(uniform(size = alpha[i].shape)) < alpha[i]
            # Where accept, make changes
            theta[i][m][accept[i]] = theta_cand[i][accept[i]]
            pred_curr[i][accept[i]] = pred_cand[i][accept[i]]
            sse_curr[i][accept[i]] = sse_cand[i][accept[i]]
            count[i][accept[i]] += 1
            count_100[i][accept[i]] += 1

        # Adaptive Metropolis Update
        if m % 100 == 0 and m > 300:
            delta = min(0.1, 1/np.sqrt(m+1)*5)
            for i in range(setup.nexp):
                tau_up[i][:] = (count_100[i] < 23).T
                tau[i][tau_up[i]] += delta
                tau[i][~tau_up[i]] -= delta
                count_100[i] *= 0

        ## Decorrelation Step
        if m % setup.decor == 0:
            for i in range(setup.nexp):
                for k in range(setup.p):
                    # Find new candidate values for theta
                    theta_cand[i][:] = theta[m][i].copy()
                    theta_cand[i][:,:,k] = initfunc((setup.ntheta[i], setup.ntemps))
                    theta_cand_mat[i][:] = theta_cand[i].reshape(setup.ntheta[i]*setup.ntemps, setup.p)
                    # Compute constraint flags
                    good_values_mat[i][:] = setup.checkConstraints(
                        tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                        )
                    good_values[i][:] = good_values_mat[i].reshape(setup.ntheta[i], setup.ntemps, setup.p)
                    # If valid, compute predictions and likelihood values
                    if np.any(good_values_mat[i]):
                        pred_cand_mat[i][good_values_mat[i]][:] = setup.models[i].eval(
                            tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys())
                            )
                    pred_cand[i][:] = pred_cand_mat[i].reshape(setup.ntheta[i], setup.ntemps, setup.y_lens[i])
                    sse_cand[i][:] = ((pred_cand[i] - setup.ys[i]) * (pred_cand[i] - setup.ys[i]) / s2_vec_curr.T).sum(axis = 2)
                    # Calculate log-probability of MCMC Accept
                    alpha[i][:] = - np.inf
                    alpha[i][good_values[i]] = (
                        - 0.5 * itl_mat[good_values[i]] * (sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]])
                        - 0.5 * itl_mat[good_values[i]] * (
                            + mvnorm_logpdf_(theta_cand[i], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            - mvnorm_logpdf_(theta_cand[i], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)[good_values[i]]
                            )
                        )
                    # MCMC Accept
                    accept[i][:] = np.log(uniform(size = alpha[i].shape)) < alpha[i]
                    # Where accept, make changes
                    theta[i][m][accept[i]] = theta_cand[i][accept[i]]
                    pred_curr[i][accept[i]] = pred_cand[i][accept[i]]
                    sse_curr[i][accept[i]] = sse_cand[i][accept[i]]
                    count_decor[i][accept[i], k] += 1

        ## Gibbs update s2 (TODO)
        for i in range(setup.nexp):
            dev_sq[i] = (pred_curr[i] - setup.ys[i]) * (pred_curr[i] - setup.ys[i]) # squared deviations
            # for j in range(setup.ns2[i]):
            #     sse[i][j] = dev_sq[:,:,setup.s2_ind[i]==j].sum(axis = 2)
            # s2[i][m] = 1 / np.random.gamma(
            #     setup.itl
            #     )
            # (array(ntemps x ny x ntheta) - vector(ny))^2 should be array(ntemps x ny x ntheta)
            for j in range(setup.ns2[i]):
                sseij = np.sum(dev_sq[i][:,:,np.where(setup.s2_ind[i]==j)[0]], 2) # sse for jth s2ind in experiment i
                s2[i][m, :, j] = 1 / np.random.gamma(setup.itl*(setup.ny_s2[i][j]/2 + np.array(setup.ig_a[i][j]) + 1) - 1,
                    1 / (setup.itl*(np.array(setup.ig_b[i][j]) + .5*sseij)))

            s2_vec_curr[i][:] = s2[i][m,:,setup.s2_ind[i]]
            sse_curr[i] = ((pred_curr[i] - setup.ys[i]) * (pred_curr[i] - setup.ys[i]) / s2_vec_curr[i].T).sum(axis = 2)

        ## Gibbs update theta0
        cc = np.linalg.inv(np.einsum('t,tpq->tpq', ntheta * setup.itl, Sigma0_inv_curr) + theta0_prior_prec)
        tbar *= 0.
        for i in range(setup.nexp):
            tbar += theta[i][m-1].sum(axis = 0)
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
            mat += np.einsum('ntp,ntq->tpq', theta[i][m] - theta0[m], theta[i][m] - theta0[m])

        Sigma0_scales = Sigma0_prior_scale + np.einsum('t,tml->tml',setup.itl,mat)
        for t in range(setup.ntemps):
            Sigma0[m][t] = invwishart.rvs(df = Sigma0_dfs[t], scale = Sigma0_scales[t])
        Sigma0_ldet_curr[:] = np.linalg.slogdet(Sigma0[m])[1]
        Sigma0_inv_curr[:] = np.linalg.inv(Sigma0[m])

        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                sw_alpha[:] = 0.
                sw_alpha += (
                    - mvnorm_logpdf(theta0[m][sw.T[1]], theta0_prior_mean, theta0_prior_prec, theta0_prior_ldet)
                    + mvnorm_logpdf(theta0[m][sw.T[0]], theta0_prior_mean, theta0_prior_prec, theta0_prior_ldet)
                    - invwishart_logpdf(Sigma0[m][sw.T[1]], Sigma0_prior_df, Sigma0_prior_scale)
                    + invwishart_logpdf(Sigma0[m][sw.T[0]], Sigma0_prior_df, Sigma0_prior_scale)
                    )
                for i in range(setup.nexp):
                    sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                        # for t_1
                        - invgamma_logpdf(s2[i][m][sw.T[1]], setup.ig_a[i], setup.ig_b[i])
                        - mvnorm_logpdf_(theta[i][m][:,sw.T[1]], theta0[m,sw.T[1]], Sigma0_inv_curr[sw.T[1]], Sigma0_ldet_curr[sw.T[1]]).sum(axis = 0)
                        + 0.5 * np.log(s2_vec_curr[i][:,sw.T[1]]).sum(axis = 0)
                        + 0.5 * sse_curr[i][:,sw.T[1]].sum(axis = 0)
                        # for t_0
                        + invgamma_logpdf(s2[i][m][sw.T[0]], setup.ig_a[i], setup.ig_b[i])
                        + mvnorm_logpdf_(theta[i][m][:,sw.T[0]], theta0[m, sw.T[0]], Sigma0_inv_curr[sw.T[0]], Sigma0_ldet_curr[sw.T[0]]).sum(axis = 0)
                        - 0.5 * np.log(s2_vec_curr[i][:,sw.T[0]]).sum(axis = 0)
                        - 0.5 * sse_curr[i][:,sw.T[0]].sum(axis = 0)
                        )
                for tt in sw[np.where(sw[np.log(uniform(size = setup.nswap_per)) < sw_alpha])[0]]:
                    count_temper[tt[0], tt[1]] += 1
                    for i in range(setup.nexp):
                        theta[i][m,:,tt[0]], theta[i][m,:,tt[1]]     = theta[i][m,:,tt[1]].copy(), theta[i][m,:,tt[0]].copy()
                        s2[i][m,tt[0]], s2[i][m,tt[1]]               = s2[i][m,tt[1]].copy(), s2[i][m,tt[0]].copy()
                        pred_curr[i][:,tt[0]], pred_curr[i][:,tt[1]] = pred_curr[i][:,tt[1]].copy(), pred_curr[i][:,tt[0]].copy()
                        sse_curr[i][:,tt[0]], sse_curr[i][:,tt[1]]   = sse_curr[i][:,tt[1]].copy(), sse_curr[i][:,tt[1]].copy()
                        s2_vec_curr[i][:]                            = s2[i][m, :, setup.s2_ind[i]]
                    theta0[m,tt[0]], theta0[m,tt[1]] = theta0[m,tt[1]].copy(), theta0[m,tt[0]].copy()
                    Sigma0[m,tt[0]], Sigma0[m,tt[1]] = Sigma0[m,tt[1]].copy(), Sigma0[m,tt[0]].copy()
                    Sigma0_inv_curr[tt[0]], Sigma0_inv_curr[tt[1]]   = Sigma0_inv_curr[tt[1]].copy(), Sigma0_inv_curr[tt[0]].copy()
                    Sigma0_ldet_curr[tt[0]], Sigma0_ldet_curr[tt[1]] = Sigma0_ldet_curr[tt[1]].copy(), Sigma0_ldet_curr[tt[0]].copy()

        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count_temper = count_temper + count_temper.T - np.diag(np.diag(count_temper))
    out = OutCalibHier(theta, s2, count, count_decor, count_100, count_temper, tau, pred_curr, theta0, Sigma0)
    return(out)

def calibPool(setup):
    t0 = time.time()
    theta = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    n_s2 = np.sum(setup.ns2)
    s2 = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    s2_vec_curr = [s2[i][0,:,setup.s2_ind[i]] for i in range(setup.nexp)]

    theta_start = initfunc(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(tran(theta_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    while np.any(~good):
        theta_start[np.where(~good)] = np.random.normal(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran(theta_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta[0] = theta_start

    pred_curr = []
    sse_curr = np.empty([setup.ntemps, setup.nexp])
    for i in range(setup.nexp):
        pred_curr.append(setup.models[i].eval(tran(theta[0, :, :], setup.bounds_mat, setup.bounds.keys())))
        sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)

    eps  = 1.0e-13
    tau  = np.repeat(-4.0, setup.ntemps)
    cc   = 2.4**2/setup.p
    S    = np.empty([setup.ntemps, setup.p, setup.p])
    S[:] = np.eye(setup.p)*1e-6
    cov  = np.empty([setup.ntemps, setup.p, setup.p])
    mu   = np.empty([setup.ntemps, setup.p])

    count = np.zeros([setup.ntemps, setup.ntemps], dtype = int)
    count_decor = np.zeros([setup.p, setup.ntemps], dtype = int)
    count_100 = np.zeros(setup.ntemps, dtype = int)

    pred_cand = np.copy(pred_curr)
    sse_cand  = np.copy(sse_curr)

    alpha    = np.ones(setup.ntemps) * (-np.inf)
    sw_alpha = np.zeros(setup.nswap_per)

    ## start MCMC
    for m in range(1,setup.nmcmc):
        theta[m] = np.copy(theta[m-1]) # current set to previous, will change if accepted

        ## adaptive Metropolis for each temperature
        if m == 300:
            mu  = theta[:(m-1)].mean(axis = 0)
            cov = cov_3d_pcm(theta[:(m-1)], mu)
            S   = cc * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))

        if m > 300:
            mu += (theta[m-1] - mu) / (m - 1)
            cov = (
                + (m-1)/(m-2) * cov
                + (m - 2)/(m - 1)/(m - 1) * np.einsum('ti,tj->tij', theta[m-1] - mu, theta[m-1] - mu)
                )
            S   = cc * np.einsum('ijk,i->ijk', cov + np.eye(setup.p) * eps, np.exp(tau))

        # generate proposal
        theta_cand  = (
            + theta[m-1]
            + np.einsum('ijk,ik->ij', cholesky(S), normal(size = (setup.ntemps, setup.p)))
            )
        good_values = setup.checkConstraints(
            tran(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
            )
        # get predictions and SSE
        pred_cand = np.copy(pred_curr)
        sse_cand = np.copy(sse_curr)
        if np.any(good_values):
            for i in range(setup.nexp):
                pred_cand[i][good_values] = setup.models[i].eval(
                    tran(theta_cand[good_values], setup.bounds_mat, setup.bounds.keys())
                    )
                sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) * (pred_cand[i] - setup.ys[i]) / s2_vec_curr[i].T, 1)

        tsq_diff = 0 # ((theta_cand * theta_cand).sum(axis = 1) - (theta[m-1] * theta[m-1]).sum(axis = 1))[good_values]
        sse_diff = (sse_cand - sse_curr).sum(axis = 1)[good_values]
        # for each temperature, accept or reject
        alpha[:] = - np.inf
        # alpha[good_values] = (- 0.5 * setup.itl[good_values] * (sse_cand[good_values] - sse_curr[good_values])).sum(axis = 1)
        alpha[good_values] = -0.5 * setup.itl[good_values] * (sse_diff + tsq_diff)
        for t in np.where(np.log(uniform(size=setup.ntemps)) < alpha)[0]: # first index because output of np.where is a tuple of arrays...
            theta[m,t] = theta_cand[t]
            count[t,t] += 1
            sse_curr[t] = sse_cand[t]
            for i in range(setup.nexp):
                pred_curr[i][t] = pred_cand[i][t]
            count_100[t] += 1

        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > 300):
            delta = min(0.1, 5 / sqrt(m + 1))
            tau[np.where(count_100 < 23)] -= delta
            tau[np.where(count_100 > 23)] += delta
            count_100 *= 0

        # decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = np.copy(theta[m,:,:])
                theta_cand[:,k] = initfunc(size = setup.ntemps) # independence proposal, will vectorize of columns
                good_values = setup.checkConstraints(
                    tran(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds,
                    )
                pred_cand = np.copy(pred_curr)
                sse_cand = np.copy(sse_curr)

                if np.any(good_values):
                    for i in range(setup.nexp):
                        pred_cand[i][good_values, :] = setup.models[i].eval(
                            tran(theta_cand[good_values, :], setup.bounds_mat, setup.bounds.keys()),
                            )
                        sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) * (pred_cand[i] - setup.ys[i]) / s2_vec_curr[i].T, 1)

                alpha[:] = -np.inf
                tsq_diff = 0. # ((theta_cand * theta_cand).sum(axis = 1) - (theta[m] * theta[m]).sum(axis = 1))[good_values]
                sse_diff = (sse_cand - sse_curr).sum(axis = 1)[good_values]
                alpha[good_values] = -0.5 * setup.itl[good_values] * (sse_diff + tsq_diff)
                for t in np.where(np.log(uniform(size = setup.ntemps)) < alpha)[0]:
                    theta[m,t,k] = theta_cand[t,k]
                    count_decor[k,t] += 1
                    for i in range(setup.nexp):
                        pred_curr[i][t] = pred_cand[i][t]
                    sse_curr[t] = sse_cand[t]

        ## Gibbs update s2
        for i in range(setup.nexp):
            dev_sq = (pred_cand[i] - setup.ys[i])**2 # squared deviations
            for j in range(setup.ns2[i]):
                sseij = np.sum(dev_sq[:,setup.s2_ind[i]==j], 1) # sse for jth s2ind in experiment i
                s2[i][m, :, j] = 1 / np.random.gamma(setup.itl*(setup.ny_s2[i][j]/2 + np.array(setup.ig_a[i][j]) + 1) - 1,
                    1 / (setup.itl * (np.array(setup.ig_b[i][j]) + .5 * sseij)))

        s2_vec_curr = [s2[i][m, :, setup.s2_ind[i]] for i in range(setup.nexp)]

        for i in range(setup.nexp):
            sse_curr[:, i] = ((pred_curr[i] - setup.ys[i]) * (pred_curr[i] - setup.ys[i]) / s2_vec_curr[i].T).sum(axis = 1)

        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(setup.ntemps, 2 * setup.nswap_per, replace = False).reshape(-1,2)
                # alpha = np.zeros(setup.nswap_per) # log probability of swap
                sw_alpha[:] = 0.
                for i in range(setup.nexp):
                    sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]])*(
                        - 0.5 * np.log(s2_vec_curr[i][:,sw.T[0]]).sum(axis = 0)
                        - 0.5 * sse_curr[sw.T[0], i]
                        - ((setup.ig_a[i] + 1) * np.log(s2[i][m][sw.T[0]])).sum(axis = 1)
                        - (setup.ig_b[i] / s2[i][m][sw.T[0]]).sum(axis = 1)
                        + 0.5 * np.log(s2_vec_curr[i][:,sw.T[1]]).sum(axis = 0)
                        + 0.5 * sse_curr[sw.T[1], i]
                        + ((setup.ig_a[i] + 1) * np.log(s2[i][m][sw.T[1]])).sum(axis = 1)
                        + (setup.ig_b[i] / s2[i][m][sw.T[1]]).sum(axis = 1)
                        )
                # sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (    # probit transform jacobian
                #     - 0.5 * (theta[m][sw.T[0]]**2).sum(axis = 1)
                #     + 0.5 * (theta[m][sw.T[1]]**2).sum(axis = 1)
                #    )
                for tt in sw[np.where(np.log(uniform(size = setup.nswap_per)) < sw_alpha)[0]]:
                    count[tt[0],tt[1]] += 1
                    theta[m][tt[0]], theta[m][tt[1]] = theta[m][tt[1]].copy(), theta[m][tt[0]].copy()
                    sse_curr[tt[0]], sse_curr[tt[1]] = sse_curr[tt[1]].copy(), sse_curr[tt[0]].copy()
                    for i in range(setup.nexp):
                        s2[i][m][tt[0]], s2[i][m][tt[1]] = s2[i][m][tt[1]].copy(), s2[i][m][tt[0]].copy()
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = pred_curr[i][tt[1]].copy(), pred_curr[i][tt[0]].copy()
                    s2_vec_curr = [s2[i][m,:,setup.s2_ind[i]] for i in range(setup.nexp)]

        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count = count + count.T - np.diag(np.diag(count))
    out = OutCalibPool(theta, s2, count, count_decor, count_100, tau, pred_curr)
    return(out)

# EOF
