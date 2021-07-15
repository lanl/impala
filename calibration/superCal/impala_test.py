import numpy as np
#import pyBASS as pb
#import physical_models_vec as pm_vec
#from scipy.interpolate import interp1d
import time
import scipy
from scipy import stats
from numpy.random import uniform, normal
from math import sqrt, floor
from scipy.special import erf, erfinv
from numpy.linalg import cholesky

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
    return means + np.einsum('...jk,...k->...j', cholesky(covs), normal(size = (means.shape)))

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
        goodi = cv(tran(cand[:,i], bounds_mat, bounds_keys),bounds)
        while np.any(~goodi):
            cand[np.where(~goodi),i] = (
                + means[i]
                + np.einsum('ik,nk->ni', chols[i], normal(size = ((~goodi).sum(), means.shape[1])))
                )
            goodi[~goodi] = cf(tran(cand[~goodi], bounds_mat, bounds_keys), bounds)
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

def mvnorm_logpdf(x, mu, Prec, ldet):
    k = x.shape[-1]
    part1 = -k * 0.5 * np.log(2 * np.pi) - 0.5 * ldet
    x = x - mu
    return part1 + np.squeeze(-x[..., None, :] @ Prec @ x[..., None] / 2)

from collections import namedtuple
OutCalibPool = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 tau pred_curr')
OutCalibHier = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 count_temper tau pred_curr theta0 Sigma0')

def calibHier(setup):
    t0 = time.time()
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    ntheta = np.sum(setup.ntheta)
    s2     = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    theta  = [np.empty([setup.nmcmc, setup.ntheta[i], setup.ntemps, setup.p]) for i in range(setup.nexp)]
    # theta  = [np.empty([setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p]) for i in range(setup.nexp)]]
    s2_vec_curr = [s2[i][0,:, setup.s2_ind[i]] for i in range(setup.nexp)]

    theta0_start = np.random.uniform(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(tran(theta0_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    while np.any(~good):
        theta0_start[np.where(~good)] = uniform(size = [(~good).sum(), setup.p])
        good[np.where(~good)] = setup.checkConstraints(
            tran(theta0_start[np.where(~good)], setup.bounds_mat, setup.bounds.keys()),
            setup.bounds,
            )
    theta0[0] = theta0_start
    Sigma0[0] = np.eye(setup.p) * 0.25**2

    for i in range(setup.nexp):
        theta[i][0] = chol_sample_nper_constraints(theta0[0], Sigma0[0], setup.ntheta[i],
                        setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds)
    pred_curr = [None] * setup.nexp
    sse_curr = [None] * setup.nexp

    for i in range(setup.nexp):
        _parlist = tran(theta[i][0].reshape(setup.ntemps * setup.ntheta[i], setup.p), setup.bounds_mat, setup.bounds.keys())
        pred_curr[i] = setup.models[i].eval(_parlist).reshape((setup.ntheta[i], setup.ntemps, setup.y_lens[i]))
        sse_curr[i] = ((pred_curr[i] - setup.ys[i]) * pred_curr[i] - setup.ys[i] / s2_vec_curr[i].T).sum(axis = 2)

    eps = 1.0e-13
    cc = 2.4**2/setup.p

    tau = [ -4 * np.ones(setup.ntemps, setup.ntheta[i]) for i in range(setup.nexp)]
    S   = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    cov = [np.empty((setup.ntemps, setup.ntheta[i], setup.p, setup.p)) for i in range(setup.nexp)]
    mu  = [np.empty((setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    S[:] = np.eye(setup.p) * 1e-6

    theta0_prior_cov = np.eye(setup.p)*1.**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = np.repeat(.5, setup.p)

    Sigma0_prior_df = setup.p
    Sigma0_prior_scale = np.eye(setup.p)*.1**2

    Sigma0_ldet_curr = np.empty(setup.ntemps)
    Sigma0_inv_curr  = np.empty([setup.ntemps, setup.p, setup.p])
    Sigma0_ldet_curr = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr  = 1 / Sigma0[0] # Sigma0[0] is np.eye(p)...so x^-1 = 1/x

    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = [np.zeros((setup.ntheta[i], setup.ntemps)) for i in range(setup.nexp)]
    count_decor = [np.zeros((setup.ntheta[i], setup.p, setup.ntemps)) for i in range(setup.nexp)]
    count_100 = [np.zeros((setup.ntheta[i], setup.ntemps)) for i in range(setup.nexp)]

    theta_cand = [np.empty([setup.ntheta[i], setup.ntemps, setup.p]) for i in setup.nexp]
    pred_cand = pred_curr.copy()
    sse_cand  = sse_curr.copy()
    ## start MCMC



    for m in range(1,setup.nmcmc):
        for i in range(setup.nexp):
            theta[i][m] = theta[i][m-1].copy() # current set to previous, will change if accepted
        ## adaptive Metropolis for each temperature
        if m == 300:
            for i in range(setup.nexp):
                mu[i][:]  = theta[i][:(m-1)].mean(axis = 0)
                cov[i][:] = cov_4d_pcm(theta[i][:(m-1)], mu[i])
                S[i][:]   = cc * np.einsum('tejl,t->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))
            # for i in range(setup.nexp):
            #     mu[i][:]  = theta[i][:(m-1)].mean(axis = 0)
            #     cov[i][:] =
            #     for it in range(setup.ntheta[i]):
            #         for t in range(setup.ntemps):
            #             mu[i][t,it,:] = np.mean(theta[i][:(m-1),it,t,:], 0)
            #             cov[i][t,it,:,:] = np.cov(theta[i][:(m-1),it,t,:], rowvar=False)
            #             S[i][t,it,:,:] = (cov[i][t,it,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[i][t,it])
        if m > 300:
            for i in range(setup.nexp):
                mu[i] += (theta[i][m-1] - mu[i]) / (m - 1)
                cov[i] = (
                    + (m-1) / (m-2) * cov
                    + (m-2) / (m-1) / (m-1) * np.einsum('tij,til->tijl', theta[i][m-1] - mu[i], theta[i][m-1] - mu[i])
                    )
                S[i] = cc * np.einsum('tejl,t->tejl', cov[i] + np.eye(setup.p) * eps, np.exp(tau[i]))
                # for it in range(setup.ntheta[i]):
                #     for t in range(setup.ntemps):
                #         mu[i][t,it,:] = mu[i][t,it,:] + (theta[i][m-1,it,t,:] - mu[i][t,it,:]) / (m-1)
                #         cov[i][t,it,:,:] = (m-1)/(m-2) * cov[i][t,it,:,:] + (m-2)/(m-1)**2 * \
                #             np.outer(theta[i][m-1,it,t,:] - mu[i][t,it,:], theta[i][m-1,it,t,:] - mu[i][t,it,:])
                #         S[i][t,it,:,:] = (cov[i][t,it,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[i][t,it])

        # generate proposal
        theta_cand = [None] * setup.nexp
        for i in range(setup.nexp):
            theta_cand[i][:] = chol_sample_1per(theta[i], S[i])
        theta_cand_mat = []
        good_values = []
        good_values_mat = []
        for i in range(setup.nexp):
            theta_cand.append(np.empty([setup.ntheta[i], setup.ntemps, setup.p]))
            #for i in range(setup.nexp):
            for it in range(setup.ntheta[i]):
                for t in range(setup.ntemps):
                    theta_cand[i][it,t,:] = chol_sample(theta[i][m-1,it,t,:], S[i][t,it,:,:])
            theta_cand_mat.append(np.reshape(theta_cand[i], [setup.ntemps*setup.ntheta[i], setup.p]))
            good_values.append(np.reshape(
                setup.checkConstraints(tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
                , [setup.ntemps, setup.ntheta[i]])) # not sure if this reshape is right
            good_values_mat.append(np.reshape(good_values[i], [setup.ntemps*setup.ntheta[i]]))


        # get predictions and SSE
        #pred_cand = []
        #sse_cand = np.empty([setup.ntemps, setup.nexp])
        pred_cand = []#np.copy(pred_curr)
        sse_cand = []#np.copy(sse_curr)
        for i in range(setup.nexp):
            if np.any(good_values[i]):
                if setup.ntheta[i] == 1:
                    parlist = tran(
                                theta_cand_mat[i][good_values_mat[i],:],
                                setup.bounds_mat,
                                setup.bounds.keys()
                            )
                else:
                    parlist = []
                    for it in range(setup.ntheta[i]):
                        parlist.append(tran(
                                    theta_cand_mat[i][good_values,:],
                                    setup.bounds_mat,
                                    setup.bounds.keys()
                                ))
                preds = np.zeros([setup.ntheta[i]*setup.ntemps, setup.y_lens[i]])
                preds[good_values_mat[i],:] = setup.models[i].eval(parlist)
                pred_cand.append(
                    np.reshape(np.copy(preds), [setup.ntheta[i], setup.ntemps, setup.y_lens[i]])
                )
                sse_cand.append(np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 2))

            else:
                sse_cand.append(0)
                pred_cand.append(0)

        # for each temperature, accept or reject
        for i in range(setup.nexp):
            for it in range(setup.ntheta[i]):
                for t in range(setup.ntemps):
                    if ~good_values[i][t,it]:
                        alpha = -9999
                    else:
                        alpha = np.sum(-0.5*setup.itl[t]*(sse_cand[i][it,t] - sse_curr[i][it,t])) + \
                            setup.itl[t]*(mvnorm_logpdf(theta_cand[i][it,t,:] ,theta0[m-1,t,:], Sigma0_inv_curr[t,:,:], Sigma0_ldet_curr[t])
                                                - mvnorm_logpdf(theta[i][m-1,it,t,:] ,theta0[m-1,t,:], Sigma0_inv_curr[t,:,:], Sigma0_ldet_curr[t])
                                )
                                #setup.itl[t]*(scipy.stats.multivariate_normal.logpdf(theta_cand[i][it,t,:] ,theta0[m-1,t,:], Sigma0[m-1,t,:,:])
                                #            - scipy.stats.multivariate_normal.logpdf(theta[i][m-1,it,t,:] ,theta0[m-1,t,:], Sigma0[m-1,t,:,:])
                                #)

                    #if m % 100 == 0:
                    #    print(alpha)
                    if np.log(np.random.rand()) < alpha:
                        theta[i][m,it,t,:] = theta_cand[i][it,t,:]
                        count[i][it,t] += 1
                        #for i in range(setup.nexp):
                        pred_curr[i][it,t,:] = pred_cand[i][it,t,:]
                        sse_curr[i][it,t] = sse_cand[i][it,t]
                        count_100[i][it,t] += 1

        # diminishing adaptation based on acceptance rate for each temperature
        if m % 100 == 0 and m > 300:
            delta = min(0.1, 1/np.sqrt(m+1)*5)
            for i in range(setup.nexp):
                for it in range(setup.ntheta[i]):
                    for t in range(setup.ntemps):
                        if count_100[i][it,t] < 23:
                            tau[i][t,it] -= delta
                        elif count_100[i][it,t] > 23:
                            tau[i][t,it] += delta
                count_100[i] = count_100[i]*0


        ## decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = []
                theta_cand_mat = []
                good_values = []
                good_values_mat = []
                for i in range(setup.nexp):
                    theta_cand.append(np.copy(theta[i][m,:,:,:]))
                    theta_cand[i][:,:,k] = np.random.rand(setup.ntemps*setup.ntheta[i]).reshape([setup.ntheta[i], setup.ntemps]) # independence proposal, will vectorize of columns
                    #good_values = setup.checkConstraints(tran(theta_cand, setup.bounds_mat, setup.bounds.keys()),
                    #                                 setup.bounds)


                    theta_cand_mat.append(np.reshape(theta_cand[i], [setup.ntemps*setup.ntheta[i], setup.p]))
                    good_values.append(np.reshape(
                        setup.checkConstraints(tran(theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
                        , [setup.ntemps, setup.ntheta[i]])) # not sure if this reshape is right
                    good_values_mat.append(np.reshape(good_values[i], [setup.ntemps*setup.ntheta[i]]))
                #pred_cand = []
                #sse_cand = np.empty([setup.ntemps, setup.nexp])

                pred_cand = []#np.copy(pred_curr)
                sse_cand = []#np.copy(sse_curr)

                for i in range(setup.nexp):
                    if np.any(good_values[i]):
                        ##np.array(range(setup.nexp))[good_values]
                        #pred_cand[i][good_values, :] = np.reshape(
                        #    setup.models[i].eval(tran(theta_cand_mat[i][good_values,:], setup.bounds_mat, setup.bounds.keys())
                        #    ), [setup.ntemps, setup.y_lens[i], setup.ntheta[i]]) # not sure if this reshape is right
                        ##sse_cand[:,i] = np.sum((pred_cand[i] - setup.ys[i])**2, 0)
                        ##sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)
                        #sse_cand[i] = np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1) # is this doing the right thing?
                        if setup.ntheta[i] == 1:
                            parlist = tran(
                                        theta_cand_mat[i][good_values_mat[i],:],
                                        setup.bounds_mat,
                                        setup.bounds.keys()
                                    )
                        else:
                            parlist = []
                            for it in range(setup.ntheta[i]):
                                parlist.append(tran(
                                            theta_cand_mat[i][good_values_mat,:],
                                            setup.bounds_mat,
                                            setup.bounds.keys()
                                        ))

                        preds = np.zeros([setup.ntheta[i]*setup.ntemps, setup.y_lens[i]])
                        preds[good_values_mat[i],:] = setup.models[i].eval(parlist)
                        pred_cand.append(
                            np.reshape(np.copy(preds), [setup.ntheta[i], setup.ntemps, setup.y_lens[i]]) # not sure if these two reshapes are right
                        )
                        #pred_cand.append(
                        #    np.reshape(setup.models[i].eval(parlist), [setup.ntemps, setup.y_lens[i], setup.ntheta[i]]) # not sure if these two reshapes are right
                        #)
                        sse_cand.append(np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 2))

                    else:
                        sse_cand.append(0)
                        pred_cand.append(0)


                for i in range(setup.nexp):
                    for it in range(setup.ntheta[i]):
                        for t in range(setup.ntemps):
                            if ~good_values[i][t,it]:
                                alpha = -9999
                            else:
                                alpha = np.sum(-0.5*setup.itl[t]*(sse_cand[i][it,t] - sse_curr[i][it,t])) + \
                                    setup.itl[t]*(mvnorm_logpdf(theta_cand[i][it,t,:] ,theta0[m-1,t,:], Sigma0_inv_curr[t,:,:], Sigma0_ldet_curr[t])
                                                - mvnorm_logpdf(theta[i][m,it,t,:] ,theta0[m-1,t,:], Sigma0_inv_curr[t,:,:], Sigma0_ldet_curr[t])
                                    )
                                    #setup.itl[t]*(scipy.stats.multivariate_normal.logpdf(theta_cand[i][it,t,:] ,theta0[m-1,t,:], Sigma0[m-1,t,:,:])
                                    #            - scipy.stats.multivariate_normal.logpdf(theta[i][m,it,t,:] ,theta0[m-1,t,:], Sigma0[m-1,t,:,:])
                                    #)

                            if np.log(np.random.rand()) < alpha:
                                theta[i][m,it,t,k] = theta_cand[i][it,t,k]
                                count_decor[i][it,k,t] += 1
                                pred_curr[i][it,t,:] = pred_cand[i][it,t,:]
                                sse_curr[i][it,t] = sse_cand[i][it,t]




        ## Gibbs update s2
        for i in range(setup.nexp):
            dev_sq = (pred_curr[i] - setup.ys[i])**2 # squared deviations
            # (array(ntemps x ny x ntheta) - vector(ny))^2 should be array(ntemps x ny x ntheta)
            for j in range(setup.ns2[i]):
                sseij = np.sum(dev_sq[:,:,setup.s2_ind[i]==j], 2) # sse for jth s2ind in experiment i
                s2[i][m, :, j] = 1 / np.random.gamma(setup.itl*(setup.ny_s2[i][j]/2 + np.array(setup.ig_a[i][j]) - 1) + 1,
                    1 / (setup.itl*(np.array(setup.ig_b[i][j]) + .5*sseij)))

        s2_vec_curr = []
        for i in range(setup.nexp):
            s2_vec_curr.append(s2[i][m, :, setup.s2_ind[i]])  # this should be ntemps x ylens[i]

        for i in range(setup.nexp):
            sse_curr[i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 2)



        ## Gibbs update theta0
        for t in range(setup.ntemps):
            #print(Sigma0[m,t,:,:])
            #S0inv = np.linalg.inv(Sigma0[m-1,t,:,:])
            cc = np.linalg.inv(ntheta*Sigma0_inv_curr[t,:,:]*setup.itl[t] + theta0_prior_prec)
            tbar = np.zeros(setup.p)
            for i in range(setup.nexp):
                tbar += np.sum(theta[i][m-1,:,t,:], axis=0)
            tbar = tbar/ntheta
            dd = np.dot(ntheta*Sigma0_inv_curr[t,:,:], tbar)*setup.itl[t] + np.dot(theta0_prior_prec, theta0_prior_mean)

            gg = False
            ii = 0
            while not gg:
                ii += 1
                tt = chol_sample(np.dot(cc, dd) , cc)
                gg = setup.checkConstraints(tran(tt, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
                #print(ii)

            theta0[m, t, :] = np.copy(tt)



        ## Gibbs update Sigma0
        for t in range(setup.ntemps):
            mat = np.zeros([setup.p, setup.p])
            for i in range(setup.nexp):
                for it in range(setup.ntheta[i]): # can probably get rid of this loop
                    mat += np.dot(theta[i][m,it,t,:] - theta0[m,t,:], (theta[i][m,it,t,:] - theta0[m,t,:]).T)

            Sigma0[m,t,:,:] = scipy.stats.invwishart.rvs(df = Sigma0_prior_df + ntheta*setup.itl[t], scale = Sigma0_prior_scale + mat*setup.itl[t])

        Sigma0_ldet_curr = np.empty(setup.ntemps)
        Sigma0_inv_curr = np.empty([setup.ntemps, setup.p, setup.p])
        for t in range(setup.ntemps):
            Sigma0_ldet_curr[t] = np.linalg.slogdet(Sigma0[m,t,:,:])[1]
            Sigma0_inv_curr[t,:,:] = np.linalg.inv(Sigma0[m,t,:,:])

        #//////////////////////////////////////////////////////////////////////////////////////
        ## NOTE: should probably get rid of all appends, just make empty lists that get changed
        #//////////////////////////////////////////////////////////////////////////////////////

        ## tempering swaps
        if m > 1 and setup.ntemps > 1:
            for _ in range(setup.ntemps): # do many swaps
                sw = np.sort(np.random.choice(range(setup.ntemps), size=2, replace=False)) # indices to swap

                alpha = (setup.itl[sw[1]] - setup.itl[sw[0]])*(
                    scipy.stats.multivariate_normal.logpdf(theta0[m,sw[0],:], theta0_prior_mean, theta0_prior_cov)#0
                    - scipy.stats.multivariate_normal.logpdf(theta0[m,sw[1],:], theta0_prior_mean, theta0_prior_cov)
                    + scipy.stats.invwishart.logpdf(Sigma0[m,sw[0],:,:], Sigma0_prior_df, Sigma0_prior_scale)
                    - scipy.stats.invwishart.logpdf(Sigma0[m,sw[1],:,:], Sigma0_prior_df, Sigma0_prior_scale)
                )
                for i in range(setup.nexp):
                    alpha += (setup.itl[sw[1]] - setup.itl[sw[0]])*(
                            - np.sum((setup.ig_a[i] + 1)*np.log(s2[i][m, sw[0], :])) - np.sum(setup.ig_b[i]/np.log(s2[i][m, sw[0], :]))
                            + np.sum((setup.ig_a[i] + 1)*np.log(s2[i][m, sw[1], :])) + np.sum(setup.ig_b[i]/np.log(s2[i][m, sw[1], :]))
                    )
                    for it in range(setup.ntheta[i]):
                        alpha += (setup.itl[sw[1]] - setup.itl[sw[0]])*(
                                - .5*np.sum(np.log(s2_vec_curr[i][:,sw[0]])) - 0.5*sse_curr[i][it, sw[0]]
                                + .5*np.sum(np.log(s2_vec_curr[i][:,sw[1]])) + 0.5*sse_curr[i][it, sw[1]]
                                #+ scipy.stats.multivariate_normal.logpdf(theta[i][m,it,sw[0],:] ,theta0[m,sw[0],:], Sigma0[m,sw[0],:,:])
                                #- scipy.stats.multivariate_normal.logpdf(theta[i][m,it,sw[1],:] ,theta0[m,sw[1],:], Sigma0[m,sw[1],:,:])
                                + mvnorm_logpdf(theta[i][m,it,sw[0],:] ,theta0[m,sw[0],:], Sigma0_inv_curr[sw[0],:,:], Sigma0_ldet_curr[sw[0]])
                                - mvnorm_logpdf(theta[i][m,it,sw[1],:] ,theta0[m,sw[1],:], Sigma0_inv_curr[sw[1],:,:], Sigma0_ldet_curr[sw[1]])
                        )

                if np.log(np.random.rand()) < alpha:
                    #theta[m, sw[0], :], theta[m, sw[1], :] = theta[m, sw[1], :], theta[m, sw[0], :]
                    #s2[m, sw[0], :], s2[m, sw[1], :] = s2[m, sw[1], :], s2[m, sw[0], :]
                    count_temper[sw[0], sw[1]] += 1
                    for i in range(setup.nexp):
                        sse_curr[i][:, sw[0]], sse_curr[i][:, sw[1]] = sse_curr[i][:, sw[1]], sse_curr[i][:, sw[0]]
                        theta[i][m,:,sw[0],:], theta[i][m,:,sw[1],:] = theta[i][m,:,sw[1],:], theta[i][m,:,sw[0],:]
                        #s2_vec_curr[i][:,sw[0]]
                        s2[i][m, sw[0], :], s2[i][m, sw[1], :] = s2[i][m, sw[1], :], s2[i][m, sw[0], :]
                        #pred_curr_temp = np.copy(pred_curr[i][sw[0], :])
                        #pred_curr[i][sw[0], :] = np.copy(pred_curr[i][sw[1], :])
                        #pred_curr[i][sw[1], :] = np.copy(pred_curr_temp)
                        pred_curr[i][:, sw[0], :], pred_curr[i][:, sw[1], :] = pred_curr[i][:, sw[1], :], pred_curr[i][:, sw[0], :]
                    s2_vec_curr = []
                    for i in range(setup.nexp):
                        s2_vec_curr.append(s2[i][m, :, setup.s2_ind[i]])  # this should be ntemps x ylens[i]
                    theta0[m,sw[0],:], theta0[m,sw[1],:] = theta0[m,sw[1],:], theta0[m,sw[0],:]
                    Sigma0[m,sw[0],:,:], Sigma0[m,sw[1],:,:] = Sigma0[m,sw[1],:,:], Sigma0[m,sw[0],:,:]
                    Sigma0_inv_curr[sw[0],:,:], Sigma0_inv_curr[sw[1],:,:] = Sigma0_inv_curr[sw[1],:,:], Sigma0_inv_curr[sw[0],:,:]
                    Sigma0_ldet_curr[sw[0]], Sigma0_ldet_curr[sw[1]] = Sigma0_ldet_curr[sw[1]], Sigma0_ldet_curr[sw[0]]

        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))

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
        alpha[:] = -np.inf
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
                s2[i][m, :, j] = 1 / np.random.gamma(setup.itl*(setup.ny_s2[i][j]/2 + np.array(setup.ig_a[i][j]) - 1) + 1,
                    1 / (setup.itl*(np.array(setup.ig_b[i][j]) + .5*sseij)))

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
                        - (setup.ig_b[i] / np.log(s2[i][m][sw.T[0]])).sum(axis = 1)
                        + 0.5 * np.log(s2_vec_curr[i][:,sw.T[1]]).sum(axis = 0)
                        + 0.5 * sse_curr[sw.T[1], i]
                        + ((setup.ig_a[i] + 1) * np.log(s2[i][m][sw.T[1]])).sum(axis = 1)
                        + (setup.ig_b[i] / np.log(s2[i][m][sw.T[1]])).sum(axis = 1)
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
