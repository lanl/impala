import numpy as np
#import pyBASS as pb
#import physical_models_vec as pm_vec
#from scipy.interpolate import interp1d
import time
import scipy
from scipy import stats

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
            def constraint_func(xvec):
                return True
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


    def setTemperatureLadder(self, temperature_ladder):
        self.tl = temperature_ladder
        self.itl = 1/self.tl
        self.ntemps = len(self.tl)

    def setMCMC(self, nmcmc, nburn, thin, decor):
        self.nmcmc = nmcmc
        self.nburn = nburn
        self.thin = thin
        self.decor = decor


def normalize(x, bounds):
    """Normalize to 0-1 scale"""
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize(z, bounds):
    """Inverse of normalize"""
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

#@profile
def tran(th, bounds, names):
    #for i in range(th.shape[1]):
    #    th[:,i] = unnormalize(th[:,i],bounds[:,i])
    th = unnormalize(th, bounds)
    return dict(zip(names,th.T))

def chol_sample(mean, cov):
    return mean + np.dot(np.linalg.cholesky(cov), np.random.standard_normal(mean.size))

#from numpy.linalg import slogdet, inv

def mvnorm_logpdf(x, mu, Prec, ldet):
    k = x.shape[-1]
    part1 = -k * 0.5 * np.log(2 * np.pi) - 0.5 * ldet
    x = x - mu
    return part1 + np.squeeze(-x[..., None, :] @ Prec @ x[..., None] / 2)

from collections import namedtuple
OutCalibPool = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 tau pred_curr')
OutCalibHier = namedtuple('OutCalibPool', 'theta s2 count count_decor count_100 count_temper tau pred_curr theta0 Sigma0')







#@profile
def calibHier(setup):
    #np.random.seed(1511)

    t0 = time.time()
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    #n_s2 = np.sum(setup.ns2)
    #s2 = np.empty([setup.nmcmc, setup.ntemps, n_s2])

    ntheta = np.sum(setup.ntheta)

    s2 = []
    theta = []
    for i in range(setup.nexp):
        s2.append(np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]))
        theta.append(np.empty([setup.nmcmc, setup.ntheta[i], setup.ntemps, setup.p]))

    s2_vec_curr = []
    for i in range(setup.nexp):
        s2_vec_curr.append(s2[i][0, :, setup.s2_ind[i]]) # this should be ntemps x ylens[i]


    theta0_start = np.random.uniform(size=[setup.ntemps, setup.p])
    ii = 0
    for t in range(setup.ntemps):
        good = setup.checkConstraints(tran(theta0_start[t, :], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
        while ~good:
            ii += 1
            theta0_start[t, :] = np.random.uniform(size=setup.p)
            good = setup.checkConstraints(tran(theta0_start[t, :], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
            #print(ii)

    theta0[0, :, :] = theta0_start

    for t in range(setup.ntemps):
        Sigma0[0, t, :, :] = np.eye(setup.p)*0.25**2

    for i in range(setup.nexp):
        theta_start = np.empty([setup.ntheta[i], setup.ntemps, setup.p])
        for it in range(setup.ntheta[i]):
            for t in range(setup.ntemps):
                good = False
                ii=0
                while ~good:
                    ii += 1
                    theta_start[it, t, :] = chol_sample(theta0[0, t, :], Sigma0[0, t, :, :])
                    good = setup.checkConstraints(tran(theta_start[it, t, :], setup.bounds_mat, setup.bounds.keys()),
                                                  setup.bounds)
                    #print(ii)

        theta[i][0, :, :, :] = np.copy(theta_start)


    pred_curr = []
    sse_curr = []#np.empty([setup.ntemps, setup.nexp, setup.ntheta])

    for i in range(setup.nexp):
        if setup.ntheta[i] == 1:
            parlist = tran(
                        np.reshape(theta[i][0, :, :, :], [setup.ntemps*setup.ntheta[i], setup.p]),
                        setup.bounds_mat,
                        setup.bounds.keys()
                    )
        else:
            parlist = []
            for it in range(setup.ntheta[i]):
                parlist.append(tran(
                            np.reshape(theta[i][0, :, :, :], [setup.ntemps*setup.ntheta[i], setup.p]),
                            setup.bounds_mat,
                            setup.bounds.keys()
                        ))
        pred_curr.append(
            np.reshape(setup.models[i].eval(parlist), [setup.ntheta[i], setup.ntemps, setup.y_lens[i]]) # not sure if these two reshapes are right
        )
        sse_curr.append(np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 2))
        #sse_curr[:, i, :] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)
        # (array(ntemps x ny x ntheta) - vector(ny))^2 / vector(ny) should be array(ntemps x ny x ntheta), then sum over the ny dimension
        
        ## need to test for ntheta[i]>1 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    eps = 1.0e-13
    cc = 2.4**2/setup.p
    tau = []
    for i in range(setup.nexp):
        tau.append(np.repeat(-4.0, setup.ntemps*setup.ntheta[i]).reshape([setup.ntemps, setup.ntheta[i]]))

    S = []
    cov = []
    mu = []
    for i in range(setup.nexp):
        S.append(np.empty([setup.ntemps, setup.ntheta[i], setup.p, setup.p]))
        cov.append(np.empty([setup.ntemps, setup.ntheta[i], setup.p, setup.p]))
        mu.append(np.empty([setup.ntemps, setup.ntheta[i], setup.p]))
        for t in range(setup.ntemps):
            for it in range(setup.ntheta[i]):
                S[i][t,it,:,:] = np.eye(setup.p)*1e-6


    theta0_prior_cov = np.eye(setup.p)*1.**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = np.repeat(.5, setup.p)

    Sigma0_prior_df = setup.p
    Sigma0_prior_scale = np.eye(setup.p)*.1**2

    Sigma0_ldet_curr = np.empty(setup.ntemps)
    Sigma0_inv_curr = np.empty([setup.ntemps, setup.p, setup.p])
    for t in range(setup.ntemps):
        Sigma0_ldet_curr[t] = np.linalg.slogdet(Sigma0[0,t,:,:])[1]
        Sigma0_inv_curr[t,:,:] = np.linalg.inv(Sigma0[0,t,:,:])

    count = []
    count_decor = []
    count_100 = []

    count_temper = np.zeros([setup.ntemps, setup.ntemps])

    for i in range(setup.nexp):
        count.append(np.zeros([setup.ntheta[i], setup.ntemps]))
        count_decor.append(np.zeros([setup.ntheta[i], setup.p, setup.ntemps]))
        count_100.append(np.zeros([setup.ntheta[i], setup.ntemps]))

    pred_cand = np.copy(pred_curr)
    sse_cand = np.copy(sse_curr)

    ## start MCMC
    for m in range(1,setup.nmcmc):
        #print(m)

        for i in range(setup.nexp):
            theta[i][m,:,:,:] = np.copy(theta[i][m-1,:,:,:]) # current set to previous, will change if accepted

        ## adaptive Metropolis for each temperature
        if m == 300:
            for i in range(setup.nexp):
                for it in range(setup.ntheta[i]):
                    for t in range(setup.ntemps):
                        mu[i][t,it,:] = np.mean(theta[i][:(m-1),it,t,:], 0)
                        cov[i][t,it,:,:] = np.cov(theta[i][:(m-1),it,t,:], rowvar=False)
                        S[i][t,it,:,:] = (cov[i][t,it,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[i][t,it])

        if m > 300:
            for i in range(setup.nexp):
                for it in range(setup.ntheta[i]):
                    for t in range(setup.ntemps):
                        mu[i][t,it,:] = mu[i][t,it,:] + (theta[i][m-1,it,t,:] - mu[i][t,it,:]) / (m-1)
                        cov[i][t,it,:,:] = (m-1)/(m-2) * cov[i][t,it,:,:] + (m-2)/(m-1)**2 * \
                            np.outer(theta[i][m-1,it,t,:] - mu[i][t,it,:], theta[i][m-1,it,t,:] - mu[i][t,it,:])
                        S[i][t,it,:,:] = (cov[i][t,it,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[i][t,it])

        # generate proposal
        theta_cand = []
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


















#@profile
def calibPool(setup):
    #np.random.seed(1511)

    t0 = time.time()
    theta = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    n_s2 = np.sum(setup.ns2)
    #s2 = np.empty([setup.nmcmc, setup.ntemps, n_s2])

    s2 = []
    for i in range(setup.nexp):
        s2.append(np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]))

    s2_vec_curr = []
    for i in range(setup.nexp):
        s2_vec_curr.append(s2[i][0, :, setup.s2_ind[i]]) # this should be ntemps x ylens[i]


    theta_start = np.random.uniform(size=[setup.ntemps, setup.p])
    #good_values = setup.checkConstraints(tran(theta_start, setup.bounds_mat, setup.bounds.keys()), setup.bounds)
    ii = 0
    for t in range(setup.ntemps):
        good = setup.checkConstraints(tran(theta_start[t, :], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
        while ~good:
            ii += 1
            theta_start[t, :] = np.random.uniform(size=setup.p)
            good = setup.checkConstraints(tran(theta_start[t, :], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
            #print(ii)


    theta[0, :, :] = theta_start

    pred_curr = []
    sse_curr = np.empty([setup.ntemps, setup.nexp])
    for i in range(setup.nexp):
        pred_curr.append(setup.models[i].eval(tran(theta[0, :, :], setup.bounds_mat, setup.bounds.keys())))
        sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)

    eps = 1.0e-13
    tau = np.repeat(-4.0, setup.ntemps)
    cc = 2.4**2/setup.p
    S = np.empty([setup.ntemps, setup.p, setup.p])
    for t in range(setup.ntemps):
        S[t,:,:] = np.eye(setup.p)*1e-6
    cov = np.empty([setup.ntemps, setup.p, setup.p])
    mu = np.empty([setup.ntemps, setup.p])

    count = np.zeros([setup.ntemps, setup.ntemps])
    count_decor = np.zeros([setup.p, setup.ntemps])
    count_100 = np.zeros(setup.ntemps)

    pred_cand = np.copy(pred_curr)
    sse_cand = np.copy(sse_curr)

    ## start MCMC
    for m in range(1,setup.nmcmc):
        theta[m,:,:] = np.copy(theta[m-1,:,:]) # current set to previous, will change if accepted

        ## adaptive Metropolis for each temperature
        if m == 300:
            for t in range(setup.ntemps):
                mu[t,:] = np.mean(theta[:(m-1),t,:], 0)
                cov[t,:,:] = np.cov(theta[:(m-1),t,:], rowvar=False)
                S[t,:,:] = (cov[t,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[t])

        if m > 300:
            for t in range(setup.ntemps):
                mu[t,:] = mu[t,:] + (theta[m-1,t,:] - mu[t,:]) / (m-1)
                cov[t,:,:] = (m-1)/(m-2) * cov[t,:,:] + (m-2)/(m-1)**2 * np.outer(theta[m-1,t,:] - mu[t,:], theta[m-1,t,:] - mu[t,:])
                S[t,:,:] = (cov[t,:,:]*cc + np.eye(setup.p)*eps*cc) * np.exp(tau[t])

        # generate proposal
        theta_cand = np.empty([setup.ntemps, setup.p])
        #skip = [False]*setup.ntemps
        for t in range(setup.ntemps):
            theta_cand[t,:] = chol_sample(theta[m-1,t,:], S[t,:,:])
            #if np.any(theta_cand[t, :] < 0) or np.any(theta_cand[t, :] > 1):
            #    skip[t] = True
            #    theta_cand[t, :] = theta_cand[t, :]*0.

        good_values = setup.checkConstraints(tran(theta_cand, setup.bounds_mat, setup.bounds.keys()), setup.bounds)

        # get predictions and SSE
        #pred_cand = []
        #sse_cand = np.empty([setup.ntemps, setup.nexp])
        pred_cand = np.copy(pred_curr)
        sse_cand = np.copy(sse_curr)
        if np.any(good_values):
            for i in range(setup.nexp):
                #np.array(range(setup.nexp))[good_values]
                pred_cand[i][good_values, :] = setup.models[i].eval(tran(theta_cand[good_values,:], setup.bounds_mat, setup.bounds.keys()))
                #sse_cand[:,i] = np.sum((pred_cand[i] - setup.ys[i])**2, 0)
                sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)

        # for each temperature, accept or reject
        for t in range(setup.ntemps):
            if ~good_values[t]:
                # todo: need to add constraint function, rescaled (also to decorrelation step)
                # right now PTW is returning -999 if any of the parameter values is bad
                # instead use None but only for the bad parameter combinations
                # todo: better yet, make a check in this file, don't even pass those parameter combinations to PTW

                alpha = -9999
            else:
                alpha = np.sum(-0.5*setup.itl[t]*(sse_cand[t,:] - sse_curr[t,:]))

            if np.log(np.random.rand()) < alpha:
                theta[m,t,:] = theta_cand[t,:]
                count[t,t] += 1
                for i in range(setup.nexp):
                    pred_curr[i][t,:] = pred_cand[i][t,:]
                sse_curr[t,:] = sse_cand[t,:]
                count_100[t] += 1

        # diminishing adaptation based on acceptance rate for each temperature
        if m % 100 == 0 and m > 300:
            delta = min(0.1, 1/np.sqrt(m+1)*5)
            for t in range(setup.ntemps):
                if count_100[t] < 23:
                    tau[t] -= delta
                elif count_100[t] > 23:
                    tau[t] += delta
            count_100 = count_100*0

        ## decorrelation step
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = np.copy(theta[m,:,:])
                theta_cand[:,k] = np.random.rand(setup.ntemps) # independence proposal, will vectorize of columns
                good_values = setup.checkConstraints(tran(theta_cand, setup.bounds_mat, setup.bounds.keys()),
                                                     setup.bounds)
                #pred_cand = []
                #sse_cand = np.empty([setup.ntemps, setup.nexp])

                pred_cand = np.copy(pred_curr)
                sse_cand = np.copy(sse_curr)

                if np.any(good_values):
                    for i in range(setup.nexp):
                        pred_cand[i][good_values, :] = setup.models[i].eval(tran(theta_cand[good_values, :], setup.bounds_mat, setup.bounds.keys()))
                        #sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) ** 2, 0)
                        sse_cand[:, i] = np.sum((pred_cand[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)

                for t in range(setup.ntemps): # need to add constraint function
                    if ~good_values[t]:
                        alpha = -9999
                    else:
                        alpha = np.sum(-0.5 * setup.itl[t] * (sse_cand[t, :] - sse_curr[t, :]))
                        #print(~good_values[t])

                    if np.log(np.random.rand()) < alpha:
                        theta[m, t, k] = theta_cand[t, k]
                        count_decor[k, t] += 1
                        for i in range(setup.nexp):
                            pred_curr[i][t, :] = pred_cand[i][t, :]
                        sse_curr[t, :] = sse_cand[t, :]

#        bb = setup.checkConstraints(tran(theta[m, 0, :], setup.bounds_mat, setup.bounds.keys()), setup.bounds)
#        if ~bb:
#            print('no')

        ## Gibbs update s2
        for i in range(setup.nexp):
            dev_sq = (pred_cand[i] - setup.ys[i])**2 # squared deviations
            for j in range(setup.ns2[i]):
                sseij = np.sum(dev_sq[:,setup.s2_ind[i]==j], 1) # sse for jth s2ind in experiment i
                s2[i][m, :, j] = 1 / np.random.gamma(setup.itl*(setup.ny_s2[i][j]/2 + np.array(setup.ig_a[i][j]) - 1) + 1,
                    1 / (setup.itl*(np.array(setup.ig_b[i][j]) + .5*sseij)))

        s2_vec_curr = []
        for i in range(setup.nexp):
            s2_vec_curr.append(s2[i][m, :, setup.s2_ind[i]])  # this should be ntemps x ylens[i]

        for i in range(setup.nexp):
            sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)

        ## tempering swaps
        if m > 1000 and setup.ntemps > 1:
            for _ in range(setup.ntemps): # do many swaps
                sw = np.sort(np.random.choice(range(setup.ntemps), size=2, replace=False)) # indices to swap

                alpha = 0
                for i in range(setup.nexp):
                    alpha += (setup.itl[sw[1]] - setup.itl[sw[0]])*(
                            - .5*np.sum(np.log(s2_vec_curr[i][:,sw[0]])) - 0.5*sse_curr[sw[0], i]
                            - np.sum((setup.ig_a[i] + 1)*np.log(s2[i][m, sw[0], :])) - np.sum(setup.ig_b[i]/np.log(s2[i][m, sw[0], :]))
                            + .5*np.sum(np.log(s2_vec_curr[i][:,sw[1]])) + 0.5*sse_curr[sw[1], i]
                            + np.sum((setup.ig_a[i] + 1)*np.log(s2[i][m, sw[1], :])) + np.sum(setup.ig_b[i]/np.log(s2[i][m, sw[1], :]))
                    )

                if np.log(np.random.rand()) < alpha:
                    theta[m, sw[0], :], theta[m, sw[1], :] = theta[m, sw[1], :], theta[m, sw[0], :]
                    #s2[m, sw[0], :], s2[m, sw[1], :] = s2[m, sw[1], :], s2[m, sw[0], :]
                    count[sw[0], sw[1]] += 1
                    sse_curr[sw[0], :], sse_curr[sw[1], :] = sse_curr[sw[1], :], sse_curr[sw[0], :]
                    for i in range(setup.nexp):
                        #s2_vec_curr[i][:,sw[0]]
                        s2[i][m, sw[0], :], s2[i][m, sw[1], :] = s2[i][m, sw[1], :], s2[i][m, sw[0], :]
                        #pred_curr_temp = np.copy(pred_curr[i][sw[0], :])
                        #pred_curr[i][sw[0], :] = np.copy(pred_curr[i][sw[1], :])
                        #pred_curr[i][sw[1], :] = np.copy(pred_curr_temp)
                        pred_curr[i][sw[0], :], pred_curr[i][sw[1], :] = pred_curr[i][sw[1], :], pred_curr[i][sw[0], :]
                    s2_vec_curr = []
                    for i in range(setup.nexp):
                        s2_vec_curr.append(s2[i][m, :, setup.s2_ind[i]])  # this should be ntemps x ylens[i]

        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))

    out = OutCalibPool(theta, s2, count, count_decor, count_100, tau, pred_curr)
    return(out)

