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
from collections import namedtuple
import impala
from impala.superCal.impala_noProbit_emu import *
#import pbar
#np.seterr(under='ignore')

# no probit tranform for hierarchical and DP versions


##############################################################################################################################################################################
def bincount2D_vectorized(a, max_count):
    """
    Applies np.bincount across a 2d array (row-wise).

    Adapted From: https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    a_offs = a + np.arange(a.shape[0])[:,None]*max_count
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*max_count).reshape(-1,max_count)

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
    https://www.mathworks.com/help/matlab/import_export/using-mapreduce-to-compute-covariance-and-related-quantities.html
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



# West 1992, dirichletprocess R package
def sample_eta(curr_eta, nclust, ndat, prior_shape, prior_rate):
    g = beta(curr_eta + 1, ndat)
    aa = prior_shape + nclust
    bb = prior_rate - np.log(g)
    eps = (aa - 1) / (ndat * bb + aa - 1)
    sel = (uniform(size = aa.shape) < eps) * 1
    aaa = np.vstack((aa,aa - 1)).T[np.arange(aa.shape[0]), sel]
    return gamma(shape = aaa, scale = 1 / bb)

    #S   = np.empty((setup.ntemps, setup.nclustmax, setup.p, setup.p))
    #S[:] = np.eye(setup.p) * 1e-4
    #nS  = np.zeros((setup.ntemps, setup.nclustmax))
    #mS  = np.zeros((setup.ntemps, setup.nclustmax, setup.p))
    #cov = [np.empty((setup.ntemps, setup.ns2[i], setup.p, setup.p)) for i in range(setup.nexp)]
    #mu  = [np.empty((setup.ntemps, setup.ns2[i], setup.p)) for i in range(setup.nexp)]

class AMcov_clust:
    """
    We want to do adaptive metropolis (AM) for theta. There are nclustmax theta vectors, but we 
    can't do AM on them because they will be switching labels. Instead we do AM on the experiment
    specific theta histories, but then need to combine them to get a sample for the 
    cluster. Within each cluster, we weight the means and covariances from experiments within
    the cluster. The way Peter has done this is to accumulate with the proper weights. He 
    loops over experiments, which each have a cov and mu, and adds them (weighted) to the 
    appropriate cluster cov and mu.

    Adaptive Metropolis for clustered calibration. Notes:
    - You cannot just do AM on 'theta' beause of label switching
    - Instead do AM on 'theta_hist' because it always exists with same dimension
    - Tricky part is that we are then proposing an update to theta instead of theta_hist
    - Combine relevant theta_hists (appropriately weighted) for updating theta
    - https://www.mathworks.com/help/matlab/import_export/using-mapreduce-to-compute-covariance-and-related-quantities.html
    """
    def __init__(self, nexp, ntheta, ntemps, p, nclustmax, start_var=1e-4, start_adapt_iter=300, tau_start=0.):
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = tau_start * np.ones(ntemps) # note this is different from hier--not sure how to have a different tau for each experiment
        self.S   = np.empty((ntemps, nclustmax, p, p)) # proposal covariance
        for t in range(ntemps):
            for c in range(nclustmax):
                self.S[t,c] = np.eye(p) * start_var
        self.cov_hist = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)] # covariance of past sample by experiment
        self.mu_hist  = [np.empty((ntemps, ntheta[i], p)) for i in range(nexp)] # mean of past samples by experiment
        self.cov_clust = np.empty((ntemps, nclustmax, p, p)) # covariance of past samples by cluster (weighted combination of cov_hist)
        self.mu_clust = np.zeros((ntemps, nclustmax, p)) # mean of past sample by cluster (weighted combination of mu_hist)
        self.n_hc = np.zeros((ntemps, nclustmax)) # count for accumulation purposes 
        self.nexp = nexp
        self.ntheta = ntheta
        self.ntemps = ntemps
        self.nclustmax = nclustmax
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = np.zeros(self.ntemps, dtype = int)
        self.nS  = np.zeros((ntemps, nclustmax))
        self.temps = np.arange(ntemps)

    def update(self, x, m, curr_delta):
        if m > self.start_adapt_iter:
            # update experiment-specific mu and cov
            for i in range(self.nexp):
                self.mu_hist[i] += (x[i][m-1] - self.mu_hist[i]) / m
                self.cov_hist[i][:] = (
                    + ((m-1) / m) * self.cov_hist[i]
                    + ((m-1) / (m * m)) * np.einsum(
                        'tej,tel->tejl', x[i][m-1] - self.mu_hist[i], x[i][m-1] - self.mu_hist[i],
                        )
                    )
            # reweight experiment-specific to get cluster-specific mu and cov
            self.cluster_covariance_update(m, curr_delta)
            # make history covariance into proposal covariance
            self.S = self.AM_SCALAR * np.einsum('tmjl,t->tmjl', self.cov_clust + np.eye(self.p) * self.eps, np.exp(self.tau)) # is identity be added correctly (correctly broadcast)?


        elif m == self.start_adapt_iter:
            # calculate experiment-specific mu and cov
            for i in range(self.nexp):
                self.mu_hist[i][:]  = x[i][:m].mean(axis = 0)
                self.cov_hist[i][:] = cov_4d_pcm(x[i][:m], self.mu_hist[i])

            # reweight experiment-specific to get cluster-specific mu and cov
            self.cluster_covariance_update(m, curr_delta)
            # make history covariance into proposal covariance
            self.S = self.AM_SCALAR * np.einsum('tmjl,t->tmjl', self.cov_clust + np.eye(self.p) * self.eps, np.exp(self.tau)) # is identity be added correctly (correctly broadcast)?

    def cluster_covariance_update(self, n, delta):
        # https://www.mathworks.com/help/matlab/import_export/using-mapreduce-to-compute-covariance-and-related-quantities.html
        self.cov_clust[:]  = 0. # accumulating covariance                                                                                   #(c1)
        self.mu_clust[:] = 0. # accumulating mean                                                                                           #(m1)
        self.n_hc[:] = 0. # accumulating count                                                                                              #(n1)
        mu_new = np.empty((self.ntemps, self.cov_clust.shape[-1]))                                                                          #(m)
        n_new = np.zeros((self.ntemps, 1))                                                                                                  #(n)
        for i in range(self.nexp): # loop over unvectorized experiments
            for j in range(delta[i].shape[1]): # loop over vectorized experiments
                n_new[:] = self.n_hc[self.temps, delta[i].T[j], None] + n                                                    #(n=n1+n2)
                mu_new[:] = (self.n_hc[self.temps, delta[i].T[j],None] * self.mu_clust[self.temps, delta[i].T[j]] + n * self.mu_hist[i][self.temps, j]) / n_new #(m=(n1*m1 + n2*m2)/n)
                self.cov_clust[self.temps, delta[i].T[j]] =  1 / n_new[:,:,None] * (                                                #1/n * 
                    + self.n_hc[self.temps, delta[i].T[j],None,None] * self.cov_clust[self.temps, delta[i].T[j]]                  # cov1 (what has accumulated so far)  #(c1)
                    + n * self.cov_hist[i][self.temps, j]                                                                   # cov2 (history for experiment ij)          #(c2)
                    + np.einsum('t,tp,tq->tpq', self.n_hc[self.temps, delta[i].T[j]],                                     # mu1                                         
                                    self.mu_clust[self.temps, delta[i].T[j]] - mu_new, self.mu_clust[self.temps, delta[i].T[j]] - mu_new)
                    + n * np.einsum('tp,tq->tpq', self.mu_hist[i][self.temps, j] - mu_new, self.mu_hist[i][self.temps, j] - mu_new)     # mu2
                    )
                self.mu_clust[self.temps, delta[i].T[j]] = mu_new       #(m1=m)
                self.n_hc[self.temps, delta[i].T[j]] = n_new.ravel()    #(n1=n)
        return
        

    def update_tau(self, m): # label switching makes it difficult to track acceptance rate, so this is turned off
        # diminishing adaptation based on acceptance rate for each temperature
        if False:
            if (m % 100 == 0) and (m > 300):
                ddelta = min(0.1, 5 / sqrt(m + 1))
                self.tau[np.where(self.count_100 < 23)] = self.tau[np.where(self.count_100 < 23)] - ddelta
                self.tau[np.where(self.count_100 > 23)] = self.tau[np.where(self.count_100 > 23)] + ddelta
                self.count_100 *= 0

    def gen_cand(self, x, m): # one sample per cluster
        #x_cand = [chol_sample_1per(x[i][m-1], self.S) for i in range(self.nexp)]
        x_cand = chol_sample_1per(x[m-1], self.S)
        return x_cand


OutCalibClust = namedtuple(
    'OutCalibClust', 'theta theta_hist s2 count count_temper pred_curr theta0 Sigma0 delta eta nclustmax theta_am'
    )

## DP Cluster Calibration

def calibClust(setup, parallel = False):
    t0 = time.time()

    if parallel:
        pool = mp.Pool(processes = mp.cpu_count())
    else:
        pool = None
    
    ## Constants Declaration    

    # boolean with dimension [i] x [ylens[i], ns2[i]] indicating which s2 to use for each output
    s2_ind_mat = [(setup.s2_ind[i][:,None] == range(setup.ns2[i])) for i in range(setup.nexp)]
    
    # boolean with dimension [i] x [ylens[i], ntheta[i]] indicating which theta to use for each output
    theta_ind_mat = [
        (setup.theta_ind[i][:,None] == range(setup.ntheta[i]))
        for i in range(setup.nexp)
        ]

    # sequence of length ntemps
    temps = np.arange(setup.ntemps)

    # matrix of repeated inverse temperatures of dimension [ntemps, nclustmax]
    itl_mat_theta = (np.ones((setup.ntemps, setup.nclustmax)).T * setup.itl).T
    
    # repeated inverse temperatures of dimension [i] x [ntemps, ns2[i]]
    itl_mat_s2 = [
        np.ones((setup.ntemps, setup.ns2[i])) * setup.itl.reshape(-1,1)
        for i in range(setup.nexp)
        ]

    ## Parameter Declaration
    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    #theta0[0] = chol_sample_1per_constraints(
    #        np.zeros((setup.ntemps, setup.p)), np.array([np.eye(setup.p)] * setup.ntemps),
    #        setup.checkConstraints, setup.bounds_mat, setup.bounds.keys(), setup.bounds,
    #        )
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

    Sigma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    Sigma0[0] = np.eye(setup.p) * 0.25**2
    # initialize delta, the cluster membership indicator (for each experiment)
    delta  = [np.empty([setup.nmcmc, setup.ntemps, setup.ntheta[i]], dtype = int) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        delta[i][0] = choice(setup.nclustmax, size = (setup.ntemps, setup.ntheta[i]))
    delta_ind_mat = [(delta[i][0,:,:,None] == range(setup.nclustmax)) for i in range(setup.nexp)]
    clust_mem_count = np.zeros((setup.ntemps, setup.nclustmax), dtype = int) # (extant) cluster weight for ijth experiment
    clust_nomem_bool = np.zeros(clust_mem_count.shape, dtype = bool) 
    #clust_nomem_bool = np.zeros(delta_ind_mat.shape, dtype = bool)                  # (njs > 0) -> false
    djs = np.empty(clust_mem_count.shape)                                    # (candidate) cluster weight for ijth experiment
    cluster_weights = np.empty(clust_mem_count.shape)
    for i in range(setup.nexp):
        clust_mem_count[:] += bincount2D_vectorized(delta[i][0], setup.nclustmax)
    curr_delta    = [delta[i][0].copy() for i in range(setup.nexp)]
    theta_unravel = [np.repeat(np.arange(setup.ntemps), setup.ntheta[i]) for i in range(setup.nexp)]
    # initialize Theta
    theta_long_shape = (setup.ntemps * setup.nclustmax, setup.p)
    theta_wide_shape = (setup.ntemps, setup.nclustmax, setup.p)
    theta      = np.empty((setup.nmcmc, setup.ntemps, setup.nclustmax, setup.p))
    theta[0]   = chol_sample_nper_constraints(
           theta0[0], Sigma0[0], setup.nclustmax, setup.checkConstraints,
           setup.bounds_mat, setup.bounds.keys(), setup.bounds,
           )
    theta_hist = [np.empty((setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p)) for i in range(setup.nexp)]
    theta_cand = np.empty(theta_wide_shape)
    for i in range(setup.nexp):
        theta_hist[i][0] = (
            theta[0, theta_unravel[i], delta[i][0].ravel()].reshape(setup.ntemps, setup.ns2[i], setup.p)
            )
    theta_eval = np.empty(theta_wide_shape)
    theta_ext = np.zeros((setup.ntemps, setup.nclustmax), dtype = bool) # array of (current) extant theta locs
    for i in range(setup.nexp):
        theta_ext[theta_unravel[i], delta[i][0].ravel()] += True
    ntheta = theta_ext.sum(axis = 1) # count extant thetas
    # initialize sigma2 and eta
    log_s2 = [np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]]) for i in range(setup.nexp)]
    eta = np.empty((setup.nmcmc, setup.ntemps))
    eta[0] = 5.  

    s2_which_mat = [
        [np.where(s2_ind_mat[i][:,j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
        ]
    theta_which_mat = [
        [np.where(theta_ind_mat[i][:,j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
        ]

    ## Initialize *Current* Variables
    
    #LAUREN COMMENTED OUT LAST THREE ELEMENTS
    #pred_curr       = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    #llik_curr        = [None] * setup.nexp # [i], ntheta[i] x ntemps ### was [i] x [ntemps, ns2[i]]
    #llik_cand  = [None] * setup.nexp # [i], ntheta[i] x ntemps
    #dev_sq          = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    pred_curr_delta = [None] * setup.nexp # [i] x [ntemps, nclustmax, ylens[i]]
    llik_curr_delta  = [None] * setup.nexp # [i] x [ntemps, nclustmax, ns2[i]]
    pred_cand_theta = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    pred_curr_theta = [None] * setup.nexp # [i] x [ntemps, ylens[i]]
    llik_cand_theta  = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    llik_curr_theta  = [None] * setup.nexp # [i] x [ntemps, ns2[i]]
    llik_cand_theta_ = np.zeros((setup.ntemps, setup.nclustmax)) # above, Summarized by cluster assignment (i.e., llik for each unique theta, summed over experiments that share that theta)
    llik_curr_theta_ = np.zeros((setup.ntemps, setup.nclustmax)) #
    marg_lik_cov_curr = [None] * setup.nexp # does not need to get summed over experiments (should be one for each experiment, regardless of cluster)
    for i in range(setup.nexp):
        ### Initialize predictions for theta_i's
        pred_curr_theta[i] = setup.models[i].eval(
            tran_unif(theta_hist[i][0].reshape(-1, setup.p),  #0 for first iteration
                    setup.bounds_mat, setup.bounds.keys()), 
            False,
            )
        pred_cand_theta[i] = pred_curr_theta[i].copy()
        marg_lik_cov_curr[i] = [None] * setup.ntemps
        llik_curr_theta[i] = np.empty([setup.ntemps, setup.ntheta[i]])
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = [None] * setup.ntheta[i]
            s2_stretched = log_s2[i][0][t,setup.theta_ind[i]]
            for j in range(setup.ntheta[i]):
                marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(s2_stretched[s2_which_mat[i][j]]))
                llik_curr_theta[i][t][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_curr_theta[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 
        llik_cand_theta[i] = llik_curr_theta[i].copy()

        ### Initialize predictions for clusters
        pred_curr_delta[i] = setup.models[i].eval(
            tran_unif(theta[0].reshape(-1, setup.p), setup.bounds_mat, setup.bounds.keys()), True,
            ).reshape(setup.ntemps, setup.nclustmax, setup.y_lens[i])
        llik_curr_delta[i] = np.empty([setup.ntemps, setup.nclustmax, setup.ntheta[i]])
        for t in range(setup.ntemps):
            s2_stretched = log_s2[i][0][t,setup.theta_ind[i]]
            for j in range(setup.ntheta[i]):
                for k in range(setup.nclustmax):
                    llik_curr_delta[i][t][k][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_curr_delta[i][t][k][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 

    ## Initialize Adaptive Metropolis related Variables
    #S   = np.empty((setup.ntemps, setup.nclustmax, setup.p, setup.p))
    #S[:] = np.eye(setup.p) * 1e-4
    #nS  = np.zeros((setup.ntemps, setup.nclustmax))
    #mS  = np.zeros((setup.ntemps, setup.nclustmax, setup.p))
    #cov = [np.empty((setup.ntemps, setup.ns2[i], setup.p, setup.p)) for i in range(setup.nexp)]
    #mu  = [np.empty((setup.ntemps, setup.ns2[i], setup.p)) for i in range(setup.nexp)]
    cov_theta_cand = AMcov_clust(
        nexp=setup.nexp, 
        ntheta=np.array([setup.ntheta[i] for i in range(setup.nexp)]), 
        ntemps=setup.ntemps, 
        p=setup.p, 
        nclustmax=setup.nclustmax,
        start_var=setup.start_var_theta, 
        start_adapt_iter=setup.start_adapt_iter, 
        tau_start=setup.start_tau_theta
        )

    ## Initialize Theta0 and Sigma0 related variables 
    theta0_prior_cov = setup.theta0_prior_cov # was 10^2, but that is far from uniform when back transforming
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_mean = setup.theta0_prior_mean #np.repeat(0.5, setup.p)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]
    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))
    Sigma0_prior_df    = setup.Sigma0_prior_df #setup.p
    Sigma0_prior_scale = setup.Sigma0_prior_scale #np.eye(setup.p)*.1**2#/setup.p
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


    ## start MCMC
    for m in range(1,setup.nmcmc):
        #------------------------------------------------------------------------------------------
        ## Gibbs Update for delta (cluster identifier)
        for i in range(setup.nexp):
            delta[i][m] = delta[i][m-1]
        
        #   Establish initial cluster weighting
        clust_mem_count[:] = 0.
        for l in range(setup.nexp):
            clust_mem_count[:] += bincount2D_vectorized(delta[l][m], setup.nclustmax)
        clust_nomem_bool[:] = (clust_mem_count == 0) # fixed at start of iteration
                                # if (non-extant) cluster j becomes extant, then set this to False.


        ##################
        ### Draw Delta ###
        ##################
        for i in range(setup.nexp):
            cluster_sample_unif[i][:] = uniform(size = (setup.ntemps, setup.ns2[i]))
            for j in range(setup.ntheta[i]):
                # weighting assigned to extant clusters for ij'th within-vectorized experiment
                clust_mem_count[temps, delta[i][m,:,j]] -= 1
                # weighting assigned to candidate (non-extant) clusters
                djs[:] = clust_nomem_bool * (eta[m-1] / (clust_nomem_bool.sum(axis = 1) + 1e-9)).reshape(-1,1)
                # djs[:] = (njs == 0) * (eta[m-1] / (njs == 0).sum(axis = 1)).reshape(-1,1)
                # unnormalized log-probability of cluster membership
                with np.errstate(divide='ignore', invalid = 'ignore'):
                    cluster_cum_prob[:] = (
                        + np.log(clust_mem_count + djs) # this line doesnt match paper? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        + llik_curr_delta[i][:,:,j] * setup.itl.reshape(-1,1)
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
                clust_mem_count[temps, delta[i][m,:,j]] += 1
                # If a candidate cluster became extant--remove candidate flag.
                clust_nomem_bool[clust_mem_count > 0] = False
            
            delta_ind_mat[i][:] = delta[i][m,:,:,None] == range(setup.nclustmax)
            curr_delta[i][:] = delta[i][m]
        
        ##########################
        ### Update Theta Evals ###
        ########################## 
        for i in range(setup.nexp):
            pred_curr_theta[i][:] = setup.models[i].eval(
                tran_unif(theta[m-1, theta_unravel[i], delta[i][m].ravel()],
                        setup.bounds_mat, setup.bounds.keys())
                ) #update after delta update before
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    llik_curr_theta[i][t][j] = setup.models[i].llik( setup.ys[i][theta_which_mat[i][j]], pred_curr_theta[i][t][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j])

                
        #------------------------------------------------------------------------------------------
        ## adaptive Metropolis per Cluster
        if False:
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
        
        ###################
        ### Draw Thetas ###
        ###################

        llik_cand_theta_[:] = 0.
        llik_curr_theta_[:] = 0.
        theta[m]      = theta[m-1]
        theta_eval[:] = theta[m-1]

        cov_theta_cand.update(theta_hist, m, curr_delta)
        theta_cand = cov_theta_cand.gen_cand(theta, m)
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
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    llik_cand_theta[i][t][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]], 
                        pred_cand_theta[i][t][theta_which_mat[i][j]], 
                        marg_lik_cov_curr[i][t][j]
                    )
                    llik_cand_theta_[t][curr_delta[i][t,j]] += llik_cand_theta[i][t][j]
                    llik_curr_theta_[t][curr_delta[i][t,j]] += llik_curr_theta[i][t][j]

        alpha[:] = - np.inf
        alpha[good_values] = (itl_mat_theta * (llik_cand_theta_ - llik_curr_theta_)
            + mvnorm_logpdf_(theta_cand, theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)
            - mvnorm_logpdf_(theta[m-1], theta0[m-1], Sigma0_inv_curr, Sigma0_ldet_curr)
            )[good_values]
        accept[:] = np.log(uniform(size = alpha.shape)) < alpha
        theta[m,accept] = theta_cand[accept]

        for i in range(setup.nexp):
            theta_hist[i][m] = (
                theta[m, theta_unravel[i], delta[i][m].ravel()
                    ].reshape(setup.ntemps, setup.ns2[i], setup.p)
                )
            pred_curr_theta[i][:] = setup.models[i].eval(
                tran_unif(theta_hist[i][m].reshape(-1,setup.p),setup.bounds_mat, setup.bounds.keys()),
                False,
                )
            llik_curr_theta[i][:] = llik_cand_theta[i][:]

                
        count += accept.sum(axis = 1)
        cov_theta_cand.count_100 += accept.sum(axis = 1)

        cov_theta_cand.update_tau(m)
        theta_ext[:] = False
        for i in range(setup.nexp):
            theta_ext[theta_unravel[i], delta[i][m].ravel()] += True
        ntheta[:] = theta_ext.sum(axis = 1)


        ########################
        ###  Gibbs update s2 ###
        ########################
        for i in range(setup.nexp):

            if setup.models[i].s2=='gibbs':
                dev_sq = (pred_curr_theta[i] - setup.ys[i])**2 @ s2_ind_mat[i] # squared deviations
                log_s2[i][m] = np.log(1 / np.random.gamma(
                        itl_mat_s2[i] * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1) - 1,
                        1 / (itl_mat_s2[i] * (setup.ig_b[i] + dev_sq / 2)),
                        ))
                for t in range(setup.ntemps):
                    s2_stretched = log_s2[i][m][t,setup.theta_ind[i]]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(s2_stretched[s2_which_mat[i][j]]))
                        llik_curr_theta[i][t][j] = setup.models[i].llik(setup.ys[i][s2_which_mat[i][j]], pred_curr_theta[i][t][s2_which_mat[i][j]], marg_lik_cov_curr[i][t][j])
                
            elif setup.models[i].s2=='fix':
                    log_s2[i][m] = np.log(setup.sd_est[i]**2)

                    for t in range(setup.ntemps):
                        s2_stretched = log_s2[i][m][t,setup.theta_ind[i]]
                        for j in range(setup.ntheta[i]):
                            marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(s2_stretched[s2_which_mat[i][j]]))
                            llik_curr_theta[i][t][j] = setup.models[i].llik(setup.ys[i][s2_which_mat[i][j]], pred_curr_theta[i][t][s2_which_mat[i][j]], marg_lik_cov_curr[i][t][j])

            else:
                print('TODO: fill in s2 sampling without gibbs')


            
        ##########################
        ### Update Delta Evals ### (THIS IS THE SLOW CHUNK THAT IS MESSING WITH RUNTIME!!!! Problem is compute_state_history looping over NHist)
        ##########################
        for i in range(setup.nexp):
            pred_curr_delta[i][:] = setup.models[i].eval(  #(IN PARTICULAR, THIS CALL!!!!)
                tran_unif(
                    theta[m].reshape(-1, setup.p).repeat(setup.ns2[i], axis = 0),
                    setup.bounds_mat, 
                    setup.bounds.keys(),
                    ),
                ).reshape(setup.ntemps, setup.nclustmax, setup.y_lens[i]) 
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    for k in range(setup.nclustmax):
                        llik_curr_delta[i][t][k][j] = setup.models[i].llik(setup.ys[i][theta_which_mat[i][j]], pred_curr_delta[i][t][k][theta_which_mat[i][j]], marg_lik_cov_curr[i][t][j]) 

                
        ###########################
        ### Gibbs Update Theta0 ###
        ###########################
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

        ###########################       
        ### Gibbs update Sigma0 ###
        ###########################
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


        ###############################################
        ### Gibbs Update for Theta (Not in Cluster) ###
        ###############################################
        theta_cand[:] = chol_sample_nper_constraints(
                theta0[m], Sigma0[m], setup.nclustmax, setup.checkConstraints,
                setup.bounds_mat, setup.bounds.keys(), setup.bounds,
                )
        theta[m,~theta_ext] = theta_cand[~theta_ext]    


        # TODO: add decorrelation step here?


        ########################
        ### Gibbs Update eta ###
        ########################
        eta[m] = sample_eta(eta[m-1], ntheta, sum(setup.ns2[i] for i in range(setup.nexp)), setup.eta_prior_shape, setup.eta_prior_rate)
        
        
        #######################
        ### Tempering Swaps ###
        #######################
        if m > setup.start_temper and setup.ntemps > 1:
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
                        + setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[0]]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                        + (mvnorm_logpdf_(theta[m,sw.T[0]], theta0[m, sw.T[0]],
                                Sigma0_inv_curr[sw.T[0]], Sigma0_ldet_curr[sw.T[0]]) * theta_ext[sw.T[0]]).sum(axis = 1)
                        #- 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[0]]
                        #- 0.5 * sse_curr[i][sw.T[0]].sum(axis = 1)
                        + llik_curr_theta[i][sw.T[0]].sum(axis=1)
                        # for t_1
                        - setup.s2_prior_kern[i](np.exp(log_s2[i][m][sw.T[1]]), setup.ig_a[i], setup.ig_b[i]).sum(axis=1)
                        - (mvnorm_logpdf_(theta[m,sw.T[1]], theta0[m,sw.T[1]],
                                Sigma0_inv_curr[sw.T[1]], Sigma0_ldet_curr[sw.T[1]]) * theta_ext[sw.T[1]]).sum(axis = 1)
                        #+ 0.5 * (setup.ny_s2[i] * np.log(s2[i][m])).sum(axis = 1)[sw.T[1]]
                        #+ 0.5 * sse_curr[i][sw.T[1]].sum(axis = 1)
                        - llik_curr_theta[i][sw.T[1]].sum(axis=1)
                        )


                    if False:
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
                        log_s2[i][m,tt[0]], log_s2[i][m,tt[1]] = log_s2[i][m,tt[1]].copy(), log_s2[i][m,tt[0]].copy()
                        pred_curr_theta[i][tt[0]], pred_curr_theta[i][tt[1]]       = \
                                            pred_curr_theta[i][tt[1]].copy(), pred_curr_theta[i][tt[0]].copy()
                        llik_curr_theta[i][tt[0]], llik_curr_theta[i][tt[1]]         = \
                                            llik_curr_theta[i][tt[1]].copy(), llik_curr_theta[i][tt[0]].copy()
                        pred_curr_delta[i][tt[0]], pred_curr_delta[i][tt[1]]       = \
                                            pred_curr_delta[i][tt[1]].copy(), pred_curr_delta[i][tt[0]].copy()
                        llik_curr_delta[i][tt[0]], llik_curr_delta[i][tt[1]]         = \
                                            llik_curr_delta[i][tt[1]].copy(), llik_curr_delta[i][tt[0]].copy()
                    theta_ext[tt[0]], theta_ext[tt[1]] = theta_ext[tt[1]].copy(), theta_ext[tt[0]].copy()
        print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')
    #------------------------------------------------------------------------------------------
    pred_curr = pred_curr_theta
    t1 = time.time()
    print('\rCalibration MCMC Complete. Time: {:f} seconds.'.format(t1 - t0))
    count_temper = count_temper + count_temper.T - np.diag(np.diag(count_temper))
    out = OutCalibClust(theta, theta_hist, log_s2, count, count_temper, 
                            pred_curr, theta0, Sigma0, delta, eta, setup.nclustmax, cov_theta_cand)
    return(out)

# EOF
