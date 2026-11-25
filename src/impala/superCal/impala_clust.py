####################################
"""Impala clustered calibration"""

####################################
import time
import warnings
from collections import namedtuple
from math import sqrt

import numpy as np
from numpy.random import beta, choice, gamma, uniform

from .impala_copula import (
    CopulaHyper,
    draw_gamma0,
    hier_logpdf,
    hyper_from_legacy,
    implied_theta_cov,
)
from .impala_noprobit_emu import (
    cf_bounds,
    chol_sample_1per,
    cov_4d_pcm,
    initfunc_unif,
    s2_swap_kern,
    tran_unif,
)

# import pbar
# np.seterr(under='ignore')

# no probit tranform for hierarchical and DP versions


##############################################################################################################################################################################
def bincount2D_vectorized(a, max_count):
    """
    Applies np.bincount across a 2d array (row-wise).

    Adapted From: https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    a_offs = a + np.arange(a.shape[0])[:, None] * max_count
    return np.bincount(
        a_offs.ravel(), minlength=a.shape[0] * max_count
    ).reshape(-1, max_count)


def sample_delta_per_temperature(
    curr_delta_t, njs_t, log_posts_t, inv_temp_t, eta_t
):
    """Given log-posteriors, current delta --
    iterates through delta assignments, reassigning probabilistically."""
    djs = (njs_t == 0) * eta_t / (sum(njs_t == 0) + 1e-9)
    temp = np.empty(log_posts_t.shape[0])
    unis = uniform(size=curr_delta_t.shape[0])
    for s in range(curr_delta_t.shape[0]):
        njs_t[curr_delta_t[s]] -= 1
        temp[:] = -np.inf
        temp[(njs_t + djs) > 0] = (  # log-posterior under each cluster
            inv_temp_t
            * (
                np.log((njs_t + djs)[njs_t + djs > 0])
                + log_posts_t.T[s, njs_t + djs > 0]
            )
        )
        temp[:] -= temp.max()
        # temp[:] = np.exp(temp) / np.exp(temp).sum()
        temp[:] = np.cumsum(np.exp(temp))
        temp[:] /= temp[
            -1
        ]  # normalized cumulative sum of probability of cluster choice
        curr_delta_t[s] = (unis[s] >= temp).sum()
        # curr_delta_t[s] = choice(log_posts_t.shape[0], p = temp)
        njs_t[curr_delta_t[s]] += 1
        djs[njs_t > 0] = 0.0
    return curr_delta_t


def sample_delta_per_temperature_wrapper(args):
    return sample_delta_per_temperature(*args)


def sample_delta(curr_delta, njs, log_posts, inv_temps, eta, pool=None):
    args = zip(curr_delta, njs, log_posts, inv_temps, eta)
    if pool is None:
        delta = np.array(list(map(sample_delta_per_temperature_wrapper, args)))
    else:
        delta = np.array(
            list(pool.map(sample_delta_per_temperature_wrapper, args))
        )
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
    nC = nSj + n  # ntemps
    mC = (nSj.reshape(-1, 1) * mSj + n * mus) / nC.reshape(-1, 1)  # ntemps x p
    Sj[:] = (
        1
        / nC.reshape(-1, 1, 1)
        * (
            +nSj.reshape(-1, 1, 1) * Sj
            + n * covs
            + np.einsum("t,tp,tq->tpq", nSj, mSj - mC, mSj - mC)
            + n * np.einsum("tp,tq->tpq", mus - mC, mus - mC)
        )
    )
    mSj[:] = mC
    nSj[:] = nC
    return Sj, mSj, nSj


# Escobar & West 1995 (JASA 90, 577-588); cf. West 1992, dirichletprocess R package
def sample_eta(curr_eta, nclust, ndat, prior_shape, prior_rate):
    g = beta(curr_eta + 1, ndat)
    aa = prior_shape + nclust
    bb = prior_rate - np.log(g)
    # Escobar & West (1995), eq. 13: mixture of Gamma(a + k, b - log g) with
    # probability eps and Gamma(a + k - 1, b - log g) with probability 1 - eps,
    # where eps / (1 - eps) = (a + k - 1) / (n (b - log g)).
    eps = (aa - 1) / (ndat * bb + aa - 1)
    sel = (uniform(size=aa.shape) >= eps) * 1  # 0 -> a + k, 1 -> a + k - 1
    aaa = np.vstack((aa, aa - 1)).T[np.arange(aa.shape[0]), sel]
    return gamma(shape=aaa, scale=1 / bb)


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

    def __init__(
        self,
        nexp,
        ntheta,
        ntemps,
        p,
        nclustmax,
        start_var=1e-4,
        start_adapt_iter=300,
        tau_start=0.0,
    ):
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = (
            tau_start * np.ones(ntemps)
        )  # note this is different from hier--not sure how to have a different tau for each experiment
        self.S = np.empty((ntemps, nclustmax, p, p))  # proposal covariance
        for t in range(ntemps):
            for c in range(nclustmax):
                self.S[t, c] = np.eye(p) * start_var
        self.cov_hist = [
            np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)
        ]  # covariance of past sample by experiment
        self.mu_hist = [
            np.empty((ntemps, ntheta[i], p)) for i in range(nexp)
        ]  # mean of past samples by experiment
        self.cov_clust = np.empty((
            ntemps,
            nclustmax,
            p,
            p,
        ))  # covariance of past samples by cluster (weighted combination of cov_hist)
        self.mu_clust = np.zeros((
            ntemps,
            nclustmax,
            p,
        ))  # mean of past sample by cluster (weighted combination of mu_hist)
        self.n_hc = np.zeros((
            ntemps,
            nclustmax,
        ))  # count for accumulation purposes
        self.nexp = nexp
        self.ntheta = ntheta
        self.ntemps = ntemps
        self.nclustmax = nclustmax
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = np.zeros(self.ntemps, dtype=int)
        self.nS = np.zeros((ntemps, nclustmax))
        self.temps = np.arange(ntemps)

    def update(self, x, m, curr_delta):
        if m > self.start_adapt_iter:
            # update experiment-specific mu and cov
            for i in range(self.nexp):
                self.mu_hist[i] += (x[i][m - 1] - self.mu_hist[i]) / m
                self.cov_hist[i][:] = +((m - 1) / m) * self.cov_hist[i] + (
                    (m - 1) / (m * m)
                ) * np.einsum(
                    "tej,tel->tejl",
                    x[i][m - 1] - self.mu_hist[i],
                    x[i][m - 1] - self.mu_hist[i],
                )
            # reweight experiment-specific to get cluster-specific mu and cov
            self.cluster_covariance_update(m, curr_delta)
            # make history covariance into proposal covariance
            self.S = self.AM_SCALAR * np.einsum(
                "tmjl,t->tmjl",
                self.cov_clust + np.eye(self.p) * self.eps,
                np.exp(self.tau),
            )  # is identity be added correctly (correctly broadcast)?

        elif m == self.start_adapt_iter:
            # calculate experiment-specific mu and cov
            for i in range(self.nexp):
                self.mu_hist[i][:] = x[i][:m].mean(axis=0)
                self.cov_hist[i][:] = cov_4d_pcm(x[i][:m], self.mu_hist[i])

            # reweight experiment-specific to get cluster-specific mu and cov
            self.cluster_covariance_update(m, curr_delta)
            # make history covariance into proposal covariance
            self.S = self.AM_SCALAR * np.einsum(
                "tmjl,t->tmjl",
                self.cov_clust + np.eye(self.p) * self.eps,
                np.exp(self.tau),
            )  # is identity be added correctly (correctly broadcast)?

    def cluster_covariance_update(self, n, delta):
        # https://www.mathworks.com/help/matlab/import_export/using-mapreduce-to-compute-covariance-and-related-quantities.html
        self.cov_clust[:] = (
            0.0  # accumulating covariance                                                                                   #(c1)
        )
        self.mu_clust[:] = (
            0.0  # accumulating mean                                                                                           #(m1)
        )
        self.n_hc[:] = (
            0.0  # accumulating count                                                                                              #(n1)
        )
        mu_new = np.empty((self.ntemps, self.cov_clust.shape[-1]))  # (m)
        n_new = np.zeros((self.ntemps, 1))  # (n)
        for i in range(self.nexp):  # loop over unvectorized experiments
            for j in range(
                delta[i].shape[1]
            ):  # loop over vectorized experiments
                n_new[:] = (
                    self.n_hc[self.temps, delta[i].T[j], None] + n
                )  # (n=n1+n2)
                mu_new[:] = (
                    self.n_hc[self.temps, delta[i].T[j], None]
                    * self.mu_clust[self.temps, delta[i].T[j]]
                    + n * self.mu_hist[i][self.temps, j]
                ) / n_new  # (m=(n1*m1 + n2*m2)/n)
                self.cov_clust[self.temps, delta[i].T[j]] = (
                    1
                    / n_new[:, :, None]
                    * (  # 1/n *
                        +self.n_hc[self.temps, delta[i].T[j], None, None]
                        * self.cov_clust[
                            self.temps, delta[i].T[j]
                        ]  # cov1 (what has accumulated so far)  #(c1)
                        + n
                        * self.cov_hist[i][
                            self.temps, j
                        ]  # cov2 (history for experiment ij)          #(c2)
                        + np.einsum(
                            "t,tp,tq->tpq",
                            self.n_hc[self.temps, delta[i].T[j]],  # mu1
                            self.mu_clust[self.temps, delta[i].T[j]] - mu_new,
                            self.mu_clust[self.temps, delta[i].T[j]] - mu_new,
                        )
                        + n
                        * np.einsum(
                            "tp,tq->tpq",
                            self.mu_hist[i][self.temps, j] - mu_new,
                            self.mu_hist[i][self.temps, j] - mu_new,
                        )  # mu2
                    )
                )
                self.mu_clust[self.temps, delta[i].T[j]] = mu_new  # (m1=m)
                self.n_hc[self.temps, delta[i].T[j]] = n_new.ravel()  # (n1=n)

    def update_tau(
        self, m
    ):  # label switching makes it difficult to track acceptance rate, so this is turned off
        # diminishing adaptation based on acceptance rate for each temperature
        if False:
            if (m % 100 == 0) and (m > 300):
                ddelta = min(0.1, 5 / sqrt(m + 1))
                self.tau[np.where(self.count_100 < 23)] = (
                    self.tau[np.where(self.count_100 < 23)] - ddelta
                )
                self.tau[np.where(self.count_100 > 23)] = (
                    self.tau[np.where(self.count_100 > 23)] + ddelta
                )
                self.count_100 *= 0

    def gen_cand(self, x, m):  # one sample per cluster
        # x_cand = [chol_sample_1per(x[i][m-1], self.S) for i in range(self.nexp)]
        x_cand = chol_sample_1per(x[m - 1], self.S)
        return x_cand


OutCalibClust = namedtuple(
    "OutCalibClust",
    "theta theta_hist s2 count count_temper pred_curr theta0 Sigma0 delta eta nclustmax theta_am v v_hist gamma0 rho copula",
)

## DP Cluster Calibration


### The following is an updated version of calibClust with expanded capabilities
### Some key changes include:
### - allowing experiment weights
### - allowing custom initializations
### - changing initial theta0, Sigma0, and eta defaults
### - estimation of separate s2 values within an experiment
### - passing indices to loglik
### - add truncated gibbs sampling for measurement errors
### - speed up clustering algorithm
def calibClust(setup):
    """
    Clustered calibration with expanded capabilities, still undergoing testing.
    Some changes include:, allowing weights, allowing custom initializations, changing initial theta0 defaults,
    estimation of separate s2 values within an experiment, adding truncated gibbs sampling for measurement errors
    """
    t0 = time.time()

    ## Constants Declaration

    # boolean with dimension [i] x [ylens[i], ns2[i]] indicating which s2 to use for each output
    s2_ind_mat = [
        (setup.s2_ind[i][:, None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
    ]

    # boolean with dimension [i] x [ylens[i], ntheta[i]] indicating which theta to use for each output
    theta_ind_mat = [
        (setup.theta_ind[i][:, None] == range(setup.ntheta[i]))
        for i in range(setup.nexp)
    ]

    # sequence of length ntemps
    temps = np.arange(setup.ntemps)

    # matrix of repeated inverse temperatures of dimension [ntemps, nclustmax]
    itl_mat_theta = (np.ones((setup.ntemps, setup.nclustmax)).T * setup.itl).T

    # repeated inverse temperatures of dimension [i] x [ntemps, ns2[i]]
    itl_mat_s2 = [
        np.ones((setup.ntemps, setup.ns2[i])) * setup.itl.reshape(-1, 1)
        for i in range(setup.nexp)
    ]

    ###################################
    ### Copula hierarchy parameters ###
    ###################################
    # The hierarchy over cluster-level theta lives on a latent Gaussian scale
    # and is mapped onto the box by a copula, so theta has the box as its exact
    # support and needs no truncation.  See impala_copula for the model and for
    # the translation of the legacy hyperprior knobs.
    margin_a = getattr(setup, "margin_a", None)
    margin_b = getattr(setup, "margin_b", None)
    rho_prior_a = getattr(setup, "rho_prior_a", None)
    rho_prior_b = getattr(setup, "rho_prior_b", None)
    if (margin_a is None or margin_b is None) and (
        setup.theta0_prior_mean is not None
    ):
        margin_a, margin_b, _rho_center, clamped = hyper_from_legacy(
            setup.theta0_prior_mean,
            setup.theta0_prior_cov,
            setup.Sigma0_prior_df,
            setup.Sigma0_prior_scale,
        )
        if np.any(clamped):
            which = np.where(clamped)[0].tolist()
            warnings.warn(
                "The hierarchical prior requested more marginal spread than a "
                "Beta with a, b >= 1 can give for its mean, in coordinate(s) "
                f"{which}; the margin was capped at the flattest such Beta "
                "(uniform where the mean is 0.5). Going further would pile "
                "prior mass at both bounds.",
                stacklevel=2,
            )
    if rho_prior_a is None or rho_prior_b is None:
        rho_prior_a, rho_prior_b = 1.0, 1.0
    copula = CopulaHyper(
        setup.p,
        a=margin_a,
        b=margin_b,
        rho_a=rho_prior_a,
        rho_b=rho_prior_b,
    )
    # The bounds are the copula's support and never need rejection.  A custom
    # constraint_func carves a region out of the box and does.
    cf_setup = None if setup._constraint_func is cf_bounds else setup
    # the hierarchy is not tempered in calibClust; see the gamma0 update
    itl_untempered = np.ones(setup.ntemps)

    gamma0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    rho = np.empty([setup.nmcmc, setup.ntemps, setup.p])

    ## Parameter Declaration
    ### Added ability to specify initial value
    if setup.theta0_start is not None:
        theta0_start = np.broadcast_to(
            np.asarray(setup.theta0_start, dtype=float),
            (setup.ntemps, setup.p),
        ).copy()
    else:
        theta0_start = initfunc_unif(size=[setup.ntemps, setup.p])
        good = setup.checkConstraints(
            tran_unif(theta0_start, setup.bounds_mat, setup.bounds.keys())
        )
        maxiter = 10000
        for j in range(maxiter):
            if np.all(good):
                break
            theta0_start[np.where(np.logical_not(good))] = initfunc_unif(
                size=[(np.logical_not(good)).sum(), setup.p]
            )
            good[np.where(np.logical_not(good))] = setup.checkConstraints(
                tran_unif(
                    theta0_start[np.where(np.logical_not(good))],
                    setup.bounds_mat,
                    setup.bounds.keys(),
                )
            )
        if j + 1 == maxiter:
            raise ValueError(
                f"Failed to find samples that fulfill the constraints after {maxiter} iterations."
            )
    gamma0[0] = copula.from_theta(theta0_start)
    rho[0] = copula.rho_a / (copula.rho_a + copula.rho_b)  # prior mean

    s2_which_mat = [
        [np.where(s2_ind_mat[i][:, j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
    ]
    theta_which_mat = [
        [np.where(theta_ind_mat[i][:, j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
    ]

    ### Add weights
    wt_mat = [None] * setup.nexp
    for i in range(setup.nexp):
        wt_mat[i] = setup.wt[i]
        if (
            np.any(
                np.asarray([
                    len(np.unique(wt_mat[i][s2_which_mat[i][j]]))
                    for j in range(len(s2_which_mat[i]))
                ])
            )
            != 1
        ) and ("gibbs" in setup.models[i].s2):
            setup.models[i].s2 = "fix"
            print(
                "Gibbs sampling for s2 only valid if weights are the same for all observations with same s2. Reverting to fixed s2. "
            )

    # initialize delta, the cluster membership indicator (for each experiment)
    delta = [
        np.empty([setup.nmcmc, setup.ntemps, setup.ntheta[i]], dtype=int)
        for i in range(setup.nexp)
    ]
    for i in range(setup.nexp):
        delta[i][0] = choice(
            setup.nclustmax, size=(setup.ntemps, setup.ntheta[i])
        )
    delta_ind_mat = [
        (delta[i][0, :, :, None] == range(setup.nclustmax))
        for i in range(setup.nexp)
    ]
    clust_mem_count = np.zeros(
        (setup.ntemps, setup.nclustmax), dtype=int
    )  # (extant) cluster weight for ijth experiment
    clust_nomem_bool = np.zeros(clust_mem_count.shape, dtype=bool)
    djs = np.empty(
        clust_mem_count.shape
    )  # (candidate) cluster weight for ijth experiment
    for i in range(setup.nexp):
        clust_mem_count[:] += bincount2D_vectorized(
            delta[i][0], setup.nclustmax
        )
    curr_delta = [delta[i][0].copy() for i in range(setup.nexp)]
    theta_unravel = [
        np.repeat(np.arange(setup.ntemps), setup.ntheta[i])
        for i in range(setup.nexp)
    ]
    # initialize Theta
    theta_long_shape = (setup.ntemps * setup.nclustmax, setup.p)
    theta_wide_shape = (setup.ntemps, setup.nclustmax, setup.p)
    # v holds the latent (unbounded) cluster parameters that the sampler moves;
    # theta is the derived box-valued parameter, kept for the emulator and for
    # every downstream consumer of the output.
    v = np.empty((setup.nmcmc, setup.ntemps, setup.nclustmax, setup.p))
    theta = np.empty((setup.nmcmc, setup.ntemps, setup.nclustmax, setup.p))

    ### Added ability to specify initial value
    if setup.theta_start is None:
        # Draw from the hierarchy, redrawing any theta a custom
        # constraint_func rejects.  The bounds cannot be violated, so with
        # constraint_func="bounds" this accepts immediately.
        v[0] = copula.parent_draw(
            gamma0[0],
            rho[0],
            n=setup.nclustmax,
            setup=cf_setup,
            latent=True,
        )
    else:
        v[0] = copula.from_theta(setup.theta_start)
    theta[0] = copula.to_theta(v[0])

    v_hist = [
        np.empty((setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p))
        for i in range(setup.nexp)
    ]
    theta_hist = [
        np.empty((setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p))
        for i in range(setup.nexp)
    ]
    theta_cand = np.empty(theta_wide_shape)
    v_cand_empty = np.empty(theta_wide_shape)
    for i in range(setup.nexp):
        v_hist[i][0] = v[0, theta_unravel[i], delta[i][0].ravel()].reshape(
            setup.ntemps, setup.ntheta[i], setup.p
        )
        theta_hist[i][0] = theta[
            0, theta_unravel[i], delta[i][0].ravel()
        ].reshape(
            setup.ntemps, setup.ntheta[i], setup.p
        )  ### edited ns2 -> ntheta
    theta_eval = np.empty(theta_wide_shape)
    v_eval = np.empty(theta_wide_shape)
    theta_ext = np.zeros(
        (setup.ntemps, setup.nclustmax), dtype=bool
    )  # array of (current) extant theta locs
    for i in range(setup.nexp):
        theta_ext[theta_unravel[i], delta[i][0].ravel()] += True
    ntheta = theta_ext.sum(axis=1)  # count extant thetas
    # initialize sigma2 and eta
    log_s2 = [
        np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]])
        for i in range(setup.nexp)
    ]
    for i in range(setup.nexp):
        log_s2[i][0] = np.log(setup.sd_est[i] ** 2)

    eta = np.empty((setup.nmcmc, setup.ntemps))
    eta[0] = (
        setup.eta_prior_shape / setup.eta_prior_rate
    )  # set to prior expectation

    ## Initialize *Current* Variables
    pred_curr_delta = [None] * setup.nexp  # [i] x [ntemps, nclustmax, ylens[i]]
    pred_cand_delta = [None] * setup.nexp  # [i] x [ntemps, nclustmax, ylens[i]]
    llik_curr_delta = [
        None
    ] * setup.nexp  # [i] x [ntemps, nclustmax, ntheta[i]]
    pred_cand_theta = [None] * setup.nexp  # [i] x [ntemps, ylens[i]]
    pred_curr_theta = [None] * setup.nexp  # [i] x [ntemps, ylens[i]]
    llik_cand_theta = [None] * setup.nexp  # [i] x [ntemps, ntheta[i]]
    llik_curr_theta = [None] * setup.nexp  # [i] x [ntemps, ntheta[i]]
    llik_cand_theta_ = np.zeros((
        setup.ntemps,
        setup.nclustmax,
    ))  # above, Summarized by cluster assignment (i.e., llik for each unique theta, summed over experiments that share that theta)
    llik_curr_theta_ = np.zeros((setup.ntemps, setup.nclustmax))
    marg_lik_cov_curr = (
        [None] * setup.nexp
    )  # does not need to get summed over experiments (should be one for each experiment, regardless of cluster)

    for i in range(setup.nexp):
        ### Initialize predictions for theta_i's
        pred_curr_theta[i] = setup.models[i].eval(
            tran_unif(
                theta_hist[i][0].reshape(-1, setup.p),  # 0 for first iteration
                setup.bounds_mat,
                setup.bounds.keys(),
            ),
            False,  # do not pool
        )
        pred_cand_theta[i] = pred_curr_theta[i].copy()
        marg_lik_cov_curr[i] = [None] * setup.ntemps
        llik_curr_theta[i] = np.empty([setup.ntemps, setup.ntheta[i]])
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = [None] * setup.ntheta[i]
            s2_stretched = log_s2[i][0][t, setup.s2_ind[i]]
            for j in range(setup.ntheta[i]):
                marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(
                    np.exp(s2_stretched[theta_which_mat[i][j]]),
                    wt_mat[i][theta_which_mat[i][j]],
                    s2_which_mat[i][j],
                )
                llik_curr_theta[i][t][j] = setup.models[i].llik(
                    setup.ys[i][theta_which_mat[i][j]],
                    pred_curr_theta[i][t][theta_which_mat[i][j]],
                    marg_lik_cov_curr[i][t][j],
                    wt_mat[i][theta_which_mat[i][j]],
                )
        llik_cand_theta[i] = llik_curr_theta[i].copy()

        ### Initialize predictions for clusters
        pred_curr_delta[i] = (
            setup
            .models[i]
            .eval(
                tran_unif(
                    theta[0].reshape(-1, setup.p),
                    setup.bounds_mat,
                    setup.bounds.keys(),
                ),
                True,
            )
            .reshape(setup.ntemps, setup.nclustmax, setup.y_lens[i])
        )  # this is now smaller because it uses more efficient pool = True code
        pred_cand_delta[i] = pred_curr_delta[i].copy()
        llik_curr_delta[i] = np.empty([
            setup.ntemps,
            setup.nclustmax,
            setup.ntheta[i],
        ])
        for t in range(setup.ntemps):
            s2_stretched = log_s2[i][0][t, setup.s2_ind[i]]
            for j in range(setup.ntheta[i]):
                for k in range(setup.nclustmax):
                    llik_curr_delta[i][t][k][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]],
                        pred_curr_delta[i][t][k][theta_which_mat[i][j]],
                        marg_lik_cov_curr[i][t][j],
                        wt_mat[i][theta_which_mat[i][j]],
                    )

    ## Initialize Adaptive Metropolis related Variables
    cov_theta_cand = AMcov_clust(
        nexp=setup.nexp,
        ntheta=np.array([setup.ntheta[i] for i in range(setup.nexp)]),
        ntemps=setup.ntemps,
        p=setup.p,
        nclustmax=setup.nclustmax,
        start_var=setup.start_var_theta,
        start_adapt_iter=setup.start_adapt_iter,
        tau_start=setup.start_tau_theta,
    )

    ## The copula hierarchy has no theta0/Sigma0 scaffolding: gamma0 is drawn
    ## from a conjugate normal and rho from an exact one-dimensional grid draw.
    ## Initialize Counters
    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = np.zeros(setup.ntemps)
    ## Initialize Alphas
    alpha_wide_shape = (
        setup.ntemps,
        setup.nclustmax,
    )
    alpha = np.ones(alpha_wide_shape) * -np.inf
    accept = np.zeros(alpha_wide_shape, dtype=bool)
    sw_alpha = np.zeros(setup.nswap_per)
    good_values = np.zeros(alpha_wide_shape, dtype=bool)

    cluster_cum_prob = np.empty((setup.ntemps, setup.nclustmax))
    cluster_sample_unif = [
        np.empty((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)
    ]

    ## start MCMC
    for m in range(1, setup.nmcmc):
        ##############################
        ### Prep for New Iteration ###
        ##############################

        for i in range(setup.nexp):
            delta[i][m] = delta[i][m - 1].copy()

        #   Establish initial cluster weighting
        clust_mem_count[:] = 0.0
        for l in range(setup.nexp):
            clust_mem_count[:] += bincount2D_vectorized(
                delta[l][m], setup.nclustmax
            )
        clust_nomem_bool[:] = clust_mem_count == 0

        ##################
        ### Draw Delta ###
        ##################
        for i in range(setup.nexp):
            cluster_sample_unif[i][:] = uniform(
                size=(setup.ntemps, setup.ntheta[i])
            )
            for j in range(setup.ntheta[i]):
                # weighting assigned to extant clusters for ij'th within-vectorized experiment
                clust_mem_count[temps, delta[i][m, :, j]] -= 1
                # weighting assigned to candidate (non-extant) clusters
                djs[:] = clust_nomem_bool * (
                    eta[m - 1] / (clust_nomem_bool.sum(axis=1) + 1e-9)
                ).reshape(-1, 1)
                # unnormalized log-probability of cluster membership
                with np.errstate(divide="ignore", invalid="ignore"):
                    cluster_cum_prob[:] = (
                        +np.log(
                            clust_mem_count + djs
                        )  # this line doesnt match paper? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        + llik_curr_delta[i][:, :, j] * setup.itl.reshape(-1, 1)
                    )
                # ---fix for numerical stability in np.exp
                cluster_cum_prob -= cluster_cum_prob.max(axis=1).reshape(-1, 1)
                # Un-normalized cumulative probability of cluster membership
                cluster_cum_prob[:] = np.exp(cluster_cum_prob).cumsum(axis=1)
                # Normalized cumulative probability of cluster membership
                cluster_cum_prob /= cluster_cum_prob[:, -1].reshape(-1, 1)
                delta[i][m, :, j] = (
                    cluster_sample_unif[i][:, j].reshape(-1, 1)
                    > cluster_cum_prob
                ).sum(axis=1)
                # Fix weights using new cluster assignments for ij'th within-vectorized experiment
                clust_mem_count[temps, delta[i][m, :, j]] += 1
                # If a candidate cluster became extant--remove candidate flag.
                clust_nomem_bool[clust_mem_count > 0] = False

            delta_ind_mat[i][:] = delta[i][m, :, :, None] == range(
                setup.nclustmax
            )
            curr_delta[i][:] = delta[i][m]

        theta_ext[:] = False
        for i in range(setup.nexp):
            theta_ext[theta_unravel[i], delta[i][m].ravel()] += True
        ntheta[:] = theta_ext.sum(axis=1)

        ##########################
        ### Update Theta Evals ### (pred_curr_delta is up to date with theta[m-1])
        ##########################
        for i in range(setup.nexp):
            theta_hist[i][m] = theta[
                m - 1, theta_unravel[i], delta[i][m].ravel()
            ].reshape(
                setup.ntemps, setup.ntheta[i], setup.p
            )  # added theta_hist update 9/19/25
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    pred_curr_theta[i][t][theta_which_mat[i][j]] = (
                        pred_curr_delta[i][
                            t, delta[i][m][t][j], theta_which_mat[i][j]
                        ].copy()
                    )  # use updated delta[i][m]
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    llik_curr_theta[i][t][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]],
                        pred_curr_theta[i][t][theta_which_mat[i][j]],
                        marg_lik_cov_curr[i][t][j],
                        wt_mat[i][theta_which_mat[i][j]],
                    )

        ########################
        ###  Gibbs update s2 ###
        ########################
        for i in range(setup.nexp):
            if setup.models[i].s2 == "gibbs":
                dev_sq = (pred_curr_theta[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
                    i
                ]  # (ntemps x ns2[i])
                for t in range(setup.ntemps):
                    log_s2[i][m][t] = np.log(
                        1
                        / np.random.gamma(
                            (
                                itl_mat_s2[i][t]
                                * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1)
                                - 1
                            ).astype("float64"),
                            (
                                1
                                / (
                                    itl_mat_s2[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ).astype("float64"),
                        ).astype("float64")
                    )
                    s2_stretched = log_s2[i][m][t, setup.s2_ind[i]]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_curr[i][t][j] = setup.models[
                            i
                        ].lik_cov_inv(
                            np.exp(s2_stretched[theta_which_mat[i][j]]),
                            wt_mat[i][theta_which_mat[i][j]],
                            s2_which_mat[i][j],
                        )
                        llik_curr_theta[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr_theta[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

            elif setup.models[i].s2 == "gibbs_trunc":
                dev_sq = (pred_curr_theta[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
                    i
                ]  # (ntemps x ns2[i])
                for t in range(setup.ntemps):
                    log_s2[i][m][t] = np.log(
                        1
                        / np.random.gamma(
                            (
                                itl_mat_s2[i][t]
                                * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1)
                                - 1
                            ).astype("float64"),
                            (
                                1
                                / (
                                    itl_mat_s2[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ).astype("float64"),
                        ).astype("float64")
                    )
                    s2_is_valid = (
                        log_s2[i][m][t] >= np.log(setup.sd_lower[i] ** 2)
                    ) * (log_s2[i][m][t] <= np.log(setup.sd_upper[i] ** 2))
                    ct = 0
                    while np.any(~s2_is_valid):
                        sub = np.where(~s2_is_valid)
                        log_s2[i][m][t][sub] = np.log(
                            1
                            / np.random.gamma(
                                (
                                    itl_mat_s2[i][t][sub]
                                    * (
                                        setup.ny_s2[i][sub] / 2
                                        + setup.ig_a[i][sub]
                                        + 1
                                    )
                                    - 1
                                ).astype("float64"),
                                (
                                    1
                                    / (
                                        itl_mat_s2[i][t][sub]
                                        * (
                                            setup.ig_b[i][sub]
                                            + dev_sq[t].flatten()[sub] / 2
                                        )
                                    )
                                ).astype("float64"),
                            ).astype("float64")
                        )
                        s2_is_valid = (
                            log_s2[i][m][t] >= np.log(setup.sd_lower[i] ** 2)
                        ) * (log_s2[i][m][t] <= np.log(setup.sd_upper[i] ** 2))
                        ct = ct + 1
                        if ct >= 50:
                            log_s2[i][m][t][
                                log_s2[i][m][t] < np.log(setup.sd_lower[i] ** 2)
                            ] = np.log(setup.sd_lower[i] ** 2)[
                                log_s2[i][m][t] < np.log(setup.sd_lower[i] ** 2)
                            ]
                            log_s2[i][m][t][
                                log_s2[i][m][t] > np.log(setup.sd_upper[i] ** 2)
                            ] = np.log(setup.sd_upper[i] ** 2)[
                                log_s2[i][m][t] > np.log(setup.sd_upper[i] ** 2)
                            ]
                            s2_is_valid = (
                                log_s2[i][m][t]
                                >= np.log(setup.sd_lower[i] ** 2)
                            ) * (
                                log_s2[i][m][t]
                                <= np.log(setup.sd_upper[i] ** 2)
                            )

                    s2_stretched = log_s2[i][m][t, setup.s2_ind[i]]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_curr[i][t][j] = setup.models[
                            i
                        ].lik_cov_inv(
                            np.exp(s2_stretched[theta_which_mat[i][j]]),
                            wt_mat[i][theta_which_mat[i][j]],
                            s2_which_mat[i][j],
                        )
                        llik_curr_theta[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr_theta[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

            elif setup.models[i].s2 == "fix":
                log_s2[i][m] = np.log(setup.sd_est[i] ** 2)

                for t in range(setup.ntemps):
                    # same indexing as the Gibbs branches: log_s2 has one entry
                    # per s2 group (s2_ind), the likelihood for theta j uses the
                    # observations of theta j (theta_which_mat)
                    s2_stretched = log_s2[i][m][t, setup.s2_ind[i]]
                    for j in range(setup.ntheta[i]):
                        marg_lik_cov_curr[i][t][j] = setup.models[
                            i
                        ].lik_cov_inv(
                            np.exp(s2_stretched[theta_which_mat[i][j]]),
                            wt_mat[i][theta_which_mat[i][j]],
                            s2_which_mat[i][j],
                        )
                        llik_curr_theta[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr_theta[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

            else:
                print("TODO: fill in s2 sampling without gibbs")

        ###########################
        ### Gibbs update gamma0 ###
        ###########################
        # Only occupied clusters belong to the hierarchy, and which ones are
        # occupied differs by temperature, so the draws are masked by
        # theta_ext.  Must use m-1: theta has not been updated this iteration.
        #
        # Only the likelihood is tempered (power posterior): the hierarchy
        # enters at full strength in every chain, as do the priors of gamma0,
        # rho, eta and delta.  Tempering the hierarchy here would have no
        # consistent stationary distribution, because unoccupied clusters are
        # refreshed from that same density untempered and the delta update
        # moves centres between occupied and unoccupied without an N^(beta-1)
        # correction.  calibHier does temper its hierarchy, which is consistent
        # there because every theta_i is always occupied.
        v_curr = [v[m - 1]]
        mask_curr = [theta_ext]
        gamma0[m][:] = draw_gamma0(
            v_curr,
            rho[m - 1],
            itl_untempered,
            copula=copula,
            setup=cf_setup,
            mask_list=mask_curr,
        )

        ########################
        ### Gibbs update rho ###
        ########################
        rho[m][:] = copula.draw_rho(
            v_curr, gamma0[m], itl_untempered, mask_list=mask_curr
        )

        ###################
        ### Draw Thetas ###
        ###################

        llik_cand_theta_[:] = 0.0
        llik_curr_theta_[:] = 0.0
        v[m] = v[m - 1].copy()
        theta[m] = theta[m - 1].copy()
        v_eval[:] = v[m - 1].copy()
        theta_eval[:] = theta[m - 1].copy()
        pred_cand_theta = pred_curr_theta.copy()

        # the adaptive proposal now lives on the latent scale
        cov_theta_cand.update(v_hist, m, curr_delta)
        ### This can sometimes help if clusters have very uneven sizes and large clusters don't accept enough.
        if setup.scale_proposal:
            cov_theta_cand.S = (
                cov_theta_cand.S
                / np.maximum(clust_mem_count, np.ones(clust_mem_count.shape))[
                    :, :, None, None
                ]
            )  # gives better performance for heavily-occupied clusters, 10/5/26

        # Random-walk on the latent scale; every candidate is inside the box
        # by construction, so only a custom constraint_func can reject one.
        v_cand = np.stack([
            chol_sample_1per(
                v[m][t][None, ...], cov_theta_cand.S[t][None, ...]
            )[0]
            for t in range(setup.ntemps)
        ])
        theta_cand = copula.to_theta(v_cand)
        good_values[:] = setup.checkConstraints(
            tran_unif(
                theta_cand.reshape(theta_long_shape),
                setup.bounds_mat,
                setup.bounds.keys(),
            ),
            setup.bounds,
        ).reshape(setup.ntemps, setup.nclustmax)
        v_eval[good_values] = v_cand[good_values]
        theta_eval[good_values] = theta_cand[good_values]

        # Unoccupied clusters are refreshed straight from the hierarchy.
        v_cand_empty[:] = copula.parent_draw(
            gamma0[m],
            rho[m],
            n=setup.nclustmax,
            setup=cf_setup,
            latent=True,
        )
        v_eval[~theta_ext] = v_cand_empty[~theta_ext].copy()
        theta_eval[~theta_ext] = copula.to_theta(v_eval[~theta_ext])
        # Automatically update for non-occupied clusters, don't want to impact
        # accept/reject of other thetas later on
        v[m, ~theta_ext] = v_eval[~theta_ext].copy()
        theta[m, ~theta_ext] = theta_eval[~theta_ext].copy()

        ### Generate candidate theta evals for all clusters. This often turns out to be faster than separate by experiment
        for i in range(setup.nexp):
            pred_cand_delta[i][:] = (
                setup
                .models[i]
                .eval(
                    tran_unif(
                        theta_eval.reshape(-1, setup.p),
                        setup.bounds_mat,
                        setup.bounds.keys(),
                    ),
                    pool=True,
                )
                .reshape(setup.ntemps, setup.nclustmax, setup.y_lens[i])
            )
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    pred_cand_theta[i][t][theta_which_mat[i][j]] = (
                        pred_cand_delta[i][
                            t, delta[i][m][t][j], theta_which_mat[i][j]
                        ].copy()
                    )
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    llik_cand_theta[i][t][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]],
                        pred_cand_theta[i][t][theta_which_mat[i][j]],
                        marg_lik_cov_curr[i][t][j],
                        wt_mat[i][theta_which_mat[i][j]],
                    )
                    llik_cand_theta_[t][curr_delta[i][t, j]] += llik_cand_theta[
                        i
                    ][t][
                        j
                    ]  # not impacted by changing non-occupied clusters' theta values
                    llik_curr_theta_[t][curr_delta[i][t, j]] += llik_curr_theta[
                        i
                    ][t][
                        j
                    ]  # not impacted by changing non-occupied clusters' theta values

        alpha[:] = -np.inf
        alpha[good_values] = (
            itl_mat_theta * (llik_cand_theta_ - llik_curr_theta_)
            + hier_logpdf(v_eval, gamma0[m], rho[m]) * theta_ext
            - hier_logpdf(v[m], gamma0[m], rho[m]) * theta_ext
        )[good_values]
        accept[:] = np.log(uniform(size=alpha.shape)) < alpha
        v[m, accept] = v_cand[accept]
        theta[m, accept] = theta_cand[accept]
        # Automatically update for non-occupied clusters
        v[m, ~theta_ext] = v_eval[~theta_ext].copy()
        theta[m, ~theta_ext] = theta_eval[~theta_ext].copy()

        for i in range(setup.nexp):
            v_hist[i][m] = v[m, theta_unravel[i], delta[i][m].ravel()].reshape(
                setup.ntemps, setup.ntheta[i], setup.p
            )
            theta_hist[i][m] = theta[
                m, theta_unravel[i], delta[i][m].ravel()
            ].reshape(setup.ntemps, setup.ntheta[i], setup.p)  # CHANGED
            pred_curr_delta[i][accept] = pred_cand_delta[i][accept].copy()
            pred_curr_delta[i][~theta_ext] = pred_cand_delta[i][
                ~theta_ext
            ].copy()  # Automatically update for non-occupied clusters
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    pred_curr_theta[i][t][theta_which_mat[i][j]] = (
                        pred_curr_delta[i][
                            t, delta[i][m][t][j], theta_which_mat[i][j]
                        ].copy()
                    )
                    llik_curr_theta[i][t][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]],
                        pred_curr_theta[i][t][theta_which_mat[i][j]],
                        marg_lik_cov_curr[i][t][j],
                        wt_mat[i][theta_which_mat[i][j]],
                    )
            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    for k in range(setup.nclustmax):
                        llik_curr_delta[i][t][k][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr_delta[i][t][k][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

        count += accept.sum(axis=1)
        cov_theta_cand.count_100 += accept.sum(axis=1)
        cov_theta_cand.update_tau(m)

        ########################
        ### Gibbs Update eta ###
        ########################
        eta[m] = sample_eta(
            eta[m - 1],
            ntheta,
            sum(setup.ntheta[i] for i in range(setup.nexp)),
            setup.eta_prior_shape,
            setup.eta_prior_rate,
        )

        #######################
        ### Tempering Swaps ###
        #######################
        if m > setup.start_temper and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(
                    setup.ntemps, 2 * setup.nswap_per, replace=False
                ).reshape(-1, 2)
                sw_alpha[:] = 0.0  # reset swap probability
                # Only the likelihood (and the s2 prior, which the s2 updates
                # temper) is tempered.  The clustering prior -- p(gamma0),
                # p(rho), p(eta), p(delta | eta) and the hierarchy of the
                # cluster centres -- is the same in every chain, so it cancels
                # here.
                for i in range(setup.nexp):
                    sw_alpha[:] = sw_alpha + (
                        setup.itl[sw.T[1]] - setup.itl[sw.T[0]]
                    ) * (
                        # for t_0
                        +s2_swap_kern(setup, i)(
                            np.exp(log_s2[i][m][sw.T[0]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        + llik_curr_theta[i][sw.T[0]].sum(axis=1)
                        # for t_1
                        - s2_swap_kern(setup, i)(
                            np.exp(log_s2[i][m][sw.T[1]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        - llik_curr_theta[i][sw.T[1]].sum(axis=1)
                    )

                # select the accepted pairs directly (previously
                # sw[np.where(sw[accepted])[0]], which indexes sw by row numbers
                # of the accepted subset and so swapped the wrong pairs)
                for tt in sw[np.log(uniform(size=setup.nswap_per)) < sw_alpha]:
                    count_temper[tt[0], tt[1]] = count_temper[tt[0], tt[1]] + 1
                    eta[m, tt[0]], eta[m, tt[1]] = (
                        eta[m, tt[1]].copy(),
                        eta[m, tt[0]].copy(),
                    )  # previously not tempering over eta correctly
                    theta[m, tt[0]], theta[m, tt[1]] = (
                        theta[m, tt[1]].copy(),
                        theta[m, tt[0]].copy(),
                    )
                    v[m, tt[0]], v[m, tt[1]] = (
                        v[m, tt[1]].copy(),
                        v[m, tt[0]].copy(),
                    )
                    gamma0[m, tt[0]], gamma0[m, tt[1]] = (
                        gamma0[m, tt[1]].copy(),
                        gamma0[m, tt[0]].copy(),
                    )
                    rho[m, tt[0]], rho[m, tt[1]] = (
                        rho[m, tt[1]].copy(),
                        rho[m, tt[0]].copy(),
                    )
                    for i in range(setup.nexp):
                        theta_hist[i][m, tt[0]], theta_hist[i][m, tt[1]] = (
                            theta_hist[i][m, tt[1]].copy(),
                            theta_hist[i][m, tt[0]].copy(),
                        )
                        v_hist[i][m, tt[0]], v_hist[i][m, tt[1]] = (
                            v_hist[i][m, tt[1]].copy(),
                            v_hist[i][m, tt[0]].copy(),
                        )
                        delta[i][m, tt[0]], delta[i][m, tt[1]] = (
                            delta[i][m, tt[1]].copy(),
                            delta[i][m, tt[0]].copy(),
                        )
                        log_s2[i][m, tt[0]], log_s2[i][m, tt[1]] = (
                            log_s2[i][m, tt[1]].copy(),
                            log_s2[i][m, tt[0]].copy(),
                        )
                        pred_curr_theta[i][tt[0]], pred_curr_theta[i][tt[1]] = (
                            pred_curr_theta[i][tt[1]].copy(),
                            pred_curr_theta[i][tt[0]].copy(),
                        )
                        llik_curr_theta[i][tt[0]], llik_curr_theta[i][tt[1]] = (
                            llik_curr_theta[i][tt[1]].copy(),
                            llik_curr_theta[i][tt[0]].copy(),
                        )
                        pred_curr_delta[i][tt[0]], pred_curr_delta[i][tt[1]] = (
                            pred_curr_delta[i][tt[1]].copy(),
                            pred_curr_delta[i][tt[0]].copy(),
                        )
                        llik_curr_delta[i][tt[0]], llik_curr_delta[i][tt[1]] = (
                            llik_curr_delta[i][tt[1]].copy(),
                            llik_curr_delta[i][tt[0]].copy(),
                        )
                    theta_ext[tt[0]], theta_ext[tt[1]] = (
                        theta_ext[tt[1]].copy(),
                        theta_ext[tt[0]].copy(),
                    )
        print(
            f"\rCalibration MCMC {m / setup.nmcmc:.01%} Complete",
            end="",
        )
    # ------------------------------------------------------------------------------------------
    pred_curr = pred_curr_theta
    t1 = time.time()
    print(f"\rCalibration MCMC Complete. Time: {t1 - t0:f} seconds.")
    count_temper = (
        count_temper + count_temper.T - np.diag(np.diag(count_temper))
    )
    s2 = [np.exp(log_s2[i]) for i in range(len(log_s2))]
    # theta0 is the copula image of the latent parent: the median of a new
    # cluster's theta, still a point in the unit box, so it can be pushed
    # through the emulator exactly as before.  Sigma0 is no longer a sampled
    # parameter -- it is reported as the implied theta-scale spread of a new
    # cluster's theta, which is what its diagonal was being read as.  To draw a
    # parent theta use copula.parent_draw(gamma0, rho).
    theta0 = copula.to_theta(gamma0)
    Sigma0 = implied_theta_cov(copula, gamma0, rho)
    out = OutCalibClust(
        theta,
        theta_hist,
        s2,  # this is s2, not log_s2
        count,
        count_temper,
        pred_curr,
        theta0,
        Sigma0,
        delta,
        eta,
        setup.nclustmax,
        cov_theta_cand,
        v,
        v_hist,
        gamma0,
        rho,
        copula,
    )
    return out


# EOF
