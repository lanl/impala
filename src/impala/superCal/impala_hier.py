####################################
####################################
"""Impala hierarchical calibration"""
####################################
####################################

###############
### Imports ###
###############

import time
from collections import namedtuple

import numpy as np
import scipy
from numpy.linalg import slogdet
from numpy.random import uniform
from scipy.stats import invwishart

from .impala_noprobit_emu import (
    chol_sample_1per,
    chol_sample_1per_constraints,
    chol_sample_nper_constraints,
    cov_4d_pcm,
    initfunc_unif,
    invwishart_logpdf,
    mvnorm_logpdf,
    mvnorm_logpdf_,
    tran_unif,
)
from .impala_pool import AMcov_pool
from .pbar import pbar

np.seterr(under="ignore")


OutCalibHier = namedtuple(
    "OutCalibHier",
    "theta s2 count count_s2 count_decor2 cov_theta_cand cov_ls2_cand count_temper pred_curr theta0 Sigma0",  # llik theta_native theta0_native theta_parent_native',
)


class AMcov_hier:
    """
    Stores and updates the covariance matrix for Adaptive Metropolis
    for a hierarchical calibration
    """

    def __init__(
        self,
        nexp,
        ntheta,
        ntemps,
        p,
        start_var=1e-4,
        start_adapt_iter=300,
        tau_start=0.0,
    ):  # ntheta is a vector of length nexp
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = [
            tau_start * np.ones((ntemps, ntheta[i])) for i in range(nexp)
        ]
        self.S = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        for i in range(nexp):
            self.S[i][:] = np.eye(p) * start_var
        self.cov = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        self.mu = [np.empty((ntemps, ntheta[i], p)) for i in range(nexp)]
        self.nexp = nexp
        self.ntemps = ntemps
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = [np.zeros((ntemps, ntheta[i])) for i in range(nexp)]

    def update(self, x, m):
        """
        updates the covariance of the previous MCMC samples;
        called in m-th iteration, so latest value is x[i][m-1]
        """
        if m > self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i] += (x[i][m - 1] - self.mu[i]) / m
                self.cov[i][:] = +((m - 1) / m) * self.cov[i] + (
                    (m - 1) / (m * m)
                ) * np.einsum(
                    "tej,tel->tejl",
                    x[i][m - 1] - self.mu[i],
                    x[i][m - 1] - self.mu[i],
                )
                self.S[i] = self.AM_SCALAR * np.einsum(
                    "tejl,te->tejl",
                    self.cov[i] + np.eye(self.p) * self.eps,
                    np.exp(self.tau[i]),
                )

        elif m == self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i][:] = x[i][:m].mean(axis=0)
                # self.mu[i][:]  = x[i].mean(axis = 0)
                self.cov[i][:] = cov_4d_pcm(x[i][:m], self.mu[i])
                self.S[i][:] = self.AM_SCALAR * np.einsum(
                    "tejl,te->tejl",
                    self.cov[i] + np.eye(self.p) * self.eps,
                    np.exp(self.tau[i]),
                )

    def update_tau(self, m):
        """
        diminishing adaptation based on acceptance rate for each temperature
        """
        if (m % 100 == 0) and (m > self.start_adapt_iter):
            delta = min(0.5, 5 / np.sqrt(m + 1))
            for i in range(self.nexp):
                self.tau[i][self.count_100[i] < 23] -= delta
                self.tau[i][self.count_100[i] > 23] += delta
                self.count_100[i] *= 0

    def gen_cand(self, x, m):
        """generate a candidate"""
        x_cand = [
            chol_sample_1per(x[i][m - 1], self.S[i]) for i in range(self.nexp)
        ]
        return x_cand


# @profile
def calibHier(setup):
    """
    Hierarchical calibration with expanded capabilities, still undergoing testing
    Some changes include:, allowing weights, allowing custom initializations, changing initial theta0 defaults,
    estimation of separate s2 values within an experiment, adding truncated gibbs sampling for measurement errors
    """
    t0 = time.time()
    theta0 = np.zeros([setup.nmcmc, setup.ntemps, setup.p])
    theta0 += 0.0
    Sigma0 = np.zeros([setup.nmcmc, setup.ntemps, setup.p, setup.p])
    Sigma0 += 0.0
    ntheta = np.sum(setup.ntheta)
    log_s2 = [
        np.zeros([setup.nmcmc, setup.ntemps, setup.ns2[i]]) + 0.0
        for i in range(setup.nexp)
    ]
    for i in range(setup.nexp):
        log_s2[i][0] = np.log(setup.sd_est[i] ** 2)
    theta = [
        np.zeros([setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p]) + 0.0
        for i in range(setup.nexp)
    ]
    theta_ind_mat = [
        (setup.theta_ind[i][:, None] == range(setup.ntheta[i]))
        for i in range(setup.nexp)
    ]
    s2_ind_mat = [
        (setup.s2_ind[i][:, None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
    ]
    s2_which_mat = [
        [np.where(s2_ind_mat[i][:, j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
    ]
    theta_which_mat = [
        [np.where(theta_ind_mat[i][:, j])[0] for j in range(setup.ntheta[i])]
        for i in range(setup.nexp)
    ]

    theta0 = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    if setup.theta0_start is not None:
        theta0_start = setup.theta0_start
    else:
        theta0_start = initfunc_unif(size=[setup.ntemps, setup.p])
        good = setup.checkConstraints(
            tran_unif(theta0_start, setup.bounds_mat, setup.bounds.keys())
        )
        maxiter = 1000000
        j = 0
        while np.any(np.logical_not(good)):
            if j >= maxiter:
                raise ValueError(
                    f"Failed to find samples that fulfill the constraints after {maxiter} iterations."
                )
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
            j += 1
    theta0[0] = theta0_start
    Sigma0[0] = setup.Sigma0_prior_scale / (
        setup.Sigma0_prior_df - setup.p - 1
    )  # initialize at prior mean

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
        ) and (setup.models[i].s2 == "gibbs"):
            setup.models[i].s2 = "MH"
            print(
                "Gibbs sampling for s2 only valid if weights are the same for all observations with same s2. Reverting to MH. "
            )

    pred_curr = [None] * setup.nexp  # [i], ntemps x ylens[i]
    pred_cand = [None] * setup.nexp  # [i], ntemps x ylens[i]
    llik_curr = [None] * setup.nexp  # [i], ntheta[i] x ntemps
    llik_cand = [None] * setup.nexp  # [i], ntheta[i] x ntemps
    itl_mat = [  # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ntheta[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
    ]

    itl_mat_s2 = [
        np.ones((setup.ntemps, setup.ns2[i])) * setup.itl.reshape(-1, 1)
        for i in range(setup.nexp)
    ]

    marg_lik_cov_curr = [None] * setup.nexp

    for i in range(setup.nexp):
        theta[i][0] = chol_sample_nper_constraints(
            theta0[0],
            Sigma0[0],
            setup.ntheta[i],
            setup.checkConstraints,
            setup.bounds_mat,
            setup.bounds.keys(),
            setup.bounds,
            setup.constants,
        )
        pred_curr[i] = setup.models[i].eval(
            tran_unif(
                theta[i][0].reshape(setup.ntemps * setup.ntheta[i], setup.p),
                setup.bounds_mat,
                setup.bounds.keys(),
            ),
            pool=False,
        )
        pred_cand[i] = pred_curr[i].copy()

        marg_lik_cov_curr[i] = [None] * setup.ntemps
        llik_curr[i] = np.empty([setup.ntemps, setup.ntheta[i]])
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = [None] * setup.ntheta[i]
            s2_stretched = log_s2[i][0][t, setup.s2_ind[i]]
            for j in range(setup.ntheta[i]):
                marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(
                    np.exp(s2_stretched[theta_which_mat[i][j]]),
                    wt_mat[i][theta_which_mat[i][j]],
                    s2_which_mat[i][j],
                )
                llik_curr[i][t][j] = setup.models[i].llik(
                    setup.ys[i][theta_which_mat[i][j]],
                    pred_curr[i][t][theta_which_mat[i][j]],
                    marg_lik_cov_curr[i][t][j],
                    wt_mat[i][theta_which_mat[i][j]],
                )
                # this isnt getting nthetas correct, probably need to change models script...
                # there should be a separate likelihood evaluation (with separate covariance)
                # for every i, t, ntheta. In diagonal case, we could vectorize over t ntheta...
                # for now, break into separate calls.  Later may be worthwhile to try to vectorize more.
        llik_cand[i] = llik_curr[i].copy()

    cov_theta_cand = AMcov_hier(
        setup.nexp,
        np.array([setup.ntheta[i] for i in range(setup.nexp)]),
        setup.ntemps,
        setup.p,
        start_var=setup.start_var_theta,
        start_adapt_iter=setup.start_adapt_iter,
        tau_start=setup.start_tau_theta,
    )
    cov_ls2_cand = [
        AMcov_pool(
            setup.ntemps,
            setup.ns2[i],
            start_var=setup.start_var_ls2,
            start_adapt_iter=setup.start_adapt_iter,
            tau_start=setup.start_tau_ls2,
        )
        for i in range(setup.nexp)
    ]

    theta0_prior_mean = setup.theta0_prior_mean  # np.repeat(0.5, setup.p)
    theta0_prior_cov = setup.theta0_prior_cov  # np.eye(setup.p)*1**2
    theta0_prior_prec = scipy.linalg.inv(theta0_prior_cov)
    theta0_prior_ldet = slogdet(theta0_prior_cov)[1]

    tbar = np.empty(theta0[0].shape)
    mat = np.zeros((setup.ntemps, setup.p, setup.p))

    Sigma0_prior_df = setup.Sigma0_prior_df  # setup.p
    Sigma0_prior_scale = (
        setup.Sigma0_prior_scale
    )  # np.eye(setup.p)*1**2#/setup.p
    Sigma0_dfs = Sigma0_prior_df + ntheta * setup.itl

    Sigma0_ldet_curr = slogdet(Sigma0[0])[1]
    Sigma0_inv_curr = np.linalg.inv(Sigma0[0])

    count_temper = np.zeros([setup.ntemps, setup.ntemps])
    count = [
        np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)
    ]
    # count_decor = [
    #     np.zeros((setup.ntemps, setup.ntheta[i], setup.p))
    #     for i in range(setup.nexp)
    # ]
    count_decor2 = np.zeros((setup.ntemps, setup.p))
    # count_100 = [np.zeros((setup.ntemps, setup.ntheta[i])) for i in range(setup.nexp)]
    count_s2 = np.zeros([setup.nexp, setup.ntemps], dtype=int)

    theta_cand = [
        np.empty([setup.ntemps, setup.ntheta[i], setup.p])
        for i in range(setup.nexp)
    ]
    theta_cand_mat = [
        np.empty([setup.ntemps * setup.ntheta[i], setup.p])
        for i in range(setup.nexp)
    ]
    theta_eval_mat = [
        np.empty(theta_cand_mat[i].shape) for i in range(setup.nexp)
    ]

    alpha = [
        np.ones((setup.ntemps, setup.ntheta[i])) * -np.inf
        for i in range(setup.nexp)
    ]
    alpha_s2 = np.ones([setup.nexp, setup.ntemps]) * (-np.inf)
    accept = [np.zeros(alpha[i].shape, dtype=bool) for i in range(setup.nexp)]
    sw_alpha = np.zeros(setup.nswap_per)
    good_values = [
        np.zeros(alpha[i].shape, dtype=bool) for i in range(setup.nexp)
    ]
    good_values_mat = [
        good_values[i].reshape(setup.ntheta[i] * setup.ntemps)
        for i in range(setup.nexp)
    ]

    ## start MCMC
    for m in pbar(range(1, setup.nmcmc)):
        for i in range(setup.nexp):
            theta[i][m] = theta[i][
                m - 1
            ].copy()  # current set to previous, will change if accepted
            log_s2[i][m] = log_s2[i][m - 1].copy()
            setup.models[i].step()
            if setup.models[i].stochastic:  # update emulator
                pred_curr[i] = setup.models[i].eval(
                    tran_unif(
                        theta[i][m].reshape(
                            setup.ntemps * setup.ntheta[i], setup.p
                        ),
                        setup.bounds_mat,
                        setup.bounds.keys(),
                    ),
                    pool=False,
                )
                for t in range(setup.ntemps):
                    for j in range(setup.ntheta[i]):
                        llik_curr[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )
        # No discrepancy for now...update here if added later

        #####################
        ### Update Thetas ###
        #####################
        cov_theta_cand.update(theta, m)

        theta_cand = cov_theta_cand.gen_cand(theta, m)

        for i in range(setup.nexp):
            # Find new candidate values for theta
            theta_eval_mat[i][:] = (
                theta[i][m - 1]
                .reshape(setup.ntemps * setup.ntheta[i], setup.p)
                .copy()
            )
            theta_cand_mat[i][:] = theta_cand[i].reshape(
                setup.ntemps * setup.ntheta[i], setup.p
            )
            # Check constraints
            good_values_mat[i][:] = setup.checkConstraints(
                tran_unif(
                    theta_cand_mat[i], setup.bounds_mat, setup.bounds.keys()
                )
            )
            good_values[i][:] = good_values_mat[i].reshape(
                setup.ntemps, setup.ntheta[i]
            )
            # Generate Predictions at new Theta values
            theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][
                good_values_mat[i]
            ]
            pred_cand[i][:] = setup.models[i].eval(
                tran_unif(
                    theta_eval_mat[i], setup.bounds_mat, setup.bounds.keys()
                ),
                pool=False,
            )  # .reshape(setup.ntemps, setup.y_lens[i])

            for t in range(setup.ntemps):
                for j in range(setup.ntheta[i]):
                    llik_cand[i][t][j] = setup.models[i].llik(
                        setup.ys[i][theta_which_mat[i][j]],
                        pred_cand[i][t][theta_which_mat[i][j]],
                        marg_lik_cov_curr[i][t][j],
                        wt_mat[i][theta_which_mat[i][j]],
                    )

            # Calculate log-probability of MCMC accept
            alpha[i][:] = -np.inf
            alpha[i][good_values[i]] = itl_mat[i][good_values[i]] * (
                # - 0.5 * (sse_cand[i][good_values[i]] - sse_curr[i][good_values[i]])
                llik_cand[i][good_values[i]]
                - llik_curr[i][good_values[i]]
                + mvnorm_logpdf_(
                    theta_cand[i],
                    theta0[m - 1],
                    Sigma0_inv_curr,
                    Sigma0_ldet_curr,
                )[good_values[i]]
                - mvnorm_logpdf_(
                    theta[i][m - 1],
                    theta0[m - 1],
                    Sigma0_inv_curr,
                    Sigma0_ldet_curr,
                )[good_values[i]]
            )
            # MCMC Accept
            accept[i][:] = np.log(uniform(size=alpha[i].shape)) < alpha[i]
            # Where accept, make changes
            theta[i][m][accept[i]] = theta_cand[i][accept[i]].copy()

            for t in range(setup.ntemps):
                accept_t = np.where(accept[i][t])[0]
                if accept_t.shape[0] > 0:
                    ind = np.hstack([theta_which_mat[i][j] for j in accept_t])
                    pred_curr[i][t][ind] = pred_cand[i][t][ind].copy()
            llik_curr[i][accept[i]] = llik_cand[i][accept[i]].copy()
            count[i][accept[i]] += 1
            cov_theta_cand.count_100[i][accept[i]] += 1

        cov_theta_cand.update_tau(m)

        # if m>10000:
        #    print('help')

        #################
        ### Update s2 ###
        #################
        for i in range(setup.nexp):
            if setup.models[i].s2 == "gibbs":
                # ## gibbs update s2
                dev_sq = (pred_curr[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
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
                            ),
                            (
                                1
                                / (
                                    itl_mat_s2[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ),
                        )
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
                        llik_curr[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

            elif setup.models[i].s2 == "gibbs_trunc":
                # ## gibbs update s2
                dev_sq = (pred_curr[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
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
                            ),
                            (
                                1
                                / (
                                    itl_mat_s2[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ),
                        )
                    )
                    s2_is_valid = (
                        log_s2[i][m][t] >= np.log(setup.sd_lower[i] ** 2)
                    ) * (log_s2[i][m][t] <= np.log(setup.sd_upper[i] ** 2))

                    ct = 0
                    while np.any(np.logical_not(s2_is_valid)):
                        sub = np.where(np.logical_not(s2_is_valid))
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
                                ),
                                (
                                    1
                                    / (
                                        itl_mat_s2[i][t][sub]
                                        * (
                                            setup.ig_b[i][sub]
                                            + dev_sq[t].flatten()[sub] / 2
                                        )
                                    )
                                ),
                            )
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
                        llik_curr[i][t][j] = setup.models[i].llik(
                            setup.ys[i][theta_which_mat[i][j]],
                            pred_curr[i][t][theta_which_mat[i][j]],
                            marg_lik_cov_curr[i][t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )

            elif setup.models[i].s2 == "fix":
                log_s2[i][m] = np.log(setup.sd_est[i] ** 2)

            # for t in range(setup.ntemps):
            #    for j in range(setup.ntheta[i]):
            #        marg_lik_cov_curr[i][t][j] = setup.models[i].lik_cov_inv(np.exp(log_s2[i][m][t, setup.s2_ind[i]])[setup.s2_ind[i]==j])
            #        llik_curr[i][t][j] = setup.models[i].llik(setup.ys[i][setup.theta_ind[i]==j], pred_curr[i][t][setup.theta_ind[i]==j], marg_lik_cov_curr[i][t][j])

            else:  # this needs to be fixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                        marg_lik_cov_candi[t][j] = setup.models[i].lik_cov_inv(
                            np.exp(ls2_candi[t, setup.s2_ind[i]])[
                                setup.s2_ind[i] == j
                            ],
                            wt_mat[i][theta_which_mat[i][j]],
                            s2_which_mat[i][j],
                        )  # s2[i][0, t, setup.s2_ind[i]])
                        llik_candi[t][j] = setup.models[i].llik(
                            setup.ys[i][setup.theta_ind[i] == j],
                            pred_curr[i][t][setup.theta_ind[i] == j],
                            marg_lik_cov_candi[t][j],
                            wt_mat[i][theta_which_mat[i][j]],
                        )
                        # something wrong still, getting way too large of variance
                    # marg_lik_cov_candi[t] = setup.models[i].lik_cov_inv(np.exp(ls2_candi[t])[setup.s2_ind[i]])#s2[i][0, t, setup.s2_ind[i]])
                    # llik_candi[t] = setup.models[i].llik(setup.ys[i], pred_curr[i][t], marg_lik_cov_candi[t])

                llik_diffi = llik_candi - llik_curr[i]
                alpha_s2 = setup.itl * (llik_diffi)
                alpha_s2 += (
                    setup.itl
                    * setup.s2_prior_kern[i](
                        np.exp(ls2_candi), setup.ig_a[i], setup.ig_b[i]
                    ).sum(axis=1)
                )  # ldhc_kern(np.exp(ls2_cand[i])).sum(axis=1)#ldig_kern(np.exp(ls2_cand[i]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 += setup.itl * ls2_candi.sum(axis=1)
                alpha_s2 -= (
                    setup.itl
                    * setup.s2_prior_kern[i](
                        np.exp(log_s2[i][m - 1]), setup.ig_a[i], setup.ig_b[i]
                    ).sum(axis=1)
                )  # ldhc_kern(np.exp(log_s2[i][m-1])).sum(axis=1)#ldig_kern(np.exp(log_s2[i][m-1]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 -= setup.itl * log_s2[i][m - 1].sum(axis=1)

                runif = np.log(uniform(size=setup.ntemps))
                for t in np.where(runif < alpha_s2)[0]:
                    count_s2[i, t] += 1
                    llik_curr[i][t] = llik_candi[t].copy()
                    log_s2[i][m][t] = ls2_candi[t].copy()
                    marg_lik_cov_curr[i][t] = marg_lik_cov_candi[t].copy()
                    cov_ls2_cand[i].count_100[t] += 1

                cov_ls2_cand[i].update_tau(m)

        ###########################
        ### Gibbs update theta0 ###
        ###########################
        cc = np.linalg.inv(
            np.einsum("t,tpq->tpq", ntheta * setup.itl, Sigma0_inv_curr)
            + theta0_prior_prec,
        )
        tbar = 0.0
        for i in range(setup.nexp):
            tbar += theta[i][m].sum(axis=1)
        tbar /= ntheta
        dd = +np.einsum(
            "t,tl->tl",
            setup.itl,
            np.einsum("tlk,tk->tl", ntheta * Sigma0_inv_curr, tbar),
        ) + np.dot(theta0_prior_prec, theta0_prior_mean)
        theta0[m][:] = chol_sample_1per_constraints(
            np.einsum("tlk,tk->tl", cc, dd),
            cc,
            setup.checkConstraints,
            setup.bounds_mat,
            setup.bounds.keys(),
            setup.bounds,
            setup.constants,
        )

        ###########################
        ### Gibbs update Sigma0 ###
        ###########################
        mat *= 0.0
        for i in range(setup.nexp):
            mat += np.einsum(
                "tnp,tnq->tpq",
                theta[i][m] - theta0[m].reshape(setup.ntemps, 1, setup.p),
                theta[i][m] - theta0[m].reshape(setup.ntemps, 1, setup.p),
            )
        Sigma0_scales = Sigma0_prior_scale + np.einsum(
            "t,tml->tml", setup.itl, mat
        )
        for t in range(setup.ntemps):
            Sigma0[m, t] = invwishart.rvs(
                df=Sigma0_dfs[t], scale=Sigma0_scales[t]
            )
        Sigma0_ldet_curr[:] = np.linalg.slogdet(Sigma0[m])[1]
        Sigma0_inv_curr[:] = np.linalg.inv(Sigma0[m])

        ################################
        ### Joint Decorrelation Step ###
        ################################
        if m % setup.decor == 0:
            for k in range(setup.p):
                z = np.random.normal() * 0.1
                theta0_cand = theta0[m].copy()
                theta0_cand[:, k] += z
                good_values_theta0 = setup.checkConstraints(
                    tran_unif(
                        theta0_cand, setup.bounds_mat, setup.bounds.keys()
                    )
                )
                for i in range(setup.nexp):
                    # Find new candidate values for theta
                    theta_cand[i][:] = theta[i][m].copy()
                    theta_eval_mat[i][:] = theta[i][m].reshape(
                        setup.ntheta[i] * setup.ntemps, setup.p
                    )
                    theta_cand[i][:, :, k] += z
                    theta_cand_mat[i][:] = theta_cand[i].reshape(
                        setup.ntheta[i] * setup.ntemps, setup.p
                    )
                    # Compute constraint flags
                    good_values_mat[i][:] = setup.checkConstraints(
                        tran_unif(
                            theta_cand_mat[i],
                            setup.bounds_mat,
                            setup.bounds.keys(),
                        )
                    )
                    # Generate predictions at "good" candidate values
                    theta_eval_mat[i][good_values_mat[i]] = theta_cand_mat[i][
                        good_values_mat[i]
                    ]
                    good_values[i][:] = (
                        good_values_mat[i]
                        .reshape(setup.ntemps, setup.ntheta[i])
                        .T
                        * good_values_theta0
                    ).T
                    pred_cand[i][:] = setup.models[i].eval(
                        tran_unif(
                            theta_eval_mat[i],
                            setup.bounds_mat,
                            setup.bounds.keys(),
                        ),
                        pool=False,
                    )  # .reshape(setup.ntemps, setup.ntheta[i], setup.y_lens[i])
                    for t in range(setup.ntemps):
                        for j in range(setup.ntheta[i]):
                            llik_cand[i][t][j] = setup.models[i].llik(
                                setup.ys[i][theta_which_mat[i][j]],
                                pred_cand[i][t][theta_which_mat[i][j]],
                                marg_lik_cov_curr[i][t][j],
                                wt_mat[i][theta_which_mat[i][j]],
                            )

                    alpha[i][:] = -np.inf
                    alpha[i][good_values[i]] = itl_mat[i][good_values[i]] * (
                        llik_cand[i][good_values[i]]
                        - llik_curr[i][good_values[i]]
                    ) + itl_mat[i][good_values[i]] * (
                        +mvnorm_logpdf_(
                            theta_cand[i],
                            theta0_cand,
                            Sigma0_inv_curr,
                            Sigma0_ldet_curr,
                        )[good_values[i]]
                        - mvnorm_logpdf_(
                            theta[i][m],
                            theta0[m],
                            Sigma0_inv_curr,
                            Sigma0_ldet_curr,
                        )[good_values[i]]
                    )
                # now sum over alpha (for each temperature), add alpha for theta0 to prior, accept or reject
                alpha_tot = (
                    sum(alpha).T
                    - 0.5
                    * setup.itl
                    * np.diag(
                        (theta0_cand - theta0_prior_mean)
                        @ theta0_prior_prec
                        @ (theta0_cand - theta0_prior_mean).T
                    )
                    + 0.5
                    * setup.itl
                    * np.diag(
                        (theta0[m] - theta0_prior_mean)
                        @ theta0_prior_prec
                        @ (theta0[m] - theta0_prior_mean).T
                    )
                )

                accept_tot = np.log(uniform(size=setup.ntemps)) < alpha_tot.sum(
                    axis=0
                )
                # Where accept, make changes
                theta0[m][accept_tot, :] = theta0_cand[accept_tot, :]
                for i in range(setup.nexp):
                    theta[i][m][accept_tot] = theta_cand[i][accept_tot].copy()
                    pred_curr[i][accept_tot, :] = pred_cand[i][
                        accept_tot, :
                    ].copy()
                    llik_curr[i][accept_tot] = llik_cand[i][accept_tot].copy()

                count_decor2[accept_tot, k] = count_decor2[accept_tot, k] + 1

        #######################
        ### Tempering Swaps ###
        #######################
        if m > setup.start_temper and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(
                    setup.ntemps, 2 * setup.nswap_per, replace=False
                ).reshape(-1, 2)
                sw_alpha[:] = 0.0  # reset swap probability
                sw_alpha[:] = sw_alpha + (
                    setup.itl[sw.T[1]] - setup.itl[sw.T[0]]
                ) * (
                    +mvnorm_logpdf(
                        theta0[m][sw.T[0]],
                        theta0_prior_mean,
                        theta0_prior_prec,
                        theta0_prior_ldet,
                    )
                    - mvnorm_logpdf(
                        theta0[m][sw.T[1]],
                        theta0_prior_mean,
                        theta0_prior_prec,
                        theta0_prior_ldet,
                    )
                    + invwishart_logpdf(
                        Sigma0[m][sw.T[0]], Sigma0_prior_df, Sigma0_prior_scale
                    )
                    - invwishart_logpdf(
                        Sigma0[m][sw.T[1]], Sigma0_prior_df, Sigma0_prior_scale
                    )
                )
                for i in range(setup.nexp):
                    sw_alpha[:] = sw_alpha + (
                        setup.itl[sw.T[1]] - setup.itl[sw.T[0]]
                    ) * (
                        # for t_0
                        +setup.s2_prior_kern[i](
                            np.exp(log_s2[i][m][sw.T[0]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        + mvnorm_logpdf_(
                            theta[i][m][sw.T[0]],
                            theta0[m, sw.T[0]],
                            Sigma0_inv_curr[sw.T[0]],
                            Sigma0_ldet_curr[sw.T[0]],
                        ).sum(axis=1)
                        + llik_curr[i][sw.T[0]].sum(axis=1)
                        # for t_1
                        - setup.s2_prior_kern[i](
                            np.exp(log_s2[i][m][sw.T[1]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        - mvnorm_logpdf_(
                            theta[i][m][sw.T[1]],
                            theta0[m, sw.T[1]],
                            Sigma0_inv_curr[sw.T[1]],
                            Sigma0_ldet_curr[sw.T[1]],
                        ).sum(axis=1)
                        - llik_curr[i][sw.T[1]].sum(axis=1)
                    )
                for tt in sw[
                    np.where(np.log(uniform(size=setup.nswap_per)) < sw_alpha)
                ]:
                    count_temper[tt[0], tt[1]] = count_temper[tt[0], tt[1]] + 1
                    for i in range(setup.nexp):
                        theta[i][m, tt[0]], theta[i][m, tt[1]] = (
                            theta[i][m, tt[1]].copy(),
                            theta[i][m, tt[0]].copy(),
                        )
                        log_s2[i][m][tt[0]], log_s2[i][m][tt[1]] = (
                            log_s2[i][m][tt[1]].copy(),
                            log_s2[i][m][tt[0]].copy(),
                        )
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = (
                            pred_curr[i][tt[1]].copy(),
                            pred_curr[i][tt[0]].copy(),
                        )
                        llik_curr[i][tt[0]], llik_curr[i][tt[1]] = (
                            llik_curr[i][tt[1]].copy(),
                            llik_curr[i][tt[0]].copy(),
                        )
                    theta0[m, tt[0]], theta0[m, tt[1]] = (
                        theta0[m, tt[1]].copy(),
                        theta0[m, tt[0]].copy(),
                    )
                    Sigma0[m, tt[0]], Sigma0[m, tt[1]] = (
                        Sigma0[m, tt[1]].copy(),
                        Sigma0[m, tt[0]].copy(),
                    )
                    Sigma0_inv_curr[tt[0]], Sigma0_inv_curr[tt[1]] = (
                        Sigma0_inv_curr[tt[1]].copy(),
                        Sigma0_inv_curr[tt[0]].copy(),
                    )
                    Sigma0_ldet_curr[tt[0]], Sigma0_ldet_curr[tt[1]] = (
                        Sigma0_ldet_curr[tt[1]].copy(),
                        Sigma0_ldet_curr[tt[0]].copy(),
                    )
                # if np.exp(log_s2[i][m,0,0])>1:
                #    print('a')
        # print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    t1 = time.time()
    print(f"\rCalibration MCMC Complete. Time: {t1 - t0:f} seconds.")

    s2 = log_s2.copy()
    for i in range(setup.nexp):
        s2[i] = np.exp(log_s2[i])

    count_temper = (
        count_temper + count_temper.T - np.diag(np.diag(count_temper))
    )
    # theta_reshape = [np.swapaxes(t,1,2) for t in theta]
    out = OutCalibHier(
        theta,
        s2,
        count,
        count_s2,
        count_decor2,
        cov_theta_cand,
        cov_ls2_cand,
        count_temper,
        pred_curr,
        theta0,
        Sigma0,
    )  # , llik, theta_native, theta0_native, theta_parent_native)
    return out
