####################################
####################################
"""Impala hierarchical calibration"""
####################################
####################################

###############
### Imports ###
###############

import time
import warnings
from collections import namedtuple

import numpy as np
from numpy.random import uniform

from .impala_copula import (
    CopulaHyper,
    draw_gamma0,
    hier_logpdf,
    hyper_from_legacy,
    implied_theta_cov,
    latent_start_var,
)
from .impala_noprobit_emu import (
    cf_bounds,
    chol_sample_1per,
    cov_4d_pcm,
    initfunc_unif,
    tran_unif,
)
from .impala_pool import AMcov_pool
from .pbar import pbar

np.seterr(under="ignore")


OutCalibHier = namedtuple(
    "OutCalibHier",
    "theta s2 count count_s2 count_decor2 cov_theta_cand cov_ls2_cand count_temper pred_curr theta0 Sigma0 v gamma0 rho copula",  # llik theta_native theta0_native theta_parent_native',
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

    ###################################
    ### Copula hierarchy parameters ###
    ###################################
    # The hierarchy lives on a latent Gaussian scale and is mapped onto the
    # box by a copula, so theta_i has the box as its exact support and needs
    # no truncation.  See impala_copula for the model.  The legacy knobs
    # (theta0_prior_mean/cov, Sigma0_prior_df/scale) are translated onto the
    # copula's (a, b, rho) parameterisation so existing run scripts keep
    # working; set margin_a/margin_b or rho_prior_a/rho_prior_b on the setup
    # to bypass the translation.
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
        # rho -- how the marginal variance splits between the parent and the
        # between-experiment spread -- is left uniform rather than centred on
        # the legacy split (_rho_center).  The old model learned Sigma0 freely,
        # so pinning rho from the legacy numbers would add information it never
        # had, and an extreme centre turns any concentration into a sharp
        # prior.  Set rho_prior_a/rho_prior_b, e.g. from
        # impala_copula.rho_prior_from_center, to encode a pooling belief.
    if rho_prior_a is None or rho_prior_b is None:
        rho_prior_a, rho_prior_b = 1.0, 1.0
    copula = CopulaHyper(
        setup.p,
        a=margin_a,
        b=margin_b,
        rho_a=rho_prior_a,
        rho_b=rho_prior_b,
    )
    # The bounds are the copula's support, so they never need rejection.  A
    # custom constraint_func carves a region out of the box and does: the model
    # keeps 1{theta0 in C} for the parent, as the truncated hierarchy did.
    cf_setup = None if setup._constraint_func is cf_bounds else setup

    gamma0 = np.zeros([setup.nmcmc, setup.ntemps, setup.p])
    rho = np.zeros([setup.nmcmc, setup.ntemps, setup.p])
    log_s2 = [
        np.zeros([setup.nmcmc, setup.ntemps, setup.ns2[i]]) + 0.0
        for i in range(setup.nexp)
    ]
    for i in range(setup.nexp):
        log_s2[i][0] = np.log(setup.sd_est[i] ** 2)
    # v holds the latent (unbounded) parameters that the sampler moves; theta
    # is the derived box-valued parameter, kept for the emulator and for every
    # downstream consumer of the output.
    v = [
        np.zeros([setup.nmcmc, setup.ntemps, setup.ntheta[i], setup.p]) + 0.0
        for i in range(setup.nexp)
    ]
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

    # The copula's support is the box, so the bounds never need rejection
    # sampling.  A custom constraint_func still does: it carves a region out of
    # the box (PTW's orderings, for instance), and the starting values have to
    # sit inside it or the first model evaluation fails.
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
    gamma0[0] = copula.from_theta(theta0_start)
    rho[0] = copula.rho_a / (copula.rho_a + copula.rho_b)  # prior mean

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
        # Draw the latent start from the hierarchy, redrawing any theta that a
        # custom constraint_func rejects.  The bounds themselves cannot be
        # violated, so with constraint_func="bounds" this accepts immediately.
        nrow = setup.ntemps * setup.ntheta[i]
        g_rep = np.repeat(gamma0[0], setup.ntheta[i], axis=0)
        sd_rep = np.repeat(np.sqrt(1.0 - rho[0]), setup.ntheta[i], axis=0)
        v_flat = g_rep + sd_rep * np.random.normal(size=(nrow, setup.p))
        good = setup.checkConstraints(
            tran_unif(
                copula.to_theta(v_flat),
                setup.bounds_mat,
                setup.bounds.keys(),
            )
        )
        maxiter = 10000
        j = 0
        while np.any(np.logical_not(good)):
            if j >= maxiter:
                raise ValueError(
                    f"Failed to find samples that fulfill the constraints after {maxiter} iterations."
                )
            bad = np.where(np.logical_not(good))[0]
            v_flat[bad] = g_rep[bad] + sd_rep[bad] * np.random.normal(
                size=(bad.size, setup.p)
            )
            good[bad] = setup.checkConstraints(
                tran_unif(
                    copula.to_theta(v_flat[bad]),
                    setup.bounds_mat,
                    setup.bounds.keys(),
                )
            )
            j += 1
        v[i][0] = v_flat.reshape(v[i][0].shape)
        theta[i][0] = copula.to_theta(v[i][0])
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
        # the adaptive proposal now lives on the latent scale
        start_var=latent_start_var(setup.start_var_theta),
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
            v[i][m] = v[i][m - 1].copy()
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
        # The adaptive proposal moves the latent v; theta is the copula image
        # of v, so every candidate is inside the box automatically and only a
        # custom constraint_func can reject one.
        cov_theta_cand.update(v, m)

        v_cand = cov_theta_cand.gen_cand(v, m)
        theta_cand = [copula.to_theta(v_cand[i]) for i in range(setup.nexp)]

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
                + hier_logpdf(v_cand[i], gamma0[m - 1], rho[m - 1])[
                    good_values[i]
                ]
                - hier_logpdf(v[i][m - 1], gamma0[m - 1], rho[m - 1])[
                    good_values[i]
                ]
            )
            # MCMC Accept
            accept[i][:] = np.log(uniform(size=alpha[i].shape)) < alpha[i]
            # Where accept, make changes
            v[i][m][accept[i]] = v_cand[i][accept[i]].copy()
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
        ### Gibbs update gamma0 ###
        ###########################
        # Conjugate normal, independent across coordinates: the latent parent
        # replaces theta0 and needs no rejection sampling.
        v_curr = [v[i][m] for i in range(setup.nexp)]
        gamma0[m][:] = draw_gamma0(
            v_curr,
            rho[m - 1],
            setup.itl,
            copula=copula,
            setup=cf_setup,
        )

        ########################
        ### Gibbs update rho ###
        ########################
        # Exact one-dimensional draw on a grid; replaces the inverse-Wishart
        # draw of Sigma0.  rho_k is the pooling strength in coordinate k.
        rho[m][:] = copula.draw_rho(v_curr, gamma0[m], setup.itl)

        ################################
        ### Joint Decorrelation Step ###
        ################################
        if m % setup.decor == 0:
            for k in range(setup.p):
                # The shared shift is applied on the latent scale, where it
                # leaves the hierarchy term invariant exactly as before.
                z = np.random.normal() * 0.1
                gamma0_cand = gamma0[m].copy()
                gamma0_cand[:, k] += z
                if cf_setup is None:
                    good_values_theta0 = np.ones(setup.ntemps, dtype=bool)
                else:
                    good_values_theta0 = setup.checkConstraints(
                        tran_unif(
                            copula.to_theta(gamma0_cand),
                            setup.bounds_mat,
                            setup.bounds.keys(),
                        )
                    )
                for i in range(setup.nexp):
                    # Find new candidate values for theta
                    v_cand[i][:] = v[i][m].copy()
                    v_cand[i][:, :, k] += z
                    theta_cand[i][:] = copula.to_theta(v_cand[i])
                    theta_eval_mat[i][:] = theta[i][m].reshape(
                        setup.ntheta[i] * setup.ntemps, setup.p
                    )
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
                        +hier_logpdf(v_cand[i], gamma0_cand, rho[m])[
                            good_values[i]
                        ]
                        - hier_logpdf(v[i][m], gamma0[m], rho[m])[
                            good_values[i]
                        ]
                    )
                # now sum over alpha (for each temperature), add alpha for the
                # parent prior, accept or reject.  Log acceptance ratio for each
                # temperature: sum over experiments and over the theta_i within
                # each experiment, plus the gamma0 prior (counted once, and not
                # tempered, consistent with the Gibbs update of gamma0).
                alpha_tot = (
                    sum(a.sum(axis=1) for a in alpha)
                    - 0.5 * (gamma0_cand**2 / rho[m]).sum(axis=1)
                    + 0.5 * (gamma0[m] ** 2 / rho[m]).sum(axis=1)
                )

                accept_tot = np.log(uniform(size=setup.ntemps)) < alpha_tot
                # Where accept, make changes
                gamma0[m][accept_tot, :] = gamma0_cand[accept_tot, :]
                for i in range(setup.nexp):
                    v[i][m][accept_tot] = v_cand[i][accept_tot].copy()
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
                # The hyperpriors p(gamma0) p(rho) are not tempered (the Gibbs
                # updates of gamma0 and rho use them at full strength in every
                # chain), so they cancel in the swap ratio and must not appear here.
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
                        + hier_logpdf(
                            v[i][m][sw.T[0]],
                            gamma0[m, sw.T[0]],
                            rho[m, sw.T[0]],
                        ).sum(axis=1)
                        + llik_curr[i][sw.T[0]].sum(axis=1)
                        # for t_1
                        - setup.s2_prior_kern[i](
                            np.exp(log_s2[i][m][sw.T[1]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        - hier_logpdf(
                            v[i][m][sw.T[1]],
                            gamma0[m, sw.T[1]],
                            rho[m, sw.T[1]],
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
                        v[i][m, tt[0]], v[i][m, tt[1]] = (
                            v[i][m, tt[1]].copy(),
                            v[i][m, tt[0]].copy(),
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
                    gamma0[m, tt[0]], gamma0[m, tt[1]] = (
                        gamma0[m, tt[1]].copy(),
                        gamma0[m, tt[0]].copy(),
                    )
                    rho[m, tt[0]], rho[m, tt[1]] = (
                        rho[m, tt[1]].copy(),
                        rho[m, tt[0]].copy(),
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
    # theta0 is the copula image of the latent parent: the median of a new
    # experiment's theta, still a point in the unit box, so it can be pushed
    # through the emulator exactly as before.  Sigma0 is no longer a sampled
    # parameter -- it is reported as the implied theta-scale spread of a new
    # experiment's theta, which is what its diagonal was being read as.  To
    # draw a parent theta use copula.parent_draw(gamma0, rho).
    theta0 = copula.to_theta(gamma0)
    Sigma0 = implied_theta_cov(copula, gamma0, rho)
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
        v,
        gamma0,
        rho,
        copula,
    )  # , llik, theta_native, theta0_native, theta_parent_native)
    return out
