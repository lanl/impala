####################################
####################################
"""Copula hierarchy for hierarchical calibration.

``calibHier`` places its hierarchy on a latent Gaussian scale and maps it onto
the parameter box with a copula, rather than truncating a normal hierarchy to
the box.

    rho_k in (0,1)                        pooling strength, coordinate k
    gamma0_k | rho  ~ N(0, rho_k)         latent parent
    v_jk | gamma0   ~ N(gamma0_k, 1-rho_k) latent experiment deviation
    theta_jk        = F_k^-1(Phi(v_jk)),  F_k = Beta(a_k, b_k)

Var(v_jk) = rho_k + (1 - rho_k) = 1 identically, so theta_jk ~ Beta(a_k, b_k)
*exactly*, whatever rho is.  With a_k = b_k = 1 that is Uniform(0,1): a
parameter the data say nothing about comes back uniform, which is the property
a normal hierarchy truncated to the box cannot deliver.  rho_k -> 1 is full
pooling, rho_k -> 0 is none.

There is no normalising constant to evaluate, no truncation, and no rejection
sampling, so none of the contraction that the truncated hierarchy suffers.

The margins cost nothing in the sampler: F_k enters only the v -> theta map
inside the likelihood, and neither ``draw_gamma0`` nor ``draw_rho`` involves
it -- both are properties of the latent Gaussian layer alone.

Tempering convention, matching the rest of ``calibHier``:
  tempered by itl[t] : the likelihood and the hierarchy prod_j N(v_j; gamma0,
                       diag(1 - rho))
  untempered         : gamma0 ~ N(0, diag(rho)) and the Beta prior on rho,
                       which play the roles theta0's and Sigma0's priors played
Consequently Var(v_jk) = rho_k + (1 - rho_k) / itl[t], which is 1 only in the
cold chain, so the exact-margin guarantee is a cold-chain statement.  The hot
chains are deliberately over-dispersed, as they already were.
"""
####################################
####################################

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm, truncnorm

from .impala_noprobit_emu import tran_unif

__all__ = [
    "CopulaHyper",
    "from_theta",
    "hier_logpdf",
    "hyper_from_legacy",
    "parent_draw",
    "rho_prior_from_center",
    "to_theta",
]

# Phi'(0): converts a theta-scale proposal variance near mid-box to the latent
# scale, so start_var_theta keeps roughly its old meaning.
_DPHI0 = 0.3989422804014327


###############################
### Hyperprior translation ####
###############################
def hyper_from_legacy(
    theta0_prior_mean,
    theta0_prior_cov,
    Sigma0_prior_df,
    Sigma0_prior_scale,
):
    """Translate the truncated-normal hyperpriors onto ``(a, b, rho)``.

    The copula fixes the *marginal* distribution of each theta_jk, so the
    parent's uncertainty and the experiment-to-experiment spread are not
    separately specifiable -- they must sum to the marginal variance.  That is
    exactly what the four legacy knobs carry, so on the diagonal the
    translation is a bijection::

        total variance = Var(parent)   [theta0_prior_cov]
                       + E Sigma0      [Sigma0_prior_scale / (df - p - 1)]
        Beta(a_k, b_k) matched to (theta0_prior_mean_k, total variance)
        rho_k          = Var(parent) / total variance

    The spread is capped at the most diffuse Beta with ``a_k, b_k >= 1`` for the
    requested mean.  Past that cap the margin turns U-shaped, piling mass at
    both bounds -- the failure mode that sank the probit-scale hierarchy -- so a
    vague legacy prior maps to the flattest sane margin instead (exactly
    Uniform(0,1) when the mean is 0.5).

    Off-diagonals of ``theta0_prior_cov`` and ``Sigma0_prior_scale`` are
    dropped; this hierarchy has no cross-coordinate correlation.

    :param theta0_prior_mean: length-p array, strictly inside (0, 1).
    :param theta0_prior_cov: p by p prior covariance for theta0 (diagonal used).
    :param Sigma0_prior_df: prior degrees of freedom, must exceed p + 1.
    :param Sigma0_prior_scale: p by p prior scale for Sigma0 (diagonal used).
    :return: ``(a, b, rho_center, clamped)``, each of length p; ``clamped``
        flags the coordinates whose requested spread hit the cap.
    """
    m_raw = np.atleast_1d(np.asarray(theta0_prior_mean, dtype=float)).ravel()
    p = m_raw.size
    if np.any((m_raw <= 0.0) | (m_raw >= 1.0)):
        raise ValueError("theta0_prior_mean must lie strictly inside (0, 1)")

    denom = float(Sigma0_prior_df) - p - 1.0
    if denom <= 0.0:
        raise ValueError(
            "Sigma0_prior_df must exceed p + 1 for E(Sigma0) to be finite"
        )

    v_par = np.diag(np.atleast_2d(np.asarray(theta0_prior_cov, dtype=float)))
    v_bet = np.diag(np.atleast_2d(np.asarray(Sigma0_prior_scale, dtype=float)))
    v_par = np.broadcast_to(v_par, (p,)).astype(float)
    v_bet = np.broadcast_to(v_bet, (p,)).astype(float) / denom
    if np.any(v_par <= 0.0) or np.any(v_bet < 0.0):
        raise ValueError("prior variances must be positive")

    # Moments of the legacy theta0 prior *as the old sampler used it*: the
    # normal truncated to the box, not the raw (mean, cov).  This matters --
    # theta0_prior_cov = eye(p) * 10**2 is so diffuse that truncating it to the
    # box leaves something indistinguishable from uniform whatever its nominal
    # mean, and that is the prior the old sampler really had.  Matching the raw
    # mean instead would turn a deliberately vague prior into a sharp one.
    sd = np.sqrt(v_par)
    tn = truncnorm((0.0 - m_raw) / sd, (1.0 - m_raw) / sd, loc=m_raw, scale=sd)
    m = np.clip(tn.mean(), 1e-6, 1.0 - 1e-6)
    v_par_trunc = tn.var()
    v_tot = v_par_trunc + v_bet

    # Beta method of moments: Var = m (1 - m) / (kappa + 1), kappa = a + b
    kappa = m * (1.0 - m) / v_tot - 1.0
    kappa_min = np.maximum(1.0 / m, 1.0 / (1.0 - m))  # keeps a, b >= 1
    clamped = kappa < kappa_min
    kappa = np.maximum(kappa, kappa_min)
    return (
        m * kappa,
        (1.0 - m) * kappa,
        np.clip(v_par_trunc / v_tot, 1e-3, 1.0 - 1e-3),
        clamped,
    )


def rho_prior_from_center(center, concentration=2.0):
    """Beta shape parameters for rho with the given mean and ``a + b``."""
    center = np.asarray(center, dtype=float)
    return concentration * center, concentration * (1.0 - center)


###############################
### The margins ###############
###############################
def to_theta(v, a=None, b=None):
    """Map the latent ``v`` (any shape ending in p) onto the unit box."""
    u = norm.cdf(v)
    if a is None or b is None:
        return u
    if np.allclose(a, 1.0) and np.allclose(b, 1.0):
        return u
    return beta_dist.ppf(u, a, b)


def from_theta(theta, a=None, b=None):
    """Inverse of :func:`to_theta`, for initialising from theta values."""
    theta = np.clip(np.asarray(theta, dtype=float), 1e-12, 1.0 - 1e-12)
    if a is None or b is None or (np.allclose(a, 1.0) and np.allclose(b, 1.0)):
        u = theta
    else:
        u = beta_dist.cdf(theta, a, b)
    return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))


###############################
### The hierarchy #############
###############################
def hier_logpdf(v, gam0, rho):
    """log prod_k N(v[..., k]; gam0[..., k], 1 - rho[..., k]).

    Replaces ``mvnorm_logpdf_`` in the hierarchy term.

    :param v: (ntemps, ntheta, p)
    :param gam0: (ntemps, p)
    :param rho: (ntemps, p)
    :return: (ntemps, ntheta)
    """
    var = (1.0 - rho)[:, None, :]
    d = v - gam0[:, None, :]
    return -0.5 * (d * d / var + np.log(var) + 1.8378770664093453).sum(axis=-1)


def _latent_moments(v_list, gam0=None, mask_list=None):
    """Totals over the latents the hierarchy applies to.

    :param v_list: list of (ntemps, nj, p) arrays.
    :param mask_list: optional list of (ntemps, nj) boolean arrays selecting
        which latents count.  calibClust needs this: only occupied clusters
        belong to the hierarchy, and which ones are occupied differs by
        temperature and changes every iteration.
    :return: ``(count, sum, sum of squares)``.  ``count`` is a scalar without a
        mask and an (ntemps,) array with one.
    """
    tot = 0
    vsum = 0.0
    sse = 0.0
    for idx, v in enumerate(v_list):
        if mask_list is None:
            tot += v.shape[1]
            vsum = vsum + v.sum(axis=1)
            if gam0 is not None:
                d = v - gam0[:, None, :]
                sse = sse + (d * d).sum(axis=1)
        else:
            mask = mask_list[idx]
            w = mask[..., None]
            tot = tot + mask.sum(axis=1)
            vsum = vsum + (v * w).sum(axis=1)
            if gam0 is not None:
                d = (v - gam0[:, None, :]) * w
                sse = sse + (d * d).sum(axis=1)
    return tot, vsum, sse


def _count_for(n, shape):
    """Broadcast a scalar or per-temperature count to ``shape``."""
    return np.reshape(n, shape) if np.ndim(n) else n


def draw_gamma0(
    v_list, rho, itl, copula=None, setup=None, mask_list=None, maxiter=10000
):
    """Conjugate normal draw of gamma0, vectorised over temperatures.

    When ``setup`` carries a custom ``constraint_func``, the model includes
    ``1{theta0 in C}`` for the parent, exactly as the truncated hierarchy did
    (its theta0 draw rejected until the constraints held), and the draw is
    rejection sampled.  That indicator does not involve rho, so conditional on
    gamma0 it is a constant: rho's draw stays exact and no normalising constant
    appears anywhere.  This is a single factor, not one per experiment, so it
    carries none of the contraction the truncated hierarchy suffered.

    :param v_list: list over experiments of (ntemps, ntheta[i], p) latents.
    :param rho: (ntemps, p)
    :param itl: (ntemps,) inverse temperatures.
    :param copula: the :class:`CopulaHyper`; required when ``setup`` is given.
    :param setup: the CalibSetup, when its constraint region must be honoured.
    :param mask_list: see :func:`_latent_moments`; selects which latents the
        hierarchy applies to (calibClust's occupied clusters).
    :return: (ntemps, p)
    """
    n, vsum, _ = _latent_moments(v_list, mask_list=mask_list)
    w = itl[:, None] / (1.0 - rho)
    prec = 1.0 / rho + _count_for(n, (-1, 1)) * w
    mu = (w * vsum) / prec
    sd = 1.0 / np.sqrt(prec)
    out = mu + np.random.normal(size=mu.shape) * sd
    if setup is None:
        return out
    if copula is None:
        raise ValueError("copula is required when setup is given")

    def _ok(x):
        return setup.checkConstraints(
            tran_unif(copula.to_theta(x), setup.bounds_mat, setup.bounds.keys())
        )

    good = _ok(out)
    j = 0
    while np.any(np.logical_not(good)):
        if j >= maxiter:
            raise ValueError(
                "Failed to find samples that fulfill the constraints after "
                f"{maxiter} iterations."
            )
        bad = np.where(np.logical_not(good))[0]
        out[bad] = (
            mu[bad] + np.random.normal(size=(bad.size, mu.shape[1])) * (sd[bad])
        )
        good[bad] = _ok(out[bad])
        j += 1
    return out


class CopulaHyper:
    """Holds the copula hyperparameters and draws ``rho``.

    ``rho_k`` has a Beta(rho_a, rho_b) prior and its full conditional is a
    one-dimensional density evaluated exactly on a grid, so the draw needs no
    tuning and no accept/reject step.
    """

    def __init__(self, p, a=None, b=None, rho_a=1.0, rho_b=1.0, ngrid=512):
        self.p = p
        ones = np.ones(p)
        self.a = (
            ones
            if a is None
            else np.broadcast_to(np.asarray(a, dtype=float), (p,)).copy()
        )
        self.b = (
            ones
            if b is None
            else np.broadcast_to(np.asarray(b, dtype=float), (p,)).copy()
        )
        self.rho_a = np.broadcast_to(
            np.asarray(rho_a, dtype=float), (p,)
        ).copy()
        self.rho_b = np.broadcast_to(
            np.asarray(rho_b, dtype=float), (p,)
        ).copy()
        self.uniform_margins = bool(
            np.allclose(self.a, 1.0) and np.allclose(self.b, 1.0)
        )
        # grid on the interior of (0, 1), shaped (ngrid, 1, 1) for broadcasting
        self.grid = ((np.arange(ngrid) + 0.5) / ngrid)[:, None, None]
        self.ngrid = ngrid
        self._lg_prior = (self.rho_a - 1.0) * np.log(self.grid) + (
            self.rho_b - 1.0
        ) * np.log1p(-self.grid)

    def to_theta(self, v):
        return to_theta(v, self.a, self.b)

    def from_theta(self, theta):
        return from_theta(theta, self.a, self.b)

    def draw_rho(self, v_list, gam0, itl, mask_list=None):
        """Exact grid draw of rho, vectorised over temperatures.

        :param v_list: list over experiments of (ntemps, ntheta[i], p) latents.
        :param gam0: (ntemps, p)
        :param itl: (ntemps,)
        :param mask_list: see :func:`_latent_moments`.
        :return: (ntemps, p)
        """
        n, _, sse = _latent_moments(v_list, gam0, mask_list=mask_list)
        g = self.grid  # (ngrid, 1, 1)
        it = itl[None, :, None]
        nn = _count_for(n, (1, -1, 1))
        lp = (
            self._lg_prior
            - 0.5 * np.log(g)
            - 0.5 * (gam0[None, :, :] ** 2) / g
            + it
            * (-0.5 * nn * np.log1p(-g) - 0.5 * sse[None, :, :] / (1.0 - g))
        )
        lp -= lp.max(axis=0, keepdims=True)
        w = np.exp(lp)
        w /= w.sum(axis=0, keepdims=True)
        cw = np.cumsum(w, axis=0)
        u = np.random.uniform(size=gam0.shape)
        idx = (cw < u[None, :, :]).sum(axis=0).clip(0, self.ngrid - 1)
        # jitter inside the grid cell so the draw is continuous
        out = (
            self.grid[:, 0, 0][idx]
            + (np.random.uniform(size=gam0.shape) - 0.5) / self.ngrid
        )
        return out.clip(1e-6, 1.0 - 1e-6)

    def parent_draw(
        self, gam0, rho, n=1, setup=None, latent=False, maxiter=10000
    ):
        """Draw theta for a new experiment from the current parent.

        Replaces ``chol_sample_nper_constraints(theta0, Sigma0, n)``.  The
        bounds need no rejection -- the copula's support is the box by
        construction -- but a custom ``constraint_func`` carves a region out of
        the box, so pass ``setup`` to reject draws outside it, exactly as the
        truncated hierarchy did.  With ``constraint_func="bounds"`` every draw
        is accepted on the first try.

        :param gam0: (..., p)
        :param rho: (..., p), broadcastable against ``gam0``.
        :param setup: the CalibSetup, if its constraint region must be honoured.
        :param latent: return the latent v instead of theta, for callers that
            store the latent state (calibClust's empty clusters).
        :return: ``gam0.shape`` when n == 1, else (..., n, p).
        """
        gam0 = np.asarray(gam0, dtype=float)
        rho = np.broadcast_to(np.asarray(rho, dtype=float), gam0.shape)
        p = gam0.shape[-1]
        sd = np.sqrt(1.0 - rho)
        if n == 1:
            shape = gam0.shape
            g_flat = gam0.reshape(-1, p)
            sd_flat = sd.reshape(-1, p)
        else:
            shape = (*gam0.shape[:-1], n, p)
            g_flat = np.repeat(gam0.reshape(-1, p), n, axis=0)
            sd_flat = np.repeat(sd.reshape(-1, p), n, axis=0)

        v_flat = g_flat + sd_flat * np.random.normal(size=g_flat.shape)
        if setup is None:
            out = v_flat if latent else self.to_theta(v_flat)
            return out.reshape(shape)

        good = setup.checkConstraints(
            tran_unif(
                self.to_theta(v_flat), setup.bounds_mat, setup.bounds.keys()
            )
        )
        j = 0
        while np.any(np.logical_not(good)):
            if j >= maxiter:
                raise ValueError(
                    "Failed to find samples that fulfill the constraints "
                    f"after {maxiter} iterations."
                )
            bad = np.where(np.logical_not(good))[0]
            v_flat[bad] = g_flat[bad] + sd_flat[bad] * np.random.normal(
                size=(bad.size, p)
            )
            good[bad] = setup.checkConstraints(
                tran_unif(
                    self.to_theta(v_flat[bad]),
                    setup.bounds_mat,
                    setup.bounds.keys(),
                )
            )
            j += 1
        out = v_flat if latent else self.to_theta(v_flat)
        return out.reshape(shape)

    def theta_sd_implied(self, gam0, rho, nsim=20000):
        """Theta-scale sd of a new experiment's theta, given the parent.

        What ``Sigma0``'s diagonal was being read as.
        """
        return self.parent_draw(gam0, rho, n=nsim).std(axis=-2)


def parent_draw(gam0, rho, a=None, b=None, n=1, setup=None):
    """Functional form of :meth:`CopulaHyper.parent_draw`."""
    return CopulaHyper(np.shape(gam0)[-1], a=a, b=b).parent_draw(
        gam0, rho, n=n, setup=setup
    )


def implied_theta_cov(copula, gam0, rho, nsim=128, chunk=512, seed=0):
    """Theta-scale covariance of a new experiment's theta, given the parent.

    The copula has no ``Sigma0``: the spread of theta_i about the parent is
    set jointly by rho and the margin, and it varies over the box because the
    map from the latent scale is nonlinear.  This returns the quantity
    ``Sigma0``'s diagonal was being read as -- how much a new experiment's
    theta varies about the parent, on the theta scale -- as a diagonal matrix,
    by Monte Carlo with common random numbers.

    The off-diagonals are zero: this hierarchy has no cross-coordinate
    correlation.

    :param copula: a :class:`CopulaHyper`.
    :param gam0: (nmcmc, ntemps, p)
    :param rho: (nmcmc, ntemps, p)
    :return: (nmcmc, ntemps, p, p), diagonal.
    """
    gam0 = np.asarray(gam0, dtype=float)
    nmcmc, ntemps, p = gam0.shape
    z = np.random.default_rng(seed).standard_normal(nsim)[:, None, None, None]
    out = np.zeros((nmcmc, ntemps, p, p))
    idx = np.arange(p)
    for lo in range(0, nmcmc, chunk):
        hi = min(lo + chunk, nmcmc)
        sd = np.sqrt(1.0 - rho[lo:hi])[None, ...]
        draws = copula.to_theta(gam0[None, lo:hi] + sd * z)
        out[lo:hi, :, idx, idx] = draws.var(axis=0)
    return out


def latent_start_var(start_var_theta):
    """Convert a theta-scale starting proposal variance to the latent scale."""
    return float(start_var_theta) / (_DPHI0**2)
