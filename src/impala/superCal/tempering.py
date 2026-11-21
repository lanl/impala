"""
Swap scheduling and temperature-ladder adaptation for parallel tempering.

This module is shared by the pooled, hierarchical and clustered samplers.  It
decides *which* pairs of temperatures attempt a state swap in each swap round,
keeps running estimates of the swap rejection rate between neighbouring
temperatures, and (optionally) re-spaces the inverse-temperature ladder during
burn-in so that every neighbouring pair has the same rejection rate.

Swap schemes
------------
``"deo"`` (default)
    Deterministic even/odd swaps between *adjacent* temperatures, the
    non-reversible parallel tempering scheme of Okabe et al. (2001) analysed by
    Syed et al. (2022).  Rounds alternate between the pairs
    ``(0,1), (2,3), ...`` and ``(1,2), (3,4), ...``.  Because the parity
    alternates deterministically, an accepted state keeps moving in the same
    direction along the ladder, so the number of swap rounds needed to carry a
    state from the hottest to the coldest chain grows roughly linearly in the
    number of temperatures instead of quadratically as under a reversible
    random-walk scheme.
``"seo"``
    Stochastic even/odd: the parity of each round is chosen at random.  This is
    the reversible counterpart of ``"deo"`` and is provided mainly for
    comparison.
``"random"``
    The scheme used by earlier versions of impala: each round draws
    ``floor(ntemps / 2)`` disjoint pairs uniformly at random from the whole
    ladder.  Pairs that are far apart on a geometric ladder are almost never
    accepted.

Ladder adaptation
-----------------
When adaptation is switched on, the rejection rate ``r_k`` between inverse
temperatures ``beta_k`` and ``beta_{k+1}`` is estimated with the
Rao-Blackwellised estimator ``1 - min(1, exp(log_alpha))``, averaged over the
swap attempts on that edge.  The cumulative rejection
``Lambda(beta_k) = sum_{j<k} r_j`` approximates the *communication barrier*
of Syed et al. (2022).  A new ladder is chosen so that
``Lambda`` is equally spaced along it (their Algorithm 4), which equalises the
rejection rate across all neighbouring pairs.  The coldest (``beta = 1``) and
hottest inverse temperatures are kept fixed.  Adaptation happens at the end of
rounds whose length doubles each time, and stops at ``adapt_until`` so the
sampler is a fixed (non-adaptive) Markov chain after burn-in.

References
----------
Okabe, T., Kawata, M., Okamoto, Y. and Mikami, M. (2001).  Replica-exchange
Monte Carlo method for the isobaric-isothermal ensemble.  Chemical Physics
Letters 335, 435-439.

Syed, S., Bouchard-Cote, A., Deligiannidis, G. and Doucet, A. (2022).
Non-reversible parallel tempering: a scalable highly parallel MCMC scheme.
Journal of the Royal Statistical Society Series B 84(2), 321-350.

Vousden, W. D., Farr, W. M. and Mandel, I. (2016).  Dynamic temperature
selection for parallel tempering in Markov chain Monte Carlo simulations.
Monthly Notices of the Royal Astronomical Society 455(2), 1919-1937.
"""

import numpy as np

SWAP_SCHEMES = ("deo", "seo", "random")


def even_odd_pairs(ntemps, parity):
    """Disjoint adjacent pairs ``(k, k+1)`` with ``k % 2 == parity``."""
    first = np.arange(parity, ntemps - 1, 2)
    return np.column_stack([first, first + 1])


def communication_barrier_ladder(itl, rejection, min_rejection=1e-4):
    """
    Re-space an inverse-temperature ladder so that the estimated cumulative
    rejection (communication barrier) is equally spaced along it.

    :param itl: inverse temperatures, decreasing, with ``itl[0] = 1``.
    :param rejection: estimated rejection rates between ``itl[k]`` and
        ``itl[k+1]``, length ``len(itl) - 1``.
    :param min_rejection: floor applied to the rejection rates so that the
        cumulative barrier is strictly increasing.
    :return: new inverse temperatures with the same end points.
    """
    itl = np.asarray(itl, dtype=float)
    rejection = np.clip(np.asarray(rejection, dtype=float), min_rejection, 1.0)
    # work on an increasing grid beta_0 < ... < beta_N (hot -> cold)
    beta = itl[::-1]
    rej = rejection[::-1]
    barrier = np.concatenate([[0.0], np.cumsum(rej)])
    targets = np.linspace(0.0, barrier[-1], len(beta))
    new_beta = np.interp(targets, barrier, beta)
    new_beta[0], new_beta[-1] = beta[0], beta[-1]
    return new_beta[::-1].copy()


class SwapScheduler:
    """
    Chooses swap pairs, tracks swap statistics and adapts the ladder.

    The sampler keeps a working copy of the inverse-temperature ladder
    (``self.itl``) and must use it in place of ``setup.itl``; it is updated in
    place, so any view of it stays current.  Arrays the sampler derives from the
    ladder must be rebuilt whenever :meth:`maybe_adapt` returns ``True``.
    """

    def __init__(
        self,
        itl,
        scheme="deo",
        adapt=False,
        adapt_until=None,
        first_round=100,
    ):
        if scheme not in SWAP_SCHEMES:
            raise ValueError(
                f"swap_scheme must be one of {SWAP_SCHEMES}, got {scheme!r}"
            )
        if adapt and scheme == "random":
            raise ValueError(
                "ladder adaptation needs adjacent swaps; use swap_scheme='deo' or 'seo'"
            )
        self.itl = np.array(itl, dtype=float)
        self.ntemps = self.itl.size
        self.scheme = scheme
        self.adapt = adapt and self.ntemps > 2
        self.adapt_until = adapt_until
        self.round_length = first_round
        self.next_adapt = None
        self.n_rounds = 0
        self.n_adaptations = 0
        nedge = max(self.ntemps - 1, 0)
        self._rej_sum = np.zeros(nedge)
        self._rej_n = np.zeros(nedge)
        # statistics accumulated since the last adaptation (or since the start)
        self.rejection_total = np.zeros(nedge)
        self.attempts_total = np.zeros(nedge)
        # round-trip bookkeeping: which replica sits at each temperature, and
        # whether it has reached the hottest temperature since it was last cold.
        # A round trip is a replica going cold -> hot -> cold.
        self._label = np.arange(self.ntemps)
        self._seen_hot = np.zeros(self.ntemps, dtype=bool)
        if self.ntemps > 1:
            self._seen_hot[self.ntemps - 1] = True
        self.round_trips = 0

    def pairs(self):
        """Pairs of temperature indices (``k`` rows of 2) for the next round."""
        if self.scheme == "deo":
            parity = self.n_rounds % 2
        elif self.scheme == "seo":
            parity = np.random.randint(2)
        else:
            npair = self.ntemps // 2
            sw = np.random.choice(self.ntemps, 2 * npair, replace=False)
            sw = np.sort(sw.reshape(-1, 2), axis=1)
            self.n_rounds += 1
            return sw
        self.n_rounds += 1
        return even_odd_pairs(self.ntemps, parity)

    def record(self, pairs, log_alpha, accepted=None):
        """
        Record one round of swap attempts.

        :param pairs: the pairs returned by :meth:`pairs`.
        :param log_alpha: log acceptance ratio of each pair.
        :param accepted: boolean array, which swaps were accepted; used to
            count round trips (cold -> hot -> cold) of each replica.
        """
        pairs = np.asarray(pairs)
        if pairs.size == 0:
            return
        if accepted is not None:
            self._track_round_trips(pairs[np.asarray(accepted, dtype=bool)])
        adjacent = pairs[:, 1] - pairs[:, 0] == 1
        if not np.any(adjacent):
            return
        edge = pairs[adjacent, 0]
        la = np.asarray(log_alpha, dtype=float)[adjacent]
        rej = 1.0 - np.exp(np.minimum(np.nan_to_num(la, nan=-np.inf), 0.0))
        np.add.at(self._rej_sum, edge, rej)
        np.add.at(self._rej_n, edge, 1.0)
        np.add.at(self.rejection_total, edge, rej)
        np.add.at(self.attempts_total, edge, 1.0)

    def _track_round_trips(self, swapped):
        if swapped.size == 0:
            return
        a, b = swapped[:, 0], swapped[:, 1]
        self._label[a], self._label[b] = (
            self._label[b].copy(),
            self._label[a].copy(),
        )
        hot = self._label[self.ntemps - 1]
        self._seen_hot[hot] = True
        cold = self._label[0]
        if self._seen_hot[cold]:
            self.round_trips += 1
            self._seen_hot[cold] = False

    def rejection_rates(self):
        """Estimated rejection rate on every adjacent edge (NaN if unvisited)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.rejection_total / self.attempts_total

    def maybe_adapt(self, m):
        """
        Adapt the ladder if a round has just finished.

        :param m: current MCMC iteration.
        :return: ``True`` if ``self.itl`` changed.
        """
        if not self.adapt:
            return False
        if self.adapt_until is not None and m > self.adapt_until:
            return False
        if self.next_adapt is None:
            self.next_adapt = m + self.round_length
            return False
        if m < self.next_adapt or np.any(self._rej_n == 0):
            return False
        rej = self._rej_sum / self._rej_n
        self.itl[:] = communication_barrier_ladder(self.itl, rej)
        self._rej_sum[:] = 0.0
        self._rej_n[:] = 0.0
        # report post-adaptation statistics only
        self.rejection_total[:] = 0.0
        self.attempts_total[:] = 0.0
        self.round_length *= 2
        self.next_adapt = m + self.round_length
        self.n_adaptations += 1
        return True
