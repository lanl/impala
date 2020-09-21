"""
Defines Parallel Tempering objects

To implement, make your sampler inherit from PTSlave; then write a new
Master class that inherits from PTMaster.  Overwrite all methods that raise
NotImplementedError's (as these are model specific).
"""
import numpy as np
from numpy.random import uniform
from random import shuffle
from math import log

class PTSlave(object):
    def set_temper_temp(self, temper_temp):
        self.temper_temp = temper_temp
        self.inv_temper_temp = 1. / self.temper_temp
        return

    def initialize_sampler(self, ns):
        raise NotImplementedError

    def iter_sample(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def log_posterior_state(self, state):
        raise NotImplementedError

    def try_swap_state_sup(self, other_chain):
        """ In each state swap, there is a 'superior' and an 'inferior' chain.  The 'superior'
        chain controls the swap--the swap probability is calculated there.

        the responsibility of the inferior chain is to calculate log-posteriors under its original
        state, and under the state of the superior chain.
        """
        state1 = self.get_state()
        state2 = other_chain.get_state()

        lp11 = self.log_posterior_state(state1)
        lp12 = self.log_posterior_state(state2)
        lp21 = other_chain.log_posterior_state(state1)
        lp22 = other_chain.log_posterior_state(state2)

        if log(uniform()) < lp21 + lp12 - lp11 - lp22:
            self.set_state(state2)
            other_chain.set_state(state1)
            return True
        else:
            return False

    def sample_k(self, k):
        for _ in range(k):
            self.iter_sample()
        return

    def initialize_chain(self, temper_temp, **kwargs):
        raise NotImplementedError

    def write_to_disk(self, path, nburn, thin):
        raise NotImplementedError

    def __init__(self, temper_temp, **kwargs):
        self.initialize_chain(temper_temp, **kwargs)
        return

class PTMaster(object):
    chains = []
    swaps  = []

    def set_temp_ladder(self, temp_ladder):
        for temp, chain in zip(temp_ladder, self.chains):
            chain.set_temper_temp(temp)
        return

    def sample_k(self, k):
        for chain in self.chains:
            chain.sample_k(k)
        return

    def get_swap_probability(self):
        swap_y = np.zeros((self.size, self.size))
        swap_n = np.zeros((self.size, self.size))

        for swap_generation in self.swaps:
            for chain_a, chain_b, swapped in swap_generation:
                swap_y[chain_a, chain_b] += swapped
                swap_n[chain_a, chain_b] += 1 - swapped

        return swap_y / (swap_n + swap_y)

    def temper_chains(self):
        chain_idx = list(range(self.size))
        shuffle(chain_idx)
        swaps = []
        for cidx1, cidx2 in zip(chain_idx[::2], chain_idx[1::2]):
            swaps.append((cidx1, cidx2, self.try_swap_states(cidx1, cidx2)))
        self.swaps.append(swaps)
        return

    def try_swap_states(self, cid1, cid2):
        return self.chains[cid1].try_swap_state_sup(self.chains[cid2])

    def sample(self, ns, k = 5):
        for chain in self.chains:
            chain.initialize_sampler(ns)

        sampled = 0
        print('\rSampling {:.1%} Complete'.format(sampled / ns), end = '')
        for _ in range(ns // k):
            self.sample_k(k)
            self.temper_chains()
            sampled += k
            print('\rSampling {:.1%} Complete'.format(sampled / ns), end = '')

        self.sample_k(ns % k)
        sampled += (ns % k)
        print('\rSampling {:.1%} Complete'.format(sampled / ns))
        return

    def initialize_chains(self, temperature_ladder, kwargs):
        raise NotImplementedError
        return

    def __init__(self, temperature_ladder, **kwargs):
        self.size = len(temperature_ladder)
        self.initialize_chains(temperature_ladder, kwargs)
        return
