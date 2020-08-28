"""
Defines Parallel Tempering objects

To implement, make your sampler inherit from PTSlave; then write a new
Master class that inherits from PTMaster.  Overwrite all methods that raise
NotImplementedError's (as these are model specific).
"""
import numpy as np

class PTSlave(object):
    def set_temper_temp(self, temper_temp):
        self.temper_temp = temper_temp
        self.inv_temper_temp = 1. / self.temper_temp
        return

    def init_sampler(self, ns):
        raise NotImplementedError

    def iter_sample(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def try_swap_states(self, other_chain):
        statel = self.get_state()
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

    def __init__(self, temper_temp, **kwargs):
        raise NotImplementedError

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
        for chain_a, chain_b, swapped in self.swaps:
            swap_y[chain_a, chain_b] += swapped
            swap_n[chain_a, chain_b] += 1 - swapped
        return swap_y / (swap_n + swap_y)

    def try_swap_states(self):
        chain_idx = list(range(self.size))
        shuffle(chain_idx)

        swaps = []
        for cidx1, cidx2 in zip(chain_idx[::2],chain_idx[1::2]):
            swaps.append((cidx1, cidx2, self.chains[cidx1].try_swap_states(self.chains[cidx2])))
        self.swaps.append(swaps)
        return

    def __init__(self, temperatures, **kwargs):
        raise NotImplementedError
