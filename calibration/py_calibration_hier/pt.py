"""
Generic rewrite of parallel_tempering.  Meant to take generic object.  Make it conform to "chain"
"""
from random import shuffle
from numpy.random import uniform
from math import log

class Samples(object):
    """ A generic samples object.  Meant to show you how I'm expecting the chain object to be
    structured. """
    x = None
    y = None
    accepted = None

    def __init__(self, ns):
        self.x = np.empty(ns + 1)
        self.y = np.empty(ns + 1)
        self.accepted = np.empty(ns + 1)
        self.accepted[0] = 0.
        return

class PTChain(object):
    """ A genetric PTChain object.  Shows user how I'm expecting PTChain to be structured """
    temper_temp     = None  # tempering temperature
    inv_temper_temp = None
    samples         = None  # class with numpy arrays for each element

    def pairwise_parameter_plot(self, path):
        raise NotImplementedError

    def get_accept_probability(self, nburn = 0):
        return self.samples.accepted[nburn:].mean()

    def get_state(self):
        """ This should return a namedtuple describing the current state of the chain. """
        raise NotImplementedError

    def set_state(self, state):
        """ state - a namedtuple describing a "state" of the chain we want to set it to. """
        raise NotImplementedError

    def log_posterior_state(self, state):
        """ returns a DOUBLE -- the log posterior for a given state """
        raise NotImplementedError

    def write_to_disk(self, path, nburn, thin):
        """ Write sampler results to disk """
        raise NotImplementedError

    def sample_n(self, n):
        """ advance the sampler by n iterations """
        for _ in range(n):
            self.iter_sample()

    def set_temperature(self, temperature):
        """ Set the tempering temperature of the sampler """
        self.temper_temp     = temperature
        self.inv_temper_temp = 1. / temperature
        return

    def iter_sample(self):
        """ advance the sampler 1 iteration """
        raise NotImplementedError

class PTSlave(object):
    def set_temperature(self, temperature):
        self.chain.set_temperature(temperature)
        return

    def get_state(self):
        return self.chain.get_state()

    def set_state(self, state):
        self.chain.set_state(state)
        return

    def log_posterior_state(self, state):
        return self.chain.log_posterior_state(state)

    def write_to_disk(self, path, nburn, thin):
        self.chain.write_to_disk(path, nburn, thin)

    def pairwise_parameter_plot(self, path):
        self.chain.pairwise_parameter_plot(self, path)
        return

    def get_accept_probability(self, nburn):
        return self.chain.get_accept_probability(nburn)

    def try_swap_state_sup(self, inf_chain):
        state1 = self.get_state()
        state2 = inf_chain.get_state()
        lp11 = self.log_posterior_state(state1)
        lp12 = self.log_posterior_state(state2)
        lp21 = inf_chain.log_posterior_state(state1)
        lp22 = inf_chain.log_posterior_state(state2)
        log_alpha = lp21 + lp12 - lp11 - lp22
        if log(uniform()) < log_alpha:
            self.set_state(state2)
            inf_chain.set_state(state1)
            return True
        else:
            return False

    def sample_n(self, n):
        return self.chain.sample_n(n)

    def initialize_sampler(self, ns):
        return self.chain.initialize_sampler(ns)

    def __init__(self, chain):
        self.chain = chain
        return

class PTMaster(object):
    swaps = []

    def try_swap_states(self, sup_rank, inf_rank):
        return self.chains[sup_rank].try_swap_state_sup(self.chains[inf_rank])

    def temper_chains(self):
        ranks = list(range(len(self.chains)))
        shuffle(ranks)

        swaps = []
        for rank1, rank2 in zip(ranks[::2], ranks[1::2]):
            swaps.append((rank1, rank2, self.try_swap_states(rank1, rank2)))
        self.swaps.append(swaps)
        return

    def sample_n(self, n):
        for chain in self.chains:
            chain.sample_n(n)
        return

    def set_temperature_ladder(self, temperature_ladder):
        assert len(temperature_ladder) == len(self.chains)
        for temp, chain in zip(temperature_ladder, self.chains):
            chain.set_temperature(temp)
        return

    def initialize_sampler(self, ns):
        for chain in self.chains:
            chain.initialize_sampler(ns)
        return

    def sample(self, ns, k = 5):
        self.initialize_sampler(ns)

        sampled = 0
        print('\rSampling {:.1%} Complete'.format(sampled / ns), end = '')
        for _ in range(ns // k):
            self.sample_n(k)
            self.temper_chains()
            sampled += k
            print('\rSampling {:.1%} Complete'.format(sampled / ns), end = '')

        self.sample_n(ns % k)
        sampled += (ns % k)
        print('\rSampling {:.1%} Complete'.format(sampled / ns))
        return

    def get_swap_probability(self):
        try:
            k = len(self.chain_ranks)
        except NameError:
            k = len(self.chain)
        swap_y = np.zeros((k,k))
        swap_n = np.zeros((k,k))
        for swap_generation in self.swaps:
            for chain_a, chain_b, swapped in swap_generation:
                swap_y[chain_a, chain_b] += swapped
                swap_n[chain_a, chain_b] += 1 - swapped

        swap_y = swap_y + swap_y.T
        swap_n = swap_n + swap_n.T
        return swap_y / (swap_y + swap_n)

    def pairwise_parameter_plot(self, path):
        self.chains[0].pairwise_parameter_plot(self, path)
        return

    def get_accept_probability(self, nburn):
        probs = np.array([chain.get_accept_probability(nburn) for chain in self.chains])
        return probs

    def write_to_disk(self, path, nburn, thin):
        self.chains[0].write_to_disk(path, nburn, thin)
        return

    def initialize_chains(self, statmodel, temperature_ladder, kwargs):
        self.chains = [PTSlave(statmodel(temperature = temp, **kwargs)) for temp in temperature_ladder]
        return

    def __init__(self, statmodel, temperature_ladder, **kwargs):
        self.initialize_chains(statmodel, temperature_ladder, kwargs)
        return

# EOF
