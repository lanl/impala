# Required Modules
import numpy as np
from numpy.random import uniform
# Builtin modules
from random import shuffle
from math import log
# Custom Modules
import parallel_tempering as pt

class BreakException(Exception):
    pass

class PTSlave(pt.PTSlave):
    dispatch = {}

    def try_swap_state_sup(self, inf_rank):
        state1 = self.get_state()
        sent = self.comm.isend(state1, dest = inf_rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, souce = inf_rank)
        lp11 = self.log_posterior_state(state1)
        state2 = recv.wait()
        lp12 = self.log_posterior_state(state2)
        recv = self.comm.irecv(source = inf_rank)
        lp21, lp22 = recv.wait()
        log_alpha = lp12 + lp21 - lp11 - lp12
        if log(uniform()) < log_alpha:
            self.comm.send('swap_yes', dest = inf_rank)
            self.set_state(state2)
            self.comm.send(True, dest = 0)
        else:
            self.comm.send('swap_no', dest = inf_rank)
            self.comm.send(False, dest = 0)
        return

    def try_swap_state_inf(self, sup_rank):
        state2 = self.get_state()
        sent = self.comm.isend(state2, dest = sup_rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = sup_rank)
        lp22 = self.log_posterior_state(state2)
        state1 = recv.wait()
        lp21 = self.log_posterior_state(state1)
        sent = self.comm.isend((lp21, lp22), dest = sup_rank)
        recv = self.comm.irecv(source = sup_rank)
        parcel = recv.wait()
        if parcel == 'swap_yes':
            self.set_state(state1)
        elif parcel == 'swap_no':
            pass
        else:
            raise ValueError('Acceptable: swap_yes, swap_no; observed {}'.format(parcel))
        return

    def complete(self, *args):
        raise BreakException('Done!')

    def watch(self):
        self.init_dispatch()
        try:
            while True:
                recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = 0)
                parcel = recv.wait()
                self.dispatch[parcel[0]](*parcel[1])
        except BreakException:
            pass
        return

    def init_dispatch(self):
        self.dispatch = {
            'initialize_chain'   : self.initialize_chain,
            'set_temperature'    : self.set_temperature,
            'initialize_sampler' : self.initialize_sampler,
            'sample_k'           : self.sample_k,
            'try_swap_state_sup' : self.try_swap_state_sup,
            'try_swap_state_inf' : self.try_swap_state_inf,
            'get_state'          : self.get_state,
            'set_state'          : self.set_state,
            'get_accept_prob'    : self.get_accept_prob,
            'get_history'        : self.get_history,
            'write_to_disk'      : self.write_to_disk,
            'complete'           : self.complete,
            }
        return

class PTMaster(pt.PTMaster):
    def initialize_chains(self, temperature_ladder, kwargs):
        self.chain_ranks = list(range(1, self.size))
        assert len(temperature_ladder) == len(self.chain_ranks)
        sent = [
            self.comm.isend(('initialize_chain', {'temp' : temp, **kwargs}), dest = rank)
            for temp, rank in zip(temperature_ladder, self.chain_ranks)
            ]
        recv = [self.comm.irecv(source = rank) for rank in self.chain_ranks]
        assert all([r.wait() for r in recv])
        return

    def set_temp_ladder(self, temp_ladder):
        sent = [
            self.comm.isend(('set_temperature', temp), dest = chain_rank)
            for temp, chain_rank in zip(temp_ladder, self.chain_ranks)
            ]
        recv = [self.comm.irecv(source = chain_rank) for chain_rank in self.chain_ranks]
        assert all([r.wait() for r in recv])
        return

    def sample_k(self, k):
        sent = [self.comm.isend(('sample_k', k), dest = rank) for rank in self.chain_ranks]
        recv = [self.comm.irecv(source = rank) for rank in self.chain_ranks]
        assert all([r.wait() for r in recv])
        return

    def try_swap_states(self, rank_1, rank_2):
        sent1 = self.comm.isend(('try_swap_states_sup', rank_2), dest = rank_1)
        sent2 = self.comm.isend(('try_swap_states_inf', rank_1), dest = rank_2)
        return self.comm.irecv(source = rank_1)

    def temper_chains(self):
        chain_ranks = self.chain_ranks.copy()
        shuffle(chain_ranks)
        swaps = []
        for rank1, rank2 in zip(chain_ranks[::2], chain_ranks[1::2]):
            swaps.append((rank1, rank2, self.try_swap_states(rank1,rank2)))
        swaps = [(swap[0],swap[1],swap[2].wait()) for swap in swaps]
        self.swaps.append(swaps)
        return

    def __init__(self, comm, size, temperature_ladder, **kwargs):
        self.comm = comm
        self.size = size
        self.initialize_chains(temperature_ladder, kwargs)
        return
