""" Extension to Statistical Models Hier """
from mpi4py import MPI
from random import sample, shuffle
from itertools import combinations
from transport import TransportHB, TransportTC, TransportFP
import statistical_models_hier as smh

class MPI_Error(Exception):
    pass

class ParallelTemperMaster(smh.ParallelTemperMaster):
    """ overwrites relevant functions from smh.ParallelTemperMaster to operate
        in an MPI environment. """

    def initialize_chains(self, kwargs):
        sent = [
            self.comm.isend(
                ('initialize_chain',{**kwargs, temperature = temp}),
                dest = chain_rank
                )
            for temp, chain_rank
            in zip(self.temperature_ladder, self.chain_ranks)
            )]
        return

    def try_swap_states(rank_1, rank_2):
        self.comm.isend(('try_swap_state_sup', rank_2), dest = rank_1)
        self.comm.isend(('try_swap_state_inf', rank_1), dest = rank_2)
        return self.comm.irecv(source = rank_1)


    def __init__(self, comm, rank, temperature_ladder, **kwargs):
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()
        self.temperature_ladder = temperature_ladder
        self.chain_ranks = tuple(range(1, self.size))
        self.initialize_chains(kwargs)
        pass

class Dispatcher(object):
    """  """
    comm = None
    rank = None
    chain = None
    dispatcher = {}

    def return_value(self, retv, dest = 0):
        self.comm.send(retv, dest = dest)
        return

    def watch(self):
        try:
            while True:
                parcel = self.comm.recv(source = 0)
                self.dispatch[parcel[0]](parcel[1])
        except BreakException:
            pass
        return

    def initialize_chain(self, args):
        self.chain = smh.Chain(**args)
        self.chain.rank = self.rank
        return

    def set_temperature(self, args):
        self.chain.set_temperature(args)
        return

    def initialize_sampler(self, args):
        self.chain.initialize_sampler(args)
        return

    def sample(self, args):
        self.chain.sample(args)
        return

    def get_state(self, args):
        self.return_value(self.chain.get_state())
        return

    def set_state(self, args):
        self.chain.set_state(args)
        return

    def get_accept_prob(self, args):
        self.return_value(self.chain.get_accept_probability())
        return

    def try_swap_state_sup(self, args):
        inf_rank = args

        state_a = self.chain.get_state()
        state_b = self.comm.recv(source = inf_rank)

        self.comm.send(state_a, dest = inf_rank)
        recv = self.comm.irecv(source = inf_rank)

        lpaa = self.chain.log_posterior_state(state_a)
        lpab = self.chain.log_posterior_state(state_b)

        lpba, lpbb = recv.wait()
        if log(uniform()) < lpab + lpba - lpaa - lpbb:
            self.comm.send('swap_yes', dest = inf_rank)
            self.chain.set_state(state_b)
            self.return_value((1, self.rank, inf_rank))
        else:
            self.comm.send('swap_no', dest = inf_rank)
            self.return_value((0, self.rank, inf_rank))
        return

    def try_swap_state_inf(self, args):
        sup_rank = args[0]

        state_b = self.chain.get_state()
        self.comm.send(state_b, dest = sup_rank)
        state_a = self.comm.recv(source = sup_rank)

        lpba = self.chain.log_posterior_state(state_a)
        lpbb = self.chain.log_posterior_state(state_b)

        self.comm.send((lpba,lpbb), dest = sup_rank)

        recv = self.comm.irecv(source = sup_rank)
        parcel = recv.wait()
        if parcel == 'swap_yes':
            self.chain.set_state(state_a)
        else:
            pass
        return

    def __init__(self, comm, rank):
        """ Initialization Routine """
        self.comm = comm
        self.rank = rank

        self.dispatch = {
            'initialize_chain'   : self.initialize_chain,
            'set_temperature'    : self.set_temperature,
            'initialize_sampler' : self.initialize_sampler,
            'sample'             : self.sample,
            'get_state'          : self.get_state,
            'set_state'          : self.set_state,
            'get_accept_prob'    : self.get_accept_prob,
            'get_history'        : self.get_history,
            'try_swap_state_inf' : self.try_swap_state_inf,
            'try_swap_state_sub' : self.try_swap_state_sub,
            }
        return

if __name__ == '__main__':
    pass

# EOF
