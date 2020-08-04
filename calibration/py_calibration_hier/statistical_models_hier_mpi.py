""" Extension to Statistical Models Hier """
from mpi4py import MPI
from random import sample, shuffle
from numpy.random import uniform
from itertools import combinations
from transport import TransportHB, TransportTC, TransportFP
import statistical_models_hier as smh
import numpy as np
from math import log

MPI_MESSAGE_SIZE = 2**18  # Maximum MPI message size for pickled sends

class MPI_Error(Exception):
    pass

class BreakException(Exception):
    pass

class ParallelTemperMaster(smh.ParallelTemperMaster):
    """ overwrites relevant functions from smh.ParallelTemperMaster to operate
        in an MPI environment. """

    def initialize_chains(self, kwargs):
        # Send initialization routine for each chain to each node
        # print('sending initialization routines to each chain')
        sent = [
            self.comm.isend(
                ('initialize_chain',{**kwargs, 'temperature' : temp}),
                dest = chain_rank
                )
            for temp, chain_rank
            in zip(self.temperature_ladder, self.chain_ranks)
            ]
        # print('Waiting on Done signals from each chain')
        # Recieve Done from chains
        rcvd = [self.comm.irecv(source = i) for i in self.chain_ranks]
        parcels = [rec.wait() for rec in rcvd]
        # Verify that all are done
        assert all(parcels)
        # print('All chains succeeded initialization, moving on')
        return

    def initialize_sampler(self, tns):
        self.tns = tns
        # Send Initialization routine for sampler to each chain
        # print('Sending initialize sampler routines to each chain')
        sent = [
            self.comm.isend(('initialize_sampler', tns), dest = chain_rank)
            for chain_rank in self.chain_ranks
            ]
        # Recieve Done from chains
        # print('waiting for initialize_sampler routine to complete')
        rcvd = [self.comm.irecv(source = i) for i in self.chain_ranks]
        parcels = [rec.wait() for rec in rcvd]
        # Verify that all are done
        # print('verifying that each chain completed initialization')
        assert all(parcels)
        return

    def write_to_disk(self, path):
        self.comm.send(('write_to_disk', path), dest = 1)
        return

    def get_state(self, rank):
        sent = self.comm.isend(('get_state',0), dest = rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = rank)
        state = recv.wait()
        return state

    def try_swap_states(self, rank_1, rank_2):
        # print('attempting to swap chain {} and {}'.format(rank_1, rank_2))
        self.comm.isend(('try_swap_state_sup', rank_2), dest = rank_1)
        self.comm.isend(('try_swap_state_inf', rank_1), dest = rank_2)
        return self.comm.irecv(source = rank_1)

    def temper_chains(self):
        # print('tempering chains, sending attempt swap instructions')
        chain_idx = list(self.chain_ranks)
        shuffle(chain_idx)

        swaps = []
        for cidx1, cidx2 in zip(chain_idx[::2], chain_idx[1::2]):
            swaps.append(self.try_swap_states(cidx1, cidx2))
            return

        # print('waiting to succeed swapping')
        received = [x.wait() for x in swaps]

        for x in received:
            if x[0] == 1: # if swap successful, append to succeeded swaps
                self.swap_yes.append(tuple(x[1:]))
            elif x[0] == 0: # if swap failed, append to failed swaps
                self.swap_no.append(tuple(x[1:]))
        # print('swapping completed, moving on')
        return

    def sample_chains(self, ns):
        # print('sampling chains {} steps'.format(ns))
        sent = [
            self.comm.isend(('sample', ns), dest = i)
            for i in self.chain_ranks
            ]
        # print('declaring recieved values for chains and waiting')
        recv = [self.comm.irecv(source = i) for i in self.chain_ranks]
        parcels = [rec.wait() for rec in recv]
        # print('asserting all chains sent True')
        assert all(parcels)
        # print('sampling {} completed, moving on'.format(ns))
        return

    def get_history(self, *args):
        state = self.get_state(1)
        d  = len(state['theta0'])
        ns = len(list(range(self.tns))[args[0]::args[1]])
        self.comm.isend(('get_history',args), dest = 1)
        history = np.empty((ns, d))
        self.comm.Recv([history, MPI.DOUBLE], source = 1)
        return history

    def parameter_pairwise_plot(self, theta, path):
        sent = self.comm.isend(('parameter_trace_plot', (path, theta.shape), dest = 1)
        sent = self.comm.Send([theta, MPI.DOUBLE], dest = 1)
        return

    def complete(self):
        for chain_rank in self.chain_ranks:
            self.comm.isend(('complete',0), dest = chain_rank)
        return

    def __init__(self, comm, size, temperature_ladder, **kwargs):
        self.comm = comm
        self.size = size
        self.temperature_ladder = temperature_ladder
        self.chain_ranks = tuple(range(1, self.size))
        self.initialize_chains(kwargs)
        pass

class Dispatcher(object):
    """ Wrapper for statistical_models_hier.Chain for an MPI environment. """
    comm = None
    rank = None
    chain = None
    dispatch = {}

    def return_value(self, retv, dest = 0):
        self.comm.send(retv, dest = dest)
        return

    def watch(self):
        try:
            while True:
                recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = 0)
                parcel = recv.wait()
                self.dispatch[parcel[0]](parcel[1])

        except BreakException:
            pass
        return

    def initialize_chain(self, args):
        self.chain = smh.Chain(**args)
        self.chain.rank = self.rank
        self.return_value(True)
        return

    def set_temperature(self, args):
        self.chain.set_temperature(args)
        self.return_value(True)
        return

    def initialize_sampler(self, args):
        self.chain.initialize_sampler(args)
        self.return_value(True)
        return

    def sample(self, args):
        #print(args)
        self.chain.sample(args)
        self.return_value(True)
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

    def get_history(self, args):
        #self.return_value(self.chain.get_history(*args))
        history = self.chain.get_history(*args)
        self.comm.Send([history, MPI.DOUBLE], dest = 0)
        return

    def try_swap_state_sup(self, args):
        inf_rank = args
        # Get Current States
        state_a = self.chain.get_state()
        sent = self.comm.isend(state_a, dest = inf_rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = inf_rank)
        # Compute Log Posteriors
        lpaa = self.chain.log_posterior_state(state_a)
        state_b = recv.wait()
        lpab = self.chain.log_posterior_state(state_b)
        # Await log-posteriors from inferior Chain
        recv = self.comm.irecv(source = inf_rank)
        lpba, lpbb = recv.wait()
        # Make decision whether to swap
        if log(uniform()) < lpab + lpba - lpaa - lpbb:
            # If yes
            self.comm.send('swap_yes', dest = inf_rank)
            self.chain.set_state(state_b)
            self.return_value((1, self.rank, inf_rank))
        else:
            # If no
            self.comm.send('swap_no', dest = inf_rank)
            self.return_value((0, self.rank, inf_rank))
        return

    def try_swap_state_inf(self, args):
        sup_rank = args
        # Get Current States
        state_b = self.chain.get_state()
        sent = self.comm.isend(state_b, dest = sup_rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = sup_rank)
        # Compute Log Posteriors
        lpbb = self.chain.log_posterior_state(state_b)
        state_a = recv.wait()
        lpba = self.chain.log_posterior_state(state_a)
        # Send Log Posteriors back to Superior Chain
        sent = self.comm.isend((lpba,lpbb), dest = sup_rank)
        # Wait for / enact superior chain's decision
        recv = self.comm.irecv(source = sup_rank)
        parcel = recv.wait()
        if parcel == 'swap_yes':
            self.chain.set_state(state_a)
        elif parcel == 'swap_no':
            pass
        else:
            raise ValueError('Acceptable: swap_yes, swap_no; Observed {}'.format(parcel))
        return

    def write_to_disk(self, args):
        path = args
        self.chain.write_to_disk(path)
        return

    def parameter_pairwise_plot(self, args):
        path, theta_shape = args
        theta = np.empty(theta_shape)
        self.comm.Recv([theta, MPI.DOUBLE], source = 0)
        self.chain.parameter_pairwise_plot(theta, path)
        return

    def complete(self, args):
        return BreakException('Done')

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
            'try_swap_state_sup' : self.try_swap_state_sup,
            'write_to_disk'      : self.write_to_disk,
            'complete'           : self.complete,
            }
        return

if __name__ == '__main__':
    pass

# EOF
