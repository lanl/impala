"""
MPI enabled Parallel Tempering
"""
import pt

class BreakException(Exception):
    pass

class PTSlave(pt.PTSlave):
    def watch(self):
        try:
            recv = self.comm.irecv(MPI_MESSAGE_SIZE, source = 0)
            parcel = recv.wait()
            self.dispatch[parcel[0]](*parcel[1:])
        except BreakException:
            pass
        return

    def try_swap_state_sup(self, inf_rank):
        state1 = self.get_state()
        sent = self.comm.isend(state1, dest = inf_rank)
        recv = self.comm.irecv(MPI_MESSAGE_SIZE, source =  inf_rank)
        lp11 = self.log_posterior_state(state1)
        state2 = recv.wait()
        lp12 = self.log_posterior_state(state2)
        recv = self.comm.irecv(source = inf_rank)
        lp21, lp22 = recv.wait()
        log_alpha = lp12 + lp21 - lp11 - lp22
        if log(uniform()) < log_alpha:
            self.comm.send('swap_yes', dest = inf_rank)
            self.set_state(state2)
            self.comm.send(True, dest = 0)
        else:
            self.comm.send('swap_no', dest = inf_rank)
            self.comm.send(False, dest = 0)
        return

    def try_swap_state_inf(self, sup_rank):
        state2 = self.get_state(sup_rank)
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
        elif parcel = 'swap_no':
            pass
        else:
            raise ValueError('Acceptable: swap_yes, swap_no; Observed {}'.format(parcel))
        return

    def initialize_chain(self, **kwargs):
        self.chain = self.statmodel(**kwargs)
        self.comm.send(True, dest = 0)
        return

    def complete(self):
        raise BreakException('Done')

    def pairwise_parameter_plot(self, path):
        super().pairwise_parameter_plot(path)
        self.comm.send(True, dest = 0)
        return

    def write_to_disk(self, path, nburn, thin):
        self.chain.write_to_disk(path, nburn, thin)
        self.comm.send(True, dest = 0)
        return

    def build_dispatch(self):
        self.dispatch = {
            'pairwise_plot'      : self.pairwise_parameter_plot,
            'init_chain'         : self.initialize_chain,
            'set_temperature'    : self.set_temperature,
            'init_sampler'       : self.initialize_sampler,
            'sample_n'           : self.sample_n,
            'get_state'          : self.get_state,
            'set_state'          : self.set_state,
            'get_accept_prob'    : self.get_accept_prob,
            'write_to_disk'      : self.write_to_disk,
            'complete'           : self.complete,
            'try_swap_state_inf' : self.try_swap_state_inf,
            'try_swap_state_sub' : self.try_swap_state_sub,
            }
        return

    def __init__(self, comm, statmodel):
        self.statmodel = statmodel
        self.build_dispatch()
        pass

class PTMaster(pt.PTMaster):
    def try_swap_states(self, sup_rank, inf_rank):
        return self.chains[sup_rank].try_swap_state_sup(self.chains[inf_rank])

    def temper_chains(self):
        ranks = list(range(len(self.chains)))
        shuffle(ranks)

        swaps = []
        for rank1, rank2 in zip(ranks[::2], ranks[1::2]):
            swaps.append(rank1, rank2, self.try_swap_states(rank_1, rank_2))
        self.swaps.append(swaps)
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

        self.sample_k(ns % k)
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
        sent = self.comm.isend(('pairwise_plot', path), dest = 1)
        recv = self.comm.irecv(source = 1)
        parcel = recv.wait()
        assert parcel
        return

    def get_accept_probability(self, nburn):
        self.comm.send(self.chain.get_acceot_probability(nburn), dest = 0)
        return

    def initialize_chains(self, temperature_ladder, kwargs):
        self.chain_ranks = list(range(1, self.size))
        sent = [
            self.comm.isend(('init_chain', {'temperature' : temp, **kwargs}), dest = rank)
            for temp,  rank in zip(temperature_ladder, self.chain_ranks)
            ]
        recv = [self.comm.irecv(source = rank) for rank in self.chain_ranks]
        assert all([r.wait() for r in recv])
        return

    def write_to_disk(self, path, nburn, thin):
        sent = self.comm.isend(('write_to_disk', path,  nburn, thin), dest = 1)
        recv = self.comm.irecv(source = 1)
        parcel = recv.wait()
        assert parcel
        return

    def __init__(self, comm, temperature_ladder, **kwargs):
        self.comm = comm
        self.size = self.comm.Get_size()
        self.initialize_chains(temperature_ladder, kwargs)
        return

# EOF
