"""Unmodified eval body from LANL Impala commit
6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e (BSD-3-Clause; see LICENSE).
Kept independent of the optimized method as a numerical regression oracle.
Do not update this oracle to make a new implementation pass.
"""

from itertools import cycle

import numpy as np
from scipy.interpolate import interp1d

epsilon = 1e-5  # Frozen baseline padding constant.


def original_material_eval(
    self, parmat, pool=None, nugget=False
):  # note: extra parameters ignored
    """parmat:  dictionary of parameters"""
    if (pool is True) or self.pool:  # Pooled Case
        # nrep = parmat['p'].shape[0]  # number of temper temps
        nrep = next(iter(parmat.values())).shape[0]
        parmat_big = {
            key: np.kron(parm, np.ones(self.nexp))
            for key, parm in parmat.items()
        }
    else:  # hierarchical case
        nrep = (
            next(iter(parmat.values())).shape[0] // self.nexp
        )  # number of temper temps
        parmat_big = parmat

    edots = np.kron(np.ones(nrep), self.edots)  # 1d vector, nexp * temper_temps
    temps = np.kron(np.ones(nrep), self.temps)  # 1d vector, nexp * temper_temps
    strain_maxs = np.kron(
        np.ones(nrep), self.meas_strain_max
    )  # 1d vector, nexp * temper_temps
    ntot = edots.shape[0]  # nexp * temper_temps
    self.model.set_history_variables(strain_maxs, edots, self.Nhist)
    self.model.initialize(parmat_big, self.constants)
    self.model.initialize_state(
        T=temps, stress=np.zeros(ntot), strain=np.zeros(ntot)
    )
    sim_state_histories = self.model.compute_state_history()
    sim_strains = sim_state_histories[:, 1].T  # 2d array: ntot, Nhist
    sim_stresses = sim_state_histories[:, 2].T  # 2d array: ntot, Nhist

    # Expand/Flatten Simulated strains to single vector--ensure no overlap.
    strain_ends = np.hstack((
        0.0,
        np.cumsum(strain_maxs + epsilon)[:-1],
    ))  # 1d vector, ntot
    flattened_sim_strain = np.hstack(  # 1d vector: ntot * Nhist
        [x + y for x, y in zip(sim_strains, strain_ends)]
    )
    # Expand/flatten simulated stress to single vector
    flattened_sim_stress = np.hstack(sim_stresses)
    # Expand/flatten measured strain to single vector, for each parameter.  Use same
    #  Computed strain ends to ensure no overlap
    flattened_strain = (
        np.hstack(  # cycle will repeat through measured strain histories
            [
                x + y
                for x, y in zip(cycle(self.meas_strain_histories), strain_ends)
            ]
        )
    )
    ifunc = interp1d(  # Generate the interpolation function.
        flattened_sim_strain,
        flattened_sim_stress,
        kind="linear",
        assume_sorted=True,
    )
    ypred = ifunc(flattened_strain).reshape(
        nrep, -1
    )  # Interpolate, and output.
    return ypred
