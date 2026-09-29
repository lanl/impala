"""Optional compiled history loop for the Cu PTW model combination.

Reuse existing numerical functions without fast-math or reduced precision.
The Python driver remains the default and the regression control.
"""

import numpy as np
from numba import njit

from . import physical_models_functions as f


@njit
def _cu_history(times, rates, T, strain, stress, params, consts):
    theta, p, s0, beta, sInf, kappa, lgamma, y0, yInf, y1, y2 = params
    (
        chi,
        c0,
        c1,
        c2,
        r0,
        r1,
        tm0,
        tm1,
        G0,
        rho_0,
        gamma_1,
        gamma_2,
        q2,
        alpha,
        rho0,
        matomic,
    ) = consts
    nrep, nhist = times.shape
    results = np.empty((nhist, 6, nrep))
    for i in range(nhist):
        edot = rates[:, max(0, i - 1)]
        dt = np.zeros(nrep) if i == 0 else times[:, i] - times[:, i - 1]
        Cv = f.Cubic_Specific_Heat(c0, c1, c2, 0, T)
        rho = f.Cubic_Density(r0, r1, 0, 0, T)
        T = T + (edot > 1.0e-6) * chi * stress * edot * dt / (Cv * rho)
        strain = strain + edot * dt
        Tmelt = f.Cubic_Melt_Temperature(tm0, tm1, 0, 0, rho)
        G = f.BGP_PW_Shear_Modulus(
            G0, rho_0, gamma_1, gamma_2, q2, alpha, rho, T, Tmelt
        )
        stress = f.PTW_Yield_Stress(
            p,
            kappa,
            s0,
            sInf,
            y0,
            yInf,
            y1,
            y2,
            beta,
            theta,
            lgamma,
            edot,
            rho0,
            matomic,
            G,
            strain,
            T,
            Tmelt,
            1.0e-10,
        )
        results[i, 0] = times[:, i]
        results[i, 1] = strain
        results[i, 2] = stress
        results[i, 3] = T
        results[i, 4] = G
        results[i, 5] = rho
    return results, T, strain, stress, Cv, rho, Tmelt, G


def compute_cu_history(model, history):
    """Validate the supported configuration, run the kernel, restore final state."""
    from . import physical_models_vec as pm

    components = (
        (model, pm.MaterialModel),
        (model.state, pm.MaterialState),
        (model.flow_stress, pm.PTW_Yield_Stress),
        (model.specific_heat, pm.Quadratic_Specific_Heat),
        (model.density, pm.Linear_Density),
        (model.melt_model, pm.Linear_Melt_Temperature),
        (model.shear_modulus, pm.BGP_PW_Shear_Modulus),
    )
    if any(type(obj) is not cls for obj, cls in components):
        raise ValueError(
            "numba history backend supports only the Cu PTW model combination"
        )
    if "update_state" in model.__dict__ or any(
        "value" in obj.__dict__ for obj, _ in components[2:]
    ):
        raise ValueError(
            "numba history backend does not support overridden model methods"
        )
    times, rates, strains = (
        history[k] for k in ("times", "strain_rate", "strains")
    )
    if (
        times.ndim != 2
        or times.shape[0] == 0
        or times.shape[1] < 2
        or strains.shape != times.shape
        or rates.shape != (times.shape[0], times.shape[1] - 1)
    ):
        raise ValueError(
            "numba history backend requires matching nonempty history arrays"
        )
    n = times.shape[0]
    state = model.state
    for array in (times, rates, strains, state.T, state.strain, state.stress):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise ValueError("numba history backend requires float64 arrays")
        if not np.all(np.isfinite(array)):
            raise ValueError("numba history backend requires finite inputs")
    if any(x.shape != (n,) for x in (state.T, state.strain, state.stress)):
        raise ValueError("numba history backend requires one state per history")
    mp = model.parameters
    names = (
        "theta",
        "p",
        "s0",
        "beta",
        "sInf",
        "kappa",
        "lgamma",
        "y0",
        "yInf",
        "y1",
        "y2",
    )
    params = tuple(
        np.broadcast_to(np.asarray(getattr(mp, k), dtype=float), (n,)).copy()
        for k in names
    )
    const_names = (
        "chi",
        "c0",
        "c1",
        "c2",
        "r0",
        "r1",
        "tm0",
        "tm1",
        "G0",
        "rho_0",
        "gamma_1",
        "gamma_2",
        "q2",
        "alpha",
        "rho0",
        "matomic",
    )
    constants = tuple(getattr(mp, k) for k in const_names)
    if any(np.ndim(x) != 0 or not np.isfinite(x) for x in constants):
        raise ValueError(
            "numba history backend requires finite scalar constants"
        )
    constants = tuple(float(x) for x in constants)
    if not all(np.all(np.isfinite(x)) for x in params):
        raise ValueError("numba history backend requires finite parameters")
    if not np.all(
        f.PTW_goodparam(
            params[2],
            params[4],
            params[7],
            params[8],
            params[9],
            params[10],
            params[3],
        )
    ):
        raise pm.ConstraintError("PTW bad val")
    result = _cu_history(
        times, rates, state.T, state.strain, state.stress, params, constants
    )
    if not all(np.all(np.isfinite(x)) for x in result):
        raise FloatingPointError("nonfinite state in numba material history")
    for name, value in zip(
        ("T", "strain", "stress", "Cv", "rho", "Tmelt", "G"), result[1:]
    ):
        setattr(state, name, value)
    return result[0]
