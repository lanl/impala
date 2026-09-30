"""Experimental Cu CUDA adapter, deliberately outside the public library API."""

from pathlib import Path

import numpy as np

PARAMS = (
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
CONSTS = (
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


class GpuCuHistory:
    """Copy all inputs/outputs each call; timings include those transfers.

    No persistent grid cache is used, so edited conditions remain visible.
    This is an experimental adapter for the documented Cu configuration only.
    """

    def __init__(self, model):
        import cupy as cp

        from impala.physics import physical_models_vec as pm

        expected = (
            pm.PTW_Yield_Stress,
            pm.Quadratic_Specific_Heat,
            pm.Linear_Density,
            pm.Linear_Melt_Temperature,
            pm.BGP_PW_Shear_Modulus,
        )
        actual = (
            model.flow_stress,
            model.specific_heat,
            model.density,
            model.melt_model,
            model.shear_modulus,
        )
        if type(model) is not pm.MaterialModel or any(
            type(a) is not e for a, e in zip(actual, expected)
        ):
            raise ValueError(
                "GPU prototype supports only the Cu PTW model combination"
            )
        if "update_state" in model.__dict__ or any(
            "value" in a.__dict__ for a in actual
        ):
            raise ValueError(
                "GPU prototype does not support overridden methods"
            )
        self.model = model
        self.cp = cp
        self.kernel = cp.RawKernel(
            (Path(__file__).with_name("cu_history.cu")).read_text(),
            "cu_history",
            options=("--std=c++11", "--fmad=false"),
        )
        self.kernel_ms = None
        self.measure_kernel = False

    def __call__(self, strain_history=None):
        from impala.physics import ConstraintError, functions

        cp = self.cp
        model = self.model
        h = model.strain_history if strain_history is None else strain_history
        times, rates = np.asarray(h["times"]), np.asarray(h["strain_rate"])
        if times.ndim != 2 or times.shape[1] < 2 or times.shape[0] == 0:
            raise ValueError("GPU prototype requires nonempty 2D histories")
        n, nh = times.shape
        if rates.shape != (n, nh - 1) or h["strains"].shape != (n, nh):
            raise ValueError("Mismatched GPU history shapes")
        state = model.state
        if any(
            not isinstance(a, np.ndarray)
            or a.shape != (n,)
            or a.dtype != np.float64
            for a in (state.T, state.strain, state.stress)
        ):
            raise ValueError("GPU prototype requires float64 vector states")
        initial = np.stack((state.T, state.strain, state.stress))
        params = np.stack([
            np.broadcast_to(
                np.asarray(getattr(model.parameters, k), dtype=float), (n,)
            )
            for k in PARAMS
        ])
        constants = np.array(
            [getattr(model.parameters, k) for k in CONSTS], dtype=float
        )
        if constants.shape != (16,):
            raise ValueError("GPU prototype requires scalar constants")
        if not all(
            np.isfinite(a).all()
            for a in (times, rates, initial, params, constants)
        ):
            raise ValueError("GPU prototype requires finite inputs")
        if not np.all(
            functions.PTW_goodparam(
                params[2],
                params[4],
                params[7],
                params[8],
                params[9],
                params[10],
                params[3],
            )
        ):
            raise ConstraintError("PTW bad val")
        device = [
            cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
            for a in (times.T, rates.T, initial, params, constants)
        ]
        output = cp.empty((nh, 6, n), dtype=cp.float64)
        aux = cp.empty((2, n), dtype=cp.float64)
        if self.measure_kernel:
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
        self.kernel(
            ((n + 127) // 128,),
            (128,),
            (
                *device,
                np.float64(functions.cbrtAvogadro),
                np.int32(n),
                np.int32(nh),
                output,
                aux,
            ),
        )
        if self.measure_kernel:
            stop.record()
        history, aux_host = cp.asnumpy(output), cp.asnumpy(aux)
        if self.measure_kernel:
            stop.synchronize()
            self.kernel_ms = cp.cuda.get_elapsed_time(start, stop)
        if not np.isfinite(history).all() or not np.isfinite(aux_host).all():
            raise FloatingPointError("Nonfinite GPU material state")
        state.strain = history[-1, 1].copy()
        state.stress = history[-1, 2].copy()
        state.T = history[-1, 3].copy()
        state.G = history[-1, 4].copy()
        state.rho = history[-1, 5].copy()
        state.Cv = aux_host[0].copy()
        state.Tmelt = aux_host[1].copy()
        return history


def attach(model):
    """Attach only to a disposable experimental model; returns timing adapter."""
    adapter = GpuCuHistory(model)
    model.compute_state_history = adapter
    return adapter
