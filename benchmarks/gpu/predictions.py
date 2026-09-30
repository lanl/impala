"""Experimental pooled Cu evaluator with bounded resident experiment geometry.

Prediction-only contract: does not update MaterialModel.state or expose histories.
Uses the same history equations; only stress histories stay on device. Fixed
geometry and interpolation indices are cached; parameters/constants stay live.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .history import CONSTS, PARAMS, GpuCuHistory

INTERPOLATE = r"""
extern "C" __global__ void interpolate(
    const double* stress, const long long* lo, const long long* hi,
    const double* dx, const double* delta, const int m, double* out) {
    const int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=m) return;
    const double a=stress[lo[j]], b=stress[hi[j]];
    out[j]=delta[j]==0.0 ? a : a+((b-a)/dx[j])*delta[j];
}
"""


class GpuCuPredictions:
    def __init__(self, model, cache_bytes=256 * 1024**2):
        from impala.superCal.models_withlik import ModelMaterialStrength

        if type(model) is not ModelMaterialStrength or not model.pool:
            raise ValueError(
                "Resident evaluator supports ordinary pooled Cu only"
            )
        # Reuse the strict physics-combination validation of the history adapter.
        validated = GpuCuHistory(model.model)
        self.cp = validated.cp
        self.model = model
        self.kernel = self.cp.RawKernel(
            Path(__file__).with_name("cu_history.cu").read_text(),
            "cu_history",
            options=("--std=c++11", "--fmad=false", "-DPREDICTION_ONLY"),
        )
        self.interpolate = self.cp.RawKernel(
            INTERPOLATE, "interpolate", options=("--fmad=false",)
        )
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.resident_bytes = 0
        self.signature = None
        self.hits = self.misses = 0
        self.input_bytes = self.output_bytes = 0

    def _geometry(self, batch):
        from impala.superCal.models_withlik import epsilon

        model, cp = self.model, self.cp
        signature = (
            model.nexp,
            model.Nhist,
            epsilon,
            *[
                (
                    np.asarray(a).shape,
                    np.asarray(a).dtype.str,
                    np.asarray(a).tobytes(),
                )
                for a in (
                    model.edots,
                    model.temps,
                    model.meas_strain_max,
                    *model.meas_strain_histories,
                )
            ],
        )
        if signature != self.signature:
            self.cache.clear()
            self.resident_bytes = 0
            self.signature = signature
        if batch in self.cache:
            self.hits += 1
            self.cache.move_to_end(batch)
            return self.cache[batch]
        self.misses += 1
        prepared = model._prepare_experiments(batch)
        h = prepared["history"]
        times, rates = h["times"], h["strain_rate"]
        n, nh = times.shape
        if nh < 2 or not all(
            np.isfinite(a).all()
            for a in (times, rates, prepared["conditions"][1])
        ):
            raise ValueError("Resident evaluator requires finite histories")
        # Match the original cumulative arithmetic, including flattened offsets.
        strain = np.zeros((n, nh))
        for i in range(1, nh):
            strain[:, i] = strain[:, i - 1] + rates[:, i - 1] * (
                times[:, i] - times[:, i - 1]
            )
        x = (strain + prepared["strain_ends"][:, None]).ravel()
        query = prepared["flattened_strain"]
        if (
            not np.all(np.diff(x) > 0)
            or query.min() < x[0]
            or query.max() > x[-1]
        ):
            raise ValueError(
                "Resident evaluator requires ordered in-range strain"
            )
        lo = np.minimum(np.searchsorted(x, query, side="right") - 1, x.size - 2)
        hi = lo + 1
        # Original flattened order is history-major, device stress is step-major.
        device_lo = (lo % nh) * n + lo // nh
        device_hi = (hi % nh) * n + hi // nh
        initial = np.stack((
            prepared["conditions"][1],
            np.zeros(n),
            np.zeros(n),
        ))
        host = (
            times.T,
            rates.T,
            initial,
            device_lo.astype(np.int64),
            device_hi.astype(np.int64),
            x[hi] - x[lo],
            query - x[lo],
        )
        arrays = tuple(cp.asarray(np.ascontiguousarray(a)) for a in host)
        size = sum(a.nbytes for a in arrays)
        self.input_bytes += size
        item = (arrays, n, nh, len(query), size)
        while self.cache and self.resident_bytes + size > self.cache_bytes:
            _, removed = self.cache.popitem(last=False)
            self.resident_bytes -= removed[-1]
        if size <= self.cache_bytes:
            self.cache[batch] = item
            self.resident_bytes += size
        return item

    def __call__(self, params, pool=None, nugget=False):
        from impala.physics import ConstraintError, functions

        if pool is False or nugget:
            raise ValueError(
                "Resident evaluator supports plain pooled predictions"
            )
        if not params or set(params) - set(PARAMS):
            raise ValueError("Unexpected Cu parameter names")
        values = [np.asarray(v, dtype=np.float64) for v in params.values()]
        if (
            any(a.ndim != 1 for a in values)
            or not values[0].size
            or any(a.shape != values[0].shape for a in values)
        ):
            raise ValueError("Invalid parameter batch shapes")
        p = np.stack([
            np.broadcast_to(
                np.asarray(
                    params[k] if k in params else self.model.constants[k],
                    dtype=np.float64,
                ),
                values[0].shape,
            )
            for k in PARAMS
        ])
        c = np.array([self.model.constants[k] for k in CONSTS], dtype=float)
        if (
            c.shape != (16,)
            or not np.isfinite(c).all()
            or not np.isfinite(p).all()
        ):
            raise ValueError(
                "Resident evaluator requires finite scalar constants/parameters"
            )
        if not np.all(
            functions.PTW_goodparam(p[2], p[4], p[7], p[8], p[9], p[10], p[3])
        ):
            raise ConstraintError("PTW bad val")
        cp = self.cp
        geometry, n, nh, m, _ = self._geometry(p.shape[1])
        times, rates, initial, lo, hi, dx, delta = geometry
        expanded = np.repeat(p, self.model.nexp, axis=1)
        device_params, constants = cp.asarray(expanded), cp.asarray(c)
        self.input_bytes += expanded.nbytes + c.nbytes
        stress = cp.empty((nh, n), dtype=cp.float64)
        aux = cp.empty((2, n), dtype=cp.float64)
        output = cp.empty(m, dtype=cp.float64)
        self.kernel(
            ((n + 127) // 128,),
            (128,),
            (
                times,
                rates,
                initial,
                device_params,
                constants,
                np.float64(functions.cbrtAvogadro),
                np.int32(n),
                np.int32(nh),
                stress,
                aux,
            ),
        )
        self.interpolate(
            ((m + 127) // 128,),
            (128,),
            (
                stress,
                lo,
                hi,
                dx,
                delta,
                np.int32(m),
                output,
            ),
        )
        result = cp.asnumpy(output).reshape(p.shape[1], -1)
        self.output_bytes += result.nbytes
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite GPU predictions")
        return result

    def statistics(self):
        return {
            k: getattr(self, k)
            for k in (
                "hits",
                "misses",
                "resident_bytes",
                "cache_bytes",
                "input_bytes",
                "output_bytes",
            )
        }


def attach(model):
    adapter = GpuCuPredictions(model)
    model.eval = adapter
    return adapter
