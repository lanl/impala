"""Experimental pooled Cu evaluator with bounded resident experiment geometry.

Prediction-only contract: does not update MaterialModel.state or expose histories.
Uses the same history equations; only stress histories stay on device. Fixed
geometry and interpolation indices are cached; parameters/constants stay live.
"""

import time
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
    def __init__(
        self,
        model,
        cache_bytes=256 * 1024**2,
        block_size=128,
        timing=False,
        hoist=False,
    ):
        from impala.superCal.models_withlik import ModelMaterialStrength

        if type(model) is not ModelMaterialStrength or not model.pool:
            raise ValueError(
                "Resident evaluator supports ordinary pooled Cu only"
            )
        # Reuse the strict physics-combination validation of the history adapter.
        validated = GpuCuHistory(model.model)
        if block_size not in (32, 64, 128, 256):
            raise ValueError("History block size must be 32, 64, 128 or 256")
        self.block_size = block_size
        self.timing = timing
        self.stage_samples = []
        self.cp = validated.cp
        self.model = model
        self.kernel = self.cp.RawKernel(
            Path(__file__).with_name("cu_history.cu").read_text(),
            "cu_history",
            options=("--std=c++11", "--fmad=false", "-DPREDICTION_ONLY")
            + (("-DHOIST_CONSTANTS",) if hoist else ()),
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

        measured = {} if self.timing else None
        if measured is not None:
            started = boundary = time.perf_counter()

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
        if measured is not None:
            measured["validation_s"] = time.perf_counter() - boundary
            boundary = time.perf_counter()
            misses = self.misses
        geometry, n, nh, m, _ = self._geometry(p.shape[1])
        if measured is not None:
            cp.cuda.get_current_stream().synchronize()
            measured["geometry_host_and_upload_s"] = (
                time.perf_counter() - boundary
            )
            measured["geometry_miss"] = self.misses != misses
            boundary = time.perf_counter()
        times, rates, initial, lo, hi, dx, delta = geometry
        expanded = np.repeat(p, self.model.nexp, axis=1)
        device_params, constants = cp.asarray(expanded), cp.asarray(c)
        self.input_bytes += expanded.nbytes + c.nbytes
        if measured is not None:
            cp.cuda.get_current_stream().synchronize()
            measured["parameter_expand_and_upload_s"] = (
                time.perf_counter() - boundary
            )
            boundary = time.perf_counter()
        stress = cp.empty((nh, n), dtype=cp.float64)
        aux = cp.empty((2, n), dtype=cp.float64)
        output = cp.empty(m, dtype=cp.float64)
        if measured is not None:
            measured["allocation_s"] = time.perf_counter() - boundary
            boundary = time.perf_counter()
            start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
            start_event.record()
        self.kernel(
            ((n + self.block_size - 1) // self.block_size,),
            (self.block_size,),
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
        if measured is not None:
            stop_event.record()
            stop_event.synchronize()
            measured["history_device_s"] = (
                cp.cuda.get_elapsed_time(start_event, stop_event) / 1000
            )
            measured["history_launch_wait_s"] = time.perf_counter() - boundary
            boundary = time.perf_counter()
            start_event.record()
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
        if measured is not None:
            stop_event.record()
            stop_event.synchronize()
            measured["interpolation_device_s"] = (
                cp.cuda.get_elapsed_time(start_event, stop_event) / 1000
            )
            measured["interpolation_launch_wait_s"] = (
                time.perf_counter() - boundary
            )
            boundary = time.perf_counter()
        result = cp.asnumpy(output).reshape(p.shape[1], -1)
        if measured is not None:
            measured["download_and_host_allocation_s"] = (
                time.perf_counter() - boundary
            )
            boundary = time.perf_counter()
        self.output_bytes += result.nbytes
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite GPU predictions")
        if measured is not None:
            measured["output_validation_s"] = time.perf_counter() - boundary
            measured["total_s"] = time.perf_counter() - started
            measured["batch"] = p.shape[1]
            self.stage_samples.append(measured)
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


def attach(model, **kwargs):
    adapter = GpuCuPredictions(model, **kwargs)
    model.eval = adapter
    return adapter
