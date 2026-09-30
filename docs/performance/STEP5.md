# Step 5: keep experiment geometry and interpolation on the GPU

Branch: `perf/gpu-resident-evaluation`, based on `perf-step4-20260930`.
Scope: the same pooled Cu calibration, with unchanged sampler, RNG, likelihood,
adaptation and four independent chains. Production library defaults, cleanup,
Mac support and hierarchical mixing changes remain outside this experiment.

## Hypothesis and measured baseline

A sampling-only cProfile run of four 1,000-state, 50-temperature chains measured
8.47 s in the coordinator, 6.60 s in grouped model evaluation, 3.67 s in the
history adapter and 2.85 s in device-to-host array reads. The latter includes
waiting for asynchronous kernel execution: it is **not pure transfer time**.
Host preparation, flattening and interpolation also consume substantial time.
Profiling perturbs timing; performance claims use separate unprofiled runs.

The initial attempt using `python -m cProfile` failed at multiprocessing spawn
because the worker could not be pickled from that entry point. No sampling ran.
The driver now has `--profile`, profiling only parent sampling coordination after
worker startup. Both the failed log and successful profile are retained.

## Implementation and boundaries

`benchmarks/gpu/predictions.py` provides a separate prediction-only adapter:

- Cache fixed experiment histories, starting conditions and interpolation indices
  on the device. Keys include batch size and content signatures for public
  experiment arrays; edits invalidate the cache. Evict least-recently-used
  geometries above a 256 MiB logical-array budget. CuPy's allocator may separately
  retain freed memory; that budget is not a total GPU-memory limit.
- Use the same CUDA history equations through a compile-time output selection.
  The existing full-history mode remains unchanged. The resident mode writes
  stress histories only, keeping them on-device.
- Precompute interpolation locations using the original cumulative strain and
  flattened experiment offsets. Interpolate stresses on the GPU, with float64
  arithmetic and fused multiply-add disabled as in step 3.
- Send live parameters/constants each call and return only predictions at measured
  strain points. Keep the original interprocess coordinator and independent RNGs.

The adapter is explicitly restricted to ordinary pooled Cu predictions. It does
not update `MaterialModel.state`, return complete histories or support continuation
of a material state. Those use cases still belong to the existing full-history
adapter. The public library is unchanged. First calls for unseen geometries count
in sampling timing; only the existing fixture warm-up precedes timing.

Initial tests caught a fixed-parameter handling mistake: beta is supplied through
constants in the Cu configuration. The adapter now accepts supported parameters
from either the parameter batch or fixed constants, matching this reference
configuration. The failed test log is retained; no fixture/tolerance was changed.

## Predeclared validation and timing

Use the unchanged numerical gates: predictions `rtol=1e-8, atol=1e-10`, weighted
log likelihood absolute error <=1e-6, noise arrays `rtol=1e-8, atol=1e-15`.
Tests cover frozen predictions/likelihoods, altered parameters/constants and
experiment conditions, unequal batch sizes, cache eviction, rate/p thresholds,
and invalid input. Full history regressions must still pass.

Three short rounds compare parallel CPU, old batched GPU and resident GPU. Rotate
backend order across rounds; run jobs sequentially. Each job has four seeds
42/123/456/789, 3,000 states and 50 temperatures. Save all temperature-chain
outputs. Check exact parameters/counters and numerical noise/prediction agreement.
Report individual and median group wall times; three runs do not establish a
hardware-independent timing guarantee.

Then compare fresh parallel CPU and resident GPU: four chains, 40,000 states,
50 temperatures. Discard 20,000, no thinning. All 16 scalar parameter/noise
quantities must have R-hat <1.01, bulk/tail ESS >=400 and MCSE/SD <0.05.
Mean differences must be <=4 combined MCSEs and <=0.1 pooled SD; prediction mean
differences <=0.1 experiment noise SD. ESS/s uses elapsed time for the entire
four-chain group. Compare against the frozen original outputs as well.

## Reproduction

Use the existing locked CPU environment plus `benchmarks/gpu/requirements.txt`
and NVIDIA device access. Every output directory must be new. Example commands
from the repository root (replace `NEW_*` with new output directories):

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_batched_chains.py --backend gpu_batched --profile \
  --iterations 1000 --temperatures 50 --output NEW_BASELINE_PROFILE
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_resident_suite.py --stage profile --output NEW_PROFILE
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_resident_suite.py --stage short --output NEW_SHORT
CUPY_CACHE_DIR=/tmp/impala-cupy-cache IMPALA_TEST_GPU=1 \
  .venv/bin/python -m pytest -q
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_resident_suite.py --stage long --output NEW_LONG
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/impala-matplotlib \
  ../benchmark/.venv/bin/python benchmarks/gpu_batched_quality.py \
  --reference NEW_LONG/cpu_parallel_long \
  --candidate NEW_LONG/gpu_resident_long --output NEW_QUALITY
```

The suite records exact driver arguments through source, commit/source hashes,
pre-job GPU telemetry and raw logs. Driver metadata records group wall time,
startup, CPU affinity and resident-cache counters/transfer-byte totals (including
fixture warm-up, excluding posterior postprocessing). CPU library threads are
fixed at one per process. See step 3 for pinned GPU versions and hardware.
