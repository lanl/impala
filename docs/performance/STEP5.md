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

## Completed short-run and regression checks

Measured implementation: `ac3392f`. The three rounds, in seconds:

| Backend | Round 1 | Round 2 | Round 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Parallel CPU | 18.775 | 21.072 | 19.250 | 19.250 |
| Previous batched GPU | 24.292 | 24.466 | 25.045 | 24.466 |
| Resident GPU | 14.730 | 15.087 | 14.935 | 14.935 |

Median resident-GPU throughput is 1.289x parallel CPU and 1.638x the previous
batched GPU. All eight comparisons with the first CPU round passed exact
all-temperature parameter/counter checks and numerical noise/prediction checks.
Maximum all-temperature noise discrepancy was 1.475e-16; CPU repetitions were
exactly identical. Full hardware-enabled suite: **98 passed in 106.07 s**.
The generated test plot is kept with external artifacts rather than in Git.

The resident sampling profile used 5.08 s in the coordinator and 3.28 s in grouped
evaluation, versus 8.47/6.60 s before. For the 1,000-state profile the old path's
sampling history output requires 5.225 GB of device-to-host payload (sum of batch
histories times `(6 * 100 + 2) * 8` bytes). The new path recorded 0.474 GB of
prediction output including fixture warm-up: approximately 11x less returned
data. These are array payloads, not measured PCIe bandwidth or pure transfer time.
The resident geometry cache recorded 988 hits and 103 misses including warm-up.

The CPU control is the existing compiled backend. This does not establish a
win over a future CPU prediction-only implementation that also avoids redundant
preparation/interpolation work. Resident GPU mode still uses the same synchronous
chain coordinator; further scheduling changes are not part of this checkpoint.

## Rebuild the report

After the quality study completes, capture the full-suite test output as
`ARTIFACTS/pytest.log` and preserve the directory names produced by the suite.
Run the following from the repository root:

```bash
.venv/bin/python benchmarks/gpu_resident_report.py \
  --artifacts ../benchmark/changes/step5 \
  --original ../benchmark/results/reference
XDG_CACHE_HOME=/tmp/impala-tex-cache ../benchmark/tools/tectonic \
  --keep-logs --outdir ../benchmark/changes/step5 \
  docs/performance/results/step5.tex
```

The report builder reads measurements rather than embedding timing outcomes.
The LaTeX source and compact JSON/CSV results are versioned; the rendered PDF and
large raw traces remain in the workspace with SHA-256 hashes recorded in Git.

## Long-run result and decision

Fresh four-chain, 40,000-state, 50-temperature sampling took **214.318 s on
resident GPU versus 252.157 s on parallel CPU**: 15.0% less elapsed time and
**17.7% higher effective samples per second**. The long GPU cache recorded
43,556 hits and 379 misses including fixture warm-up (99.1% hit rate).

All predeclared posterior-quality gates passed. Maximum R-hat: 1.002585; minimum
bulk ESS: 2253.48; minimum tail ESS: 3345.25. Minimum bulk ESS/s increased from
8.937 to 10.515. GPU parameters and all recorded counters matched CPU exactly;
maximum noise-array error was 6.29e-18 and prediction error 4.91e-15. The fresh
CPU theta, noise, acceptance and prediction arrays reproduced the frozen original
saved reference exactly. The full long-array comparison also passed the declared
noise/prediction tolerances.

This checkpoint establishes a measured GPU advantage for this pooled Cu workload.
Keep it experimental and opt-in through `--backend gpu_resident`. Production
integration, automatic fallback, Mac testing and hierarchical sampler changes
remain deferred. No algorithmic mixing changes were made. Further GPU tuning
would need evidence on the remaining kernel/synchronization costs; it is not
included in these results.

Git checkpoint: `perf-step5-20260930`, branch `perf/gpu-resident-evaluation`.
Measured numerical/driver code is `ac3392f`; the final documentation commit adds
the report builder, compact results, diagnostics, profiles and artifact hashes.
All commits are local; nothing was pushed upstream.

After rebuilding the PDF, regenerate and verify the artifact manifest:

```bash
.venv/bin/python benchmarks/gpu_resident_manifest.py \
  --artifacts ../benchmark/changes/step5
```

This checks all short/long measured source hashes against the current files and
hashes raw outputs, report files, benchmark/library source, fixtures, Cu inputs
and frozen original reference traces. Development profiles precede the code
checkpoint and are retained as diagnostic evidence rather than timing gates.
