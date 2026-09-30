# Step 8: batch proposal likelihoods across temperatures

Branch `perf/gpu-proposal-likelihood`, based on `perf-step7-20260930` (`d3385fe`).
Author: Ayan Biswas <ayanju04@gmail.com>.

## Profile and hypothesis

A fresh step-7 GPU profile (four chains, 3,000 states, 50 temperatures) records
164,400 scalar likelihood calls for seed 42: 0.539 s cumulative, out of 3.040 s
worker process CPU and 11.582 s instrumented wall time. Remote requests take
9.067 s wall but 0.534 s process CPU. Remote waiting includes useful GPU work,
other workers and scheduling; it is not an estimate of avoidable IPC overhead.

Hypothesis: batch likelihood calculations for ordinary and decorrelation
parameter proposals across temperatures to reduce scalar Python calls. Keep
Gibbs batching enabled in both controls and candidates. This does not change
proposal generation, acceptance, RNG consumption, adaptation, swaps or GPU math.

## Implementation and safeguards

Private `setup._batch_pooled_proposals` requires the guarded step-7 Gibbs path.
The benchmark exposes `--worker-batch-proposals` alongside `--worker-batch-gibbs`.
The existing model restrictions and default scalar behavior are unchanged.
`DiagonalGibbsBatch.proposal_likelihood` reads the current covariance dictionaries
on every call, including changes due to temperature swaps or new Gibbs draws.
No covariance cache can become stale. C-contiguous residuals preserve the scalar
per-observation reduction order even for Fortran or strided input arrays.
Calculations still include all temperatures when any proposal satisfies constraints,
matching existing behavior. Decorrelation parameter proposals remain sequential.

Nine new tests compare exact scalar/batched results with swapped and replaced
covariances at 1/8/50 temperatures and C/Fortran/strided layouts; also verify that
no global RNG state changes. Combined with step-7 focused tests: 27 passed.

## Predeclared validation

Run fresh profiled before/after controls; compare saved full-temperature arrays.
Then three sequential short rounds of four backends (CPU/GPU with scalar/batched
proposal likelihoods), four seeds 42/123/456/789, 3,000 states, 50 temperatures.
Reverse round 2 and rotate round 3. Require exact parameters/counters and unchanged
noise tolerances (rtol 1e-8, atol 1e-15), predictions (rtol 1e-8, atol 1e-10).
Run the full GPU-enabled regression suite before long sampling.

Long controls: previous GPU proposal path, candidate GPU, equally optimized CPU;
four 40,000-state chains per backend, 50 temperatures. Discard 20,000 states,
no thinning. All 16 parameter/noise quantities require R-hat <1.01, bulk/tail
ESS >=400 and MCSE/SD <0.05. Posterior means must agree within four combined
MCSEs and 0.1 pooled SD; prediction means within 0.1 experiment noise SD.
Also compare optimized CPU traces with frozen original traces independently.
Use group wall time for ESS/s, not summed worker time. Startup/JIT and saving
are excluded; CPU work, initialization inside sampling, transfers and coordination
remain included. Short repetitions support long timings (one run per backend).

## Reproduction

Use existing pinned `.venv`, distinct output paths, and no overlapping workloads:

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_proposal_suite.py --stage profile --output NEW_PROFILE
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_proposal_suite.py --stage short --output NEW_SHORT
CUPY_CACHE_DIR=/tmp/impala-cupy-cache IMPALA_TEST_GPU=1 \
  .venv/bin/python -m pytest -q
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_proposal_suite.py --stage long --output NEW_LONG
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/impala-matplotlib \
  ../benchmark/.venv/bin/python benchmarks/gpu_batched_quality.py \
  --reference NEW_LONG/cpu_candidate --candidate NEW_LONG/gpu_candidate \
  --output NEW_QUALITY
```

Raw artifacts are in workspace `benchmark/changes/step8/`. Initial profiles use
`profile-before`/`profile-after`, corresponding to suite `gpu_control`/`gpu_candidate`.
Those initial profiles record their own source hashes before the code checkpoint.
All timing runs will use committed code. Hierarchical mixing, Mac validation,
asynchronous scheduling and public backend cleanup remain separate.
