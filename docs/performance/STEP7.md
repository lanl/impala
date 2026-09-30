# Step 7: profile workers and batch independent Gibbs bookkeeping

Branch: `perf/gpu-worker-overhead`, based on `perf-step6-20260930` (`ad741ae`).
Author: Ayan Biswas <ayanju04@gmail.com>. Pooled Cu only; hierarchical mixing,
Mac testing and public-backend cleanup remain separate. This is an implementation
change to the existing Gibbs step, not a change to its distributions or adaptation.

## Evidence before optimization

`--worker-profile` adds cProfile inside each worker, process CPU time, and both
wall and CPU time spent in remote model requests. Profiles are written only after
all sampling finishes. Normal sampling metadata also records process CPU time.
Worker CPU times are per process; their sum is not elapsed time. Remote wall time
includes GPU evaluation, other workers, parent scheduling and communication.
It must not be interpreted as avoidable IPC overhead.

The initial seed-42 3,000-state profile recorded 14.558 s wall and 5.692 s process
CPU. Remote requests used 9.365 s wall but just 0.513 s CPU. Likelihood evaluation
was called 314,350 times (1.094 s cumulative), and covariance preparation 150,000
times (0.485 s). Much of the sampler's own 2.439 s involved the temperature loops.
This points to worker computation before asynchronous scheduling or shared memory.

## Selected implementation

A private, opt-in `setup._batch_pooled_gibbs` path in `calibPool` delegates the
ordinary Gibbs noise update to `DiagonalGibbsBatch`. The benchmark exposes it as
`--worker-batch-gibbs`; the default scalar path is unchanged.

- Precompute fixed gamma shape parameters once per run using the same formula.
- Draw the independent temperatures/errors in one NumPy gamma call. Explicit
  C-contiguous inputs preserve the original temperature-then-error traversal and
  random-number consumption. There is no new RNG, reordering or GPU randomness.
- Batch log variances, diagonal inverse covariance/log determinants and likelihood
  arithmetic. Keep the original expression order and contiguous per-observation
  reductions. Return the same covariance-list representation used by tempering.
- Preserve proposal generation, acceptance, adaptation, decorrelation, swaps,
  priors, residual sum-of-squares calculation, history grid and GPU configuration.

The guard accepts the ordinary deterministic `ModelMaterialStrength` diagonal
likelihood with untruncated Gibbs noise and no discrepancy. Custom likelihood,
covariance or step overrides, stochastic models, discrepancy and other noise
updates are rejected when explicitly requesting this path. No generic batching
claim is made for unsupported models. No automatic production default changes.

Direct tests compare noise draws, every covariance and likelihood, and the complete
NumPy RNG state exactly against the scalar loop for four seeds and 1/8/50
temperatures. Unsupported model variants are checked explicitly. Initial tests
passed, followed by four profiled 3,000-state chains: all saved arrays, including
all-temperature noise/parameters and predictions, matched the prior GPU path
bit-for-bit. Seed-42 process CPU time fell from 5.692 to 3.033 s; instrumented
wall time from 14.558 to 11.572 s. Profiled timings are diagnostic, not speed gates.

## Predeclared validation and timing

Keep the original 256-case fixture and all earlier tolerance gates. Three short
rounds compare old/new worker paths on both parallel CPU and the tuned GPU, each
with four seeds 42/123/456/789, 3,000 states and 50 temperatures. Reverse the
second round and rotate the third. Save all-temperature outputs; require exact
parameters/counters, noise `rtol=1e-8, atol=1e-15`, predictions
`rtol=1e-8, atol=1e-10`. Run GPU-enabled full regressions before long sampling.

Long jobs: prior-worker GPU, batched-worker GPU, then batched-worker parallel CPU,
each with four 40,000-state chains and 50 temperatures. The CPU long control gets
the same worker optimization, so any GPU comparison is against the improved CPU.
The unoptimized CPU worker path has fresh repeated short controls; its earlier
long timings are historical and are not used for a fresh long speedup claim.

Discard 20,000 states, no thinning. All 16 parameter/noise quantities must satisfy
R-hat <1.01, bulk/tail ESS >=400, MCSE/SD <0.05. Mean differences <=4 combined
MCSEs and <=0.1 pooled SD; prediction-mean differences <=0.1 experiment noise SD.
Compare optimized CPU arrays independently with frozen original reference outputs.
ESS/s uses group wall time. All timing jobs run sequentially with one numerical
thread per worker. Startup/JIT and postprocessing are excluded, while preparation,
transfers, worker computation and coordination remain included.

## Reproduction

Use the existing pinned numerical/GPU environment and new output directories.

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_worker_suite.py --stage profile --output NEW_PROFILE
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_worker_suite.py --stage short --output NEW_SHORT
CUPY_CACHE_DIR=/tmp/impala-cupy-cache IMPALA_TEST_GPU=1 \
  .venv/bin/python -m pytest -q
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_worker_suite.py --stage long --output NEW_LONG
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/impala-matplotlib \
  ../benchmark/.venv/bin/python benchmarks/gpu_batched_quality.py \
  --reference NEW_LONG/cpu_candidate --candidate NEW_LONG/gpu_candidate \
  --output NEW_QUALITY
```

The suite saves exact driver settings, source hashes (including the sampler and
Gibbs helper), GPU telemetry and raw outputs. Initial before/after profiles are
retained under workspace `benchmark/changes/step7/profile-before` and
`profile-after`; their development source hashes precede the final code checkpoint.

## Completed short checks and full regressions

Measured implementation: `d6a53cc`. Direct Gibbs/RNG/unsupported-input tests:
18 passed in 1.14 s. Full hardware-enabled suite: **126 passed in 105.55 s**.
All eleven short comparisons with the first scalar CPU round passed exact
all-temperature parameter/counter checks and the declared numerical gates.

| Backend | Round 1 (s) | Round 2 (s) | Round 3 (s) | Median (s) |
| --- | ---: | ---: | ---: | ---: |
| CPU, scalar workers | 19.122 | 18.846 | 21.867 | 19.122 |
| CPU, batched workers | 15.588 | 16.612 | 15.823 | 15.823 |
| GPU, scalar workers | 13.111 | 13.437 | 13.581 | 13.437 |
| GPU, batched workers | 11.126 | 11.120 | 11.124 | 11.124 |

Median short-run throughput improves by 1.209x on CPU and 1.208x on GPU. The
variation in CPU controls is retained rather than hidden by combining historical
timings. Only the separate long-run measurements support the final ESS/s claim.

## Rebuild the report and inventory

Keep the artifact subdirectories `profile-before`, `profile-after`, `short`,
`long`, `quality`, and the files `profile-agreement.json`, `pytest.log`.
`profile-before`/`profile-after` correspond to the profile suite's `gpu_control`/
`gpu_candidate`; its agreement file contains the same per-seed comparison nested
under `gpu_candidate`. The recorded artifact layout retains the initial direct
profile calls. A direct profile comparison can be regenerated with the shared
`gpu_resident_suite.compare` function.

```bash
.venv/bin/python benchmarks/gpu_worker_report.py \
  --artifacts ../benchmark/changes/step7 \
  --original ../benchmark/results/reference
XDG_CACHE_HOME=/tmp/impala-tex-cache ../benchmark/tools/tectonic \
  --keep-logs --outdir ../benchmark/changes/step7 \
  docs/performance/results/step7.tex
.venv/bin/python benchmarks/gpu_worker_manifest.py \
  --artifacts ../benchmark/changes/step7
```

The manifest checks measured short/long source hashes and records all raw outputs,
benchmark/library source, tests, fixtures, Cu inputs and frozen original traces.

## Final long-run result

Four chains, 40,000 states each, 50 temperatures; elapsed group sampling time:

| Backend | Seconds |
| --- | ---: |
| Prior GPU workers (step-6 configuration) | 196.576 |
| GPU with batched Gibbs workers | 157.594 |
| CPU with the same batched Gibbs workers | 217.575 |

The new GPU implementation takes 19.83% less time and delivers 1.247x throughput
versus the prior GPU workers. It delivers 1.381x throughput versus the equally
optimized CPU. These are fresh paired long controls, one run per backend; the
three short repetitions above provide supporting timing evidence.

All saved GPU arrays match the prior GPU worker path bit-for-bit for all four
seeds. The optimized CPU parameter/noise/counter/prediction arrays independently
match the frozen original CPU traces exactly. GPU versus CPU parameters and
counters match exactly; maximum noise difference is 6.29e-18 and prediction
difference 4.91e-15, within unchanged gates.

All 16 posterior quantities pass the predeclared checks: maximum R-hat 1.002585,
minimum bulk ESS 2253.48, minimum tail ESS 3345.25. Minimum bulk ESS/s increases
from 10.357 on the improved CPU to 14.299 on GPU (1.381x). Posterior and prediction
mean gates pass. Identical saved GPU samples imply the same ESS as prior GPU
workers, so their 1.247x throughput gain is also a 1.247x ESS/s gain.

Implementation commit: `d6a53cc`; final local checkpoint: `perf-step7-20260930`.
Compact results, profiles, CSV diagnostics, test output, LaTeX and checksums are
versioned in `results/step7*`. PDF: workspace `benchmark/changes/step7/step7.pdf`.
No upstream push was performed. This does not resolve hierarchical mixing or
establish performance on other models, workloads or hardware.
