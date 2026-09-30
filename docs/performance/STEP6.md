# Step 6: measure GPU stages and tune the history kernel

Branch: `perf/gpu-kernel-tuning`, based on the step-5 branch after author/email
corrections (`12934c3`). Author: Ayan Biswas <ayanju04@gmail.com>.
Scope remains four independent pooled Cu chains at 50 temperatures, unchanged
sampler/RNG/adaptation/likelihoods, float64 and no fused multiply-add. Production
library code and defaults are unchanged. Mac and hierarchical mixing work remain
separate. Earlier measured commit IDs are mapped in `AUTHORSHIP_REWRITE.json`.

## Measurement before optimization

The driver now records parent receive/wait/decode and send/encode wall time.
Receive time includes useful worker-side sampling work and scheduling delay;
it is not a measurement of avoidable IPC overhead alone.

Optional `--gpu-timing` instruments the resident evaluator. It explicitly
synchronizes at stage boundaries, measures history/interpolation CUDA event
intervals, and times uploads and downloads after prior GPU work has completed.
Transfers include allocation, Python/runtime and synchronization costs; they are
not pure PCIe DMA measurements. CUDA event intervals may include launch gaps.
Geometry misses include CPU construction and geometry uploads. Device kernel
intervals are nested inside host launch/wait intervals and must not be added to
them. Diagnostic timing perturbs execution; all speed claims use separate
uninstrumented runs. Fixture warm-up is excluded from stage samples.

An initial four-chain, 1,000-state diagnostic measured 4.997 s overall:

| Category | Seconds | Interpretation |
| --- | ---: | --- |
| Material history CUDA events | 2.338 | Largest GPU cost; nested in evaluation |
| Interpolation CUDA events | 0.012 | Small GPU cost |
| Geometry preparation/upload | 0.345 | Includes cache misses |
| Parameter expansion/upload | 0.244 | Live inputs, synchronized |
| Prediction download/allocation | 0.092 | Prior GPU work completed first |
| Parent receive/wait/decode | 1.607 | Includes worker sampling and synchronization |
| Parent send/encode | 0.152 | Sending predictions back to workers |

These rows do not form a complete disjoint wall-time budget: host validation,
allocation, launch and other overhead are recorded separately in raw JSON.

## Targeted changes

1. Expose history-kernel block sizes 32, 64, 128 and 256 through
   `--gpu-block-size`. Each history is independent; no reductions or shared
   numerical state are involved. The previous control remains 128 threads.
2. Optional `--gpu-hoist` evaluates invariant reference-density cube-root/power
   expressions and a constant cube-root factor once per history, outside the
   100-step loop. Subexpressions and operation order are retained. The same CUDA
   source has a compile-time guard; the default/full-history path is preserved.
3. Keep interpolation launch size and the chain coordinator unchanged. Buffer
   allocation and interpolation were too small to prioritize in this experiment.

Rotating 40-call warmed tests cover batches 4/50/100/159/200/317 and all four
block sizes, plus hoisting at 32 and 128. Predictions must match the prior GPU
path bitwise at every call. At batch 159, observed medians were 2.335 ms for the
128-thread control, 2.142 ms with 32 threads, 1.991 ms with hoisting at 128, and
1.831 ms with both. These are screening measurements, not sampler speedups.
The selected candidate is 32 threads plus hoisting. Timing samples and kernel
stage events are retained for each option, including the slower 256-thread case.

## Predeclared validation

Frozen 256-case predictions retain `rtol=1e-8, atol=1e-10` and weighted likelihood
absolute tolerance 1e-6. Test all block sizes with/without hoisting, live conditions
and constants, cache eviction, unequal batches, small-p/rate thresholds, rejected
inputs, and timing instrumentation preserving predictions. Run full GPU-enabled
regressions before long sampling. Original fixtures are never regenerated.

Three sequential short rounds rotate parallel CPU / control GPU / candidate GPU.
Each has four seeds 42/123/456/789, 3,000 states and 50 temperatures. Save all
parameter/noise temperatures; require exact parameters and counters, noise
`rtol=1e-8, atol=1e-15` and prediction `rtol=1e-8, atol=1e-10` agreement.
Two 1,000-state instrumented diagnostic jobs compare control/candidate separately.

Fresh long jobs run control GPU, candidate GPU, then parallel CPU, each with
four 40,000-state chains and 50 temperatures. Discard 20,000 states, no thinning.
All 16 parameter/noise quantities must have R-hat <1.01, bulk/tail ESS >=400,
MCSE/SD <0.05. Posterior mean differences must be <=4 combined MCSEs and <=0.1
pooled SD; prediction means <=0.1 experiment noise SD. Original saved outputs
remain an independent reference. ESS/s uses group wall time, not summed worker
times. Long timing is one run per backend; short repeated timings characterize
some local variability, not a universal hardware speed guarantee.

## Reproduction

Use existing pinned optional GPU dependencies and the same numerical environment.
Choose a new output directory for every command; no artifact is overwritten.

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_block_sizes.py --output NEW_SCREEN
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_kernel_suite.py --stage diagnostic --output NEW_DIAGNOSTICS
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_kernel_suite.py --stage short --output NEW_SHORT
CUPY_CACHE_DIR=/tmp/impala-cupy-cache IMPALA_TEST_GPU=1 \
  .venv/bin/python -m pytest -q
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_kernel_suite.py --stage long --output NEW_LONG
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/impala-matplotlib \
  ../benchmark/.venv/bin/python benchmarks/gpu_batched_quality.py \
  --reference NEW_LONG/cpu --candidate NEW_LONG/candidate --output NEW_QUALITY
```

Sampling excludes startup/JIT and postprocessing, but includes cache misses,
transfers, CPU work and synchronization. The suite fixes numerical threads to
one per worker and records pre-job GPU telemetry, logs and source/commit hashes.
CPU/GPU jobs run sequentially. The normal resident GPU option still defaults to
128 threads without hoisting; select the candidate explicitly:

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_batched_chains.py --backend gpu_resident \
  --gpu-block-size 32 --gpu-hoist --iterations 40000 --temperatures 50 \
  --output NEW_CANDIDATE
```
