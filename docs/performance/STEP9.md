# Step 9: separate GPU execution from coordinator/worker waiting

Branch `perf/gpu-coordination-profile`, based on `perf-step8-20260930` (`76aade0`).
Author: Ayan Biswas <ayanju04@gmail.com>. Measurement only: sampler, model physics,
GPU kernels and scheduling policy are unchanged. Both step-7 Gibbs and step-8
proposal batching remain enabled to measure the latest experimental path.

## Questions and measurements

- How much elapsed coordinator time is model evaluation versus receiving/waiting,
  batch assembly/splitting, reply transmission, and other bookkeeping?
- Within model evaluation, how much is measured CUDA kernel execution versus
  host validation, geometry/cache preparation, parameter upload, allocation,
  kernel launch/synchronization and output download?
- Do requests arrive together, or do early workers wait for late workers?
- How much does instrumentation distort timing, and how do larger calls differ?

`--coordination-timing` enables worker request timestamps (same-machine monotonic
clock), worker send/receive/CPU counters, coordinator poll/decode counters and a
per-batch timeline. No cProfile is enabled. The coordinator still receives one
request per active chain in the original order, combines them, evaluates once,
and replies in the original order. Worker termination remains unchanged.

The timeline records request start and coordinator receive completion per worker,
model start/end, batch size and reply-send start/end. A request timestamp precedes
serialization, so request-to-model latency includes serialization, pipe handling,
scheduling, waiting for peers and assembly; it is not pure IPC latency.
Request-start spread indicates worker skew but is not an additive wall-time cost.
Coordinator polling waits include necessary worker sampler computation.

`--gpu-timing` is the existing synchronized stage profiler. CUDA events time
history/interpolation execution; host timers cover validation, geometry preparation
and upload, parameter expansion/upload, allocation, launches/waits, output download
and validation. Upload/download categories include host work, not just device DMA.
Device event durations are nested inside launch/wait wall times: never add both.
Measured device-time/wall-time is not hardware occupancy or a utilization counter.
GPU stream synchronization and CUDA events perturb timings, hence separate controls.

## Predeclared experiment and validation

Three standard rounds: four chains, 3,000 states, 50 temperatures. Each round runs
plain, coordination-only and coordination+GPU-stage instrumentation, rotating order.
An exploratory larger-call sweep uses four chains, 1,000 states, 200 temperatures,
once per instrumentation mode. No inference-quality claim is made for these short
larger-temperature chains. Never compare the two workloads as equivalent posteriors.

Save full-temperature outputs. All saved arrays must match exactly within each
workload across instrumentation modes and repeated rounds. Compare the standard
plain run to the saved step-8 candidate; retain the original 256-case fixture check
inside every launch. No sampler changes imply no new long ESS/convergence study;
step-8 diagnostics remain the existing evidence, conditional on unchanged outputs.
Focused tests cover request ordering, heterogeneous batches, retiring workers,
timing accounting, GPU prediction accuracy and existing GPU timing behavior.

Run sequentially, one numerical thread per worker. Exclude startup/JIT and saving;
include cache misses, initialization inside sampling, transfers and coordination.
Do not treat summed worker time as elapsed group time. Do not infer hypothetical
asynchronous speedups by subtracting all waiting or summing overlapping intervals.

## Reproduction

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache IMPALA_TEST_GPU=1 .venv/bin/python -m pytest -q \
  tests/test_coordination_timing.py tests/test_batched_gpu_requests.py \
  tests/test_gpu_resident_predictions.py
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_coordination_suite.py --output NEW_RUNS
```

Raw traces, GPU telemetry, full-temperature samples and logs reside in workspace
`benchmark/changes/step9/`. Source hashes identify measured code; final compact
JSON, LaTeX, checksums and interpretation will be versioned in `results/step9*`.
No asynchronous scheduling or new production default is part of this experiment.
