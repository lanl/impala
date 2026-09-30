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
The initially planned 200-temperature larger-call sweep failed during warm-up
(see the adjustment below). Its replacement uses eight chains, 1,000 states and
50 temperatures, once per instrumentation mode. No inference-quality or ESS/s
comparison is made between these different workloads.

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

Implementation checkpoint: `694fd8d`. Focused CPU tests: 4 passed, 1 GPU-only skip;
hardware-enabled focused tests: **19 passed in 5.64 s**. No library sampler or
GPU-kernel source changed. Detailed GPU stage timing already existed; new timers
are confined to the benchmark driver. Worker timing JSON is written before
postprocessing, and stage/timeline arrays are captured before predictive requests.
The parent postprocessing/saving time is excluded from recorded sampling time.

After the suite completes, rebuild summaries, the wall-time breakdown plot, PDF
and checksums using:

```bash
MPLCONFIGDIR=/tmp/impala-matplotlib .venv/bin/python \
  benchmarks/gpu_coordination_report.py --artifacts ../benchmark/changes/step9
XDG_CACHE_HOME=/tmp/impala-tex-cache ../benchmark/tools/tectonic \
  --keep-logs --outdir ../benchmark/changes/step9 docs/performance/results/step9.tex
.venv/bin/python benchmarks/gpu_coordination_manifest.py \
  --artifacts ../benchmark/changes/step9
```

The report's reference comparison uses the existing sibling artifact
`step8/short/gpu_candidate_0`. Preserve that reference when relocating artifacts.
The report includes per-worker CPU/request counters, stage totals and batch-size/
request-latency distributions. Raw timeline timestamps remain outside Git;
compact summaries and the standalone scientific plot PDF are versioned.

## Larger-call adjustment and preserved failure

The nine standard runs completed successfully. The first 200-temperature warm-up
then failed with `ValueError: shape < 0` in the existing scalar Gamma noise draw,
before timing or candidate instrumentation. The default ladder extends to invalid
Gamma shape values at that temperature count. This is a setup limitation, not a
GPU/timestamp failure. No attempt was made to change the ladder or Gibbs model.

Preserved failure artifacts: `failed-200-temperatures/`, its top-level log and
`suite.log`. The corrected larger-call sweep uses eight independent chains at
50 temperatures (seeds 42/123/456/789/1011/1213/1415/1617), 1,000 states each.
The comparison checks every saved array of all eight chains. `--resume` reuses
the nine completed standard runs after checking their recorded settings; new
experiments omit that flag and use a fresh output directory. The benchmark driver
and numerical implementation are unchanged between standard and eight-chain runs.

## Findings

All twelve successful runs pass exact saved-array checks within workload, including
full-temperature parameters/noise and all eight chains in the larger sweep.
The standard plain output also matches the step-8 candidate bit-for-bit. The
original fixture and four original long CPU traces remain unchanged. This is
measurement evidence, not a new sampler speedup or convergence study.

| Standard mode | Round 1 (s) | Round 2 (s) | Round 3 (s) | Median (s) |
| --- | ---: | ---: | ---: | ---: |
| Plain | 10.726 | 10.811 | 10.841 | 10.811 |
| Coordination timestamps | 10.801 | 10.740 | 10.847 | 10.801 |
| Coordination + GPU stages | 11.052 | 10.964 | 11.158 | 11.052 |

Coordination-only median is within ordinary variation of plain (0.999x).
GPU stage synchronization/timing raises the median by about 2.2% (1.022x).
This is not a precise causal overhead estimate from three trials. Eight-chain
plain/coordination/stage runs take 4.923/5.025/5.047 s, one trial each.

Representative standard detailed-stage round 2, with disjoint parent intervals:

| Component | Seconds | Fraction of group wall time |
| --- | ---: | ---: |
| GPU history + interpolation kernels | 5.460 | 49.8% |
| Other model-call time | 1.951 | 17.8% |
| Polling for worker requests | 2.663 | 24.3% |
| Receiving/decoding requests | 0.293 | 2.7% |
| Sending/encoding replies | 0.430 | 3.9% |
| Batch assembly/splitting | 0.127 | 1.2% |
| Other coordinator/start bookkeeping | 0.040 | 0.4% |

The three standard detailed runs put kernels at 49.0–49.8% of group wall time.
Of the representative 5.460 kernel seconds, 5.421 is history and 0.039 is
interpolation. Inside model-call wall time, parameter expansion/upload takes
0.725 s, geometry preparation/upload 0.491 s, and download/host allocation
0.263 s. These are combined host/device categories, not isolated DMA costs.
CPU process time can include CUDA-driver spinning/waiting; it is not a measure
of avoidable Python overhead. Do not add nested stage timings to the table above.

Standard coordination-only round 2 averages 159.1 parameter vectors per batch
(p95 200); eight chains average 332.7 (p95 400). Request-start spread is median
0.132 ms/p95 0.220 ms for four chains, versus 0.305/0.846 ms for eight. Standard
latest-request-to-model-start is median 0.164 ms/p95 0.294 ms. These measures
include host serialization/scheduling and are not estimates of pure IPC or
critical-path time that could be removed.

Larger-call geometry preparation/upload is 1.190 s, 23.6% of its 5.047 s run;
kernels are 1.829 s, 36.2%. There are 143 distinct batch sizes and 162 geometry
misses, including 19 misses for previously seen sizes. First-observed misses
cost 1.028 s; repeated misses 0.141 s. In the standard run, 130 sizes/144 misses
include 14 repeated misses costing 0.054 s. The 256 MiB resident cache is nearly
full in both cases. Thus eviction occurs, but most measured geometry cost is
first-use preparation, especially in the shorter 1,000-state larger-call run.
Simply increasing cache capacity cannot eliminate those first-use costs. These
short-run fractions must not be extrapolated to long-run steady-state costs.

## Recommended order

1. Profile the **history kernel** at the measured batch-size distribution (roughly
   150–200 and 300–400 parameter vectors) to determine compute versus memory
   limits before another kernel change. It is the largest standard-run component.
2. Investigate geometry construction/reuse and parameter expansion/upload. Separate
   cold and warm costs, and use longer larger-batch controls before claiming gains.
   A bounded cache-size sweep can isolate the smaller eviction cost; allocation
   reuse or less repeated geometry construction would address different costs.
3. Consider communication changes only with a concrete measured target. Request
   decode plus reply send is about 6.6% in the representative standard run, and
   even that is not all removable. Worker polling includes necessary sampler work.
   Asynchronous scheduling may trade larger efficient GPU batches for overlap;
   no speedup from that tradeoff has been demonstrated here.

Final local checkpoint: `perf-step9-20260930`, branch
`perf/gpu-coordination-profile`. PDF: workspace `benchmark/changes/step9/step9.pdf`.
Versioned results include the standalone wall-time breakdown plot PDF. Nothing
was pushed. Numerical implementation, default settings and hierarchical mixing
are unchanged.
