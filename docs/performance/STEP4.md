# Step 4: batch independent chains into larger GPU evaluations

Branch: `perf/gpu-batched-chains`, based on `perf-step3-20260929`.
Scope: pooled copper calibration only. Automatic backend selection, production
cleanup, Mac testing, and hierarchical algorithm changes are deferred.

## Preserve the inference problem

Four spawned worker processes each own their setup, sampler adaptation, and
Python/NumPy random state. Seeds remain 42/123/456/789. The existing sampler runs
unchanged. A coordinator waits for one material-evaluation request from each
active worker, concatenates parameter arrays, evaluates them together, then
returns each worker's own prediction rows. Chains can have different request
sizes/counts; completed workers leave the active group. No random draws or
likelihoods are combined, and no worker observes another chain's parameters.
The adapter is restricted to the same Cu configuration and pooled evaluations.

Three controls distinguish batching from acceleration:

- `cpu_parallel`: independent compiled-CPU workers, without a coordinator for
  model evaluation. This is the practical multicore CPU comparison.
- `cpu_batched`: the same coordinator and IPC, evaluating combined requests on
  compiled CPU. This checks the batching boundary separately from GPU rounding.
- `gpu_batched`: coordinator evaluates combined requests on the existing float64
  CUDA prototype. Transfers, IPC, synchronization and host work count in timing.

Original per-request interpolation uses flattened, offset strain coordinates.
Combining requests can alter rounding of those offsets, even on CPU. Therefore
numerical prediction/likelihood tolerances remain the original gates; exact
parameter/counter equality is checked explicitly rather than assumed.

## Validation and timing protocol

First compare 3,000 states at 50 temperatures, all three backends, four seeds.
Save all temperature-chain theta/noise arrays and counters. Request-boundary
regressions exercise unequal request sizes, preserved ordering, invalid batches,
and CPU/GPU predictions and likelihoods. Use prediction `rtol=1e-8, atol=1e-10`
and absolute log-likelihood error <=1e-6. Compare parameter and acceptance arrays
exactly; record noise discrepancies separately, within `rtol=1e-8, atol=1e-15`.

Then compare full independent CPU and batched GPU runs: 40,000 states, 50
temperatures, discard 20,000, four seeds, unthinned cold-chain diagnostics.
Use the step-3 posterior agreement gates: all 16 scalar quantities with R-hat
<1.01, bulk/tail ESS >=400, MCSE/SD <0.05; mean differences <=4 combined MCSEs
and <=0.1 pooled SD; prediction mean differences <=0.1 experiment noise SD.

Measure elapsed wall time for all four chains together, from simultaneous
release after warm-up until all samplers finish. Exclude startup/JIT and output
compression, but record startup separately. A barrier defers postprocessing
until all chains finish, so predictions/output writing cannot contend with a
still-running sampler. ESS/s must use this four-chain wall time, not the sum of
worker times. Run competing backends sequentially. Keep original fixtures and
prior artifacts unchanged. Any throughput claim must distinguish a gain over
serial CPU execution from a gain over independent multicore CPU workers.

## Reproduction

Use the existing optional requirements in `benchmarks/gpu/requirements.txt` and
NVIDIA device access for GPU runs. Each output directory must be new.

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_batched_chains.py --backend gpu_batched \
  --iterations 3000 --temperatures 50 --save-full --output NEW_DIRECTORY
```

Repeat with `cpu_parallel` and `cpu_batched`. For the longer quality study use
`--iterations 40000` and omit `--save-full` to retain cold-chain outputs only.
Ordinary chain outputs follow the existing diagnostic format. Coordinator
metadata additionally records group sizes, combined batches, warm-up and wall
time, source hashes and CPU affinity count. Worker errors are propagated and
remaining workers terminated, rather than allowing a failed chain to deadlock
other chains indefinitely. This is an experimental benchmark driver, not a
production worker service.

An additional scaling probe uses eight independent seeds
42/123/456/789/111/222/333/444, 3,000 states and 50 temperatures. Compare CPU
parallel and GPU batched with identical seeds, checking cold-chain outputs and
counters. This probe is for timing/scaling, not a convergence claim. Hardware:
24 CPU cores (Intel Core Ultra 9 285HX), approximately 64 GB RAM, the same RTX
PRO 5000 GPU as step 3. CPU numerical libraries use one thread per worker.

For the long-run comparison, use the diagnostics environment:

```bash
../benchmark/.venv/bin/python benchmarks/gpu_batched_quality.py \
  --reference CPU_PARALLEL_OUTPUT --candidate GPU_BATCHED_OUTPUT \
  --output NEW_QUALITY_DIRECTORY
```

This applies the existing posterior gates and explicitly replaces summed worker
times with the simultaneous-chain wall times when calculating ESS/s.

Driver validation refinement: independent CPU workers do not send evaluation
messages while they sample, so their completion timeout scales with the requested
run length. Request-serving waits retain a 180-second bound. This was corrected
between measured jobs, followed by a two-worker smoke check, before the long runs.
It changes driver liveness handling, not sampler arithmetic or scheduling.

## Results and decision

Checkpoint: `perf-step4-20260930`. Initial implementation: `5588570`;
long-run implementation: `3e1ad43`. All timings below are elapsed sampling wall
time for the entire group, including transfers and coordination.

| Chains | States per chain | Parallel CPU | Batched GPU | CPU/GPU speedup |
| --- | --- | --- | --- | --- |
| 4 | 3,000 | 18.53 s | 23.99 s | 0.772x |
| 8 | 3,000 | 20.59 s | 39.26 s | 0.525x |
| 4 | 40,000 | 252.64 s | 334.53 s | 0.755x |

The four-chain batched CPU control took 46.24 s. GPU batching beats that single
CPU coordinator, but loses to independent CPU workers. Mean combined request
sizes were 159.1 and 317.3 for four/eight chains. Larger calls alone do not make
this workflow faster than multicore CPU execution. These are individual measured
runs, not timing confidence intervals.

All long-run quality gates passed for all 16 parameter/noise quantities. Maximum
R-hat was 1.002585; minimum bulk ESS 2253.48 and tail ESS 3345.25. GPU cold-chain
parameter draws and acceptance counts matched CPU exactly for all four seeds.
Maximum noise-array discrepancy was 6.29e-18; maximum prediction discrepancy
4.91e-15. The parallel CPU runs reproduced the original saved theta, noise,
acceptance and prediction arrays exactly. Four-chain short runs also checked
all temperatures; the eight-chain scaling check compared cold-chain outputs.

Minimum bulk ESS/s was 8.920 on parallel CPU versus 6.736 on batched GPU.
GPU sampling took 32.4% longer, with 24.5% lower ESS/s. The measured quality is
preserved, but the performance gate fails. Keep compiled CPU parallelism for
this workload and preserve GPU batching as an experimental benchmark.

Validation: 94 hardware-enabled tests passed in 115.67 s; the subsequent driver
timeout correction passed a two-worker smoke check and the full long CPU run.
Initial short runs recorded the development HEAD, with runner source hashes
matching `5588570`; eight-chain runs used that commit and long runs used
`3e1ad43`. No production sampler or model source changed in this step.

Compact results, CSV diagnostics, test log, LaTeX source and artifact hashes are
in `docs/performance/results/step4*`. Raw traces/logs and the PDF are in workspace
`benchmark/changes/step4/`. The next GPU hypothesis would need to reduce host
preparation, transfers or coordination costs. Cleanup, Mac testing and the
hierarchical mixing investigation remain separate future work.
