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

## Initial profile result and artifact rebuild

All saved arrays in the four before/after profiled chains match exactly, including
all-temperature outputs. Seed-42 worker process CPU time falls from 3.040 to
2.588 s; instrumented wall time 11.582 to 11.061 s. Batched likelihoods take
0.259 s cumulative across 3,287 calls, replacing scalar proposal calls (scalar
initialization remains). Profiles perturb timings, so use unprofiled runs for
performance conclusions. Measured implementation commit: `a27110d`.

After collecting `short`, `long`, `quality`, `profile-before`, `profile-after`,
`profile-agreement.json` and `pytest.log` under the artifact root:

```bash
.venv/bin/python benchmarks/gpu_proposal_report.py \
  --artifacts ../benchmark/changes/step8 \
  --original ../benchmark/results/reference
XDG_CACHE_HOME=/tmp/impala-tex-cache ../benchmark/tools/tectonic \
  --keep-logs --outdir ../benchmark/changes/step8 docs/performance/results/step8.tex
.venv/bin/python benchmarks/gpu_proposal_manifest.py \
  --artifacts ../benchmark/changes/step8
```

The profile suite's `gpu_control`/`gpu_candidate` directories correspond to
`profile-before`/`profile-after`; extract its `agreement.json["gpu_candidate"]`
as `profile-agreement.json` when rebuilding from that suite. Original direct
profile calls used those before/after names. Report summaries and LaTeX are
versioned; the PDF and raw traces remain in the workspace artifact directory.
The manifest verifies measured source hashes and inventories raw artifacts,
source, tests, model inputs, the original fixture and frozen original traces.

## Short-run finding and added CPU control

All eleven short comparisons passed the declared accuracy gates. Median seconds:
CPU control 16.167, CPU candidate 17.263; GPU control 11.203, GPU candidate 10.875.
The GPU candidate is faster in all three rounds (median throughput +3.01%),
but CPU batching is slower in two of three rounds (median time +6.78%). This is
not evidence for a general CPU speedup. Keep the candidate opt-in.

Because of this CPU regression, add a fresh long CPU control with proposal
batching disabled, after the suite's three long jobs. Compare GPU against both
CPU paths, not only the potentially slower CPU candidate. This extension was
chosen from short timing evidence before long results were available. Reproduce:

```bash
CUPY_CACHE_DIR=/tmp/impala-cupy-cache .venv/bin/python \
  benchmarks/gpu_batched_chains.py --backend cpu_parallel \
  --worker-batch-gibbs --iterations 40000 --temperatures 50 \
  --output NEW_LONG/cpu_control
```

The report builder includes this extra control's results and independently checks
its saved outputs against the CPU candidate. Its source hashes are checked by the
manifest along with the original suite jobs. Long controls run in the recorded
order GPU control, GPU candidate, CPU candidate, CPU control; this is one timing
per backend, so small differences need cautious interpretation.

Profiling overhead is especially relevant here: eliminating many scalar calls
also eliminates profiler events. The profiled CPU reduction alone does not
establish an unprofiled CPU speedup, and fresh array assembly has its own cost.
The short CPU regression is retained in the results rather than dismissed as
noise. The current evidence supports a small GPU-path experiment, not enabling
proposal batching by default on CPU.

Full hardware-enabled regression suite: **135 passed in 120.10 s**, including the
nine new proposal-likelihood cases. Test-created pairplot output is retained in
the external artifact directory, not mixed with frozen input fixtures.

## Long GPU result

Fresh prior GPU path: 156.062 s; candidate: 154.173 s for four 40,000-state,
50-temperature chains. All saved GPU arrays match bit-for-bit across all four
seeds. Throughput and ESS/s improve by 1.0123x (about 1.2%) in this paired run.
This is smaller than the 3.01% short median gain. Without repeated long trials,
this does not establish a robust general speedup beyond timing variability.
Keep the implementation as an opt-in experiment; do not promote it to a default.
The larger step-7 improvement remains distinct from this incremental result.

## Final controls, quality and decision

| Backend (all retain step-7 Gibbs batching) | Group sampling seconds |
| --- | ---: |
| GPU, scalar proposal likelihoods | 156.062 |
| GPU, batched proposal likelihoods | 154.173 |
| CPU, scalar proposal likelihoods | 212.907 |
| CPU, batched proposal likelihoods | 205.902 |

The additional CPU control takes 212.907 s versus 205.902 s for the candidate:
the long CPU result improves, opposite to the short medians. CPU timing evidence
is mixed. GPU throughput is 1.336x the faster fresh CPU path, and 1.381x the
scalar-proposal CPU path. These comparisons include earlier optimizations and
must not be attributed solely to proposal batching.

CPU control/candidate saved arrays match exactly. Candidate CPU parameter/noise/
counter/prediction arrays also match the frozen original traces exactly. GPU
parameters/counters match CPU exactly; noise/predictions remain within unchanged
numerical tolerances. All 16 posterior quantities pass: maximum R-hat 1.002585,
minimum bulk ESS 2253.48, minimum tail ESS 3345.25. Minimum bulk ESS/s is 14.617
on GPU and 10.944 on batched-proposal CPU. Posterior-mean and prediction-mean
gates pass. Original fixture/reference SHA-256 hashes remain unchanged.

Decision: retain a guarded opt-in experimental implementation, with no default
change and no claim of a robust general speedup. The exactness results are stronger
than the performance evidence. More worker-side vectorization is showing diminishing
returns. Further work should measure GPU execution and coordinator/worker idle time
before attempting more complex asynchronous scheduling; remote wall time alone is
not enough to justify it. Hierarchical mixing remains a separate project.

Implementation `a27110d`; checkpoint `perf-step8-20260930` on
`perf/gpu-proposal-likelihood`. Results, profiles, diagnostics, test output,
LaTeX and manifest are versioned under `results/step8*`. PDF is workspace
`benchmark/changes/step8/step8.pdf`. Commits use Ayan Biswas <ayanju04@gmail.com>.
Nothing was pushed upstream.
