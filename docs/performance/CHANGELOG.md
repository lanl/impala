# Performance change log

## Baseline — 2026-09-29

- Source: `6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e`.
- Local tag: `perf-baseline-20260929`.
- Original report/artifacts: workspace `benchmark/report/report.pdf` and
  `benchmark/results/reference/` (outside this checkout).
- Pooled inference passed the initial statistical screen; hierarchical mixing
  did not. Neither this patch nor exact short-chain matching resolves that issue.

## Step 1 — cache fixed material preparation — 2026-09-29

- Branch: `perf/cache-material-preparation`.
- Added a four-entry cache with content-based invalidation and protection against
  mutation of cached arrays through exposed physics-model state.
- Preserved SciPy interpolation, physical equations, parameter checks, numerical
  precision, history resolution, and MCMC algorithms.
- Added a frozen original evaluation method, an independent saved copper fixture,
  focused regression tests, and a paired accuracy/timing benchmark.
- Full test suite: **66 passed** (including 14 new regression tests); see
  `results/step1-pytest.txt`. Lint and whitespace checks passed.
- Final paired measurements and accuracy evidence are recorded below.

### Array-layout refinement

- Initial implementation checkpoint: `7ac9310`.
- Preserved original C/Fortran memory layout when copying history arrays, avoiding
  an unnecessary additional compiled specialization. Regression tests now check
  layout as well as numerical output.
- Warm the evaluator before measuring preparation-cache misses so compilation
  time is excluded. Record independent golden-fixture errors in benchmark JSON.
- Repeated full suite: **66 passed in 100.93 seconds**; see
  `results/step1-final-pytest.txt`. Initial measurements remain in external
  `benchmark/changes/step1/`; final measurements use a separate directory.

### Final paired results

- Measured code: `5c5cb01`; final checkpoint tag: `perf-step1-20260929`.
- All 256 original-fixture predictions and likelihoods matched exactly (zero error).
- All four 3,000-state, eight-temperature original/cached sampler comparisons
  matched exactly, including recorded parameter/noise and acceptance arrays.
- Model speedups (batches 1/4/50/200): 1.029x, 1.041x, 1.090x, 1.107x.
- Full sampler speedups: pooled seed 42: 1.026x, pooled seed 123: 1.043x, hier seed 42: 1.134x, hier seed 123: 1.039x.
- These are small paired timing measurements, not a long-run ESS/s or convergence
  study. Hierarchical mixing remains unresolved. No GPU work was included.
- See `results/step1-results.json`, `results/step1-manifest.json` and
  `results/step1.tex`; full traces and PDF are in external
  `benchmark/changes/step1-final/`. Baseline fixtures remain unchanged.

## Step 2 — compile the Cu history loop — 2026-09-29

- Branch: `perf/compile-material-history`, based on `perf-step1-20260929`.
- Post-cache profile: history driver consumed 0.889/1.023 seconds (87%) for
  200 batch-50 evaluations.
- Added an opt-in Numba history driver for the copper PTW model combination,
  reusing existing numerical functions and preserving float64 arithmetic.
- Added a constructor/configuration setting; Python remains the default.
- Added history/state, edge-case and independent-fixture regressions, plus a
  subprocess benchmark against an untouched original checkout. Final measurements
  and validation are recorded below.
- See [STEP2.md](STEP2.md) for scope, limitations, enabling and reproduction.

### Step-2 results

- Measured implementation: `ac87663`; results checkpoint: `perf-step2-20260929`.
- Full suite: **83 passed in 114.72 seconds** (17 new compiled-driver tests).
- Independent original-checkout comparisons: all 256 fixture predictions and
  likelihoods matched exactly (zero error); all four seeded sampler runs per
  candidate matched the original arrays and counters exactly.
- Model speedups versus cached Python (batches 1/4/50/200): 3.63x, 2.85x, 1.34x, 1.13x.
- Sampler speedups versus cached Python: pooled seed 42: 1.98x, pooled seed 123: 1.97x, hier seed 42: 1.51x, hier seed 123: 1.55x.
- Python remains the default; enable the Cu backend explicitly. These are warmed
  timings on this machine, not a convergence study or a universal speedup claim.
- See `results/step2-results.json`, `results/step2-manifest.json`,
  `results/step2-pytest.txt`, and `results/step2.tex`. Full artifacts and PDF:
  workspace `benchmark/changes/step2/`.

## Step 3 — GPU Cu history prototype — 2026-09-29

- Branch: `perf/gpu-material-history`, based on `perf-step2-20260929`.
- Added an experimental float64 CUDA history kernel and benchmark-only adapter;
  production CPU code and default configuration are unchanged.
- Full suite with opt-in hardware tests: **91 passed in 113.57 seconds**,
  including eight new GPU tests. See `results/step3-pytest.txt`.
- Added transfer-inclusive paired timing and calibration runners, and a pooled
  multi-seed quality comparison with predeclared gates in [STEP3.md](STEP3.md).
- Final results are recorded below; this remains a benchmark-only prototype.

### Step-3 results

- Prototype code: `7817d9a`; results checkpoint: `perf-step3-20260929`.
- All four 40,000-state pooled GPU chains passed the predeclared quality gates
  against the original CPU reference. Parameter draws/counters matched exactly;
  noise and prediction differences were at rounding scale. Max R-hat 1.002585,
  min bulk ESS 2253.48, min tail ESS 3345.25.
- GPU evaluation gains depend on batch size: 1.16x at 50, 2.90x at 200, 4.87x at
  1000; slower at batches 1–8. All timings include transfers and host work.
- Real pooled seed-42 calibration at 50 temperatures: compiled CPU 186.68 s,
  GPU 211.88 s (13.5% slower). Small-batch calibration also favored CPU.
- Decision: retain compiled CPU for this workflow; keep GPU as an experimental
  adapter. Historical-original ESS/s gains are not gains over compiled CPU.
- Added reproducible suite/call-size scripts and array-level diagnostic detail.
  Complete results, CSV diagnostics, PDF source and hashes are versioned in
  `results/step3*`; raw artifacts and PDF are in workspace
  `benchmark/changes/step3/`. See [STEP3.md](STEP3.md) for scope and limitations.

## Step 4 — batch independent chains — 2026-09-30

- Branch: `perf/gpu-batched-chains`, based on `perf-step3-20260929`.
- Added separate-process pooled chains with independent RNG/adaptation state and
  an evaluation-only batching coordinator. No sampler algorithm changes.
- Added multicore CPU, batched CPU and batched GPU controls, request-boundary
  regression tests, and posterior diagnostics using elapsed four-chain wall time.
- Full hardware-enabled suite: **94 passed in 115.67 seconds**. Initial short
  runs matched all-temperature parameter/counter arrays exactly; numerical
  noise/prediction differences passed the existing tolerances.
- Full four-chain run: GPU 334.53 s versus parallel CPU 252.64 s; all quality
  gates passed, with exact parameter draws/counts and rounding-scale noise
  differences. GPU ESS/s was 24.5% lower. Eight-chain scaling also favored CPU.
- Preserve this experimental implementation at `perf-step4-20260930`; retain
  CPU parallelism for the tested workflow. Results, diagnostics, LaTeX and hashes
  are versioned. See [STEP4.md](STEP4.md).

### Step-4 driver refinement

- Initial code checkpoint: `5588570`.
- Made the CPU-only completion wait scale with run length; the coordinator has no
  evaluation messages to receive from those workers while sampling. Paused the
  suite controller between jobs to avoid editing a measured run's code.
- Two-worker CPU smoke check passed after this correction. Added the sequential
  scaling/long-run suite driver for reproduction. Sampler/math code is unchanged.

## Step 5 — resident GPU predictions — 2026-09-30

- Branch `perf/gpu-resident-evaluation`, measured implementation `ac3392f`,
  checkpoint `perf-step5-20260930`.
- Added a bounded device geometry cache and on-device interpolation; return only
  measured-point predictions. Shared CUDA equations retain the original history
  mode. Prediction-only mode is explicit and experimental; production unchanged.
- Three short paired rounds: median resident GPU 14.935 s, old GPU 24.466 s,
  parallel CPU 19.250 s. All-temperature parameter/counter checks passed exactly.
- Fresh four-chain long runs: resident GPU 214.318 s versus parallel CPU 252.157 s,
  17.7% higher ESS/s. All posterior gates passed; parameter draws/counters exact,
  noise and predictions differed only at rounding scale. CPU reproduced original
  saved baseline arrays exactly. Full hardware-enabled suite: 98 passed.
- Versioned reproducible suite/report scripts, LaTeX, JSON/CSV diagnostics,
  profiles, test log and artifact hashes. See [STEP5.md](STEP5.md) for commands,
  retained failed attempts, interface limits and deferred work.
