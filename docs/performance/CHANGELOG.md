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
