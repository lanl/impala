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
- Paired sampler/timing measurements will be attached in the results checkpoint.
  No performance claim is made by the code checkpoint.

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
