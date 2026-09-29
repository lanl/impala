# Performance changes and accuracy gates

Baseline: `6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e`, retained by local tag
`perf-baseline-20260929` and branch `perf/cpu-baseline`.
Development branch: `perf/cache-material-preparation`.

Each optimization gets a separate code commit, an accuracy check against the
baseline, and a dated result entry in [CHANGELOG.md](CHANGELOG.md). Keep numerical
and sampler changes separate. Do not regenerate baseline fixtures using optimized
code. Git tags are local checkpoints; no changes have been pushed upstream.

## Step 1: cache material-model preparation

`ModelMaterialStrength` caches expanded experimental conditions, integration
histories, experiment offsets, and flattened observation/interpolation coordinates.
The cache stores at most four batch geometries using least-recently-used eviction.
Its key includes the batch size, history resolution, padding constant, experiment
count, and the shapes/dtypes/contents of experimental arrays. In-place input edits
therefore invalidate the cache. Material parameters and constants are evaluated
fresh every time, and model state is reinitialized with fresh mutable arrays.
Copies preserve the original memory layout to avoid extra compiled specializations.

Actual simulated strain coordinates are compared before reusing flattened
interpolation coordinates. The existing `interp1d` operation, floating-point
arithmetic, 100-step integration, parameter checks, and sampler are unchanged.
Older pickled model objects acquire a cache on their first evaluation.

Interpolation **weights** are not substituted for SciPy's existing calculation:
that would change arithmetic and complicate boundary/rounding behavior. Parameter
validity checks remain in place because direct physics calls also rely on them.
Moving those checks or fusing the history loop requires a separate patch.
The wrapper's preparation cache assumes its normal `MaterialModel` grid generator;
custom overrides of `set_history_variables` with externally changing behavior are
not a supported cached-input source. Dynamic simulated output grids are checked.
Concurrent use of one mutable material-model instance remains unsupported.

## Accuracy gates for every change

From the repository root, with the installed locked environment:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/impala-matplotlib OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_material_preparation_cache.py

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/impala-matplotlib OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q --durations=10
```

There are two independent controls:

1. `tests/reference/material_eval.py` freezes the original evaluation method.
   Pooled and hierarchical calls, changing batch sizes/parameters, cache eviction,
   input mutation, pool overrides, simulated-grid changes, and exception behavior
   are compared to it. Predictions and likelihoods must be exactly equal in the
   same environment. This control shares the physics implementation, so it is
   appropriate for this preparation-only patch, not sufficient on its own for a
   future physics-kernel rewrite.
2. `tests/data/material_preparation_baseline.npz` preserves pre-optimization
   predictions and likelihoods for 128 feasible prior-bound inputs and 128 sampled
   inputs, plus boundary/out-of-bounds constraint cases. It was copied from the
   original CPU benchmark fixture, not recalculated by this patch. Prediction
   tolerances are `rtol=1e-8, atol=1e-10 Mbar`; log-likelihood absolute tolerance
   is `1e-6`; constraint masks must agree exactly. Its standalone model configuration
   is in `tests/data/material_preparation_config.json`.

## Paired sampler and timing check

```bash
.venv/bin/python benchmarks/cache_material_preparation.py \
  --output ../benchmark/changes/step1-rerun
```

Choose a new output directory for a rerun; the script refuses to overwrite one.
Run after tests, with no other benchmark jobs running. Compilation is warmed
before model timings. Original and cached calls alternate order for 60 repetitions
at batch sizes 1, 4, 50 and 200. Cache-miss latency is recorded separately.

Both modes also run original/cached samplers at seeds 42 and 123 for 3,000 states,
eight temperatures, 327 observations, and the original 100-step histories.
Tempering starts at iteration 2,000. Order is reversed for the second seed.
All temperature-chain parameter/noise arrays, hierarchical means/covariances,
and recorded acceptance/swap counters must match exactly. This is a trajectory
regression test, not a new claim of convergence for those short chains.

The timing control swaps only `eval` to the frozen original method. All other
library code is shared. For later patches that change physics or samplers, compare
against a separate checkout/environment of the baseline commit as well as the
saved numerical fixture.

Results record the code commit and source SHA-256. Compact results and test logs
are committed under `docs/performance/results/`; full traces and raw timings stay
in the specified external output directory. Record their hashes in the versioned
manifest. Timings are observations on this machine, not universal speedup promises.

Useful history commands:

```bash
git log --oneline perf-baseline-20260929..perf/cache-material-preparation
git diff perf-baseline-20260929..perf/cache-material-preparation -- src tests benchmarks docs/performance
```

Use a separate worktree for side-by-side baseline work rather than overwriting
uncommitted development files. A future optimization should start from a named
checkpoint and repeat these accuracy gates before comparing performance.

## Recorded step-1 checkpoint

Final measured code: `5c5cb01` (initial implementation: `7ac9310`).
The final documentation/results checkpoint is tagged `perf-step1-20260929`.
See `results/step1-results.json`, `results/step1-manifest.json`, and
`results/step1-final-pytest.txt`. The LaTeX report source is `results/step1.tex`;
its PDF and full paired traces are in workspace `benchmark/changes/step1-final/`.
The earlier baseline report remains at `benchmark/report/report.pdf`.

The new regression tests are collected by ordinary `pytest` and the existing
pull-request test workflow. Long paired timing runs remain an explicit benchmark
command, since shared CI hardware is unsuitable for precise speed comparisons.
Before accepting each future optimization, rerun the accuracy tests and paired
benchmark, use a new artifact directory, and commit its results and manifest.
If arithmetic changes break exact trajectory equality, require predeclared
numerical tolerances and independent multi-seed posterior/predictive comparisons,
including R-hat, ESS, and ESS per second, before calling the change successful.
