# Step 2: compile the Cu material-history driver

Branch: `perf/compile-material-history`, starting from `perf-step1-20260929`.
The original checkout is retained separately at workspace `impala-original/`,
commit `6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e`.

## What changed

The post-cache profile attributed 0.889 of 1.023 seconds (87%) to the history
loop across 200 evaluations at batch size 50. Individual numerical functions
already use Numba; the outer driver still crossed from Python into those
functions on each of 100 steps. The new optional backend compiles that driver
and calls the existing physics functions directly. It does not use fast-math,
parallel execution, reduced precision, a different integration grid, or a new
sampler. It restores all final material-state fields after the compiled call.

This first compiled backend supports exactly the copper example's combination:
PTW stress, quadratic specific heat, linear density, linear melt temperature,
and BGP/PW shear modulus. The Python backend remains the default. Other model
combinations and overridden model classes/methods are rejected explicitly when
requesting the compiled backend. Numba remains an optional dependency, imported
only when the backend is selected. Old objects without a backend attribute
continue using Python.

The compiled path accepts finite float64 histories and state arrays, scalar
material constants, and parameters broadcastable to one value per history.
PTW parameter constraints are checked before entering the loop, because they
are invariant within one history calculation. Direct Python physics calls
retain their existing checks. Nonfinite compiled results raise an error.
This experimental backend does not promise identical NumPy warning/underflow
or partially updated state behavior on invalid inputs. Keep the Python backend
for custom models and unsupported inputs.

## Enable it

In a copy of `examples/cu_config.yaml`, set:

```yaml
settings:
  history_backend: numba
```

Keep the other settings in that mapping. `multi_calib.build_main_model` forwards
this setting for the main stress/strain block. Separate Z blocks are unchanged.
The constructor API also accepts:

```python
model = sc.ModelMaterialStrength(..., history_backend="numba")
```

For an existing model: `model.model.history_backend = "numba"`.
For direct physics construction: `MaterialModel(..., history_backend="numba")`.
Numba must be installed (the project's `numba` extra). First use compiles the
kernel; report startup separately from warmed steady-state performance.

## Accuracy and timing protocol

Run the complete suite, including the new driver regressions:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/impala-matplotlib OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q --durations=10
```

Tests compare predictions and every final state field with the Python driver,
including pooled/hierarchical batches, modified parameters/constants/conditions,
history resolution, nonuniform supplied histories, continued nonzero states,
small-p branches and strain-rate thresholds. They also check unsupported inputs
and original-fixture prediction/likelihood tolerances. Original fixtures are
never regenerated with optimized code.

The independent benchmark imports the untouched original checkout in separate
processes; no changed physics module is shared with that reference:

```bash
git worktree add --detach ../impala-original perf-baseline-20260929
.venv/bin/python benchmarks/compiled_material_history.py \
  --reference-root ../impala-original \
  --output ../benchmark/changes/step2-rerun
```

Skip the first command when that worktree already exists. The benchmark refuses
an existing output directory or a dirty/wrong reference checkout. It compares
original, cached Python, and compiled versions at seeds 42 and 123, reversing
variant execution order for the second seed. Each worker checks the saved
256-case fixture, records 60 warmed evaluations at batch sizes 1/4/50/200,
and runs pooled/hierarchical samplers for 3,000 states and eight temperatures.
Compilation is warmed before sampler timing. Every recorded chain/state and
acceptance/swap counter must match the original exactly; a mismatch fails the
benchmark rather than silently relaxing the gate.

Raw timings, full traces, per-process logs, source hashes and metadata stay in
the external artifact directory. Compact results, tests, manifest, profile and
LaTeX report source are versioned here. Timings across variants are sequential,
not simultaneous, and are observations on this machine. Exact short trajectories
do not establish convergence; the known hierarchical mixing issue remains.

## Recorded result — 2026-09-29

Measured code: `ac87663`; final results tag: `perf-step2-20260929`.
All **83 tests passed**, and all independent original-checkout fixture and
short-chain comparisons matched exactly. The original 256-case fixture had zero
prediction and likelihood error. Both candidates (cached and compiled) matched
the four original sampler runs, including the recorded arrays and counters.

Relative to cached Python, the compiled backend measured **1.97–1.98x** faster
for pooled sampling and **1.51–1.55x** for hierarchical sampling. Model evaluation
speedups at batch sizes 1/4/50/200 were **3.63x / 2.85x / 1.34x / 1.13x**.
The full sampler comparisons used eight temperatures; do not assume these gains
apply unchanged to the earlier 50-temperature workload.

See `results/step2-results.json`, `results/step2-pytest.txt`, the before/after
profiles, and `results/step2-manifest.json`. The two-page PDF is in workspace
`benchmark/changes/step2/step2.pdf`; its source is `results/step2.tex`.
