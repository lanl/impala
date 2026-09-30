# Step 3: experimental GPU Cu material-history calculation

Branch: `perf/gpu-material-history`, starting from `perf-step2-20260929`.
This prototype is in `benchmarks/gpu/`; the production CPU library and default
configuration remain unchanged. The optional adapter attaches to a disposable
model created by the benchmark scripts, not a new public backend option.

## Implementation and environment

One CUDA thread advances one complete material history, using float64 and the
same equations/order as the existing Cu PTW model. CUDA fused multiply-add is
disabled (`--fmad=false`); no fast-math option is enabled. CPU interpolation and
MCMC remain unchanged. All conditions, parameters and initial states are copied
to the GPU on every call, and full histories/final state are copied back. This
avoids a new cache-invalidation dependency and includes transfer cost honestly.
CuPy memory pooling can reuse allocations; inputs are still transferred each call.

The GPU is an NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, compute
capability 12.0, with approximately 24 GB memory. The system CUDA toolkit was
root-readable only. We installed pinned CuPy and minimal CUDA runtime/compiler
wheels into the existing virtual environment without changing NumPy, SciPy,
Numba, system permissions, or the other process using the GPU.

Optional dependencies are in `benchmarks/gpu/requirements.txt`.
Installation reference: https://docs.cupy.dev/en/stable/install.html
GPU commands need access to the NVIDIA devices (outside this session's sandbox).
Use `CUPY_CACHE_DIR=/tmp/impala-cupy-cache` for the writable compilation cache.
The first-call measurement includes compilation if the disk cache is cold; it
must not be described as a guaranteed cold compilation measurement on reruns.

## Accuracy gates declared before calibration runs

- Original 256-case predictions: `rtol=1e-8`, `atol=1e-10 Mbar`.
- Original weighted log likelihoods: absolute error <= `1e-6`.
- Complete histories/final fields and edge cases: `rtol=1e-8`, `atol=1e-10`.
- Invalid PTW parameters must still raise `ConstraintError`.
- CPU/GPU trajectory equality is recorded, not assumed: different transcendental
  implementations can cause rounding changes and eventually different chains.
- For pooled quality comparison, use four independent seeds (42/123/456/789),
  40,000 states, 50 temperatures, discard the first 20,000 states, no thinning.
  Compare with the preserved original CPU runs under the same setup.
- Require all 16 parameter/noise quantities to have rank R-hat < 1.01, bulk and
  tail ESS >= 400, and MCSE(mean)/SD < 0.05.
- Require posterior means to differ by <= four combined MCSEs and <= 0.1 pooled
  posterior SD; report credible-interval endpoint differences separately.
- Require maximum posterior prediction-mean difference across 327 observations
  <= 0.1 posterior noise SD for that experiment. Prediction checks use the same
  200 retained draws per run as the baseline; they supplement full-chain tests.
- Compare ESS per sampling second, not only iteration throughput. The saved CPU
  reference timings are historical; a fresh compiled-CPU run at the same setup
  provides a current runtime comparison, with that distinction kept explicit.

The existing hierarchical reference is not converged. Short hierarchical runs
can check execution/throughput, but cannot certify posterior equivalence.
No production GPU integration is justified from kernel timing alone.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/impala-uv-cache uv pip install --python .venv/bin/python \
  -r benchmarks/gpu/requirements.txt
IMPALA_TEST_GPU=1 CUPY_CACHE_DIR=/tmp/impala-cupy-cache \
  MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
CUPY_CACHE_DIR=/tmp/impala-cupy-cache OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/gpu_material_history.py --output NEW_DIRECTORY
CUPY_CACHE_DIR=/tmp/impala-cupy-cache OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/gpu_calibration.py --backend gpu --mode pooled \
  --iterations 40000 --temperatures 50 --seed 42 --output NEW_RUN_DIRECTORY
```

Output directories must not already exist. Repeat the last command at the four
seeds for the quality study. `--backend cpu` selects the compiled CPU control.
For small-batch smoke/timing comparisons, use 3,000 states and eight temperatures.
The hardware tests are skipped unless `IMPALA_TEST_GPU=1`; ordinary CPU CI does
not require CuPy or a GPU. Do not regenerate any baseline fixture with this code.

To reproduce the complete sequential calibration job order, use
`benchmarks/gpu_suite.py --output NEW_SUITE_DIRECTORY`. It runs the two-mode
small-batch comparisons, one fresh 50-temperature compiled-CPU control, then the
four pooled GPU chains. Run quality diagnostics with the existing ArviZ environment:

```bash
../benchmark/.venv/bin/python benchmarks/gpu_quality.py \
  --reference ../benchmark/results/reference \
  --candidate NEW_SUITE_DIRECTORY --output NEW_DIAGNOSTICS_DIRECTORY
```

The original CPU reference remains untouched. Its sampling-time comparison is
historical; only the new compiled-CPU control provides a contemporaneous runtime.

## Result and decision — 2026-09-29

**Retain the compiled CPU for the current calibration workflow.** The GPU
prototype is accurate and helps large batches, but does not improve the tested
sampler. Production library code and default configuration are unchanged.

- Implementation measured: `7817d9a`; final checkpoint: `perf-step3-20260929`.
- Full hardware-enabled suite: **91 passed in 113.57 seconds**.
- Original 256-case maximum prediction error: `5.97e-16 Mbar`; maximum weighted
  log-likelihood error: `1.75e-10`, both within the existing numerical gates.
- Four 40,000-state pooled GPU runs at 50 temperatures passed **all 16**
  convergence/quality checks. Maximum R-hat: **1.002585**; minimum bulk ESS:
  **2253.48**; minimum tail ESS: **3345.25**.
- Parameter draws and acceptance counters matched the original exactly at all
  four seeds. Noise-variance differences were below `3.7e-18`; saved prediction
  differences were below `7.1e-17 Mbar`. Joint parameter/noise arrays are therefore
  not bitwise identical, although parameter trajectories are identical.
- Transfer-inclusive model speedups vs compiled CPU at batches 1/4/8/50/200/1000:
  **0.19x / 0.28x / 0.38x / 1.16x / 2.90x / 4.87x**. Values below 1 mean slower.
- Eight-temperature, 3,000-state seed-42 pooled sampling: CPU **3.55 s**, GPU
  **8.90 s**. Hierarchical: CPU **8.01 s**, GPU **12.32 s**.
- Matched 50-temperature, 40,000-state pooled seed 42: CPU **186.68 s**, GPU
  **211.88 s**, so GPU was **13.5% slower**. The four GPU runs took
  211.88/215.08/211.44/202.79 seconds. The fresh CPU control exactly reproduced
  the original seed-42 parameter/noise arrays, saved predictions and counters.
- A separate instrumented 3,000-state CPU run at 50 temperatures recorded 3,288
  model calls with mean batch **39.72**, range **1–50**. Constraint filtering
  reduces actual batch sizes; the fixed batch-50 result is not an end-to-end gain.
  This early-chain sample is explanatory evidence, not a full-run profile.
- GPU ESS/s was 1.14x the **historical original** CPU reference. That older CPU
  predates steps 1 and 2; do not present this as a gain over compiled CPU.
- The new quality comparator also passed an original-versus-itself sanity check.
  Hierarchical posterior equivalence remains unestablished because its original
  reference did not converge; only short execution/numerical checks were made.

Raw artifacts and PDF: workspace `benchmark/changes/step3/`. Versioned compact
results, diagnostics, test log, LaTeX source and manifest are under
`docs/performance/results/step3*`. No baseline artifacts were overwritten.
A future GPU experiment should target naturally larger batches, such as batching
independent chains, before considering integration into the sampler. The current
prototype alone does not justify increasing temperatures or changing inference
settings merely to make GPU timing look better.
