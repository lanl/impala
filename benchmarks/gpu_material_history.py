"""Float64 GPU prototype: accuracy and transfer-inclusive paired timings."""

import argparse
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
from cache_material_preparation import ROOT, mc, setup_for
from gpu.history import attach


def main(output):
    import cupy as cp

    output.mkdir(parents=True, exist_ok=False)
    old = setup_for("pooled").models[0]
    old.model.history_backend = "numba"
    candidate = deepcopy(old)
    adapter = attach(candidate.model)
    with np.load(ROOT / "tests/data/material_preparation_baseline.npz") as f:
        saved = {k: f[k] for k in f.files}
    params = mc.sc.tran_unif(
        saved["normalized_parameters"],
        saved["bounds"],
        saved["parameter_names"],
    )
    start = time.perf_counter()
    expected = old.eval(params)
    cpu_first = time.perf_counter() - start
    start = time.perf_counter()
    actual = candidate.eval(params)
    gpu_first = time.perf_counter() - start
    cov = old.lik_cov_inv(saved["variance"], saved["weights"])
    ll = np.array([
        old.llik(saved["observed"], row, cov, saved["weights"])
        for row in actual
    ])
    np.testing.assert_allclose(
        actual, saved["predictions"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(ll, saved["log_likelihood"], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(expected, saved["predictions"])
    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "kernel_sha256": hashlib.sha256(
            (ROOT / "benchmarks/gpu/cu_history.cu").read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            (ROOT / "benchmarks/gpu/history.py").read_bytes()
        ).hexdigest(),
        "gpu": {
            "name": props["name"].decode(),
            "compute_capability": [props["major"], props["minor"]],
            "total_memory": props["totalGlobalMem"],
        },
        "packages": {
            k: version(k)
            for k in (
                "numpy",
                "scipy",
                "numba",
                "cupy-cuda13x",
                "cuda-toolkit",
                "nvidia-cuda-runtime",
                "nvidia-cuda-nvrtc",
            )
        },
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "first_evaluation_seconds": {"cpu": cpu_first, "gpu": gpu_first},
        "fixture": {
            "cases": 256,
            "max_prediction_error": float(
                np.max(np.abs(actual - saved["predictions"]))
            ),
            "max_likelihood_error": float(
                np.max(np.abs(ll - saved["log_likelihood"]))
            ),
            "exact_predictions": bool(
                np.array_equal(actual, saved["predictions"])
            ),
        },
        "timings": [],
    }
    np.savez_compressed(
        output / "fixture.npz", cpu=expected, gpu=actual, gpu_likelihood=ll
    )
    for batch in (1, 4, 8, 50, 200, 1000):
        native = {k: v[np.arange(batch) % 256] for k, v in params.items()}
        reference = old.eval(native)
        candidate.eval(native)
        durations = {"cpu": [], "gpu": []}
        for i in range(30):
            jobs = [("cpu", old), ("gpu", candidate)]
            if i % 2:
                jobs.reverse()
            for label, model in jobs:
                start = time.perf_counter()
                pred = model.eval(native)
                elapsed = time.perf_counter() - start
                durations[label].append(elapsed)
                np.testing.assert_allclose(
                    pred, reference, rtol=1e-8, atol=1e-10
                )
        # Separate events, excluded from end-to-end measurements above.
        adapter.measure_kernel = True
        kernel = []
        for _ in range(5):
            candidate.eval(native)
            kernel.append(adapter.kernel_ms)
        adapter.measure_kernel = False
        cpu, gpu = [
            float(np.median(durations[k]) * 1000) for k in ("cpu", "gpu")
        ]
        result["timings"].append({
            "batch": batch,
            "histories": batch * 6,
            "cpu_ms": cpu,
            "gpu_ms": gpu,
            "speedup": cpu / gpu,
            "kernel_ms": float(np.median(kernel)),
            "raw_seconds": durations,
        })
        (output / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                **result,
                "timings": [
                    {k: v for k, v in r.items() if k != "raw_seconds"}
                    for r in result["timings"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
