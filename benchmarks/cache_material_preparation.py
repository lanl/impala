"""Paired original/cached timing and exact sampler regression checks.

Run with .venv/bin/python benchmarks/cache_material_preparation.py --output DIR.
Refuses to overwrite results. The control uses the frozen pre-change eval body;
all other library code is shared (valid for this preparation-only change).
"""

import os

for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/impala-matplotlib")

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path
from types import MethodType

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = load("multi_calib", ROOT / "examples/multi_calib.py")
oracle = load(
    "original_eval", ROOT / "tests/reference/material_eval.py"
).original_material_eval


def setup_for(mode, original=False, iterations=3000, temperatures=8):
    cfg = yaml.safe_load((ROOT / "examples/cu_config.yaml").read_text())
    cfg["paths"]["path_to_data"] = str(ROOT / "data/Cu")
    cfg["settings"]["mcmc"]["nmcmc"] = iterations
    cfg["settings"]["tempering"]["n"] = temperatures
    pooled = mode == "pooled"
    data, temps, rates, _ = mc.load_main_stress_strain(cfg)
    model, indices = mc.build_main_model(
        cfg, data, temps, rates, pooled, mc.get_model_choices(cfg)
    )
    if original:
        model.eval = MethodType(oracle, model)
    setup = mc.build_setup(cfg, model, data, indices, False, [], None, pooled)
    return setup


def snapshot(out, mode):
    arrays = {"s2": out.s2[0].copy(), "count": np.asarray(out.count).copy()}
    if mode == "pooled":
        arrays["theta"] = out.theta.copy()
    else:
        arrays.update(
            theta=out.theta[0].copy(),
            theta0=out.theta0.copy(),
            Sigma0=out.Sigma0.copy(),
            count_temper=out.count_temper.copy(),
        )
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "baseline_commit": "6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e",
        "candidate_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_source_sha256": hashlib.sha256(
            (ROOT / "src/impala/superCal/models_withlik.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "microbenchmarks": [],
        "samplers": [],
    }
    with np.load(ROOT / "tests/data/material_preparation_baseline.npz") as f:
        normalized, bounds, names = (
            f["normalized_parameters"],
            f["bounds"],
            f["parameter_names"],
        )
    baseline = setup_for("pooled", original=True).models[0]
    candidate = setup_for("pooled").models[0]
    for batch in (1, 4, 50, 200):
        params = mc.sc.tran_unif(normalized[:batch], bounds, names)
        expected = baseline.eval(params)  # JIT compilation excluded.
        candidate._experiment_cache.clear()
        start = time.perf_counter()
        actual = candidate.eval(params)
        cold_ms = 1000 * (time.perf_counter() - start)
        np.testing.assert_array_equal(actual, expected)
        timings = {"original": [], "cached": []}
        # Alternate order to reduce systematic clock/thermal bias.
        for repeat in range(60):
            jobs = [("original", baseline), ("cached", candidate)]
            if repeat % 2:
                jobs.reverse()
            for label, model in jobs:
                start = time.perf_counter()
                value = model.eval(params)
                timings[label].append(time.perf_counter() - start)
                np.testing.assert_array_equal(value, expected)
        old, new = [
            float(np.median(timings[k])) for k in ("original", "cached")
        ]
        result["microbenchmarks"].append({
            "batch": batch,
            "original_ms": old * 1000,
            "cached_ms": new * 1000,
            "speedup": old / new,
            "cache_miss_ms": cold_ms,
            "max_absolute_prediction_error": 0.0,
            "paired_repetitions": 60,
            "raw_seconds": timings,
        })
    for mode in ("pooled", "hier"):
        for seed in (42, 123):
            outputs, times = {}, {}
            jobs = [("original", True), ("cached", False)]
            if seed == 123:
                jobs.reverse()
            for label, original in jobs:
                setup = setup_for(mode, original)
                mc.seeds(seed)
                start = time.perf_counter()
                out = mc.run_calibration(setup, mode == "pooled")
                times[label] = time.perf_counter() - start
                outputs[label] = snapshot(out, mode)
            for name in outputs["original"]:
                np.testing.assert_array_equal(
                    outputs["cached"][name],
                    outputs["original"][name],
                    err_msg=f"{mode}/{seed}/{name}",
                )
            result["samplers"].append({
                "mode": mode,
                "seed": seed,
                "states": 3000,
                "temperatures": 8,
                "observations": len(setup.ys[0]),
                "seconds": times,
                "speedup": times["original"] / times["cached"],
                "exactly_equal_arrays": list(outputs["original"]),
            })
            # Save exact equality evidence and full traces for later audit.
            np.savez_compressed(
                args.output / f"{mode}_{seed}_original.npz",
                **outputs["original"],
            )
            np.savez_compressed(
                args.output / f"{mode}_{seed}_cached.npz", **outputs["cached"]
            )
            (args.output / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "All predictions and seeded sampler arrays exactly match the original."
    )


if __name__ == "__main__":
    main()
