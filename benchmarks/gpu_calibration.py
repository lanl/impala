"""Save cold-chain draws and predictions in the original diagnostic format."""

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from cache_material_preparation import ROOT, mc, setup_for
from gpu.history import attach


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)

    def build(states):
        s = setup_for(
            args.mode, iterations=states, temperatures=args.temperatures
        )
        s.models[0].model.history_backend = "numba"
        if args.backend == "gpu":
            attach(s.models[0].model)
        return s

    mc.seeds(8675309)
    start = time.perf_counter()
    warm = mc.run_calibration(build(5), args.mode == "pooled")
    warm_seconds = time.perf_counter() - start
    del warm
    gc.collect()
    setup = build(args.iterations)
    mc.seeds(args.seed)
    start = time.perf_counter()
    out = mc.run_calibration(setup, args.mode == "pooled")
    seconds = time.perf_counter() - start
    arrays = {
        "s2": out.s2[0][:, 0].copy(),
        "count": np.asarray(out.count),
        "observed": setup.ys[0],
        "experiment_index": setup.s2_ind[0],
    }
    if args.mode == "pooled":
        arrays["theta"] = out.theta[:, 0].copy()
    else:
        arrays.update(
            theta=out.theta[0][:, 0].copy(),
            theta0=out.theta0[:, 0].copy(),
            Sigma0=out.Sigma0[:, 0].copy(),
            count_temper=out.count_temper,
        )
    for a in arrays.values():
        assert np.isfinite(a).all()
    selected = np.linspace(
        args.iterations // 2,
        args.iterations - 1,
        min(200, args.iterations // 2),
        dtype=int,
    )
    native = mc.sc.tran_unif(
        arrays["theta"][selected].reshape(-1, setup.p),
        setup.bounds_mat,
        setup.bounds.keys(),
    )
    predicted = setup.models[0].eval(native, pool=args.mode == "pooled")
    arrays["prediction_indices"] = selected
    arrays["predictions"] = predicted
    rng = np.random.default_rng(args.seed + 100000)
    arrays["predictive_observations"] = predicted + rng.normal(
        size=predicted.shape
    ) * np.sqrt(arrays["s2"][selected][:, setup.s2_ind[0]])
    np.savez_compressed(args.output / "draws.npz", **arrays)
    meta = {
        "backend": args.backend,
        "mode": args.mode,
        "seed": args.seed,
        "iterations": args.iterations,
        "temperatures": args.temperatures,
        "warmup_seconds": warm_seconds,
        "sampling_seconds": seconds,
        "iterations_per_second": (args.iterations - 1) / seconds,
        "parameter_names": list(setup.bounds),
        "bounds": setup.bounds_mat.tolist(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                ROOT / "benchmarks/gpu/history.py",
                ROOT / "benchmarks/gpu/cu_history.cu",
                ROOT / "benchmarks/gpu_calibration.py",
            )
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    p.add_argument("--mode", choices=["pooled", "hier"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--temperatures", type=int, default=8)
    run(p.parse_args())
