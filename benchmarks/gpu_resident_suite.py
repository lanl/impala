"""Sequential, paired resident-GPU experiments; never overwrite artifacts."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, candidate):
    result = {}
    for seed in (42, 123, 456, 789):
        with (
            np.load(reference / f"pooled_{seed}/draws.npz") as a,
            np.load(candidate / f"pooled_{seed}/draws.npz") as b,
        ):
            row = {}
            for key in a.files:
                if key not in b:
                    raise ValueError(f"Missing {key}")
                exact = bool(np.array_equal(a[key], b[key]))
                error = float(np.max(np.abs(a[key] - b[key])))
                row[key] = {"exact": exact, "max_absolute_difference": error}
                if key in ("s2", "full_s2"):
                    np.testing.assert_allclose(
                        a[key], b[key], rtol=1e-8, atol=1e-15
                    )
                elif key in ("predictions", "predictive_observations"):
                    np.testing.assert_allclose(
                        a[key], b[key], rtol=1e-8, atol=1e-10
                    )
                else:
                    np.testing.assert_array_equal(a[key], b[key])
            result[str(seed)] = row
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    if args.stage == "profile":
        jobs = [("resident_profile", "gpu_resident", 1000, True)]
    elif args.stage == "short":
        jobs = []
        for repeat in range(3):
            order = ["cpu_parallel", "gpu_batched", "gpu_resident"]
            order = order[repeat:] + order[:repeat]
            jobs.extend(
                (f"{backend}_{repeat}", backend, 3000, False)
                for backend in order
            )
    else:
        jobs = [
            ("gpu_resident_long", "gpu_resident", 40000, False),
            ("cpu_parallel_long", "cpu_parallel", 40000, False),
        ]
    results = {}
    for label, backend, iterations, profile in jobs:
        cmd = [
            sys.executable,
            "benchmarks/gpu_batched_chains.py",
            "--backend",
            backend,
            "--iterations",
            str(iterations),
            "--temperatures",
            "50",
            "--output",
            str((args.output / label).resolve()),
        ]
        if profile:
            cmd.append("--profile")
        if args.stage == "short":
            cmd.append("--save-full")
        print("Starting", label, flush=True)
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used",
                    "--format=csv",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results[label] = json.loads(
            (args.output / label / "results.json").read_text()
        )
        print(label, results[label]["sampling_wall_seconds"], flush=True)
    if args.stage == "short":
        comparisons = {
            label: compare(args.output / "cpu_parallel_0", args.output / label)
            for label in results
            if label != "cpu_parallel_0"
        }
        (args.output / "agreement.json").write_text(
            json.dumps(comparisons, indent=2) + "\n"
        )
    (args.output / "suite.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=["profile", "short", "long"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
