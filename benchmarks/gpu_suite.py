"""Run step-3 calibration jobs sequentially to avoid timing contention."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(base):
    base.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    jobs = [
        ("cpu", "pooled", 42, 3000, 8, "smoke/cpu_pooled_42"),
        ("gpu", "pooled", 42, 3000, 8, "smoke/gpu_pooled_42"),
        ("gpu", "hier", 42, 3000, 8, "smoke/gpu_hier_42"),
        ("cpu", "hier", 42, 3000, 8, "smoke/cpu_hier_42"),
        ("cpu", "pooled", 42, 40000, 50, "control/pooled_42"),
    ]
    jobs += [
        ("gpu", "pooled", seed, 40000, 50, f"pooled_{seed}")
        for seed in (42, 123, 456, 789)
    ]
    for backend, mode, seed, iterations, temperatures, label in jobs:
        destination = base / label
        print("Starting", label, flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "benchmarks/gpu_calibration.py"),
            "--backend",
            backend,
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--iterations",
            str(iterations),
            "--temperatures",
            str(temperatures),
            "--output",
            str(destination.resolve()),
        ]
        with (base / (label.replace("/", "_") + ".log")).open("w") as log:
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used",
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
        metadata = json.loads((destination / "metadata.json").read_text())
        print(
            "Completed",
            label,
            "seconds",
            metadata["sampling_seconds"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
