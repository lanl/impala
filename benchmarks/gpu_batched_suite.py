"""Sequential scaling and long-run jobs for the batched-chain experiment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    four = [42, 123, 456, 789]
    eight = four + [111, 222, 333, 444]
    jobs = [
        ("cpu_parallel_8", "cpu_parallel", 3000, eight),
        ("gpu_batched_8", "gpu_batched", 3000, eight),
        ("gpu_batched_long", "gpu_batched", 40000, four),
        ("cpu_parallel_long", "cpu_parallel", 40000, four),
    ]
    for label, backend, states, seeds in jobs:
        print("Starting", label, flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "benchmarks/gpu_batched_chains.py"),
            "--backend",
            backend,
            "--iterations",
            str(states),
            "--temperatures",
            "50",
            "--seeds",
            *[str(s) for s in seeds],
            "--output",
            str((output / label).resolve()),
        ]
        with (output / f"{label}.log").open("w") as log:
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
        result = json.loads((output / label / "results.json").read_text())
        print(
            "Finished",
            label,
            "wall seconds",
            result["sampling_wall_seconds"],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
