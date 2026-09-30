"""Step-6 paired calibration controls, with isolated diagnostics kept separate."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from gpu_resident_suite import compare

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {
    "cpu": ["--backend", "cpu_parallel"],
    "control": ["--backend", "gpu_resident"],
    "candidate": [
        "--backend",
        "gpu_resident",
        "--gpu-block-size",
        "32",
        "--gpu-hoist",
    ],
}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    if args.stage == "short":
        jobs = []
        for i in range(3):
            order = list(OPTIONS)
            order = order[i:] + order[:i]
            jobs.extend((f"{label}_{i}", label, 3000) for label in order)
    elif args.stage == "long":
        jobs = [
            (label, label, 40000) for label in ("control", "candidate", "cpu")
        ]
    else:
        jobs = [(label, label, 1000) for label in ("control", "candidate")]
    results = {}
    for label, kind, iterations in jobs:
        cmd = [
            sys.executable,
            "benchmarks/gpu_batched_chains.py",
            *OPTIONS[kind],
            "--iterations",
            str(iterations),
            "--temperatures",
            "50",
            "--output",
            str((args.output / label).resolve()),
        ]
        if args.stage == "short":
            cmd.append("--save-full")
        if args.stage == "diagnostic":
            cmd.extend(["--save-full", "--gpu-timing"])
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
    reference = "cpu_0" if args.stage == "short" else "control"
    comparisons = {
        label: compare(args.output / reference, args.output / label)
        for label in results
        if label != reference
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
        "--stage", choices=["short", "long", "diagnostic"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
