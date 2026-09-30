"""Paired instrumentation controls: ordinary and larger GPU request batches."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from gpu_resident_suite import ROOT, compare

OPTIONS = {
    "plain": [],
    "coordination": ["--coordination-timing"],
    "stages": ["--coordination-timing", "--gpu-timing"],
}


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    jobs = []
    for round_ in range(3):
        order = list(OPTIONS)
        order = order[round_:] + order[:round_]
        jobs.extend(
            (f"standard_{kind}_{round_}", kind, 3000, 50) for kind in order
        )
    jobs.extend((f"large_{kind}", kind, 1000, 200) for kind in OPTIONS)
    results = {}
    for name, kind, iterations, temperatures in jobs:
        print("Starting", name, flush=True)
        cmd = [
            sys.executable,
            "benchmarks/gpu_batched_chains.py",
            "--backend",
            "gpu_resident",
            "--gpu-block-size",
            "32",
            "--gpu-hoist",
            "--worker-batch-gibbs",
            "--worker-batch-proposals",
            "--save-full",
            "--iterations",
            str(iterations),
            "--temperatures",
            str(temperatures),
            *OPTIONS[kind],
            "--output",
            str((output / name).resolve()),
        ]
        with (output / f"{name}.log").open("w") as log:
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
        results[name] = json.loads((output / name / "results.json").read_text())
        print(name, results[name]["sampling_wall_seconds"], flush=True)
    agreement = {
        name: compare(
            output
            / (
                "large_plain"
                if name.startswith("large")
                else "standard_plain_0"
            ),
            output / name,
        )
        for name in results
    }
    assert all(
        v["exact"]
        for comparison in agreement.values()
        for seed in comparison.values()
        for v in seed.values()
    ), "Instrumentation changed saved arrays"
    (output / "suite.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "agreement.json").write_text(
        json.dumps(agreement, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
