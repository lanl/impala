"""Paired instrumentation controls: ordinary and larger GPU request batches."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from gpu_resident_suite import ROOT

OPTIONS = {
    "plain": [],
    "coordination": ["--coordination-timing"],
    "stages": ["--coordination-timing", "--gpu-timing"],
}


def compare(reference, candidate):
    result = {}
    paths = sorted(reference.glob("pooled_*/draws.npz"))
    assert {p.parent.name for p in paths} == {
        p.parent.name for p in candidate.glob("pooled_*/draws.npz")
    }
    for path in paths:
        with (
            np.load(path) as a,
            np.load(candidate / path.parent.name / "draws.npz") as b,
        ):
            assert set(a.files) == set(b.files)
            result[path.parent.name] = {
                key: {"exact": bool(np.array_equal(a[key], b[key]))}
                for key in a.files
            }
    return result


def main(output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", "/tmp/impala-cupy-cache")
    jobs = []
    for round_ in range(3):
        order = list(OPTIONS)
        order = order[round_:] + order[:round_]
        jobs.extend(
            (f"standard_{kind}_{round_}", kind, 3000, 50) for kind in order
        )
    jobs.extend((f"large_{kind}", kind, 1000, 50) for kind in OPTIONS)
    results = {}
    for name, kind, iterations, temperatures in jobs:
        seeds = [42, 123, 456, 789] + (
            [1011, 1213, 1415, 1617] if name.startswith("large") else []
        )
        saved = output / name / "results.json"
        if resume and saved.exists():
            prior = json.loads(saved.read_text())
            assert (
                prior["iterations"] == iterations
                and prior["temperatures"] == temperatures
                and prior["seeds"] == seeds
            )
            assert prior["coordination_timing"] == (kind != "plain") and prior[
                "gpu_timing"
            ] == (kind == "stages")
            for filename, expected in prior["source_sha256"].items():
                assert (
                    hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                    == expected
                ), filename
            results[name] = prior
            print("Reusing completed", name, flush=True)
            continue
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
            "--seeds",
            *map(str, seeds),
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.output, args.resume)
