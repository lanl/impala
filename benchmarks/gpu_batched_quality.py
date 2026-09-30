"""Step-3 posterior gates with ESS/s based on simultaneous-chain wall time."""

import argparse
import json
from pathlib import Path

import pandas as pd
from gpu_quality import main as compare_posterior


def main(args):
    compare_posterior(args)
    summary = json.loads((args.output / "summary.json").read_text())
    walls = {
        label: json.loads((root / "results.json").read_text())[
            "sampling_wall_seconds"
        ]
        for label, root in [
            ("reference", args.reference),
            ("gpu", args.candidate),
        ]
    }
    frames = {}
    for label, filename in [
        ("reference", "original_diagnostics.csv"),
        ("gpu", "gpu_diagnostics.csv"),
    ]:
        frame = pd.read_csv(args.output / filename, index_col=0)
        frame["ess_per_second"] = frame.ess_bulk / walls[label]
        frame.to_csv(args.output / filename)
        summary[label]["sampling_seconds"] = walls[label]
        summary[label]["min_ess_per_second"] = float(frame.ess_per_second.min())
        frames[label] = frame
    ratios = frames["gpu"].ess_per_second / frames["reference"].ess_per_second
    summary["timing_basis"] = (
        "wall time until all four chains finish, not sum of worker times"
    )
    summary["min_ess_per_second_ratio_cpu"] = float(ratios.min())
    summary["max_ess_per_second_ratio_cpu"] = float(ratios.max())
    summary["sampling_speedup_cpu_parallel_over_gpu_batched"] = (
        walls["reference"] / walls["gpu"]
    )
    del summary["min_ess_per_second_ratio_historical_cpu"]
    del summary["max_ess_per_second_ratio_historical_cpu"]
    comparison = pd.read_csv(args.output / "comparison.csv", index_col=0)
    comparison = comparison.drop(
        columns=["ess_per_second_ratio_historical_cpu"]
    )
    comparison["ess_per_second_ratio_cpu"] = ratios
    comparison.to_csv(args.output / "comparison.csv")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("Final wall-time-adjusted comparison:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
