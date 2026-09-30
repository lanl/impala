"""Record actual model-call batch sizes in a short pooled CPU calibration."""

import argparse
import json
from collections import Counter
from pathlib import Path

from cache_material_preparation import mc, setup_for


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    setup = setup_for("pooled", iterations=3000, temperatures=50)
    model = setup.models[0]
    model.model.history_backend = "numba"
    evaluate = model.eval
    counts = Counter()

    def recorded(params, **kwargs):
        counts[len(next(iter(params.values())))] += 1
        return evaluate(params, **kwargs)

    model.eval = recorded
    mc.seeds(42)
    mc.run_calibration(setup, True)
    total = sum(counts.values())
    result = {
        "states": 3000,
        "temperatures": 50,
        "seed": 42,
        "scope": "instrumented early-chain calls, not steady-state timing",
        "calls": total,
        "mean_batch": sum(n * k for n, k in counts.items()) / total,
        "minimum_batch": min(counts),
        "maximum_batch": max(counts),
        "batch_counts": dict(sorted(counts.items())),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
