"""Rotate warmed block-size timings and check exact prediction equality."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from cache_material_preparation import ROOT, mc, setup_for
from gpu.predictions import GpuCuPredictions


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    model = setup_for("pooled").models[0]
    adapters = {
        str(b): GpuCuPredictions(model, block_size=b)
        for b in (32, 64, 128, 256)
    }
    adapters.update({
        f"{b}-hoist": GpuCuPredictions(model, block_size=b, hoist=True)
        for b in (32, 128)
    })
    with np.load(ROOT / "tests/data/material_preparation_baseline.npz") as f:
        params = mc.sc.tran_unif(
            f["normalized_parameters"], f["bounds"], f["parameter_names"]
        )
    rows = []
    for batch in (4, 50, 100, 159, 200, 317):
        native = {k: v[np.arange(batch) % 256] for k, v in params.items()}
        reference = adapters["128"](native)
        durations = {b: [] for b in adapters}
        for adapter in adapters.values():
            np.testing.assert_array_equal(adapter(native), reference)
        for rep in range(40):
            sizes = list(adapters)
            sizes = sizes[rep % len(sizes) :] + sizes[: rep % len(sizes)]
            for b in sizes:
                start = time.perf_counter()
                actual = adapters[b](native)
                durations[b].append(time.perf_counter() - start)
                np.testing.assert_array_equal(actual, reference)
        events = {}
        for b, adapter in adapters.items():
            adapter.timing = True
            adapter.stage_samples.clear()
            for _ in range(10):
                np.testing.assert_array_equal(adapter(native), reference)
            events[b] = adapter.stage_samples
            adapter.timing = False
        rows.append({
            "batch": batch,
            "durations_s": durations,
            "isolated_stages": events,
        })
        print(
            batch,
            {
                b: round(float(np.median(v)) * 1000, 3)
                for b, v in durations.items()
            },
            flush=True,
        )
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
