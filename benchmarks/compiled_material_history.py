"""Compare original checkout, cached Python, and compiled Cu history drivers.

Run with --reference-root ../impala-original --output NEW_DIRECTORY.
Each variant runs in an isolated subprocess importing its specified checkout.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/impala-matplotlib")
    sys.path.insert(0, str(args.source_root / "src"))
    import numpy as np
    import yaml

    import impala

    assert (
        Path(impala.__file__)
        .resolve()
        .is_relative_to(args.source_root.resolve())
    )
    spec = importlib.util.spec_from_file_location(
        "mc", args.source_root / "examples/multi_calib.py"
    )
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    cfg = yaml.safe_load(
        (args.source_root / "examples/cu_config.yaml").read_text()
    )
    cfg["paths"]["path_to_data"] = str(args.source_root / "data/Cu")
    cfg["settings"]["tempering"]["n"] = 8
    if args.variant != "original":
        cfg["settings"]["history_backend"] = (
            "numba" if args.variant == "compiled" else "python"
        )

    def setup(mode, states=3000):
        cfg["settings"]["mcmc"]["nmcmc"] = states
        pooled = mode == "pooled"
        data, temps, rates, _ = mc.load_main_stress_strain(cfg)
        model, indices = mc.build_main_model(
            cfg, data, temps, rates, pooled, mc.get_model_choices(cfg)
        )
        return mc.build_setup(
            cfg, model, data, indices, False, [], None, pooled
        )

    args.output.mkdir(parents=True, exist_ok=False)
    meta = {
        "variant": args.variant,
        "seed": args.seed,
        "source_root": str(args.source_root.resolve()),
        "commit": subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(args.source_root)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (args.source_root / "src/impala").rglob("*.py")
        },
        "micro": [],
        "samplers": {},
    }
    f = np.load(ROOT / "tests/data/material_preparation_baseline.npz")
    model = setup("pooled").models[0]
    native = mc.sc.tran_unif(
        f["normalized_parameters"], f["bounds"], f["parameter_names"]
    )
    start = time.perf_counter()
    predicted = model.eval(native)
    meta["first_evaluation_seconds"] = time.perf_counter() - start
    cov = model.lik_cov_inv(f["variance"], f["weights"])
    llik = np.array([
        model.llik(f["observed"], x, cov, f["weights"]) for x in predicted
    ])
    np.testing.assert_allclose(
        predicted, f["predictions"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(llik, f["log_likelihood"], rtol=0, atol=1e-6)
    meta["fixture"] = {
        "cases": len(predicted),
        "max_prediction_error": float(
            np.max(np.abs(predicted - f["predictions"]))
        ),
        "max_likelihood_error": float(
            np.max(np.abs(llik - f["log_likelihood"]))
        ),
    }
    np.savez_compressed(
        args.output / "fixture.npz", predictions=predicted, likelihood=llik
    )
    for batch in (1, 4, 50, 200):
        params = {k: v[:batch] for k, v in native.items()}
        model.eval(params)
        seconds = []
        for _ in range(60):
            start = time.perf_counter()
            value = model.eval(params)
            seconds.append(time.perf_counter() - start)
            np.testing.assert_array_equal(value, predicted[:batch])
        meta["micro"].append({
            "batch": batch,
            "median_ms": float(np.median(seconds) * 1000),
            "raw_seconds": seconds,
        })
    if args.seed == 42:
        params = {k: v[:50] for k, v in native.items()}
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(200):
            model.eval(params)
        profile.disable()
        profile.dump_stats(str(args.output / "evaluation.prof"))
        with (args.output / "profile.txt").open("w") as stream:
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
                "cumtime"
            ).print_stats(30)
    for mode in ("pooled", "hier"):
        mc.seeds(8675309)
        mc.run_calibration(setup(mode, 5), mode == "pooled")
        s = setup(mode)
        mc.seeds(args.seed)
        start = time.perf_counter()
        out = mc.run_calibration(s, mode == "pooled")
        elapsed = time.perf_counter() - start
        arrays = {"s2": out.s2[0], "count": np.asarray(out.count)}
        if mode == "pooled":
            arrays["theta"] = out.theta
        else:
            arrays.update(
                theta=out.theta[0],
                theta0=out.theta0,
                Sigma0=out.Sigma0,
                count_temper=out.count_temper,
            )
        for a in arrays.values():
            assert np.all(np.isfinite(a))
        np.savez_compressed(args.output / f"{mode}.npz", **arrays)
        meta["samplers"][mode] = {
            "seconds": elapsed,
            "states": 3000,
            "temperatures": 8,
            "arrays": list(arrays),
        }
    (args.output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


def main(args):
    import numpy as np

    baseline = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert baseline == "6ccd5a1ce8ea4b79755ad4387cf43605b4f5aa0e", baseline
    assert not subprocess.check_output(
        ["git", "-C", str(args.reference_root), "status", "--porcelain"],
        text=True,
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"baseline_commit": baseline, "workers": [], "comparisons": []}
    for seed in (42, 123):
        variants = (
            ("original", "cached", "compiled")
            if seed == 42
            else ("compiled", "cached", "original")
        )
        for variant in variants:
            dest = args.output / f"{variant}_{seed}"
            source = args.reference_root if variant == "original" else ROOT
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--source-root",
                str(source.resolve()),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--output",
                str(dest.resolve()),
            ]
            with (args.output / f"{variant}_{seed}.log").open("w") as log:
                subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            meta = json.loads((dest / "metadata.json").read_text())
            for b in meta["micro"]:
                b.pop("raw_seconds")
            summary["workers"].append(meta)
            print(f"Finished {variant} seed {seed}", flush=True)
        for variant in ("cached", "compiled"):
            for name in ("fixture", "pooled", "hier"):
                with (
                    np.load(args.output / f"original_{seed}/{name}.npz") as old,
                    np.load(
                        args.output / f"{variant}_{seed}/{name}.npz"
                    ) as new,
                ):
                    for key in old.files:
                        np.testing.assert_array_equal(
                            new[key],
                            old[key],
                            err_msg=f"{variant}/{seed}/{name}/{key}",
                        )
                    summary["comparisons"].append({
                        "variant": variant,
                        "seed": seed,
                        "output": name,
                        "exactly_equal_arrays": old.files,
                    })
        (args.output / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    print(
        "All fixture and sampler arrays match the independent original checkout exactly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--variant", choices=("original", "cached", "compiled"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.reference_root is None:
            parser.error("--reference-root is required")
        main(args)
