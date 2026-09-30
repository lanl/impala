"""Batch independent pooled chains' evaluations without changing the sampler.

Each spawned process owns its NumPy/Python RNG state. A coordinator only joins
material evaluations; it never combines targets, adaptation, or random draws.
"""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/impala-matplotlib")

import argparse
import contextlib
import cProfile
import hashlib
import json
import multiprocessing as mp
import pstats
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from cache_material_preparation import ROOT, mc, setup_for


def evaluate_group(model, requests):
    """Combine heterogeneous batch lengths; split in the same request order."""
    names = tuple(requests[0])
    if any(tuple(r) != names for r in requests):
        raise ValueError("Request parameter names/order differ")
    sizes = [len(r[names[0]]) for r in requests]
    if any(n <= 0 for n in sizes) or any(
        any(len(v) != n for v in r.values()) for r, n in zip(requests, sizes)
    ):
        raise ValueError("Invalid parameter batch shapes")
    combined = {k: np.concatenate([r[k] for r in requests]) for k in names}
    predictions = model.eval(combined, pool=True)
    return np.split(predictions, np.cumsum(sizes)[:-1]), sum(sizes)


def worker(connection, config, seed, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with (
        (folder / "worker.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            warm = setup_for(
                "pooled", iterations=5, temperatures=config["temperatures"]
            )
            warm.models[0].model.history_backend = "numba"
            mc.seeds(8675309)
            mc.run_calibration(warm, True)
            setup = setup_for(
                "pooled",
                iterations=config["iterations"],
                temperatures=config["temperatures"],
            )
            model = setup.models[0]
            model.model.history_backend = "numba"
            if config["backend"] != "cpu_parallel":

                def remote(params, pool=None, **kwargs):
                    if pool is False or kwargs:
                        raise ValueError(
                            "Only ordinary pooled evaluations are supported"
                        )
                    connection.send(("evaluate", params))
                    tag, predictions = connection.recv()
                    if tag != "result":
                        raise RuntimeError("Invalid coordinator reply")
                    return predictions

                model.eval = remote
            connection.send(("ready", None))
            assert connection.recv()[0] == "start"
            mc.seeds(seed)
            start = time.perf_counter()
            out = mc.run_calibration(setup, True)
            seconds = time.perf_counter() - start
            connection.send(("sampled", seconds))
            assert connection.recv()[0] == "postprocess"
            arrays = {
                "theta": out.theta[:, 0].copy(),
                "s2": out.s2[0][:, 0].copy(),
                "count": np.asarray(out.count),
                "count_s2": np.asarray(out.count_s2),
                "count_decor": np.asarray(out.count_decor),
                "observed": setup.ys[0],
                "experiment_index": setup.s2_ind[0],
            }
            if config["save_full"]:
                arrays.update(full_theta=out.theta, full_s2=out.s2[0])
            selected = np.linspace(
                config["iterations"] // 2,
                config["iterations"] - 1,
                min(200, config["iterations"] // 2),
                dtype=int,
            )
            native = mc.sc.tran_unif(
                arrays["theta"][selected], setup.bounds_mat, setup.bounds.keys()
            )
            arrays["predictions"] = model.eval(native, pool=True)
            arrays["prediction_indices"] = selected
            rng = np.random.default_rng(seed + 100000)
            arrays["predictive_observations"] = arrays[
                "predictions"
            ] + rng.normal(size=arrays["predictions"].shape) * np.sqrt(
                arrays["s2"][selected][:, setup.s2_ind[0]]
            )
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(folder / "draws.npz", **arrays)
            meta = {
                **config,
                "seed": seed,
                "mode": "pooled",
                "sampling_seconds": seconds,
                "iterations_per_second": (config["iterations"] - 1) / seconds,
                "parameter_names": list(setup.bounds),
                "bounds": setup.bounds_mat.tolist(),
            }
            (folder / "metadata.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            connection.send(("done", None))
        except BaseException:
            connection.send(("error", traceback.format_exc()))
            raise
        finally:
            connection.close()


def serve(connections, model, terminal, timeout=180):
    active = list(connections)
    batches = Counter()
    groups = Counter()
    completed = {}
    eval_seconds = 0.0
    receive_seconds = send_seconds = 0.0
    while active:
        pending = []
        retired = []
        for seed in active:
            connection = connections[seed]
            receive_start = time.perf_counter()
            if not connection.poll(timeout):
                raise TimeoutError(f"Chain {seed} stalled")
            tag, payload = connection.recv()
            receive_seconds += time.perf_counter() - receive_start
            if tag == "error":
                raise RuntimeError(f"Chain {seed}: {payload}")
            if tag == terminal:
                completed[seed] = payload
                retired.append(seed)
            elif tag == "evaluate":
                pending.append((seed, payload))
            else:
                raise RuntimeError(f"Unexpected message {tag}")
        for seed in retired:
            active.remove(seed)
        if pending:
            start = time.perf_counter()
            predictions, total = evaluate_group(model, [p for _, p in pending])
            eval_seconds += time.perf_counter() - start
            batches[total] += 1
            groups[len(pending)] += 1
            send_start = time.perf_counter()
            for (seed, _), prediction in zip(pending, predictions):
                connections[seed].send(("result", prediction))
            send_seconds += time.perf_counter() - send_start
    return {
        "chain_seconds": completed,
        "combined_batch_counts": dict(sorted(batches.items())),
        "group_sizes": dict(sorted(groups.items())),
        "evaluation_seconds": eval_seconds,
        "receive_wait_and_decode_seconds": receive_seconds,
        "send_encode_seconds": send_seconds,
    }


def run(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "backend": args.backend,
        "iterations": args.iterations,
        "temperatures": args.temperatures,
        "save_full": args.save_full,
        "profile": args.profile,
        "gpu_block_size": args.gpu_block_size,
        "gpu_timing": args.gpu_timing,
        "gpu_hoist": args.gpu_hoist,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    startup = time.perf_counter()
    model = None
    if args.backend != "cpu_parallel":
        model = setup_for("pooled").models[0]
        model.model.history_backend = "numba"
        if args.backend == "gpu_batched":
            from gpu.history import attach

            attach(model.model)
        if args.backend == "gpu_resident":
            from gpu.predictions import attach

            attach(
                model,
                block_size=args.gpu_block_size,
                timing=args.gpu_timing,
                hoist=args.gpu_hoist,
            )
        with np.load(
            ROOT / "tests/data/material_preparation_baseline.npz"
        ) as f:
            native = mc.sc.tran_unif(
                f["normalized_parameters"], f["bounds"], f["parameter_names"]
            )
            predicted = model.eval(native)
            np.testing.assert_allclose(
                predicted, f["predictions"], rtol=1e-8, atol=1e-10
            )
    context = mp.get_context("spawn")
    processes = []
    connections = {}
    try:
        for seed in args.seeds:
            parent, child = context.Pipe()
            process = context.Process(
                target=worker,
                args=(child, config, seed, str(args.output / f"pooled_{seed}")),
            )
            process.start()
            child.close()
            processes.append(process)
            connections[seed] = parent
        for seed, c in connections.items():
            if not c.poll(180):
                raise TimeoutError(f"Chain {seed} did not initialize")
            tag, payload = c.recv()
            if tag != "ready":
                raise RuntimeError(str(payload))
        if args.backend == "gpu_resident":
            model.eval.stage_samples.clear()  # Exclude fixture warm-up.
        startup_seconds = time.perf_counter() - startup
        start = time.perf_counter()
        for c in connections.values():
            c.send(("start", None))
        # CPU-only workers send no requests while sampling. Their completion
        # wait must scale with run length; broker requests retain a short bound.
        timeout = max(180, args.iterations / 20) if model is None else 180
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        stats = serve(connections, model, "sampled", timeout=timeout)
        if profiler:
            profiler.disable()
        wall = time.perf_counter() - start
        if profiler:
            profiler.dump_stats(str(args.output / "sampling.prof"))
            with (args.output / "profile.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(60)
        if args.backend == "gpu_resident":
            stats["resident_evaluation"] = model.eval.statistics()
            if args.gpu_timing:
                (args.output / "gpu-stages.json").write_text(
                    json.dumps(model.eval.stage_samples, indent=2) + "\n"
                )
        for c in connections.values():
            c.send(("postprocess", None))
        serve(connections, model, "done")
        for p in processes:
            p.join(10)
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exit {p.exitcode}")
        result = {
            **config,
            "seeds": args.seeds,
            "startup_seconds": startup_seconds,
            "sampling_wall_seconds": wall,
            "sampling": stats,
            "cpu_affinity_count": len(os.sched_getaffinity(0)),
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (
                    Path(__file__),
                    ROOT / "benchmarks/gpu/history.py",
                    ROOT / "benchmarks/gpu/cu_history.cu",
                    ROOT / "benchmarks/gpu/predictions.py",
                )
            },
        }
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))
    finally:
        for c in connections.values():
            c.close()
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["cpu_parallel", "cpu_batched", "gpu_batched", "gpu_resident"],
        required=True,
    )
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--temperatures", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    p.add_argument("--save-full", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--gpu-block-size", type=int, choices=[32, 64, 128, 256], default=128
    )
    p.add_argument("--gpu-timing", action="store_true")
    p.add_argument("--gpu-hoist", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
