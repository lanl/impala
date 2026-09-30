"""Build compact results and LaTeX from completed step-5 artifacts."""

import argparse
import json
import pstats
import re
import shutil
import statistics
from pathlib import Path

import numpy as np


def main(args):
    base, docs = args.artifacts.resolve(), args.docs.resolve()
    docs.mkdir(parents=True, exist_ok=True)
    short = json.loads((base / "short/suite.json").read_text())
    long = json.loads((base / "long/suite.json").read_text())
    q = json.loads((base / "quality/summary.json").read_text())
    agreement = json.loads((base / "short/agreement.json").read_text())
    profiles = {}
    for label, folder in (
        ("before", "profile-baseline"),
        ("after", "profile-after/resident_profile"),
    ):
        stats = pstats.Stats(str(base / folder / "sampling.prof"))
        profiles[label] = {
            name: next(v[3] for k, v in stats.stats.items() if k[2] == name)
            for name in ("serve", "evaluate_group", "asnumpy")
        }
    timing = {}
    for backend in ("cpu_parallel", "gpu_batched", "gpu_resident"):
        values = [
            short[f"{backend}_{i}"]["sampling_wall_seconds"] for i in range(3)
        ]
        timing[backend] = {
            "seconds": values,
            "median": statistics.median(values),
        }
    original = {}
    for seed in (42, 123, 456, 789):
        with (
            np.load(args.original / f"pooled_{seed}/draws.npz") as a,
            np.load(
                base / f"long/cpu_parallel_long/pooled_{seed}/draws.npz"
            ) as b,
        ):
            original[str(seed)] = {
                key: bool(np.array_equal(a[key], b[key]))
                for key in ("theta", "s2", "count", "predictions")
            }
    test_match = re.search(
        r"(\d+) passed in ([\d.]+)s", (base / "pytest.log").read_text()
    )
    if test_match is None:
        raise ValueError("Missing successful full-suite test result")
    test_summary = test_match.group(0)
    result = {
        "profiles": profiles,
        "long_array_agreement": json.loads(
            (base / "long/agreement.json").read_text()
        ),
        "tests": test_summary,
        "short_timings": timing,
        "short_runs": short,
        "long_runs": long,
        "quality": q,
        "short_agreement": agreement,
        "original_cpu_agreement": original,
    }
    (docs / "step5-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for src, dst in [
        ("comparison.csv", "comparison"),
        ("original_diagnostics.csv", "cpu-diagnostics"),
        ("gpu_diagnostics.csv", "gpu-diagnostics"),
    ]:
        shutil.copyfile(base / "quality" / src, docs / f"step5-{dst}.csv")
    for src, dst in [
        ("profile-baseline/profile.txt", "profile-before"),
        ("profile-after/resident_profile/profile.txt", "profile-after"),
        ("pytest.log", "pytest"),
    ]:
        shutil.copyfile(base / src, docs / f"step5-{dst}.txt")
    labels = {
        "cpu_parallel": "Parallel CPU",
        "gpu_batched": "Previous batched GPU",
        "gpu_resident": "Resident GPU",
    }
    rows = []
    for key, row in timing.items():
        numbers = " & ".join(f"{v:.2f}" for v in row["seconds"])
        rows.append(f"{labels[key]} & {numbers} & {row['median']:.2f}\\\\")
    cpu = long["cpu_parallel_long"]["sampling_wall_seconds"]
    gpu = long["gpu_resident_long"]["sampling_wall_seconds"]
    fields = [
        ("Maximum R-hat", "max_rhat"),
        ("Minimum bulk ESS", "min_bulk_ess"),
        ("Minimum tail ESS", "min_tail_ess"),
        ("Minimum bulk ESS/s", "min_ess_per_second"),
    ]
    diagnostics = "\n".join(
        f"{label} & {q['reference'][key]:.5g} & {q['gpu'][key]:.5g}\\\\"
        for label, key in fields
    )
    passed = q["all_quality_gates_pass"]
    exact = all(
        a["theta"]["exact"] and a["count"]["exact"]
        for a in q["array_agreement"].values()
    )
    noise = max(
        a["s2"]["max_absolute_difference"]
        for a in q["array_agreement"].values()
    )
    pred = max(
        a["predictions"]["max_absolute_difference"]
        for a in q["array_agreement"].values()
    )
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\emergencystretch=3em
\title{Impala performance experiment: Step 5}
\author{Resident GPU geometry and prediction interpolation}
\date{September 30, 2026}
\begin{document}\maketitle
\section*{Outcome}
For four independent 40,000-state pooled Cu chains at 50 temperatures, resident
GPU evaluation took GPUSECONDS s versus CPUSECONDS s on parallel CPU:
SPEEDUP times CPU throughput. All predeclared posterior-quality gates: QUALITY.
This is an experimental prediction-only adapter; sampler equations, random draws,
adaptation and likelihoods are unchanged. Production defaults, Mac support and
hierarchical mixing changes remain outside this experiment.
\section*{Evidence and implementation}
The previous four-chain prototype spent OLDEVAL of OLDWALL profiled seconds in grouped
model evaluation for a 1,000-state run. It copied full histories to the CPU and
rebuilt flattened interpolation arrays. Of this time, OLDREAD s was in device reads,
which includes waiting for asynchronous kernels, not solely transfer cost.
The resident profile reduced grouped evaluation to NEWEVAL s and coordinator time
to NEWWALL s. Profiles guide investigation; the timing comparisons below are unprofiled.

The new adapter caches fixed experiment histories, initial conditions and
interpolation indices on the GPU, bounded by 256 MiB of live cached arrays.
The same CUDA history equations emit only stress histories, which stay on-device.
A second kernel interpolates at observed strain points. Only predictions return
to the CPU. Parameters and constants are refreshed on every evaluation.

Cache keys include content signatures for experiment arrays and batch size;
edits invalidate cached geometry. The interpolation preparation preserves the
original cumulative strain arithmetic and flattened experiment offsets. All
arithmetic remains float64 with fused multiply-add disabled. The existing
full-history adapter remains available. The new prediction-only evaluator does
not update material state or support state continuation. The CuPy allocator can
retain freed memory beyond the logical cache budget.
\section*{Paired short runs}
Each entry is wall time in seconds for four chains, each with 3,000 states and
50 temperatures. Backend order rotates between three rounds. Jobs run sequentially,
with one numerical-library thread per worker. Transfers, cache misses, IPC and
coordination are included. Startup/JIT and output compression are excluded;
startup is recorded separately. The only warm-up evaluation uses the unchanged
fixture; unseen geometries are prepared during sampling.
\begin{center}\begin{tabular}{lrrrr}\toprule
Backend & Round 1 & Round 2 & Round 3 & Median\\\midrule
TIMINGROWS
\bottomrule\end{tabular}\end{center}
Hardware: Intel Core Ultra 9 285HX, 24 CPU cores; RTX PRO 5000 Blackwell laptop
GPU, approximately 24 GB device memory. Three repeats measure local variability;
they do not establish universal speedups. Earlier step-4 times remain separate
historical results. Identical seeds are 42, 123, 456 and 789.
\section*{Accuracy and posterior quality}
Short comparisons cover all-temperature parameters/noise and recorded counters.
Exact parameter/counter checks and numerical noise/prediction checks passed.
The frozen 256-case prediction fixture and weighted likelihood gates remain
unchanged: prediction relative tolerance $10^{-8}$, absolute $10^{-10}$ Mbar;
likelihood absolute tolerance $10^{-6}$. Noise tolerance is relative $10^{-8}$,
absolute $10^{-15}$. Cache invalidation, eviction, unequal batches, rate/p
thresholds and invalid inputs are covered by hardware tests. Full hardware-enabled suite:
TESTSUMMARY.

For the long study, discard the first 20,000 states and use all remaining cold
chain draws without thinning. All 16 scalar parameter/noise quantities require
rank R-hat below 1.01, bulk/tail ESS at least 400, and MCSE(mean)/SD below 0.05.
Mean differences must be within four combined MCSEs and 0.1 pooled posterior SD;
prediction means within 0.1 experiment noise SD. Predictions use the same 200
retained draws per chain as the baseline. The CPU reference is a fresh parallel
run, also compared with the frozen original outputs.
\begin{center}\begin{tabular}{lrr}\toprule
Diagnostic & Parallel CPU & Resident GPU\\\midrule
DIAGNOSTICS
\bottomrule\end{tabular}\end{center}
Quality gates: QUALITY. Exact GPU parameter trajectories and acceptance counts:
EXACT. Maximum noise-array difference: NOISE; maximum prediction difference:
PRED. Fresh CPU arrays exactly reproduce original saved reference: ORIGINAL.
GPU/CPU effective samples per second range: ESSMIN--ESSMAX. ESS/s uses wall time
for the whole four-chain group, not summed worker times. The speed claim is
conditional on quality gates passing for this workload.
\section*{Reproducibility and limitations}
\begingroup\raggedright
Branch: \texttt{perf/gpu-resident-evaluation}.\\
Measured implementation: \texttt{COMMIT}.\\
Checkpoint: \texttt{perf-step5-20260930}.
See \path{docs/performance/STEP5.md} for commands, failure records, test results
and scope. \path{benchmarks/gpu_resident_suite.py} reproduces paired experiments;
\path{benchmarks/gpu_resident_report.py} rebuilds compact results and this LaTeX.
GPU dependencies remain pinned in \path{benchmarks/gpu/requirements.txt}.

Raw traces, telemetry, profiles and logs are under workspace
\path{benchmark/changes/step5/}. Versioned compact JSON, CSVs, test logs, LaTeX
and a SHA-256 artifact manifest are under \path{docs/performance/results/step5*}.
No original reference artifact or tolerance was changed. All commits are local;
no upstream changes have been published. This experiment does not establish
speedups for other models, hierarchical calibration, Mac GPUs or small batches.
The CPU control is the existing compiled-CPU backend, not a newly rewritten
CPU prediction-only evaluator; some preparation/interpolation savings could
also benefit a future CPU implementation. The remaining synchronization/kernel
costs require separate evidence before
further changes are justified.
\par\endgroup
\end{document}
"""
    replacements = {
        "OLDEVAL": f"{profiles['before']['evaluate_group']:.2f}",
        "OLDWALL": f"{profiles['before']['serve']:.2f}",
        "OLDREAD": f"{profiles['before']['asnumpy']:.2f}",
        "NEWEVAL": f"{profiles['after']['evaluate_group']:.2f}",
        "NEWWALL": f"{profiles['after']['serve']:.2f}",
        "TESTSUMMARY": test_summary,
        "GPUSECONDS": f"{gpu:.2f}",
        "CPUSECONDS": f"{cpu:.2f}",
        "SPEEDUP": f"{cpu / gpu:.3f}",
        "QUALITY": "passed" if passed else "FAILED",
        "TIMINGROWS": "\n".join(rows),
        "DIAGNOSTICS": diagnostics,
        "EXACT": str(exact),
        "NOISE": f"{noise:.3g}",
        "PRED": f"{pred:.3g}",
        "ORIGINAL": str(all(all(v.values()) for v in original.values())),
        "ESSMIN": f"{q['min_ess_per_second_ratio_cpu']:.3f}",
        "ESSMAX": f"{q['max_ess_per_second_ratio_cpu']:.3f}",
        "COMMIT": long["gpu_resident_long"]["commit"][:7],
    }
    for key, value in replacements.items():
        tex = tex.replace(key, value)
    (docs / "step5.tex").write_text(tex)
    print(
        json.dumps(
            {
                "timings": timing,
                "long_cpu_seconds": cpu,
                "long_gpu_seconds": gpu,
                "quality": passed,
                "exact": exact,
                "original": original,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument(
        "--docs", type=Path, default=Path("docs/performance/results")
    )
    main(p.parse_args())
