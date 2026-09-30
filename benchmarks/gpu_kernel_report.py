"""Rebuild the step-6 timing/quality report from completed recorded artifacts."""

import argparse
import json
import re
import shutil
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    base = args.artifacts.resolve()
    docs = ROOT / "docs/performance/results"
    short = json.loads((base / "short/suite.json").read_text())
    long = json.loads((base / "long/suite.json").read_text())
    diagnostic = json.loads((base / "diagnostic/suite.json").read_text())
    q = json.loads((base / "quality/summary.json").read_text())
    screening = json.loads((base / "kernel-options/results.json").read_text())
    screening_summary = [
        {
            "batch": row["batch"],
            "median_evaluation_s": {
                key: statistics.median(values)
                for key, values in row["durations_s"].items()
            },
            "median_history_device_s": {
                key: statistics.median(v["history_device_s"] for v in values)
                for key, values in row["isolated_stages"].items()
            },
        }
        for row in screening
    ]
    profiles = {}
    for label in ("control", "candidate"):
        samples = json.loads(
            (base / f"diagnostic/{label}/gpu-stages.json").read_text()
        )
        profiles[label] = {
            k: sum(row[k] for row in samples)
            for k in samples[0]
            if k.endswith("_s")
        }
        profiles[label]["calls"] = len(samples)
    medians = {
        label: statistics.median(
            short[f"{label}_{i}"]["sampling_wall_seconds"] for i in range(3)
        )
        for label in ("cpu", "control", "candidate")
    }
    tests = re.search(
        r"\d+ passed in [\d.]+s", (base / "pytest.log").read_text()
    )
    if tests is None:
        raise ValueError("No successful full test result")
    original = {}
    for seed in (42, 123, 456, 789):
        with (
            np.load(args.original / f"pooled_{seed}/draws.npz") as a,
            np.load(base / f"long/cpu/pooled_{seed}/draws.npz") as b,
        ):
            original[str(seed)] = {
                k: bool(np.array_equal(a[k], b[k]))
                for k in ("theta", "s2", "count", "predictions")
            }
    all_gpu_arrays_exact = all(
        v["exact"]
        for seed in json.loads((base / "long/agreement.json").read_text())[
            "candidate"
        ].values()
        for v in seed.values()
    )
    result = {
        "control_candidate_all_saved_arrays_exact": all_gpu_arrays_exact,
        "screening_summary": screening_summary,
        "short": short,
        "long": long,
        "diagnostic": diagnostic,
        "stage_totals": profiles,
        "short_medians": medians,
        "quality": q,
        "original_cpu_agreement": original,
        "tests": tests.group(),
        "short_agreement": json.loads(
            (base / "short/agreement.json").read_text()
        ),
        "long_agreement": json.loads(
            (base / "long/agreement.json").read_text()
        ),
        "diagnostic_agreement": json.loads(
            (base / "diagnostic/agreement.json").read_text()
        ),
    }
    (docs / "step6-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for src, dest in [
        ("quality/comparison.csv", "comparison.csv"),
        ("quality/original_diagnostics.csv", "cpu-diagnostics.csv"),
        ("quality/gpu_diagnostics.csv", "gpu-diagnostics.csv"),
        ("pytest.log", "pytest.txt"),
    ]:
        shutil.copyfile(base / src, docs / f"step6-{dest}")
    labels = {
        "cpu": "Parallel CPU",
        "control": "Previous resident GPU",
        "candidate": "Tuned GPU",
    }
    timingrows = []
    for label, display_name in labels.items():
        values = [
            short[f"{label}_{i}"]["sampling_wall_seconds"] for i in range(3)
        ]
        timingrows.append(
            display_name
            + " & "
            + " & ".join(f"{v:.2f}" for v in [*values, medians[label]])
            + r"\\"
        )
    stagefields = [
        ("History CUDA events", "history_device_s"),
        ("Interpolation CUDA events", "interpolation_device_s"),
        ("Geometry host + upload", "geometry_host_and_upload_s"),
        ("Parameters + upload", "parameter_expand_and_upload_s"),
        ("Download + host allocation", "download_and_host_allocation_s"),
    ]
    stagerows = [
        f"{name} & {profiles['control'][key]:.3f} & {profiles['candidate'][key]:.3f}\\\\"
        for name, key in stagefields
    ]
    for name, key in [
        ("Parent receive/wait/decode", "receive_wait_and_decode_seconds"),
        ("Parent send/encode", "send_encode_seconds"),
    ]:
        stagerows.append(
            f"{name} & {diagnostic['control']['sampling'][key]:.3f} & {diagnostic['candidate']['sampling'][key]:.3f}\\\\"
        )
    stagerows.append(
        f"Total instrumented wall time & {diagnostic['control']['sampling_wall_seconds']:.3f} & {diagnostic['candidate']['sampling_wall_seconds']:.3f}\\\\"
    )
    cpu, control, candidate = [
        long[k]["sampling_wall_seconds"]
        for k in ("cpu", "control", "candidate")
    ]
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\emergencystretch=3em
\title{Impala performance experiment: Step 6}
\author{GPU timing breakdown and exact-math kernel tuning}
\date{September 30, 2026}
\begin{document}\maketitle
\section*{Outcome}
Four independent pooled Cu chains, 40,000 states each, 50 temperatures:
previous resident GPU CONTROL s, tuned GPU CANDIDATE s, parallel CPU CPUSECONDS s.
The tuned GPU delivers SPEEDUPCONTROL times the previous GPU throughput and
SPEEDUPCPU times parallel CPU throughput. All predeclared posterior-quality gates:
QUALITY. The full hardware-enabled suite: TESTS.

The selected candidate uses 32 threads per history-kernel block and computes
invariant reference-density cube-root/power terms and a constant cube-root factor
once per history. The control uses 128 threads and the original loop expressions.
There are no changes to the physical equations, float64 precision, sampler,
likelihood, random draws, integration grid or adaptation. Fused multiply-add
remains disabled. The coordinator and interpolation kernel launch size are unchanged.
\section*{Where the time goes}
Each diagnostic run uses four chains with 1,000 states and 50 temperatures.
Explicit synchronization separates transfer wall time from outstanding kernels.
CUDA event intervals measure history and interpolation execution. Their host
launch/wait intervals, recorded in JSON, include those device intervals.
\begin{center}\begin{tabular}{lrr}\toprule
Category (seconds) & Control GPU & Tuned GPU\\\midrule
STAGEROWS
\bottomrule\end{tabular}\end{center}
These rows are not an additive full budget: host validation, launch and other
overhead also contribute. Receive time includes useful sampler work in worker
processes, scheduling, IPC and synchronization; it is not all avoidable overhead.
Transfer intervals include runtime/allocation costs, not pure DMA. CUDA event
intervals can include launch gaps. Stage synchronization perturbs timing, so
speed claims use the separate uninstrumented runs below. Warm-up is excluded
from stage samples.
\section*{Screening and repeated calibration timings}
Screening rotates 40 warmed calls across block sizes 32/64/128/256 and hoisting
at 32/128, for batches 4/50/100/159/200/317. Predictions must match the control
bitwise at every call. The selected combination was fastest in the representative
screen; raw timings and stage events include unsuccessful options too.

Short end-to-end comparisons use four chains with 3,000 states and 50 temperatures.
Backend order rotates across three rounds. Each row gives elapsed sampling seconds
for the entire chain group, not the sum of worker times.
\begin{center}\begin{tabular}{lrrrr}\toprule
Backend & Round 1 & Round 2 & Round 3 & Median\\\midrule
TIMINGROWS
\bottomrule\end{tabular}\end{center}
All jobs run sequentially; numerical-library threads are fixed to one per worker.
Startup/JIT and output saving are excluded, but transfers, preparation/cache misses,
CPU work and coordination are included. The same fixture warm-up precedes sampling.
The long study has one fresh run per backend. Three short repeats characterize
local variation; these are not universal timing guarantees. Hardware remains the
24-core Intel Core Ultra 9 285HX and RTX PRO 5000 Blackwell laptop GPU.
\section*{Accuracy and sampling quality}
All-temperature short-chain checks require exact parameter/counter arrays, noise
relative/absolute tolerances $10^{-8}/10^{-15}$, and prediction tolerances
$10^{-8}/10^{-10}$. The original frozen 256-case fixture retains its prediction
and weighted-likelihood gates (absolute likelihood error at most $10^{-6}$).
Regression tests cover all block sizes with/without hoisting, changes to conditions
and constants, cache eviction, unequal batches, rate/p thresholds and invalid inputs.
Instrumentation is checked to preserve predictions.

Long diagnostics discard 20,000 states, without thinning. All 16 parameter/noise
quantities require rank R-hat below 1.01, bulk/tail ESS at least 400, and MCSE/SD
below 0.05. Means must agree within four combined MCSEs and 0.1 pooled SD;
prediction means within 0.1 experiment noise SD. Predictions use the same 200
retained draws per chain as the reference.

Maximum R-hat: RHAT. Minimum bulk/tail ESS: BULK/TAIL.
Minimum bulk ESS/s: CPUESS on CPU and GPUESS on tuned GPU.
Exact tuned-GPU parameter draws and acceptance counts versus CPU: EXACT.
Maximum noise-array difference: NOISE; maximum prediction difference: PRED.
The fresh CPU arrays reproduce the frozen original baseline exactly: ORIGINAL.
Every saved tuned GPU array matches the previous GPU exactly: GPUEXACT.
Long model-evaluation time fell from OLDEVAL to NEWEVAL s, while parent
receive/wait/decode rose from OLDWAIT to NEWWAIT s. This category includes useful
worker computation; profile workers before attributing it to avoidable IPC.
Quality gates: QUALITY. ESS/s always uses group elapsed wall time.
\section*{Reproduction and scope}
\begingroup\raggedright
Branch: \texttt{perf/gpu-kernel-tuning}.\\
Measured implementation: \texttt{COMMIT}. Checkpoint: \texttt{perf-step6-20260930}.\\
See \path{docs/performance/STEP6.md} for exact commands and limitations.
\path{benchmarks/gpu_kernel_suite.py} reproduces diagnostic, short and long jobs;
\path{benchmarks/gpu_kernel_report.py} rebuilds compact results and this LaTeX.
The candidate is explicit: \texttt{--backend gpu\_resident --gpu-block-size 32 --gpu-hoist}.
The normal experimental default remains the previous GPU configuration.

Raw traces, telemetry, timing samples and logs are under workspace
\path{benchmark/changes/step6/}. Versioned JSON/CSV summaries, test output, LaTeX
and an artifact hash manifest are under \path{docs/performance/results/step6*}.
Original fixtures and earlier results remain unchanged. Commits use Ayan Biswas
\texttt{<ayanju04@gmail.com>} and are local; nothing was pushed upstream.
No production-library, Mac, hierarchical-mixing or asynchronous scheduling changes
are included. Performance gains apply to this tested pooled workload.
\par\endgroup
\end{document}
"""
    replacements = {
        "GPUEXACT": str(all_gpu_arrays_exact),
        "OLDEVAL": f"{long['control']['sampling']['evaluation_seconds']:.2f}",
        "NEWEVAL": f"{long['candidate']['sampling']['evaluation_seconds']:.2f}",
        "OLDWAIT": f"{long['control']['sampling']['receive_wait_and_decode_seconds']:.2f}",
        "NEWWAIT": f"{long['candidate']['sampling']['receive_wait_and_decode_seconds']:.2f}",
        "CONTROL": f"{control:.2f}",
        "CANDIDATE": f"{candidate:.2f}",
        "CPUSECONDS": f"{cpu:.2f}",
        "SPEEDUPCONTROL": f"{control / candidate:.3f}",
        "SPEEDUPCPU": f"{cpu / candidate:.3f}",
        "QUALITY": "passed" if q["all_quality_gates_pass"] else "FAILED",
        "TESTS": tests.group(),
        "STAGEROWS": "\n".join(stagerows),
        "TIMINGROWS": "\n".join(timingrows),
        "RHAT": f"{q['gpu']['max_rhat']:.6f}",
        "BULK": f"{q['gpu']['min_bulk_ess']:.2f}",
        "TAIL": f"{q['gpu']['min_tail_ess']:.2f}",
        "CPUESS": f"{q['reference']['min_ess_per_second']:.3f}",
        "GPUESS": f"{q['gpu']['min_ess_per_second']:.3f}",
        "EXACT": str(
            all(
                v["theta"]["exact"] and v["count"]["exact"]
                for v in q["array_agreement"].values()
            )
        ),
        "NOISE": f"{max(v['s2']['max_absolute_difference'] for v in q['array_agreement'].values()):.3g}",
        "PRED": f"{max(v['predictions']['max_absolute_difference'] for v in q['array_agreement'].values()):.3g}",
        "ORIGINAL": str(all(all(v.values()) for v in original.values())),
        "COMMIT": long["candidate"]["commit"][:7],
    }
    # Longest keys first: CONTROL also occurs within SPEEDUPCONTROL.
    for key in sorted(replacements, key=len, reverse=True):
        tex = tex.replace(key, replacements[key])
    (docs / "step6.tex").write_text(tex)
    print(
        json.dumps(
            {
                "short_medians": medians,
                "long_seconds": {
                    k: long[k]["sampling_wall_seconds"] for k in labels
                },
                "quality_pass": q["all_quality_gates_pass"],
                "original": original,
                "stages": profiles,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    main(p.parse_args())
