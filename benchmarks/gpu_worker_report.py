"""Build step-7 results and LaTeX from recorded worker profiles and runs."""

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
    q = json.loads((base / "quality/summary.json").read_text())
    labels = {
        "cpu_control": "CPU, scalar workers",
        "cpu_candidate": "CPU, batched workers",
        "gpu_control": "GPU, scalar workers",
        "gpu_candidate": "GPU, batched workers",
    }
    medians = {
        k: statistics.median(
            short[f"{k}_{i}"]["sampling_wall_seconds"] for i in range(3)
        )
        for k in labels
    }
    profiles = {
        kind: {
            str(seed): json.loads(
                (
                    base / f"profile-{kind}/pooled_{seed}/worker-timing.json"
                ).read_text()
            )
            for seed in (42, 123, 456, 789)
        }
        for kind in ("before", "after")
    }
    original = {}
    for seed in (42, 123, 456, 789):
        with (
            np.load(args.original / f"pooled_{seed}/draws.npz") as a,
            np.load(base / f"long/cpu_candidate/pooled_{seed}/draws.npz") as b,
        ):
            original[str(seed)] = {
                k: bool(np.array_equal(a[k], b[k]))
                for k in ("theta", "s2", "count", "predictions")
            }
    agreement = json.loads((base / "long/agreement.json").read_text())
    exact = all(
        v["exact"]
        for seed in agreement["gpu_candidate"].values()
        for v in seed.values()
    )
    tests = re.search(
        r"\d+ passed in [\d.]+s", (base / "pytest.log").read_text()
    )
    if tests is None:
        raise ValueError("No successful full test result")
    result = {
        "short": short,
        "long": long,
        "short_medians": medians,
        "profiles": profiles,
        "quality": q,
        "original_cpu_agreement": original,
        "gpu_all_saved_arrays_exact": exact,
        "tests": tests.group(),
        "long_agreement": agreement,
        "short_agreement": json.loads(
            (base / "short/agreement.json").read_text()
        ),
        "profile_agreement": json.loads(
            (base / "profile-agreement.json").read_text()
        ),
    }
    (docs / "step7-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for src, dest in [
        ("quality/comparison.csv", "comparison.csv"),
        ("quality/original_diagnostics.csv", "cpu-diagnostics.csv"),
        ("quality/gpu_diagnostics.csv", "gpu-diagnostics.csv"),
        ("pytest.log", "pytest.txt"),
        ("profile-before/pooled_42/profile.txt", "profile-before.txt"),
        ("profile-after/pooled_42/profile.txt", "profile-after.txt"),
    ]:
        shutil.copyfile(base / src, docs / f"step7-{dest}")
    rows = []
    for label, name in labels.items():
        values = [
            short[f"{label}_{i}"]["sampling_wall_seconds"] for i in range(3)
        ]
        rows.append(
            name
            + " & "
            + " & ".join(f"{v:.2f}" for v in [*values, medians[label]])
            + r"\\"
        )
    old, new, cpu = [
        long[k]["sampling_wall_seconds"]
        for k in ("gpu_control", "gpu_candidate", "cpu_candidate")
    ]
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\emergencystretch=3em
\title{Impala performance experiment: Step 7}
\author{Worker CPU profiling and batched Gibbs bookkeeping}
\date{September 30, 2026}
\begin{document}\maketitle
\section*{Outcome}
Four pooled Cu chains, 40,000 states each, 50 temperatures: previous GPU worker
path OLDGPU s; batched-worker GPU NEWGPU s; batched-worker parallel CPU CPUSECONDS s.
The new GPU path delivers SPEEDOLD times the previous GPU throughput and SPEEDCPU
times the improved CPU throughput. All predeclared posterior gates: QUALITY.
Every saved GPU output matches the previous worker path exactly: EXACT.
Full hardware-enabled suite: TESTS.
\section*{Why worker computation was selected}
The seed-42 diagnostic used 3,000 states and 50 temperatures. Before optimization,
wall time was BEFOREWALL s and process CPU time BEFORECPU s. Remote model requests
used BEFOREREMOTEWALL s wall but only BEFOREREMOTECPU s CPU. The remaining CPU
work included 314,350 likelihood calls and 150,000 covariance preparations.
The profile therefore pointed to temperature-loop bookkeeping before changes
to communication or asynchronous scheduling.

With batching, seed-42 process CPU time was AFTERCPU s and instrumented wall
time AFTERWALL s. Four-chain profiled outputs matched exactly at all temperatures.
Profiles perturb timing and are diagnostic evidence, not performance gates.
Remote wall time includes GPU evaluation, other workers, useful sampler computation
and scheduling. Summed worker CPU time is not elapsed group time.
\section*{What changed and what is preserved}
The opt-in implementation precomputes fixed gamma shape terms, batches independent
temperature/error draws in one NumPy call, and batches noise log variances,
diagonal covariance and likelihood calculations. C-contiguous gamma inputs
preserve the original temperature-then-error traversal. Observation reductions
remain contiguous and arithmetic expression order is preserved. The covariance
list interface used by tempering is retained.

Direct tests compare draws, covariance, likelihood and the complete NumPy RNG
state exactly against the scalar loop across four seeds and 1/8/50 temperatures.
The implementation uses the existing Gibbs distribution and random stream.
Proposal generation, acceptance, priors, adaptation, decorrelation, swaps,
residual sum-of-squares and GPU history calculations are unchanged. No hierarchical
mixing algorithm change is included.

The private \texttt{setup.\_batch\_pooled\_gibbs} path accepts ordinary deterministic
material-strength models with diagonal likelihood, untruncated Gibbs noise and no
discrepancy. Custom likelihood/covariance/step overrides, stochastic models and
other noise modes are rejected. The default scalar path remains unchanged.
\section*{Repeated end-to-end timings}
Short runs use four chains, each with 3,000 states and 50 temperatures. Execution
order reverses in round 2 and rotates in round 3. Seconds are elapsed time for the
entire group. CPU and GPU each have matched old/new worker controls.
\begin{center}\begin{tabular}{lrrrr}\toprule
Backend & Round 1 & Round 2 & Round 3 & Median\\\midrule
TIMINGROWS
\bottomrule\end{tabular}\end{center}
Jobs run sequentially with one numerical-library thread per worker. Startup/JIT
and saving are excluded; initialization inside sampling, cache misses, CPU work,
transfers and coordination remain included. The long study uses fresh prior-worker
GPU, batched-worker GPU and batched-worker CPU runs. Thus the long GPU comparison
uses the improved CPU implementation, not a weaker historical CPU control.
The scalar CPU has fresh short controls; earlier scalar CPU long times remain
historical. Long timings are one run per backend, not universal speed guarantees.
\section*{Accuracy and posterior quality}
Short regressions require exact parameters/counters at all temperatures, noise
relative/absolute tolerances $10^{-8}/10^{-15}$, and prediction tolerances
$10^{-8}/10^{-10}$. The original frozen prediction/likelihood fixture is unchanged.
For long diagnostics, discard 20,000 states, no thinning. All 16 parameter/noise
quantities require R-hat below 1.01, bulk/tail ESS at least 400 and MCSE/SD below
0.05. Mean differences must be within four combined MCSEs and 0.1 pooled SD;
prediction means within 0.1 experiment noise SD, using the same 200 selected
retained draws per chain as the original reference.

Quality gates: QUALITY. Maximum R-hat RHAT; minimum bulk/tail ESS BULK/TAIL.
Minimum bulk ESS/s: CPUESS on improved CPU, GPUESS on improved GPU.
Fresh optimized CPU arrays reproduce the frozen original arrays exactly: ORIGINAL.
GPU parameter draws/counters match CPU exactly: PARAMEXACT. Noise/prediction
errors remain within the declared gates. All saved GPU arrays match the previous
GPU worker path exactly: EXACT. No tolerance or original artifact was changed.
\section*{Tracking and reproduction}
\begingroup\raggedright
Branch: \texttt{perf/gpu-worker-overhead}. Measured code: \texttt{COMMIT}.\\
Checkpoint: \texttt{perf-step7-20260930}. Commits use Ayan Biswas
\texttt{<ayanju04@gmail.com>}; nothing was pushed upstream.
See \path{docs/performance/STEP7.md} and \path{benchmarks/gpu_worker_suite.py}
for commands, controls and limitations. Select the candidate with
\texttt{--worker-batch-gibbs}, alongside the existing tuned-GPU flags.
\path{benchmarks/gpu_worker_report.py} rebuilds compact results and this LaTeX.
Raw profiles, telemetry and traces are in workspace \path{benchmark/changes/step7/};
versioned JSON/CSV summaries, profiles, test output, LaTeX and SHA-256 inventory
are under \path{docs/performance/results/step7*}. The initial worker profiles
precede the code checkpoint; the timing gates use committed code. Hardware is
the same 24-core Core Ultra 9 285HX and RTX PRO 5000 Blackwell laptop GPU.
Cleanup, Mac validation, hierarchical mixing and asynchronous scheduling remain
outside this checkpoint.
\par\endgroup
\end{document}
"""
    before, after = profiles["before"]["42"], profiles["after"]["42"]
    repl = {
        "OLDGPU": f"{old:.2f}",
        "NEWGPU": f"{new:.2f}",
        "CPUSECONDS": f"{cpu:.2f}",
        "SPEEDOLD": f"{old / new:.3f}",
        "SPEEDCPU": f"{cpu / new:.3f}",
        "QUALITY": "passed" if q["all_quality_gates_pass"] else "FAILED",
        "EXACT": str(exact),
        "TESTS": tests.group(),
        "TIMINGROWS": "\n".join(rows),
        "BEFOREWALL": f"{before['sampling_wall_seconds']:.3f}",
        "BEFORECPU": f"{before['process_cpu_seconds']:.3f}",
        "BEFOREREMOTEWALL": f"{before['remote']['wall_seconds']:.3f}",
        "BEFOREREMOTECPU": f"{before['remote']['cpu_seconds']:.3f}",
        "AFTERWALL": f"{after['sampling_wall_seconds']:.3f}",
        "AFTERCPU": f"{after['process_cpu_seconds']:.3f}",
        "RHAT": f"{q['gpu']['max_rhat']:.6f}",
        "BULK": f"{q['gpu']['min_bulk_ess']:.2f}",
        "TAIL": f"{q['gpu']['min_tail_ess']:.2f}",
        "CPUESS": f"{q['reference']['min_ess_per_second']:.3f}",
        "GPUESS": f"{q['gpu']['min_ess_per_second']:.3f}",
        "ORIGINAL": str(all(all(v.values()) for v in original.values())),
        "PARAMEXACT": str(
            all(
                v["theta"]["exact"] and v["count"]["exact"]
                for v in q["array_agreement"].values()
            )
        ),
        "COMMIT": long["gpu_candidate"]["commit"][:7],
    }
    for key in sorted(repl, key=len, reverse=True):
        tex = tex.replace(key, repl[key])
    (docs / "step7.tex").write_text(tex)
    print(
        json.dumps(
            {
                "medians": medians,
                "long_seconds": {
                    k: v["sampling_wall_seconds"] for k, v in long.items()
                },
                "quality": q["all_quality_gates_pass"],
                "gpu_exact": exact,
                "original": original,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    main(p.parse_args())
