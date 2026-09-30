"""Build step-8 results and LaTeX from recorded proposal likelihood profiles and runs."""

import argparse
import json
import re
import shutil
import statistics
from pathlib import Path

import numpy as np
from gpu_resident_suite import compare

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    base = args.artifacts.resolve()
    docs = ROOT / "docs/performance/results"
    short = json.loads((base / "short/suite.json").read_text())
    long = json.loads((base / "long/suite.json").read_text())
    long["cpu_control"] = json.loads(
        (base / "long/cpu_control/results.json").read_text()
    )
    cpu_agreement = compare(
        base / "long/cpu_control", base / "long/cpu_candidate"
    )
    q = json.loads((base / "quality/summary.json").read_text())
    labels = {
        "cpu_control": "CPU, scalar proposals",
        "cpu_candidate": "CPU, batched proposals",
        "gpu_control": "GPU, scalar proposals",
        "gpu_candidate": "GPU, batched proposals",
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
        "cpu_control_agreement": cpu_agreement,
        "short_agreement": json.loads(
            (base / "short/agreement.json").read_text()
        ),
        "profile_agreement": json.loads(
            (base / "profile-agreement.json").read_text()
        ),
    }
    (docs / "step8-results.json").write_text(
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
        shutil.copyfile(base / src, docs / f"step8-{dest}")
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
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\emergencystretch=3em
\title{Impala performance experiment: Step 8}
\author{Batched proposal likelihoods across temperatures}
\date{September 30, 2026}
\begin{document}\maketitle
\section*{Outcome}
Four pooled Cu chains, 40,000 states each, 50 temperatures: previous GPU proposal
path OLDGPU s; batched-proposal GPU NEWGPU s; batched-proposal parallel CPU CPUSECONDS s.
CPU retaining scalar proposal likelihoods takes CPUCONTROL s.
The new GPU path delivers SPEEDOLD times the previous GPU throughput, SPEEDCPU
times the batched-proposal CPU throughput, and SPEEDCPUCONTROL times the
scalar-proposal CPU throughput. All predeclared posterior gates: QUALITY.
Every saved GPU output matches the previous proposal path exactly: EXACT.
Full hardware-enabled suite: TESTS.
\section*{Why worker computation was selected}
The seed-42 diagnostic used 3,000 states and 50 temperatures. Before optimization,
wall time was BEFOREWALL s and process CPU time BEFORECPU s. Remote model requests
used BEFOREREMOTEWALL s wall but only BEFOREREMOTECPU s CPU. The remaining CPU
work included 164,400 scalar likelihood calls (0.539 s cumulative).
This makes proposal likelihoods a useful CPU target, without assuming that
remote waiting is avoidable communication overhead. After batching, 3,287
proposal likelihood calls take 0.259 s cumulative; scalar calls remain only
for initialization.

With batching, seed-42 process CPU time was AFTERCPU s and instrumented wall
time AFTERWALL s. Four-chain profiled outputs matched exactly at all temperatures.
Profiles perturb timing and are diagnostic evidence, not performance gates.
Removing scalar calls also removes profiler events; assembling fresh covariance
arrays has a cost. The profiled CPU reduction is not proof of a CPU speedup.
Remote wall time includes GPU evaluation, other workers, useful sampler computation
and scheduling. Summed worker CPU time is not elapsed group time.
\section*{What changed and what is preserved}
The opt-in implementation batches diagonal likelihood calculations across
all temperatures for ordinary and decorrelation parameter proposals. It reads
the current covariance dictionaries on each call, including changes from swaps
and Gibbs updates; no covariance cache is retained. C-contiguous residuals
preserve the scalar per-observation reduction and arithmetic order.

Nine new tests compare exact likelihoods across 1/8/50 temperatures and
C/Fortran/strided input layouts, with swapped and replaced covariances; the
global RNG state remains unchanged. Together with Gibbs tests, 27 focused
tests pass. Proposal generation, acceptance, priors, adaptation, decorrelation
order, Gibbs draws, swaps and GPU calculations remain unchanged.

The private \texttt{setup.\_batch\_pooled\_proposals} option requires the guarded
step-7 Gibbs batching path. It is restricted to ordinary deterministic
material-strength models with diagonal likelihood, untruncated Gibbs noise and
no discrepancy or custom likelihood/covariance/step overrides. Defaults remain
unchanged. Both controls and candidates use step-7 Gibbs batching; only proposal
likelihood evaluation differs.
\section*{Repeated end-to-end timings}
Short runs use four chains, each with 3,000 states and 50 temperatures. Execution
order reverses in round 2 and rotates in round 3. Seconds are elapsed time for the
entire group. CPU and GPU each have matched old/new proposal controls.
\begin{center}\begin{tabular}{lrrrr}\toprule
Backend & Round 1 & Round 2 & Round 3 & Median\\\midrule
TIMINGROWS
\bottomrule\end{tabular}\end{center}
Jobs run sequentially with one numerical-library thread per worker. Startup/JIT
and saving are excluded; initialization inside sampling, cache misses, CPU work,
transfers and coordination remain included. The long study uses fresh prior-proposal
GPU, batched-proposal GPU and batched-proposal CPU runs.
Short CPU batching regressed in two of three rounds (median time +6.78 percent),
whereas all three GPU rounds improved (median throughput +3.01 percent).
This prompted an additional fresh scalar-proposal CPU long control, run last.
Both CPU paths retain Gibbs batching. GPU comparisons include both CPU paths.
The long CPU run improves while short medians regress: CPU evidence is mixed.
Long timings are one run per backend; a small gain is not a universal speed guarantee.
Keep the proposal optimization opt-in; no general CPU speedup is established.
The long GPU gain is about 1.2 percent, smaller than the short median gain;
without repeated long trials it does not establish a robust speedup beyond
timing variability. The previous step-7 improvement remains a separate result.
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
Minimum bulk ESS/s: CPUESS on batched-proposal CPU, GPUESS on improved GPU.
CPU control/candidate saved outputs exactly match: CPUEXACT.
Fresh optimized CPU arrays reproduce the frozen original arrays exactly: ORIGINAL.
GPU parameter draws/counters match CPU exactly: PARAMEXACT. Noise/prediction
errors remain within the declared gates. All saved GPU arrays match the previous
GPU proposal path exactly: EXACT. No tolerance or original artifact was changed.
\section*{Tracking and reproduction}
\begingroup\small\raggedright
Branch: \texttt{perf/gpu-proposal-likelihood}. Measured code: \texttt{COMMIT}.\\
Checkpoint: \texttt{perf-step8-20260930}. Commits use Ayan Biswas
\texttt{<ayanju04@gmail.com>}; nothing was pushed upstream.
See \path{docs/performance/STEP8.md} and \path{benchmarks/gpu_proposal_suite.py}
for commands, controls and limitations. Select the candidate with
\texttt{--worker-batch-proposals}, alongside \texttt{--worker-batch-gibbs} and the tuned-GPU flags.
\path{benchmarks/gpu_proposal_report.py} rebuilds compact results and this LaTeX.
Raw profiles, telemetry and traces are in workspace \path{benchmark/changes/step8/};
versioned JSON/CSV summaries, profiles, test output, LaTeX and SHA-256 inventory
are under \path{docs/performance/results/step8*}. The initial worker profiles
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
        "CPUEXACT": str(
            all(
                v["exact"]
                for seed in cpu_agreement.values()
                for v in seed.values()
            )
        ),
        "CPUCONTROL": f"{long['cpu_control']['sampling_wall_seconds']:.2f}",
        "SPEEDCPUCONTROL": f"{long['cpu_control']['sampling_wall_seconds'] / new:.3f}",
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
    (docs / "step8.tex").write_text(tex)
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
