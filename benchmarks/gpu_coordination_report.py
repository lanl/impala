"""Summarize exclusive coordinator time and nested CUDA stage measurements."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
from gpu_resident_suite import ROOT, compare


def distribution(values):
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def analyze(folder, result):
    wall = result["sampling_wall_seconds"]
    stats = result["sampling"]
    summary = {"wall_s": wall}
    if not result["coordination_timing"]:
        return summary
    trace = json.loads((folder / "coordination-timeline.json").read_text())
    detail = stats["coordination"]
    model_s = sum(row["model_end"] - row["model_start"] for row in trace)
    assembly_s = sum(row["assembly_s"] + row["split_s"] for row in trace)
    summary.update(
        parent_cpu_s=stats["parent_process_cpu_seconds"],
        model_wall_s=model_s,
        assembly_split_s=assembly_s,
        poll_wait_s=detail["poll_wait_seconds"],
        receive_decode_s=detail["receive_decode_seconds"],
        reply_send_s=stats["send_encode_seconds"],
        batch_size=distribution([row["batch"] for row in trace]),
        calls=len(trace),
    )
    skew, latest, first, latencies = [], [], [], []
    for row in trace:
        starts = [request["request_start"] for request in row["requests"]]
        assert len(starts) == row["workers"]
        skew.append(max(starts) - min(starts))
        latest.append(row["model_start"] - max(starts))
        first.append(row["model_start"] - min(starts))
        latencies.extend(row["model_start"] - value for value in starts)
    summary["request_start_spread_s"] = distribution(skew)
    summary["latest_request_to_model_s"] = distribution(latest)
    summary["earliest_request_to_model_s"] = distribution(first)
    summary["per_request_to_model_s"] = distribution(latencies)
    workers = {}
    for seed in result["seeds"]:
        timing = json.loads(
            (folder / f"pooled_{seed}/worker-timing.json").read_text()
        )
        timing["outside_remote_wall_s"] = (
            timing["sampling_wall_seconds"] - timing["remote"]["wall_seconds"]
        )
        timing["outside_remote_cpu_s"] = (
            timing["process_cpu_seconds"] - timing["remote"]["cpu_seconds"]
        )
        workers[str(seed)] = timing
    summary["workers"] = workers
    accounted = (
        model_s
        + assembly_s
        + summary["poll_wait_s"]
        + summary["receive_decode_s"]
        + summary["reply_send_s"]
    )
    summary["other_wall_s"] = wall - accounted
    assert summary["other_wall_s"] >= 0
    if result["gpu_timing"]:
        stages = json.loads((folder / "gpu-stages.json").read_text())
        assert len(stages) == len(trace)
        for stage, event in zip(stages, trace):
            assert stage["batch"] == event["batch"]
        totals = {
            key: sum(stage[key] for stage in stages)
            for key in stages[0]
            if key.endswith("_s")
        }
        summary["gpu_stage_totals_s"] = totals
        device_s = totals["history_device_s"] + totals["interpolation_device_s"]
        summary["device_kernel_s"] = device_s
        summary["device_kernel_fraction_of_wall"] = device_s / wall
        summary["model_other_wall_s"] = model_s - device_s
        assert model_s >= device_s
        summary["geometry_misses"] = sum(
            stage["geometry_miss"] for stage in stages
        )
        seen = set()
        repeated_misses = 0
        first_miss_s = repeated_miss_s = 0.0
        for stage in stages:
            if stage["geometry_miss"]:
                if stage["batch"] in seen:
                    repeated_misses += 1
                    repeated_miss_s += stage["geometry_host_and_upload_s"]
                else:
                    first_miss_s += stage["geometry_host_and_upload_s"]
            seen.add(stage["batch"])
        summary["unique_batch_sizes"] = len(seen)
        summary["repeated_geometry_misses"] = repeated_misses
        summary["first_geometry_miss_s"] = first_miss_s
        summary["repeated_geometry_miss_s"] = repeated_miss_s
        summary["resident_cache"] = stats["resident_evaluation"]

    return summary


def main(base):
    base = base.resolve()
    runs = base / "runs"
    results = json.loads((runs / "suite.json").read_text())
    summaries = {
        name: analyze(runs / name, result) for name, result in results.items()
    }
    medians = {
        mode: statistics.median(
            summaries[f"standard_{mode}_{i}"]["wall_s"] for i in range(3)
        )
        for mode in ("plain", "coordination", "stages")
    }
    agreement = json.loads((runs / "agreement.json").read_text())
    assert all(
        v["exact"]
        for r in agreement.values()
        for seed in r.values()
        for v in seed.values()
    )
    step8 = compare(
        base.parent / "step8/short/gpu_candidate_0", runs / "standard_plain_0"
    )
    assert all(v["exact"] for seed in step8.values() for v in seed.values())
    result = {
        "measured_commit": results["standard_plain_0"]["commit"],
        "medians_s": medians,
        "runs": summaries,
        "instrumentation_all_arrays_exact": True,
        "step8_all_arrays_exact": True,
        "agreement": agreement,
        "step8_agreement": step8,
    }
    docs = ROOT / "docs/performance/results"
    (docs / "step9-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    shutil.copyfile(base / "tests-gpu.log", docs / "step9-tests.txt")

    # Plot nonoverlapping parent wall-time components, with CUDA kernel intervals
    # subtracted from model wall time. These are not hardware occupancy metrics.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["standard_stages_1", "large_stages"]
    fields = [
        ("poll_wait_s", "Wait for requests"),
        ("receive_decode_s", "Receive/decode"),
        ("assembly_split_s", "Assemble/split"),
        ("device_kernel_s", "CUDA kernels"),
        ("model_other_wall_s", "Other model time*"),
        ("reply_send_s", "Send replies"),
        ("other_wall_s", "Other coordinator"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = np.zeros(2)
    for field, label in fields:
        values = np.array([
            summaries[name][field] / summaries[name]["wall_s"] * 100
            for name in names
        ])
        ax.barh([0, 1], values, left=left, label=label)
        left += values
    ax.set_yticks(
        [0, 1], ["4 chains, 50 temperatures", "8 chains, 50 temperatures"]
    )
    ax.set_xlabel("Percentage of instrumented group wall time")
    ax.set_xlim(0, 100)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8
    )
    fig.tight_layout()
    fig.savefig(docs / "step9-breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    table = []
    for mode in ("plain", "coordination", "stages"):
        times = [summaries[f"standard_{mode}_{i}"]["wall_s"] for i in range(3)]
        table.append(
            mode
            + " & "
            + " & ".join(f"{v:.3f}" for v in [*times, medians[mode]])
            + r"\\"
        )
    stages_rows = []
    standard, large = [summaries[name] for name in names]
    for field, label in fields:
        stages_rows.append(
            label
            + " & "
            + " & ".join(f"{s[field]:.3f}" for s in [standard, large])
            + r"\\"
        )
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs,graphicx,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\emergencystretch=3em
\title{Impala Step 9: GPU execution and coordination}
\author{Measurement of the step-8 experimental path}
\date{September 30, 2026}
\begin{document}\maketitle
\section*{Experiment and accuracy}
This is instrumentation only. Sampler, kernels, transfers and synchronous
scheduling policy remain unchanged. Four independent standard chains use seeds
42/123/456/789, 3,000 states and 50 temperatures. An exploratory larger-call
sweep uses eight independent chains, 1,000 states and the same 50 temperatures.
The additional seeds are 1011/1213/1415/1617. All saved outputs,
including full-temperature arrays, match exactly across instrumentation modes.
The standard plain run also matches the recorded step-8 GPU candidate exactly.
The original prediction fixture is checked at every launch. TESTRESULT

There is no new long posterior-quality study: this experiment measures unchanged
computation. Short eight-chain runs do not establish convergence or an
ESS/s comparison with the standard workload. No production default changes.
An initial 200-temperature attempt failed before sampling: the existing ladder
yields a negative Gamma shape during warm-up. Its logs are preserved; larger
batches instead use more independent chains without changing the sampler.
\section*{Instrumentation overhead}
No cProfile is used. Coordination timers separate polling, receive/decode,
assembly/split, model evaluation and sends, and record worker request timestamps.
Detailed GPU timing additionally uses CUDA events and stage synchronization.
Three standard rounds rotate mode order, with no overlapping runs. Seconds below
are elapsed group sampling time; startup/JIT and saving are excluded, while cache
misses, CPU work, transfers and coordination remain included.
\begin{center}\begin{tabular}{lrrrr}\toprule
Mode & Round 1 & Round 2 & Round 3 & Median\\\midrule
TIMINGROWS
\bottomrule\end{tabular}\end{center}
Coordination median/plain ratio: COORDRATIO; detailed-stage median/plain ratio:
STAGERATIO. These ratios include timing variation, not just instrumentation cost.
Larger-call times (one run each): plain LARGEPLAIN s, coordination LARGECOORD s,
detailed stages LARGESTAGES s. They are exploratory, not repeated speed estimates.
\section*{Nonoverlapping coordinator wall-time breakdown}
The figure uses standard detailed-stage round 2 and the larger detailed-stage run.
Each bar totals elapsed group time. Device event durations are nested within model
wall time and subtracted from other model time; they are never counted twice.
\begin{center}\includegraphics[width=0.94\linewidth]{step9-breakdown.pdf}\end{center}
\noindent *Other model time includes host validation, cache preparation, transfers,
allocation, launches and synchronization outside measured history/interpolation
kernel intervals. It is not all avoidable overhead. Kernel time/wall time is not
hardware occupancy, and these fractions describe instrumented runs.
\begin{center}\begin{tabular}{lrr}\toprule
Component & 4 chains (s) & 8 chains (s)\\\midrule
BREAKDOWNROWS
\bottomrule\end{tabular}\end{center}
\section*{Interpretation}
In the illustrated standard run, CUDA history/interpolation kernels occupy
DEVICEPCT percent of elapsed time; request polling occupies POLLPCT percent,
and receiving/decoding plus sending replies occupy PIPEPCT percent. In the
larger-call run, kernel time accounts for LARGEDEVICEPCT percent. The detailed
JSON separates geometry preparation, parameter expansion/upload and output download.
Geometry preparation/upload takes GEOMETRYPCT percent of standard elapsed time
and LARGEGEOMETRYPCT percent in the larger-call run. The latter includes
FIRSTGEOMETRY s of first-observed cache misses and REPEATGEOMETRY s of repeated
misses. Its shorter duration makes first-use costs proportionally larger;
this is not proof of a sustained 24-percent geometry cost in long runs.
These measurements support examining the history kernel and model-side setup
before rewriting communication. Poll waiting cannot simply be removed: workers
must compute their next sampler transition after each result. Asynchronous
scheduling might overlap some work, but smaller GPU batches and new coordination
costs could offset that gain; no asynchronous speedup has been demonstrated.
\section*{Worker skew and request waiting}
In coordination-only standard round 2, request-start spread across workers has
median SKEWP50 ms and p95 SKEWP95 ms. Latest-request-to-model-start median is
LATESTP50 ms (p95 LATESTP95 ms). Combined parameter batch size averages BATCHMEAN
(p95 BATCHP95). Larger coordination-only requests average LARGEBATCH parameters.

A timestamp precedes worker serialization. Request-to-model latency therefore
includes serialization, pipes, scheduling, waiting for peers and assembly.
Polling waits include necessary sampler work. Worker request waits overlap each
other and the GPU; do not add them to parent wall time. Request spread is not
an additive cost or an asynchronous speedup estimate. Parent CPU time and each
worker's inside/outside-request wall and CPU time are retained in the JSON.
\section*{Reproduction and tracking}
\begingroup\small\raggedright
Measured commit: \texttt{COMMIT}. Branch: \texttt{perf/gpu-coordination-profile};
checkpoint: \texttt{perf-step9-20260930}. Authored by Ayan Biswas
\texttt{<ayanju04@gmail.com>}; no push. See \path{docs/performance/STEP9.md} for
interpretation and commands. \path{benchmarks/gpu_coordination_suite.py} runs
controls; \path{benchmarks/gpu_coordination_report.py} rebuilds this report.
JSON, plot PDF, LaTeX, tests and SHA-256 inventory are versioned under
\path{docs/performance/results/step9*}; raw traces and samples reside in workspace
\path{benchmark/changes/step9/}. Hardware: Core Ultra 9 285HX, RTX PRO 5000
Blackwell laptop GPU. Mac testing, hierarchical mixing and asynchronous scheduling
remain separate. See the accompanying Markdown findings for the next experiment.
\par\endgroup
\end{document}
"""
    c = summaries["standard_coordination_1"]
    repl = {
        "TIMINGROWS": "\n".join(table),
        "GEOMETRYPCT": f"{standard['gpu_stage_totals_s']['geometry_host_and_upload_s'] / standard['wall_s'] * 100:.1f}",
        "LARGEGEOMETRYPCT": f"{large['gpu_stage_totals_s']['geometry_host_and_upload_s'] / large['wall_s'] * 100:.1f}",
        "FIRSTGEOMETRY": f"{large['first_geometry_miss_s']:.3f}",
        "REPEATGEOMETRY": f"{large['repeated_geometry_miss_s']:.3f}",
        "DEVICEPCT": f"{standard['device_kernel_fraction_of_wall'] * 100:.1f}",
        "POLLPCT": f"{standard['poll_wait_s'] / standard['wall_s'] * 100:.1f}",
        "PIPEPCT": f"{(standard['receive_decode_s'] + standard['reply_send_s']) / standard['wall_s'] * 100:.1f}",
        "LARGEDEVICEPCT": f"{large['device_kernel_fraction_of_wall'] * 100:.1f}",
        "BREAKDOWNROWS": "\n".join(stages_rows),
        "COORDRATIO": f"{medians['coordination'] / medians['plain']:.3f}",
        "STAGERATIO": f"{medians['stages'] / medians['plain']:.3f}",
        "COMMIT": result["measured_commit"][:7],
        "TESTRESULT": (base / "tests-gpu.log").read_text().splitlines()[-1],
        "SKEWP50": f"{c['request_start_spread_s']['p50'] * 1000:.3f}",
        "SKEWP95": f"{c['request_start_spread_s']['p95'] * 1000:.3f}",
        "LATESTP50": f"{c['latest_request_to_model_s']['p50'] * 1000:.3f}",
        "LATESTP95": f"{c['latest_request_to_model_s']['p95'] * 1000:.3f}",
        "BATCHMEAN": f"{c['batch_size']['mean']:.1f}",
        "BATCHP95": f"{c['batch_size']['p95']:.1f}",
        "LARGEBATCH": f"{summaries['large_coordination']['batch_size']['mean']:.1f}",
        **{
            f"LARGE{mode.upper()}": f"{summaries['large_' + key]['wall_s']:.3f}"
            for mode, key in [
                ("plain", "plain"),
                ("coord", "coordination"),
                ("stages", "stages"),
            ]
        },
    }
    for key in sorted(repl, key=len, reverse=True):
        tex = tex.replace(key, repl[key])
    (docs / "step9.tex").write_text(tex)
    print(
        json.dumps(
            {
                "medians": medians,
                "standard_coordination": c,
                "standard_stages": standard,
                "large_stages": large,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    main(parser.parse_args().artifacts)
