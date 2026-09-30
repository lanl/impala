"""Analyze four GPU pooled chains against the preserved original CPU study.

Use the diagnostics environment with ArviZ (benchmark/.venv/bin/python).
Acceptance thresholds are documented in docs/performance/STEP3.md.
"""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np

SEEDS = (42, 123, 456, 789)


def load(root):
    metas = []
    runs = []
    for seed in SEEDS:
        folder = root / f"pooled_{seed}"
        meta = json.loads((folder / "metadata.json").read_text())
        assert (
            meta["iterations"] == 40000
            and meta["temperatures"] == 50
            and meta["seed"] == seed
        )
        metas.append(meta)
        with np.load(folder / "draws.npz") as f:
            runs.append({k: f[k] for k in f.files})
    assert all(
        m["bounds"] == metas[0]["bounds"]
        and m["parameter_names"] == metas[0]["parameter_names"]
        for m in metas
    )
    bounds = np.array(metas[0]["bounds"])
    names = metas[0]["parameter_names"]
    theta = (
        np.stack([r["theta"][20000:] for r in runs])
        * (bounds[:, 1] - bounds[:, 0])
        + bounds[:, 0]
    )
    noise = np.sqrt(np.stack([r["s2"][20000:] for r in runs]))
    posterior = {name: theta[:, :, j] for j, name in enumerate(names)}
    posterior.update({
        f"sigma_exp{e}": noise[:, :, e] for e in range(noise.shape[-1])
    })
    diag = az.summary(
        az.from_dict(posterior=posterior), hdi_prob=0.95, round_to="none"
    )
    seconds = sum(m["sampling_seconds"] for m in metas)
    diag["ess_per_second"] = diag.ess_bulk / seconds
    diag["mcse_over_sd"] = diag.mcse_mean / diag.sd
    gate = (
        (diag.r_hat < 1.01)
        & (diag.ess_bulk >= 400)
        & (diag.ess_tail >= 400)
        & (diag.mcse_over_sd < 0.05)
    )
    summary = {
        "all_converged": bool(gate.all()),
        "max_rhat": float(diag.r_hat.max()),
        "min_bulk_ess": float(diag.ess_bulk.min()),
        "min_tail_ess": float(diag.ess_tail.min()),
        "min_ess_per_second": float(diag.ess_per_second.min()),
        "sampling_seconds": seconds,
        "median_sampling_seconds": float(
            np.median([m["sampling_seconds"] for m in metas])
        ),
        "passing_quantities": int(gate.sum()),
    }
    return metas, runs, posterior, diag, summary, noise


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rm, rr, rp, rd, rs, rn = load(args.reference)
    gm, gr, gp, gd, gs, gn = load(args.candidate)
    assert (
        rm[0]["bounds"] == gm[0]["bounds"]
        and rm[0]["parameter_names"] == gm[0]["parameter_names"]
    )
    for r, g in zip(rr, gr):
        np.testing.assert_array_equal(r["observed"], g["observed"])
        np.testing.assert_array_equal(
            r["experiment_index"], g["experiment_index"]
        )
    comparison = (
        rd[["mean", "sd", "mcse_mean"]]
        .add_prefix("cpu_")
        .join(gd[["mean", "sd", "mcse_mean"]].add_prefix("gpu_"))
    )
    delta = np.abs(comparison.gpu_mean - comparison.cpu_mean)
    comparison["combined_mcse_z"] = delta / np.sqrt(
        comparison.cpu_mcse_mean**2 + comparison.gpu_mcse_mean**2
    )
    pooled_sd = np.sqrt((comparison.cpu_sd**2 + comparison.gpu_sd**2) / 2)
    comparison["mean_difference_in_sd"] = delta / pooled_sd
    comparison["ess_per_second_ratio_historical_cpu"] = (
        gd.ess_per_second / rd.ess_per_second
    )
    for q in (0.025, 0.975):
        comparison[f"quantile_{q}_difference_in_sd"] = [
            abs(np.quantile(gp[k], q) - np.quantile(rp[k], q)) / pooled_sd[k]
            for k in comparison.index
        ]
    means_pass = bool(
        (
            (comparison.combined_mcse_z <= 4)
            & (comparison.mean_difference_in_sd <= 0.1)
        ).all()
    )
    rpred = np.concatenate([r["predictions"] for r in rr]).mean(axis=0)
    gpred = np.concatenate([r["predictions"] for r in gr]).mean(axis=0)
    noise = (rn.mean(axis=(0, 1)) + gn.mean(axis=(0, 1))) / 2
    pred_diff = np.abs(rpred - gpred) / noise[rr[0]["experiment_index"]]
    exact = {
        str(seed): bool(
            np.array_equal(r["theta"], g["theta"])
            and np.array_equal(r["s2"], g["s2"])
        )
        for seed, r, g in zip(SEEDS, rr, gr)
    }
    result = {
        "reference": rs,
        "gpu": gs,
        "posterior_means_pass": means_pass,
        "max_mean_combined_mcse_z": float(comparison.combined_mcse_z.max()),
        "max_mean_difference_in_sd": float(
            comparison.mean_difference_in_sd.max()
        ),
        "prediction_means_pass": bool(np.max(pred_diff) <= 0.1),
        "max_prediction_mean_difference_in_noise_sd": float(np.max(pred_diff)),
        "min_ess_per_second_ratio_historical_cpu": float(
            comparison.ess_per_second_ratio_historical_cpu.min()
        ),
        "max_ess_per_second_ratio_historical_cpu": float(
            comparison.ess_per_second_ratio_historical_cpu.max()
        ),
        "exact_cold_chains": exact,
        "array_agreement": {
            str(seed): {
                key: {
                    "exact": bool(np.array_equal(r[key], g[key])),
                    "max_absolute_difference": float(
                        np.max(np.abs(r[key] - g[key]))
                    ),
                }
                for key in ("theta", "s2", "count", "predictions")
            }
            for seed, r, g in zip(SEEDS, rr, gr)
        },
    }
    result["all_quality_gates_pass"] = bool(
        rs["all_converged"]
        and gs["all_converged"]
        and means_pass
        and result["prediction_means_pass"]
    )
    rd.to_csv(args.output / "original_diagnostics.csv")
    gd.to_csv(args.output / "gpu_diagnostics.csv")
    comparison.to_csv(args.output / "comparison.csv")
    np.savez_compressed(
        args.output / "prediction_comparison.npz",
        original=rpred,
        gpu=gpred,
        difference_in_noise_sd=pred_diff,
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
