#!/usr/bin/env python3
### Author: Alex Most
"""
Unified PTW calibration script.

Behavior:
- Select calibration mode with --mode pooled or --mode hier
- Loads supported data groups if they are present in YAML:
    QS_SHPB, S200F, Z, RMI
- Lets you exclude any present group with:
    --exclude-qs-shpb
    --exclude-s200f
    --exclude-z
    --exclude-rmi
- Uses the YAML `models:` block for the main PTW material model.
- Appends RMI to the main stress-strain block, matching the legacy behavior.
- Keeps Z as separate vector experiment blocks.
- In hierarchical mode, creates experiment_*.png plots.
- If --pooled-overlay-dir is supplied in hierarchical mode, overlays the pooled
  best-SSE curve on each experiment_*.png plot.
- Use --seed to set random seed for both numpy and random. eg. --seed 42 will set np.random.seed(42) and random.seed(42).
"""

import argparse
import json
import os
import pickle
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401
import yaml
from scipy.stats import qmc

import impala
import impala.superCal.post_process as pp  # noqa: F401
from impala import superCal as sc
from impala.superCal.post_process import (
    get_outcome_predictions_impala,
    total_temperature_swaps,
)


def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the ptw-bp repo root.
    Expected repo-root markers:
      - impala/
      - Helper_Functions/
    """
    start = start.resolve()

    for p in [start] + list(start.parents):
        if (p / "impala").is_dir() and (p / "Helper_Functions").is_dir():
            return p

    raise RuntimeError(
        f"Could not find repo root above {start}. "
        "Expected directories named 'impala' and 'Helper_Functions'."
    )


ROOT = find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(ROOT))

PHYS = ROOT / "impala" / "physics"
sys.path.insert(0, str(PHYS))


np.seterr(under="ignore", over="ignore", divide="ignore", invalid="ignore")


# ----------------------------------------------------------------------
# Supported model classes
# ----------------------------------------------------------------------

MODEL_KINDS = {
    "flow_stress_model": [
        "Constant_Yield_Stress",
        "JC_Yield_Stress",
        "PTW_Yield_Stress",
        "PTWbp_Yield_Stress",
        "Stein_Flow_Stress",
    ],
    "melt_model": [
        "Constant_Melt_Temperature",
        "Linear_Melt_Temperature",
        "Quadratic_Melt_Temperature",
        "Cubic_Melt_Temperature",
        "BGP_Melt_Temperature",
    ],
    "shear_model": [
        "Constant_Shear_Modulus",
        "Linear_Cold_PW_Shear_Modulus",
        "Quadratic_Cold_PW_Shear_Modulus",
        "Simple_Shear_Modulus",
        "BGP_PW_Shear_Modulus",
        "Stein_Shear_Modulus",
    ],
    "specific_heat_model": [
        "Constant_Specific_Heat",
        "Linear_Specific_Heat",
        "Quadratic_Specific_Heat",
        "Cubic_Specific_Heat",
        "Piecewise_Linear_Specific_Heat",
        "Piecewise_Quadratic_Specific_Heat",
        "Piecewise_Cubic_Specific_Heat",
    ],
    "density_model": [
        "Constant_Density",
        "Linear_Density",
        "Quadratic_Density",
        "Cubic_Density",
    ],
}


DEFAULT_MODELS = {
    "flow_stress_model": "PTW_Yield_Stress",
    "melt_model": "Cubic_Melt_Temperature",
    "shear_model": "BGP_PW_Shear_Modulus",
    "specific_heat_model": "Cubic_Specific_Heat",
    "density_model": "Cubic_Density",
}


SUPPORTED_GROUPS = ("QS_SHPB", "S200F", "Z", "RMI")


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------

def seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def expand_home(path_str: str) -> Path:
    return Path(path_str.replace("HOME", os.path.expanduser("~")))


def subsample_mask(n, target):
    if n <= target:
        return np.ones(n, dtype=bool)

    step = max(int(np.floor(n / target)),1)

    idx1 = np.linspace(1, n, n)
    keep = (idx1 % step) == 0
    return keep.astype(bool)


def log_to_gamma(values, base="exp"):
    if base == "exp":
        return np.exp(values)
    if base == "10":
        return np.power(10.0, values)
    raise ValueError(f"Unsupported gamma_log_base={base}")


def convert_lgamma_to_gamma_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if "lgamma" in out.columns:
        base = cfg.get("settings", {}).get("gamma_log_base", "exp")
        out["gamma"] = log_to_gamma(out["lgamma"].to_numpy(), base=base)
        out = out.drop(columns=["lgamma"])
    return out


# ----------------------------------------------------------------------
# Model selection and registration
# ----------------------------------------------------------------------

def get_model_choices(cfg):
    model_cfg = dict(DEFAULT_MODELS)
    model_cfg.update(cfg.get("models", {}))

    for key, allowed in MODEL_KINDS.items():
        name = model_cfg[key]

        if name not in allowed:
            raise ValueError(
                f"Invalid {key}: {name}. Allowed values are: {allowed}"
            )

        if not hasattr(impala.physics, name):
            raise AttributeError(
                f"{name} is listed for {key}, but it does not exist in physical_models_vec.py"
            )

    return model_cfg


def validate_model_constants_and_bounds(cfg, model_cfg):
    test_model = impala.physics.MaterialModel(
        flow_stress_model=getattr(impala.physics, model_cfg["flow_stress_model"]),
        melt_model=getattr(impala.physics, model_cfg["melt_model"]),
        shear_modulus_model=getattr(impala.physics, model_cfg["shear_model"]),
        specific_heat_model=getattr(impala.physics, model_cfg["specific_heat_model"]),
        density_model=getattr(impala.physics, model_cfg["density_model"]),
    )

    required_params = set(test_model.get_parameter_list())
    required_consts = set(test_model.get_constants_list())

    bounds = set(cfg["bounds_ptw"].keys())
    consts = set(cfg["consts_ptw"].keys())

    missing_model_params = sorted(required_params - bounds - consts)
    missing_model_consts = sorted(required_consts - consts)
    extra_bounds = sorted(bounds - required_params)

    if missing_model_params:
        raise ValueError(
            "The selected PTW model choices require these parameters, but they are "
            f"missing from both bounds_ptw and consts_ptw: {missing_model_params}"
        )

    if missing_model_consts:
        raise ValueError(
            "The selected PTW model choices require these constants, but they are "
            f"missing from consts_ptw: {missing_model_consts}"
        )

    if extra_bounds:
        print(
            "[WARN] bounds_ptw contains parameters not used by the selected main model: "
            f"{extra_bounds}"
        )

    fixed_model_params = sorted(required_params & consts)
    calibrated_model_params = sorted(required_params & bounds)

    print("calibrated PTW parameters:", calibrated_model_params)
    print("fixed PTW parameters:", fixed_model_params)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def has_data_group(cfg, group_name):
    files = cfg.get("data_files", {}).get(group_name, [])
    return files is not None and len(files) > 0


def warn_unsupported_groups(cfg):
    groups = list(cfg.get("data_files", {}).keys())
    unsupported = [g for g in groups if g not in SUPPORTED_GROUPS]
    for g in unsupported:
        print(f"WARNING: Unsupported data_files group '{g}' found in config. It will be ignored.")
    return unsupported


def load_curve_group(cfg, group_name, temps_key, edots_key, stress_divisor=None):
    """
    Load one stress-strain style group.

    stress_divisor:
      None -> do not change stress
      1e5 -> divide stress column by 1e5, matching legacy Z/RMI behavior
    """

    PDATA = expand_home(cfg["paths"]["path_to_data"])

    files = cfg.get("data_files", {}).get(group_name, [])
    if not files:
        return [], [], [], []

    temps = list(cfg.get(temps_key, []))
    edots = list(cfg.get(edots_key, []))

    if not (len(files) == len(temps) == len(edots)):
        raise ValueError(
            f"{group_name} length mismatch: "
            f"files={len(files)}, {temps_key}={len(temps)}, {edots_key}={len(edots)}"
        )

    dat = []
    for f in files:
        arr = np.loadtxt(PDATA / f, dtype=np.float64)
        if stress_divisor is not None:
            arr = arr.copy()
            arr[:, 1] /= stress_divisor
        dat.append(arr)

    return dat, temps, edots, list(files)


def load_main_stress_strain(cfg, include_qs_shpb=True, include_s200f=True):
    dat_all = []
    temps_all = []
    edots_all = []
    files_all = []

    if include_qs_shpb:
        dat, temps, edots, files = load_curve_group(
            cfg, "QS_SHPB", "temps_qs_shpb", "edots_qs_shpb"
        )
        if dat:
            print(f"loaded {len(dat)} QS_SHPB experiments")
            dat_all.extend(dat)
            temps_all.extend(temps)
            edots_all.extend(edots)
            files_all.extend(files)

    if include_s200f:
        dat, temps, edots, files = load_curve_group(
            cfg, "S200F", "temps_s200f", "edots_s200f"
        )
        if dat:
            print(f"loaded {len(dat)} S200F experiments")
            dat_all.extend(dat)
            temps_all.extend(temps)
            edots_all.extend(edots)
            files_all.extend(files)

    return dat_all, temps_all, edots_all, files_all


def load_z(cfg):
    dat_z, temps_z, edots_z, z_files = load_curve_group(
        cfg, "Z", "temps_z", "edots_z", stress_divisor=1e5
    )

    if dat_z:
        print(f"loaded {len(dat_z)} Z experiments")

    return dat_z, temps_z, edots_z, z_files


def load_rmi(cfg):
    dat_rmi, temps_rmi, edots_rmi, rmi_files = load_curve_group(
        cfg, "RMI", "temps_rmi", "edots_rmi", stress_divisor=1e5
    )

    if dat_rmi:
        print(f"loaded {len(dat_rmi)} RMI experiments")

    return dat_rmi, temps_rmi, edots_rmi, rmi_files


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------

def build_main_model(cfg, dat_all, temps, edots, pooled: bool, model_cfg):
    nsamp = cfg["settings"]["nsamp"]

    strain_hist_list = [np.asarray(v)[:, 0] for v in dat_all]
    inds = [subsample_mask(len(s), nsamp) for s in strain_hist_list]

    model = sc.ModelMaterialStrength(
        temps=np.array(temps, dtype=float),
        edots=np.array(edots, dtype=float) * 1e-6,
        consts=cfg["consts_ptw"],
        strain_histories=[strain_hist_list[j][inds[j]] for j in range(len(dat_all))],
        flow_stress_model=model_cfg["flow_stress_model"],
        melt_model=model_cfg["melt_model"],
        shear_model=model_cfg["shear_model"],
        specific_heat_model=model_cfg["specific_heat_model"],
        density_model=model_cfg["density_model"],
        pool=pooled,
        s2="gibbs",
    )

    return model, inds


def build_z_models(cfg, dat_z, temps_z, edots_z, pooled: bool):
    if not dat_z:
        return [], []

    dat_all_z = [
        np.repeat(arr[0, :].reshape(1, -1), 100, axis=0)
        for arr in dat_z
    ]

    strain_hist_list_z = [np.asarray(v)[:, 0] for v in dat_all_z]
    z_stress = [np.asarray(v)[:, 1] for v in dat_all_z]

    density = cfg["z_density"]
    if len(density) != len(dat_all_z):
        raise ValueError(
            "Expected len(z_density) == number of Z experiments, "
            f"got {len(density)} vs {len(dat_all_z)}"
        )

    c = cfg["consts_ptw"]

    consts_ptw_z = [
        {
            "beta": c["beta"],
            "matomic": c["matomic"],
            "Tmelt0": c["Tmelt0"],
            "rho0": c["rho0"],
            "rho_Z": density[j],
            "Cv0": 0.0001,
            "chi": 0,
            "G0": c["G0"],
            "rho_0": c.get("rho_0", 1.844),
            "Tm_0": c["Tm_0"],
            "rho_m": c["rho_m"],
            "gamma_1": c["gamma_1"],
            "gamma_2": c["gamma_2"],
            "gamma_3": c["gamma_3"],
            "q3": c["q3"],
            "q2": c["q2"],
            "alpha": c["alpha"],
            "tm0": c.get("tm0", 0.0),
            "tm1": c.get("tm1", 0.0),
            "tm2": c.get("tm2", 0.0),
            "tm3": c.get("tm3", 0.0),
        }
        for j in range(len(dat_all_z))
    ]

    models_z = [
        sc.ModelMaterialStrength(
            temps=np.array(temps_z, dtype=float)[j],
            edots=np.array(edots_z, dtype=float)[j] * 1e-6,
            consts=consts_ptw_z[j],
            strain_histories=[strain_hist_list_z[j]],
            flow_stress_model="PTW_Yield_Stress",
            melt_model="Cubic_Melt_Temperature",
            shear_model="BGP_PW_Shear_Modulus",
            specific_heat_model="Constant_Specific_Heat",
            density_model="Constant_Density",
            pool=pooled,
            s2="gibbs",
        )
        for j in range(len(dat_all_z))
    ]

    return models_z, z_stress


def build_setup(cfg, model, dat_all, inds, include_z, models_z, z_stress, pooled: bool):
    bounds = cfg["bounds_ptw"]
    setup = sc.CalibSetup(bounds, sc.constraints_ptw)

    yobs_main = np.hstack([
        np.asarray(dat_all[j])[inds[j], 1]
        for j in range(len(dat_all))
    ]).astype(np.float64)

    s2_ind_main = np.hstack([
        [j] * len(np.asarray(dat_all[j])[inds[j], 1])
        for j in range(len(dat_all))
    ]).astype(int)

    theta_ind_main = s2_ind_main

    sd_main_default = cfg["settings"]["sd_est"]["main_default"]

    if pooled:
        sd_est_main = np.full(len(dat_all), sd_main_default, dtype=np.float64)
    else:
        sd_main_last = cfg["settings"]["sd_est"].get("last_main", sd_main_default)
        sd_est_main = np.array(
            [sd_main_default] * (len(dat_all) - 1) + [sd_main_last],
            dtype=np.float64,
        )

    s2_df_main = np.array([50] * len(dat_all), dtype=int)

    setup.addVecExperiments(
        yobs=yobs_main,
        model=model,
        sd_est=sd_est_main,
        s2_df=s2_df_main,
        s2_ind=s2_ind_main,
        theta_ind=theta_ind_main,
    )

    if include_z:
        if z_stress is None or len(z_stress) != len(models_z):
            raise ValueError("z_stress must be provided for all Z models")

        for j, mz in enumerate(models_z):
            yobs_z = np.asarray(z_stress[j], dtype=np.float64)
            n_obs_z = yobs_z.shape[0]

            setup.addVecExperiments(
                yobs=yobs_z,
                model=mz,
                sd_est=np.array([cfg["settings"]["sd_est"]["z_default"]], dtype=np.float64),
                s2_df=np.array([50], dtype=int),
                s2_ind=np.zeros(n_obs_z, dtype=int),
                theta_ind=np.zeros(n_obs_z, dtype=int),
            )

    temp_cfg = cfg["settings"]["tempering"]
    setup.setTemperatureLadder(
        temp_cfg["base"] ** np.arange(temp_cfg["n"]),
        start_temper=temp_cfg["start_temper"],
    )

    mcmc = cfg["settings"]["mcmc"]
    tau = float(mcmc.get("start_tau_theta", -4.0))

    setup.setMCMC(
        nmcmc=mcmc["nmcmc"],
        nburn=mcmc["nburn"],
        thin=mcmc["thin"],
        decor=mcmc["decor"],
        start_tau_theta=tau,
    )

    if not hasattr(setup, "start_tau_theta"):
        setup.start_tau_theta = tau

    if not pooled:
        p = setup.p
        pri = cfg["settings"]["priors"]
        setup.setHierPriors(
            theta0_prior_mean=np.repeat(pri["theta0_prior_mean"], p),
            theta0_prior_cov=np.eye(p) * (pri["theta0_prior_scale"] ** 2),
            Sigma0_prior_df=p + pri["Sigma0_prior_df_offset"],
            Sigma0_prior_scale=np.eye(p) * (pri["Sigma0_prior_scale"] ** 2),
        )

    return setup


def run_calibration(setup, pooled: bool):
    with np.errstate(under="ignore", over="ignore", divide="ignore", invalid="ignore"):
        if pooled:
            return sc.calibPool(setup)
        return sc.calibHier(setup)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------

def scale_draws_to_native(raw_draws, setup):
    l_bounds = np.array(pd.DataFrame(setup.bounds.values()))[:, 0]
    u_bounds = np.array(pd.DataFrame(setup.bounds.values()))[:, 1]
    return qmc.scale(np.asarray(raw_draws), l_bounds, u_bounds)


def sse_by_draw(preds, setup):
    total = None

    for b, pred in enumerate(preds):
        pred = np.asarray(pred)
        y = np.asarray(setup.ys[b]).reshape(1, -1)

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        block_sse = ((pred - y) ** 2).mean(axis=1)

        if total is None:
            total = block_sse
        else:
            total = total + block_sse

    return total


def mape_one_theta(preds, setup):
    vals = []

    for b, pred in enumerate(preds):
        pred = np.asarray(pred)
        if pred.ndim == 2:
            pred = pred[0]

        y = np.asarray(setup.ys[b])
        vals.append(np.mean(np.abs(pred - y) / y))

    return 100 * float(np.mean(vals))


def save_best_from_native_draws(native_draws, setup, cfg, results_dir: Path):
    preds = get_outcome_predictions_impala(setup, theta_input=native_draws)["outcome_draws"]
    sse = sse_by_draw(preds, setup)
    best_idx = int(np.argmin(sse))

    median_theta = np.median(native_draws, axis=0).reshape(1, -1)
    median_preds = get_outcome_predictions_impala(setup, theta_input=median_theta)["outcome_draws"]

    best_preds = [p[best_idx].reshape(1, -1) for p in preds]

    cols = list(setup.bounds.keys())
    rows = []

    rows.append({
        **dict(zip(cols, median_theta.flatten())),
        "method": "parent_median",
        "sse": float(sse_by_draw(median_preds, setup)[0]),
        "mape": mape_one_theta(median_preds, setup),
    })

    rows.append({
        **dict(zip(cols, native_draws[best_idx])),
        "method": "parent_minsse",
        "sse": float(sse[best_idx]),
        "mape": mape_one_theta(best_preds, setup),
    })

    best_internal = pd.DataFrame(rows)
    best_internal.to_csv(results_dir / "best_internal.csv", index=False)

    best_human = convert_lgamma_to_gamma_df(best_internal, cfg)
    best_human.to_csv(results_dir / "best.csv", index=False)


def save_draws_pooled(setup, out, results_dir: Path, cfg: dict):
    ensure_dir(results_dir)

    raw = pd.DataFrame(out.theta[:, 0, :], columns=setup.bounds.keys())
    raw.to_csv(results_dir / "theta_draws.csv", index=False)

    n_draws = raw.shape[0]
    start = min(25000, max(0, n_draws // 2))
    uu = np.arange(start, n_draws, 10, dtype=int)

    native_arr = scale_draws_to_native(raw.to_numpy()[uu, :], setup)
    native = pd.DataFrame(native_arr, columns=setup.bounds.keys())
    native = convert_lgamma_to_gamma_df(native, cfg)
    native.to_csv(results_dir / "theta_draws_native.csv", index=False)

    save_best_from_native_draws(native_arr, setup, cfg, results_dir)


def save_draws_hier(setup, out, results_dir: Path, cfg: dict, include_z: bool, models_z_cnt: int):
    ensure_dir(results_dir)

    theta0 = pd.DataFrame(out.theta0[:, 0, :], columns=setup.bounds.keys())
    theta0.to_csv(results_dir / "theta0_draws.csv", index=False)

    parent_raw = sc.chol_sample_1per_constraints(
        out.theta0[:, 0],
        out.Sigma0[:, 0],
        setup.checkConstraints,
        setup.bounds_mat,
        setup.bounds.keys(),
        setup.bounds,
        setup.models[0].constants,
    )

    parent = pd.DataFrame(parent_raw, columns=setup.bounds.keys())
    parent.to_csv(results_dir / "parent_draws.csv", index=False)

    nexp_main = setup.ns2[0]
    theta_exp_main = [out.theta[0][:, 0, j, :] for j in range(nexp_main)]

    with open(results_dir / "thetai_draws.pkl", "wb") as f:
        pickle.dump(theta_exp_main, f, pickle.HIGHEST_PROTOCOL)

    block_idx = 1
    if include_z and models_z_cnt > 0:
        for k in range(models_z_cnt):
            if block_idx >= len(out.theta):
                break

            arr = out.theta[block_idx][:, 0, 0, :]
            with open(results_dir / f"thetai_draws_z{k + 1}.pkl", "wb") as f:
                pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)

            block_idx += 1

    n_draws = parent.shape[0]
    start = min(20000, max(0, n_draws // 2))
    uu = np.arange(start, n_draws, 10, dtype=int)

    native_arr = scale_draws_to_native(parent.to_numpy()[uu, :], setup)
    native = pd.DataFrame(native_arr, columns=setup.bounds.keys())
    native = convert_lgamma_to_gamma_df(native, cfg)
    native.to_csv(results_dir / "theta_draws_native.csv", index=False)

    save_best_from_native_draws(native_arr, setup, cfg, results_dir)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def parameter_trace_plot_labeled(df_trace: pd.DataFrame, ylim=(0, 1)):
    arr = df_trace.to_numpy()
    labels = list(df_trace.columns)
    n, d = arr.shape

    fig_h = max(3.0, 1.8 * d)
    fig, axes = plt.subplots(d, 1, figsize=(14, fig_h), sharex=True)

    if d == 1:
        axes = [axes]

    palette = plt.get_cmap("Set1")

    for i, ax in enumerate(axes):
        ax.plot(range(n), arr[:, i], color=palette(i), linewidth=1.2)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_ylabel(
            labels[i],
            fontsize=10,
            rotation=0,
            labelpad=55,
            ha="right",
            va="center",
        )
        ax.set_yticks([0.0, 0.5, 1.0])

    axes[-1].set_xlabel("Iteration")
    fig.subplots_adjust(left=0.12, hspace=0.35)

    return fig, axes


def load_pooled_theta(results_pooled: Path, setup, cfg: dict, which: str = "parent_minsse") -> np.ndarray:
    """
    Load a pooled parameter vector with shape (1, p) from a pooled results directory.

    Prefers best_internal.csv because it preserves lgamma.
    Falls back to best.csv and converts gamma -> lgamma if needed.
    """

    preferred = results_pooled / "best_internal.csv"
    fallback = results_pooled / "best.csv"

    best_path = preferred if preferred.exists() else fallback

    if not best_path.exists():
        raise FileNotFoundError(f"Missing pooled best file at {preferred} or {fallback}")

    dfb = pd.read_csv(best_path)

    if "method" not in dfb.columns:
        raise ValueError(f"{best_path} has no 'method' column")

    row = dfb.loc[dfb["method"] == which]

    if row.empty:
        raise ValueError(f"Could not find method='{which}' in {best_path}")

    theta_cols = list(setup.bounds.keys())
    row = row.iloc[0].copy()

    if "lgamma" in theta_cols and "lgamma" not in dfb.columns and "gamma" in dfb.columns:
        base = cfg.get("settings", {}).get("gamma_log_base", "exp")
        gamma_value = float(row["gamma"])

        if base == "exp":
            row["lgamma"] = np.log(gamma_value)
        elif base == "10":
            row["lgamma"] = np.log10(gamma_value)
        else:
            raise ValueError(f"Unsupported gamma_log_base={base}")

    return row[theta_cols].to_numpy(dtype=float).reshape(1, -1)


def make_all_plots(
    cfg,
    setup,
    out,
    results_dir: Path,
    plots_dir: Path,
    pooled: bool,
    include_z: bool,
    include_rmi: bool,
    dat_rmi=None,
    pooled_overlay_dir: Path = None,
):
    ensure_dir(plots_dir)

    # -----------------------
    # observed_data_rmi.png
    # -----------------------
    if include_rmi and dat_rmi and len(dat_rmi) > 0:
        try:
            plt.close("all")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

            for j, arr in enumerate(dat_rmi):
                ax.plot(
                    arr[:, 0],
                    arr[:, 1],
                    linewidth=1.5,
                    label=f"RMI {j + 1}" if len(dat_rmi) > 1 else "RMI",
                )

            ax.set_xlabel("Strain")
            ax.set_ylabel("Stress (Mbar)")
            ax.set_title("Observed RMI stress-strain")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "observed_data_rmi.png", dpi=200)
            plt.close("all")

        except Exception as e:
            print(f"[WARN] Could not create observed_data_rmi.png: {e}")

    # -----------------------
    # tempering.png
    # -----------------------
    try:
        plt.close("all")
        total_temperature_swaps(out, setup)
        plt.tight_layout()
        plt.savefig(plots_dir / "tempering.png", dpi=200)
        plt.close("all")

    except Exception as e:
        print(f"[WARN] Could not create tempering.png: {e}")

    # -----------------------
    # experiment_*.png and experiment_zmachine_*.png
    # hierarchical only
    # -----------------------
    if not pooled:
        try:
            df_parent = pd.read_csv(results_dir / "parent_draws.csv")
            n_draws = df_parent.shape[0]
            start = min(20000, max(0, n_draws // 2))
            uu = np.arange(start, n_draws, 10, dtype=int)

            parent_native = scale_draws_to_native(df_parent.to_numpy()[uu, :], setup)

            PARENT_Y = get_outcome_predictions_impala(
                setup,
                theta_input=parent_native,
            )["outcome_draws"]

            QUANTS_PARENT_Y = [
                np.quantile(PARENT_Y[j], [0.025, 0.5, 0.975], axis=0)
                for j in range(len(PARENT_Y))
            ]

            # Optional pooled overlay.
            POOLED_Y = None
            if pooled_overlay_dir is not None:
                try:
                    theta_pooled_best = load_pooled_theta(
                        Path(pooled_overlay_dir),
                        setup,
                        cfg,
                        which="parent_minsse",
                    )
                    POOLED_Y = get_outcome_predictions_impala(
                        setup,
                        theta_input=theta_pooled_best,
                    )["outcome_draws"]
                    print(f"loaded pooled overlay from {pooled_overlay_dir}")

                except Exception as e:
                    print(f"[WARN] Could not load pooled overlay from {pooled_overlay_dir}: {e}")

            with open(results_dir / "thetai_draws.pkl", "rb") as f:
                theta_exp_list = pickle.load(f)

            THETAi_Y = [
                get_outcome_predictions_impala(
                    setup,
                    theta_input=scale_draws_to_native(theta_exp_list[j][uu, :], setup),
                )["outcome_draws"]
                for j in range(len(theta_exp_list))
            ]

            QUANTS_THETAi_Y = [
                np.quantile(THETAi_Y[j][0], [0.025, 0.5, 0.975], axis=0)
                for j in range(len(THETAi_Y))
            ]

            n_exp_main = len(np.unique(setup.s2_ind[0]))
            s2_inds = setup.s2_ind[0]

            for exp_ind in range(n_exp_main):
                plt.close("all")
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))

                mask = np.where(s2_inds == exp_ind)[0]
                x = setup.models[0].meas_strain_histories[exp_ind]
                yobs = setup.ys[0][mask]

                # Model stores edot internally as 1/us because the script multiplies YAML 1/s by 1e-6.
                # Multiply back by 1e6 so the title shows the original YAML strain rate in 1/s.
                exp_edot = setup.models[0].edots[exp_ind] * 1e6
                exp_temp = setup.models[0].temps[exp_ind]

                ax.fill_between(
                    x,
                    QUANTS_PARENT_Y[0][0, mask],
                    QUANTS_PARENT_Y[0][2, mask],
                    color="lightgray",
                    zorder=1,
                    label="Hier parent 95% interval",
                )

                ax.plot(
                    x,
                    QUANTS_PARENT_Y[0][1, mask],
                    color="darkgray",
                    linewidth=2,
                    label="Hier parent median",
                    zorder=2,
                )

                ax.fill_between(
                    x,
                    QUANTS_THETAi_Y[exp_ind][0, mask],
                    QUANTS_THETAi_Y[exp_ind][2, mask],
                    color="pink",
                    zorder=3,
                    label="Hier experiment-specific 95% interval",
                )

                ax.plot(
                    x,
                    QUANTS_THETAi_Y[exp_ind][1, mask],
                    color="red",
                    linewidth=2,
                    label="Hier experiment-specific median",
                    zorder=4,
                )

                if POOLED_Y is not None:
                    ax.plot(
                        x,
                        POOLED_Y[0][0, mask],
                        color="blue",
                        linewidth=2,
                        label="Pooled best SSE",
                        zorder=5,
                    )

                ax.scatter(
                    x,
                    yobs,
                    color="black",
                    s=12,
                    label="Data",
                    zorder=6,
                )

                ax.set_xlabel("Strain")
                ax.set_ylabel("Stress")
                ax.set_title(
                    f"Experiment {exp_ind}: strain rate = {exp_edot:.3g} 1/s, T = {exp_temp:.0f} K"
                )
                ax.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"experiment_{exp_ind}.png", dpi=200)
                plt.close("all")

            if include_z:
                z_idx = 1

                while True:
                    zfile = results_dir / f"thetai_draws_z{z_idx}.pkl"

                    if not zfile.exists():
                        break

                    with open(zfile, "rb") as f:
                        theta_z = pickle.load(f)

                    THETA_Y_z = get_outcome_predictions_impala(
                        setup,
                        theta_input=scale_draws_to_native(theta_z[uu, :], setup),
                    )["outcome_draws"]

                    block = z_idx

                    if block >= len(THETA_Y_z):
                        break

                    q = np.quantile(THETA_Y_z[block], [0.025, 0.5, 0.975], axis=0)
                    x0 = setup.models[block].meas_strain_histories[0][0]
                    y0 = setup.ys[block][0]

                    exp_edot_z = setup.models[block].edots[0] * 1e6
                    exp_temp_z = setup.models[block].temps[0]

                    plt.close("all")
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

                    NUDGE = 1e-3
                    xx = np.array([x0 - NUDGE, x0, x0 + NUDGE])

                    ax.fill_between(
                        xx,
                        np.repeat(q[0].flatten()[0], 3),
                        np.repeat(q[2].flatten()[0], 3),
                        color="pink",
                        zorder=1,
                        label="Hier Z experiment-specific 95% interval",
                    )

                    ax.plot(
                        [x0],
                        [q[1].flatten()[0]],
                        marker="o",
                        color="red",
                        label="Hier Z experiment-specific median",
                        zorder=2,
                    )

                    ax.scatter(
                        [x0],
                        [y0],
                        color="black",
                        label="Data",
                        zorder=3,
                    )

                    ax.set_xlim(x0 - 20 * NUDGE, x0 + 20 * NUDGE)
                    ax.set_xlabel("Strain")
                    ax.set_ylabel("Stress")
                    ax.set_title(
                        f"Z Experiment {z_idx}: strain rate = {exp_edot_z:.3g} 1/s, T = {exp_temp_z:.0f} K"
                    )
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"experiment_zmachine_{z_idx}.png", dpi=200)
                    plt.close("all")

                    z_idx += 1

        except Exception as e:
            print(f"[WARN] Could not create experiment-level plots: {e}")

    # -----------------------
    # best_all.png
    # pooled + hierarchical
    # -----------------------
    try:
        plt.close("all")

        OBS = setup.ys[0]

        if pooled:
            df_draws = pd.read_csv(results_dir / "theta_draws.csv")
            start_target = 25000
            median_label = "Pooled posterior median"
            best_label = "Pooled best SSE"
        else:
            df_draws = pd.read_csv(results_dir / "parent_draws.csv")
            start_target = 20000
            median_label = "Hier parent median"
            best_label = "Hier parent best SSE"

        n_draws = df_draws.shape[0]
        start = min(start_target, max(0, n_draws // 2))
        uu = np.arange(start, n_draws, 10, dtype=int)

        native = scale_draws_to_native(df_draws.to_numpy()[uu, :], setup)

        pred_median = get_outcome_predictions_impala(
            setup,
            theta_input=np.median(native, axis=0).reshape(1, -1),
        )["outcome_draws"]

        THETA_Y = get_outcome_predictions_impala(
            setup,
            theta_input=native,
        )["outcome_draws"]

        s2_inds = setup.s2_ind[0]
        n_exp_main = len(np.unique(s2_inds))

        pred_median_main = np.hstack(
            pred_median[0][0, np.where(s2_inds == i)[0]]
            for i in range(n_exp_main)
        ).reshape(-1)

        parent_sse = sum([
            (
                (
                    THETA_Y[0][:, np.where(s2_inds == i)[0]]
                    - setup.ys[0][np.where(s2_inds == i)]
                ) ** 2
            ).mean(axis=1)
            for i in range(n_exp_main)
        ])

        best_idx = int(np.argmin(parent_sse))

        pred_minsse_main = np.hstack([
            THETA_Y[0][best_idx, np.where(s2_inds == i)[0]]
            for i in range(n_exp_main)
        ]).reshape(-1)

        n = len(OBS)
        pred_median_main = pred_median_main[:n]
        pred_minsse_main = pred_minsse_main[:n]

        _fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.plot(
            np.arange(n),
            pred_median_main,
            linewidth=2,
            label=median_label,
            zorder=2,
        )
        ax.plot(
            np.arange(n),
            pred_minsse_main,
            linewidth=2,
            label=best_label,
            zorder=3,
        )
        ax.scatter(
            np.arange(n),
            OBS[:n],
            color="black",
            s=10,
            label="Data",
            zorder=5,
        )

        ax.set_xlabel("Index")
        ax.set_ylabel("Stress")
        ax.set_title("Prediction for All Experiments, Main Block")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "best_all.png", dpi=200)
        plt.close("all")

    except Exception as e:
        print(f"[WARN] Could not create best_all.png: {e}")

    # -----------------------
    # trace.png
    # -----------------------
    try:
        plt.close("all")

        if pooled:
            df_trace = pd.read_csv(results_dir / "theta_draws.csv")
        else:
            df_trace = pd.read_csv(results_dir / "parent_draws.csv")

        parameter_trace_plot_labeled(df_trace)
        plt.savefig(plots_dir / "trace.png", dpi=200)
        plt.close("all")

    except Exception as e:
        print(f"[WARN] Could not create trace.png: {e}")

# ----------------------------------------------------------------------
# Run metadata
# ----------------------------------------------------------------------

def effective_groups_from_flags(cfg, include_qs, include_s200f, include_z, include_rmi):
    groups = []

    if include_qs and has_data_group(cfg, "QS_SHPB"):
        groups.append("qs-shpb")

    if include_s200f and has_data_group(cfg, "S200F"):
        groups.append("s200f")

    if include_z:
        groups.append("z")

    if include_rmi:
        groups.append("rmi")

    return groups


def build_run_paths(cfg, mode: str, effective_groups):
    root = expand_home(cfg["paths"]["path_to_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = "-".join(effective_groups) if effective_groups else "none"
    run_name = f"{timestamp}_{mode}__{group_tag}"
    run_dir = root / "runs" / run_name

    return run_dir, run_dir / "results", run_dir / "plots"


def write_run_files(run_dir: Path, cfg: dict, args, model_cfg, included_groups, unsupported_groups):
    ensure_dir(run_dir)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "seed": args.seed,
        "config_path": str(args.config.resolve()),
        "included_groups": included_groups,
        "unsupported_groups_ignored": unsupported_groups,
        "model_spec_main": model_cfg,
        "settings_snapshot": deepcopy(cfg.get("settings", {})),
        "bounds_ptw": deepcopy(cfg.get("bounds_ptw", {})),
        "consts_ptw": deepcopy(cfg.get("consts_ptw", {})),
        "z_density": deepcopy(cfg.get("z_density", [])),
        "pooled_overlay_dir": str(args.pooled_overlay_dir) if args.pooled_overlay_dir else None,
    }

    with open(run_dir / "run_settings.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    resolved_cfg = deepcopy(cfg)
    resolved_cfg.setdefault("run_metadata", {})
    resolved_cfg["run_metadata"].update(manifest)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved_cfg, fh, sort_keys=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified PTW calibration script.")
    ap.add_argument("--mode", choices=["hier", "pooled"], required=True)
    ap.add_argument("--config", type=Path, default=Path("be_config.yaml"))
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--exclude-qs-shpb", action="store_true")
    ap.add_argument("--exclude-s200f", action="store_true")
    ap.add_argument("--exclude-z", action="store_true")
    ap.add_argument("--exclude-rmi", action="store_true")

    ap.add_argument(
        "--pooled-overlay-dir",
        type=Path,
        default=None,
        help="Optional pooled results directory to overlay pooled best-SSE curve on hierarchical experiment plots.",
    )

    args = ap.parse_args()

    print(f"seed={args.seed}")
    seeds(args.seed)

    pooled = args.mode == "pooled"

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    unsupported_groups = warn_unsupported_groups(cfg)

    include_qs = has_data_group(cfg, "QS_SHPB") and not args.exclude_qs_shpb
    include_s200f = has_data_group(cfg, "S200F") and not args.exclude_s200f
    include_z_requested = has_data_group(cfg, "Z") and not args.exclude_z
    include_rmi_requested = has_data_group(cfg, "RMI") and not args.exclude_rmi

    print(f"include_qs_shpb={include_qs}")
    print(f"include_s200f={include_s200f}")
    print(f"include_z={include_z_requested}")
    print(f"include_rmi={include_rmi_requested}")

    model_cfg = get_model_choices(cfg)
    validate_model_constants_and_bounds(cfg, model_cfg)

    print("using physical model choices:")
    for key, value in model_cfg.items():
        print(f"  {key}: {value}")

    if pooled:
        print("using pooled mode")
    else:
        for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            if os.environ.get(v, None) not in (None, "1"):
                print(
                    f"WARNING: {v}={os.environ.get(v)}. "
                    "Hierarchical mode may run very slowly unless this is set to 1."
                )
        print("using hierarchical mode")

    dat_all, temps, edots, _main_files = load_main_stress_strain(
        cfg,
        include_qs_shpb=include_qs,
        include_s200f=include_s200f,
    )

    dat_z, temps_z, edots_z, _z_files = ([], [], [], [])
    if include_z_requested:
        dat_z, temps_z, edots_z, _z_files = load_z(cfg)

    use_z = bool(include_z_requested and len(dat_z) > 0)

    dat_rmi, temps_rmi, edots_rmi, _rmi_files = ([], [], [], [])
    if include_rmi_requested:
        dat_rmi, temps_rmi, edots_rmi, _rmi_files = load_rmi(cfg)

    use_rmi = bool(include_rmi_requested and len(dat_rmi) > 0)

    if use_rmi:
        k = int(cfg.get("settings", {}).get("rmi_replicate", 5))
        dat_rmi_rep = dat_rmi * k
        temps_rmi_rep = temps_rmi * k
        edots_rmi_rep = edots_rmi * k

        dat_all = dat_all + dat_rmi_rep
        temps = temps + temps_rmi_rep
        edots = edots + edots_rmi_rep

        print(f"appended RMI to main block with replicate={k}")

    if len(dat_all) == 0:
        raise ValueError(
            "No main-block experiments selected. "
            "At least one of QS_SHPB, S200F, or RMI must be included."
        )

    effective_groups = effective_groups_from_flags(
        cfg,
        include_qs=include_qs,
        include_s200f=include_s200f,
        include_z=use_z,
        include_rmi=use_rmi,
    )

    run_dir, results_dir, plots_dir = build_run_paths(cfg, args.mode, effective_groups)
    ensure_dir(run_dir)
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    write_run_files(
        run_dir,
        cfg,
        args,
        model_cfg=model_cfg,
        included_groups=effective_groups,
        unsupported_groups=unsupported_groups,
    )

    model_main, inds = build_main_model(
        cfg,
        dat_all,
        temps,
        edots,
        pooled=pooled,
        model_cfg=model_cfg,
    )

    models_z, z_stress = [], None
    if use_z:
        models_z, z_stress = build_z_models(cfg, dat_z, temps_z, edots_z, pooled=pooled)

    setup = build_setup(
        cfg,
        model_main,
        dat_all,
        inds,
        include_z=use_z,
        models_z=models_z,
        z_stress=z_stress,
        pooled=pooled,
    )

    out = run_calibration(setup, pooled=pooled)

    if pooled:
        save_draws_pooled(setup, out, results_dir, cfg)
    else:
        save_draws_hier(
            setup,
            out,
            results_dir,
            cfg,
            include_z=use_z,
            models_z_cnt=len(models_z),
        )

    make_all_plots(
        cfg=cfg,
        setup=setup,
        out=out,
        results_dir=results_dir,
        plots_dir=plots_dir,
        pooled=pooled,
        include_z=use_z,
        include_rmi=use_rmi,
        dat_rmi=dat_rmi,
        pooled_overlay_dir=args.pooled_overlay_dir,
    )

    print("Done.")
    print("Run directory:", str(run_dir))
    print("Results in:", str(results_dir))
    print("Plots in:", str(plots_dir))


if __name__ == "__main__":
    main()
