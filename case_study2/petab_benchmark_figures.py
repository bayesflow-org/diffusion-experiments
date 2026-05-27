#%%
# PEtab benchmark model summary table/figure generation.
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from case_study2.helper_visualize import plot_model_comparison_grid


PROBLEM_NAME = "Beer_MolBioSystems2014"
BASE = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = BASE / "metrics"
LUSTRE_METRICS_DIR = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/metrics")
METRICS_DIR = LUSTRE_METRICS_DIR if LUSTRE_METRICS_DIR.exists() else DEFAULT_METRICS_DIR
PLOTS_DIR = BASE / "plots"

METRIC_COLUMNS = [
    "nrmse",
    "posterior_calibration_error",
    "posterior_contraction",
    "c2st",
    "posterior_tarp",
]


def _to_float(value) -> float:
    """Robust scalar conversion for numpy/jax/tensor-like objects."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return np.nan
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return np.nan


def canonical_model_name(name: str) -> str:
    """Collapse versioned model names like '<base>_3' back to '<base>'."""
    m = re.match(r"^(.*)_(\d+)$", str(name))
    if m:
        return m.group(1)
    return str(name)


def extract_run_id(name: str) -> int:
    """Extract run id from '<base>_<run_id>', default to 0."""
    m = re.match(r"^(.*)_(\d+)$", str(name))
    if m:
        return int(m.group(2))
    return 0


def is_fusion_model(name: str) -> bool:
    return "_ft" in str(name)


def normalize_model_key(name: str) -> str:
    """Map fusion-transformer model keys to the corresponding base model."""
    return str(name).replace("_ft_ema", "_ema").replace("_ft", "")


def load_diffusion_metrics(problem_name: str) -> pd.DataFrame:
    """Load all diffusion metric pickles in a single DataFrame."""
    rows = []
    metric_files = sorted(METRICS_DIR.glob(f"{problem_name}_metrics_*.pkl"))
    if not metric_files:
        logging.warning(f"No diffusion metrics found for {problem_name}")
        return pd.DataFrame()

    for metric_file in metric_files:
        model_name_from_file = metric_file.stem.replace(f"{problem_name}_metrics_", "", 1)
        with open(metric_file, "rb") as f:
            metric_list = pickle.load(f)

        for row in metric_list:
            raw_model_name = row.get("model", model_name_from_file)
            row = dict(row)
            row["run_id"] = extract_run_id(raw_model_name)
            row["model"] = canonical_model_name(raw_model_name)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].map(_to_float)

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").fillna(0).astype(int)
    # Use distance to ideal C2ST=0.5, so higher is better consistently.
    df["c2st"] = 0.5 + (df["c2st"] - 0.5).abs()
    return df


def load_mcmc_metrics(problem_name: str) -> pd.DataFrame:
    def _extract_mcmc_run_id_from_path(fp: Path) -> int:
        m = re.search(r"_mcmc_metrics_(\d+)$", fp.stem)
        return int(m.group(1)) if m else 0

    run_files = sorted(METRICS_DIR.glob(f"{problem_name}_mcmc_metrics_*.csv"))
    legacy_file = METRICS_DIR / f"{problem_name}_mcmc_metrics.csv"
    if run_files:
        parts = []
        for fp in run_files:
            with open(fp, "rb") as f:
                part = pd.read_csv(f, index_col=0)
            # Ensure run id is available for per-run files even if not present in the CSV content.
            if "run_id" not in part.columns:
                part["run_id"] = _extract_mcmc_run_id_from_path(fp)
            else:
                run_id_from_file = _extract_mcmc_run_id_from_path(fp)
                part["run_id"] = pd.to_numeric(part["run_id"], errors="coerce").fillna(run_id_from_file)
            parts.append(part)
        mcmc_df = pd.concat(parts, ignore_index=True, sort=False)
    elif legacy_file.exists():
        with open(legacy_file, "rb") as f:
            mcmc_df = pd.read_csv(f, index_col=0)
    else:
        logging.info(f"No MCMC metrics found for {problem_name}")
        return pd.DataFrame()

    if mcmc_df.empty:
        return mcmc_df

    for col in METRIC_COLUMNS:
        if col not in mcmc_df.columns:
            mcmc_df[col] = np.nan
        mcmc_df[col] = mcmc_df[col].map(_to_float)

    if "run_id" not in mcmc_df.columns:
        mcmc_df["run_id"] = mcmc_df["model"].map(extract_run_id)
    mcmc_df["run_id"] = pd.to_numeric(mcmc_df["run_id"], errors="coerce").fillna(0).astype(int)
    mcmc_df["model"] = mcmc_df["model"].map(canonical_model_name)
    mcmc_df["c2st"] = 0.5 + (mcmc_df["c2st"] - 0.5).abs()
    return mcmc_df


def build_combined_run_metrics(diffusion_df: pd.DataFrame, mcmc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build raw observation table across all runs and both summary networks.

    - `model_key` removes `_ft` suffix so TimeSeries and FusionTransformer
      variants are grouped under the same model family.
    - No per-run averaging is performed here.
    """
    if diffusion_df.empty and mcmc_df.empty:
        return pd.DataFrame()

    # For MCMC, each chain is considered a run-equivalent unit.
    # Encode chain id into run_id so chain rows stay separate downstream.
    if not mcmc_df.empty and "chain_idx" in mcmc_df.columns:
        chain_idx = pd.to_numeric(mcmc_df["chain_idx"], errors="coerce").fillna(0).astype(int)
        base_run_id = pd.to_numeric(mcmc_df["run_id"], errors="coerce").fillna(0).astype(int)
        mcmc_df = mcmc_df.copy()
        mcmc_df["run_id"] = base_run_id * 1000 + chain_idx

    all_df = pd.concat([diffusion_df, mcmc_df], ignore_index=True, sort=False)
    all_df["model_key"] = all_df["model"].map(normalize_model_key)
    all_df["run_id"] = pd.to_numeric(all_df["run_id"], errors="coerce").fillna(0).astype(int)
    return all_df[["run_id", "model_key", "sampler"] + METRIC_COLUMNS].copy()


def _safe_standardize(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def compute_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c2st_distance = (df["c2st"] - 0.5).abs()
    tarp_distance = df["posterior_tarp"].abs()
    score = (
            -_safe_standardize(df["nrmse"]).fillna(0.0)
            - _safe_standardize(df["posterior_calibration_error"]).fillna(0.0)
            + _safe_standardize(df["posterior_contraction"]).fillna(0.0)
            - _safe_standardize(c2st_distance).fillna(0.0)
            - _safe_standardize(tarp_distance).fillna(0.0)
    )
    df["score"] = score
    df["rank"] = df["score"].rank(ascending=False, method="min")
    return df


def add_rank_per_run(run_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite score/rank independently for each run.

    If both summary networks are present for the same
    (run_id, model_key, sampler), their score contributions are summed
    before ranking.
    """
    if run_df.empty:
        return run_df

    ranked_runs = []
    for run_id, group in run_df.groupby("run_id", sort=True):
        scored = compute_score_and_rank(group)
        merged = (
            scored.groupby(["run_id", "model_key", "sampler"], as_index=False)
            .agg(
                nrmse=("nrmse", "mean"),
                posterior_calibration_error=("posterior_calibration_error", "mean"),
                posterior_contraction=("posterior_contraction", "mean"),
                c2st=("c2st", "mean"),
                posterior_tarp=("posterior_tarp", "mean"),
                score=("score", "sum"),
            )
        )
        merged["rank"] = merged["score"].rank(ascending=False, method="min")
        ranked_runs.append(merged)

    return pd.concat(ranked_runs, ignore_index=True)


def summarize_over_runs(run_scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std over runs for each model/sampler, including rank stats."""
    if run_scored_df.empty:
        return run_scored_df

    summary = (
        run_scored_df.groupby(["model_key", "sampler"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            nrmse=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            posterior_calibration_error=("posterior_calibration_error", "mean"),
            posterior_calibration_error_std=("posterior_calibration_error", "std"),
            posterior_contraction=("posterior_contraction", "mean"),
            posterior_contraction_std=("posterior_contraction", "std"),
            c2st=("c2st", "mean"),
            c2st_std=("c2st", "std"),
            posterior_tarp=("posterior_tarp", "mean"),
            posterior_tarp_std=("posterior_tarp", "std"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
        )
    )
    summary = compute_score_and_rank(summary)
    return summary


def add_labels_and_order(summary_df: pd.DataFrame, diffusion_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    base_df = diffusion_df[~diffusion_df["model"].map(is_fusion_model)].copy()
    base_df["model_key"] = base_df["model"]
    order_map = (
        base_df[["model_key", "sampler"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(order=lambda x: range(len(x)))
    )

    display_names = base_df[["model_key", "sampler", "model"]].drop_duplicates()
    out = summary_df.merge(display_names, on=["model_key", "sampler"], how="left")
    out["model"] = out["model"].fillna(out["model_key"])
    out = (
        out.merge(order_map, on=["model_key", "sampler"], how="left")
        .sort_values(["order"], na_position="last", kind="stable")
        .drop(columns=["order"])
        .reset_index(drop=True)
    )

    out["family"] = out["model"].apply(
        lambda x: "Flow Matching" if "flow_matching" in x else (
            "Consistency Model" if "consistency" in x else (
                "Diffusion" if "diffusion" in x else "MCMC"
            )
        )
    )
    out = out.sort_values(["family", "rank_mean", "rank_std"], kind="stable").reset_index(drop=True)
    return out


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if pd.isna(mean):
        return "---"
    if pd.isna(std):
        return f"{mean:.{digits}f} (---)"
    if digits == 0:
        return f"{np.ceil(mean):.{digits}f} ({np.ceil(std):.{digits}f})"
    return f"{mean:.{digits}f} ({std:.{digits+1}f})"


def write_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        logging.warning("No summary rows to export.")
        return

    df = summary_df.copy()
    df = df.rename(
        columns={
            "family": "Family",
            "model": "Design Choice",
            "sampler": "Sampler",
        }
    )

    design_choice_map = {
        "ot_flow_matching": "OT",
        "flow_matching": "Uniform",
        "flow_matching_edm": r"$\rho=-0.6$",
        "diffusion_cosine_v": r"Cosine, VP, $\v$",
        "diffusion_cosine_F": r"Cosine, VP, $\mathbf{F}$",
        "diffusion_edm_vp": r"EDM, VP, $\mathbf{F}$",
        "diffusion_edm_ve": r"EDM, VE, $\mathbf{F}$",
        "diffusion_cosine_noise": r"Cosine, VP, $\epsilonb$",
        "diffusion_edm_vp_ema": r"EMA, EDM, VP, $\mathbf{F}$",
        "consistency_model": "Discrete",
        "stable_consistency_model": "Continuous",
        "MCMC": "-",
    }

    df["Design Choice"] = df["Design Choice"].map(design_choice_map).fillna(df["Design Choice"])
    df["Sampler"] = df["Sampler"].astype(str).str.upper()
    df["NRMSE"] = df.apply(lambda r: fmt_mean_std(r["nrmse"], r["nrmse_std"]), axis=1)
    df["Calibration Error"] = df.apply(
        lambda r: fmt_mean_std(r["posterior_calibration_error"], r["posterior_calibration_error_std"]),
        axis=1,
    )
    df["Contraction"] = df.apply(
        lambda r: fmt_mean_std(r["posterior_contraction"], r["posterior_contraction_std"]),
        axis=1,
    )
    df["C2ST"] = df.apply(lambda r: fmt_mean_std(r["c2st"], r["c2st_std"]), axis=1)
    df["TARP"] = df.apply(lambda r: fmt_mean_std(r["posterior_tarp"], r["posterior_tarp_std"]), axis=1)
    #df["Rank"] = df.apply(lambda r: fmt_mean_std(r["rank_mean"], r["rank_std"]), axis=1)
    #df["Rank_overall"] = df['rank']
    df["ranked"] = df["rank_mean"].rank(ascending=True, method="min")
    df["Rank"] = df.apply(lambda r: fmt_mean_std(r["ranked"], r["rank_std"], digits=0), axis=1)

    cols = [
        "Family",
        "Design Choice",
        "Sampler",
        "NRMSE",
        "Calibration Error",
        "Contraction",
        "C2ST",
        "TARP",
        "Rank",
        #"Rank_overall"
    ]

    latex_code = df[cols].to_latex(index=False, escape=False)
    with open(PLOTS_DIR / "metrics_table.tex", "w") as f:
        f.write(latex_code)


def main() -> None:
    logging.info(f"Using METRICS_DIR={METRICS_DIR}")
    diffusion_df = load_diffusion_metrics(PROBLEM_NAME)
    mcmc_df = load_mcmc_metrics(PROBLEM_NAME)

    run_level_df = build_combined_run_metrics(diffusion_df, mcmc_df)
    run_scored_df = add_rank_per_run(run_level_df)
    summary_df = summarize_over_runs(run_scored_df)
    summary_df = add_labels_and_order(summary_df, diffusion_df)

    if summary_df.empty:
        logging.info("No summary data available.")
        return

    write_table(summary_df)
    logging.info(f"Wrote table to {PLOTS_DIR / 'metrics_table.tex'}")

    # Keep plotting behavior: use run-mean values
    plot_df = summary_df.copy()
    plot_df["nrmse"] = plot_df["nrmse"]
    plot_df["posterior_calibration_error"] = plot_df["posterior_calibration_error"]
    plot_df["posterior_contraction"] = plot_df["posterior_contraction"]
    plot_df["c2st"] = plot_df["c2st"]
    plot_df["posterior_tarp"] = plot_df["posterior_tarp"]
    plot_df["rank"] = plot_df["rank"]
    logging.info(np.corrcoef(plot_df[["posterior_calibration_error", "c2st"]].values.T))
    plot_model_comparison_grid(plot_df, save_path=PLOTS_DIR, plot_shade=True)


if __name__ == "__main__":
    main()
