# -*- coding: utf-8 -*-
"""
SHAP analysis for ML models

Requirements
-----------
- models_paper.py in PYTHONPATH 
- shap >= 0.42
- sklearn, numpy, pandas, matplotlib
"""

author = "Gerfried Millner"
version = "1.0.0"
date = "31.03.2026"
email = "gerfried.millner@ntu.edu.sg"
status = "Developing"

import os
import pickle

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# from models_paper import train_final_xgb_full

# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path: str) -> None:
    """Save Python object as pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------
# SHAP computation helpers
# ---------------------------------------------------------------------

def get_representative_background(
    X,
    strategy: str = "kmeans",
    n_clusters: int = 50,
    random_state: int = 42,
):
    """
    Get representative background for SHAP.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (already scaled).
    strategy : str
        'auto': k-means if n > 500, else full.
        'full': Use all data.
        'kmeans': Use k-means centroids (RECOMMENDED).
    n_clusters : int
        Number of k-means clusters.
    random_state : int
        Random seed.

    Returns
    -------
    background : np.ndarray
        Background data.
    method : str
        Method used (e.g., 'full' or 'kmeans_50').
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    n_samples = len(X_array)

    # Determine strategy
    if strategy == "auto":
        use_kmeans = n_samples > 500
    elif strategy == "kmeans":
        use_kmeans = True
    else:  # 'full'
        use_kmeans = False

    # Apply strategy
    if use_kmeans:
        n_clusters_actual = min(n_clusters, max(2, n_samples // 2))
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=random_state)
        background = kmeans.fit(X_array).cluster_centers_
        method = f"kmeans_{n_clusters_actual}"
    else:
        background = X_array
        method = "full"

    return background, method


def get_scaled_features_from_pipeline(model: Pipeline,
                                      X: pd.DataFrame) -> pd.DataFrame:
    """
    Pass X through the model's scaler to obtain the scaled feature matrix.
    """
    scaler = model.named_steps["scaler"]
    X_scaled_values = scaler.transform(X)  # shape (n_samples, n_features_scaled)

    n_features_scaled = X_scaled_values.shape[1]
    feature_names = list(X.columns[:n_features_scaled])

    X_scaled = pd.DataFrame(
        X_scaled_values,
        index=X.index,
        columns=feature_names,
    )
    return X_scaled

def compute_shap_xgb(
    model: Pipeline,
    X_scaled: pd.DataFrame,
):
    """
    Compute SHAP values for XGBoost via TreeExplainer.
    Assumes model is a Pipeline([... ('scaler', ...), ('xgb', XGBRegressor)]).
    """
    xgb = model.named_steps["xgb"]
    explainer = shap.TreeExplainer(
        xgb,
        data=X_scaled,
        feature_perturbation="interventional",
    )
    shap_values = explainer.shap_values(
        X_scaled,
        check_additivity=False,
    )
    # XGBRegressor returns (n_samples, n_features) for regression
    return shap_values, explainer

# ---------------------------------------------------------------------
# SHAP statistics & saving
# ---------------------------------------------------------------------

def create_shap_df(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Create summary DataFrame from SHAP values.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    std_abs_shap = np.abs(shap_values).std(axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "std_abs_shap": std_abs_shap,
        }
    )

    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df["cumulative_importance"] = df["mean_abs_shap"].cumsum()
    total_importance = df["mean_abs_shap"].sum()
    df["percent_cumulative"] = (
        df["cumulative_importance"] / total_importance * 100.0
    )

    return df

def save_shap_results(
    shap_values: np.ndarray,
    explainer,
    X_scaled: pd.DataFrame,
    output_dir: str,
    model_name: str,
    target_name: str,
) -> pd.DataFrame:
    """
    Save SHAP arrays, explainer, and SHAP summary DataFrame.
    """
    ensure_directory(output_dir)

    df_shap = create_shap_df(shap_values, X_scaled.columns.tolist())

    base = f"{model_name}_{target_name}".lower()
    pkl_path = os.path.join(output_dir, f"shap_{base}.pkl")
    csv_path = os.path.join(output_dir, f"shapsummary_{base}.csv")

    save_pickle(
        {
            "shap_values": shap_values,
            "feature_names": X_scaled.columns.tolist(),
            "df_shap": df_shap,
            "explainer": explainer,
        },
        pkl_path,
    )
    df_shap.to_csv(csv_path, index=False)
    print(f"SHAP results saved to {output_dir} for {model_name}, target={target_name}")
    return df_shap


# ---------------------------------------------------------------------
# Plotting utilities 
# ---------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_scaled: pd.DataFrame,
    max_display: int = 20,
    save_path: str | None = None,
    figsize=(10, 8),
):
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=X_scaled.columns.tolist(),
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Beeswarm plot saved to {save_path}")
    else:
        plt.show()
    return fig


def plot_shap_bar(
    shap_values: np.ndarray,
    X_scaled: pd.DataFrame,
    max_display: int = 20,
    save_path: str | None = None,
    figsize=(10, 8),
):
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=X_scaled.columns.tolist(),
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar plot saved to {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------
# XGB: full-data fit + SHAP
# ---------------------------------------------------------------------

def fit_full_xgb_and_compute_shap(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    shap_outdir: str = None,
    n_trials: int = 100,
    n_splits_inner: int = 5,
    rands: int = 42,
):
    """
    1) Tune XGB HPs on full dataset via inner CV (train_final_xgb_full).
    2) Fit final XGB pipeline on full dataset.
    3) Compute SHAP values on full dataset (TreeExplainer, scaled X).
    """
    from models_paper import train_final_xgb_full

    print(f"[XGB] Using X with shape {X.shape} for target={target_name}")

    xgb_model, xgb_study = train_final_xgb_full(
        X,
        y,
        groups=False,              # SHAP runs: no group-aware CV required here
        rands=rands,
        n_splits_inner=n_splits_inner,
        n_trials=n_trials,
    )

    # Scale X via pipeline's scaler
    X_scaled = get_scaled_features_from_pipeline(xgb_model, X)

    shap_values, explainer = compute_shap_xgb(
        model=xgb_model,
        X_scaled=X_scaled,
    )

    if shap_outdir is not None:
        ensure_directory(shap_outdir)
        save_pickle(xgb_model, os.path.join(shap_outdir, f"xgb_full_{target_name}.pkl"))
        save_pickle(xgb_study, os.path.join(shap_outdir, f"xgb_study_{target_name}.pkl"))
        df_shap = save_shap_results(shap_values, explainer, X_scaled, shap_outdir,
        model_name="xgb", target_name=target_name)
    else:
        df_shap = create_shap_df(shap_values, X_scaled.columns.tolist())
        
    return xgb_model, xgb_study, df_shap, X_scaled, shap_values

def shap_for_transfer_xgb(
    X_source: pd.DataFrame,
    y_source: pd.Series,
    X_target: pd.DataFrame,
    y_target: pd.Series,
    param_xgb: dict | None = None,
    rand_state: int = 42,
    n_splits: int = 5,
    output_dir: str = "results/shap_transfer",
    task_name: str = "target",
    save_results: bool = False,
    tune_n_splits_inner: int = 5,
    tune_n_trials: int = 50,
):
    """
    Compute SHAP values for a transfer-learning XGB model.

    1) If param_xgb is None, tune XGB hyperparameters on the SOURCE dataset
       (X_source, y_source) using tune_xgb_on_source.
    2) Fit a target-only baseline and a source→target XGB transfer model
       on the given target dataset.
    3) Select one fitted source→target XGB Pipeline.
    4) Compute SHAP values on the same target feature matrix.
    5) Optionally save SHAP arrays and a SHAP summary DataFrame.
    """
    from transfer_learning_curve import eval_transfer_for_subset_xgb, tune_xgb_on_source

    # 0) If needed, tune XGB HPs on the source dataset
    if param_xgb is None:
        param_xgb = tune_xgb_on_source(
            X_ss=X_source,
            y_ss=y_source,
            n_splits_inner=tune_n_splits_inner,
            n_trials=tune_n_trials,
            rands=rand_state,
        )

    # 1) Fit transfer models on the provided target set
    metrics_base, metrics_trans, deltas, xgb_transfer_model = eval_transfer_for_subset_xgb(
        X_ss=X_source,
        y_ss=y_source,
        X_dp_subset=X_target,
        y_dp_subset=y_target,
        param_xgb=param_xgb,
        rand_state=rand_state,
        n_splits=n_splits,
        clip_min=0.1,
        clip_max=10.0,
        return_model=True,
    )
    if xgb_transfer_model is None:
        raise RuntimeError("No transfer XGB model returned from eval_transfer_for_subset_xgb.")

    # 2) Prepare target features for SHAP (drop sample_id if present)
    if "sample_id" in X_target.columns:
        X_target_use = X_target.drop(columns=["sample_id"])
    else:
        X_target_use = X_target

    # 3) Scale via the model's scaler
    X_target_scaled = get_scaled_features_from_pipeline(xgb_transfer_model, X_target_use)

    # 4) Compute SHAP values
    shap_values, explainer = compute_shap_xgb(
        model=xgb_transfer_model,
        X_scaled=X_target_scaled,
    )

    df_shap = None
    shap_dir = None
    if save_results:
        run_name = "transfer_XGB_full_target"
        shap_dir = os.path.join(output_dir, task_name, run_name)
        df_shap = save_shap_results(
            shap_values=shap_values,
            explainer=explainer,
            X_scaled=X_target_scaled,
            output_dir=shap_dir,
            model_name="xgb_transfer",
            target_name=task_name,
        )
    else:
        df_shap = create_shap_df(shap_values, X_target_scaled.columns.tolist())

    results = {
        "model": xgb_transfer_model,
        "param_xgb": param_xgb,
        "X_target": X_target,
        "X_target_scaled": X_target_scaled,
        "shap_values": shap_values,
        "df_shap": df_shap,
        "metrics_target_only": metrics_base,
        "metrics_transfer": metrics_trans,
        "deltas": deltas,
        "shap_dir": shap_dir,
    }
    return results

if __name__ == "__main__":
    import datetime, sys
    from pathlib import Path

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    sys.path.append(module_path)

    xy_dir = os.path.join(module_path, "data", "Xy")

    GROUPS  = ["ss_sem", "ss_om"] #"dp_sem"
    TARGETS = ["uts", "ys", "el"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(module_path, "results", "shap", timestamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("Saving results to:", out_dir)

    for group in GROUPS:
        print(f"\n{'#'*70}\nGROUP: {group.upper()}\n{'#'*70}")

        tasks = []
        for target in TARGETS:
            X = pd.read_csv(os.path.join(xy_dir, f"X_{group}_{target}.csv"))
            y = pd.read_csv(os.path.join(xy_dir, f"y_{group}_{target}.csv")).squeeze()
            tasks.append((target.upper(), X, y))

        for target_name, X, y in tasks:
            print(f"\n{'#'*70}\nRunning SHAP: {group} / {target_name}\n{'#'*70}\n")

            if "sample_id" in X.columns:
                X = X.drop(columns=["sample_id"])

            shap_dir = os.path.join(out_dir, group, target_name)

            xgb_model, xgb_study, df_shap, X_scaled, shap_values = (
                fit_full_xgb_and_compute_shap(
                    X=X,
                    y=y,
                    target_name=f"{group}_{target_name}",
                    shap_outdir=shap_dir,
                    n_trials=100,
                    n_splits_inner=5,
                    rands=2026,
                )
            )
            print(f"\n✓ SHAP complete: {group} / {target_name}")

    print("\nDone.")