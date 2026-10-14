# -*- coding: utf-8 -*-
"""
SHAP analysis for the paper models (XGBoost, group-aware, fold-local cleaning).

Requirements
------------
- models_paper.py and the data_processing package importable (same project root)
- shap >= 0.42, xgboost, sklearn, numpy, pandas, matplotlib, optuna

Design notes
------------
- The final model is tuned and fitted by models_paper.train_final_xgb_full with
  the SAME settings as the nested CV (n_splits_inner=3, random_state=2026).
- SHAP values are computed in-sample on the cleaned, scaled feature matrix that
  the fitted model actually saw. They describe the fitted model, not
  out-of-sample behaviour.
- Several rows can share one sample_id (multiple images per sample). Besides
  the row-level summary, a per-sample_id summary is produced in which every
  physical sample contributes equally.
- Additivity (expected_value + sum(SHAP) == model prediction) is verified
  explicitly after every SHAP computation.
"""

from __future__ import annotations

import copy
import os
import pickle
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

__author__ = "Gerfried Millner"
__version__ = "1.3.0"
__date__ = "13.10.2026"
__email__ = "gerfried.millner@ntu.edu.sg"
__status__ = "Deliver"


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


def slugify(text: str) -> str:
    """File-system-safe lower-case label, e.g. 'Total EL (%)' -> 'total_el'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


# ---------------------------------------------------------------------
# SHAP computation helpers
# ---------------------------------------------------------------------

def get_representative_background(
    X,
    strategy: str = "sample",
    n_background: int = 200,
    random_state: int = 42,
):
    """
    Get a background data set for interventional SHAP.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (already scaled).
    strategy : str
        'sample' : random subset of REAL rows (default, recommended for
                   trees with mixed binary/continuous features).
        'kmeans' : k-means centroids (centroids are not real data points and
                   give fractional values for binary features).
        'full'   : all rows (cost grows quadratically with n).
        'auto'   : 'sample' if n > n_background, else 'full'.
    n_background : int
        Number of background rows (or clusters for 'kmeans').
    random_state : int
        Random seed.

    Returns
    -------
    background : np.ndarray
    method : str
    """
    X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    n_samples = len(X_array)

    if strategy == "auto":
        strategy = "sample" if n_samples > n_background else "full"

    if strategy == "full" or n_samples <= n_background and strategy == "sample":
        return X_array, "full"

    if strategy == "sample":
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=n_background, replace=False)
        return X_array[np.sort(idx)], f"sample_{n_background}"

    if strategy == "kmeans":
        n_clusters = min(n_background, max(2, n_samples // 2))
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        centers = km.fit(X_array).cluster_centers_
        return centers, f"kmeans_{n_clusters}"

    raise ValueError(f"Unknown background strategy: {strategy!r}")


def get_scaled_features_from_pipeline(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pass X through the model's scaler and return a labelled DataFrame.

    Column names come from the scaler itself (scaler.columns_), never from
    positional slicing. X must contain exactly the scaler's columns in the
    same order; otherwise CustomScaler.transform would silently reindex and
    produce NaNs.
    """
    scaler = model.named_steps["scaler"]
    expected = list(scaler.columns_)

    if list(X.columns) != expected:
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        raise ValueError(
            "Feature columns of X do not match the fitted scaler. "
            f"Missing: {missing[:10]}, extra: {extra[:10]}, "
            f"or order differs (n_expected={len(expected)}, n_given={X.shape[1]})."
        )

    X_scaled_values = scaler.transform(X)

    if not np.isfinite(X_scaled_values).all():
        raise ValueError("Scaled feature matrix contains NaN or inf.")

    return pd.DataFrame(X_scaled_values, index=X.index, columns=expected)


def _additivity_stats(
    expected_value,
    shap_values: np.ndarray,
    pred: np.ndarray,
    rel_tol: float = 1e-2,
) -> dict:
    """
    Compare expected_value + sum(SHAP) with the model prediction.

    tol = rel_tol * max(1, std(prediction)). Besides the maximum error the
    statistics tell the error pattern apart: a roughly constant signed error
    points to a base_score/expected_value problem, row-specific errors to a
    traversal mismatch between shap and xgboost.
    """
    base = float(np.ravel(expected_value)[0])
    signed = base + shap_values.sum(axis=1) - pred
    abs_err = np.abs(signed)
    tol = rel_tol * max(1.0, float(np.std(pred)))
    return {
        "ok": bool(abs_err.max() <= tol),
        "tol": tol,
        "max_err": float(abs_err.max()),
        "p99_err": float(np.percentile(abs_err, 99)),
        "mean_err": float(abs_err.mean()),
        "n_over_tol": int((abs_err > tol).sum()),
        "n_rows": int(len(abs_err)),
        "mean_signed": float(signed.mean()),
        "std_signed": float(signed.std()),
    }


def _print_additivity(label: str, st: dict) -> None:
    status = "OK" if st["ok"] else "FAILED"
    print(f"Additivity [{label}] {status}: max {st['max_err']:.4g}, "
          f"p99 {st['p99_err']:.4g}, mean {st['mean_err']:.4g}, "
          f"rows over tol {st['n_over_tol']}/{st['n_rows']} (tol {st['tol']:.4g})")
    if not st["ok"]:
        constant = (
            abs(st["mean_signed"]) > st["tol"]
            and st["std_signed"] < 0.25 * abs(st["mean_signed"])
        )
        pattern = (
            "roughly constant offset (suggests base_score / expected_value handling)"
            if constant else
            "row-specific errors (suggests a traversal mismatch between shap and xgboost)"
        )
        print(f"  error pattern: {pattern}; mean signed error {st['mean_signed']:.4g}, "
              f"std {st['std_signed']:.4g}")


def _shap_interventional(xgb_model, X_values, background):
    explainer = shap.TreeExplainer(
        xgb_model, data=background, feature_perturbation="interventional"
    )
    values = np.asarray(explainer.shap_values(X_values, check_additivity=False))
    return values, explainer


def _shap_path_dependent(xgb_model, X_values):
    explainer = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(X_values, check_additivity=False))
    return values, explainer


def _shap_xgboost_native(xgb_model, X_values):
    """XGBoost's own TreeSHAP (pred_contribs): exact w.r.t. its predictions."""
    import xgboost as xgb_lib

    booster = xgb_model.get_booster()
    contribs = booster.predict(xgb_lib.DMatrix(X_values), pred_contribs=True)
    values = np.asarray(contribs[:, :-1])
    explainer = SimpleNamespace(expected_value=float(np.mean(contribs[:, -1])))
    return values, explainer


SHAP_METHODS = (
    "interventional_float32",
    "interventional",
    "tree_path_dependent",
    "xgboost_native",
)


def compute_shap_xgb(
    model: Pipeline,
    X_scaled: pd.DataFrame,
    background_strategy: str = "sample",
    n_background: int = 200,
    random_state: int = 42,
    X_background: pd.DataFrame | None = None,
    method: str = "auto",
):
    """
    Compute SHAP values for XGBoost and verify additivity.

    Assumes model is Pipeline([('scaler', ...), ('xgb', XGBRegressor)]).
    All rows of X_scaled are explained.

    method
    ------
    'interventional_float32' : shap TreeExplainer with a background of real
                       rows (see get_representative_background), inputs and
                       background cast to float32. XGBoost casts its inputs to
                       float32, whereas shap 0.47 explains an XGBRegressor in
                       float64; for rows lying exactly on a split threshold
                       the two then take different branches and additivity
                       breaks. Casting first removes this mismatch.
    'interventional' : same explainer on float64 inputs (strict; raises if
                       additivity fails). If X_background (scaled, same
                       columns) is given, the background is drawn from it.
    'auto' (default) : try, in this order, until additivity holds:
                       'interventional_float32', 'interventional',
                       'tree_path_dependent' (shap, no background) and
                       'xgboost_native' (XGBoost's own pred_contribs). The
                       method used is stored on the returned explainer as
                       .shap_method. Both interventional variants compute the
                       same quantity; only the last two are path-dependent
                       TreeSHAP and must be named in the methods section.
    Any single name from SHAP_METHODS can also be forced.

    Returns (shap_values, explainer).
    """
    xgb = model.named_steps["xgb"]
    X_values = X_scaled.to_numpy(dtype=float)

    if method != "auto" and method not in SHAP_METHODS:
        raise ValueError(f"method must be 'auto' or one of {SHAP_METHODS}.")

    if X_background is not None and list(X_background.columns) != list(X_scaled.columns):
        raise ValueError("X_background must have the same columns as X_scaled.")

    # Recent XGBoost versions (>= 3.1) default to enable_categorical=True, which
    # shap's interventional TreeExplainer rejects even without categorical
    # features. Explain a shallow copy with the flag off (same booster, so the
    # predictions are identical) and leave the fitted model untouched.
    xgb_for_shap = copy.copy(xgb)
    if getattr(xgb_for_shap, "enable_categorical", False):
        xgb_for_shap.set_params(enable_categorical=False)

    pred = np.asarray(xgb.predict(X_values), dtype=float)

    order = list(SHAP_METHODS) if method == "auto" else [method]
    needs_background = any(m.startswith("interventional") for m in order)
    background = None
    if needs_background:
        background, bg_method = get_representative_background(
            X_scaled if X_background is None else X_background,
            strategy=background_strategy,
            n_background=n_background,
            random_state=random_state,
        )
        print(f"SHAP background: {bg_method} ({len(background)} rows), "
              f"explaining {len(X_values)} rows.")

    attempts = {
        "interventional": lambda: _shap_interventional(xgb_for_shap, X_values, background),
        "interventional_float32": lambda: _shap_interventional(
            xgb_for_shap, X_values.astype(np.float32), background.astype(np.float32)
        ),
        "tree_path_dependent": lambda: _shap_path_dependent(xgb_for_shap, X_values),
        "xgboost_native": lambda: _shap_xgboost_native(xgb_for_shap, X_values),
    }

    failures = []
    for name in order:
        try:
            shap_values, explainer = attempts[name]()
        except Exception as exc:  # noqa: BLE001 (any backend error triggers the next method)
            print(f"SHAP method '{name}' raised {type(exc).__name__}: {str(exc)[:200]}")
            failures.append(f"{name}: {type(exc).__name__}")
            continue

        if shap_values.shape != X_values.shape:
            failures.append(f"{name}: unexpected shape {shap_values.shape}")
            continue

        stats = _additivity_stats(explainer.expected_value, shap_values, pred)
        _print_additivity(name, stats)

        if stats["ok"]:
            explainer.shap_method = name
            if not name.startswith("interventional") and name != order[0]:
                print(f"WARNING: SHAP computed with '{name}' (path-dependent "
                      "TreeSHAP) because the interventional methods failed the "
                      "additivity check. State this method in the paper.")
            elif name != order[0]:
                print(f"NOTE: '{order[0]}' failed, '{name}' passed (same "
                      "interventional SHAP quantity).")
            return shap_values, explainer
        failures.append(f"{name}: max error {stats['max_err']:.4g} > tol {stats['tol']:.4g}")

    raise RuntimeError(
        "SHAP additivity could not be satisfied. Attempts: " + "; ".join(failures)
        + ". Check shap/xgboost versions and that the matrix passed here equals "
        "the matrix the model was fitted on."
    )


# ---------------------------------------------------------------------
# SHAP statistics and saving
# ---------------------------------------------------------------------

def _summarize_abs_shap(abs_matrix: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Summary table from a (n_units x n_features) matrix of |SHAP| values."""
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": abs_matrix.mean(axis=0),
            "std_abs_shap": abs_matrix.std(axis=0),
        }
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["cumulative_importance"] = df["mean_abs_shap"].cumsum()
    total = df["mean_abs_shap"].sum()
    df["percent_cumulative"] = df["cumulative_importance"] / total * 100.0
    return df


def create_shap_df(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Row-level summary: every row (image) counts equally."""
    return _summarize_abs_shap(np.abs(shap_values), feature_names)


def create_shap_df_by_sample(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_ids,
) -> pd.DataFrame:
    """
    Sample-level summary: |SHAP| is first averaged within each sample_id, then
    across samples, so samples with many images do not dominate.
    """
    abs_df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    abs_df["_sample_id"] = np.asarray(sample_ids)
    per_sample = abs_df.groupby("_sample_id").mean()
    df = _summarize_abs_shap(per_sample.to_numpy(), feature_names)
    df["n_samples"] = len(per_sample)
    return df


def save_shap_results(
    shap_values: np.ndarray,
    explainer,
    X_scaled: pd.DataFrame,
    output_dir: str,
    model_name: str,
    target_name: str,
    sample_ids=None,
    X_display: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Save SHAP arrays, explainer, and SHAP summary DataFrame(s).
    Returns the row-level summary.

    If X_display (cleaned, unscaled feature matrix, same columns and row order
    as X_scaled) is given, it is written to X_<model>_<target>.csv. The scaled
    matrix is always written to X_scaled_<model>_<target>.csv. Row i of
    both files belongs to row i of shap_values in the pickle, so they can be
    loaded in a notebook to re-create beeswarm, dependence and other SHAP plots.
    """
    ensure_directory(output_dir)

    feature_names = X_scaled.columns.tolist()
    df_shap = create_shap_df(shap_values, feature_names)

    base = f"{slugify(model_name)}_{slugify(target_name)}"
    pkl_path = os.path.join(output_dir, f"shap_{base}.pkl")
    csv_path = os.path.join(output_dir, f"shapsummary_{base}.csv")

    # Scaled matrix: the one the SHAP values were actually computed on.
    X_scaled.to_csv(
        os.path.join(output_dir, f"X_scaled_{base}.csv"),
        index=True,
        index_label="index",
    )

    if X_display is not None:
        if list(X_display.columns) != feature_names or len(X_display) != len(X_scaled):
            raise ValueError("X_display must match X_scaled in columns and row count.")
        X_display.to_csv(
            os.path.join(output_dir, f"X_{base}.csv"),
            index=True,
            index_label="index",
        )

    payload = {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "df_shap": df_shap,
        "explainer": explainer,
        "shap_method": getattr(explainer, "shap_method", "interventional_float32"),
        "index": X_scaled.index.to_numpy(),
    }

    if sample_ids is not None:
        df_sample = create_shap_df_by_sample(shap_values, feature_names, sample_ids)
        df_sample.to_csv(
            os.path.join(output_dir, f"shapsummary_bysample_{base}.csv"),
            index=False,
        )
        payload["df_shap_by_sample"] = df_sample
        payload["sample_ids"] = np.asarray(sample_ids)

    save_pickle(payload, pkl_path)
    df_shap.to_csv(csv_path, index=False)
    print(f"SHAP results saved to {output_dir} for {model_name}, target={target_name}")
    return df_shap


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_scaled: pd.DataFrame,
    X_display: pd.DataFrame | None = None,
    max_display: int = 20,
    save_path: str | None = None,
    figsize=(10, 8),
):
    """
    Beeswarm plot. SHAP values always belong to X_scaled; if X_display (the
    cleaned, unscaled matrix with identical columns and row order) is given,
    the colour scale uses raw feature values, which are easier to read.
    """
    features = X_scaled if X_display is None else X_display
    shap.summary_plot(
        shap_values,
        features,
        feature_names=X_scaled.columns.tolist(),
        max_display=max_display,
        plot_size=figsize,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Beeswarm plot saved to {save_path}")
        plt.close(fig)
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
    """Mean |SHAP| bar plot."""
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=X_scaled.columns.tolist(),
        plot_type="bar",
        max_display=max_display,
        plot_size=figsize,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------
# XGB: full-data fit + SHAP
# ---------------------------------------------------------------------

def fit_full_xgb_and_compute_shap(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    shap_out_dir: str | None = None,
    target_label: str | None = None,
    n_trials: int = 100,
    n_splits_inner: int = 3,
    random_state: int = 2026,
    background_strategy: str = "sample",
    n_background: int = 200,
    shap_method: str = "auto",
) -> dict:
    """
    Tune and fit the final group-aware XGBoost model, rebuild the cleaned
    feature matrix with the fitted cleaning pipeline, and compute SHAP values
    on the exact cleaned and scaled matrix used by the fitted model.

    X must still contain 'sample_id' (needed for group-aware inner CV; it is
    removed by the cleaning step). target_col must be the same column name
    that models_paper uses, e.g. 'UTS (MPa)'. target_label is only used for
    file names (default: slug of target_col).

    Returns
    -------
    dict with keys:
        bundle, study, df_shap, df_shap_by_sample, X_scaled, X_clean,
        shap_values, y_clean, sample_ids, explainer
    """
    from models_paper import train_final_xgb_full
    from data_processing.clean_data import apply_cleaning_pipeline

    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)
    label = target_label or slugify(target_col)

    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} rows.")

    if target_col in X.columns:
        raise ValueError(
            f"'{target_col}' is already in X. Pass X without its target column."
        )

    if "sample_id" not in X.columns:
        raise KeyError("X must retain 'sample_id' for group-aware inner CV.")

    print(f"XGB: fitting final model for {target_col} using {len(X)} raw rows.")

    fitted_bundle, xgb_study = train_final_xgb_full(
        X=X,
        y=y,
        target_col=target_col,
        groups=True,
        random_state=random_state,
        n_splits_inner=n_splits_inner,
        n_trials=n_trials,
    )

    xgb_model = fitted_bundle["model"]
    expected_columns = fitted_bundle["feature_columns"]

    # Guard against a fit/apply inconsistency in clean_data: image descriptors
    # flagged as too-missing by fit_image_missingness_filter are dropped again
    # by apply_cleaning_pipeline, but fit_cleaning_pipeline only removes them
    # if the generic missingness rule also catches them afterwards. Any such
    # feature is in the model but would be rebuilt as a constant median column.
    image_diag = fitted_bundle["cleaning_pipeline"].get("image_missingness", {})
    inconsistent = sorted(
        set(image_diag.get("dropped_image_features", [])) & set(expected_columns)
    )
    has_training_matrix = (
        "X_train_clean" in fitted_bundle and "y_train_clean" in fitted_bundle
    )
    if inconsistent and not has_training_matrix:
        raise RuntimeError(
            f"{len(inconsistent)} image features are in the fitted model but would "
            "be dropped and re-imputed by apply_cleaning_pipeline "
            f"(e.g. {inconsistent[:5]}). Return X_train_clean / y_train_clean "
            "from train_final_xgb_full, or fix fit_image_missingness_filter "
            "to drop dropped_image_features at fit time."
        )
    if inconsistent:
        print(f"WARNING: {len(inconsistent)} image features are affected by the "
              "fit/apply inconsistency in clean_data; the training matrix from "
              "the bundle is used, so SHAP is unaffected.")

    # Prefer the matrix the model was fitted on, if the bundle carries it;
    # otherwise rebuild it with the fitted cleaning pipeline.
    if "X_train_clean" in fitted_bundle and "y_train_clean" in fitted_bundle:
        X_clean = fitted_bundle["X_train_clean"]
        y_clean = fitted_bundle["y_train_clean"]
    else:
        dataframe_full = X.copy()
        dataframe_full[target_col] = y.to_numpy()
        X_clean, y_clean = apply_cleaning_pipeline(
            dataframe=dataframe_full,
            pipeline=fitted_bundle["cleaning_pipeline"],
            target_col=target_col,
            verbose=False,
        )

    if list(X_clean.columns) != list(expected_columns):
        raise RuntimeError(
            "The cleaned SHAP matrix does not match the final model feature schema."
        )

    if len(X_clean) != fitted_bundle["n_train_clean"]:
        raise RuntimeError(
            "The number of SHAP-eligible rows does not match the number of "
            "rows used to fit the final model."
        )

    if not X_clean.index.equals(y_clean.index):
        raise RuntimeError("Cleaned X and y indices are not aligned.")

    X_scaled = get_scaled_features_from_pipeline(model=xgb_model, X=X_clean)

    shap_values, explainer = compute_shap_xgb(
        model=xgb_model,
        X_scaled=X_scaled,
        background_strategy=background_strategy,
        n_background=n_background,
        random_state=random_state,
        method=shap_method,
    )

    # Map cleaned rows back to their sample_id. This relies on the cleaning
    # step preserving the raw row index (as prepare_model_fold documents).
    sample_ids = None
    if X_clean.index.is_unique and X_clean.index.isin(X.index).all():
        sample_ids = X.loc[X_clean.index, "sample_id"].to_numpy()
    else:
        print("WARNING: cleaned index does not map onto raw rows; "
              "per-sample SHAP summary skipped.")

    if shap_out_dir is not None:
        ensure_directory(shap_out_dir)
        save_pickle(fitted_bundle, os.path.join(shap_out_dir, f"xgb_full_{label}.pkl"))
        save_pickle(xgb_study, os.path.join(shap_out_dir, f"xgb_study_{label}.pkl"))
        df_shap = save_shap_results(
            shap_values=shap_values,
            explainer=explainer,
            X_scaled=X_scaled,
            output_dir=shap_out_dir,
            model_name="xgb",
            target_name=label,
            sample_ids=sample_ids,
            X_display=X_clean,
        )
    else:
        df_shap = create_shap_df(shap_values, X_scaled.columns.tolist())

    df_shap_by_sample = None
    if sample_ids is not None:
        df_shap_by_sample = create_shap_df_by_sample(
            shap_values, X_scaled.columns.tolist(), sample_ids
        )

    return {
        "bundle": fitted_bundle,
        "study": xgb_study,
        "df_shap": df_shap,
        "df_shap_by_sample": df_shap_by_sample,
        "X_scaled": X_scaled,
        "X_clean": X_clean,
        "shap_values": shap_values,
        "y_clean": y_clean,
        "sample_ids": sample_ids,
        "explainer": explainer,
        "shap_method": getattr(explainer, "shap_method", "interventional_float32"),
    }


# ---------------------------------------------------------------------
# Transfer-learning XGB + SHAP
# ---------------------------------------------------------------------

def shap_for_transfer_xgb(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp: pd.DataFrame,
    y_dp: pd.Series,
    target_col: str | None = None,
    shap_out_dir: str | None = None,
    target_label: str | None = None,
    n_trials: int = 100,
    n_splits_inner: int = 3,
    random_state: int = 2026,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    explain: str = "dp",
    background_from: str = "dp",
    background_strategy: str = "sample",
    n_background: int = 200,
    tune_on_dp_only: bool = False,
    shap_method: str = "auto",
) -> dict:
    """
    SHAP for the final importance-weighted SS+DP XGB model.

    X_ss, X_dp are the RAW source tables (sample_id retained, as exported by
    clean_data.py). transfer_learning_curve.fit_final_transfer_xgb fits the
    cleaning once on SS + DP, computes cross-fitted importance weights, tunes
    with group-aware fold-local inner CV and fits the final model. SHAP values
    are computed on that exact cleaned and scaled training matrix.

    explain         : 'dp' (explain the DP rows, default) or 'pool' (all rows).
    background_from : 'dp' (target-domain baseline, default) or 'pool'.
                      The baseline defines what "average" means, so state the
                      choice in the methods.

    Returns a dict like fit_full_xgb_and_compute_shap, plus is_dp (aligned with
    X_scaled) and the fitted bundle (which also holds the sample weights).
    """
    from transfer_learning_curve import fit_final_transfer_xgb

    if explain not in {"dp", "pool"}:
        raise ValueError("explain must be 'dp' or 'pool'.")
    if background_from not in {"dp", "pool"}:
        raise ValueError("background_from must be 'dp' or 'pool'.")

    bundle, study = fit_final_transfer_xgb(
        X_ss=X_ss,
        y_ss=y_ss,
        X_dp=X_dp,
        y_dp=y_dp,
        target_col=target_col,
        n_trials=n_trials,
        n_splits_inner=n_splits_inner,
        random_state=random_state,
        clip_min=clip_min,
        clip_max=clip_max,
        tune_on_dp_only=tune_on_dp_only,
    )

    model = bundle["model"]
    X_clean = bundle["X_train_clean"]
    y_clean = bundle["y_train_clean"]
    is_dp = np.asarray(bundle["is_dp"]).astype(bool)
    sample_ids = np.asarray(bundle["sample_ids"])
    label = target_label or slugify(f"transfer_{bundle['target_column']}")

    if list(X_clean.columns) != list(bundle["feature_columns"]):
        raise RuntimeError("Training matrix does not match the model feature schema.")
    if len(X_clean) != bundle["n_train_clean"]:
        raise RuntimeError("Training matrix row count differs from n_train_clean.")

    print(f"Transfer XGB: {len(X_clean)} cleaned rows "
          f"({int(is_dp.sum())} DP, {int((~is_dp).sum())} SS), "
          f"{X_clean.shape[1]} features.")

    X_scaled_all = get_scaled_features_from_pipeline(model=model, X=X_clean)

    explain_pos = np.arange(len(X_scaled_all)) if explain == "pool" else np.flatnonzero(is_dp)
    bg_pos = np.arange(len(X_scaled_all)) if background_from == "pool" else np.flatnonzero(is_dp)

    X_scaled = X_scaled_all.iloc[explain_pos]
    X_bg = X_scaled_all.iloc[bg_pos]
    X_display = X_clean.iloc[explain_pos]
    explain_ids = sample_ids[explain_pos]

    shap_values, explainer = compute_shap_xgb(
        model=model,
        X_scaled=X_scaled,
        background_strategy=background_strategy,
        n_background=n_background,
        random_state=random_state,
        X_background=X_bg,
        method=shap_method,
    )

    if shap_out_dir is not None:
        ensure_directory(shap_out_dir)
        save_pickle(bundle, os.path.join(shap_out_dir, f"xgb_transfer_full_{label}.pkl"))
        save_pickle(study, os.path.join(shap_out_dir, f"xgb_transfer_study_{label}.pkl"))
        df_shap = save_shap_results(
            shap_values=shap_values,
            explainer=explainer,
            X_scaled=X_scaled,
            output_dir=shap_out_dir,
            model_name="xgb_transfer",
            target_name=label,
            sample_ids=explain_ids,
            X_display=X_display,
        )
    else:
        df_shap = create_shap_df(shap_values, X_scaled.columns.tolist())

    df_shap_by_sample = create_shap_df_by_sample(
        shap_values, X_scaled.columns.tolist(), explain_ids
    )

    return {
        "bundle": bundle,
        "study": study,
        "df_shap": df_shap,
        "df_shap_by_sample": df_shap_by_sample,
        "X_scaled": X_scaled,
        "X_clean": X_display,
        "shap_values": shap_values,
        "y_clean": y_clean.iloc[explain_pos],
        "sample_ids": explain_ids,
        "is_dp": is_dp[explain_pos],
        "explainer": explainer,
        "shap_method": getattr(explainer, "shap_method", "interventional_float32"),
    }


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import sys
    from pathlib import Path

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    xy_dir = os.path.join(module_path, "data", "Xy")

    GROUPS = ["ss_sem", "ss_om"]  # "dp_sem"
    # Same target column names as in models_paper.py
    TARGETS = {
        "uts": "UTS (MPa)",
        "ys": "YS (MPa)",
        "el": "Total EL (%)",
    }

    # Must match the nested-CV settings in models_paper.run_paper_models
    N_TRIALS = 100
    N_SPLITS_INNER = 3
    RANDOM_STATE = 2026

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(module_path, "results", "shap", timestamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("Saving results to:", out_dir)

    for group in GROUPS:
        print(f"\n{'#' * 70}\nGROUP: {group.upper()}\n{'#' * 70}")

        for target_short, target_col in TARGETS.items():
            print(f"\n{'#' * 70}\nRunning SHAP: {group} / {target_short.upper()}\n{'#' * 70}\n")

            # sample_id must stay in X (group-aware CV); cleaning removes it.
            X = pd.read_csv(
                os.path.join(xy_dir, f"X_{group}_{target_short}.csv"),
                low_memory=False,
            )
            y = pd.read_csv(
                os.path.join(xy_dir, f"y_{group}_{target_short}.csv"),
                low_memory=False,
            ).squeeze("columns")

            shap_dir = os.path.join(out_dir, group, target_short)

            res = fit_full_xgb_and_compute_shap(
                X=X,
                y=y,
                target_col=target_col,
                shap_out_dir=shap_dir,
                target_label=target_short,
                n_trials=N_TRIALS,
                n_splits_inner=N_SPLITS_INNER,
                random_state=RANDOM_STATE,
            )

            plot_shap_beeswarm(
                res["shap_values"],
                res["X_scaled"],
                X_display=res["X_clean"],
                save_path=os.path.join(shap_dir, f"beeswarm_{target_short}.png"),
            )
            plot_shap_bar(
                res["shap_values"],
                res["X_scaled"],
                save_path=os.path.join(shap_dir, f"bar_{target_short}.png"),
            )

            print(f"\nTop features ({group} / {target_short.upper()}):")
            print(res["df_shap"].head(10)[["feature", "mean_abs_shap", "percent_cumulative"]]
                  .to_string(index=False))
            print(f"\nSHAP complete: {group} / {target_short.upper()} "
                  f"(method: {res['shap_method']})")

    print("\nDone.")