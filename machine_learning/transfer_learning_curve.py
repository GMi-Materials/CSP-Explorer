# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 11:30:20 2026

@author: gerfried.millner
"""
author = "Gerfried Millner"
version = "1.0.0"
date = "15.06.2026"
email = "gerfried.millner@ntu.edu.sg"
status = "Developing"

import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import optuna


def create_group_id(df, sample_id_col="sample_id", verbose=False):
    """
    Create group IDs from sample_id column and remove sample_id from dataframe.
    """
    if sample_id_col not in df.columns:
        raise ValueError(
            f"Column '{sample_id_col}' not found in DataFrame."
            f"Available columns: {df.columns.tolist()}"
        )
    groups = pd.factorize(df[sample_id_col])[0]
    groups = pd.Series(groups, index=df.index, name="group_id")
    df_clean = df.drop(columns=[sample_id_col])

    if verbose:
        print(f"  Created {groups.nunique()} groups from {len(groups)} observations")
        print(f"  Removed '{sample_id_col}' column from features")

    return groups, df_clean


def align_X(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Align two feature matrices to their common feature columns.

    Parameters
    ----------
    X_source, X_target : pd.DataFrame
    verbose : bool
        If True, prints the number and names of dropped features.

    Returns
    -------
    X_source_common, X_target_common : pd.DataFrame
    common_features : list
    """
    common_features = X_source.columns.intersection(X_target.columns).tolist()

    if verbose:
        dropped_source = set(X_source.columns) - set(common_features)
        dropped_target = set(X_target.columns) - set(common_features)
        if dropped_source:
            print(f"  align_X: dropped {len(dropped_source)} source-only features: {sorted(dropped_source)}")
        if dropped_target:
            print(f"  align_X: dropped {len(dropped_target)} target-only features: {sorted(dropped_target)}")
        print(f"  align_X: retaining {len(common_features)} common features")

    X_source_common = X_source[common_features].copy()
    X_target_common = X_target[common_features].copy()
    return X_source_common, X_target_common, common_features


def compute_regression_metrics(y_true, y_pred, tol_rel=0.15):
    """
    Compute regression metrics including a relative tolerance metric (tol15).

    tol15: percentage of points where |y_true - y_pred| <= tol_rel * |y_true|.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, _ = pearsonr(y_true, y_pred)
    else:
        r = np.nan

    denom = np.abs(y_true)
    mask = denom > 0
    if np.any(mask):
        within = np.abs(y_true[mask] - y_pred[mask]) <= tol_rel * denom[mask]
        tol15 = 100.0 * within.sum() / within.size
    else:
        tol15 = np.nan

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearsons_r": r,
        "tol15": tol15,
    }


def compute_importance_weights(
    X_pool: pd.DataFrame,
    is_dp_flag: np.ndarray,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> np.ndarray:
    """
    Estimate importance weights via a domain classifier (logistic regression).

    For source (SS) samples the weight is p(DP|x) / (1 - p(DP|x)), clipped to
    [clip_min, clip_max].  Target (DP) samples receive weight 1.

    Parameters
    ----------
    X_pool     : pd.DataFrame, shape (n_ss + n_dp, n_features)
    is_dp_flag : np.ndarray of int, 0 = SS, 1 = DP
    clip_min, clip_max : float

    Returns
    -------
    w : np.ndarray, shape (n_ss + n_dp,)
    """
    dom_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    dom_clf.fit(X_pool, is_dp_flag)
    p_dp = dom_clf.predict_proba(X_pool)[:, 1]

    w = np.where(
        is_dp_flag == 1,
        1.0,
        np.clip(p_dp / np.clip(1.0 - p_dp, 1e-6, 1.0), clip_min, clip_max),
    )
    return w


def _tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits_inner: int = 3,
    n_trials: int = 100,
    rands: int = 42,
    sample_weight: np.ndarray = None,
    groups: np.ndarray = None,
) -> dict:
    """
    Tune XGBoost hyperparameters on an arbitrary (X, y) partition via inner CV.

    Parameters
    ----------
    X, y           : feature matrix and target for this fold's training data
    n_splits_inner : number of inner CV folds
    n_trials       : number of Optuna trials (default 100)
    rands          : random seed
    sample_weight  : optional per-sample weights forwarded to XGB fit
    groups         : optional group array for GroupKFold; when provided,
                     GroupKFold is used instead of KFold so that images from
                     the same physical sample never straddle inner fold boundaries

    Returns
    -------
    best_params : dict
    """
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_splits = min(n_splits_inner, n_unique)
        cv = GroupKFold(n_splits=actual_splits)
        cv_splits = list(cv.split(X, groups=groups))
    else:
        cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=rands)
        cv_splits = list(cv.split(X))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 800, step=100),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha":         trial.suggest_categorical("reg_alpha", [0.0, 1e-3, 1e-2]),
        }
        scores = []
        for tr_idx, te_idx in cv_splits:
            X_tr_inner, y_tr_inner = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te_inner, y_te_inner = X.iloc[te_idx], y.iloc[te_idx]
            w_tr_inner = sample_weight[tr_idx] if sample_weight is not None else None

            xgb = XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=rands,
                n_jobs=-1,
                tree_method="hist",
            )
            xgb.fit(X_tr_inner, y_tr_inner, sample_weight=w_tr_inner)
            y_pred = xgb.predict(X_te_inner)
            scores.append(compute_regression_metrics(y_te_inner, y_pred)["R2"])

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits_inner: int = 3,
    n_trials: int = 25,
    rands: int = 42,
    sample_weight: np.ndarray = None,
    groups: np.ndarray = None,
) -> dict:
    """
    Tune Random Forest hyperparameters on an arbitrary (X, y) partition via inner CV.

    Parameters
    ----------
    X, y           : feature matrix and target for this fold's training data
    n_splits_inner : number of inner CV folds
    n_trials       : number of Optuna trials
    rands          : random seed
    sample_weight  : optional per-sample weights forwarded to RF fit
    groups         : optional group array for GroupKFold; when provided,
                     GroupKFold is used instead of KFold so that images from
                     the same physical sample never straddle inner fold boundaries

    Returns
    -------
    best_params : dict
    """
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_splits = min(n_splits_inner, n_unique)
        cv = GroupKFold(n_splits=actual_splits)
        cv_splits = list(cv.split(X, groups=groups))
    else:
        cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=rands)
        cv_splits = list(cv.split(X))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
        }
        scores = []
        for tr_idx, te_idx in cv_splits:
            X_tr_inner, y_tr_inner = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te_inner, y_te_inner = X.iloc[te_idx], y.iloc[te_idx]
            w_tr_inner = sample_weight[tr_idx] if sample_weight is not None else None

            rf = RandomForestRegressor(
                **params,
                max_features="sqrt",
                bootstrap=True,
                random_state=rands,
                n_jobs=-1,
            )
            rf.fit(X_tr_inner, y_tr_inner, sample_weight=w_tr_inner)
            y_pred = rf.predict(X_te_inner)
            scores.append(compute_regression_metrics(y_te_inner, y_pred)["R2"])

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ============================================================
# Nested CV - XGB
# ============================================================

def eval_transfer_nested_xgb(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp_subset: pd.DataFrame,
    y_dp_subset: pd.Series,
    rand_state: int = 42,
    n_splits_outer: int = 3,
    n_splits_inner: int = 3,
    n_trials_inner: int = 25,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    tol_rel: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fully nested CV comparing DP-only XGB and instance-reweighted pooled
    XGB (SS+DP) on a DP subset.

    Both models share identical outer CV folds (group-aware) and are each
    tuned independently inside every outer fold, ensuring a controlled,
    unbiased comparison across all N values.

    Outer CV  : GroupKFold on DP groups
    Inner CV  : KFold on the respective training partition
    DP-only   : inner CV on DP_train_k
    SS->DP    : inner CV on SS union DP_train_k with importance weights

    Parameters
    ----------
    X_ss, y_ss               : source (SS) features and target (full dataset)
    X_dp_subset, y_dp_subset : target (DP) features and target (subset of size N)
    rand_state               : random seed
    n_splits_outer           : outer GroupKFold splits
    n_splits_inner           : inner KFold splits for HP tuning
    n_trials_inner           : Optuna trials per inner tuning call
    clip_min, clip_max       : importance weight clip bounds
    tol_rel                  : relative tolerance for tol15 metric

    Returns
    -------
    df_metrics_dp_only : pd.DataFrame, per-fold metrics for DP-only model
    df_metrics_ss2dp   : pd.DataFrame, per-fold metrics for SS->DP model
    """
    X_dp_sub = X_dp_subset.copy()
    y_dp_sub = y_dp_subset.copy()

    groups_dp, X_dp_noid = create_group_id(X_dp_sub, sample_id_col="sample_id")
    gkf = GroupKFold(n_splits=n_splits_outer)

    X_ss_noid = X_ss.copy()
    if "sample_id" in X_ss_noid.columns:
        X_ss_noid = X_ss_noid.drop(columns=["sample_id"])

    metrics_dp_only_list = []
    metrics_ss2dp_list = []

    for fold_idx, (tr_idx, te_idx) in enumerate(
        gkf.split(X_dp_noid, y_dp_sub, groups_dp)
    ):
        X_dp_tr = X_dp_noid.iloc[tr_idx].reset_index(drop=True)
        y_dp_tr = y_dp_sub.iloc[tr_idx].reset_index(drop=True)
        X_dp_te = X_dp_noid.iloc[te_idx].reset_index(drop=True)
        y_dp_te = y_dp_sub.iloc[te_idx].reset_index(drop=True)
        groups_dp_tr = groups_dp.iloc[tr_idx].reset_index(drop=True).values

        # DP-only branch
        param_dp = _tune_xgb(
            X_dp_tr, y_dp_tr,
            n_splits_inner=n_splits_inner,
            n_trials=n_trials_inner,
            rands=rand_state + fold_idx,
            sample_weight=None,
            groups=groups_dp_tr,
        )
        xgb_dp = XGBRegressor(
            **param_dp,
            objective="reg:squarederror",
            random_state=rand_state,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_dp.fit(X_dp_tr, y_dp_tr)
        m_dp = compute_regression_metrics(y_dp_te, xgb_dp.predict(X_dp_te), tol_rel=tol_rel)
        metrics_dp_only_list.append(m_dp)

        # SS->DP branch
        X_pool_tr = pd.concat(
            [X_ss_noid.reset_index(drop=True), X_dp_tr], axis=0, ignore_index=True
        )
        y_pool_tr = pd.concat(
            [y_ss.reset_index(drop=True), y_dp_tr], axis=0, ignore_index=True
        )
        is_dp_tr_flag = np.array(
            [0] * len(X_ss_noid) + [1] * len(X_dp_tr), dtype=int
        )
        w_pool_tr = compute_importance_weights(
            X_pool_tr, is_dp_tr_flag, clip_min, clip_max
        )
        # SS samples each receive a unique group ID; DP groups are offset past
        # the SS range so that GroupKFold inner CV respects DP sample boundaries.
        groups_pool_tr = np.concatenate([
            np.arange(len(X_ss_noid)),
            groups_dp_tr + len(X_ss_noid),
        ])
        param_pool = _tune_xgb(
            X_pool_tr, y_pool_tr,
            n_splits_inner=n_splits_inner,
            n_trials=n_trials_inner,
            rands=rand_state + fold_idx,
            sample_weight=w_pool_tr,
            groups=groups_pool_tr,
        )
        xgb_pool = XGBRegressor(
            **param_pool,
            objective="reg:squarederror",
            random_state=rand_state,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_pool.fit(X_pool_tr, y_pool_tr, sample_weight=w_pool_tr)
        m_pool = compute_regression_metrics(
            y_dp_te, xgb_pool.predict(X_dp_te), tol_rel=tol_rel
        )
        metrics_ss2dp_list.append(m_pool)

    return pd.DataFrame(metrics_dp_only_list), pd.DataFrame(metrics_ss2dp_list)


# ============================================================
# Nested CV - RF
# ============================================================

def eval_transfer_nested_rf(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp_subset: pd.DataFrame,
    y_dp_subset: pd.Series,
    rand_state: int = 42,
    n_splits_outer: int = 3,
    n_splits_inner: int = 3,
    n_trials_inner: int = 25,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    tol_rel: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fully nested CV comparing DP-only RF and instance-reweighted pooled
    RF (SS+DP) on a DP subset.

    Mirrors eval_transfer_nested_xgb in design; see that docstring for details.
    """
    X_dp_sub = X_dp_subset.copy()
    y_dp_sub = y_dp_subset.copy()

    groups_dp, X_dp_noid = create_group_id(X_dp_sub, sample_id_col="sample_id")
    gkf = GroupKFold(n_splits=n_splits_outer)

    X_ss_noid = X_ss.copy()
    if "sample_id" in X_ss_noid.columns:
        X_ss_noid = X_ss_noid.drop(columns=["sample_id"])

    metrics_dp_only_list = []
    metrics_ss2dp_list = []

    for fold_idx, (tr_idx, te_idx) in enumerate(
        gkf.split(X_dp_noid, y_dp_sub, groups_dp)
    ):
        X_dp_tr = X_dp_noid.iloc[tr_idx].reset_index(drop=True)
        y_dp_tr = y_dp_sub.iloc[tr_idx].reset_index(drop=True)
        X_dp_te = X_dp_noid.iloc[te_idx].reset_index(drop=True)
        y_dp_te = y_dp_sub.iloc[te_idx].reset_index(drop=True)
        groups_dp_tr = groups_dp.iloc[tr_idx].reset_index(drop=True).values

        # DP-only branch
        param_dp = _tune_rf(
            X_dp_tr, y_dp_tr,
            n_splits_inner=n_splits_inner,
            n_trials=n_trials_inner,
            rands=rand_state + fold_idx,
            sample_weight=None,
            groups=groups_dp_tr,
        )
        rf_dp = RandomForestRegressor(
            **param_dp,
            max_features="sqrt",
            bootstrap=True,
            random_state=rand_state,
            n_jobs=-1,
        )
        rf_dp.fit(X_dp_tr, y_dp_tr)
        m_dp = compute_regression_metrics(y_dp_te, rf_dp.predict(X_dp_te), tol_rel=tol_rel)
        metrics_dp_only_list.append(m_dp)

        # SS->DP branch
        X_pool_tr = pd.concat(
            [X_ss_noid.reset_index(drop=True), X_dp_tr], axis=0, ignore_index=True
        )
        y_pool_tr = pd.concat(
            [y_ss.reset_index(drop=True), y_dp_tr], axis=0, ignore_index=True
        )
        is_dp_tr_flag = np.array(
            [0] * len(X_ss_noid) + [1] * len(X_dp_tr), dtype=int
        )
        w_pool_tr = compute_importance_weights(
            X_pool_tr, is_dp_tr_flag, clip_min, clip_max
        )
        # SS samples each receive a unique group ID; DP groups are offset past
        # the SS range so that GroupKFold inner CV respects DP sample boundaries.
        groups_pool_tr = np.concatenate([
            np.arange(len(X_ss_noid)),
            groups_dp_tr + len(X_ss_noid),
        ])
        param_pool = _tune_rf(
            X_pool_tr, y_pool_tr,
            n_splits_inner=n_splits_inner,
            n_trials=n_trials_inner,
            rands=rand_state + fold_idx,
            sample_weight=w_pool_tr,
            groups=groups_pool_tr,
        )
        rf_pool = RandomForestRegressor(
            **param_pool,
            max_features="sqrt",
            bootstrap=True,
            random_state=rand_state,
            n_jobs=-1,
        )
        rf_pool.fit(X_pool_tr, y_pool_tr, sample_weight=w_pool_tr)
        m_pool = compute_regression_metrics(
            y_dp_te, rf_pool.predict(X_dp_te), tol_rel=tol_rel
        )
        metrics_ss2dp_list.append(m_pool)

    return pd.DataFrame(metrics_dp_only_list), pd.DataFrame(metrics_ss2dp_list)


# ============================================================
# Learning-curve experiment (transfer effect)
# ============================================================

def run_transfer_learning_curve(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp: pd.DataFrame,
    y_dp,
    N_values=None,
    n_reps: int = 10,
    n_splits_outer: int = 3,
    n_splits_inner: int = 3,
    n_trials_inner: int = 25,
    max_time=None,
    base_seed: int = 42,
    model: str = "XGB",
    clip_min: float = 0.1,
    clip_max: float = 10.0,
):
    """
    Run repeated subsampling experiments for multiple N using RF or XGB,
    with fully nested hyperparameter tuning per outer fold and per model branch.

    Parameters
    ----------
    X_ss, y_ss       : source (SS) features and target
    X_dp, y_dp       : target (DP) features and target (full available set)
    N_values         : list of DP subset sizes to evaluate
    n_reps           : number of random subsample repetitions per N
    n_splits_outer   : outer GroupKFold splits (3 recommended for N_min < 30)
    n_splits_inner   : inner KFold splits for HP tuning
    n_trials_inner   : Optuna trials per inner tuning call, default 25.
    max_time         : optional wall-clock time limit in seconds
    base_seed        : base random seed; repetition r uses seed base_seed + r
    model            : "XGB" or "RF"
    clip_min, clip_max : importance weight clip bounds

    Returns
    -------
    df_runs    : pd.DataFrame with per-repetition raw results
    df_summary : pd.DataFrame with aggregated mean +/- 95% CI per N
    """
    model = model.upper()
    if model not in {"RF", "XGB"}:
        raise ValueError(f"Unsupported model '{model}', choose 'RF' or 'XGB'.")

    if N_values is None:
        N_values = [len(X_dp)]

    results = []
    start_time = time.time()

    for N in N_values:
        if N > len(X_dp):
            print(f"Skipping N={N}: exceeds available DP samples ({len(X_dp)})")
            continue

        print(f"[{model}] N = {N} DP samples")

        for r in range(n_reps):
            if max_time is not None and (time.time() - start_time) > max_time:
                print("Wall-clock time limit reached. Stopping early.")
                df_runs = pd.DataFrame(results)
                df_summary = summarize_transfer_results(df_runs)
                return df_runs, df_summary

            rng = np.random.RandomState(base_seed + r)
            idx = rng.choice(len(X_dp), size=N, replace=False)

            X_dp_sub = X_dp.iloc[idx].reset_index(drop=True)
            y_dp_sub = (
                y_dp.iloc[idx].reset_index(drop=True)
                if hasattr(y_dp, "iloc")
                else pd.Series(y_dp[idx]).reset_index(drop=True)
            )

            if model == "XGB":
                df_dp_only, df_ss2dp = eval_transfer_nested_xgb(
                    X_ss=X_ss,
                    y_ss=y_ss,
                    X_dp_subset=X_dp_sub,
                    y_dp_subset=y_dp_sub,
                    rand_state=base_seed + r,
                    n_splits_outer=n_splits_outer,
                    n_splits_inner=n_splits_inner,
                    n_trials_inner=n_trials_inner,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                df_dp_only, df_ss2dp = eval_transfer_nested_rf(
                    X_ss=X_ss,
                    y_ss=y_ss,
                    X_dp_subset=X_dp_sub,
                    y_dp_subset=y_dp_sub,
                    rand_state=base_seed + r,
                    n_splits_outer=n_splits_outer,
                    n_splits_inner=n_splits_inner,
                    n_trials_inner=n_trials_inner,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            metrics_dp_only = {k: float(df_dp_only[k].mean()) for k in df_dp_only.columns}
            metrics_ss2dp   = {k: float(df_ss2dp[k].mean())   for k in df_ss2dp.columns}

            row = {"N": N, "rep": r, "model": model}
            for k, v in metrics_dp_only.items():
                row[f"base_{k}"] = v
            for k, v in metrics_ss2dp.items():
                row[f"tr_{k}"] = v

            row["d_R2"]    = metrics_ss2dp["R2"]    - metrics_dp_only["R2"]
            row["d_MAE"]   = metrics_ss2dp["MAE"]   - metrics_dp_only["MAE"]
            row["d_RMSE"]  = metrics_ss2dp["RMSE"]  - metrics_dp_only["RMSE"]
            row["d_tol15"] = metrics_ss2dp["tol15"] - metrics_dp_only["tol15"]

            results.append(row)

        print(f"  Done N={N} | last row: {results[-1]}")

    df_runs = pd.DataFrame(results)
    df_summary = summarize_transfer_results(df_runs)
    return df_runs, df_summary


# ============================================================
# Aggregation and plotting helpers
# ============================================================

def summarize_transfer_results(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-repetition results vs N to mean and 95% CI.

    The 95% CI is computed as 1.96 * SE of the mean across n_reps repetitions.
    Each repetition value is itself the mean over outer CV folds for that
    (N, repetition) combination.
    """
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    grouped = df_runs.groupby("N").agg(
        d_RMSE_mean=("d_RMSE", "mean"),
        d_RMSE_se=("d_RMSE", se),
        d_MAE_mean=("d_MAE", "mean"),
        d_MAE_se=("d_MAE", se),
        d_R2_mean=("d_R2", "mean"),
        d_R2_se=("d_R2", se),
        d_tol15_mean=("d_tol15", "mean"),
        d_tol15_se=("d_tol15", se),
        base_MAE_mean=("base_MAE", "mean"),
        base_MAE_se=("base_MAE", se),
        tr_MAE_mean=("tr_MAE", "mean"),
        tr_MAE_se=("tr_MAE", se),
    ).reset_index()

    for metric in ("d_RMSE", "d_MAE", "d_R2", "d_tol15", "base_MAE", "tr_MAE"):
        grouped[f"{metric}_ci95"] = 1.96 * grouped[f"{metric}_se"]

    return grouped


def plot_transfer_learning_curve(df_summary: pd.DataFrame, prop: str = ""):
    """
    Plot absolute MAE for DP-only and SS->DP together with delta-MAE vs N,
    including 95% CIs on the mean across repetitions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    title_suffix = f" - {prop.upper()}" if prop else ""

    ax = axes[0]
    ax.errorbar(
        df_summary["N"], df_summary["base_MAE_mean"],
        yerr=df_summary["base_MAE_ci95"],
        fmt="o-", capsize=3, label="DP-only",
    )
    ax.errorbar(
        df_summary["N"], df_summary["tr_MAE_mean"],
        yerr=df_summary["tr_MAE_ci95"],
        fmt="s--", capsize=3, label="SS->DP",
    )
    ax.set_xlabel("Number of DP training samples $N_T$")
    ax.set_ylabel("MAE (MPa)")
    ax.set_title(f"Absolute MAE{title_suffix}")
    ax.legend()

    ax = axes[1]
    ax.errorbar(
        df_summary["N"], df_summary["d_MAE_mean"],
        yerr=df_summary["d_MAE_ci95"],
        fmt="s-", capsize=3, color="C1", label="delta-MAE (SS->DP minus DP-only)",
    )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of DP training samples $N_T$")
    ax.set_ylabel("delta-MAE (MPa)")
    ax.set_title(f"Transfer improvement{title_suffix}")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import pickle
    import sys
    import os

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    sys.path.append(module_path)
    print("Module path:", module_path)

    xy_dir = os.path.join(module_path, "data", "Xy")

    # Experiment configuration
    N_VALUES        = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 239, 260, 280]
    N_REPS          = 10
    N_SPLITS_OUTER  = 3    # 3-fold: ensures ~7 test samples even at N=20
    N_SPLITS_INNER  = 3
    N_TRIALS_INNER  = 100 
    MODEL           = "XGB"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_root = os.path.join(
        module_path, "results", "transfer_learning", timestamp
    )
    Path(out_dir_root).mkdir(parents=True, exist_ok=True)
    print("Saving results to:", out_dir_root)

    properties = ["uts", "ys", "el"]

    for prop in properties:
        print("=" * 60)
        print(f"Running nested transfer-learning curve for: {prop.upper()}")
        print("=" * 60)

        X_ss = pd.read_csv(os.path.join(xy_dir, f"X_ss_sem_{prop}.csv"))
        y_ss = pd.read_csv(os.path.join(xy_dir, f"y_ss_sem_{prop}.csv")).squeeze()

        X_dp = pd.read_csv(os.path.join(xy_dir, f"X_dp_sem_{prop}.csv"))
        y_dp = pd.read_csv(os.path.join(xy_dir, f"y_dp_sem_{prop}.csv")).squeeze()

        X_ss_common, X_dp_common, _ = align_X(X_ss, X_dp, verbose=True)

        df_runs, df_summary = run_transfer_learning_curve(
            X_ss=X_ss_common,
            y_ss=y_ss,
            X_dp=X_dp_common,
            y_dp=y_dp,
            N_values=N_VALUES,
            n_reps=N_REPS,
            n_splits_outer=N_SPLITS_OUTER,
            n_splits_inner=N_SPLITS_INNER,
            n_trials_inner=N_TRIALS_INNER,
            base_seed=42,
            model=MODEL,
            clip_min=0.1,
            clip_max=10.0,
        )

        out_dir_prop = os.path.join(out_dir_root, prop)
        Path(out_dir_prop).mkdir(parents=True, exist_ok=True)

        df_runs.to_csv(
            os.path.join(out_dir_prop, f"transfer_runs_{MODEL}_{prop}.csv"),
            index=False,
        )
        df_summary.to_csv(
            os.path.join(out_dir_prop, f"transfer_summary_{MODEL}_{prop}.csv"),
            index=False,
        )
        with open(
            os.path.join(out_dir_prop, f"transfer_runs_{MODEL}_{prop}.pkl"), "wb"
        ) as f:
            pickle.dump(df_runs, f)
        with open(
            os.path.join(out_dir_prop, f"transfer_summary_{MODEL}_{prop}.pkl"), "wb"
        ) as f:
            pickle.dump(df_summary, f)

        plot_transfer_learning_curve(df_summary, prop=prop)

    print("Done.")