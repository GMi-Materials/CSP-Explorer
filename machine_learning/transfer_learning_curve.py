# -*- coding: utf-8 -*-
"""
Transfer-learning curve (SS -> DP) with fold-local cleaning.

Usage
-----
Edit the "Experiment configuration" block under `if __name__ == "__main__":`
and run `python transfer_learning_curve.py`.

    MODE = "profile"      time the stages of one split, extrapolate, exit
    MODE = "trial_check"  best inner R2 vs number of Optuna trials, exit
    MODE = "run"          full experiment (default)

Set RUN_DIR to an existing results folder to resume an unfinished run.

Requirements
------------
models_paper.py and the data_processing package must be importable.
joblib >= 1.3 is recommended (streaming results for checkpointing).
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__author__ = "Gerfried Millner"
__version__ = "1.2.0"
__date__ = "13.10.2026"
__email__ = "gerfried.millner@ntu.edu.sg"
__status__ = "Developing"

MODULE_PATH = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

from data_processing.clean_data import fit_cleaning_pipeline, prepare_clean_fold  # noqa: E402
from models_paper import (  # noqa: E402
    CLEANING_KWARGS,
    build_xgb_pipeline,
    suggest_xgb_parameters,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# Seeds and thread control
# ============================================================

def _derive_seed(*parts: int) -> int:
    """Deterministic 31-bit seed from integer parts (SeedSequence hashing)."""
    ss = np.random.SeedSequence([int(p) for p in parts])
    return int(ss.generate_state(1)[0] % (2**31 - 1))


# Thread count for the models of THIS process. None keeps whatever
# models_paper.build_xgb_pipeline / _build_rf use (n_jobs=-1). Parallel
# workers set it to 1 so that n_jobs workers do not oversubscribe the CPU.
_MODEL_THREADS: int | None = None


def set_model_threads(n: int | None) -> None:
    global _MODEL_THREADS
    _MODEL_THREADS = None if n is None else int(n)


def _make_xgb(params: dict, rands: int) -> Pipeline:
    """models_paper XGB pipeline, optionally forced to a fixed thread count."""
    model = build_xgb_pipeline(params, random_state=rands)
    if _MODEL_THREADS is not None:
        model.set_params(xgb__n_jobs=_MODEL_THREADS)
    return model


# ============================================================
# Data preparation
# ============================================================

def _resolve_target_col(target_col, y_ss, y_dp) -> str:
    """Use target_col if given, otherwise the (common) name of the y Series."""
    if target_col is not None:
        return target_col
    names = {getattr(y_ss, "name", None), getattr(y_dp, "name", None)}
    names.discard(None)
    if len(names) != 1:
        raise ValueError(
            "target_col not given and it cannot be inferred from y_ss / y_dp "
            f"names: {names or 'no names'}."
        )
    return names.pop()


def prepare_domain_frame(X: pd.DataFrame, y, target_col: str, prefix: str) -> pd.DataFrame:
    """
    Raw table + target column, with a domain-prefixed sample_id.

    The prefix keeps SS and DP groups in separate namespaces, so ids from the
    two domains can never collide in a pooled GroupKFold.
    """
    if "sample_id" not in X.columns:
        raise KeyError("X must retain 'sample_id' (needed for group-aware CV).")

    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} rows.")
    if target_col in X.columns:
        raise ValueError(
            f"Target column '{target_col}' is already present in X."
        )

    X["sample_id"] = prefix + "_" + X["sample_id"].astype(str)
    X[target_col] = y.to_numpy()
    return X


def align_X(X_source, X_target, verbose: bool = False):
    """
    Restrict two RAW tables to their common columns.

    Optional with the fold-local cleaning used here: pooled cleaning handles
    columns that exist in only one domain (missing values are median-imputed or
    the feature is removed by the missingness rules). Kept for diagnostics.
    """
    common = X_source.columns.intersection(X_target.columns).tolist()
    if verbose:
        only_s = sorted(set(X_source.columns) - set(common))
        only_t = sorted(set(X_target.columns) - set(common))
        print(f"  align_X: {len(common)} common columns, "
              f"{len(only_s)} source-only, {len(only_t)} target-only")
    return X_source[common].copy(), X_target[common].copy(), common


# ============================================================
# Metrics and importance weights
# ============================================================

def compute_regression_metrics(y_true, y_pred, tol_rel=0.15):
    """
    Regression metrics including tol15: percentage of points with
    |y_true - y_pred| <= tol_rel * |y_true|.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    # np.ptp (range) instead of np.std: near-constant float predictions can have
    # std > 0 and still trigger scipy's ConstantInputWarning.
    if len(y_true) > 1 and np.ptp(y_true) > 0 and np.ptp(y_pred) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Pearsons_r": r, "tol15": tol15}


def _domain_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def compute_importance_weights(
    X_pool: pd.DataFrame,
    is_dp_flag: np.ndarray,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Importance weights via a domain classifier (logistic regression).

    SS rows get p(DP|x) / (1 - p(DP|x)), clipped to [clip_min, clip_max];
    DP rows get weight 1. When groups are given, p(DP|x) is CROSS-FITTED with
    StratifiedGroupKFold, so no row is scored by a classifier that saw its own
    sample (in-sample probabilities are overconfident when DP rows are few).
    Falls back to the in-sample fit if too few groups are available.

    X_pool must be cleaned (finite, numeric).
    """
    X_arr = np.asarray(X_pool, dtype=float)
    flag = np.asarray(is_dp_flag).astype(int)

    if not np.isfinite(X_arr).all():
        raise ValueError("X_pool must be cleaned (NaN/inf found).")
    if flag.sum() == 0 or flag.sum() == len(flag):
        raise ValueError("Importance weights need rows from both domains.")

    p_dp = np.full(len(X_arr), np.nan)
    crossfit_ok = False

    if groups is not None:
        groups = np.asarray(groups)
        k = min(
            n_splits,
            len(np.unique(groups[flag == 1])),
            len(np.unique(groups[flag == 0])),
        )
        if k >= 2:
            crossfit_ok = True
            splitter = StratifiedGroupKFold(
                n_splits=k, shuffle=True, random_state=random_state
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                for tr, te in splitter.split(X_arr, flag, groups):
                    if len(np.unique(flag[tr])) < 2:
                        crossfit_ok = False
                        break
                    clf = _domain_classifier().fit(X_arr[tr], flag[tr])
                    p_dp[te] = clf.predict_proba(X_arr[te])[:, 1]

    if not crossfit_ok or np.isnan(p_dp).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = _domain_classifier().fit(X_arr, flag)
        p_dp = clf.predict_proba(X_arr)[:, 1]

    return np.where(
        flag == 1,
        1.0,
        np.clip(p_dp / np.clip(1.0 - p_dp, 1e-6, 1.0), clip_min, clip_max),
    )


# ============================================================
# Inner folds (fold-local cleaning, optional importance weights)
# ============================================================

def _build_inner_folds(
    df_train_raw: pd.DataFrame,
    target_col: str,
    n_splits_inner: int,
    cleaning_kwargs: dict,
    is_dp_raw: np.ndarray | None = None,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    tune_on_dp_only: bool = False,
    random_state: int = 42,
) -> list[dict]:
    """
    Group-aware inner folds on a RAW training table (RangeIndex, sample_id,
    target column). Cleaning is fitted on each inner training part only; for
    the pooled branch (is_dp_raw given) the importance weights are also
    computed inside each inner training part.

    If tune_on_dp_only is True, pooled inner folds are scored on their DP
    validation rows only (when at least 2 are available).
    """
    groups = df_train_raw["sample_id"].to_numpy()
    k = min(n_splits_inner, len(np.unique(groups)))
    if k < 2:
        raise ValueError("Fewer than 2 sample_id groups available for inner CV.")

    folds = []
    for tr, va in GroupKFold(n_splits=k).split(df_train_raw, groups=groups):
        df_tr = df_train_raw.iloc[tr]
        df_va = df_train_raw.iloc[va]

        X_tr, y_tr, X_va, y_va, _ = prepare_clean_fold(
            df_tr, df_va, target_col, verbose=False, **cleaning_kwargs
        )

        w_tr = None
        if is_dp_raw is not None:
            pos_tr = X_tr.index.to_numpy()
            w_tr = compute_importance_weights(
                X_tr,
                is_dp_raw[pos_tr],
                clip_min,
                clip_max,
                groups=groups[pos_tr],
                random_state=random_state,
            )
            if tune_on_dp_only:
                keep = is_dp_raw[X_va.index.to_numpy()] == 1
                if keep.sum() >= 2:
                    X_va, y_va = X_va.loc[keep], y_va.loc[keep]

        folds.append({
            "X_train": X_tr, "y_train": y_tr, "w_train": w_tr,
            "X_valid": X_va, "y_valid": y_va,
        })
    return folds


# ============================================================
# Tuning and fitting
# ============================================================

def _score_inner(model_factory, inner_folds) -> float:
    """Mean inner-fold R2 (weights are used for fitting, not for scoring)."""
    scores = []
    for f in inner_folds:
        model = model_factory()
        _fit_with_weights(model, f["X_train"], f["y_train"], f["w_train"])
        pred = model.predict(f["X_valid"])
        scores.append(compute_regression_metrics(f["y_valid"], pred)["R2"])
    scores = np.asarray(scores, dtype=float)
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else -np.inf


def _fit_with_weights(model, X, y, w) -> None:
    """Fit a Pipeline (xgb step) or a plain estimator with optional weights."""
    if isinstance(model, Pipeline):
        model.fit(X, y, xgb__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)


def _tune_xgb(inner_folds, n_trials: int = 100, rands: int = 42) -> tuple[dict, optuna.Study]:
    """Tune XGB with the models_paper search space and a seeded TPE sampler."""
    def objective(trial):
        params = suggest_xgb_parameters(trial)
        return _score_inner(
            lambda: _make_xgb(params, rands), inner_folds
        )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=rands)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def _build_rf(params: dict, rands: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        **params, max_features="sqrt", bootstrap=True,
        random_state=rands,
        n_jobs=-1 if _MODEL_THREADS is None else _MODEL_THREADS,
    )


def _tune_rf(inner_folds, n_trials: int = 25, rands: int = 42) -> tuple[dict, optuna.Study]:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        return _score_inner(lambda: _build_rf(params, rands), inner_folds)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=rands)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def _tune(family: str, inner_folds, n_trials: int, rands: int) -> dict:
    if family == "XGB":
        return _tune_xgb(inner_folds, n_trials, rands)[0]
    return _tune_rf(inner_folds, n_trials, rands)[0]


def _fit_final(family: str, params: dict, X, y, w, rands: int):
    model = _make_xgb(params, rands) if family == "XGB" \
        else _build_rf(params, rands)
    _fit_with_weights(model, X, y, w)
    return model


# ============================================================
# Nested CV (XGB or RF)
# ============================================================

def eval_transfer_nested(
    family: str,
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp_subset: pd.DataFrame,
    y_dp_subset: pd.Series,
    target_col: str | None = None,
    rand_state: int = 42,
    n_splits_outer: int = 3,
    n_splits_inner: int = 3,
    n_trials_inner: int = 25,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    tol_rel: float = 0.15,
    cleaning_kwargs: dict | None = None,
    tune_on_dp_only: bool = False,
    return_oof: bool = False,
):
    """
    Fully nested CV comparing a DP-only model with an importance-weighted
    pooled SS+DP model on a DP subset (see module docstring).

    X_ss, X_dp_subset are RAW source tables that still contain sample_id.
    Returns per-fold metrics (df_dp_only, df_ss2dp), both evaluated on the
    same DP test rows. With return_oof=True a third frame with the pooled
    out-of-fold predictions (fold, sample_id, y_true, pred_dp_only,
    pred_ss2dp) is returned as well.
    """
    family = family.upper()
    if family not in {"XGB", "RF"}:
        raise ValueError("family must be 'XGB' or 'RF'.")

    ck = dict(cleaning_kwargs or CLEANING_KWARGS)
    target_col = _resolve_target_col(target_col, y_ss, y_dp_subset)

    df_ss = prepare_domain_frame(X_ss, y_ss, target_col, "ss")
    df_dp = prepare_domain_frame(X_dp_subset, y_dp_subset, target_col, "dp")

    dp_groups = pd.factorize(df_dp["sample_id"])[0]
    if len(np.unique(dp_groups)) < n_splits_outer:
        raise ValueError(
            f"Only {len(np.unique(dp_groups))} DP sample_id groups for "
            f"n_splits_outer={n_splits_outer}."
        )

    metrics_dp_only, metrics_ss2dp, oof_parts = [], [], []

    for fold_idx, (tr, te) in enumerate(
        GroupKFold(n_splits=n_splits_outer).split(df_dp, groups=dp_groups)
    ):
        seed = _derive_seed(rand_state, fold_idx)
        df_dp_tr = df_dp.iloc[tr].reset_index(drop=True)
        df_dp_te = df_dp.iloc[te].reset_index(drop=True)

        # ---- DP-only branch: cleaning fitted on DP train only ----
        X_tr, y_tr, X_te, y_te, _ = prepare_clean_fold(
            df_dp_tr, df_dp_te, target_col, verbose=False, **ck
        )
        inner_dp = _build_inner_folds(
            df_dp_tr, target_col, n_splits_inner, ck, random_state=seed
        )
        params_dp = _tune(family, inner_dp, n_trials_inner, seed)
        model_dp = _fit_final(family, params_dp, X_tr, y_tr, None, seed)

        # ---- SS -> DP branch: cleaning fitted on SS + DP train ----
        df_pool = pd.concat([df_ss, df_dp_tr], ignore_index=True)
        is_dp_raw = np.r_[
            np.zeros(len(df_ss), dtype=int), np.ones(len(df_dp_tr), dtype=int)
        ]
        Xp_tr, yp_tr, Xp_te, yp_te, _ = prepare_clean_fold(
            df_pool, df_dp_te, target_col, verbose=False, **ck
        )
        pos = Xp_tr.index.to_numpy()
        w_pool = compute_importance_weights(
            Xp_tr, is_dp_raw[pos], clip_min, clip_max,
            groups=df_pool["sample_id"].to_numpy()[pos], random_state=seed,
        )
        inner_pool = _build_inner_folds(
            df_pool, target_col, n_splits_inner, ck,
            is_dp_raw=is_dp_raw, clip_min=clip_min, clip_max=clip_max,
            tune_on_dp_only=tune_on_dp_only, random_state=seed,
        )
        params_pool = _tune(family, inner_pool, n_trials_inner, seed)
        model_pool = _fit_final(family, params_pool, Xp_tr, yp_tr, w_pool, seed)

        # ---- Paired evaluation on the same DP test rows ----
        common = X_te.index.intersection(Xp_te.index)
        if len(common) == 0:
            raise RuntimeError(f"No common DP test rows in outer fold {fold_idx}.")

        y_true = y_te.loc[common]
        pred_dp = np.asarray(model_dp.predict(X_te.loc[common])).ravel()
        pred_pool = np.asarray(model_pool.predict(Xp_te.loc[common])).ravel()
        m_dp = compute_regression_metrics(y_true, pred_dp, tol_rel=tol_rel)
        m_pool = compute_regression_metrics(y_true, pred_pool, tol_rel=tol_rel)
        m_dp["n_test"] = len(common)
        m_pool["n_test"] = len(common)
        metrics_dp_only.append(m_dp)
        metrics_ss2dp.append(m_pool)

        oof_parts.append(pd.DataFrame({
            "fold": fold_idx,
            "sample_id": df_dp_te.loc[common, "sample_id"].to_numpy(),
            "y_true": y_true.to_numpy(),
            "pred_dp_only": pred_dp,
            "pred_ss2dp": pred_pool,
        }))

    df_dp_only = pd.DataFrame(metrics_dp_only)
    df_ss2dp = pd.DataFrame(metrics_ss2dp)
    if return_oof:
        return df_dp_only, df_ss2dp, pd.concat(oof_parts, ignore_index=True)
    return df_dp_only, df_ss2dp


def metrics_from_oof(df_oof: pd.DataFrame, tol_rel: float = 0.15):
    """Metrics of both branches on the pooled out-of-fold predictions."""
    m_base = compute_regression_metrics(
        df_oof["y_true"], df_oof["pred_dp_only"], tol_rel=tol_rel
    )
    m_tr = compute_regression_metrics(
        df_oof["y_true"], df_oof["pred_ss2dp"], tol_rel=tol_rel
    )
    return m_base, m_tr


def eval_transfer_nested_xgb(*args, **kwargs):
    """Backward-compatible wrapper: nested CV with XGB."""
    return eval_transfer_nested("XGB", *args, **kwargs)


def eval_transfer_nested_rf(*args, **kwargs):
    """Backward-compatible wrapper: nested CV with RF."""
    return eval_transfer_nested("RF", *args, **kwargs)


# ============================================================
# Final pooled model (for SHAP and deployment)
# ============================================================

def fit_final_transfer_xgb(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp: pd.DataFrame,
    y_dp: pd.Series,
    target_col: str | None = None,
    n_trials: int = 100,
    n_splits_inner: int = 3,
    random_state: int = 2026,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    cleaning_kwargs: dict | None = None,
    tune_on_dp_only: bool = False,
):
    """
    Tune and fit the final importance-weighted SS+DP XGB model on ALL data.

    Cleaning is fitted once on SS + DP; weights come from a cross-fitted
    domain classifier on the cleaned pool; hyperparameters are tuned with
    group-aware inner CV (fold-local cleaning and weights).

    Returns
    -------
    bundle : dict
        model (Pipeline: CustomScaler + XGB), cleaning_pipeline,
        feature_columns, target_column, best_params, best_inner_R2,
        n_train_raw, n_train_clean, n_features, X_train_clean, y_train_clean,
        is_dp (0 = SS, 1 = DP, aligned with X_train_clean), sample_ids,
        sample_weight.
    study : optuna.Study
    """
    ck = dict(cleaning_kwargs or CLEANING_KWARGS)
    target_col = _resolve_target_col(target_col, y_ss, y_dp)

    df_ss = prepare_domain_frame(X_ss, y_ss, target_col, "ss")
    df_dp = prepare_domain_frame(X_dp, y_dp, target_col, "dp")
    df_pool = pd.concat([df_ss, df_dp], ignore_index=True)
    is_dp_raw = np.r_[
        np.zeros(len(df_ss), dtype=int), np.ones(len(df_dp), dtype=int)
    ]

    X_pool, y_pool, cleaning_pipeline = fit_cleaning_pipeline(
        df_pool, target_col=target_col, verbose=False, **ck
    )
    pos = X_pool.index.to_numpy()
    is_dp = is_dp_raw[pos]
    sample_ids = df_pool["sample_id"].to_numpy()[pos]

    if is_dp.sum() == 0:
        raise RuntimeError("No DP rows remain after cleaning.")

    weights = compute_importance_weights(
        X_pool, is_dp, clip_min, clip_max,
        groups=sample_ids, random_state=random_state,
    )

    inner_folds = _build_inner_folds(
        df_pool, target_col, n_splits_inner, ck,
        is_dp_raw=is_dp_raw, clip_min=clip_min, clip_max=clip_max,
        tune_on_dp_only=tune_on_dp_only, random_state=random_state,
    )
    best_params, study = _tune_xgb(inner_folds, n_trials, random_state)

    model = _make_xgb(best_params, random_state)
    model.fit(X_pool, y_pool, xgb__sample_weight=weights)

    bundle = {
        "model": model,
        "cleaning_pipeline": cleaning_pipeline,
        "feature_columns": X_pool.columns.tolist(),
        "target_column": target_col,
        "best_params": best_params,
        "best_inner_R2": study.best_value,
        "n_train_raw": len(df_pool),
        "n_train_clean": len(X_pool),
        "n_features": X_pool.shape[1],
        "X_train_clean": X_pool,
        "y_train_clean": y_pool,
        "is_dp": is_dp,
        "sample_ids": sample_ids,
        "sample_weight": weights,
    }
    return bundle, study


# ============================================================
# Learning-curve experiment (transfer effect)
# ============================================================

def _subsample_dp(X_dp, y_dp, N: int, rng, unit: str):
    """Draw N DP rows ('row') or N DP samples with all their rows ('sample')."""
    y_dp = pd.Series(y_dp).reset_index(drop=True)
    X_dp = X_dp.reset_index(drop=True)

    if unit == "row":
        idx = rng.choice(len(X_dp), size=N, replace=False)
    elif unit == "sample":
        ids = X_dp["sample_id"].drop_duplicates().to_numpy()
        chosen = rng.choice(ids, size=N, replace=False)
        idx = np.flatnonzero(X_dp["sample_id"].isin(chosen).to_numpy())
    else:
        raise ValueError("subsample_unit must be 'row' or 'sample'.")

    return X_dp.iloc[idx].reset_index(drop=True), y_dp.iloc[idx].reset_index(drop=True)


def _run_unit(
    N: int,
    r: int,
    *,
    model: str,
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp: pd.DataFrame,
    y_dp,
    target_col: str,
    base_seed: int,
    subsample_unit: str,
    n_splits_outer: int,
    n_splits_inner: int,
    n_trials_inner: int,
    clip_min: float,
    clip_max: float,
    cleaning_kwargs: dict | None,
    tune_on_dp_only: bool,
    model_threads: int | None,
    deadline: float | None,
) -> dict | None:
    """
    One (N, repetition) unit: subsample, nested CV, metrics row.

    Module-level so joblib workers can run it. Returns None if the wall-clock
    deadline has passed before the unit started.
    """
    if deadline is not None and time.time() > deadline:
        return None

    set_model_threads(model_threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()

    # Subsampling seed is base_seed + r exactly as in 1.1.0 (same DP subsets,
    # nested across N within a repetition).
    rng = np.random.RandomState(base_seed + r)
    X_dp_sub, y_dp_sub = _subsample_dp(X_dp, y_dp, N, rng, subsample_unit)

    df_dp_only, df_ss2dp, df_oof = eval_transfer_nested(
        model,
        X_ss=X_ss,
        y_ss=y_ss,
        X_dp_subset=X_dp_sub,
        y_dp_subset=y_dp_sub,
        target_col=target_col,
        rand_state=_derive_seed(base_seed, N, r),
        n_splits_outer=n_splits_outer,
        n_splits_inner=n_splits_inner,
        n_trials_inner=n_trials_inner,
        clip_min=clip_min,
        clip_max=clip_max,
        cleaning_kwargs=cleaning_kwargs,
        tune_on_dp_only=tune_on_dp_only,
        return_oof=True,
    )

    # Primary metrics: pooled out-of-fold predictions of this repetition.
    m_base, m_tr = metrics_from_oof(df_oof)

    row = {"N": int(N), "rep": int(r), "model": model, "target": target_col,
           "n_test": int(len(df_oof))}
    row.update({f"base_{k}": float(v) for k, v in m_base.items()})
    row.update({f"tr_{k}": float(v) for k, v in m_tr.items()})
    row["d_R2"] = row["tr_R2"] - row["base_R2"]
    row["d_MAE"] = row["tr_MAE"] - row["base_MAE"]
    row["d_RMSE"] = row["tr_RMSE"] - row["base_RMSE"]
    row["d_tol15"] = row["tr_tol15"] - row["base_tol15"]

    # Per-fold means (the 1.1.0 definition), kept for comparison.
    for k in df_dp_only.columns:
        if k == "n_test":
            continue
        row[f"foldmean_base_{k}"] = float(df_dp_only[k].mean())
        row[f"foldmean_tr_{k}"] = float(df_ss2dp[k].mean())
    row["foldmean_d_R2"] = row["foldmean_tr_R2"] - row["foldmean_base_R2"]
    row["foldmean_d_MAE"] = row["foldmean_tr_MAE"] - row["foldmean_base_MAE"]

    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return row


def _load_checkpoint(path: str | None) -> list[dict]:
    """Read finished units from a JSONL file (a truncated last line is ignored)."""
    rows: list[dict] = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _append_checkpoint(path: str | None, row: dict) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def run_transfer_learning_curve(
    X_ss: pd.DataFrame,
    y_ss: pd.Series,
    X_dp: pd.DataFrame,
    y_dp,
    target_col: str | None = None,
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
    subsample_unit: str = "row",
    cleaning_kwargs: dict | None = None,
    tune_on_dp_only: bool = False,
    n_jobs: int = 1,
    model_threads: int | None = None,
    checkpoint_path: str | None = None,
    drop_incomplete: bool = True,
):
    """
    Repeated subsampling experiments for multiple N (RF or XGB) with fully
    nested, fold-local cleaning and tuning.

    X_ss, X_dp are RAW source tables (sample_id retained); y_* are Series
    holding the target (their name is used as target_col if not given).
    subsample_unit: 'row' draws N DP rows (images, as in earlier versions);
    'sample' draws N DP samples (sample_id) with all of their rows.

    n_jobs           : worker processes over (N, rep) units (1 = in-process).
    model_threads    : threads per XGB/RF model. Default: 1 when n_jobs != 1,
                       otherwise whatever models_paper uses.
    checkpoint_path  : JSONL file; finished units are appended as they
                       complete and skipped on restart.
    max_time         : wall-clock limit in seconds; units not started before
                       the deadline are skipped (running units finish).
    drop_incomplete  : N values with fewer than n_reps finished repetitions
                       are reported and left out of the summary.

    Returns
    -------
    df_runs, df_summary
    """
    model = model.upper()
    if model not in {"RF", "XGB"}:
        raise ValueError(f"Unsupported model '{model}', choose 'RF' or 'XGB'.")

    target_col = _resolve_target_col(target_col, y_ss, y_dp)
    n_available = (
        len(X_dp) if subsample_unit == "row" else X_dp["sample_id"].nunique()
    )
    if N_values is None:
        N_values = [n_available]

    N_list = []
    for N in N_values:
        if N > n_available:
            print(f"Skipping N={N}: exceeds available DP {subsample_unit}s ({n_available})")
        else:
            N_list.append(int(N))

    # ---- resume from checkpoint ----
    done: dict[tuple[int, int], dict] = {}
    for row in _load_checkpoint(checkpoint_path):
        if row.get("model") == model and row.get("target") == target_col:
            done[(int(row["N"]), int(row["rep"]))] = row
    wanted = {(N, r) for N in N_list for r in range(n_reps)}
    n_done = len(wanted & set(done))
    pending = [(N, r) for N in N_list for r in range(n_reps) if (N, r) not in done]
    print(f"N values used: {N_list} | reps: {n_reps}")
    print(f"[{model}] {target_col}: {len(wanted)} units, {n_done} already done, "
          f"{len(pending)} to run (n_jobs={n_jobs})")

    if model_threads is None and n_jobs != 1:
        model_threads = 1
    deadline = None if max_time is None else time.time() + max_time

    shared = dict(
        model=model, X_ss=X_ss, y_ss=y_ss, X_dp=X_dp, y_dp=y_dp,
        target_col=target_col, base_seed=base_seed,
        subsample_unit=subsample_unit, n_splits_outer=n_splits_outer,
        n_splits_inner=n_splits_inner, n_trials_inner=n_trials_inner,
        clip_min=clip_min, clip_max=clip_max, cleaning_kwargs=cleaning_kwargs,
        tune_on_dp_only=tune_on_dp_only, model_threads=model_threads,
        deadline=deadline,
    )

    t_start = time.time()
    n_new = 0
    n_skipped = 0

    def _handle(row):
        nonlocal n_new, n_skipped
        if row is None:
            n_skipped += 1
            return
        done[(row["N"], row["rep"])] = row
        _append_checkpoint(checkpoint_path, row)
        n_new += 1
        print(f"  [{n_done + n_new}/{len(wanted)}] N={row['N']} rep={row['rep']} "
              f"d_MAE={row['d_MAE']:+.2f} d_R2={row['d_R2']:+.3f} "
              f"n_test={row['n_test']} ({row['elapsed_s']:.0f}s, "
              f"{(time.time() - t_start) / 60:.1f} min total)")

    if pending:
        if n_jobs == 1:
            for N, r in pending:
                _handle(_run_unit(N, r, **shared))
                if n_skipped:
                    break
        else:
            from joblib import Parallel, delayed
            try:
                from joblib import parallel_config
                ctx = parallel_config(backend="loky", inner_max_num_threads=1)
            except ImportError:  # joblib < 1.3
                ctx = contextlib.nullcontext()
            with ctx:
                try:
                    stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                        delayed(_run_unit)(N, r, **shared) for N, r in pending
                    )
                except TypeError:  # joblib < 1.3: no streaming, results at the end
                    stream = iter(Parallel(n_jobs=n_jobs)(
                        delayed(_run_unit)(N, r, **shared) for N, r in pending
                    ))
                for row in stream:
                    _handle(row)
        if n_skipped:
            print(f"Wall-clock time limit reached: {n_skipped} unit(s) not started.")

    # ---- assemble results (only the requested N / reps) ----
    rows = [done[k] for k in sorted(wanted & set(done))]
    df_runs = pd.DataFrame(rows)
    if df_runs.empty:
        return df_runs, pd.DataFrame()

    counts = df_runs.groupby("N")["rep"].nunique()
    incomplete = counts[counts < n_reps]
    df_for_summary = df_runs
    if len(incomplete):
        print("Incomplete N values (finished reps): "
              + ", ".join(f"N={n}: {c}/{n_reps}" for n, c in incomplete.items()))
        if drop_incomplete:
            df_for_summary = df_runs[~df_runs["N"].isin(incomplete.index)]
            print("  -> left out of the summary (kept in df_runs).")

    df_summary = (
        summarize_transfer_results(df_for_summary)
        if len(df_for_summary) else pd.DataFrame()
    )
    return df_runs, df_summary


# ============================================================
# Aggregation and plotting helpers
# ============================================================

def summarize_transfer_results(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-repetition results vs N to mean and 95% CI.

    CI = t(0.975, n_reps - 1) * SE of the mean over repetitions. Repetitions
    are random subsamples of the same DP pool and overlap, so they are not
    independent; the interval therefore still understates the true
    uncertainty and should be described as the variability over subsampling
    repetitions.
    """
    def se(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    spec = dict(
        n_reps=("rep", "nunique"),
        d_RMSE_mean=("d_RMSE", "mean"), d_RMSE_se=("d_RMSE", se),
        d_MAE_mean=("d_MAE", "mean"), d_MAE_se=("d_MAE", se),
        d_R2_mean=("d_R2", "mean"), d_R2_se=("d_R2", se),
        d_tol15_mean=("d_tol15", "mean"), d_tol15_se=("d_tol15", se),
        base_MAE_mean=("base_MAE", "mean"), base_MAE_se=("base_MAE", se),
        tr_MAE_mean=("tr_MAE", "mean"), tr_MAE_se=("tr_MAE", se),
        base_R2_mean=("base_R2", "mean"), tr_R2_mean=("tr_R2", "mean"),
    )
    if "n_test" in df_runs.columns:
        spec["n_test_mean"] = ("n_test", "mean")

    grouped = df_runs.groupby("N").agg(**spec).reset_index()

    dof = grouped["n_reps"] - 1
    t_crit = np.where(dof >= 1, student_t.ppf(0.975, np.maximum(dof, 1)), np.nan)

    for metric in ("d_RMSE", "d_MAE", "d_R2", "d_tol15", "base_MAE", "tr_MAE"):
        grouped[f"{metric}_ci95"] = t_crit * grouped[f"{metric}_se"]

    return grouped


def plot_transfer_learning_curve(
    df_summary: pd.DataFrame,
    prop: str = "",
    save_path: str | None = None,
):
    """
    Absolute MAE (DP-only vs SS->DP) and delta-MAE vs N with 95% CIs.
    Saves to save_path if given (and closes the figure), otherwise shows it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    title_suffix = f" - {prop.upper()}" if prop else ""

    ax = axes[0]
    ax.errorbar(df_summary["N"], df_summary["base_MAE_mean"],
                yerr=df_summary["base_MAE_ci95"], fmt="o-", capsize=3, label="DP-only")
    ax.errorbar(df_summary["N"], df_summary["tr_MAE_mean"],
                yerr=df_summary["tr_MAE_ci95"], fmt="s--", capsize=3, label="SS->DP")
    ax.set_xlabel("Number of DP training samples $N_T$")
    ax.set_ylabel("MAE")
    ax.set_title(f"Absolute MAE{title_suffix}")
    ax.legend()

    ax = axes[1]
    ax.errorbar(df_summary["N"], df_summary["d_MAE_mean"],
                yerr=df_summary["d_MAE_ci95"], fmt="s-", capsize=3, color="C1",
                label="delta-MAE (SS->DP minus DP-only)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of DP training samples $N_T$")
    ax.set_ylabel("delta-MAE")
    ax.set_title(f"Transfer improvement{title_suffix}")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ============================================================
# Diagnostics: where does the time go, how many trials are enough?
# ============================================================

def _setup_split(X_ss, y_ss, X_dp, y_dp, target_col, N, rep, base_seed,
                 subsample_unit, n_splits_outer):
    """First outer fold of one (N, rep) unit (same seeds as the real run)."""
    target_col = _resolve_target_col(target_col, y_ss, y_dp)
    rng = np.random.RandomState(base_seed + rep)
    X_sub, y_sub = _subsample_dp(X_dp, y_dp, N, rng, subsample_unit)
    df_ss = prepare_domain_frame(X_ss, y_ss, target_col, "ss")
    df_dp = prepare_domain_frame(X_sub, y_sub, target_col, "dp")
    dp_groups = pd.factorize(df_dp["sample_id"])[0]
    tr, te = next(iter(
        GroupKFold(n_splits=n_splits_outer).split(df_dp, groups=dp_groups)
    ))
    seed = _derive_seed(_derive_seed(base_seed, N, rep), 0)
    return (target_col, df_ss, df_dp.iloc[tr].reset_index(drop=True),
            df_dp.iloc[te].reset_index(drop=True), seed)


def profile_single_split(
    X_ss, y_ss, X_dp, y_dp, target_col=None, N: int = 100, rep: int = 0,
    n_trials_probe: int = 5, n_trials_full: int = 100,
    n_units: int | None = None, n_workers: int = 1,
    n_splits_outer: int = 3, n_splits_inner: int = 3, base_seed: int = 42,
    subsample_unit: str = "row", clip_min: float = 0.1, clip_max: float = 10.0,
    cleaning_kwargs: dict | None = None, tune_on_dp_only: bool = False,
):
    """
    Time every stage of ONE outer fold of one (N, rep) unit (XGB) and
    extrapolate to the full run. Tuning is probed with n_trials_probe trials
    and scaled linearly to n_trials_full (a rough estimate).
    """
    ck = dict(cleaning_kwargs or CLEANING_KWARGS)
    target_col, df_ss, df_tr, df_te, seed = _setup_split(
        X_ss, y_ss, X_dp, y_dp, target_col, N, rep, base_seed,
        subsample_unit, n_splits_outer,
    )
    times: dict[str, float] = {}

    def timed(label, fn):
        t0 = time.perf_counter()
        out = fn()
        times[label] = time.perf_counter() - t0
        return out

    # DP-only branch
    X_tr, y_tr, X_te, y_te, _ = timed(
        "DP-only | outer cleaning",
        lambda: prepare_clean_fold(df_tr, df_te, target_col, verbose=False, **ck))
    inner_dp = timed(
        "DP-only | inner folds (cleaning)",
        lambda: _build_inner_folds(df_tr, target_col, n_splits_inner, ck,
                                   random_state=seed))
    params = timed(
        f"DP-only | tuning ({n_trials_probe} trials)",
        lambda: _tune("XGB", inner_dp, n_trials_probe, seed))
    timed("DP-only | final fit", lambda: _fit_final("XGB", params, X_tr, y_tr, None, seed))

    # pooled branch
    df_pool = pd.concat([df_ss, df_tr], ignore_index=True)
    is_dp_raw = np.r_[np.zeros(len(df_ss), dtype=int), np.ones(len(df_tr), dtype=int)]
    Xp_tr, yp_tr, Xp_te, yp_te, _ = timed(
        "Pooled  | outer cleaning",
        lambda: prepare_clean_fold(df_pool, df_te, target_col, verbose=False, **ck))
    pos = Xp_tr.index.to_numpy()
    w_pool = timed(
        "Pooled  | importance weights",
        lambda: compute_importance_weights(
            Xp_tr, is_dp_raw[pos], clip_min, clip_max,
            groups=df_pool["sample_id"].to_numpy()[pos], random_state=seed))
    inner_pool = timed(
        "Pooled  | inner folds (cleaning + weights)",
        lambda: _build_inner_folds(
            df_pool, target_col, n_splits_inner, ck, is_dp_raw=is_dp_raw,
            clip_min=clip_min, clip_max=clip_max,
            tune_on_dp_only=tune_on_dp_only, random_state=seed))
    params_p = timed(
        f"Pooled  | tuning ({n_trials_probe} trials)",
        lambda: _tune("XGB", inner_pool, n_trials_probe, seed))
    timed("Pooled  | final fit",
          lambda: _fit_final("XGB", params_p, Xp_tr, yp_tr, w_pool, seed))

    total = sum(times.values())
    print(f"\nProfile: N={N}, rep={rep}, outer fold 0 "
          f"(SS rows {len(df_ss)}, DP train rows {len(df_tr)}, "
          f"pooled train rows after cleaning {len(Xp_tr)})")
    for label, sec in times.items():
        print(f"  {label:<44s}{sec:8.2f} s  ({100 * sec / total:4.1f} %)")

    scale = n_trials_full / max(n_trials_probe, 1)
    tuning = sum(v for k, v in times.items() if "tuning" in k)
    per_fold = (total - tuning) + tuning * scale
    per_unit = per_fold * n_splits_outer
    print(f"\nEstimate with {n_trials_full} trials: ~{per_fold:.0f} s per outer fold, "
          f"~{per_unit / 60:.1f} min per (N, rep) unit")
    if n_units:
        hours = per_unit * n_units / max(n_workers, 1) / 3600
        print(f"{n_units} units on {n_workers} worker(s): ~{hours:.1f} h "
              f"(ideal speed-up; larger N is slower than this N={N} probe)")
    return times


def trial_budget_check(
    X_ss, y_ss, X_dp, y_dp, target_col=None, N: int = 100, rep: int = 0,
    max_trials: int = 100, marks=(10, 25, 30, 50, 75, 100),
    n_splits_outer: int = 3, n_splits_inner: int = 3, base_seed: int = 42,
    subsample_unit: str = "row", clip_min: float = 0.1, clip_max: float = 10.0,
    cleaning_kwargs: dict | None = None, tune_on_dp_only: bool = False,
):
    """
    Best inner-CV R2 after k Optuna trials (outer fold 0 of one (N, rep),
    both branches). Shows how much a smaller n_trials_inner would cost.
    """
    ck = dict(cleaning_kwargs or CLEANING_KWARGS)
    target_col, df_ss, df_tr, _, seed = _setup_split(
        X_ss, y_ss, X_dp, y_dp, target_col, N, rep, base_seed,
        subsample_unit, n_splits_outer,
    )
    inner_dp = _build_inner_folds(df_tr, target_col, n_splits_inner, ck, random_state=seed)
    df_pool = pd.concat([df_ss, df_tr], ignore_index=True)
    is_dp_raw = np.r_[np.zeros(len(df_ss), dtype=int), np.ones(len(df_tr), dtype=int)]
    inner_pool = _build_inner_folds(
        df_pool, target_col, n_splits_inner, ck, is_dp_raw=is_dp_raw,
        clip_min=clip_min, clip_max=clip_max,
        tune_on_dp_only=tune_on_dp_only, random_state=seed,
    )

    print(f"\nBest inner R2 after k trials (N={N}, rep={rep}, outer fold 0)")
    for label, inner in (("DP-only", inner_dp), ("SS->DP ", inner_pool)):
        _, study = _tune_xgb(inner, n_trials=max_trials, rands=seed)
        vals = np.array([t.value for t in study.trials if t.value is not None],
                        dtype=float)
        best = np.maximum.accumulate(vals)
        txt = "  ".join(f"{k}: {best[k - 1]:.3f}" for k in marks if k <= len(best))
        print(f"  {label}  {txt}")


# ============================================================
# Entry point
# ============================================================

# Same target column names as models_paper.py / clean_data.py
TARGETS = {
    "uts": "UTS (MPa)",
    "ys": "YS (MPa)",
    "el": "Total EL (%)",
}


def _load_tables(xy_dir: str, prop: str):
    """RAW source tables: sample_id retained, no global cleaning."""
    X_ss = pd.read_csv(os.path.join(xy_dir, f"X_ss_sem_{prop}.csv"), low_memory=False)
    y_ss = pd.read_csv(os.path.join(xy_dir, f"y_ss_sem_{prop}.csv"),
                       low_memory=False).squeeze("columns")
    X_dp = pd.read_csv(os.path.join(xy_dir, f"X_dp_sem_{prop}.csv"), low_memory=False)
    y_dp = pd.read_csv(os.path.join(xy_dir, f"y_dp_sem_{prop}.csv"),
                       low_memory=False).squeeze("columns")
    return X_ss, y_ss, X_dp, y_dp


def _check_run_config(cfg_path: str, cfg: dict, force: bool) -> None:
    """Store the configuration; refuse to resume a run whose settings differ."""
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            old_cfg = json.load(f)
        if old_cfg != cfg and not force:
            changed = sorted(k for k in set(cfg) | set(old_cfg)
                             if cfg.get(k) != old_cfg.get(k))
            raise SystemExit(
                f"Stored configuration in {cfg_path} differs in {changed}.\n"
                "Use a new RUN_DIR, or set FORCE_RESUME = True to resume anyway."
            )
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    module_path = MODULE_PATH
    print("Module path:", module_path)

    xy_dir = os.path.join(module_path, "data", "Xy")

    # ----------------------------------------------------------
    # Experiment configuration (edit here)
    # ----------------------------------------------------------
    MODE            = "run"      # "run" | "profile" | "trial_check"
    PROPERTIES      = ["uts", "ys", "el"]   # run in this order, one after another
    MODEL           = "XGB"      # "XGB" or "RF"
    N_VALUES        = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280]
    N_REPS          = 10
    N_SPLITS_OUTER  = 3          # 3-fold: ensures ~6 test rows even at N=20
    N_SPLITS_INNER  = 3
    N_TRIALS_INNER  = 100        # Optuna trials per inner tuning (check with MODE = "trial_check")
    SUBSAMPLE_UNIT  = "row"      # 'row' (images) or 'sample' (sample_id)
    BASE_SEED       = 42
    CLIP_MIN        = 0.1
    CLIP_MAX        = 10.0
    TUNE_ON_DP_ONLY = False      # True: score pooled inner folds on DP validation rows only
    ALIGN_COLUMNS   = False      # True: restrict SS and DP to their common columns (align_X)

    # Execution
    N_JOBS          = max(1, (os.cpu_count() or 2) - 1)   # workers over (N, rep) units; 1 = serial
    MAX_TIME        = None       # wall-clock limit per target in seconds, or None
    RUN_DIR         = os.path.join(module_path, "results", "transfer_learning", "full_run_20261013") # None: new timestamped folder. Existing folder: resume it.
    FORCE_RESUME    = False      # resume even if the stored configuration differs
    SHOW_PLOTS      = False      # True: also open each plot window (blocks until closed)
    PROBE_N         = 100        # N used by MODE = "profile" / "trial_check"

    unknown = [p for p in PROPERTIES if p not in TARGETS]
    if unknown:
        raise SystemExit(f"Unknown properties {unknown}; choose from {list(TARGETS)}.")

    n_units = len(N_VALUES) * N_REPS
    print("-" * 60)
    print(f"MODE            : {MODE}")
    print(f"Properties      : {PROPERTIES} (sequential)")
    print(f"Model           : {MODEL}")
    print(f"N values ({len(N_VALUES):>2})   : {N_VALUES}")
    print(f"Repetitions     : {N_REPS}  ->  {n_units} (N, rep) units per property")
    print(f"CV              : outer {N_SPLITS_OUTER}-fold, inner {N_SPLITS_INNER}-fold, "
          f"{N_TRIALS_INNER} Optuna trials")
    print(f"Subsample unit  : {SUBSAMPLE_UNIT} | base seed {BASE_SEED} | "
          f"clip [{CLIP_MIN}, {CLIP_MAX}] | tune_on_dp_only={TUNE_ON_DP_ONLY}")
    print(f"Workers         : {N_JOBS} | max_time={MAX_TIME} | align_columns={ALIGN_COLUMNS}")
    print("-" * 60)

    common = dict(
        n_splits_outer=N_SPLITS_OUTER, n_splits_inner=N_SPLITS_INNER,
        base_seed=BASE_SEED, subsample_unit=SUBSAMPLE_UNIT,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX,
        cleaning_kwargs=CLEANING_KWARGS, tune_on_dp_only=TUNE_ON_DP_ONLY,
    )

    # ---- diagnostics only (first property) ----
    if MODE in ("profile", "trial_check"):
        prop = PROPERTIES[0]
        X_ss, y_ss, X_dp, y_dp = _load_tables(xy_dir, prop)
        if ALIGN_COLUMNS:
            X_ss, X_dp, _ = align_X(X_ss, X_dp, verbose=True)
        print(f"Diagnostics on target: {prop.upper()} "
              f"(SS rows {len(X_ss)}, DP rows {len(X_dp)})")
        if MODE == "profile":
            profile_single_split(
                X_ss, y_ss, X_dp, y_dp, TARGETS[prop], N=PROBE_N,
                n_trials_full=N_TRIALS_INNER, n_units=n_units,
                n_workers=N_JOBS, **common,
            )
        else:
            trial_budget_check(
                X_ss, y_ss, X_dp, y_dp, TARGETS[prop], N=PROBE_N,
                max_trials=N_TRIALS_INNER, **common,
            )
        sys.exit(0)

    if MODE != "run":
        raise SystemExit(f"Unknown MODE '{MODE}'.")

    # ---- full run ----
    if RUN_DIR:
        out_dir_root = RUN_DIR
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir_root = os.path.join(module_path, "results", "transfer_learning", timestamp)
    Path(out_dir_root).mkdir(parents=True, exist_ok=True)
    print("Saving results to:", out_dir_root)

    for prop in PROPERTIES:
        target_col = TARGETS[prop]
        print("=" * 60)
        print(f"Running nested transfer-learning curve for: {prop.upper()}")
        print("=" * 60)

        out_dir_prop = os.path.join(out_dir_root, prop)
        Path(out_dir_prop).mkdir(parents=True, exist_ok=True)

        # Configuration fingerprint: everything that changes the numbers of a
        # unit. N_VALUES, N_REPS, N_JOBS and MAX_TIME may change between resumes.
        cfg = {
            "model": MODEL, "target_col": target_col,
            "n_splits_outer": N_SPLITS_OUTER, "n_splits_inner": N_SPLITS_INNER,
            "n_trials_inner": N_TRIALS_INNER, "subsample_unit": SUBSAMPLE_UNIT,
            "base_seed": BASE_SEED, "tune_on_dp_only": TUNE_ON_DP_ONLY,
            "align_columns": ALIGN_COLUMNS, "clip": [CLIP_MIN, CLIP_MAX],
            "cleaning_kwargs": json.loads(json.dumps(CLEANING_KWARGS, default=str)),
            "seed_scheme": "seedsequence-v1",
        }
        _check_run_config(os.path.join(out_dir_prop, "config.json"), cfg, FORCE_RESUME)

        X_ss, y_ss, X_dp, y_dp = _load_tables(xy_dir, prop)
        if ALIGN_COLUMNS:
            X_ss, X_dp, _ = align_X(X_ss, X_dp, verbose=True)
        n_dp_samples = X_dp["sample_id"].nunique()
        print(f"Data: SS {len(X_ss)} rows, DP {len(X_dp)} rows "
              f"({n_dp_samples} DP samples), {X_dp.shape[1]} DP columns")

        df_runs, df_summary = run_transfer_learning_curve(
            X_ss=X_ss, y_ss=y_ss, X_dp=X_dp, y_dp=y_dp,
            target_col=target_col,
            N_values=N_VALUES,
            n_reps=N_REPS,
            n_trials_inner=N_TRIALS_INNER,
            max_time=MAX_TIME,
            model=MODEL,
            n_jobs=N_JOBS,
            checkpoint_path=os.path.join(
                out_dir_prop, f"transfer_runs_{MODEL}_{prop}.jsonl"),
            **common,
        )
        if df_runs.empty:
            print("No finished units; nothing to save.")
            continue

        stem = f"{MODEL}_{prop}"
        df_runs.to_csv(os.path.join(out_dir_prop, f"transfer_runs_{stem}.csv"), index=False)
        df_summary.to_csv(os.path.join(out_dir_prop, f"transfer_summary_{stem}.csv"), index=False)
        with open(os.path.join(out_dir_prop, f"transfer_runs_{stem}.pkl"), "wb") as f:
            pickle.dump(df_runs, f)
        with open(os.path.join(out_dir_prop, f"transfer_summary_{stem}.pkl"), "wb") as f:
            pickle.dump(df_summary, f)

        if len(df_summary):
            plot_transfer_learning_curve(
                df_summary, prop=prop,
                save_path=os.path.join(out_dir_prop, f"transfer_curve_{stem}.png"),
            )
            if SHOW_PLOTS:
                plot_transfer_learning_curve(df_summary, prop=prop)

    print("Done.")
    
    ##### Test-run 

    # # ----------------------------------------------------------
    # # Experiment configuration (edit here)
    # # ----------------------------------------------------------
    # MODE            = "run"      # "run" | "profile" | "trial_check"
    # PROPERTIES      = ["uts", "ys", "el"]   # run in this order, one after another
    # MODEL           = "XGB"      # "XGB" or "RF"
    # N_VALUES        = [20, 280]
    # N_REPS          = 1
    # N_SPLITS_OUTER  = 3          # 3-fold: ensures ~7 test rows even at N=20
    # N_SPLITS_INNER  = 3
    # N_TRIALS_INNER  = 3        # Optuna trials per inner tuning (check with MODE = "trial_check")
    # SUBSAMPLE_UNIT  = "row"      # 'row' (images) or 'sample' (sample_id)
    # BASE_SEED       = 42
    # CLIP_MIN        = 0.1
    # CLIP_MAX        = 10.0
    # TUNE_ON_DP_ONLY = False      # True: score pooled inner folds on DP validation rows only
    # ALIGN_COLUMNS   = False      # True: restrict SS and DP to their common columns (align_X)

    # # Execution
    # N_JOBS          = 2
    # MAX_TIME        = None       # wall-clock limit per target in seconds, or None
    # RUN_DIR         = None       # None: new timestamped folder. Existing folder: resume it.
    # FORCE_RESUME    = False      # resume even if the stored configuration differs
    # SHOW_PLOTS      = False      # True: also open each plot window (blocks until closed)
    # PROBE_N         = 100        # N used by MODE = "profile" / "trial_check"

    # unknown = [p for p in PROPERTIES if p not in TARGETS]
    # if unknown:
    #     raise SystemExit(f"Unknown properties {unknown}; choose from {list(TARGETS)}.")

    # n_units = len(N_VALUES) * N_REPS
    # print("-" * 60)
    # print(f"MODE            : {MODE}")
    # print(f"Properties      : {PROPERTIES} (sequential)")
    # print(f"Model           : {MODEL}")
    # print(f"N values ({len(N_VALUES):>2})   : {N_VALUES}")
    # print(f"Repetitions     : {N_REPS}  ->  {n_units} (N, rep) units per property")
    # print(f"CV              : outer {N_SPLITS_OUTER}-fold, inner {N_SPLITS_INNER}-fold, "
    #       f"{N_TRIALS_INNER} Optuna trials")
    # print(f"Subsample unit  : {SUBSAMPLE_UNIT} | base seed {BASE_SEED} | "
    #       f"clip [{CLIP_MIN}, {CLIP_MAX}] | tune_on_dp_only={TUNE_ON_DP_ONLY}")
    # print(f"Workers         : {N_JOBS} | max_time={MAX_TIME} | align_columns={ALIGN_COLUMNS}")
    # print("-" * 60)

    # common = dict(
    #     n_splits_outer=N_SPLITS_OUTER, n_splits_inner=N_SPLITS_INNER,
    #     base_seed=BASE_SEED, subsample_unit=SUBSAMPLE_UNIT,
    #     clip_min=CLIP_MIN, clip_max=CLIP_MAX,
    #     cleaning_kwargs=CLEANING_KWARGS, tune_on_dp_only=TUNE_ON_DP_ONLY,
    # )

    # # ---- diagnostics only (first property) ----
    # if MODE in ("profile", "trial_check"):
    #     prop = PROPERTIES[0]
    #     X_ss, y_ss, X_dp, y_dp = _load_tables(xy_dir, prop)
    #     if ALIGN_COLUMNS:
    #         X_ss, X_dp, _ = align_X(X_ss, X_dp, verbose=True)
    #     print(f"Diagnostics on target: {prop.upper()} "
    #           f"(SS rows {len(X_ss)}, DP rows {len(X_dp)})")
    #     if MODE == "profile":
    #         profile_single_split(
    #             X_ss, y_ss, X_dp, y_dp, TARGETS[prop], N=PROBE_N,
    #             n_trials_full=N_TRIALS_INNER, n_units=n_units,
    #             n_workers=N_JOBS, **common,
    #         )
    #     else:
    #         trial_budget_check(
    #             X_ss, y_ss, X_dp, y_dp, TARGETS[prop], N=PROBE_N,
    #             max_trials=N_TRIALS_INNER, **common,
    #         )
    #     sys.exit(0)

    # if MODE != "run":
    #     raise SystemExit(f"Unknown MODE '{MODE}'.")

    # # ---- full run ----
    # if RUN_DIR:
    #     out_dir_root = RUN_DIR
    # else:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     out_dir_root = os.path.join(module_path, "results", "transfer_learning", timestamp)
    # Path(out_dir_root).mkdir(parents=True, exist_ok=True)
    # print("Saving results to:", out_dir_root)

    # for prop in PROPERTIES:
    #     target_col = TARGETS[prop]
    #     print("=" * 60)
    #     print(f"Running nested transfer-learning curve for: {prop.upper()}")
    #     print("=" * 60)

    #     out_dir_prop = os.path.join(out_dir_root, prop)
    #     Path(out_dir_prop).mkdir(parents=True, exist_ok=True)

    #     # Configuration fingerprint: everything that changes the numbers of a
    #     # unit. N_VALUES, N_REPS, N_JOBS and MAX_TIME may change between resumes.
    #     cfg = {
    #         "model": MODEL, "target_col": target_col,
    #         "n_splits_outer": N_SPLITS_OUTER, "n_splits_inner": N_SPLITS_INNER,
    #         "n_trials_inner": N_TRIALS_INNER, "subsample_unit": SUBSAMPLE_UNIT,
    #         "base_seed": BASE_SEED, "tune_on_dp_only": TUNE_ON_DP_ONLY,
    #         "align_columns": ALIGN_COLUMNS, "clip": [CLIP_MIN, CLIP_MAX],
    #         "cleaning_kwargs": json.loads(json.dumps(CLEANING_KWARGS, default=str)),
    #         "seed_scheme": "seedsequence-v1",
    #     }
    #     _check_run_config(os.path.join(out_dir_prop, "config.json"), cfg, FORCE_RESUME)

    #     X_ss, y_ss, X_dp, y_dp = _load_tables(xy_dir, prop)
    #     if ALIGN_COLUMNS:
    #         X_ss, X_dp, _ = align_X(X_ss, X_dp, verbose=True)
    #     n_dp_samples = X_dp["sample_id"].nunique()
    #     print(f"Data: SS {len(X_ss)} rows, DP {len(X_dp)} rows "
    #           f"({n_dp_samples} DP samples), {X_dp.shape[1]} DP columns")

    #     df_runs, df_summary = run_transfer_learning_curve(
    #         X_ss=X_ss, y_ss=y_ss, X_dp=X_dp, y_dp=y_dp,
    #         target_col=target_col,
    #         N_values=N_VALUES,
    #         n_reps=N_REPS,
    #         n_trials_inner=N_TRIALS_INNER,
    #         max_time=MAX_TIME,
    #         model=MODEL,
    #         n_jobs=N_JOBS,
    #         checkpoint_path=os.path.join(
    #             out_dir_prop, f"transfer_runs_{MODEL}_{prop}.jsonl"),
    #         **common,
    #     )
    #     if df_runs.empty:
    #         print("No finished units; nothing to save.")
    #         continue

    #     stem = f"{MODEL}_{prop}"
    #     df_runs.to_csv(os.path.join(out_dir_prop, f"transfer_runs_{stem}.csv"), index=False)
    #     df_summary.to_csv(os.path.join(out_dir_prop, f"transfer_summary_{stem}.csv"), index=False)
    #     with open(os.path.join(out_dir_prop, f"transfer_runs_{stem}.pkl"), "wb") as f:
    #         pickle.dump(df_runs, f)
    #     with open(os.path.join(out_dir_prop, f"transfer_summary_{stem}.pkl"), "wb") as f:
    #         pickle.dump(df_summary, f)

    #     if len(df_summary):
    #         plot_transfer_learning_curve(
    #             df_summary, prop=prop,
    #             save_path=os.path.join(out_dir_prop, f"transfer_curve_{stem}.png"),
    #         )
    #         if SHOW_PLOTS:
    #             plot_transfer_learning_curve(df_summary, prop=prop)

    # print("Done.")
    
    
    
    
    
    