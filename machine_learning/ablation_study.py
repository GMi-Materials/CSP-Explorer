# -*- coding: utf-8 -*-
"""
Feature-set ablation for the nested group-aware XGBoost pipeline.

Compares chemistry-only, image-only and combined models on identical outer
splits, identical fold-local cleaned rows and identical inner CV partitions.
Only the feature columns differ between the three models.

Stage 1: build_fold_datasets()  (cleaning, done once, optionally cached)
Stage 2: run_ablation()         (tuning, refit and evaluation per feature set)
Report : summarize_ablation()

__author__ = "Gerfried Millner"
__version__ = "1.0.0"
__date__ = "13.10.2026"
__email__ = "gerfried.millner@ntu.edu.sg"
__status__ = "Developing"

"""
from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

# models_paper must be imported first: it prepends the project root to
# sys.path, which is required for the data_processing import below.
from models_paper import (
    CLEANING_KWARGS,
    build_xgb_pipeline,
    combine_X_y,
    compute_regression_metrics,
    get_cv_splits,
    make_xgb_optuna_objective,
    precompute_cleaned_inner_folds,
    prepare_model_fold,
)
from data_processing.clean_data import (
    CHEMISTRY_COLUMNS,
    DEFAULT_ID_COLUMNS,
    IMAGE_DESCRIPTOR_PREFIXES,
    TARGET_COLUMNS,
)


# =============================================================================
# Configuration
# =============================================================================

FEATURE_SET_NAMES: tuple[str, ...] = ("chem", "image", "all")

# Columns that describe the morphology pipeline rather than the microstructure.
# morph_source matches the "morph_" prefix; morphology_valid does not.
MORPHOLOGY_FLAG_COLUMNS: tuple[str, ...] = ("morphology_valid", "morph_source")

FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    set(TARGET_COLUMNS) | set(DEFAULT_ID_COLUMNS) | {"sample_id"}
)

METRIC_COLUMNS: tuple[str, ...] = (
    "R2",
    "MAE",
    "RMSE",
    "Pearson_r",
    "pct_within_tol",
    "n_features",
    "best_inner_R2",
)


# =============================================================================
# Feature sets
# =============================================================================

def split_feature_sets(
    columns: Iterable[str],
    chemistry_columns: Iterable[str] = CHEMISTRY_COLUMNS,
    image_prefixes: Iterable[str] = IMAGE_DESCRIPTOR_PREFIXES,
    morphology_flag_columns: Iterable[str] = MORPHOLOGY_FLAG_COLUMNS,
    morphology_flags_in_image: bool = False,
    extra_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Split retained columns into {"chem", "image", "all"}.

    chem:  columns listed in chemistry_columns.
    image: columns starting with image_prefixes.
    all:   every retained column, in the given order.

    morphology_flag_columns (default: morphology_valid, morph_source) are
    excluded from "image" and appear only in "all", unless
    morphology_flags_in_image=True. Any further column that is neither
    chemistry nor image must be declared in extra_groups (group name ->
    columns); such columns also appear only in "all".
    """
    columns = list(columns)
    prefixes = tuple(image_prefixes)
    chem_lookup = set(chemistry_columns)
    flag_lookup = set(morphology_flag_columns)
    extra_groups = dict(extra_groups or {})
    extra_lookup = {c for group in extra_groups.values() for c in group}

    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if duplicated:
        raise ValueError(f"Duplicate feature columns: {duplicated}")

    chem = [c for c in columns if c in chem_lookup]

    if morphology_flags_in_image:
        image = [c for c in columns if c.startswith(prefixes) or c in flag_lookup]
    else:
        image = [
            c for c in columns
            if c.startswith(prefixes) and c not in flag_lookup
        ]

    overlap = sorted(set(chem) & set(image))
    if overlap:
        raise ValueError(f"Columns assigned to both chem and image: {overlap}")

    extra_overlap = sorted(extra_lookup & (set(chem) | set(image)))
    if extra_overlap:
        raise ValueError(
            "extra_groups contains columns already assigned to chem or "
            f"image: {extra_overlap}"
        )

    assigned = set(chem) | set(image) | extra_lookup | flag_lookup
    unassigned = [c for c in columns if c not in assigned]
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} column(s) belong to no feature group: "
            f"{unassigned}. Declare them via extra_groups (kept only in "
            "'all') or fix the schema."
        )

    sets = {"chem": chem, "image": image, "all": columns}

    empty = [name for name, cols in sets.items() if not cols]
    if empty:
        raise ValueError(f"Empty feature set(s): {empty}")

    return sets


def _assert_no_leakage(
    columns: Iterable[str],
    target_col: str,
    context: str,
) -> None:
    """Raise if a target or identifier column is among the features."""
    hits = sorted((FORBIDDEN_FEATURE_COLUMNS | {target_col}) & set(columns))
    if hits:
        raise RuntimeError(
            f"Target/identifier column(s) in features ({context}): {hits}"
        )


# =============================================================================
# Persistence
# =============================================================================

def save_fold_datasets(folds: list[dict[str, Any]], path: str | Path) -> None:
    """Pickle the stage 1 fold datasets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(folds, file)


def load_fold_datasets(path: str | Path) -> list[dict[str, Any]]:
    """Load stage 1 fold datasets written by save_fold_datasets()."""
    with open(path, "rb") as file:
        return pickle.load(file)


# =============================================================================
# Stage 1: fold datasets
# =============================================================================

def build_fold_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    groups: bool = True,
    random_state: int = 2026,
    n_splits_outer: int = 5,
    n_splits_inner: int = 3,
    cleaning_kwargs: dict[str, Any] | None = None,
    cache_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Clean every outer fold once, exactly as comp_XGB_model does.

    Seeds follow comp_XGB_model (inner splits use random_state + fold), so
    the "all" feature set reproduces the main nested-CV result.

    Per fold the dict holds: fold, target_col, X_train, y_train, X_valid,
    y_valid (cleaned, unscaled, full feature set), cleaned_inner_folds,
    valid_index (retained rows), outer_test_index (all raw test rows),
    y_test_raw, feature_columns, config.

    If cache_path exists, the cached list is returned after checking that it
    was built with identical settings; otherwise it is built and saved.
    """
    cleaning_kwargs = dict(cleaning_kwargs or CLEANING_KWARGS)

    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)

    config = {
        "target_col": target_col,
        "groups": groups if isinstance(groups, bool) else "custom",
        "random_state": random_state,
        "n_splits_outer": n_splits_outer,
        "n_splits_inner": n_splits_inner,
        "cleaning_kwargs": cleaning_kwargs,
        "n_rows_raw": len(X),
        "n_columns_raw": X.shape[1],
    }

    if cache_path is not None and Path(cache_path).exists():
        folds = load_fold_datasets(cache_path)
        if len(folds) != n_splits_outer or any(
            fold["config"] != config for fold in folds
        ):
            raise ValueError(
                f"Cache {cache_path} was built with different settings; "
                "delete it or choose another path."
            )
        return folds

    dataframe_full = combine_X_y(X=X, y=y, target_col=target_col)

    outer_splits = get_cv_splits(
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits_outer,
        random_state=random_state,
    )

    folds: list[dict[str, Any]] = []

    for fold_index, (outer_train_index, outer_test_index) in enumerate(
        outer_splits
    ):
        print(
            f"  Cleaning outer fold {fold_index + 1}/{n_splits_outer}: "
            f"{len(outer_train_index)} raw train, "
            f"{len(outer_test_index)} raw test rows"
        )

        dataframe_outer_train = dataframe_full.iloc[outer_train_index].copy()
        dataframe_outer_test = dataframe_full.iloc[outer_test_index].copy()

        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            _,
        ) = prepare_model_fold(
            dataframe_train=dataframe_outer_train,
            dataframe_validation=dataframe_outer_test,
            target_col=target_col,
            cleaning_kwargs=cleaning_kwargs,
            verbose=False,
        )

        cleaned_inner_folds = precompute_cleaned_inner_folds(
            dataframe_train=dataframe_outer_train,
            target_col=target_col,
            groups=groups,
            n_splits_inner=n_splits_inner,
            random_state=random_state + fold_index,
            cleaning_kwargs=cleaning_kwargs,
        )

        feature_columns = X_train.columns.tolist()
        _assert_no_leakage(feature_columns, target_col, f"fold {fold_index}")

        if not X_valid.index.isin(dataframe_outer_test.index).all():
            raise RuntimeError(
                f"Fold {fold_index}: validation rows outside the outer test set."
            )

        folds.append({
            "fold": fold_index,
            "target_col": target_col,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "cleaned_inner_folds": cleaned_inner_folds,
            "valid_index": X_valid.index.copy(),
            "outer_test_index": dataframe_outer_test.index.copy(),
            "y_test_raw": dataframe_outer_test[target_col].copy(),
            "feature_columns": feature_columns,
            "config": config,
        })

    if cache_path is not None:
        save_fold_datasets(folds, cache_path)

    return folds


# =============================================================================
# Stage 2: ablation
# =============================================================================

def _slice_inner_folds(
    cleaned_inner_folds: list[dict[str, Any]],
    feature_set: str,
    target_col: str,
    split_kwargs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Restrict every cleaned inner fold to one feature set.

    The column set is derived from each inner fold's own fitted columns,
    because inner-fold cleaning can retain a different schema than the
    outer fold. Rows and inner partitions are untouched.
    """
    sliced = []

    for inner in cleaned_inner_folds:
        try:
            columns = split_feature_sets(
                inner["X_train"].columns, **split_kwargs
            )[feature_set]
        except ValueError as error:
            raise ValueError(
                f"Inner fold {inner['fold']}, feature set "
                f"'{feature_set}': {error}"
            ) from error

        _assert_no_leakage(columns, target_col, f"inner fold {inner['fold']}")

        sliced.append({
            "fold": inner["fold"],
            "X_train": inner["X_train"][columns],
            "y_train": inner["y_train"],
            "X_valid": inner["X_valid"][columns],
            "y_valid": inner["y_valid"],
        })

    return sliced


def run_ablation(
    folds: list[dict[str, Any]],
    feature_sets: Sequence[str] = FEATURE_SET_NAMES,
    n_trials: int = 100,
    random_state: int = 2026,
    tol_rel: float = 0.15,
    split_kwargs: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tune, refit and evaluate one XGBoost model per fold and feature set.

    Returns (metrics_df, oof_df, best_params_df). oof_df is indexed by the
    raw row index and contains NaN predictions for validation rows removed
    by the fold-local cleaning (identically for all feature sets).
    split_kwargs is forwarded to split_feature_sets (e.g. extra_groups).
    """
    unknown = sorted(set(feature_sets) - set(FEATURE_SET_NAMES))
    if unknown:
        raise ValueError(f"Unknown feature set(s): {unknown}")

    split_kwargs = dict(split_kwargs or {})

    metrics_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    valid_indices: dict[int, pd.Index] = {}
    run_settings: dict[int, dict[str, tuple]] = {}

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        for fold in folds:
            fold_index = fold["fold"]
            target_col = fold["target_col"]
            n_folds = len(folds)

            if fold["config"]["random_state"] != random_state:
                raise ValueError(
                    "random_state differs from the one used to build the "
                    f"fold datasets ({fold['config']['random_state']})."
                )

            seed = random_state + fold_index
            column_sets = split_feature_sets(
                fold["feature_columns"], **split_kwargs
            )

            for name in feature_sets:
                columns = column_sets[name]
                _assert_no_leakage(columns, target_col, f"fold {fold_index}")

                X_train = fold["X_train"][columns]
                X_valid = fold["X_valid"][columns]
                y_train = fold["y_train"]
                y_valid = fold["y_valid"]

                inner_folds = _slice_inner_folds(
                    fold["cleaned_inner_folds"], name, target_col, split_kwargs
                )

                for original, restricted in zip(
                    fold["cleaned_inner_folds"], inner_folds
                ):
                    if not (
                        original["X_train"].index.equals(
                            restricted["X_train"].index
                        )
                        and original["X_valid"].index.equals(
                            restricted["X_valid"].index
                        )
                    ):
                        raise RuntimeError("Inner-fold rows changed by slicing.")

                # Row identity across feature sets
                if fold_index in valid_indices:
                    if not valid_indices[fold_index].equals(X_valid.index):
                        raise RuntimeError(
                            f"Fold {fold_index}: validation index differs "
                            "between feature sets."
                        )
                else:
                    valid_indices[fold_index] = X_valid.index

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=seed),
                )
                study.optimize(
                    make_xgb_optuna_objective(
                        cleaned_inner_folds=inner_folds,
                        random_state=seed,
                    ),
                    n_trials=n_trials,
                    show_progress_bar=False,
                )

                run_settings.setdefault(fold_index, {})[name] = (
                    seed,
                    seed,
                    len(study.trials),
                    tuple(len(f["X_valid"]) for f in inner_folds),
                )

                model = build_xgb_pipeline(
                    params=study.best_params,
                    random_state=seed,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)

                metrics = compute_regression_metrics(
                    y_true=y_valid,
                    y_pred=y_pred,
                    tol_rel=tol_rel,
                )

                metrics_rows.append({
                    "fold": fold_index,
                    "feature_set": name,
                    "n_features": len(columns),
                    "R2": metrics["R2"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "Pearson_r": metrics["Pearsons_r"],
                    "pct_within_tol": metrics["pct_within_tol"],
                    "best_inner_R2": study.best_value,
                })

                params_rows.append({
                    "fold": fold_index,
                    "feature_set": name,
                    **study.best_params,
                })

                oof = pd.DataFrame({"y_true": fold["y_test_raw"]})
                oof["y_pred"] = np.nan
                oof.loc[X_valid.index, "y_pred"] = y_pred
                oof["fold"] = pd.array([pd.NA] * len(oof), dtype="Int64")
                oof.loc[X_valid.index, "fold"] = fold_index
                oof["feature_set"] = name
                oof_frames.append(oof)

                print(
                    f"  Fold {fold_index + 1}/{n_folds} | {name:<5} | "
                    f"n_features={len(columns):>4} | "
                    f"inner R2={study.best_value:.3f} | "
                    f"outer R2={metrics['R2']:.3f}"
                )
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

    # Seeds, trial counts and inner partition sizes identical across sets
    for fold_index, settings in run_settings.items():
        if len(set(settings.values())) != 1:
            raise RuntimeError(
                f"Fold {fold_index}: seeds, trial counts or inner "
                f"partitions differ between feature sets: {settings}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    best_params_df = pd.DataFrame(params_rows)
    oof_df = pd.concat(oof_frames).sort_index(kind="stable")
    oof_df = oof_df[["y_true", "y_pred", "fold", "feature_set"]]

    # Identical OOF missing-row pattern across feature sets
    masks = {
        name: oof_df.loc[oof_df["feature_set"] == name, "y_pred"]
        .isna()
        .sort_index()
        for name in feature_sets
    }
    reference = masks[feature_sets[0]]
    for name, mask in masks.items():
        if not mask.equals(reference):
            raise RuntimeError(
                f"OOF missing-row pattern of '{name}' differs from "
                f"'{feature_sets[0]}'."
            )

    return metrics_df, oof_df, best_params_df


# =============================================================================
# Reporting
# =============================================================================

def summarize_ablation(
    metrics_df: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]] = (("all", "chem"), ("all", "image")),
    paired_metrics: Sequence[str] = ("R2", "MAE"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarise the ablation over outer folds (no significance test).

    Returns
    -------
    summary:        mean and std over folds per feature_set.
    paired:         per-fold differences "a - b" for paired_metrics. For R2 a
                    positive value favours a; for MAE a negative value does.
    paired_summary: mean and std over folds of those differences.
    """
    summary = (
        metrics_df.groupby("feature_set", sort=False)[list(METRIC_COLUMNS)]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

    wide = {
        metric: metrics_df.pivot(
            index="fold", columns="feature_set", values=metric
        )
        for metric in paired_metrics
    }

    frames = []
    for first, second in comparisons:
        available = wide[paired_metrics[0]].columns
        if first not in available or second not in available:
            continue

        frame = pd.DataFrame({
            f"delta_{metric}": wide[metric][first] - wide[metric][second]
            for metric in paired_metrics
        })
        frame.insert(0, "comparison", f"{first} - {second}")
        frames.append(frame.reset_index())

    delta_columns = [f"delta_{metric}" for metric in paired_metrics]

    if not frames:
        empty = pd.DataFrame(columns=["fold", "comparison", *delta_columns])
        return summary, empty, empty

    paired = pd.concat(frames, ignore_index=True)

    paired_summary = (
        paired.groupby("comparison", sort=False)[delta_columns]
        .agg(["mean", "std"])
    )
    paired_summary.columns = [
        f"{metric}_{stat}" for metric, stat in paired_summary.columns
    ]

    return summary, paired, paired_summary


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    import os
    import warnings
    from datetime import datetime

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    
    SPLIT_KWARGS = {"image_prefixes": (*IMAGE_DESCRIPTOR_PREFIXES, "is_bse")}

    results_root = Path(module_path) / "results" / "ML_ablation"
    cache_directory = results_root / "cache"
    output_directory = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    xy_directory = Path(module_path) / "data" / "Xy"

    N_TRIALS = 100
    RANDOM_STATE = 2026
    GROUPS = ["ss_sem", "ss_om"]
    TARGETS = {
        "uts": "UTS (MPa)",
        "ys": "YS (MPa)",
        "el": "Total EL (%)",
    }

    for group_name in GROUPS:
        for target_short_name, target_column in TARGETS.items():
            tag = f"{group_name}_{target_short_name}"
            print(f"\n{'=' * 70}\n{tag.upper()}\n{'=' * 70}")

            X = pd.read_csv(xy_directory / f"X_{tag}.csv", low_memory=False)
            y = pd.read_csv(
                xy_directory / f"y_{tag}.csv", low_memory=False
            ).squeeze("columns")

            folds = build_fold_datasets(
                X=X,
                y=y,
                target_col=target_column,
                groups=True,
                random_state=RANDOM_STATE,
                cache_path=cache_directory / f"{tag}_folds.pkl",
            )

            metrics_df, oof_df, best_params_df = run_ablation(
                folds,
                n_trials=N_TRIALS,
                random_state=RANDOM_STATE,
                split_kwargs=SPLIT_KWARGS,
            )

            summary, paired, paired_summary = summarize_ablation(metrics_df)

            print("\nMean/std over outer folds:")
            print(summary.round(3).to_string())
            print("\nPaired differences (mean/std):")
            print(paired_summary.round(3).to_string())

            metrics_df.to_csv(output_directory / f"{tag}_metrics.csv", index=False)
            oof_df.to_csv(output_directory / f"{tag}_oof.csv")
            best_params_df.to_csv(
                output_directory / f"{tag}_best_params.csv", index=False
            )
            summary.to_csv(output_directory / f"{tag}_summary.csv")
            paired.to_csv(output_directory / f"{tag}_paired.csv", index=False)
            paired_summary.to_csv(output_directory / f"{tag}_paired_summary.csv")

    print(f"\nAll results saved to: {output_directory}")



