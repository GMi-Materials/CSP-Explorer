# -*- coding: utf-8 -*-
"""
Nested group-aware XGBoost modelling with fold-local dynamic cleaning.

All data-derived cleaning decisions are fitted exclusively on each training
partition and then applied unchanged to its paired validation/test partition.
"""

author = "Gerfried Millner"
version = "1.1.0"
date = "08.10.2026"
email = "gerfried.millner@ntu.edu.sg"
status = "Deliver"

import os
import sys
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# =============================================================================
# Import fold-local cleaning functions
# =============================================================================

MODULE_PATH = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]

if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

from data_processing.clean_data import prepare_clean_fold


# =============================================================================
# Configuration
# =============================================================================

CLEANING_KWARGS = {
    "missing_feature_threshold": 0.80,
    "handle_inf": "clip",
    "drop_fe": True,
}


# =============================================================================
# Scaling
# =============================================================================

def X_scaling(X_in, ref=None):
    """
    Apply z-score scaling using statistics fitted on ref.

    Binary and constant features are retained without scaling.
    """
    X_in = pd.DataFrame(X_in).copy()

    if ref is None:
        ref = X_in

    ref = pd.DataFrame(ref).copy()

    X_scaled = pd.DataFrame(
        index=X_in.index,
        columns=X_in.columns,
        dtype=float,
    )

    for column in X_in.columns:
        unique_count = ref[column].nunique(dropna=False)

        if unique_count <= 2:
            X_scaled[column] = X_in[column]
            continue

        mean_value = ref[column].mean()
        std_value = ref[column].std()

        if std_value == 0 or np.isnan(std_value):
            X_scaled[column] = X_in[column] - mean_value
        else:
            X_scaled[column] = (X_in[column] - mean_value) / std_value

    return X_scaled


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible z-score transformer.

    Statistics are fitted only on each model-training subset.
    """

    def __init__(self):
        self.reference_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.reference_ = X.copy()
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        X_scaled = X_scaling(X, ref=self.reference_)
        return X_scaled.to_numpy(dtype=float)


# =============================================================================
# Metrics
# =============================================================================

def compute_regression_metrics(y_true, y_pred, tol_rel=0.15):
    """
    Compute regression metrics for one validation/test partition.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearsons_r, _ = pearsonr(y_true, y_pred)
    else:
        pearsons_r = np.nan

    denominator = np.abs(y_true)
    nonzero_mask = denominator > 0

    if np.any(nonzero_mask):
        within_tolerance = (
            np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])
            <= tol_rel * denominator[nonzero_mask]
        )

        pct_within_tol = (
            100.0 * within_tolerance.sum() / within_tolerance.size
        )
    else:
        pct_within_tol = np.nan

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearsons_r": pearsons_r,
        "pct_within_tol": pct_within_tol,
    }


# =============================================================================
# Data and split utilities
# =============================================================================

def combine_X_y(X, y, target_col):
    """
    Reconstruct the raw input table required by prepare_clean_fold().

    The target must not already be present in X. The guard prevents accidental
    duplication or overwrite of a modelling target.
    """
    X = X.copy()
    y = pd.Series(y).copy()

    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} rows, but y has {len(y)} rows."
        )

    if target_col in X.columns:
        raise ValueError(
            f"Target column '{target_col}' is already present in X. "
            "Regenerate source X files after removing their own target."
        )

    dataframe = X.copy()
    dataframe[target_col] = y.to_numpy()

    return dataframe


def get_cv_splits(X, y, groups=True, n_splits=5, random_state=42):
    """
    Construct explicit cross-validation partitions.

    With groups=True, all rows with the same sample_id remain in one fold.
    """
    if isinstance(groups, bool):
        if groups:
            if "sample_id" not in X.columns:
                raise KeyError(
                    "groups=True requires the 'sample_id' column in X."
                )

            group_values = X["sample_id"].to_numpy()
            n_unique_groups = len(np.unique(group_values))

            if n_unique_groups < n_splits:
                raise ValueError(
                    f"Only {n_unique_groups} unique sample_id values are "
                    f"available for n_splits={n_splits}."
                )

            splitter = GroupKFold(n_splits=n_splits)

            return list(
                splitter.split(
                    X=X,
                    y=y,
                    groups=group_values,
                )
            )

        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        return list(splitter.split(X=X, y=y))

    group_values = np.asarray(groups)

    if len(group_values) != len(X):
        raise ValueError(
            "Custom group labels must have the same length as X."
        )

    splitter = GroupKFold(n_splits=n_splits)

    return list(
        splitter.split(
            X=X,
            y=y,
            groups=group_values,
        )
    )


# =============================================================================
# Fold-local cleaning
# =============================================================================

def prepare_model_fold(
    dataframe_train,
    dataframe_validation,
    target_col,
    cleaning_kwargs=None,
    verbose=False,
):
    """
    Fit cleaning on dataframe_train only and apply it to dataframe_validation.

    The retained validation index is deliberately preserved. It identifies the
    rows eligible for OOF prediction.
    """
    if cleaning_kwargs is None:
        cleaning_kwargs = CLEANING_KWARGS

    (
        X_train_clean,
        y_train_clean,
        X_validation_clean,
        y_validation_clean,
        cleaning_pipeline,
    ) = prepare_clean_fold(
        dataframe_train=dataframe_train,
        dataframe_validation=dataframe_validation,
        target_col=target_col,
        verbose=verbose,
        **cleaning_kwargs,
    )

    if "sample_id" in X_train_clean.columns:
        raise RuntimeError(
            "sample_id reached the final training feature matrix."
        )

    if "sample_id" in X_validation_clean.columns:
        raise RuntimeError(
            "sample_id reached the final validation feature matrix."
        )

    if list(X_train_clean.columns) != list(X_validation_clean.columns):
        raise RuntimeError(
            "Training and validation feature schemas differ after cleaning."
        )

    if not X_train_clean.index.equals(y_train_clean.index):
        raise RuntimeError(
            "Cleaned training X and y indices are not aligned."
        )

    if not X_validation_clean.index.equals(y_validation_clean.index):
        raise RuntimeError(
            "Cleaned validation X and y indices are not aligned."
        )

    return (
        X_train_clean,
        y_train_clean,
        X_validation_clean,
        y_validation_clean,
        cleaning_pipeline,
    )


def precompute_cleaned_inner_folds(
    dataframe_train,
    target_col,
    groups=True,
    n_splits_inner=3,
    random_state=42,
    cleaning_kwargs=None,
):
    """
    Precompute deterministic inner-fold cleaning once per outer fold.

    The cleaning procedure depends on inner training data only, not Optuna
    hyperparameters; therefore reuse across Optuna trials is valid.
    """
    if target_col not in dataframe_train.columns:
        raise KeyError(
            f"Target column '{target_col}' is not present in dataframe_train."
        )

    X_for_split = dataframe_train.drop(columns=[target_col])
    y_for_split = dataframe_train[target_col]

    inner_splits = get_cv_splits(
        X=X_for_split,
        y=y_for_split,
        groups=groups,
        n_splits=n_splits_inner,
        random_state=random_state,
    )

    cleaned_inner_folds = []

    for inner_fold_index, (inner_train_index, inner_valid_index) in enumerate(
        inner_splits
    ):
        dataframe_inner_train = (
            dataframe_train.iloc[inner_train_index].copy()
        )

        dataframe_inner_valid = (
            dataframe_train.iloc[inner_valid_index].copy()
        )

        (
            X_inner_train,
            y_inner_train,
            X_inner_valid,
            y_inner_valid,
            _,
        ) = prepare_model_fold(
            dataframe_train=dataframe_inner_train,
            dataframe_validation=dataframe_inner_valid,
            target_col=target_col,
            cleaning_kwargs=cleaning_kwargs,
            verbose=False,
        )

        cleaned_inner_folds.append({
            "fold": inner_fold_index,
            "X_train": X_inner_train,
            "y_train": y_inner_train,
            "X_valid": X_inner_valid,
            "y_valid": y_inner_valid,
        })

    return cleaned_inner_folds


# =============================================================================
# XGBoost model construction
# =============================================================================

def suggest_xgb_parameters(trial):
    """
    Define the focused XGBoost hyperparameter search space.
    """
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            800,
            step=100,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            10.0,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-2,
            10.0,
            log=True,
        ),
        "reg_alpha": trial.suggest_categorical(
            "reg_alpha",
            [0.0, 1e-3, 1e-2],
        ),
    }


def build_xgb_regressor(params, random_state=42):
    """
    Build an XGBRegressor from an Optuna parameter dictionary.
    """
    return XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=0.8,
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def build_xgb_pipeline(params, random_state=42):
    """
    Build the fitted preprocessing/model pipeline for one XGBoost configuration.
    """
    return Pipeline([
        ("scaler", CustomScaler()),
        (
            "xgb",
            build_xgb_regressor(
                params=params,
                random_state=random_state,
            ),
        ),
    ])


# =============================================================================
# Optuna tuning
# =============================================================================

def make_xgb_optuna_objective(
    cleaned_inner_folds,
    random_state=42,
):
    """
    Create an Optuna objective evaluated on pre-cleaned inner CV folds.
    """
    def objective(trial):
        params = suggest_xgb_parameters(trial)
        fold_scores = []

        for fold_data in cleaned_inner_folds:
            model = build_xgb_pipeline(
                params=params,
                random_state=random_state,
            )

            model.fit(
                fold_data["X_train"],
                fold_data["y_train"],
            )

            y_valid_pred = model.predict(
                fold_data["X_valid"]
            )

            fold_scores.append(
                r2_score(
                    fold_data["y_valid"],
                    y_valid_pred,
                )
            )

        return float(np.mean(fold_scores))

    return objective


# =============================================================================
# Nested cross-validation
# =============================================================================

def comp_XGB_model(
    X,
    y,
    target_col,
    groups=True,
    random_state=42,
    n_trials=100,
    n_splits_outer=5,
    n_splits_inner=3,
    cleaning_kwargs=None,
):
    """
    Run nested group-aware CV with fold-local cleaning and Optuna tuning.

    Cleaning is fitted only on each training partition. OOF predictions are
    retained only for validation rows surviving the fitted scale requirement.
    """
    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)

    dataframe_full = combine_X_y(
        X=X,
        y=y,
        target_col=target_col,
    )

    outer_splits = get_cv_splits(
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits_outer,
        random_state=random_state,
    )

    metrics_list = []
    fold_models = []

    y_pred_oof = pd.Series(
        index=dataframe_full.index,
        dtype=float,
    )

    fold_oof = pd.Series(
        index=dataframe_full.index,
        dtype="Int64",
    )

    for fold_index, (outer_train_index, outer_test_index) in enumerate(
        outer_splits
    ):
        print(
            f"  Outer fold {fold_index + 1}/{n_splits_outer}: "
            f"{len(outer_train_index)} raw train rows, "
            f"{len(outer_test_index)} raw test rows"
        )

        # Keep global indices. They are required for OOF assignment
        dataframe_outer_train = (
            dataframe_full.iloc[outer_train_index].copy()
        )

        dataframe_outer_test = (
            dataframe_full.iloc[outer_test_index].copy()
        )

        cleaned_inner_folds = precompute_cleaned_inner_folds(
            dataframe_train=dataframe_outer_train,
            target_col=target_col,
            groups=groups,
            n_splits_inner=n_splits_inner,
            random_state=random_state + fold_index,
            cleaning_kwargs=cleaning_kwargs,
        )

        objective = make_xgb_optuna_objective(
            cleaned_inner_folds=cleaned_inner_folds,
            random_state=random_state + fold_index,
        )

        sampler = optuna.samplers.TPESampler(
            seed=random_state + fold_index,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        (
            X_outer_train_clean,
            y_outer_train_clean,
            X_outer_test_clean,
            y_outer_test_clean,
            cleaning_pipeline,
        ) = prepare_model_fold(
            dataframe_train=dataframe_outer_train,
            dataframe_validation=dataframe_outer_test,
            target_col=target_col,
            cleaning_kwargs=cleaning_kwargs,
            verbose=False,
        )

        model = build_xgb_pipeline(
            params=study.best_params,
            random_state=random_state + fold_index,
        )

        model.fit(
            X_outer_train_clean,
            y_outer_train_clean,
        )

        y_pred = model.predict(X_outer_test_clean)

        fold_metrics = compute_regression_metrics(
            y_true=y_outer_test_clean,
            y_pred=y_pred,
            tol_rel=0.15,
        )

        fold_metrics["fold"] = fold_index
        fold_metrics["n_train_raw"] = len(dataframe_outer_train)
        fold_metrics["n_train_clean"] = len(X_outer_train_clean)
        fold_metrics["n_test_raw"] = len(dataframe_outer_test)
        fold_metrics["n_test_clean"] = len(X_outer_test_clean)
        fold_metrics["n_features"] = X_outer_train_clean.shape[1]

        metrics_list.append(fold_metrics)

        fold_models.append({
            "fold": fold_index,
            "model": model,
            "cleaning_pipeline": cleaning_pipeline,
            "feature_columns": X_outer_train_clean.columns.tolist(),
            "best_params": study.best_params,
            "best_inner_R2": study.best_value,
            "outer_metrics": fold_metrics,
        })

        # Use retained global validation indices, not outer_test_index. This
        # correctly leaves ineligible no-scale rows without OOF predictions.
        y_pred_oof.loc[y_outer_test_clean.index] = y_pred
        fold_oof.loc[y_outer_test_clean.index] = fold_index

    metrics_dataframe = pd.DataFrame(metrics_list)

    oof_dataframe = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred_oof,
        "fold": fold_oof,
    })

    oof_dataframe["evaluated"] = oof_dataframe["y_pred"].notna()

    expected_missing = int(
        (
            metrics_dataframe["n_test_raw"]
            - metrics_dataframe["n_test_clean"]
        ).sum()
    )

    observed_missing = int(
        oof_dataframe["y_pred"].isna().sum()
    )

    if observed_missing != expected_missing:
        raise RuntimeError(
            "Unexpected number of missing OOF predictions: "
            f"observed={observed_missing}, expected={expected_missing}."
        )

    return metrics_dataframe, fold_models, oof_dataframe


# =============================================================================
# Final model fitting
# =============================================================================

def train_final_xgb_full(
    X,
    y,
    target_col,
    groups=True,
    random_state=42,
    n_splits_inner=3,
    n_trials=100,
    cleaning_kwargs=None,
):
    """
    Tune XGBoost using group-aware inner CV, then fit on all eligible data.

    The returned bundle contains the fitted cleaning pipeline and fitted model,
    both of which are necessary to predict from a future raw source table.
    """
    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)

    dataframe_full = combine_X_y(
        X=X,
        y=y,
        target_col=target_col,
    )

    cleaned_inner_folds = precompute_cleaned_inner_folds(
        dataframe_train=dataframe_full,
        target_col=target_col,
        groups=groups,
        n_splits_inner=n_splits_inner,
        random_state=random_state,
        cleaning_kwargs=cleaning_kwargs,
    )

    objective = make_xgb_optuna_objective(
        cleaned_inner_folds=cleaned_inner_folds,
        random_state=random_state,
    )

    sampler = optuna.samplers.TPESampler(
        seed=random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    (
        X_full_clean,
        y_full_clean,
        _,
        _,
        cleaning_pipeline,
    ) = prepare_clean_fold(
        dataframe_train=dataframe_full,
        dataframe_validation=dataframe_full.copy(),
        target_col=target_col,
        verbose=False,
        **(cleaning_kwargs or CLEANING_KWARGS),
    )

    final_model = build_xgb_pipeline(
        params=study.best_params,
        random_state=random_state,
    )

    final_model.fit(
        X_full_clean,
        y_full_clean,
    )

    fitted_bundle = {
        "model": final_model,
        "cleaning_pipeline": cleaning_pipeline,
        "feature_columns": X_full_clean.columns.tolist(),
        "target_column": target_col,
        "best_params": study.best_params,
        "best_inner_R2": study.best_value,
        "n_train_raw": len(dataframe_full),
        "n_train_clean": len(X_full_clean),
        "n_features": X_full_clean.shape[1],
    }

    return fitted_bundle, study


# =============================================================================
# User-facing wrapper
# =============================================================================

def run_paper_models(
    X,
    y,
    target_col,
    groups=True,
    random_state=2026,
    n_trials=100,
    cleaning_kwargs=None,
):
    """
    Run nested CV for one material cohort and one mechanical-property target.
    """
    metrics_XGB, fold_models_XGB, oof_XGB = comp_XGB_model(
        X=X,
        y=y,
        target_col=target_col,
        groups=groups,
        random_state=random_state,
        n_trials=n_trials,
        n_splits_outer=5,
        n_splits_inner=3,
        cleaning_kwargs=cleaning_kwargs,
    )

    summary = pd.DataFrame([{
        "model": "Model_XGB",
        "mean_R2": metrics_XGB["R2"].mean(),
        "std_R2": metrics_XGB["R2"].std(),
        "mean_MAE": metrics_XGB["MAE"].mean(),
        "std_MAE": metrics_XGB["MAE"].std(),
        "mean_RMSE": metrics_XGB["RMSE"].mean(),
        "std_RMSE": metrics_XGB["RMSE"].std(),
        "mean_Pearsons_r": metrics_XGB["Pearsons_r"].mean(),
        "std_Pearsons_r": metrics_XGB["Pearsons_r"].std(),
        "mean_pct_within_tol": metrics_XGB["pct_within_tol"].mean(),
        "std_pct_within_tol": metrics_XGB["pct_within_tol"].std(),
        "mean_n_features": metrics_XGB["n_features"].mean(),
        "std_n_features": metrics_XGB["n_features"].std(),
        "mean_n_train_clean": metrics_XGB["n_train_clean"].mean(),
        "mean_n_test_clean": metrics_XGB["n_test_clean"].mean(),
        "n_oof_evaluated": int(oof_XGB["evaluated"].sum()),
        "n_oof_ineligible_no_scale": int((~oof_XGB["evaluated"]).sum()),
    }])

    raw = {
        "Model_XGB": {
            "fold_metrics": metrics_XGB,
            "fold_models": fold_models_XGB,
            "oof_predictions": oof_XGB,
        }
    }

    return summary, raw


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING,
    )

    module_path = str.rsplit(
        os.path.abspath(""),
        os.path.sep,
        1,
    )[0]

    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    print("Module path:", module_path)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = (
        Path(module_path)
        / "results"
        / "ML"
        / timestamp
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    xy_directory = os.path.join(
        module_path,
        "data",
        "Xy",
    )

    N_TRIALS = 100

    GROUPS = [
        "ss_sem",
        "ss_om",
    ]

    TARGETS = {
        "uts": "UTS (MPa)",
        "ys": "YS (MPa)",
        "el": "Total EL (%)",
    }

    for group_name in GROUPS:
        print(f"\n{'=' * 70}")
        print(f"GROUP: {group_name.upper()}")
        print(f"{'=' * 70}")

        for target_short_name, target_column in TARGETS.items():
            x_path = os.path.join(
                xy_directory,
                f"X_{group_name}_{target_short_name}.csv",
            )

            y_path = os.path.join(
                xy_directory,
                f"y_{group_name}_{target_short_name}.csv",
            )

            X = pd.read_csv(
                x_path,
                low_memory=False,
            )

            y = pd.read_csv(
                y_path,
                low_memory=False,
            ).squeeze("columns")

            print(f"\n--- {target_short_name.upper()} ---")
            print(
                f"Raw source data: "
                f"{X.shape[0]} rows x {X.shape[1]} columns"
            )

            summary, raw = run_paper_models(
                X=X,
                y=y,
                target_col=target_column,
                groups=True,
                random_state=2026,
                n_trials=N_TRIALS,
                cleaning_kwargs=CLEANING_KWARGS,
            )

            print("\nNested-CV summary:")
            print(summary.to_string(index=False))

            summary_path = (
                output_directory
                / f"{group_name}_{target_short_name}_summary.csv"
            )

            oof_path = (
                output_directory
                / f"{group_name}_{target_short_name}_oof.csv"
            )

            raw_path = (
                output_directory
                / f"{group_name}_{target_short_name}_nested_cv.pkl"
            )

            summary.to_csv(
                summary_path,
                index=False,
            )

            raw["Model_XGB"]["oof_predictions"].to_csv(
                oof_path,
                index=False,
            )

            with open(raw_path, "wb") as file:
                pickle.dump(raw, file)

    print(f"\nAll results saved to: {output_directory}")