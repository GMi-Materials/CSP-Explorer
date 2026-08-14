# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:09:36 2026

@author: gerfried.millner
"""

author = "Gerfried Millner"
version = "1.0.0"
date = "31.03.2026"
email = "gerfried.millner@ntu.edu.sg"
status = "Deliver"

import numpy as np
import pandas as pd
import optuna  
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

def X_scaling(X_in, ref=None):
    """
    Scale input data using z-score normalization.
    Binary or constant features are kept unscaled.
    """
    X_scal = pd.DataFrame(index=X_in.index)

    if ref is None:
        ref = X_in

    for col in X_in.columns:
        if ref[col].nunique() <= 2:
            X_scal[col] = X_in[col]
        else:
            mean = ref[col].mean()
            std = ref[col].std()
            if std == 0 or np.isnan(std):
                X_scal[col] = X_in[col] - mean
            else:
                X_scal[col] = (X_in[col] - mean) / std
    return X_scal

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around X_scaling.
    - Learns per-feature mean/std on X in fit().
    - Applies same stats to new data in transform().
    - Leaves binary/constant features unscaled.
    """

    def __init__(self):
        self.ref_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.ref_ = X.copy()
        self.columns_ = X.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        X_scaled = X_scaling(X, ref=self.ref_)
        return X_scaled.values

def compute_regression_metrics(y_true, y_pred, tol_rel=0.15):
    """
    Compute R2, MAE, RMSE, Pearson r, and percentage within relative tolerance.
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
        pct_within = 100.0 * within.sum() / within.size
    else:
        pct_within = np.nan

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearsons_r": r,
        "pct_within_tol": pct_within,
    }

########################################
################# XGB ##################
########################################

def make_xgb_full_data_objective(
    X,
    Y,
    groups: bool = False,
    rands: int = 42,
    n_splits_inner: int = 5,
):
    """
    Build an Optuna objective for tuning XGBoost on the full dataset.
    Uses inner CV only (no outer loop here), respecting group structure.
    The design matrix is:
        - X.drop(columns=["sample_id"]) if groups is True
        - X as-is otherwise
    """

    # ----- inner CV splitter -----
    if isinstance(groups, bool):
        if groups is True:
            if "sample_id" not in X.columns:
                raise ValueError("groups=True requires 'sample_id' column in X.")
            group_id = pd.factorize(X["sample_id"])[0]
            X_use = X.drop(columns=["sample_id"])
            kf = GroupKFold(n_splits=n_splits_inner)
            cv_splits = list(kf.split(X_use, Y, group_id))
        else:
            X_use = X
            kf = KFold(
                n_splits=n_splits_inner,
                shuffle=True,
                random_state=rands,
            )
            cv_splits = list(kf.split(X_use))
    else:
        # groups is array-like with custom group labels
        X_use = X
        kf = GroupKFold(n_splits=n_splits_inner)
        cv_splits = list(kf.split(X_use, Y, groups))

    def objective(trial):
        # ----- search space -----
        # Keep search modest to avoid overfitting for small n
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=100)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

        xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=rands,
            n_jobs=-1,
            tree_method="hist",
        )

        pipe = Pipeline([
            ("scaler", CustomScaler()),
            ("xgb", xgb),
        ])

        scores = cross_val_score(
            pipe,
            X_use.values,
            np.ravel(Y.values),
            cv=cv_splits,
            scoring="r2",
            n_jobs=-1,
        )
        return scores.mean()

    return objective

def train_final_xgb_full(
    X,
    Y,
    groups: bool = False,
    rands: int = 42,
    n_splits_inner: int = 5,
    n_trials: int = 100,
):
    """
    Tune XGBoost hyperparameters with Optuna inner CV and refit on all data.
    XGBRegressor is trained on the full feature set (all descriptors),
    dropping only 'sample_id' if groups=True.

    Returns
    -------
    final_model : Pipeline
        Fitted Pipeline(CustomScaler, XGBRegressor).
    study : optuna.study.Study
        Optuna study with history and best_params.
    """
    objective = make_xgb_full_data_objective(
        X,
        Y,
        groups=groups,
        rands=rands,
        n_splits_inner=n_splits_inner,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    # Prepare full design matrix for final fit
    if isinstance(groups, bool) and groups is True:
        if "sample_id" not in X.columns:
            raise ValueError("groups=True requires 'sample_id' column in X.")
        X_use = X.drop(columns=["sample_id"])
    else:
        X_use = X

    xgb = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        reg_lambda=best_params["reg_lambda"],
        reg_alpha=best_params["reg_alpha"],
        objective="reg:squarederror",
        random_state=rands,
        n_jobs=-1,
        tree_method="hist",
    )

    final_model = Pipeline([
        ("scaler", CustomScaler()),
        ("xgb", xgb),
    ])
    final_model.fit(X_use, Y)

    return final_model, study

def make_xgb_optuna_objective(
    X,
    Y,
    groups: bool = False,
    rands: int = 42,
    n_splits_inner: int = 5,
):
    """
    XGB + Optuna objective with a focused search space for N≈220.
    """

    # ----- inner CV splitter -----
    if isinstance(groups, bool):
        if groups is True:
            if "sample_id" not in X.columns:
                raise ValueError("groups=True requires 'sample_id' column in X.")
            group_id = pd.factorize(X["sample_id"])[0]
            X_use = X.drop(columns=["sample_id"])
            kf = GroupKFold(n_splits=n_splits_inner)
            cv_splits = list(kf.split(X_use, Y, group_id))
        else:
            X_use = X
            kf = KFold(
                n_splits=n_splits_inner,
                shuffle=True,
                random_state=rands,
            )
            cv_splits = list(kf.split(X_use))
    else:
        X_use = X
        kf = GroupKFold(n_splits=n_splits_inner)
        cv_splits = list(kf.split(X_use, Y, groups))

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 300, 800, step=100)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)

        # Keep one main sampling knob, fix the other
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = 0.8

        # Keep tree-structure regularization
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0)

        # Prefer L2; keep L1 simple
        reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True)
        reg_alpha = trial.suggest_categorical("reg_alpha", [0.0, 1e-3, 1e-2])

        xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=rands,
            n_jobs=-1,
            tree_method="hist",
        )

        pipe = Pipeline([
            ("scaler", CustomScaler()),
            ("xgb", xgb),
        ])

        scores = cross_val_score(
            pipe,
            X_use.values,
            np.ravel(Y.values),
            cv=cv_splits,
            scoring="r2",
            n_jobs=-1,
        )
        return scores.mean()

    return objective


def comp_XGB_model(
    X,
    Y,
    groups=False,
    rands=42,
    n_trials=50,
    n_splits_outer=5,
    n_splits_inner=5
):
    """
    Nested CV with Optuna-tuned XGBoost.
    Returns
    -------
    df : DataFrame
        R2, MAE, RMSE, Pearson r, pct_within_tol per outer fold.
    list_models : list
        Fitted XGB pipelines per outer fold.
    oof : DataFrame
        Out-of-fold predictions aligned with the index of X/Y.
        Columns: 'y_true', 'y_pred', 'fold'.
    """
    # ----- outer CV splitter -----
    if isinstance(groups, bool):
        if groups is True:
            if "sample_id" not in X.columns:
                raise ValueError("groups=True requires 'sample_id' column in X.")
            group_id_full = pd.factorize(X["sample_id"])[0]
            X_use_full = X.drop(columns=["sample_id"])
            kf_outer = GroupKFold(n_splits=n_splits_outer)
            outer_splits = kf_outer.split(X_use_full, Y, group_id_full)
        else:
            X_use_full = X
            kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=rands)
            outer_splits = kf_outer.split(X_use_full)
    else:
        X_use_full = X
        kf_outer = GroupKFold(n_splits=n_splits_outer)
        outer_splits = kf_outer.split(X_use_full, Y, groups)

    metrics_list = []
    list_models = []

    # OOF containers (aligned to original index)
    y_pred_oof = pd.Series(index=Y.index, dtype=float)
    fold_oof = pd.Series(index=Y.index, dtype=int)

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        X_train = X.iloc[outer_train_idx]
        Y_train = Y.iloc[outer_train_idx]
        X_test = X.iloc[outer_test_idx]
        Y_test = Y.iloc[outer_test_idx]

        # Inner Optuna on outer-train data
        objective = make_xgb_optuna_objective(
            X_train,
            Y_train,
            groups=groups,
            rands=rands,
            n_splits_inner=n_splits_inner,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params

        xgb = XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=0.8,  # fixed, consistent with make_xgb_optuna_objective
            min_child_weight=best_params["min_child_weight"],
            reg_lambda=best_params["reg_lambda"],
            reg_alpha=best_params["reg_alpha"],
            objective="reg:squarederror",
            random_state=rands,
            n_jobs=-1,
            tree_method="hist",
        )

        if isinstance(groups, bool) and groups is True:
            if "sample_id" not in X_train.columns:
                raise ValueError("groups=True requires 'sample_id' column in X.")
            X_train_use = X_train.drop(columns=["sample_id"])
            X_test_use = X_test.drop(columns=["sample_id"])
        else:
            X_train_use = X_train
            X_test_use = X_test

        pipe = Pipeline([
            ("scaler", CustomScaler()),
            ("xgb", xgb),
        ])

        model = pipe.fit(X_train_use, Y_train)
        y_pred = model.predict(X_test_use)

        # store metrics per fold
        m = compute_regression_metrics(Y_test, y_pred, tol_rel=0.15)
        metrics_list.append(m)
        list_models.append(model)

        # fill OOF containers
        y_pred_oof.iloc[outer_test_idx] = y_pred
        fold_oof.iloc[outer_test_idx] = fold_idx

    df = pd.DataFrame(metrics_list)

    oof = pd.DataFrame({
        "y_true": Y,
        "y_pred": y_pred_oof,
        "fold": fold_oof,
    })

    return df, list_models, oof

############################################
################# Wrapper ##################
############################################

def run_paper_models(
    X,
    Y,
    groups: bool = True,
    rands: int = 42,
    n_trials: int=100,
):
    """
    Run the XGB model with nested CV and Optuna tuning.

    Parameters
    ----------
    X : pd.DataFrame
    Y : pd.Series or array-like
    groups : bool, default True
        If True, use group-aware CV based on X['sample_id'].
    rands : int, default 42

    Returns
    -------
    summary : pd.DataFrame
        Single-row summary of metrics for XGB.
    raw : dict
        {'Model_XGB': (df_per_fold, model_list, oof_df)}.
    """
    results = []
    raw = {}

    df_XGB, models_XGB, oof_XGB = comp_XGB_model(
        X, Y,
        groups=groups,
        rands=rands,
        n_trials=n_trials,
        n_splits_outer=5,
        n_splits_inner=3,
    )
    raw["Model_XGB"] = (df_XGB, models_XGB, oof_XGB)
    results.append({
        "model": "Model_XGB",
        "mean_R2": df_XGB["R2"].mean(),
        "std_R2": df_XGB["R2"].std(),
        "mean_MAE": df_XGB["MAE"].mean(),
        "std_MAE": df_XGB["MAE"].std(),
        "mean_RMSE": df_XGB["RMSE"].mean(),
        "std_RMSE": df_XGB["RMSE"].std(),
        "mean_Pearsons_r": df_XGB["Pearsons_r"].mean(),
        "std_Pearsons_r": df_XGB["Pearsons_r"].std(),
        "mean_pct_within_tol": df_XGB["pct_within_tol"].mean(),
        "std_pct_within_tol": df_XGB["pct_within_tol"].std(),
    })

    summary = pd.DataFrame(results)
    return summary, raw

if __name__ == "__main__":
    import warnings, sys, os, pickle
    from datetime import datetime
    from pathlib import Path
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    sys.path.append(module_path)
    print("Module path:", module_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(module_path) / "results" / "ML" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    xy_dir = os.path.join(module_path, "data", "Xy")
    N_TRIALS = 100

    GROUPS   = ["ss_sem", "ss_om"] # "dp_sem"
    TARGETS  = ["uts", "ys", "el"]

    for group in GROUPS:
        print(f"\n{'='*60}\nGROUP: {group.upper()}\n{'='*60}")

        tasks = []
        for target in TARGETS:
            X = pd.read_csv(os.path.join(xy_dir, f"X_{group}_{target}.csv"))
            y = pd.read_csv(os.path.join(xy_dir, f"y_{group}_{target}.csv")).squeeze()
            tasks.append((target.upper(), X, y))

        for target_name, X, y in tasks:
            print(f"\n=== {target_name} ({group}) ===")
            summary, raw = run_paper_models(
                X, y,
                groups=True, rands=2026, n_trials=N_TRIALS,
            )
            print(summary)
            summary.to_csv(out_dir / f"{group}_{target_name.lower()}_summary.csv", index=False)
            with open(out_dir / f"{group}_{target_name.lower()}.pkl", "wb") as f:
                pickle.dump(raw, f)

    print(f"\nAll results saved to: {out_dir}")
    

