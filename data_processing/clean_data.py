#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fold-safe data cleaning utilities for CSP Explorer.

Input
-----
This module expects the deterministic, target-independent reduced descriptor
table at:

    data/descriptors_reduced.csv

Output
------
When run directly, this module writes target-specific source tables to:

    data/Xy/X_ss_sem_uts.csv
    data/Xy/y_ss_sem_uts.csv
    data/Xy/X_ss_sem_ys.csv
    data/Xy/y_ss_sem_ys.csv
    data/Xy/X_ss_sem_el.csv
    data/Xy/y_ss_sem_el.csv

    data/Xy/X_ss_om_uts.csv
    data/Xy/y_ss_om_uts.csv
    data/Xy/X_ss_om_ys.csv
    data/Xy/y_ss_om_ys.csv
    data/Xy/X_ss_om_el.csv
    data/Xy/y_ss_om_el.csv

    data/Xy/X_dp_sem_uts.csv
    data/Xy/y_dp_sem_uts.csv
    data/Xy/X_dp_sem_ys.csv
    data/Xy/y_dp_sem_ys.csv
    data/Xy/X_dp_sem_el.csv
    data/Xy/y_dp_sem_el.csv

The exported X files are source tables, not final globally cleaned ML matrices.
They retain sample_id for grouped cross-validation. Data-derived operations
such as rare-support chemistry removal, constant removal, morphology coverage
screening, high-missingness removal, infinity clipping, and
median imputation must be fitted on each training fold only through
fit_cleaning_pipeline() and then applied unchanged to the validation fold with
apply_cleaning_pipeline().

Author: Gerfried Millner
Version: 1.3.0
Date: 12.10.2026
Email: gerfried.millner@ntu.edu.sg
Status: Deliver

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# =============================================================================
# Fixed schema definitions
# =============================================================================

TARGET_COLUMNS = (
    "YS (MPa)",
    "UTS (MPa)",
    "Uniform EL (%)",
    "Total EL (%)",
    "Area Reduction (%)",
    "Absorbed Energy (J/cm²)",
    "Hardness (HV)",
    "n2",
    "Young modulus [GPa]",
)

CHEMISTRY_COLUMNS = (
    "Al",
    "B",
    "C",
    "Ca",
    "Ce",
    "Co",
    "Cr",
    "Cu",
    "Er",
    "La",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Nb",
    "Ni",
    "O",
    "P",
    "S",
    "Si",
    "Sm",
    "Sn",
    "Ti",
    "Ta",
    "V",
    "W",
    "Zn",
    "Zr",
    "Fe",
)

# Never pass identifiers to the model. sample_id is retained in exported source
# tables but removed by _build_feature_matrix() during fold-level cleaning.
DEFAULT_ID_COLUMNS = (
    "obs_id",
    "sample_id",
    "Sample",
    "DOI",
    "ID_name",
    "id_name",
    "image_file",
    "image_file.1",
    "image_name",
    "image_number",
    "Paper_ID",
    "subfolder",
    "Fig",
    "doi",
    "doi_norm",
    "doi_url",
    "pdf_filename",
    "extraction_date",
    "image_name_norm",
)

# These fields identify source, acquisition, or annotation context rather than
# transferable physical predictors. 
DEFAULT_METADATA_COLUMNS = (
    "Material",
    "Category",
    "method",
    "resolution_um",
    "resolution_px",
    "cls_type",
    "cls_confidence",
    "morph_source",
)


# =============================================================================
# Fixed cohorts and exported targets
# =============================================================================

STAINLESS_CATEGORIES = (
    "Ferritic Stainless Steel",
    "Duplex Stainless Steel",
    "Austenitic Stainless Steel",
)

DUAL_PHASE_CATEGORIES = (
    "Conventional DP",
)

GROUP_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "ss_sem": {
        "categories": STAINLESS_CATEGORIES,
        "methods": (
            "SEM: SE",
            "SEM: BSE",
        ),
    },
    "ss_om": {
        "categories": STAINLESS_CATEGORIES,
        "methods": (
            "OM",
        ),
    },
    "dp_sem": {
        "categories": DUAL_PHASE_CATEGORIES,
        "methods": (
            "SEM: SE",
            "SEM: BSE",
        ),
    },
}

TARGET_DEFINITIONS = {
    "uts": "UTS (MPa)",
    "ys": "YS (MPa)",
    "el": "Total EL (%)",
}


IMAGE_DESCRIPTOR_PREFIXES = (
    "morph_",
    "glcm_",
    "lbp_",
    "s2_",
    "conv1_",
    "layer1_",
    "layer2_",
    "layer3_",
    "layer4_",
    "cross_",
    "global_",
    "multiscale_",
)

# Canonical morphology predictors produced by feat_reducer.MorphologyReducer.
# morph_n_particles and morph_source are never NaN by
# construction, so they are not screened here.
CANONICAL_MORPHOLOGY_FEATURES = (
    "morph_area_mean",
    "morph_diameter_mean",
    "morph_aspect_ratio_mean",
    "morph_eccentricity_mean",
)

# =============================================================================
# Cohort filtering and row alignment
# =============================================================================

def filter_by_subcategory(
    dataframe: pd.DataFrame,
    subcategory: str | Iterable[str],
    category_column: str = "Category",
    drop_col: bool = False,
) -> pd.DataFrame:
    """
    Filter rows by one or more Category values.

    This fixed cohort-selection operation is safe before cross-validation.
    """
    if category_column not in dataframe.columns:
        raise KeyError(
            f"Required category column '{category_column}' was not found."
        )

    subcategories = (
        [subcategory]
        if isinstance(subcategory, str)
        else list(subcategory)
    )

    dataframe_filtered = dataframe.loc[
        dataframe[category_column].isin(subcategories)
    ].copy()

    if drop_col:
        dataframe_filtered = dataframe_filtered.drop(
            columns=[category_column],
            errors="ignore",
        )

    return dataframe_filtered


def filter_by_method(
    dataframe: pd.DataFrame,
    microscopy_method: str | Iterable[str],
    method_column: str = "method",
) -> pd.DataFrame:
    """
    Filter rows by one or more microscopy methods.

    This fixed cohort-selection operation is safe before cross-validation.
    """
    if method_column not in dataframe.columns:
        raise KeyError(
            f"Required method column '{method_column}' was not found."
        )

    methods = (
        [microscopy_method]
        if isinstance(microscopy_method, str)
        else list(microscopy_method)
    )

    return dataframe.loc[
        dataframe[method_column].isin(methods)
    ].copy()


def get_aligned_groups(
    dataframe: pd.DataFrame,
    index: pd.Index,
    group_col: str = "sample_id",
) -> pd.Series:
    """
    Return sample_id values aligned exactly to rows retained by the cleaner.
    """
    if group_col not in dataframe.columns:
        raise KeyError(
            f"Group column '{group_col}' was not found in the dataframe."
        )

    return dataframe.loc[index, group_col].copy()


# =============================================================================
# Diagnostics
# =============================================================================

def diagnose_nan_patterns(
    dataframe: pd.DataFrame,
    target_col: str = "UTS (MPa)",
) -> pd.DataFrame:
    """
    Return missing-data diagnostics without modifying the dataframe.

    Do not use this output for global feature selection. Missingness thresholds
    are fitted only from cross-validation training folds.
    """
    feature_groups = {
        "Target": [target_col],
        "Chemistry": list(CHEMISTRY_COLUMNS),
        "Morphology": [
            column
            for column in dataframe.columns
            if column.startswith("morph_")
        ],
        "GLCM": [
            column
            for column in dataframe.columns
            if column.startswith("glcm_")
        ],
        "LBP": [
            column
            for column in dataframe.columns
            if column.startswith("lbp_")
        ],
        "S2": [
            column
            for column in dataframe.columns
            if column.startswith("s2_")
        ],
        "Gram": [
            column
            for column in dataframe.columns
            if column.startswith(
                (
                    "conv1_",
                    "layer1_",
                    "layer2_",
                    "layer3_",
                    "layer4_",
                    "cross_",
                    "global_",
                    "multiscale_",
                )
            )
        ],
    }

    records: list[dict[str, Any]] = []

    for group_name, columns in feature_groups.items():
        available_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        if not available_columns:
            continue

        group_data = dataframe.loc[:, available_columns]

        records.append(
            {
                "group": group_name,
                "n_features": len(available_columns),
                "mean_missing_percent": round(
                    100.0 * group_data.isna().mean().mean(),
                    4,
                ),
                "rows_with_any_missing": int(
                    group_data.isna().any(axis=1).sum()
                ),
                "rows_complete": int(
                    (~group_data.isna().any(axis=1)).sum()
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "mean_missing_percent",
        ascending=False,
        ignore_index=True,
    )


def diagnose_infinity_sources(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-feature counts of positive and negative infinity."""
    numeric_columns = dataframe.select_dtypes(include=np.number).columns
    records: list[dict[str, Any]] = []

    for column in numeric_columns:
        positive_infinity = int((dataframe[column] == np.inf).sum())
        negative_infinity = int((dataframe[column] == -np.inf).sum())
        total_infinity = positive_infinity + negative_infinity

        if total_infinity > 0:
            records.append(
                {
                    "feature": column,
                    "positive_infinity": positive_infinity,
                    "negative_infinity": negative_infinity,
                    "total_infinity": total_infinity,
                    "percent_infinity": round(
                        100.0 * total_infinity / len(dataframe),
                        4,
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature",
                "positive_infinity",
                "negative_infinity",
                "total_infinity",
                "percent_infinity",
            ]
        )

    return pd.DataFrame(records).sort_values(
        "total_infinity",
        ascending=False,
        ignore_index=True,
    )


# =============================================================================
# Internal validation and matrix construction helpers
# =============================================================================

def get_image_descriptor_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric image-derived descriptor columns only."""
    numeric_columns = set(
        dataframe.select_dtypes(include=np.number).columns
    )

    return [
        column
        for column in dataframe.columns
        if column in numeric_columns
        and column.startswith(IMAGE_DESCRIPTOR_PREFIXES)
    ]


def calculate_image_completeness(
    dataframe: pd.DataFrame,
    image_columns: list[str],
) -> pd.Series:
    """Fraction of finite retained image descriptors available per row."""
    if not image_columns:
        raise ValueError(
            "No image descriptor columns are available for completeness filtering."
        )

    image_data = dataframe.reindex(columns=image_columns).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return image_data.notna().mean(axis=1)


def fit_image_missingness_filter(
    feature_dataframe: pd.DataFrame,
    feature_missingness_threshold: float = 0.80,
    row_completeness_threshold: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit feature and row missingness rules on one training fold.

    Feature removal: image descriptors with missing fraction strictly greater
    than feature_missingness_threshold are removed.

    Row removal: rows must have at least row_completeness_threshold of the
    retained image descriptors observed and finite.
    """
    if not 0.0 <= feature_missingness_threshold <= 1.0:
        raise ValueError(
            "feature_missingness_threshold must be between 0 and 1."
        )

    if not 0.0 <= row_completeness_threshold <= 1.0:
        raise ValueError(
            "row_completeness_threshold must be between 0 and 1."
        )

    image_columns = get_image_descriptor_columns(feature_dataframe)

    if not image_columns:
        raise ValueError("No numeric image descriptor columns were found.")

    image_data = feature_dataframe[image_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_missing_fraction = image_data.isna().mean(axis=0)

    dropped_image_features = (
        feature_missing_fraction.loc[
            feature_missing_fraction > feature_missingness_threshold
        ]
        .index
        .tolist()
    )

    retained_image_features = [
        column
        for column in image_columns
        if column not in dropped_image_features
    ]

    row_completeness = calculate_image_completeness(
        feature_dataframe,
        retained_image_features,
    )

    retained_row_mask = (
        row_completeness >= row_completeness_threshold
    )

    diagnostics = {
        "all_image_features": image_columns,
        "feature_missing_fraction": feature_missing_fraction,
        "dropped_image_features": dropped_image_features,
        "retained_image_features": retained_image_features,
        "feature_missingness_threshold": feature_missingness_threshold,
        "row_completeness": row_completeness,
        "row_completeness_threshold": row_completeness_threshold,
        "retained_row_mask": retained_row_mask,
        "dropped_row_indices": feature_dataframe.index[
            ~retained_row_mask
        ].tolist(),
    }

    return feature_dataframe.loc[retained_row_mask].copy(), diagnostics


def apply_image_missingness_filter(
    feature_dataframe: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply training-fitted image descriptor schema and row-completeness rule.

    No missingness threshold is re-estimated on validation/query data.
    """
    retained_image_features = diagnostics["retained_image_features"]
    row_completeness_threshold = diagnostics[
        "row_completeness_threshold"
    ]

    missing_features = [
        column
        for column in retained_image_features
        if column not in feature_dataframe.columns
    ]

    if missing_features:
        raise KeyError(
            "Validation/query data lack retained image descriptors: "
            f"{missing_features}"
        )

    feature_dataframe = feature_dataframe.drop(
        columns=diagnostics["dropped_image_features"],
        errors="ignore",
    )

    row_completeness = calculate_image_completeness(
        feature_dataframe,
        retained_image_features,
    )

    retained_row_mask = (
        row_completeness >= row_completeness_threshold
    )

    return (
        feature_dataframe.loc[retained_row_mask].copy(),
        row_completeness,
    )


def fit_morphology_feature_guardrail(
    feature_dataframe: pd.DataFrame,
    maximum_missing_fraction: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit a morphology-specific coverage rule on one training fold.

    Canonical morphology predictors whose missing (NaN or infinite) fraction
    is strictly greater than maximum_missing_fraction are removed. Rows are
    never removed: dropping images with failed segmentation would change the
    study population and could introduce method-dependent selection bias.
    """
    if not 0.0 <= maximum_missing_fraction <= 1.0:
        raise ValueError(
            "maximum_missing_fraction must be between 0 and 1."
        )

    output = feature_dataframe.copy()

    available_features = [
        column
        for column in CANONICAL_MORPHOLOGY_FEATURES
        if column in output.columns
    ]

    if not available_features:
        return output, {
            "available_features": [],
            "dropped_features": [],
            "missing_fraction": {},
            "maximum_missing_fraction": maximum_missing_fraction,
        }

    morphology_values = (
        output[available_features]
        .replace([np.inf, -np.inf], np.nan)
    )

    missing_fraction = morphology_values.isna().mean()

    dropped_features = missing_fraction[
        missing_fraction > maximum_missing_fraction
    ].index.tolist()

    output = output.drop(columns=dropped_features, errors="ignore")

    return output, {
        "available_features": available_features,
        "dropped_features": dropped_features,
        "missing_fraction": missing_fraction.to_dict(),
        "maximum_missing_fraction": maximum_missing_fraction,
    }


def _validate_target_column(
    dataframe: pd.DataFrame,
    target_col: str,
) -> None:
    if target_col not in dataframe.columns:
        raise KeyError(
            f"Target column '{target_col}' was not found in the dataframe."
        )


def _validate_handle_inf(handle_inf: str) -> None:
    valid_methods = {"clip", "nan", "drop_features"}

    if handle_inf not in valid_methods:
        raise ValueError(
            f"handle_inf must be one of {sorted(valid_methods)}; "
            f"received '{handle_inf}'."
        )


def _require_unique_index(dataframe: pd.DataFrame) -> None:
    if not dataframe.index.is_unique:
        raise ValueError(
            "The dataframe index must be unique. Call "
            "reset_index(drop=True) after cohort filtering."
        )


def _get_numeric_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    return dataframe.select_dtypes(include=np.number).columns.tolist()


def _drop_missing_target_rows(
    dataframe: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    _validate_target_column(dataframe, target_col)

    valid_target = dataframe[target_col].notna()
    dataframe_valid = dataframe.loc[valid_target].copy()
    target = dataframe_valid[target_col].copy()

    return dataframe_valid, target


def _fill_chemistry(
    dataframe: pd.DataFrame,
    chemistry_columns: list[str],
) -> int:
    """
    Fill chemistry NaNs with 0.0 using the predetermined database convention.

    Returns the number of rows with no recorded chemistry values before fill.
    """
    if not chemistry_columns:
        return 0

    chemistry_block = dataframe.loc[:, chemistry_columns]

    rows_without_composition = int(
        chemistry_block.isna().all(axis=1).sum()
    )

    dataframe.loc[:, chemistry_columns] = chemistry_block.fillna(0.0)

    return rows_without_composition


def find_rare_support_chemistry_features(
    dataframe: pd.DataFrame,
    chemistry_columns: list[str],
    group_column: str = "sample_id",
    minimum_nonzero_groups: int = 5,
) -> list[str]:
    """
    Return non-Fe chemistry features with non-zero concentrations in fewer
    than minimum_nonzero_groups unique sample groups.

    This function must be fitted on a training fold only. Chemistry NaN
    values must already have been filled with 0.0 by _fill_chemistry().
    """
    if minimum_nonzero_groups < 1:
        raise ValueError(
            "minimum_nonzero_groups must be at least 1."
        )

    if group_column not in dataframe.columns:
        raise KeyError(
            f"Required grouping column '{group_column}' was not found."
        )

    alloy_columns = [
        column
        for column in chemistry_columns
        if column != "Fe" and column in dataframe.columns
    ]

    if not alloy_columns:
        return []

    chemistry_block = dataframe.loc[:, alloy_columns]

    # NaN != 0 evaluates to True, so unfilled NaNs would be counted as
    # reported non-zero values. Fail loudly instead of miscounting.
    if chemistry_block.isna().any().any():
        raise ValueError(
            "Chemistry NaN values must be filled with 0.0 before "
            "find_rare_support_chemistry_features() is called."
        )

    # A group counts for an element if any of its rows has a non-zero value.
    nonzero_by_group = (
        chemistry_block.ne(0.0)
        .groupby(dataframe[group_column], sort=False)
        .any()
    )

    nonzero_group_counts = nonzero_by_group.sum(axis=0)

    rare_support_features = nonzero_group_counts.loc[
        nonzero_group_counts < minimum_nonzero_groups
    ].index.tolist()

    return rare_support_features


def _build_feature_matrix(
    dataframe: pd.DataFrame,
    id_cols: Iterable[str],
    metadata_cols: Iterable[str],
    drop_fe: bool = True,
) -> pd.DataFrame:
    """
    Build candidate predictors using fixed, schema-based exclusions.
    """
    columns_to_drop = set(TARGET_COLUMNS)
    columns_to_drop.update(id_cols)
    columns_to_drop.update(metadata_cols)
    columns_to_drop.add("Other")

    if drop_fe:
        columns_to_drop.add("Fe")

    return dataframe.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in dataframe.columns
        ],
        errors="ignore",
    )


def _calculate_infinity_replacement(
    series: pd.Series,
) -> dict[str, float]:
    """Calculate training-fold-only finite clipping limits."""
    finite_values = series.loc[np.isfinite(series)]

    if finite_values.empty:
        return {
            "positive_clip": 1.0e10,
            "negative_clip": -1.0e10,
        }

    series_max = float(finite_values.max())
    series_min = float(finite_values.min())
    series_std = float(finite_values.std())

    if not np.isfinite(series_std):
        series_std = 0.0

    if series_max > 0:
        positive_clip = min(
            series_max * 10.0,
            series_max + 5.0 * series_std,
        )
    else:
        positive_clip = series_max + 5.0 * series_std

    if series_min < 0:
        negative_clip = max(
            series_min * 10.0,
            series_min - 5.0 * series_std,
        )
    else:
        negative_clip = series_min - 5.0 * series_std

    return {
        "positive_clip": float(positive_clip),
        "negative_clip": float(negative_clip),
    }

def _replace_infinities(
    dataframe: pd.DataFrame,
    handle_inf: str,
    infinity_replacements: dict[str, dict[str, float]] | None = None,
    infinite_features_to_drop: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Apply an existing infinity-handling policy.

    No feature-removal decision is made from validation or query data.
    """
    dataframe_processed = dataframe.copy()
    numeric_columns = _get_numeric_feature_columns(dataframe_processed)

    if handle_inf == "nan":
        for column in numeric_columns:
            infinity_mask = np.isinf(dataframe_processed[column])

            if infinity_mask.any():
                dataframe_processed.loc[infinity_mask, column] = np.nan

        return dataframe_processed

    if handle_inf == "drop_features":
        return dataframe_processed.drop(
            columns=list(infinite_features_to_drop or []),
            errors="ignore",
        )

    if infinity_replacements is None:
        raise ValueError(
            "Infinity clipping requires training-fitted replacement limits."
        )

    for column, replacement in infinity_replacements.items():
        if column not in dataframe_processed.columns:
            continue

        positive_mask = dataframe_processed[column] == np.inf
        negative_mask = dataframe_processed[column] == -np.inf

        dataframe_processed.loc[
            positive_mask,
            column,
        ] = replacement["positive_clip"]

        dataframe_processed.loc[
            negative_mask,
            column,
        ] = replacement["negative_clip"]

    return dataframe_processed


def _fit_median_values(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Fit a training-fold median for every retained numeric feature.

    A column without finite training values receives a deterministic 0.0 fill.
    """
    imputation_values: dict[str, float] = {}

    for column in _get_numeric_feature_columns(dataframe):
        median_value = dataframe[column].median(skipna=True)

        if pd.isna(median_value):
            median_value = 0.0

        imputation_values[column] = float(median_value)

    return imputation_values


def _apply_median_values(
    dataframe: pd.DataFrame,
    imputation_values: dict[str, float],
) -> pd.DataFrame:
    """Apply only training-fitted median values."""
    dataframe_processed = dataframe.copy()

    for column, median_value in imputation_values.items():
        if column in dataframe_processed.columns:
            dataframe_processed[column] = dataframe_processed[
                column
            ].fillna(median_value)

    return dataframe_processed

def _validate_composition(
    dataframe: pd.DataFrame,
    chemistry_columns: list[str],
) -> None:
    """
    Raise an error if any row has no reported alloying-element composition.

    Fe is the steel balance element and is excluded from this test. A valid
    steel record must contain at least one non-Fe elemental value.
    """
    alloy_columns = [
        column
        for column in chemistry_columns
        if column != "Fe"
    ]

    if not alloy_columns:
        raise KeyError(
            "No alloying-element columns from CHEMISTRY_COLUMNS were found."
        )

    rows_without_composition = dataframe[alloy_columns].isna().all(axis=1)

    if not rows_without_composition.any():
        return

    if "sample_id" in dataframe.columns:
        bad_sample_ids = dataframe.loc[
            rows_without_composition,
            "sample_id",
        ].astype(str).tolist()

        identifier_text = f"sample_id values: {bad_sample_ids}"
    else:
        bad_indices = dataframe.index[rows_without_composition].tolist()
        identifier_text = f"row indices: {bad_indices}"

    raise ValueError(
        "Rows with no reported alloying-element composition were found. "
        "This is a dataset error and must be corrected before modelling. "
        f"Affected {identifier_text}"
    )

def _is_constant(series: pd.Series) -> bool:
    """
    Return True when a feature has zero or one distinct finite values.

    NaN and +/-inf do not make a predictor informative because they are later
    imputed or handled by the cleaning pipeline.
    """
    finite_values = series.loc[np.isfinite(series)]

    return finite_values.nunique() <= 1


# =============================================================================
# Leakage-safe fold cleaning
# =============================================================================

def fit_cleaning_pipeline(
    dataframe: pd.DataFrame,
    target_col: str = "UTS (MPa)",
    id_cols: Iterable[str] | None = None,
    metadata_cols: Iterable[str] | None = None,
    missing_feature_threshold: float = 0.80,
    handle_inf: str = "clip",
    drop_fe: bool = True,
    minimum_nonzero_chemical_groups: int = 5,
    verbose: bool = True,
    morphology_missing_threshold: float = 0.20,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Fit a cleaning pipeline only on a CV training partition.

    Fixed operations:
    - Drop missing target rows.
    - Validate that every retained steel row has at least one reported
      non-Fe chemistry value.
    - Fill missing values in reported chemistry columns with 0.0.
    - Remove identifiers, metadata, all target columns, Fe, and Other.

    Training-fitted operations:
    - Remove non-Fe chemistry features with non-zero values in fewer than
      minimum_nonzero_chemical_groups unique sample_id groups (after the
      chemistry NaN fill and all training-row removals).
    - Remove finite-value constant features.
    - Remove canonical morphology features whose missing fraction exceeds
      morphology_missing_threshold (stricter than the generic rule; rows are
      retained). Runs after the image-level completeness rule.
    - Remove features above the missingness threshold.
    - Learn infinity handling from training data.
    - Learn per-feature median-imputation values from training data.
    """
    _validate_target_column(dataframe, target_col)
    _validate_handle_inf(handle_inf)
    _require_unique_index(dataframe)

    if not 0.0 <= missing_feature_threshold <= 1.0:
        raise ValueError(
            "missing_feature_threshold must be between 0.0 and 1.0."
        )

    if not 0.0 <= morphology_missing_threshold <= 1.0:
        raise ValueError(
            "morphology_missing_threshold must be between 0.0 and 1.0."
        )

    if id_cols is None:
        id_cols = list(DEFAULT_ID_COLUMNS)
    else:
        id_cols = list(id_cols)

    if metadata_cols is None:
        metadata_cols = list(DEFAULT_METADATA_COLUMNS)
    else:
        metadata_cols = list(metadata_cols)

    n_input_rows = len(dataframe)

    dataframe_train, target_train = _drop_missing_target_rows(
        dataframe,
        target_col,
    )

    if dataframe_train.empty:
        raise ValueError(
            f"No rows with observed '{target_col}' are available for fitting."
        )

    chemistry_available = [
        column
        for column in CHEMISTRY_COLUMNS
        if column in dataframe_train.columns
    ]

    _validate_composition(
        dataframe_train,
        chemistry_available,
    )

    _fill_chemistry(
        dataframe_train,
        chemistry_available,
    )

    features_train = _build_feature_matrix(
        dataframe_train,
        id_cols=id_cols,
        metadata_cols=metadata_cols,
        drop_fe=drop_fe,
    )

    numeric_columns = _get_numeric_feature_columns(features_train)

    nonnumeric_features = [
        column
        for column in features_train.columns
        if column not in numeric_columns
    ]

    if nonnumeric_features:
        features_train = features_train.drop(
            columns=nonnumeric_features,
            errors="ignore",
        )

    n_rows_removed_scale = 0

    constant_features = [
        column
        for column in features_train.columns
        if _is_constant(features_train[column])
    ]

    features_train = features_train.drop(
        columns=constant_features,
        errors="ignore",
    )
    
    features_train, image_missingness = fit_image_missingness_filter(
    features_train,
    feature_missingness_threshold=0.80,
    row_completeness_threshold=0.80,
    )
    
    target_train = target_train.loc[features_train.index].copy()

    # Morphology coverage guardrail: features only, rows are retained.
    features_train, morphology_guardrail = fit_morphology_feature_guardrail(
        features_train,
        maximum_missing_fraction=morphology_missing_threshold,
    )

    # Chemistry NaNs are already 0.0 here. Counting uses the rows that
    # actually remain in the training fold, so the support estimate matches
    # what the model sees.
    rare_support_chemistry_features = find_rare_support_chemistry_features(
        dataframe_train.loc[features_train.index],
        chemistry_columns=[
            column
            for column in chemistry_available
            if column in features_train.columns
        ],
        group_column="sample_id",
        minimum_nonzero_groups=minimum_nonzero_chemical_groups,
    )

    features_train = features_train.drop(
        columns=rare_support_chemistry_features,
        errors="ignore",
    )

    missing_fraction = (
        features_train
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .mean()
    )

    high_missing_features = missing_fraction.loc[
        missing_fraction > missing_feature_threshold
    ].index.tolist()

    features_train = features_train.drop(
        columns=high_missing_features,
        errors="ignore",
    )

    infinity_replacements: dict[str, dict[str, float]] = {}
    infinite_features_to_drop: list[str] = []

    for column in _get_numeric_feature_columns(features_train):
        if not np.isinf(features_train[column]).any():
            continue

        if handle_inf == "clip":
            infinity_replacements[column] = _calculate_infinity_replacement(
                features_train[column]
            )

        elif handle_inf == "drop_features":
            infinite_features_to_drop.append(column)

    features_train = features_train.drop(
        columns=infinite_features_to_drop,
        errors="ignore",
    )

    features_train = _replace_infinities(
        features_train,
        handle_inf=handle_inf,
        infinity_replacements=infinity_replacements,
        infinite_features_to_drop=infinite_features_to_drop,
    )

    n_features_with_missing = int(features_train.isna().any().sum())

    imputation_values = _fit_median_values(features_train)

    features_train = _apply_median_values(
        features_train,
        imputation_values,
    )

    if features_train.isna().any().any():
        columns_remaining_missing = features_train.columns[
            features_train.isna().any()
        ].tolist()

        raise ValueError(
            "Missing values remain after training-fold imputation. "
            f"Columns: {columns_remaining_missing}"
        )

    if np.isinf(features_train.to_numpy(dtype=float)).any():
        raise ValueError(
            "Infinite values remain after training-fold infinity handling."
        )

    forbidden_features = set(TARGET_COLUMNS).intersection(
        features_train.columns
    )

    if forbidden_features:
        raise RuntimeError(
            "Target columns reached the final training feature matrix: "
            f"{sorted(forbidden_features)}"
        )

    pipeline = {
        "target_col": target_col,
        "id_cols": id_cols,
        "metadata_cols": metadata_cols,
        "chemistry_cols": chemistry_available,
        "drop_fe": drop_fe,
        "nonnumeric_features": nonnumeric_features,
        "constant_features": constant_features,
        "rare_support_chemistry_features": rare_support_chemistry_features,
        "minimum_nonzero_chemical_groups": minimum_nonzero_chemical_groups,
        "high_missing_features": high_missing_features,
        "missing_feature_threshold": missing_feature_threshold,
        "handle_inf": handle_inf,
        "infinite_features_to_drop": infinite_features_to_drop,
        "infinity_replacements": infinity_replacements,
        "imputation_values": imputation_values,
        "final_columns": features_train.columns.tolist(),
        "image_missingness": image_missingness,
        "morphology_guardrail": morphology_guardrail,
    }

    if verbose:
        print("=" * 80)
        print("FITTED CLEANING PIPELINE")
        print("=" * 80)
        print(f"Target:                     {target_col}")
        print(f"Input rows:                 {n_input_rows}")
        print(f"Training rows retained:     {features_train.shape[0]}")
        print(f"Training features retained: {features_train.shape[1]}")
        print(f"Rows removed (no scale):    {n_rows_removed_scale}")
        print(f"Constant features removed:  {len(constant_features)}")
        print(
            "Rare-support chem removed:  "
            f"{len(rare_support_chemistry_features)}"
        )
        print(
            "Morphology feats removed:   "
            f"{len(morphology_guardrail['dropped_features'])}"
        )
        print(f"High-NaN features removed:  {len(high_missing_features)}")
        print(f"Inf features removed:       {len(infinite_features_to_drop)}")
        print(f"Inf features clipped:       {len(infinity_replacements)}")
        print(f"Features with NaN imputed:  {n_features_with_missing}")
        print(f"Fe removed:                 {drop_fe}")
        print("=" * 80)

    return features_train, target_train, pipeline


def apply_cleaning_pipeline(
    dataframe: pd.DataFrame,
    pipeline: dict[str, Any],
    target_col: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Apply a training-fitted cleaning pipeline to validation or query data.

    No data-derived statistic is fitted in this function. Output columns and
    their order exactly match pipeline["final_columns"].
    """
    _require_unique_index(dataframe)

    if target_col is None:
        target_col = pipeline["target_col"]

    dataframe_apply = dataframe.copy()
    target_apply: pd.Series | None = None

    if target_col in dataframe_apply.columns:
        valid_target = dataframe_apply[target_col].notna()

        dataframe_apply = dataframe_apply.loc[valid_target].copy()
        target_apply = dataframe_apply[target_col].copy()

    chemistry_columns = [
        column
        for column in pipeline["chemistry_cols"]
        if column in dataframe_apply.columns
    ]

    _validate_composition(
        dataframe_apply,
        chemistry_columns,
    )

    _fill_chemistry(
        dataframe_apply,
        chemistry_columns,
    )

    features_apply = _build_feature_matrix(
        dataframe_apply,
        id_cols=pipeline["id_cols"],
        metadata_cols=pipeline["metadata_cols"],
        drop_fe=pipeline["drop_fe"],
    )

    n_rows_removed_scale = 0

    for key in (
        "nonnumeric_features",
        "constant_features",
        "rare_support_chemistry_features",
        "high_missing_features",
        "infinite_features_to_drop",
    ):
        features_apply = features_apply.drop(
            columns=pipeline[key],
            errors="ignore",
        )

    features_apply, validation_image_completeness = (
        apply_image_missingness_filter(
            features_apply,
            pipeline["image_missingness"],
        )
    )
    
    if target_apply is not None:
        target_apply = target_apply.loc[features_apply.index].copy()

    # Must follow the image filter: row completeness there is computed over
    # the fitted retained_image_features, which still include these columns.
    features_apply = features_apply.drop(
        columns=pipeline["morphology_guardrail"]["dropped_features"],
        errors="ignore",
    )

    features_apply = _replace_infinities(
        features_apply,
        handle_inf=pipeline["handle_inf"],
        infinity_replacements=pipeline["infinity_replacements"],
        infinite_features_to_drop=pipeline["infinite_features_to_drop"],
    )

    absent_columns = [
        column
        for column in pipeline["final_columns"]
        if column not in features_apply.columns
    ]

    features_apply = features_apply.reindex(
        columns=pipeline["final_columns"],
    )

    numeric_apply_columns = _get_numeric_feature_columns(features_apply)

    n_unfitted_infinities = int(
        np.isinf(
            features_apply[numeric_apply_columns].to_numpy(dtype=float)
        ).sum()
    )

    if n_unfitted_infinities > 0:
        features_apply.loc[:, numeric_apply_columns] = features_apply.loc[
            :, numeric_apply_columns
        ].replace([np.inf, -np.inf], np.nan)

    features_apply = _apply_median_values(
        features_apply,
        pipeline["imputation_values"],
    )

    if features_apply.isna().any().any():
        columns_remaining_missing = features_apply.columns[
            features_apply.isna().any()
        ].tolist()

        raise RuntimeError(
            "Missing values remain after applying training-fold medians. "
            f"Columns: {columns_remaining_missing}"
        )

    if np.isinf(features_apply.to_numpy(dtype=float)).any():
        raise ValueError(
            "Infinite values remain after applying the cleaning pipeline."
        )

    if list(features_apply.columns) != pipeline["final_columns"]:
        raise RuntimeError(
            "Applied feature order differs from the fitted training schema."
        )

    forbidden_features = set(TARGET_COLUMNS).intersection(
        features_apply.columns
    )

    if forbidden_features:
        raise RuntimeError(
            "Target columns reached the applied feature matrix: "
            f"{sorted(forbidden_features)}"
        )

    if target_apply is not None:
        target_apply = target_apply.loc[features_apply.index].copy()

    if verbose:
        print(
            "Applied fitted cleaning pipeline: "
            f"{features_apply.shape[0]} rows x "
            f"{features_apply.shape[1]} features"
        )

        if n_unfitted_infinities > 0:
            print(
                f"Warning: {n_unfitted_infinities} infinity values not "
                "encountered in training were converted to NaN and then "
                "filled with training-fold medians."
            )

        if absent_columns:
            print(
                "Warning: absent validation/query columns were filled with "
                "training-fold medians:"
            )
            print(absent_columns)

    return features_apply, target_apply

# =============================================================================
# Convenience helper for an explicit outer-CV fold
# =============================================================================

def prepare_clean_fold(
    dataframe_train: pd.DataFrame,
    dataframe_validation: pd.DataFrame,
    target_col: str,
    verbose: bool = False,
    **fit_kwargs: Any,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series | None,
    dict[str, Any],
]:
    """
    Fit cleaning on one outer-CV training fold and apply it to its paired
    validation fold without refitting any data-derived cleaning statistic.

    Example
    -------
    X_train, y_train, X_valid, y_valid, cleaner = prepare_clean_fold(
        dataframe_train=df_train,
        dataframe_validation=df_valid,
        target_col="UTS (MPa)",
        missing_feature_threshold=0.80,
        handle_inf="clip",
        drop_fe=True,
    )
    """
    X_train, y_train, cleaning_pipeline = fit_cleaning_pipeline(
        dataframe_train,
        target_col=target_col,
        verbose=verbose,
        **fit_kwargs,
    )

    X_validation, y_validation = apply_cleaning_pipeline(
        dataframe_validation,
        pipeline=cleaning_pipeline,
        target_col=target_col,
        verbose=verbose,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        cleaning_pipeline,
    )


# =============================================================================
# Main: diagnostics and export of raw cohort/target source files
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    print(f"Module path: {module_path}")

    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    INPUT_FILE = "descriptors_reduced.csv"

    data_dir = os.path.join(module_path, "data")
    xy_dir = os.path.join(data_dir, "Xy")
    input_path = os.path.join(data_dir, INPUT_FILE)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            "Reduced descriptor CSV not found:\n"
            f"{input_path}\n\n"
            "Run feat_reducer.py first to create "
            "data/descriptors_reduced.csv."
        )

    dataframe_full = pd.read_csv(input_path, low_memory=False)

    required_columns = {
        "sample_id",
        "Category",
        "method",
    }

    missing_required = sorted(
        required_columns.difference(dataframe_full.columns)
    )

    if missing_required:
        raise KeyError(
            "The descriptor table lacks required cohort columns: "
            f"{missing_required}"
        )

    print("=" * 80)
    print("CLEAN_DATA: DIAGNOSTICS AND X/Y SOURCE EXPORT")
    print("=" * 80)
    print(f"Input file:     {input_path}")
    print(
        f"Loaded dataset: {dataframe_full.shape[0]} rows x "
        f"{dataframe_full.shape[1]} columns"
    )

    nan_summary = diagnose_nan_patterns(
        dataframe_full,
        target_col="UTS (MPa)",
    )

    infinity_summary = diagnose_infinity_sources(dataframe_full)

    print("\nMISSINGNESS DIAGNOSTICS")
    print(nan_summary.to_string(index=False))

    print("\nINFINITY DIAGNOSTICS")

    if infinity_summary.empty:
        print("No infinity values found.")
    else:
        print(infinity_summary.to_string(index=False))

    os.makedirs(xy_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("EXPORTING RAW COHORT/TARGET SOURCE TABLES")
    print("=" * 80)

    for group_name, group_definition in GROUP_DEFINITIONS.items():

        group_mask = (
            dataframe_full["Category"].isin(
                group_definition["categories"]
            )
            & dataframe_full["method"].isin(
                group_definition["methods"]
            )
        )

        dataframe_group = dataframe_full.loc[group_mask].copy()
        dataframe_group = dataframe_group.reset_index(drop=True)

        print(f"\nGROUP: {group_name.upper()}")
        print(f"Rows after cohort selection: {len(dataframe_group)}")

        if dataframe_group.empty:
            print("Warning: no rows matched; no output files were written.")
            continue

        print(
            f"Unique sample_id values: "
            f"{dataframe_group['sample_id'].nunique()}"
        )

        for target_short_name, target_col in TARGET_DEFINITIONS.items():

            if target_col not in dataframe_group.columns:
                print(
                    f"Warning: missing target '{target_col}'; "
                    "no files written."
                )
                continue

            dataframe_target = dataframe_group.loc[
                dataframe_group[target_col].notna()
            ].copy()

            if dataframe_target.empty:
                print(
                    f"Warning: no observed values for '{target_col}'; "
                    "no files written."
                )
                continue

            # These retain sample_id for GroupKFold. Dynamic cleaning remains
            # inside the training/validation fold loop in the model script.
            X_source = dataframe_target.drop(columns=[target_col]).copy()
            y_source = dataframe_target.loc[:, [target_col]].copy()

            x_filename = (
                f"X_{group_name}_{target_short_name}.csv"
            )
            y_filename = (
                f"y_{group_name}_{target_short_name}.csv"
            )

            x_path = os.path.join(xy_dir, x_filename)
            y_path = os.path.join(xy_dir, y_filename)

            X_source.to_csv(x_path, index=False)
            y_source.to_csv(y_path, index=False)

            n_x_columns = X_source.shape[1]
            n_predictors = n_x_columns - int("sample_id" in X_source.columns)
            n_y_columns = y_source.shape[1]
            
            print(
                f"  {target_short_name:<3} | "
                f"{len(X_source):4d} rows | "
                f"{X_source['sample_id'].nunique():4d} samples | "
                f"X: {n_x_columns:3d} columns "
                f"({n_predictors:3d} predictors) | "
                f"y: {n_y_columns} column | "
                f"{x_filename} | {y_filename}"
            )

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(
        "The X files retain raw predictor values and sample_id. In modelling, "
        "split by sample_id first; then call prepare_clean_fold() within "
        "each training/validation fold."
    )