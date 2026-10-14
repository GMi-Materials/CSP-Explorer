#!/usr/bin/env python3
"""
Deterministic, target-independent descriptor reduction for CSP Explorer.

Run directly in Spyder using Run file (F5).

Purpose
-------
1. Read data/descriptors_full.csv.
2. Collapse the five candidate segmentation methods (standard, fine, inverse,
   adaptive, multiotsu) into one canonical morphology block. For every row the
   method with the highest particle count is selected (ties go to the earlier
   method in MorphologyReducer.METHODS). The row-wise rule uses no other rows
   and no targets. Provenance is kept in morph_source
3. Keep every other non-Gram column unchanged.
4. Replace the complete Gram-derived descriptor family with a predefined,
   interpretable 30-column Gram panel.
5. Save the reduced CSV file and a JSON audit report.

Author: Gerfried Millner
Version: 1.3.0
Date: 12.10.2026
Status: Deliver

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Five Gram sources present in descriptors_full.csv.
GRAM_SOURCES = (
    "conv1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)

# Six predefined within-layer descriptors.
# Five Gram sources x six descriptors = exactly 30 retained Gram descriptors.
GRAM_STATISTICS = (
    "trace",
    "frobenius_norm",
    "spectral_norm",
    "condition_number",
    "eigen_entropy",
    "effective_rank",
)

# Exact column names expected in descriptors_full.csv.
FIXED_GRAM_COLUMNS = tuple(
    f"{source}_{statistic}"
    for source in GRAM_SOURCES
    for statistic in GRAM_STATISTICS
)

# All of these fields are considered Gram-derived and are removed unless
# explicitly included in FIXED_GRAM_COLUMNS.
GRAM_FAMILY_PREFIXES = (
    "conv1_",
    "layer1_",
    "layer2_",
    "layer3_",
    "layer4_",
    "cross_",
    "global_",
    "multi_scale_",
)

# =============================================================================
# Deterministic reduction functions
# =============================================================================

class SchemaValidationError(ValueError):
    """Raised if the input dataset is incompatible with the fixed schema."""


def is_gram_family_column(column_name: str) -> bool:
    """
    Return True if the column is Gram-derived.

    This rule depends only on the column name. It never evaluates data values,
    target variables, or the composition of a specific study subset.
    """
    return column_name.startswith(GRAM_FAMILY_PREFIXES)


def get_gram_family_columns(columns) -> list[str]:
    """Return all Gram-family columns while preserving their original order."""
    return [
        column
        for column in columns
        if is_gram_family_column(column)
    ]


def validate_input_schema(dataframe: pd.DataFrame) -> None:
    """
    Verify only the schema needed for deterministic feature reduction.

    No feature values, targets, missingness, variance, or correlations are used.
    """
    if dataframe.empty:
        raise SchemaValidationError(
            "The input dataframe contains no rows."
        )

    if dataframe.shape[1] == 0:
        raise SchemaValidationError(
            "The input dataframe contains no columns."
        )

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()

    if duplicate_columns:
        examples = ", ".join(str(column) for column in duplicate_columns[:10])

        raise SchemaValidationError(
            "Duplicate column names were found. The output schema would be "
            f"ambiguous. Examples: {examples}"
        )

    missing_columns = [
        column
        for column in FIXED_GRAM_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = "\n".join(
            f"    - {column}"
            for column in missing_columns
        )

        raise SchemaValidationError(
            "The input dataframe is missing one or more required fixed "
            "Gram descriptors:\n"
            f"{missing_text}\n\n"
            "No missing Gram descriptor is created or filled with zero. "
            "Check the upstream descriptor extraction or revise the fixed "
            "Gram schema intentionally."
        )

    if not get_gram_family_columns(dataframe.columns):
        raise SchemaValidationError(
            "No Gram-family columns were detected."
        )


def build_reduced_column_order(input_columns) -> list[str]:
    """
    Create a deterministic output schema.

    All non-Gram columns remain in their original order. The 30 retained Gram
    fields replace the entire original Gram block at the position where the
    first Gram-family field occurred.
    """
    output_columns = []
    inserted_fixed_gram_panel = False

    for column in input_columns:

        if is_gram_family_column(column):

            if not inserted_fixed_gram_panel:
                output_columns.extend(FIXED_GRAM_COLUMNS)
                inserted_fixed_gram_panel = True

            continue

        output_columns.append(column)

    if not inserted_fixed_gram_panel:
        raise SchemaValidationError(
            "No Gram-family block was found for replacement."
        )

    output_column_index = pd.Index(output_columns)

    duplicate_output_columns = output_column_index[
        output_column_index.duplicated()
    ].tolist()

    if duplicate_output_columns:
        examples = ", ".join(
            str(column)
            for column in duplicate_output_columns[:10]
        )

        raise SchemaValidationError(
            "The reduced schema would contain duplicate columns: "
            f"{examples}"
        )

    return output_columns


def reduce_gram_descriptors(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Replace all Gram-derived columns with a fixed 30-column Gram panel.

    Parameters
    ----------
    dataframe:
        A full PSP descriptor dataframe. Targets may be present, but they are
        neither read nor used to choose, transform, or evaluate descriptors.

    Returns
    -------
    dataframe_reduced:
        The reduced dataframe.

    report:
        A JSON-serializable audit record.
    """
    validate_input_schema(dataframe)

    input_columns = dataframe.columns.tolist()

    input_gram_columns = get_gram_family_columns(input_columns)

    discarded_gram_columns = [
        column
        for column in input_gram_columns
        if column not in FIXED_GRAM_COLUMNS
    ]

    output_columns = build_reduced_column_order(input_columns)

    # Direct column selection only. No values are transformed.
    dataframe_reduced = dataframe.loc[:, output_columns].copy()

    # -------------------------------------------------------------------------
    # Defensive validation: guarantee the stated transformation.
    # -------------------------------------------------------------------------
    expected_non_gram_columns = [
        column
        for column in input_columns
        if not is_gram_family_column(column)
    ]

    actual_non_gram_columns = [
        column
        for column in dataframe_reduced.columns
        if not is_gram_family_column(column)
    ]

    if actual_non_gram_columns != expected_non_gram_columns:
        raise RuntimeError(
            "Internal validation failed: non-Gram columns were altered, "
            "removed, or reordered."
        )

    if not dataframe_reduced.loc[:, FIXED_GRAM_COLUMNS].equals(
        dataframe.loc[:, FIXED_GRAM_COLUMNS]
    ):
        raise RuntimeError(
            "Internal validation failed: retained Gram fields do not exactly "
            "match their input values."
        )

    if dataframe_reduced.shape[0] != dataframe.shape[0]:
        raise RuntimeError(
            "Internal validation failed: row count changed."
        )

    report = {
        "script": Path(__file__).name,
        "operation": (
            "deterministic_target_independent_gram_schema_reduction"
        ),
        "input_rows": int(dataframe.shape[0]),
        "output_rows": int(dataframe_reduced.shape[0]),
        "input_columns": int(dataframe.shape[1]),
        "output_columns": int(dataframe_reduced.shape[1]),
        "input_gram_columns": int(len(input_gram_columns)),
        "retained_gram_columns": int(len(FIXED_GRAM_COLUMNS)),
        "discarded_gram_columns": int(len(discarded_gram_columns)),
        "gram_reduction_percent": round(
            100.0 * (
                1.0 - len(FIXED_GRAM_COLUMNS) / len(input_gram_columns)
            ),
            4,
        ),
        "retained_gram_column_names": list(FIXED_GRAM_COLUMNS),
        "discarded_gram_column_names": discarded_gram_columns,
        "non_gram_columns_retained_unchanged": int(
            len(expected_non_gram_columns)
        ),
        "target_columns_used": [],
        "data_fitted_operations_used": [],
        "composition_columns_modified": [],
    }

    return dataframe_reduced, report


class MorphologyReducer:
    """Collapse candidate segmentations into one canonical morphology block."""

    METHODS = ("standard", "fine", "inverse", "adaptive", "multiotsu")

    METRICS = (
        "n_particles",
        "area_mean",
        "diameter_mean",
        "aspect_ratio_mean",
        "eccentricity_mean",
    )

    # Candidate-specific bookkeeping fields that are removed with the candidates.
    LEGACY_COLUMNS = ("morph_method_used",)

    def reduce(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Select, per row, the segmentation method with the highest particle
        count and copy its five metrics into morph_<metric>.

        Rows without any segmentation of at least one particle get
        morph_n_particles = 0, NaN shape metrics, morph_source = "none"
        Rows are never dropped here.
        """
        output = dataframe.copy()
        n_rows = len(output)

        canonical_columns = [f"morph_{metric}" for metric in self.METRICS]

        candidate_columns = [
            f"morph_{method}_{metric}"
            for method in self.METHODS
            for metric in self.METRICS
            if f"morph_{method}_{metric}" in output.columns
        ]

        if not candidate_columns:
            return output, {
                "applied": False,
                "n_candidate_columns_removed": 0,
                "source_counts": {},
                "canonical_columns": [
                    column
                    for column in output.columns
                    if column.startswith("morph_")
                ],
                "failed_morphology_rows": 0,
            }

        selected_source = np.full(n_rows, "none", dtype=object)
        selected_count = np.full(n_rows, -1.0)

        for method in self.METHODS:
            count_column = f"morph_{method}_n_particles"

            if count_column not in output.columns:
                continue

            counts = (
                pd.to_numeric(output[count_column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-1.0)
                .to_numpy(dtype=float)
            )

            # Strict ">" keeps the earlier method on ties: deterministic.
            replace = counts > selected_count

            selected_count[replace] = counts[replace]
            selected_source[replace] = method

        valid_selection = selected_count >= 1
        failed_mask = ~valid_selection

        # A method with zero particles is not a usable source.
        selected_source[failed_mask] = "none"

        overwritten_columns = [
            column for column in canonical_columns if column in output.columns
        ]

        for metric in self.METRICS:
            values = np.full(n_rows, np.nan)

            for method in self.METHODS:
                source_column = f"morph_{method}_{metric}"

                if source_column not in output.columns:
                    continue

                method_mask = valid_selection & (selected_source == method)

                if not method_mask.any():
                    continue

                source_values = pd.to_numeric(
                    output[source_column],
                    errors="coerce",
                ).to_numpy(dtype=float)

                values[method_mask] = source_values[method_mask]

            if metric == "n_particles":
                values[failed_mask] = 0.0

            output[f"morph_{metric}"] = values

        output["morph_source"] = selected_source

        output = output.drop(
            columns=candidate_columns + list(self.LEGACY_COLUMNS),
            errors="ignore",
        )

        # Defensive validation of the stated transformation.
        leftover = [
            column
            for column in output.columns
            if column.startswith(
                tuple(f"morph_{method}_" for method in self.METHODS)
            )
        ]

        if leftover:
            raise RuntimeError(
                "Internal validation failed: candidate morphology columns "
                f"remain: {leftover}"
            )

        if not set(canonical_columns).issubset(output.columns):
            raise RuntimeError(
                "Internal validation failed: canonical morphology columns "
                "are missing."
            )

        if len(output) != n_rows:
            raise RuntimeError(
                "Internal validation failed: row count changed."
            )

        return output, {
            "applied": True,
            "n_candidate_columns_removed": len(candidate_columns),
            "source_counts": {
                str(key): int(value)
                for key, value in (
                    pd.Series(selected_source).value_counts().items()
                )
            },
            "canonical_columns": canonical_columns,
            "provenance_columns": ["morph_source"],
            "pre_existing_canonical_columns_overwritten": overwritten_columns,
            "failed_morphology_rows": int(failed_mask.sum()),
        }


def reduce_descriptors(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Full deterministic reduction: morphology collapse, then Gram panel.

    Every column that is neither Gram-derived nor morph_-prefixed is verified
    to be unchanged in values, order and row count.
    """
    n_input_columns = int(dataframe.shape[1])

    untouched_columns = [
        column
        for column in dataframe.columns
        if not is_gram_family_column(column)
        and not column.startswith("morph_")
    ]

    dataframe_morphology, morphology_report = MorphologyReducer().reduce(
        dataframe
    )

    dataframe_reduced, report = reduce_gram_descriptors(dataframe_morphology)

    untouched_set = set(untouched_columns)

    if [
        column
        for column in dataframe_reduced.columns
        if column in untouched_set
    ] != untouched_columns:
        raise RuntimeError(
            "Internal validation failed: untouched columns were removed "
            "or reordered."
        )

    if not dataframe_reduced.loc[:, untouched_columns].equals(
        dataframe.loc[:, untouched_columns]
    ):
        raise RuntimeError(
            "Internal validation failed: untouched columns changed values."
        )

    report["operation"] = (
        "deterministic_target_independent_morphology_and_gram_schema_reduction"
    )
    report["input_columns"] = n_input_columns
    report["non_gram_columns_retained_unchanged"] = int(
        len(untouched_columns)
    )
    report["morphology"] = morphology_report

    return dataframe_reduced, report


def save_report(report: dict, report_path: Path) -> None:
    """Save one JSON audit report."""
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")


def print_summary(
    group_name: str,
    dataframe_original: pd.DataFrame,
    dataframe_reduced: pd.DataFrame,
    report: dict,
) -> None:
    """Print a concise processing summary in Spyder's console."""
    print("\n" + "=" * 80)
    print(f"GROUP: {group_name}")
    print("=" * 80)
    print(f"Rows:                  {dataframe_original.shape[0]}")
    print(
        f"Columns:               "
        f"{dataframe_original.shape[1]} -> {dataframe_reduced.shape[1]}"
    )
    print(
        f"Gram descriptors:      "
        f"{report['input_gram_columns']} -> "
        f"{report['retained_gram_columns']}"
    )
    print(
        f"Removed Gram fields:   "
        f"{report['discarded_gram_columns']}"
    )
    print(
        f"Gram reduction:        "
        f"{report['gram_reduction_percent']:.1f}%"
    )
    print(
        f"Non-Gram fields kept:  "
        f"{report['non_gram_columns_retained_unchanged']}"
    )
    print("Target columns used:   none")
    print("Fitted operations:     none")
    print("Composition modified:  none")


# =============================================================================
# Main: run this part directly using F5 in Spyder
# =============================================================================

if __name__ == "__main__":

    import os, sys
    module_path = str.rsplit(os.path.abspath(""), os.path.sep, 1)[0]
    print(f"Module path: {module_path}")

    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    input_path = os.path.join(
        module_path,
        "data",
        "descriptors_full.csv",
    )

    output_path = os.path.join(
        module_path,
        "data",
        "descriptors_reduced.csv",
    )

    report_path = os.path.join(
        module_path,
        "data",
        "descriptors_reduced_report.json",
    )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            "Input CSV not found:\n"
            f"{input_path}\n\n"
            "Expected location: <project_root>/data/descriptors_full.csv"
        )

    print("=" * 80)
    print("DETERMINISTIC DESCRIPTOR REDUCTION (MORPHOLOGY + GRAM)")
    print("=" * 80)
    print(f"Input file:     {input_path}")

    dataframe_full = pd.read_csv(input_path, low_memory=False)

    print(
        f"Input dataset:  "
        f"{dataframe_full.shape[0]} rows x "
        f"{dataframe_full.shape[1]} columns"
    )

    dataframe_reduced, report = reduce_descriptors(dataframe_full)

    dataframe_reduced.to_csv(output_path, index=False)

    save_report(
        report,
        Path(report_path),
    )

    print("\n" + "=" * 80)
    print("REDUCTION SUMMARY")
    print("=" * 80)
    print(f"Rows preserved:        {report['input_rows']}")
    print(
        f"Total columns:         "
        f"{report['input_columns']} -> "
        f"{report['output_columns']}"
    )
    print(
        f"Gram columns:          "
        f"{report['input_gram_columns']} -> "
        f"{report['retained_gram_columns']}"
    )
    print(
        f"Discarded Gram columns: "
        f"{report['discarded_gram_columns']}"
    )
    print(
        f"Gram reduction:        "
        f"{report['gram_reduction_percent']:.2f}%"
    )
    print(
        f"Other columns kept:    "
        f"{report['non_gram_columns_retained_unchanged']} "
        f"(non-Gram, non-morphology)"
    )

    morphology_report = report["morphology"]

    if morphology_report["applied"]:
        print(
            f"Morphology candidates: "
            f"{morphology_report['n_candidate_columns_removed']} columns "
            f"collapsed into {len(morphology_report['canonical_columns'])} "
            f"canonical + morph_source"
        )
        print(f"Morphology sources:    {morphology_report['source_counts']}")
        print(
            f"Failed morphology rows: "
            f"{morphology_report['failed_morphology_rows']} "
        )
    else:
        print("Morphology:            no candidate columns found; unchanged")

    print("Target columns used:   none")
    print("Fitted operations:     none")
    print("Composition modified:  none")
    print(f"\nSaved CSV:    {output_path}")
    print(f"Saved report: {report_path}")
    print("=" * 80)