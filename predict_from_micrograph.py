# -*- coding: utf-8 -*-
"""
Predict Material Properties from a Single Micrograph
=====================================================
Author : Gerfried Millner
Version: 1.2.0
Date   : 12.10.2026
Email  : gerfried.millner@ntu.edu.sg
Status : Development

Overview
--------
Given:
(1) A micrograph (PNG / TIFF / JPG) already prepared by the user
(2) A pre-built mask (.npy) produced by doi_and_mask-3.py
(3) User-supplied metadata (modality, um_per_pixel, composition, …)
(4) A literature training dataset CSV

This script:
Step 0  – Configuration
Step 1  – Load image + mask
Step 2  – Extract descriptors
Step 3  – Assemble query DataFrame with metadata
Step 4-6 – Deterministic descriptor reduction (feat_reducer.reduce_descriptors:
          morphology collapse + fixed 30-column Gram panel; nothing is fitted,
          so training frame and query rows are reduced by the same row-wise rule)
Step 7  – Cleaning (fold-safe clean_data pipeline, fitted by the final model)
Step 8  – Feature schema check
Step 9  – Model training and validation (XGBoost, per target + modality)

Pre-requisite for the user
--------------------------
Run doi_and_mask.py on the GIMP colour-annotated image first to produce
the _mask.npy file.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

import pickle
import numpy as np
import pandas as pd
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap – resolve project root so sibling packages are importable
# ---------------------------------------------------------------------------
MODULE_PATH = str(Path(__file__).resolve().parent)
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

from image_processing.extract_descriptors import (
    extract_all_descriptors,
    GramDescriptorExtractor,
)
from data_processing.combine_data import split_images_to_observations
from data_processing.feat_reducer import (
    reduce_descriptors,
    SchemaValidationError,
)
from data_processing.clean_data import (
    fit_cleaning_pipeline,
    apply_cleaning_pipeline,
    CHEMISTRY_COLUMNS,
    TARGET_COLUMNS,
)

from machine_learning.models_paper import (
    train_final_xgb_full,
    CLEANING_KWARGS,
)

# ===========================================================================
# Constants
# ===========================================================================

TARGETS = ["UTS (MPa)", "YS (MPa)"]

# ===========================================================================
# Step 0 – Configuration
# ===========================================================================

@dataclass
class QueryConfig:
    """
    All user-supplied inputs for a single micrograph prediction run.

    Required inputs
    ---------------
    name : str
        Image stem without extension, e.g. "C_tran_1".
    image_folder : str
        Absolute path to the folder that contains the image file AND a
        "masks" subfolder with the pre-built .npy mask.

    Example
    -------
    image_folder = r"C:\\...\\data\\Alleima_Validation\\processed_images\\OM"
    name = "C_tran_1"

    Derived automatically
    ~~~~~~~~~~~~~~~~~~~~~
    image_path → \\C_tran_1.png
    mask_path  → \\masks\\C_tran_1_mask.npy

    Works with any absolute path; no project-root assumption is made.

    Imaging metadata
    ----------------
    image_modality : str
        Imaging mode matching the 'method' column in the training CSV.
        Examples: "SEM: BSE", "SEM: SE", "OM".
        Use "SEM" to aggregate both SEM: SE and SEM: BSE from the training CSV.
    um_per_pixel : float | None
        Physical scale in µm per pixel.
        If None, provide both resolution_um and resolution_px instead.
    resolution_um : float | None
        Physical length of the scale bar in µm.
    resolution_px : float | None
        Pixel length of the same scale bar.

    Material and composition
    ------------------------
    material : str
        Material category, e.g. "Steel", "Aluminum".
    material_subcategory : str
        Sub-category describing THIS image, e.g. "Duplex Stainless Steel".
        Used as metadata on the query row; does not filter training data.
    material_categories : list[str]
        Subcategory / category labels used to filter the TRAINING SET before
        fitting any transformer or ML model. Must match values in the
        "Material_Subcategory" or "Category" column of the training CSV.
        Pass an empty list to skip this filter.
    composition : dict
        Elemental composition in wt-%, e.g. {"C": 0.04, "Cr": 22.0}.
        Elements absent from the dict are treated as 0 wt-%.

    Dataset and target
    ------------------
    training_dataset_csv : str
        Path to the combined literature descriptor CSV (training set).
    target_property : str
        Column name of the target property in the training CSV.
        Examples: "UTS (MPa)", "YS (MPa)", "Total EL (%)".
    id_cols : list[str]
        Kept for backward compatibility only. The cleaning pipeline now uses
        its own DEFAULT_ID_COLUMNS / DEFAULT_METADATA_COLUMNS (um_per_pixel is
        deliberately a feature), so this list is NOT passed to it any more.
    max_imputed_feature_frac : float
        A warning is logged when more than this fraction of the model
        features is absent or non-finite in a query and therefore filled
        with training medians.

    Feature extraction
    ------------------
    use_gram : bool
        Whether to run the VGG19/ResNet Gram extractor.
    gram_device : str | None
        "cuda", "cpu", or None for auto-detection.

    ML model (Step 9)
    -----------------
    run_ml : bool
        If True, train XGBoost and predict for each target after process_folder.
    n_trials : int
        Number of Optuna trials for XGBoost hyperparameter search (default 100).
    n_splits_inner : int
        Number of inner CV folds for Optuna (default 5).

    Output
    ------
    output_dir : str
        Directory for output files. Defaults to a "predictions" subfolder
        next to image_folder.
    verbose : bool
    """
    # --- Required ---
    name: str = ""
    image_folder: str = ""

    # --- Imaging metadata ---
    image_modality: str = "SEM: BSE"
    um_per_pixel: Optional[float] = None
    resolution_um: Optional[float] = None
    resolution_px: Optional[float] = None

    # --- Material and composition ---
    material: str = "Steel"
    material_subcategory: str = ""
    composition: Dict[str, float] = field(default_factory=dict)

    # --- Dataset and target ---
    training_dataset_csv: str = ""
    target_property: str = "UTS (MPa)"
    material_categories: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=lambda: [
        # Core identifiers
        "obs_id", "ID_name", "id_name", "sample_id", "Sample",
        "Paper_ID", "DOI", "DOI_Enhanced", "doi", "doi_norm", "doi_url",
        # Image bookkeeping
        "image_file", "image_filename", "image_name", "image_number",
        "image_name_norm", "Fig",
        # Paper / provenance
        "subfolder", "pdf_filename", "extraction_date",
        # Material labels (encoded separately, not used as features)
        "Material", "Material_Subcategory", "Category", "method",
        # Classification metadata
        "cls_type", "cls_confidence",
        "Classification_Confidence",
        # Morphology source bookkeeping
        "morph_source", "feat_morph_source",
        # Scale (already converted to physical features)
        "um_per_pixel", "resolution_um", "resolution_px",
        # Balance element – not an independent predictor
        "Fe",
    ])

    # --- Feature extraction ---
    use_gram: bool = True
    gram_device: Optional[str] = None

    # --- ML model (Step 9) ---
    run_ml: bool = True
    n_trials: int = 100
    n_splits_inner: int = 5
    max_imputed_feature_frac: float = 0.20

    # --- Output ---
    output_dir: str = ""
    verbose: bool = True
    output_timestamp: str = ""

    def __post_init__(self):
        if not self.image_folder:
            raise ValueError(
                "QueryConfig.image_folder must be set to the folder "
                "containing the image and the masks/ subfolder."
            )

        folder = Path(self.image_folder)

        if self.name:
            self.image_path = str(folder / f"{self.name}.png")
            self.mask_path = str(folder / "masks" / f"{self.name}_mask.npy")
        else:
            self.image_path = ""
            self.mask_path = ""
            
        if not self.output_dir:
            predictions_dir = folder / "predictions"
            if self.output_timestamp:
                self.output_dir = str(predictions_dir / self.output_timestamp)
            else:
                self.output_dir = str(predictions_dir)


# ===========================================================================
# Step 1 – Load image and mask
# ===========================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Load micrograph as a 2-D uint8 grayscale array.

    Handles RGB, RGBA, and single-channel inputs via scikit-image.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    np.ndarray shape (H, W), dtype uint8
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = io.imread(str(p))

    if image.ndim == 3 and image.shape[2] == 4:   # RGBA → RGB
        image = image[:, :, :3]
    if image.ndim == 3:                            # RGB → gray
        image = rgb2gray(image)
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2-D array after grayscale conversion; got {image.shape}."
        )

    logger.info(f"Image loaded : '{p.name}' shape={image.shape} dtype=uint8")
    return image


def load_mask(mask_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load a pre-built boolean exclusion mask from a .npy file.

    The mask was produced by doi_and_mask-3.py (build_mask function).
    Convention: True = pixel excluded from all descriptors.

    Parameters
    ----------
    mask_path : str | None
        If None, no mask is applied.

    Returns
    -------
    np.ndarray (bool) or None
    """
    if mask_path is None:
        logger.info("No mask supplied → all pixels included.")
        return None

    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}\n"
            f"Run doi_and_mask-3.py on the GIMP-annotated image first."
        )

    raw = np.load(str(p), allow_pickle=False)

    if raw.dtype == np.uint8:
        mask_bool = raw.astype(bool)
    elif raw.dtype == bool:
        mask_bool = raw
    else:
        mask_bool = raw.astype(bool)
        logger.warning(f"Unexpected mask dtype '{raw.dtype}'; converted to bool.")

    pct = 100.0 * mask_bool.sum() / mask_bool.size
    logger.info(
        f"Mask loaded : '{p.name}' "
        f"{mask_bool.sum():,} / {mask_bool.size:,} px excluded ({pct:.1f}%)"
    )
    return mask_bool


def check_mask_shape(image: np.ndarray, mask: Optional[np.ndarray]) -> None:
    """Raise if a mask does not have the same (H, W) as its image."""
    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape} vs mask {mask.shape}. "
            "Ensure the mask was generated from the same image."
        )


# ===========================================================================
# Utility – Filter training data by modality and material categories
# ===========================================================================

def filter_training_data(
    df_train: pd.DataFrame,
    image_modality: str,
    material_categories: List[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Restrict the training DataFrame to the rows used for fitting all
    transformers (feature reducers, cleaning pipeline) and the ML model.

    For SEM, image_modality="SEM" aggregates both "SEM: SE" and "SEM: BSE"
    from the 'method' column of the training CSV.
    """
    n_start = len(df_train)

    # --- Modality filter ---
    if "method" in df_train.columns:
        if image_modality == "SEM":
            methods = ["SEM: SE", "SEM: BSE"]
            df_filtered = df_train[df_train["method"].isin(methods)].copy()
        else:
            df_filtered = df_train[df_train["method"] == image_modality].copy()

        if len(df_filtered) == 0:
            available = df_train["method"].dropna().unique().tolist()
            raise ValueError(
                f"No training samples with method='{image_modality}'. "
                f"Available: {available}"
            )
    else:
        logger.warning("'method' column not found — modality filter skipped.")
        df_filtered = df_train.copy()

    n_after_modality = len(df_filtered)

    # --- Category filter ---
    if material_categories:
        cat_col = None
        for col in ("Material_Subcategory", "Category"):
            if col in df_filtered.columns:
                cat_col = col
                break

        if cat_col is None:
            logger.warning(
                "Neither 'Material_Subcategory' nor 'Category' column found "
                "— material category filter skipped."
            )
        else:
            df_filtered = df_filtered[
                df_filtered[cat_col].isin(material_categories)
            ].copy()
            if len(df_filtered) == 0:
                available = df_train[cat_col].dropna().unique().tolist()
                raise ValueError(
                    f"No training samples matching categories "
                    f"{material_categories} in column '{cat_col}'. "
                    f"Available: {available}"
                )

    df_filtered = df_filtered.reset_index(drop=True)

    if verbose:
        logger.info(
            f"Training filter : {n_start} → {n_after_modality} (modality) "
            f"→ {len(df_filtered)} (category) samples retained."
        )
        if material_categories:
            logger.info(f" Categories      :  {material_categories}")

    return df_filtered


# ===========================================================================
# Step 2 – Extract descriptors
# ===========================================================================

def extract_descriptors(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    image_name: str,
    use_gram: bool = True,
    gram_device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Extract the full descriptor vector from a single micrograph.

    Delegates to extract_all_descriptors() which performs:
    - Adaptive ensemble segmentation → morphology descriptors
    - GLCM texture (multi-distance, multi-angle)
    - Local Binary Pattern histogram
    - 2-point spatial correlation (S2)
    - VGG19/ResNet Gram matrix features (optional)

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    mask : np.ndarray | None
        Boolean exclusion mask. True = excluded pixel.
    image_name : str
        Used as the 'filename' argument inside extract_all_descriptors.
    use_gram : bool
    gram_device : str | None
    verbose : bool

    Returns
    -------
    dict {feature_name: scalar}
    """
    gram_extractor = None
    if use_gram:
        try:
            gram_extractor = GramDescriptorExtractor(device=gram_device)
            logger.info(f"Gram extractor: device='{gram_extractor.device}'")
        except Exception as exc:
            logger.warning(
                f"GramDescriptorExtractor init failed ({exc}); "
                f"continuing without Gram features."
            )

    descriptors = extract_all_descriptors(
        image=image,
        mask=mask,
        filename=image_name,
        subfolder="",
        use_adaptive=True,
        gram_extractor=gram_extractor,
        verbose=verbose,
    )

    logger.info(f"Descriptors : {len(descriptors)} features extracted.")
    return descriptors


# ===========================================================================
# Step 3 – Assemble query DataFrame
# ===========================================================================

def _resolve_um_per_pixel(
    config: "QueryConfig",
    strict: bool = True,
) -> Optional[float]:
    """
    Resolve scale from QueryConfig.

    strict=True → raise if nothing is provided (single-image mode).
    strict=False → return None and rely on external metadata (e.g. obs_df).
    """
    if config.um_per_pixel is not None:
        return config.um_per_pixel

    if config.resolution_um is not None and config.resolution_px is not None:
        if config.resolution_px <= 0:
            raise ValueError("resolution_px must be > 0.")
        return config.resolution_um / config.resolution_px

    if strict:
        raise ValueError(
            "Provide either 'um_per_pixel' or both "
            "'resolution_um' and 'resolution_px' in QueryConfig."
        )

    logger.warning(
        "No scale specified in QueryConfig; will rely on per-image metadata "
        "(um_per_pixel from validation dataframe)."
    )
    return None


def build_query_dataframe(
    descriptors: Dict[str, Any],
    config: "QueryConfig",
    strict_scale: bool = True,
    image_name: str = "",
) -> pd.DataFrame:
    """
    Combine raw descriptors with user-supplied metadata into a single-row
    DataFrame matching the column schema of the training dataset.

    Columns added
    -------------
    image_filename, image_name, id_name, subfolder, method, um_per_pixel,
    Material, Material_Subcategory, is_bse,
    + one column per composition element in config.composition.

    Note
    ----
    In the batch/folder path, enrich_query_with_observation() overwrites
    method, is_bse, Material, Material_Subcategory, um_per_pixel, and
    composition with authoritative per-image values from the validation
    DataFrame, so the values set here are only the ones actually used in
    single-image mode (run_prediction()).

    Parameters
    ----------
    descriptors : dict
    config : QueryConfig
    strict_scale : bool
    image_name : str
        Image stem; falls back to config.name when empty (batch mode passes it
        explicitly because config.name is "" there).

    Returns
    -------
    pd.DataFrame (1 row)
    """
    um_per_pixel = _resolve_um_per_pixel(config, strict=strict_scale)
    image_stem = image_name or (Path(config.image_path).stem if config.image_path else "")
    row: Dict[str, Any] = dict(descriptors)
    row.update({
        "image_filename": f"{image_stem}.png",
        "image_name": image_stem,
        "id_name": image_stem,
        "subfolder": Path(config.image_folder).name if config.image_folder else "",
        "method": config.image_modality,
        "um_per_pixel": um_per_pixel,
        "Material": config.material,
        "Material_Subcategory": config.material_subcategory,
    })

    # Elemental composition (wt-%) — one column per element supplied in
    # config.composition. Elements absent from the dict are left unset
    # here and are zero-filled by clean_query() before the cleaning pipeline
    # is applied, which is equivalent to "treated as 0 wt-%".
    unknown_elements = [e for e in config.composition if e not in CHEMISTRY_COLUMNS]
    if unknown_elements:
        logger.warning(
            f"Composition elements not in CHEMISTRY_COLUMNS are not model "
            f"features and will be ignored: {unknown_elements}"
        )
    for element, wt_pct in config.composition.items():
        row[element] = wt_pct

    df_query = pd.DataFrame([row])

    # Derive is_bse now that 'method' is known (mirrors the equivalent
    # step in enrich_query_with_observation() for the batch path).
    df_query["is_bse"] = df_query["method"].apply(
        lambda m: 1 if "BSE" in str(m).upper() else 0
    )

    return df_query


# ===========================================================================
# Steps 4-6 – Deterministic descriptor reduction
# ===========================================================================

def reduce_features(df: pd.DataFrame, label: str = "frame") -> pd.DataFrame:
    """
    Apply feat_reducer.reduce_descriptors() to a frame.

    The reduction collapses the five candidate segmentations into the
    canonical morph_* block (highest particle count wins) and replaces the
    whole Gram family by the fixed 30-column panel. It is row-wise and uses no
    target and no fitted statistic, so the training frame and every query row
    go through exactly the same rule that produced the training data.

    Physical-unit conversion (add_physical_units_to_features) is no longer
    part of the pipeline: it only converts columns named feat_*, which neither
    the descriptor extractor nor the training CSV contains, and um_per_pixel is
    a model feature on its own.
    """
    try:
        df_reduced, _ = reduce_descriptors(df)
    except SchemaValidationError as exc:
        raise RuntimeError(
            f"[{label}] Descriptor reduction failed: {exc}\n"
            "For a query image this usually means that the Gram extractor "
            "did not run (use_gram=False or GramDescriptorExtractor init "
            "failed); the fixed Gram panel is required and is never "
            "filled or zero-padded."
        ) from exc

    # Candidate-only bookkeeping field. The reducer removes it together with
    # the per-method columns; a query that took the adaptive path has no
    # per-method columns, so it is removed here to keep the schemas equal.
    return df_reduced.drop(columns=["morph_method_used"], errors="ignore")


# ===========================================================================
# Step 7 – Cleaning pipeline
# ===========================================================================

class QueryRemovedByCleaning(ValueError):
    """The cleaning pipeline dropped the query row (see clean_query)."""


def clean_query(
    df_query: pd.DataFrame,
    pipeline: Dict[str, Any],
    name: str = "query",
    max_imputed_frac: float = 0.20,
    return_imputed: bool = False,
):
    """
    Clean a one-row query frame with a FITTED cleaning pipeline.

    Steps before apply_cleaning_pipeline
    ------------------------------------
    1. Drop all target columns (a NaN target would make the row disappear).
    2. Require at least one supplied alloying element, then zero-fill every
       chemistry column of the pipeline that is absent from the query. Without
       this, apply_cleaning_pipeline fills absent columns with the training
       MEDIAN, which for elements such as Mn or Si is not "not present".
    3. Add model features that are absent from the query as NaN columns and
       record which features are absent or non-finite; the pipeline either
       drops the row (fewer than 80% of the retained image descriptors
       present) or fills them with training medians.

    Raises
    ------
    ValueError               : no alloying element supplied.
    QueryRemovedByCleaning   : the row was dropped by the cleaning rules.

    Returns X_query (one row, columns == pipeline["final_columns"]) and, if
    return_imputed is True, also the list of imputed feature names.
    """
    df_q = df_query.drop(
        columns=[c for c in TARGET_COLUMNS if c in df_query.columns]
    ).copy()

    chem_cols = list(pipeline["chemistry_cols"])
    supplied = [
        c for c in chem_cols
        if c != "Fe" and c in df_q.columns and df_q[c].notna().any()
    ]
    if not supplied:
        raise ValueError(
            f"[{name}] No alloying element was supplied in the composition; "
            "a prediction without chemistry is not meaningful."
        )
    for c in chem_cols:
        if c not in df_q.columns:
            df_q[c] = 0.0

    final_columns = list(pipeline["final_columns"])
    imputed: List[str] = []
    for c in final_columns:
        if c in chem_cols:
            continue                      # chemistry NaN is zero-filled, not imputed
        if c not in df_q.columns:
            # An absent descriptor column would make apply_cleaning_pipeline
            # raise before its 80% completeness rule can act. Add it as NaN so
            # the pipeline decides: row dropped, or value filled by the median.
            df_q[c] = np.nan
            imputed.append(c)
            continue
        values = pd.to_numeric(df_q[c], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            imputed.append(c)

    X_q, _ = apply_cleaning_pipeline(
        dataframe=df_q,
        pipeline=pipeline,
        target_col=None,
        verbose=False,
    )

    if X_q.empty:
        raise QueryRemovedByCleaning(
            f"[{name}] The query row was removed by the cleaning pipeline "
            "(um_per_pixel missing, or fewer than 80% of the retained image "
            "descriptors present)."
        )

    if imputed:
        frac = len(imputed) / max(len(final_columns), 1)
        message = (
            f"[{name}] {len(imputed)}/{len(final_columns)} features "
            f"({frac:.0%}) absent or non-finite in the query; filled with "
            f"training medians. First 10: {imputed[:10]}"
        )
        (logger.warning if frac > max_imputed_frac else logger.info)(message)

    return (X_q, imputed) if return_imputed else X_q


def fit_final_model(
    df_train_r: pd.DataFrame,
    target: str,
    n_trials: int = 100,
    n_splits_inner: int = 5,
    random_state: int = 42,
):
    """
    Tune and fit the final XGBoost model for one target on the RAW reduced
    training frame. train_final_xgb_full performs the cleaning itself
    (fold-local during tuning, full data for the final fit) and returns a
    bundle holding the model and the fitted cleaning pipeline.

    Inner CV is group-aware when 'sample_id' is available, so images of one
    sample never sit on both sides of a tuning split.

    Returns (bundle, study).
    """
    if target not in df_train_r.columns:
        raise KeyError(f"Target column '{target}' not found in the training frame.")

    X_raw = df_train_r.drop(columns=[target]).reset_index(drop=True)
    y_raw = df_train_r[target].reset_index(drop=True)

    use_groups = "sample_id" in X_raw.columns
    if not use_groups:
        logger.warning(
            "'sample_id' not found in the training frame; tuning falls back to "
            "non-grouped inner CV (images of one sample can leak between folds)."
        )

    return train_final_xgb_full(
        X=X_raw,
        y=y_raw,
        target_col=target,
        groups=use_groups,
        random_state=random_state,
        n_splits_inner=n_splits_inner,
        n_trials=n_trials,
    )


def get_training_matrix(bundle: Dict[str, Any], df_train_r: pd.DataFrame, target: str):
    """
    Cleaned training matrix and labels belonging to a fitted bundle.

    Uses bundle["X_train_clean"] / ["y_train_clean"] when train_final_xgb_full
    provides them (exact training matrix). Otherwise the fitted pipeline is
    applied to the training frame, which gives the same columns and rows.
    """
    if "X_train_clean" in bundle and "y_train_clean" in bundle:
        return bundle["X_train_clean"], bundle["y_train_clean"]

    return apply_cleaning_pipeline(
        dataframe=df_train_r.reset_index(drop=True),
        pipeline=bundle["cleaning_pipeline"],
        target_col=target,
        verbose=False,
    )


def apply_cleaning(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    config: "QueryConfig",
) -> tuple:
    """
    Fit the cleaning pipeline on the modality-filtered training set and
    apply it to the query row (standalone helper; the prediction paths use
    the pipeline stored in the fitted model bundle instead).

    Returns
    -------
    X_train, y_train, X_query, pipeline
    """
    X_train, y_train, pipeline = fit_cleaning_pipeline(
        dataframe=df_train,
        target_col=config.target_property,
        verbose=config.verbose,
        **CLEANING_KWARGS,
    )

    X_query = clean_query(
        df_query,
        pipeline,
        name=config.name or "query",
        max_imputed_frac=config.max_imputed_feature_frac,
    )

    logger.info(f"Cleaning done : X_train {X_train.shape} X_query {X_query.shape}")
    return X_train, y_train, X_query, pipeline


# ===========================================================================
# Step 8 – Feature alignment
# ===========================================================================

def align_features(
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Verify that X_query has exactly the columns of X_train, in the same order.

    The fitted cleaning pipeline already guarantees this schema. Missing
    columns are NOT zero-filled here any more: that would silently hide a
    schema problem, and absent chemistry is handled in clean_query().
    Raises RuntimeError on any mismatch.
    """
    train_cols = list(X_train.columns)
    query_cols = list(X_query.columns)

    if query_cols != train_cols:
        missing = sorted(set(train_cols) - set(query_cols))
        extra = sorted(set(query_cols) - set(train_cols))
        raise RuntimeError(
            "Query features do not match the training schema. "
            f"Missing: {missing[:10]}, extra: {extra[:10]}, "
            "or the column order differs."
        )

    if verbose:
        logger.info(f"Schema check OK: {len(train_cols)} features in train and query.")
    return X_query.copy()


def _safe_target_name(target: str) -> str:
    """Convert a target-property label to a filesystem-safe filename stem."""
    return (
        target.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )

# ===========================================================================
# Metadata helpers
# ===========================================================================

def build_observation_metadata(
    df_val: pd.DataFrame,
    image_modality: str,
    material_categories: List[str],
) -> pd.DataFrame:
    """
    Build image-level observation table from a sample-level validation
    DataFrame, filtered to the requested imaging modality and material
    categories.
    """
    df = df_val.dropna(how="all").copy()

    if "sample_id" not in df.columns:
        df["sample_id"] = range(1, len(df) + 1)

    obs_df = split_images_to_observations(df)

    obs_df["image_name"] = (
        obs_df["image_name"].astype(str)
        .str.strip()
        .str.replace(r"\.(?:png|tiff?|jpe?g)$", "", case=False, regex=True)
        .str.lower()
    )

    # Filter 1: modality subfolder
    before = len(obs_df)
    obs_df = obs_df[
        obs_df["subfolder"].astype(str).str.strip().str.upper()
        == image_modality.strip().upper()
    ]
    logger.info(
        f"Modality filter ('{image_modality}'): "
        f"{before} → {len(obs_df)} observations"
    )

    # Filter 2: material category (empty list = no filter)
    if material_categories:
        before = len(obs_df)
        obs_df = obs_df[obs_df["Category"].isin(material_categories)]
        logger.info(
            f"Category filter {material_categories}: "
            f"{before} → {len(obs_df)} observations"
        )

    if obs_df.empty:
        raise ValueError(
            f"No observations remain after filtering for "
            f"modality='{image_modality}' and "
            f"categories={material_categories}. "
            f"Check your df_val and QueryConfig."
        )

    obs_df = obs_df.set_index("image_name", drop=False)
    return obs_df


def enrich_query_with_observation(
    df_query: pd.DataFrame,
    obs_df: pd.DataFrame,
    image_name: str,
) -> pd.DataFrame:
    """
    Merge composition + metadata from obs_df into a 1-row query DataFrame.
    """
    key = image_name.lower()
    if key not in obs_df.index:
        raise ValueError(
            f"No observation metadata found for image_name='{image_name}'."
        )

    obs_row = obs_df.loc[key]
    if isinstance(obs_row, pd.DataFrame):
        raise ValueError(
            f"image_name='{image_name}' occurs {len(obs_row)} times in the "
            f"observation table; image names must be unique per modality."
        )
    df = df_query.copy()

    if "Material" in obs_row.index:
        df.loc[:, "Material"] = obs_row["Material"]

    if "Category" in obs_row.index:
        df.loc[:, "Material_Subcategory"] = obs_row["Category"]
        df.loc[:, "Category"] = obs_row["Category"]

    if "Sample" in obs_row.index:
        df.loc[:, "Sample"] = obs_row["Sample"]

    if "um_per_pixel" in obs_row.index and pd.notna(obs_row["um_per_pixel"]):
        df.loc[:, "um_per_pixel"] = float(obs_row["um_per_pixel"])

    if "method" in obs_row.index and pd.notna(obs_row["method"]):
        df.loc[:, "method"] = obs_row["method"]

    for col in ["subfolder", "DOI", "ID_name", "sample_id", "obs_id"]:
        if col in obs_row.index:
            df.loc[:, col] = obs_row[col]

    skip = {"image_name", "image_number", "resolution_um", "resolution_px"}
    handled = {
        "Material", "Category", "Sample",
        "um_per_pixel", "method",
        "subfolder", "DOI", "ID_name", "sample_id", "obs_id",
    }

    for col, val in obs_row.items():
        if col in skip or col in handled:
            continue
        df.loc[:, col] = val

    # Derive is_bse from the per-image method now that it is known
    if "method" in df.columns:
        df["is_bse"] = df["method"].apply(
            lambda m: 1 if "BSE" in str(m).upper() else 0
        )

    return df


# ===========================================================================
# Main orchestrator – single image
# ===========================================================================

def run_single_image(
    name: str,
    image_folder: str,
    df_val: pd.DataFrame,
    config: QueryConfig,
    targets: List[str] = None,
    save_bundle: bool = False,       # ← new
) -> Dict[str, Any]:
    """
    Run the full pipeline (Steps 0-9) for a single named image,
    pulling composition, scale, and modality metadata from df_val.

    Parameters
    ----------
    name         : image stem without extension, e.g. "A_rad_1"
    image_folder : absolute path to the folder containing the .png and
                   a masks/ subfolder with the _mask.npy file
    df_val       : validation DataFrame
    config       : QueryConfig — modality, training CSV, material_categories,
                   and ML settings (n_trials, n_splits_inner) are read from here
    targets      : target properties to predict; defaults to TARGETS
    save_bundle : if True, saves the full results bundle as a .pkl file
                  to config.output_dir / {name}_bundle.pkl

    Returns
    -------
    dict keyed by target property, each containing:
        y_pred   : float — predicted value
        model    : fitted sklearn Pipeline (scaler + XGBRegressor)
        bundle   : dict with model, cleaning_pipeline, feature_columns, ...
        study    : Optuna study object
        X_train  : pd.DataFrame — cleaned training feature matrix
        y_train  : pd.Series   — training labels
        X_query  : pd.DataFrame — aligned query feature vector (1 row)

    """
    if targets is None:
        targets = TARGETS

    folder_label = Path(image_folder).name   # "OM" or "SEM"

    obs_df = build_observation_metadata(
        df_val,
        image_modality=folder_label,
        material_categories=config.material_categories,
    )

    fit_res = fit_feature_pipeline(config)
    df_train_r   = fit_res["df_train_reduced"]

    img_path  = Path(image_folder) / f"{name}.png"
    mask_path = Path(image_folder) / "masks" / f"{name}_mask.npy"

    image = load_image(str(img_path))
    mask  = load_mask(str(mask_path))
    check_mask_shape(image, mask)

    descriptors = extract_descriptors(image, mask, name,
                                      use_gram=config.use_gram,
                                      gram_device=config.gram_device,
                                      verbose=config.verbose)

    df_query = build_query_dataframe(descriptors, config, strict_scale=False, image_name=name)
    df_query = enrich_query_with_observation(df_query, obs_df, name)  # ← Excel lookup

    df_query_r = reduce_features(df_query, label=name)

    results = {}
    for target in targets:
        config.target_property = target

        fitted, study = fit_final_model(
            df_train_r,
            target,
            n_trials=config.n_trials,
            n_splits_inner=config.n_splits_inner,
        )
        model = fitted["model"]
        pipeline = fitted["cleaning_pipeline"]

        X_train, y_train = get_training_matrix(fitted, df_train_r, target)

        X_q = clean_query(
            df_query_r,
            pipeline,
            name=name,
            max_imputed_frac=config.max_imputed_feature_frac,
        )
        X_q_aligned = align_features(X_train, X_q, verbose=False)
        X_q_aligned.index = [name]

        y_pred = model.predict(X_q_aligned)
        results[target] = {
            "y_pred": float(y_pred[0]),
            "model": model,
            "bundle": fitted,
            "study": study,
            "X_train": X_train,
            "y_train": y_train,
            "X_query": X_q_aligned,
        }

    if save_bundle:
        os.makedirs(config.output_dir, exist_ok=True)
        bundle = {
            "results":       results,
            "df_validation": df_val,
        }
        bundle_path = os.path.join(config.output_dir, f"{name}_bundle.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"Bundle saved → {bundle_path}")

    return results


# ===========================================================================
# Batch helpers – folder-level feature pipeline
# ===========================================================================

def fit_feature_pipeline(
    config: QueryConfig,
    target_properties: List[str] = None,
) -> Dict[str, Any]:
    if target_properties is None:
        target_properties = [config.target_property]

    df_train_full = pd.read_csv(config.training_dataset_csv)
    logger.info(
        f"Training CSV : {df_train_full.shape[0]} samples × "
        f"{df_train_full.shape[1]} columns "
        f"[{Path(config.training_dataset_csv).name}]"
    )

    df_train = filter_training_data(
        df_train_full,
        image_modality=config.image_modality,
        material_categories=config.material_categories,
        verbose=config.verbose,
    )

    df_train_r = reduce_features(df_train, label="training")
    if config.verbose:
        logger.info(f"After reduction : train {df_train_r.shape}")

    return {"df_train_reduced": df_train_r}


def transform_single_image(
    image_path: str,
    mask_path: Optional[str],
    config: QueryConfig,
    pipeline: Dict[str, Any],
    X_train: pd.DataFrame,
    obs_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Apply the fitted feature pipeline to one image using observation-level
    metadata from ``obs_df``.

    The transformation order is:
        1. Load image and exclusion mask
        2. Extract raw descriptors
        3. Enrich metadata from the observation table
        4. Deterministic descriptor reduction (morphology collapse, Gram panel)
        5. Apply the fitted cleaning pipeline
        8. Align query features with the training feature matrix

    Parameters
    ----------
    image_path : str
        Path to the input micrograph.
    mask_path : str or None
        Path to the corresponding ``.npy`` exclusion mask. If None, all
        image pixels are retained.
    config : QueryConfig
        Pipeline configuration.
    pipeline : dict
        Fitted cleaning pipeline returned by ``fit_cleaning_pipeline``.
    X_train : pandas.DataFrame
        Final cleaned training feature matrix. Its column order defines the
        required query-feature schema.
    obs_df : pandas.DataFrame
        Image-level metadata table indexed by lowercase image stem.

    Returns
    -------
    dict
        Dictionary containing:
        - ``X_query_aligned``: final one-row model input matrix.
        - ``df_query_raw``: descriptor and metadata row before reduction,
          cleaning, and alignment.
    """
    image_name = Path(image_path).stem

    image = load_image(image_path)
    mask = load_mask(mask_path)

    check_mask_shape(image, mask)

    descriptors = extract_descriptors(
        image=image,
        mask=mask,
        image_name=image_name,
        use_gram=config.use_gram,
        gram_device=config.gram_device,
        verbose=config.verbose,
    )

    df_query = build_query_dataframe(
        descriptors,
        config,
        strict_scale=False,
        image_name=image_name,
    )
    df_query = enrich_query_with_observation(
        df_query,
        obs_df,
        image_name,
    )

    df_query_r = reduce_features(df_query, label=image_name)

    X_query = clean_query(
        df_query_r,
        pipeline,
        name=image_name,
        max_imputed_frac=config.max_imputed_feature_frac,
    )

    X_query_aligned = align_features(
        X_train,
        X_query,
        verbose=False,
    )
    X_query_aligned.index = [image_name]

    return {
        "X_query_aligned": X_query_aligned,
        "df_query_raw": df_query,
    }


def validation_predictions(
    batch: dict,
    target: str = "UTS (MPa)",
    n_trials: int = 100,
    n_splits_inner: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Train the final XGB model on the raw reduced training frame of a batch
    and predict every query image with the cleaning pipeline of that model.

    Parameters
    ----------
    batch         : output dict from process_folder()
    target        : target property column name
    n_trials      : Optuna trials (default 100; use 20 for quick runs)
    n_splits_inner: inner CV folds (default 5; use 3 for quick runs)

    Returns
    -------
    model, study, y_pred (pd.Series indexed by validation image names), bundle
    """
    df_train_r = batch["df_train_raw"]
    queries = batch["query_rows_reduced"]
    max_frac = batch.get("max_imputed_feature_frac", 0.20)

    bundle, study = fit_final_model(
        df_train_r,
        target,
        n_trials=n_trials,
        n_splits_inner=n_splits_inner,
        random_state=random_state,
    )
    pipeline = bundle["cleaning_pipeline"]

    rows_X: List[pd.DataFrame] = []
    for name, df_query_r in queries:
        try:
            X_q = clean_query(
                df_query_r, pipeline, name=name, max_imputed_frac=max_frac
            )
        except QueryRemovedByCleaning as exc:
            logger.warning(f"{exc} Skipped.")
            continue
        X_q.index = [name]
        rows_X.append(X_q)

    if not rows_X:
        raise RuntimeError(f"No query image survived cleaning for target '{target}'.")

    X_val = pd.concat(rows_X)
    y_pred = bundle["model"].predict(X_val)
    y_pred_ser = pd.Series(y_pred, index=X_val.index, name=f"y_pred_{target}")
    return bundle["model"], study, y_pred_ser, bundle


def run_ml_validation(
    all_batch_outputs: Dict[str, dict],
    out_dir: str,
    targets: List[str] = None,
    n_trials: int = 100,
    n_splits_inner: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Step 9: train one XGB model per (modality, target) combination and
    save models, studies, and prediction CSVs.

    Parameters
    ----------
    all_batch_outputs : {modality_name: batch_output} from process_folder()
    out_dir           : directory for models + prediction CSVs
    targets           : list of target columns; defaults to TARGETS
    n_trials          : Optuna trials
    n_splits_inner    : inner CV folds

    Returns
    -------
    results : {modality: {target: {"model", "study", "y_pred"}}}
    """
    if targets is None:
        targets = TARGETS

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Step 9 – ML validation. Output dir: {out_dir}")

    results: Dict[str, Dict[str, Any]] = {}

    for modality_name, batch in all_batch_outputs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 9 | Modality: {modality_name}")
        logger.info(f"{'='*60}")

        results[modality_name] = {}

        for target in targets:
            logger.info(f"  → Training XGB for target: {target}")

            model, study, y_pred, bundle = validation_predictions(
                batch,
                target=target,
                n_trials=n_trials,
                n_splits_inner=n_splits_inner,
            )

            results[modality_name][target] = {
                "model":  model,
                "bundle": bundle,
                "study":  study,
                "y_pred": y_pred,
            }

            safe_t = _safe_target_name(target)

            # Save model, study and fitted bundle (model + cleaning pipeline) as pickles
            model_pkl = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_model.pkl")
            study_pkl = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_study.pkl")
            with open(model_pkl, "wb") as f_out:
                pickle.dump(model, f_out)
            with open(study_pkl, "wb") as f_out:
                pickle.dump(study, f_out)
            bundle_pkl = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_bundle.pkl")
            with open(bundle_pkl, "wb") as f_out:
                pickle.dump(bundle, f_out)

            # Save predictions as CSV
            pred_csv = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_predictions.csv")
            y_pred.to_csv(pred_csv, header=True)

            logger.info(f"    Saved model, study and predictions for {target}.")

    # Bundle all results in one pickle
    bundle_path = os.path.join(out_dir, "all_xgb_validation_results.pkl")
    with open(bundle_path, "wb") as f_out:
        pickle.dump(results, f_out)
    logger.info("✓ All models and predictions saved.")

    return results


# ===========================================================================
# Batch folder pipeline (Steps 0-8 for a whole image folder)
# ===========================================================================

def process_folder(
    config: QueryConfig,
    df_val: pd.DataFrame,
    target_properties: List[str] = None,
) -> Dict[str, Any]:
    """
    Run Steps 0-8 for every image in config.image_folder.

    target_properties : list of target columns to produce X/y sets for.
                        Defaults to [config.target_property].

    Returns
    -------
    batch_output dict with keys:
        results_by_target, df_query_raw_batch, df_train_raw, query_rows_reduced
    """
    if target_properties is None:
        target_properties = [config.target_property]

    # 0) Observation metadata – use disk subfolder label, not training method
    folder_label = Path(config.image_folder).name   # "OM" or "SEM"

    obs_df = build_observation_metadata(
        df_val,
        image_modality=folder_label,
        material_categories=config.material_categories,
    )

    # 1) Reduce the training frame ONCE (deterministic, nothing fitted)
    fit_res      = fit_feature_pipeline(config)
    df_train_r   = fit_res["df_train_reduced"]

    image_folder = Path(config.image_folder)
    out_dir      = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Extract descriptors + reduce, once per image
    rows_reduced: List[tuple] = []
    rows_raw:     List[pd.DataFrame] = []

    image_files = sorted(image_folder.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No .png images found in {image_folder}")

    for img_path in image_files:
        name      = img_path.stem
        mask_path = image_folder / "masks" / f"{name}_mask.npy"

        logger.info(f"Extracting descriptors: {img_path.name}")

        image = load_image(str(img_path))
        mask  = load_mask(str(mask_path))
        check_mask_shape(image, mask)

        descriptors = extract_descriptors(
            image=image, mask=mask, image_name=name,
            use_gram=config.use_gram, gram_device=config.gram_device,
            verbose=config.verbose,
        )

        df_query = build_query_dataframe(descriptors, config, strict_scale=False, image_name=name)
        df_query = enrich_query_with_observation(df_query, obs_df, name)

        df_query_r = reduce_features(df_query, label=name)

        rows_reduced.append((name, df_query_r))
        rows_raw.append(df_query)

    df_query_raw_batch = pd.concat(rows_raw, ignore_index=True)

    # 3) Per-target: cleaning + schema check
    #    (inspection copy; the prediction step re-cleans the query rows with
    #    the pipeline stored in the final model bundle)
    results_by_target: Dict[str, Any] = {}

    for target in target_properties:
        logger.info(f"Cleaning pipeline for target: {target}")

        config.target_property = target

        X_train, y_train, pipeline = fit_cleaning_pipeline(
            dataframe=df_train_r,
            target_col=target,
            verbose=config.verbose,
            **CLEANING_KWARGS,
        )

        rows_X: List[pd.DataFrame] = []
        imputed_by_image: Dict[str, List[str]] = {}
        for name, df_query_r in rows_reduced:
            try:
                X_q, imputed = clean_query(
                    df_query_r,
                    pipeline,
                    name=name,
                    max_imputed_frac=config.max_imputed_feature_frac,
                    return_imputed=True,
                )
            except QueryRemovedByCleaning as exc:
                logger.warning(f"{exc} Skipped.")
                continue

            X_q_aligned = align_features(X_train, X_q, verbose=False)
            X_q_aligned.index = [name]
            rows_X.append(X_q_aligned)
            imputed_by_image[name] = imputed

        if not rows_X:
            logger.warning(f"No query rows survived cleaning for target '{target}'.")
            continue

        X_query_batch = pd.concat(rows_X)

        safe_t = _safe_target_name(target)
        X_train.to_csv(out_dir / f"X_train_{safe_t}.csv")
        y_train.to_csv(out_dir / f"y_train_{safe_t}.csv")
        X_query_batch.to_csv(out_dir / f"X_query_batch_{safe_t}.csv")
        pd.Series(
            {n: len(v) for n, v in imputed_by_image.items()},
            name="n_imputed_features",
        ).to_csv(out_dir / f"query_imputed_feature_counts_{safe_t}.csv")

        results_by_target[target] = {
            "X_train":      X_train,
            "y_train":      y_train,
            "X_query_batch": X_query_batch,
            "pipeline":     pipeline,
            "query_imputed_features": imputed_by_image,
        }

    df_query_raw_batch.to_csv(out_dir / "query_descriptors_batch.csv", index=False)

    batch_output = {
        "results_by_target":  results_by_target,
        "df_query_raw_batch": df_query_raw_batch,
        "df_train_raw":       fit_res["df_train_reduced"],
        "query_rows_reduced": rows_reduced,
        "max_imputed_feature_frac": config.max_imputed_feature_frac,
    }
    with open(out_dir / "batch_prediction_output.pkl", "wb") as f:
        pickle.dump(batch_output, f)

    return batch_output

# ===========================================================================
# __main__  – Steps 0-8 (process_folder) + Step 9 (ML validation)
# ===========================================================================

if __name__ == "__main__":
    from datetime import datetime

    print(MODULE_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_validation = pd.read_excel(
        os.path.join(
            MODULE_PATH, "data", "validation", "csp_validation.xlsx",
        )
    )

    # "SEM" aggregates SEM: SE + SEM: BSE from the training CSV.
    modalities = ["OM", "SEM"]
    # modalities = ["OM"]

    folder_map = {
        "OM":  "OM",
        "SEM": "SEM",
    }

    all_batch_outputs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Steps 0-8: descriptor extraction + feature pipeline for each modality
    # ------------------------------------------------------------------
    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing modality: {modality}  (Steps 0-8)")
        print(f"{'='*60}")

        image_folder = os.path.join(
            MODULE_PATH, "data", "validation",
            "processed_images", f'{modality}',
        )

        config = QueryConfig(
            image_folder=image_folder,
            image_modality=modality,       # "OM" or "SEM" (SEM → SE+BSE)
            um_per_pixel=None,             # per-image from df_validation
            material="Steel",
            material_subcategory="",       # per-image from df_validation
            material_categories=[
                "Ferritic Stainless Steel",
                "Duplex Stainless Steel",
                "Austenitic Stainless Steel",
            ],
            composition={},                # per-image from df_validation
            use_gram=True,
            gram_device=None,
            training_dataset_csv=os.path.join(
                MODULE_PATH, "data", "descriptors_full.csv"
            ),
            output_timestamp=timestamp,
            verbose=True,
            # --- Step 9 options ---
            run_ml=True,
            n_trials=100,        # reduce to 20 for quick validation runs
            n_splits_inner=5,    # reduce to 3 for quick validation runs
        )

        print(f"Image folder: {config.image_folder}")
        print(f"Train CSV  : {config.training_dataset_csv}")
        print(f"Output dir : {config.output_dir}")
                       
        batch_output = process_folder(
            config,
            df_validation,
            target_properties=TARGETS,
        )

        all_batch_outputs[modality] = batch_output

    # ------------------------------------------------------------------
    # Step 9: XGBoost model training + predictions, one per (modality, target)
    # # ------------------------------------------------------------------
    if any(True for _ in all_batch_outputs):   # at least one modality ran
        print(f"\n{'='*60}")
        print("Step 9 – ML model training and validation")
        print(f"{'='*60}")

        ml_out_dir = os.path.join(
            MODULE_PATH, "data", "validation",
            "predictions", f"models_{timestamp}",
        )

        ml_results = run_ml_validation(
            all_batch_outputs=all_batch_outputs,
            out_dir=ml_out_dir,
            targets=TARGETS,
            n_trials=config.n_trials,
            n_splits_inner=config.n_splits_inner,
        )

        print(f"\n✓ All done. Models and predictions saved to:\n  {ml_out_dir}")
    
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #                         Single Image
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    
    # config_single = QueryConfig(
    # image_folder=os.path.join(
    #     MODULE_PATH, "data", "validation", "processed_images", "SEM"
    # ),
    # image_modality="SEM",          # aggregates SEM: SE + SEM: BSE from training CSV
    # um_per_pixel=None,             # read from df_validation via enrich_query_with_observation
    # material="Steel",
    # material_subcategory="",       # overwritten per-image from df_validation
    # material_categories=[
    #     "Ferritic Stainless Steel",
    #     "Duplex Stainless Steel",
    #     "Austenitic Stainless Steel",
    # ],
    # use_gram=True,
    # training_dataset_csv=os.path.join(
    #     MODULE_PATH, "data","descriptors_full.csv"
    # ),
    # gram_device=None,     
    # verbose=True,
    # run_ml=True,
    # n_trials=100,
    # n_splits_inner=5,
    # )
    
    # results_single = run_single_image(
    #     name="12_BSE",            # stem only, no extension
    #     image_folder=config_single.image_folder,
    #     df_val=df_validation,
    #     config=config_single,
    #     targets=["UTS (MPa)", "YS (MPa)"],
    #     save_bundle=True, 
    # )
    
    # print({t: results_single[t]["y_pred"] for t in results_single})