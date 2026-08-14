# -*- coding: utf-8 -*-
"""
Predict Material Properties from a Single Micrograph
=====================================================
Author : Gerfried Millner
Version: 1.1.0
Date   : 14.08.2026
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
Step 4  – Unify morphology
Step 5  – Physical units
Step 6  – Feature reduction
Step 7  – Cleaning pipeline
Step 8  – Feature alignment
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
from data_processing.combine_data import (
    unify_morphology_for_ml,
    add_physical_units_to_features,
    split_images_to_observations,
)
from data_processing.feat_reducer import (
    SmartGramReducer,
    GLCMReducer,
    LBPReducer,
)
from data_processing.clean_data import (
    fit_cleaning_pipeline,
    apply_cleaning_pipeline,
)

from machine_learning.models_paper import train_final_xgb_full

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
        Metadata/bookkeeping columns excluded from feature matrices.
        Defaults cover the full set used across all existing scripts.

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

    Returns
    -------
    pd.DataFrame (1 row)
    """
    um_per_pixel = _resolve_um_per_pixel(config, strict=strict_scale)
    image_stem = Path(config.image_path).stem if config.image_path else ""
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
    # here and are zero-filled later by align_features() against X_train,
    # which is equivalent to "treated as 0 wt-%".
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
# Step 4 – Unify morphology
# ===========================================================================

def unify_morphology(
    df_query: pd.DataFrame,
    df_train: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Collapse per-method morphology columns into the canonical morph_* block
    using unify_morphology_for_ml() from combine_data.

    Parameters
    ----------
    df_query : pd.DataFrame
    df_train : pd.DataFrame
    verbose : bool

    Returns
    -------
    (df_query_unified, df_train_unified)
    """
    df_train_u = unify_morphology_for_ml(df_train, drop_per_method=True)
    df_query_u = unify_morphology_for_ml(df_query, drop_per_method=True)

    if verbose:
        logger.info(
            f"Morphology : unified — "
            f"train {df_train_u.shape}, query {df_query_u.shape}"
        )
    return df_query_u, df_train_u


# ===========================================================================
# Step 5 – Physical unit conversion
# ===========================================================================

def convert_physical_units(
    df_query: pd.DataFrame,
    df_train: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Convert pixel-scale features to µm / µm² using 'um_per_pixel'.

    Parameters
    ----------
    df_query : pd.DataFrame
    df_train : pd.DataFrame
    verbose : bool

    Returns
    -------
    (df_query_physical, df_train_physical)
    """
    def _convert_safe(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if "um_per_pixel" in df.columns and df["um_per_pixel"].notna().any():
            return add_physical_units_to_features(df)
        logger.warning(
            f"{label}: 'um_per_pixel' absent or all-NaN — "
            f"physical unit conversion skipped."
        )
        return df.copy()

    df_train_p = _convert_safe(df_train, "Training")
    df_query_p = _convert_safe(df_query, "Query")

    if verbose:
        logger.info("Physical units: conversion complete.")
    return df_query_p, df_train_p


# ===========================================================================
# Step 6 – Feature reduction
# ===========================================================================

def apply_feature_reduction(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Fit SmartGramReducer and GLCMReducer on the training set, then
    transform both training set and query rows.

    Returns
    -------
    df_train_reduced, df_query_reduced, gram_reducer, glcm_reducer
    """
    GRAM_PREFIXES = (
        "conv1_", "layer1_", "layer2_", "layer3_", "layer4_",
        "cross_", "multi_scale", "global_",
    )

    gram_cols_train = [
        c for c in df_train.columns
        if any(c.startswith(p) for p in GRAM_PREFIXES)
    ]

    has_query = df_query is not None and not df_query.empty

    # --- Gram ---
    if gram_cols_train:
        logger.info(f"SmartGramReducer: fitting on {len(gram_cols_train)} Gram columns …")
        gram_reducer = SmartGramReducer()
        gram_reducer.fit(df_train, verbose=verbose)
        df_train_g = gram_reducer.transform(df_train, verbose=False)
        df_query_g = gram_reducer.transform(df_query, verbose=False) if has_query else df_query.copy()
    else:
        logger.warning("No Gram columns found — SmartGramReducer skipped.")
        gram_reducer = None
        df_train_g = df_train.copy()
        df_query_g = df_query.copy()

    # --- GLCM ---
    glcm_cols_post = [c for c in df_train_g.columns if c.startswith("glcm_")]
    if glcm_cols_post:
        logger.info(f"GLCMReducer : fitting on {len(glcm_cols_post)} GLCM columns …")
        glcm_reducer = GLCMReducer()
        glcm_reducer.fit(df_train_g, verbose=verbose)
        df_train_r = glcm_reducer.transform(df_train_g, verbose=False)
        df_query_r = glcm_reducer.transform(df_query_g, verbose=False) if has_query else df_query_g.copy()
    else:
        logger.warning("No GLCM columns found — GLCMReducer skipped.")
        glcm_reducer = None
        df_train_r = df_train_g.copy()
        df_query_r = df_query_g.copy()

        # --- LBP ---
    lbp_cols_post = [c for c in df_train_r.columns if c.startswith("lbp_")]
    if lbp_cols_post:
        logger.info(f"LBPReducer  : fitting on {len(lbp_cols_post)} LBP columns …")
        lbp_reducer = LBPReducer()
        lbp_reducer.fit(df_train_r, verbose=verbose)
        df_train_r = lbp_reducer.transform(df_train_r, verbose=False)
        df_query_r = lbp_reducer.transform(df_query_r, verbose=False) if has_query else df_query_r
    else:
        logger.warning("No LBP columns found — LBPReducer skipped.")
        lbp_reducer = None
    
    if verbose:
        logger.info(f"After reduction : train {df_train_r.shape} query {df_query_r.shape}")
    
    return df_train_r, df_query_r, gram_reducer, glcm_reducer, lbp_reducer

# ===========================================================================
# Step 7 – Cleaning pipeline
# ===========================================================================

def apply_cleaning(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    config: "QueryConfig",
) -> tuple:
    """
    Fit the cleaning pipeline on the modality-filtered training set and
    apply it to the query row.

    Returns
    -------
    X_train, y_train, X_query, pipeline
    """
    id_cols_present = [c for c in config.id_cols if c in df_train.columns]

    X_train, y_train, pipeline = fit_cleaning_pipeline(
        df=df_train,
        target_col=config.target_property,
        id_cols=id_cols_present,
        impute_method="category",
        handle_inf="clip",
        verbose=config.verbose,
    )

    X_query, _ = apply_cleaning_pipeline(
        df=df_query,
        pipeline=pipeline,
        target_col=None,
        verbose=False,
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
    Ensure X_query has exactly the same columns as X_train, in the same order.

    - Columns present in X_train but absent from X_query → zero-filled.
    - Columns present in X_query but absent from X_train → dropped silently.

    Returns
    -------
    X_query_aligned : pd.DataFrame
    """
    train_cols = set(X_train.columns)
    query_cols = set(X_query.columns)

    missing = train_cols - query_cols
    extra   = query_cols - train_cols

    if missing:
        logger.warning(
            f"Feature alignment: {len(missing)} column(s) absent from query "
            f"→ zero-filled. First 10: {sorted(missing)[:10]}"
        )
        X_query = X_query.copy()
        for col in missing:
            X_query[col] = 0.0

    if extra and verbose:
        logger.info(
            f"Feature alignment: {len(extra)} query column(s) not in "
            f"training set → dropped."
        )
    X_query = X_query.drop(columns=list(extra), errors="ignore")
    X_query_aligned = X_query[X_train.columns]

    status = "✓ exact match" if (not missing and not extra) else "⚠ padded/trimmed"
    logger.info(
        f"Alignment [{status}]: "
        f"{X_train.shape[1]} features in both train and query."
    )
    return X_query_aligned


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
        .str.replace(".png",  "", case=False)
        .str.replace(".tif",  "", case=False)
        .str.replace(".tiff", "", case=False)
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

    # Filter 2: material category
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
    gram_reducer = fit_res["gram_reducer"]
    glcm_reducer = fit_res["glcm_reducer"]
    lbp_reducer  = fit_res["lbp_reducer"]  
    df_train_r   = fit_res["df_train_reduced"]

    img_path  = Path(image_folder) / f"{name}.png"
    mask_path = Path(image_folder) / "masks" / f"{name}_mask.npy"

    image = load_image(str(img_path))
    mask  = load_mask(str(mask_path))

    descriptors = extract_descriptors(image, mask, name,
                                      use_gram=config.use_gram,
                                      gram_device=config.gram_device,
                                      verbose=config.verbose)

    df_query = build_query_dataframe(descriptors, config, strict_scale=False)
    df_query = enrich_query_with_observation(df_query, obs_df, name)  # ← Excel lookup

    df_query_u = unify_morphology_for_ml(df_query, drop_per_method=True)
    if "um_per_pixel" in df_query_u.columns and df_query_u["um_per_pixel"].notna().any():
        df_query_p = add_physical_units_to_features(df_query_u)
    else:
        df_query_p = df_query_u.copy()

    df_query_g = gram_reducer.transform(df_query_p, verbose=False) if gram_reducer else df_query_p
    df_query_r = glcm_reducer.transform(df_query_g, verbose=False) if glcm_reducer else df_query_g
    df_query_r = lbp_reducer.transform(df_query_r, verbose=False) if lbp_reducer else df_query_r 

    results = {}
    for target in targets:
        config.target_property = target
        id_cols_present = [c for c in config.id_cols if c in df_train_r.columns]
        X_train, y_train, pipeline = fit_cleaning_pipeline(
            df=df_train_r, target_col=target,
            id_cols=id_cols_present, impute_method="category",
            handle_inf="clip", verbose=config.verbose,
        )
        df_q = df_query_r.drop(columns=[target], errors="ignore")
        X_q, _ = apply_cleaning_pipeline(df=df_q, pipeline=pipeline,
                                          target_col=None, verbose=False)
        X_q_aligned = align_features(X_train, X_q, verbose=False)
        X_q_aligned.index = [name]

        model, study = train_final_xgb_full(X_train, y_train, groups=False,
                                             n_trials=config.n_trials,
                                             n_splits_inner=config.n_splits_inner)
        y_pred = model.predict(X_q_aligned)
        results[target] = {"y_pred": float(y_pred[0]), "model": model, "study": study,
                           "X_train": X_train, "y_train": y_train, "X_query": X_q_aligned}
    
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

    _, df_train_u = unify_morphology(pd.DataFrame(), df_train, verbose=False)
    _, df_train_p = convert_physical_units(pd.DataFrame(), df_train_u, verbose=config.verbose)

    df_train_r, _, gram_reducer, glcm_reducer, lbp_reducer = apply_feature_reduction(
        df_train_p, pd.DataFrame(), verbose=config.verbose
    )

    return {
        "df_train_reduced": df_train_r,
        "gram_reducer":     gram_reducer,
        "glcm_reducer":     glcm_reducer,
        "lbp_reducer":      lbp_reducer,   #
    }

def transform_single_image(
    image_path: str,
    mask_path: Optional[str],
    config: QueryConfig,
    gram_reducer,
    glcm_reducer,
    lbp_reducer,
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
        4. Unify morphology descriptors
        5. Convert pixel-scale descriptors to physical units
        6. Apply fitted Gram, GLCM, and LBP reducers
        7. Apply the fitted cleaning pipeline
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
    gram_reducer : SmartGramReducer or None
        Fitted Gram-feature reducer.
    glcm_reducer : GLCMReducer or None
        Fitted GLCM-feature reducer.
    lbp_reducer : LBPReducer or None
        Fitted LBP-feature reducer.
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
        - ``df_query_raw``: descriptor and metadata row before morphology,
          physical-unit conversion, reduction, cleaning, and alignment.
    """
    image_name = Path(image_path).stem

    image = load_image(image_path)
    mask = load_mask(mask_path)

    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape} vs mask {mask.shape}. "
            "Ensure the mask was generated from the same image."
        )

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
    )
    df_query = enrich_query_with_observation(
        df_query,
        obs_df,
        image_name,
    )

    df_query_u = unify_morphology_for_ml(
        df_query,
        drop_per_method=True,
    )

    if (
        "um_per_pixel" in df_query_u.columns
        and df_query_u["um_per_pixel"].notna().any()
    ):
        df_query_p = add_physical_units_to_features(df_query_u)
    else:
        logger.warning(
            f"[{image_name}] 'um_per_pixel' absent or NaN; "
            "physical-unit conversion skipped."
        )
        df_query_p = df_query_u.copy()

    df_query_g = (
        gram_reducer.transform(df_query_p, verbose=False)
        if gram_reducer is not None
        else df_query_p.copy()
    )

    df_query_r = (
        glcm_reducer.transform(df_query_g, verbose=False)
        if glcm_reducer is not None
        else df_query_g.copy()
    )

    df_query_r = (
        lbp_reducer.transform(df_query_r, verbose=False)
        if lbp_reducer is not None
        else df_query_r.copy()
    )

    df_query_for_cleaning = df_query_r.drop(
        columns=[config.target_property],
        errors="ignore",
    )

    X_query, _ = apply_cleaning_pipeline(
        df=df_query_for_cleaning,
        pipeline=pipeline,
        target_col=None,
        verbose=False,
    )

    if X_query.empty:
        raise ValueError(
            f"[{image_name}] Query row is empty after applying the "
            "cleaning pipeline."
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
) -> tuple:
    """
    Train final XGB on batch X_train / y_train and predict on X_query_batch.

    Parameters
    ----------
    batch         : output dict from process_folder()
    target        : target property column name
    n_trials      : Optuna trials (default 100; use 20 for quick runs)
    n_splits_inner: inner CV folds (default 5; use 3 for quick runs)

    Returns
    -------
    model, study, y_pred (pd.Series indexed by validation image names)
    """
    res_t   = batch["results_by_target"][target]
    X_train = res_t["X_train"]
    y_train = res_t["y_train"]
    X_val   = res_t["X_query_batch"]

    model, study = train_final_xgb_full(
        X_train,
        y_train,
        groups=False,
        n_trials=n_trials,
        n_splits_inner=n_splits_inner,
    )

    y_pred = model.predict(X_val)
    y_pred_ser = pd.Series(y_pred, index=X_val.index, name=f"y_pred_{target}")
    return model, study, y_pred_ser


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

            model, study, y_pred = validation_predictions(
                batch,
                target=target,
                n_trials=n_trials,
                n_splits_inner=n_splits_inner,
            )

            results[modality_name][target] = {
                "model":  model,
                "study":  study,
                "y_pred": y_pred,
            }

            safe_t = _safe_target_name(target)

            # Save model + study as pickles
            model_pkl = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_model.pkl")
            study_pkl = os.path.join(out_dir, f"xgb_{modality_name}_{safe_t}_study.pkl")
            with open(model_pkl, "wb") as f_out:
                pickle.dump(model, f_out)
            with open(study_pkl, "wb") as f_out:
                pickle.dump(study, f_out)

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
        results_by_target, df_query_raw_batch, df_train_raw
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

    # 1) Fit feature reducers ONCE on training set
    fit_res      = fit_feature_pipeline(config)
    gram_reducer = fit_res["gram_reducer"]
    glcm_reducer = fit_res["glcm_reducer"]
    lbp_reducer  = fit_res["lbp_reducer"]
    df_train_r   = fit_res["df_train_reduced"]

    image_folder = Path(config.image_folder)
    out_dir      = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Extract descriptors + apply reducers ONCE per image
    rows_reduced: List[tuple] = []
    rows_raw:     List[pd.DataFrame] = []

    for img_path in sorted(image_folder.glob("*.png")):
        name      = img_path.stem
        mask_path = image_folder / "masks" / f"{name}_mask.npy"

        logger.info(f"Extracting descriptors: {img_path.name}")

        image = load_image(str(img_path))
        mask  = load_mask(str(mask_path))

        descriptors = extract_descriptors(
            image=image, mask=mask, image_name=name,
            use_gram=config.use_gram, gram_device=config.gram_device,
            verbose=config.verbose,
        )

        df_query = build_query_dataframe(descriptors, config, strict_scale=False)
        df_query = enrich_query_with_observation(df_query, obs_df, name)

        df_query_u = unify_morphology_for_ml(df_query, drop_per_method=True)
        if "um_per_pixel" in df_query_u.columns and df_query_u["um_per_pixel"].notna().any():
            df_query_p = add_physical_units_to_features(df_query_u)
        else:
            df_query_p = df_query_u.copy()

        df_query_g = gram_reducer.transform(df_query_p, verbose=False) if gram_reducer is not None else df_query_p.copy()
        df_query_r = glcm_reducer.transform(df_query_g, verbose=False) if glcm_reducer is not None else df_query_g.copy()
        df_query_r = lbp_reducer.transform(df_query_r, verbose=False) if lbp_reducer is not None else df_query_r.copy()

        rows_reduced.append((name, df_query_r))
        rows_raw.append(df_query)

    df_query_raw_batch = pd.concat(rows_raw, ignore_index=True)

    # 3) Per-target: cleaning + alignment
    results_by_target: Dict[str, Any] = {}

    for target in target_properties:
        logger.info(f"Cleaning pipeline for target: {target}")

        config.target_property = target

        id_cols_present = [c for c in config.id_cols if c in df_train_r.columns]
        X_train, y_train, pipeline = fit_cleaning_pipeline(
            df=df_train_r,
            target_col=target,
            id_cols=id_cols_present,
            impute_method="category",
            handle_inf="clip",
            verbose=config.verbose,
        )

        rows_X: List[pd.DataFrame] = []
        for name, df_query_r in rows_reduced:
            df_q = df_query_r.copy()

            if target in df_q.columns:
                df_q = df_q.drop(columns=[target])

            X_q, _ = apply_cleaning_pipeline(
                df=df_q, pipeline=pipeline, target_col=None, verbose=False,
            )

            if X_q.empty:
                logger.warning(
                    f"[{name}] Skipping: query row empty after cleaning "
                    f"(df_query_r shape={df_query_r.shape})."
                )
                continue

            X_q_aligned = align_features(X_train, X_q, verbose=False)
            X_q_aligned.index = [name]
            rows_X.append(X_q_aligned)

        if not rows_X:
            logger.warning(f"No query rows survived cleaning for target '{target}'.")
            continue

        X_query_batch = pd.concat(rows_X)

        safe_t = _safe_target_name(target)
        X_train.to_csv(out_dir / f"X_train_{safe_t}.csv")
        y_train.to_csv(out_dir / f"y_train_{safe_t}.csv")
        X_query_batch.to_csv(out_dir / f"X_query_batch_{safe_t}.csv")

        results_by_target[target] = {
            "X_train":      X_train,
            "y_train":      y_train,
            "X_query_batch": X_query_batch,
            "pipeline":     pipeline,
        }

    df_query_raw_batch.to_csv(out_dir / "query_descriptors_batch.csv", index=False)

    batch_output = {
        "results_by_target":  results_by_target,
        "df_query_raw_batch": df_query_raw_batch,
        "df_train_raw":       fit_res["df_train_reduced"],
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
        print(f"Image folder: {config.image_folder}")
                       
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
    
    

            
        
        
