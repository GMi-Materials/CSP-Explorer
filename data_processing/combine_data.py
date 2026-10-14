# -*- coding: utf-8 -*-
"""
Author: Gerfried Millner
Version: 1.0.0
Date: 31.03.2026
Email: gerfried.millner@ntu.edu.sg
Status: Deliver
"""

import pandas as pd
import numpy as np
import os, re

# ---------------------------------------------------
# Normalizers
# ---------------------------------------------------
def _normalize_doi(s):
    """Normalize DOI: strip URL/‘doi:’ prefixes, lowercase; return None for NaN."""
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r'^(https?://(dx\.)?doi\.org/)+', '', s)
    s = re.sub(r'^(doi:)\s*', '', s)
    return s.strip()

def _normalize_filename(s):
    """Return lowercase filename stem (no path/extension); None for NaN."""
    if pd.isna(s):
        return None
    fname = os.path.basename(str(s)).strip().lower()
    return os.path.splitext(fname)[0]

# ---------------------------------------------------
# Step 1: Load literature data from Excel
# ---------------------------------------------------

def load_literature_data(
    excel_file: str,
    sheet_name: str = 'papers',
    *,
    cutoff_col_name: str = "Notes",
    write_subset_to: str = None
) -> pd.DataFrame:
    """Load sheet, trim right of cutoff, ffill IDs for non-empty ‘Sample’, clean numerics."""

    # Read without assuming a header row
    raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    # Detect the row that contains the header (must contain 'DOI')
    header_row_idx = raw.index[raw.apply(lambda r: "DOI" in r.values, axis=1)][0]

    # Set header and drop all rows up to and including the header row
    raw.columns = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1 :].reset_index(drop=True)

    if cutoff_col_name in df.columns:
        df = df.iloc[:, :df.columns.get_loc(cutoff_col_name)]

    if write_subset_to:
        df.to_excel(write_subset_to, index=False)

    if 'Sample' in df.columns:
        mask = df['Sample'].notna()
        for col in ('Paper_ID', "subfolder", 'DOI'):
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col].ffill()

    prop_cols = [
        "YS (MPa)", "UTS (MPa)", "Uniform EL (%)", "Total EL (%)",
        "Area Reduction (%)", "Absorbed Energy (J/cm²)",
        "Hardness (HV)", "n"
    ]
    for col in df.columns:
        if col in prop_cols:
            s = df[col].astype(str)
            if s.str.contains('±').any():
                s = s.str.split('±').str[0].str.strip()
            if s.str.contains('+', regex=False).any():
                s = s.str.split('+').str[0].str.strip()
            df[col] = pd.to_numeric(s, errors='coerce')

    df['sample_id'] = range(1, len(df) + 1)
    print(f"Loaded {len(df)} samples from Excel (sheet='{sheet_name}')")
    return df

# ---------------------------------------------------
# Step 2: Split images into separate observations
# ---------------------------------------------------
def split_images_to_observations(literature_df):
    """Explode wide image columns to one row per image; compute um_per_pixel; build IDs."""
    observations = []

    # Only consider string columns and keep your original pattern
    name_cols = [
        col for col in literature_df.columns
        if isinstance(col, str) and col.startswith('Name_')
    ]

    for _, sample_row in literature_df.iterrows():
        sample_has_images = False

        for name_col in name_cols:
            if pd.notna(sample_row[name_col]) and str(sample_row[name_col]).strip():
                sample_has_images = True
                image_number = name_col.split('_')[1]

                res_um_col = f'Resolution_{image_number} (um)'
                res_px_col = f'Resolution_{image_number} (pixel)'
                um_per_pixel_col = f'umperpixel{image_number}'  # from your Excel
                method_col = next(
                    (col for col in literature_df.columns
                     if isinstance(col, str)
                     and 'Method' in col and image_number in col),
                    None
                )

                obs_row = {}
                for col in literature_df.columns:
                    if not (
                        isinstance(col, str)
                        and any(x in col for x in ['Name_', 'Resolution_', 'Method', 'um_per_pixel', 'umperpixel'])
                    ):
                        obs_row[col] = sample_row[col]

                obs_row['image_name'] = str(sample_row[name_col]).replace('.png', '').strip()
                obs_row['image_number'] = int(image_number)
                obs_row['resolution_um'] = sample_row[res_um_col] if res_um_col in literature_df.columns else None
                obs_row['resolution_px'] = sample_row[res_px_col] if res_px_col in literature_df.columns else None
                obs_row['method'] = sample_row[method_col] if method_col and pd.notna(sample_row[method_col]) else None

                if um_per_pixel_col in literature_df.columns and pd.notna(sample_row[um_per_pixel_col]):
                    obs_row['um_per_pixel'] = sample_row[um_per_pixel_col]
                elif (
                    pd.notna(obs_row['resolution_um'])
                    and pd.notna(obs_row['resolution_px'])
                    and obs_row['resolution_px'] > 0
                ):
                    obs_row['um_per_pixel'] = obs_row['resolution_um'] / obs_row['resolution_px']
                else:
                    obs_row['um_per_pixel'] = None

                observations.append(obs_row)

        if not sample_has_images:
            obs_row = {}
            for col in literature_df.columns:
                if not (
                    isinstance(col, str)
                    and any(x in col for x in ['Name_', 'Resolution_', 'Method', 'um_per_pixel', 'umperpixel'])
                ):
                    obs_row[col] = sample_row[col]
            obs_row['image_name'] = None
            obs_row['image_number'] = None
            obs_row['resolution_um'] = None
            obs_row['resolution_px'] = None
            obs_row['um_per_pixel'] = None
            obs_row['method'] = None
            observations.append(obs_row)

    obs_df = pd.DataFrame(observations)
    obs_df['obs_id'] = range(1, len(obs_df) + 1)
    obs_df['doi_norm'] = obs_df['DOI'].map(_normalize_doi)
    obs_df['image_name_norm'] = obs_df['image_name'].map(_normalize_filename)
    obs_df["ID_name"] = obs_df['subfolder'] + "_" + obs_df['image_name_norm'].fillna('')

    return obs_df

# ---------------------------------------------------
# Step 3: Add image features (merge by DOI + filename)
# ---------------------------------------------------

def add_image_features(observations_df, features_csv):
    """
    Merge observations with feature CSV on ID_name.

    Keeps the observation-side metadata as authoritative. Overlapping columns
    (e.g. Sample, DOI, image_name) are disambiguated during the merge by
    appending '_feat' to the features-side copies — a necessary intermediate
    step to prevent a pandas MergeError — and then immediately dropped.
    Only the numeric image features and the resolved image_file path are
    retained from the features CSV.
    """
    features_df = pd.read_csv(features_csv)

    # Keep ID/file column names consistent
    if 'ID_name' not in features_df.columns and 'id_name' in features_df.columns:
        features_df = features_df.rename(columns={'id_name': 'ID_name'})
    if 'filename' not in features_df.columns and 'image_filename' in features_df.columns:
        features_df = features_df.rename(columns={'image_filename': 'filename'})

    # Normalize merge key
    observations_df = observations_df.copy()
    features_df['ID_name'] = features_df['ID_name'].astype(str).str.strip().str.lower()
    observations_df['ID_name'] = observations_df['ID_name'].astype(str).str.strip().str.lower()

    # Merge; overlapping columns from the features side receive "_feat"
    merged_df = observations_df.merge(
        features_df,
        on='ID_name',
        how='left',
        suffixes=('', '_feat')
    )

    # Preserve the actual feature-file filename, if present
    if 'filename' in merged_df.columns:
        merged_df = merged_df.rename(columns={'filename': 'image_file'})
    elif 'image_filename' in merged_df.columns:
        merged_df = merged_df.rename(columns={'image_filename': 'image_file'})
    else:
        merged_df['image_file'] = None

    # Drop all collision-generated duplicate columns from the features side
    feat_dup_cols = [c for c in merged_df.columns if c.endswith('_feat')]
    if feat_dup_cols:
        merged_df = merged_df.drop(columns=feat_dup_cols)

    # Put identifiers/metadata first
    id_cols = [
        'obs_id', 'sample_id', 'Sample', 'DOI', 'ID_name',
        'image_file', 'image_name', 'image_number',
        'resolution_um', 'resolution_px', 'um_per_pixel', 'method'
    ]
    other_cols = [c for c in merged_df.columns if c not in id_cols]
    final_cols = [c for c in id_cols if c in merged_df.columns] + other_cols

    return merged_df[final_cols]


# ---------------------------------------------------
# Step 3.5: Unify morphology to a single block for ML
# ---------------------------------------------------

def unify_morphology_for_ml(df: pd.DataFrame, drop_per_method: bool = True) -> pd.DataFrame:
    """Collapse per-method morph columns into morph_*; keep source; drop method-specific cols."""
    out = df.copy()

    methods = ("standard", "fine", "inverse", "adaptive", "multiotsu")
    metrics = ("n_particles", "area_mean", "diameter_mean", "aspect_ratio_mean", "eccentricity_mean")

    # Ensure canonical columns exist
    for met in metrics:
        col = f"morph_{met}"
        if col not in out.columns:
            out[col] = np.nan

    n = len(out)
    if n == 0:
        return out

    # Choose best method by n_particles (row-wise), if available
    counts_cols = [f"morph_{m}_n_particles" for m in methods if f"morph_{m}_n_particles" in out.columns]
    chosen = np.array([None] * n, dtype=object)
    if counts_cols:
        counts = out[counts_cols].fillna(-1).to_numpy()
        best_idx = counts.argmax(axis=1)
        best_methods = [re.sub(r"^morph_(.+)_n_particles$", r"\1", counts_cols[i]) for i in best_idx]
        best_vals = counts[np.arange(n), best_idx]
        chosen = np.where(best_vals >= 1, np.array(best_methods, dtype=object), None)

    # Fallback to explicit method_used if present
    if "morph_method_used" in out.columns:
        mu = out["morph_method_used"].astype(str).str.lower()
        for m in methods:
            mask = (pd.isna(chosen)) & (mu.eq(m))
            np.put(chosen, np.where(mask)[0], m)

    # If canonical already valid (n_particles>0) and nothing chosen, keep canonical
    canon_ok = out.get("morph_n_particles", pd.Series(np.nan, index=out.index)).fillna(0) > 0
    chosen = np.where(canon_ok & pd.isna(chosen), "canonical", chosen)

    # Fill canonical columns from the chosen source, but only where canonical is NaN
    for met in metrics:
        dst = f"morph_{met}"
        for m in methods:
            src = f"morph_{m}_{met}"
            if src in out.columns:
                mask = (chosen == m) & out[dst].isna()
                if mask.any():
                    out.loc[mask, dst] = out.loc[mask, src].values

    # Record source category (no one-hot here to keep columns minimal)
    out["morph_source"] = pd.Series(chosen, index=out.index).where(pd.notna(chosen), "none").astype(str)

    # Drop per-method morph columns to DECREASE feature count
    if drop_per_method:
        drop_cols = [c for c in out.columns if re.match(r"^morph_(standard|fine|inverse|adaptive|multiotsu)_", c)]
        drop_cols += [c for c in ("morph_method_used",) if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    return out


# ---------------------------------------------------
# Step 4: Convert features to physical units (µm)
# ---------------------------------------------------

def add_physical_units_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pixel-based features to µm/µm² using um_per_pixel; add Hall–Petch predictor."""
    import numpy as np

    if 'um_per_pixel' not in df.columns:
        raise ValueError(
            "ERROR: 'um_per_pixel' column not found in dataframe!\n"
            "Please ensure your Excel data includes resolution measurements."
        )

    def _pick(series_df, names):
        for name in names:
            if name in series_df.columns:
                return series_df[name]
        return None

    df_out = df.copy()

    total_images = df_out['image_name'].notna().sum() if 'image_name' in df_out.columns else len(df_out)
    n_available = df_out['um_per_pixel'].notna().sum()
    print("\n======================================================================")
    print("POST-PROCESSING: Converting Features to Physical Units")
    print("======================================================================")
    print(f"\nResolution data:")
    print(f"  Total rows: {total_images}")
    print(f"  With um_per_pixel: {n_available} ({(100*n_available/max(total_images,1)):.1f}%)")

    if n_available == 0:
        print("\nNo um_per_pixel present; returning input unchanged.")
        return df_out

    measured = df_out[df_out['um_per_pixel'].notna()]
    if len(measured):
        print(f"\nResolution statistics:")
        print(f"  Range:  {measured['um_per_pixel'].min():.4f} - {measured['um_per_pixel'].max():.4f} µm/px")
        print(f"  Mean:   {measured['um_per_pixel'].mean():.4f} µm/px")
        print(f"  Median: {measured['um_per_pixel'].median():.4f} µm/px")

    # LENGTH conversions (new names first, fallback to old)
    length_specs = {
        "feat_morph_diameter_mean_um": ["feat_morph_diameter_mean", "feat_diameter_mean"],
        "feat_s2_correlation_length_um": ["feat_s2_correlation_length", "feat_CorrelationLength"],
        "feat_s2_first_peak_pos_um": ["feat_s2_first_peak_pos", "feat_FirstPeakPos"],
        "feat_morph_equivalent_diameter_mean_um": ["feat_morph_equivalent_diameter_mean", "feat_equivalent_diameter_mean"],
        "feat_morph_perimeter_mean_um": ["feat_morph_perimeter_mean", "feat_perimeter_mean"],
        "feat_morph_major_axis_mean_um": ["feat_morph_major_axis_mean", "feat_major_axis_mean"],
        "feat_morph_minor_axis_mean_um": ["feat_morph_minor_axis_mean", "feat_minor_axis_mean"],
    }

    print("\n----------------------------------------------------------------------")
    print("Converting LENGTH features:")
    print("----------------------------------------------------------------------")

    converted_lengths = 0
    for dest_um, sources in length_specs.items():
        src_series = _pick(df_out, sources)
        if src_series is not None:
            df_out[dest_um] = src_series * df_out['um_per_pixel']
            converted_lengths += 1
            print(f"  ✓ {sources[0]} → {dest_um}")

    if converted_lengths == 0:
        print("  No length features found for conversion.")

    # AREA conversions (new names first, fallback to old)
    area_specs = {
        "feat_morph_area_mean_um2": ["feat_morph_area_mean", "feat_area_mean"],
        "feat_morph_area_total_um2": ["feat_morph_area_total", "feat_area_total"],
        "feat_morph_area_median_um2": ["feat_morph_area_median", "feat_area_median"],
    }

    print("\n----------------------------------------------------------------------")
    print("Converting AREA features:")
    print("----------------------------------------------------------------------")

    converted_areas = 0
    for dest_um2, sources in area_specs.items():
        src_series = _pick(df_out, sources)
        if src_series is not None:
            df_out[dest_um2] = src_series * (df_out['um_per_pixel'] ** 2)
            converted_areas += 1
            print(f"  ✓ {sources[0]} → {dest_um2}")

    if converted_areas == 0:
        print("  No area features found for conversion.")

    # Physics-based derived example (Hall–Petch from correlation length)
    if "feat_s2_correlation_length_um" in df_out.columns:
        df_out["feat_hall_petch_predictor"] = 1 / np.sqrt(
            df_out["feat_s2_correlation_length_um"].replace(0, np.nan)
        )
        print("\nDerived:")
        print("  ✓ feat_hall_petch_predictor from feat_s2_correlation_length_um")

    print("\n======================================================================")
    print("Done converting to physical units.")
    print("======================================================================")

    return df_out

# ---------------------------------------------------
# Pipeline
# ---------------------------------------------------

def process_steel_data(literature_excel, features_csv, output_csv, sheet_name='papers'):
    """
    Pipeline:
    1. Load literature data
    2. Split images to observations
    3. Merge with features
    4. Unify morphology
    5. Convert to physical units
    6. Normalize by imaging mode
    7. Save
    """
    
    lit_df = load_literature_data(literature_excel, sheet_name)
    obs_df = split_images_to_observations(lit_df)
    df_image = add_image_features(obs_df, features_csv)
    df_image = unify_morphology_for_ml(df_image, drop_per_method=True)
    final_df = add_physical_units_to_features(df_image)
    # Calibration has been used for conversion, not as a predictor.
    final_df = final_df.drop(columns=["um_per_pixel"],errors="ignore")
    final_df["is_bse"] = (final_df["method"] == "SEM:BSE").astype(int)
    
    if output_csv:
        final_df.to_csv(output_csv, index=False)
        print(f"\n✓ Complete! Final dataset saved to {output_csv}")
    
    return final_df

def calculate_balance_composition(Xin, col_chemistry):
    """Fill balance element to 100%: Fe for Steel, else the element named in ‘Material’."""
    import pandas as pd
    X_out = Xin.copy(deep=True)
    for idx, row in X_out.iterrows():
        material = row["Material"]
        if material == "Steel":
            other_elements = [col for col in col_chemistry if col != "Fe"]
            other_elements_sum = sum(row[col] if pd.notna(row[col]) else 0 
                                   for col in other_elements if col in X_out.columns)
            X_out.loc[idx, "Fe"] = 100 - other_elements_sum
        elif material in col_chemistry:
            other_elements = [col for col in col_chemistry if col != material]
            other_elements_sum = sum(row[col] if pd.notna(row[col]) else 0 
                                   for col in other_elements if col in X_out.columns)
            X_out.loc[idx, material] = 100 - other_elements_sum
        else:
            pass
    return X_out

def check_doi_consistency(df, doi_col_obs="DOI", doi_col_feat="feat_doi"):
    """Compare normalized DOIs from obs vs. features; print mismatches; return mismatch slice."""
    if doi_col_obs not in df.columns or doi_col_feat not in df.columns:
        print("DOI check skipped: required DOI columns not found.")
        return
    obs_doi = df[doi_col_obs].map(_normalize_doi)
    feat_doi = df[doi_col_feat].map(_normalize_doi)
    mismatches = (obs_doi.notna() & feat_doi.notna() & (obs_doi != feat_doi))
    n_bad = mismatches.sum()
    if n_bad > 0:
        print(f"⚠️ DOI mismatch detected for {n_bad} rows. Example(s):")
        print(df.loc[mismatches, [doi_col_obs, doi_col_feat, "ID_name"]].head())
        return df.loc[mismatches, ["ID_name", "obs_id","sample_id",doi_col_obs, doi_col_feat]]
    else:
        print("✓ DOI check passed: all matched DOIs agree.")
        return 

def combine_morphology(
    df: pd.DataFrame,
    *,
    prefix: str = "",                       # use '' for raw descriptors, 'feat_' after merge
    methods = ("standard","fine","inverse","adaptive","multiotsu"),
    metrics = ("n_particles","area_mean","diameter_mean","aspect_ratio_mean","eccentricity_mean"),
    min_particles: int = 1,
    use_method_used_fallback: bool = True,
    one_hot: bool = True,
    drop_per_method: bool = True,
) -> pd.DataFrame:
    """General collapse: pick best method, fill {prefix}morph_*, label source, optional one-hot/drop."""
    out = df.copy()

    # Ensure canonical columns exist
    for met in metrics:
        col = f"{prefix}morph_{met}"
        if col not in out.columns:
            out[col] = np.nan

    n = len(out)
    if n == 0:
        return out

    # Build counts matrix (methods x rows)
    counts = np.full((len(methods), n), -1.0)
    have_any_method_cols = False
    for k, m in enumerate(methods):
        col = f"{prefix}morph_{m}_n_particles"
        if col in out.columns:
            have_any_method_cols = True
            counts[k, :] = out[col].fillna(-1).to_numpy()

    chosen = np.array([None]*n, dtype=object)

    if have_any_method_cols:
        best_idx = counts.argmax(axis=0)
        best_cnt = counts[best_idx, np.arange(n)]
        chosen = np.where(best_cnt >= float(min_particles),
                          np.array(methods, dtype=object)[best_idx],
                          None)

    # Fallback to morph_method_used if provided
    mmu_col = f"{prefix}morph_method_used"
    if use_method_used_fallback and (mmu_col in out.columns):
        mmu = out[mmu_col].astype(str).str.lower()
        for m in methods:
            mask = (pd.isna(chosen)) & (mmu.eq(m))
            np.put(chosen, np.where(mask)[0], m)

    # If canonical block already valid, mark source as 'canonical'
    canon_ok = (out.get(f"{prefix}morph_n_particles", pd.Series(np.nan, index=out.index)).fillna(0) > 0)
    chosen = np.where(canon_ok & pd.isna(chosen), "canonical", chosen)

    # Write canonical values from chosen per-method blocks (overwrite to ensure one source of truth)
    for met in metrics:
        dst = f"{prefix}morph_{met}"
        for m in methods:
            src = f"{prefix}morph_{m}_{met}"
            if src in out.columns:
                mask = (chosen == m)
                if mask.any():
                    out.loc[mask, dst] = out.loc[mask, src].values

    # Source label
    src_col = f"{prefix}morph_source"
    out[src_col] = pd.Series(chosen, index=out.index).where(pd.notna(chosen), "none").astype(str)

    # One-hot encode source (use fixed columns to keep schema stable)
    if one_hot:
        for m in list(methods) + ["canonical","none"]:
            oh = f"{prefix}morph_source_{m}"
            out[oh] = (out[src_col] == m).astype(int)

    # Optionally drop per-method numeric columns
    if drop_per_method:
        pat = re.compile(rf"^{re.escape(prefix)}morph_({'|'.join(methods)})_")
        drop_cols = [c for c in out.columns if pat.match(c)]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    return out


def rename_old_descriptors(df: pd.DataFrame,
                           *,
                           lbp_P: int = 8,
                           lbp_R: int = 1,
                           id_name_target: str = "id_name"  # or "ID_name"
                           ) -> pd.DataFrame:
    """
    Upgrade old descriptor column names to the new convention.

    Old example columns:
      filename, subfolder, img_type, classification_confidence,
      standard_area_mean, fine_area_mean, inverse_area_mean, ...
      contrast, homogeneity, energy, anisotropy_d1, ..., correlation,
      lbp_bin_0..9,
      Baseline, CorrelationIntegral, CorrelationLength, SlopeAt0, SpecificSurface,
      FirstPeakPos, FirstPeakVal, MeanExcess, VarianceExcess, SkewnessExcess, KurtosisExcess,
      area_mean, diameter_mean, aspect_ratio_mean, eccentricity_mean, n_particles,
      pdf_filename, doi, doi_url, extraction_date, ID_name

    New target convention:
      image_filename, subfolder, cls_type, cls_confidence,
      morph_standard_*, morph_fine_*, morph_inverse_*,
      glcm_*, lbp_p{P}r{R}_bin_*, s2_*,
      morph_area_mean, morph_diameter_mean, morph_aspect_ratio_mean,
      morph_eccentricity_mean, morph_n_particles,
      pdf_filename, doi, doi_url, extraction_date, id_name (or ID_name)
    """
    out = df.copy()

    rename_map = {}

    # --- ID / file / class ---
    if "filename" in out.columns:
        rename_map["filename"] = "image_filename"
    if "img_type" in out.columns:
        rename_map["img_type"] = "cls_type"
    if "classification_confidence" in out.columns:
        rename_map["classification_confidence"] = "cls_confidence"
    if "ID_name" in out.columns and id_name_target != "ID_name":
        rename_map["ID_name"] = id_name_target
    if "ID_name" not in out.columns and "id_name" in out.columns and id_name_target == "ID_name":
        rename_map["id_name"] = "ID_name"

    # --- Canonical morphology block (no method) ---
    canon_morph = {
        "area_mean": "morph_area_mean",
        "diameter_mean": "morph_diameter_mean",
        "aspect_ratio_mean": "morph_aspect_ratio_mean",
        "eccentricity_mean": "morph_eccentricity_mean",
        "n_particles": "morph_n_particles",
    }
    for old, new in canon_morph.items():
        if old in out.columns:
            rename_map[old] = new

    # --- Per-method morphology: standard/fine/inverse ---
    methods = ("standard", "fine", "inverse")
    morph_metrics = ("n_particles","area_mean","diameter_mean","aspect_ratio_mean","eccentricity_mean")
    for m in methods:
        for met in morph_metrics:
            old = f"{m}_{met}"
            new = f"morph_{m}_{met}"
            if old in out.columns:
                rename_map[old] = new

    # --- GLCM block: add glcm_ prefix to known keys ---
    # Simple keys
    glcm_simple = ["contrast","homogeneity","energy","correlation"]
    for k in glcm_simple:
        if k in out.columns:
            rename_map[k] = f"glcm_{k}"

    # Distance-based anisotropy bundles: d1/d3/d5
    dists = ("1","3","5")
    glcm_pref = {
        "anisotropy_d{d}": "glcm_anisotropy_d{d}",
        "homogeneity_anisotropy_d{d}": "glcm_homogeneity_anisotropy_d{d}",
        "max_contrast_d{d}": "glcm_max_contrast_d{d}",
        "min_contrast_d{d}": "glcm_min_contrast_d{d}",
        "homogeneity_d{d}": "glcm_homogeneity_d{d}",
    }
    for d in dists:
        for old_t, new_t in glcm_pref.items():
            old = old_t.format(d=d)
            new = new_t.format(d=d)
            if old in out.columns:
                rename_map[old] = new

    # Remaining glcm scalars
    more_glcm = {
        "anisotropy_index": "glcm_anisotropy_index",
        "homogeneity_anisotropy": "glcm_homogeneity_anisotropy",
        "anisotropy_ratio": "glcm_anisotropy_ratio",
        # note: 'correlation' already handled above
    }
    for old, new in more_glcm.items():
        if old in out.columns:
            rename_map[old] = new

    # --- LBP bins ---
    # old: lbp_bin_0..9  -> new: lbp_p{P}r{R}_bin_i
    lbp_pat = re.compile(r"^lbp_bin_(\d+)$")
    for c in list(out.columns):
        m = lbp_pat.match(c)
        if m:
            i = int(m.group(1))
            rename_map[c] = f"lbp_p{lbp_P}r{lbp_R}_bin_{i}"

    # --- 2-point correlation (S2) ---
    s2_map = {
        "Baseline": "s2_baseline",
        "CorrelationIntegral": "s2_correlation_integral",
        "CorrelationLength": "s2_correlation_length",
        "SlopeAt0": "s2_slope_at_0",
        "SpecificSurface": "s2_specific_surface",
        "FirstPeakPos": "s2_first_peak_pos",
        "FirstPeakVal": "s2_first_peak_val",
        "MeanExcess": "s2_mean_excess",
        "VarianceExcess": "s2_variance_excess",
        "SkewnessExcess": "s2_skewness_excess",
        "KurtosisExcess": "s2_kurtosis_excess",
    }
    for old, new in s2_map.items():
        if old in out.columns:
            rename_map[old] = new

    # Apply rename
    out = out.rename(columns=rename_map)

    # Optional: stable, friendly column order (keys first, then features)
    key_cols = [c for c in [
        "image_filename", "subfolder", "cls_type", "cls_confidence",
        "pdf_filename", "doi", "doi_url", "extraction_date",
        id_name_target
    ] if c in out.columns]

    # Put morph canonical block next
    morph_canon_cols = [c for c in [
        "morph_area_mean","morph_diameter_mean","morph_aspect_ratio_mean",
        "morph_eccentricity_mean","morph_n_particles"
    ] if c in out.columns]

    # Then per-method morph (if you want to keep them for now)
    morph_method_cols = sorted([c for c in out.columns if re.match(r"^morph_(standard|fine|inverse)_", c)])

    glcm_cols = sorted([c for c in out.columns if c.startswith("glcm_")])
    lbp_cols  = sorted([c for c in out.columns if c.startswith(f"lbp_p{lbp_P}r{lbp_R}_bin_")])
    s2_cols   = sorted([c for c in out.columns if c.startswith("s2_")])

    others = [c for c in out.columns if c not in (key_cols + morph_canon_cols + morph_method_cols + glcm_cols + lbp_cols + s2_cols)]
    new_order = key_cols + morph_canon_cols + morph_method_cols + glcm_cols + lbp_cols + s2_cols + others
    out = out[new_order]

    return out


def derived_chem_features(X):
    X_new = X.copy()
    X_new["Cr_eq"] = X_new["Cr"]+ X_new.get("Mo", 0.0) + 1. * X_new.get("Si", 0.0) + 0.5 *X_new.get("Nb", 0.0)
    X_new["Ni_eq"] = X_new["Ni"] + 30.0 * X_new.get("C", 0.0) + 0.5 * X_new.get("Mn", 0.0) + 30.0 * X_new.get("N", 0.0)
    X_new["Creq_Nieq_ratio"] = X_new["Cr_eq"] / (X_new["Ni_eq"] + 1e-6)
    return X_new

def analyze_failed_matches(df, features_csv=None):
    """
    Analyze which images/masks failed to match or have missing data.
    Returns detailed DataFrames for inspection.
    """
    import pandas as pd
    
    print(f"\n{'='*80}")
    print("FAILED MATCH ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Images with NO feature data at all
    # Identify feature columns (anything that starts with common prefixes)
    feature_prefixes = ['feat_', 'morph_', 'glcm_', 's2_', 'lbp_']
    feature_cols = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in feature_prefixes)]
    
    if not feature_cols:
        print("No feature columns found!")
        return None
    
    # Create a mask for rows that have ANY feature data
    has_any_feature = df[feature_cols].notna().any(axis=1)
    
    # Images with names but no features
    has_image_name = df['image_name'].notna()
    images_no_features = df[has_image_name & ~has_any_feature].copy()
    
    print(f"\n1. IMAGES WITH NO FEATURE DATA:")
    print(f"   Total images: {has_image_name.sum()}")
    print(f"   Images with features: {has_any_feature.sum()}")
    print(f"   Images WITHOUT features: {len(images_no_features)}")
    
    if len(images_no_features) > 0:
        print(f"\n   First 20 examples:")
        for idx, row in images_no_features.head(20).iterrows():
            print(f"   - obs_id: {row.get('obs_id', 'N/A')}, "
                  f"ID_name: {row.get('ID_name', 'N/A')}, "
                  f"Image: {row.get('image_name', 'N/A')}")
        
    
    # 2. Images missing resolution data (um_per_pixel)
    images_no_resolution = df[has_image_name & df['um_per_pixel'].isna()].copy()
    
    print(f"\n\n2. IMAGES MISSING RESOLUTION DATA:")
    print(f"   Total images: {has_image_name.sum()}")
    print(f"   Images WITH resolution: {df['um_per_pixel'].notna().sum()}")
    print(f"   Images WITHOUT resolution: {len(images_no_resolution)}")
    
    if len(images_no_resolution) > 0:
        print(f"\n   First 10 examples:")
        for idx, row in images_no_resolution.head(10).iterrows():
            print(f"   - obs_id: {row.get('obs_id', 'N/A')}, "
                  f"Image: {row.get('image_name', 'N/A')}, "
                  f"Method: {row.get('method', 'N/A')}")
    
    # 3. Check specific feature types that might be missing
    print(f"\n\n3. FEATURE TYPE AVAILABILITY:")
    
    feature_groups = {
        'Morphology': ['morph_'],
        'GLCM': ['glcm_'],
        'S2': ['s2_'],
        'LBP': ['lbp_']
    }
    
    for group_name, prefixes in feature_groups.items():
        group_cols = [col for col in df.columns 
                     if any(col.startswith(prefix) for prefix in prefixes)]
        if group_cols:
            has_group_feature = df[group_cols].notna().any(axis=1)
            print(f"   {group_name}: {has_group_feature.sum()} images have features")
    
    # 4. Analyze by imaging method
    print(f"\n\n4. ANALYSIS BY IMAGING METHOD:")
    if 'method' in df.columns:
        method_stats = df[has_image_name].groupby('method').agg(
            total=('image_name', 'size'),
            with_features=('image_name', lambda x: has_any_feature[x.index].sum()),
            with_resolution=('um_per_pixel', lambda x: x.notna().sum())
        ).reset_index()
        
        method_stats['features_pct'] = (method_stats['with_features'] / method_stats['total'] * 100).round(1)
        method_stats['resolution_pct'] = (method_stats['with_resolution'] / method_stats['total'] * 100).round(1)
        
        print(method_stats.to_string(index=False))
    
    # 5. Cross-reference with original features CSV if provided
    if features_csv is not None and os.path.exists(features_csv):
        print(f"\n\n5. CROSS-REFERENCE WITH FEATURES CSV:")
        features_df = pd.read_csv(features_csv)
        
        # Get ID_name column name from features
        id_col = 'ID_name' if 'ID_name' in features_df.columns else 'id_name'
        if id_col not in features_df.columns:
            id_col = next((c for c in features_df.columns if 'name' in c.lower()), None)
        
        if id_col:
            features_ids = set(features_df[id_col].astype(str).str.lower().str.strip().dropna().unique())
            obs_ids = set(df[has_image_name]['ID_name'].astype(str).str.lower().str.strip().dropna().unique())
            
            in_features_not_obs = features_ids - obs_ids
            in_obs_not_features = obs_ids - features_ids
            
            print(f"   IDs in features CSV but not in observations: {len(in_features_not_obs)}")
            if in_features_not_obs:
                print(f"   Example IDs: {list(in_features_not_obs)[:5]}")
            
            print(f"   IDs in observations but not in features CSV: {len(in_obs_not_features)}")
            if in_obs_not_features:
                print(f"   Example IDs: {list(in_obs_not_features)[:5]}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"  • Total observations: {len(df)}")
    print(f"  • Observations with images: {has_image_name.sum()}")
    print(f"  • Images with ANY features: {has_any_feature.sum()}")
    print(f"  • Images missing ALL features: {len(images_no_features)}")
    print(f"  • Images missing resolution: {len(images_no_resolution)}")
    print(f"{'='*80}")
    
    # Return DataFrames for further inspection
    results = {
        'images_no_features': images_no_features,
        'images_no_resolution': images_no_resolution,
        'has_any_feature': has_any_feature
    }
    
    return results


if __name__ == "__main__":
    import sys
    module_path = str.rsplit(os.path.abspath(''), os.path.sep, 1)[0]
    sys.path.append(module_path)
    print(module_path)
    
    # ============================================================================
    # Combine literature data and image recognition
    # ============================================================================

    excel_file_path = os.path.join(module_path, "data", "literature_data.xlsx")
    features_csv_path = os.path.join(module_path, "data", "descriptors.csv")
    output_csv_path = os.path.join(module_path, "data", "descriptors_full.csv")
    
    df_descriptors = pd.read_csv(features_csv_path)

    lit_df = load_literature_data(excel_file_path, sheet_name='papers')
    obs_df = split_images_to_observations(lit_df)
    df_image = add_image_features(obs_df, features_csv_path)
    df_image = unify_morphology_for_ml(df_image, drop_per_method=True)
    final_df = add_physical_units_to_features(df_image)
    # Calibration has been used for conversion, not as a predictor.
    final_df = final_df.drop(columns=["um_per_pixel"],errors="ignore")
    final_df["is_bse"] = (final_df["method"] == "SEM: BSE").astype(int)
    ### df_missmatch = check_doi_consistency(final_df)
    
    final_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Done. Saved combined dataset to {output_csv_path}")
    