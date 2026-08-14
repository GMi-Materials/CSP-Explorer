# -*- coding: utf-8 -*-
"""
Author: Gerfried Millner
Version: 1.0.1
Date: 03.06.2026
Email: gerfried.millner@ntu.edu.sg
Status: Deliver

@author: gerfried.millner
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

def filter_by_subcategory(df, subcategory, category_coulmn="Category", drop_col=True):
    """
    Filter dataframe by Material Subcategory.
    
    Args:
        df: DataFrame with Category column
        subcategory: String name of subcategory (e.g., 'Dual-Phase (DP) Steel')
                     OR list of subcategory names
        category_coulmn: String, name of category column
    
    Returns:
        Filtered DataFrame
    """
    if category_coulmn not in df.columns:
        print("Error: "+str(category_coulmn)+" column not found.")
        return None
    
    # Convert single string to list for uniform handling
    if isinstance(subcategory, str):
        subcategory_list = [subcategory]
    else:
        subcategory_list = subcategory
    
    # Filter using isin() to match any subcategory in the list
    filtered_df = df[df[category_coulmn].isin(subcategory_list)].copy()
    
    # Print filtering results
    if len(subcategory_list) == 1:
        print(f"Filtered to: {subcategory_list[0]}")
    else:
        print(f"Filtered to {len(subcategory_list)} subcategories: {', '.join(subcategory_list)}")
    print(f"Samples: {len(filtered_df)}")
    
    if drop_col:
        filtered_df = filtered_df.drop(category_coulmn, axis=1)
    
    return filtered_df

def calculate_balance_composition(Xin, col_chemistry):
    """
    Calculate and fill the balance composition for each row.
    For Steel, balance is Fe. For other materials, balance is the base element.
    
    Parameters:
        Xin (pd.DataFrame): Input dataframe
        col_chemistry (list): List of chemical element columns
    
    Returns:
        pd.DataFrame: Dataframe with updated balance compositions
    """
    X_out = Xin.copy(deep=True)
    
    for idx, row in X_out.iterrows():
        material = row.get("Material", "Steel")
        
        if material == "Steel":
            other_elements = [col for col in col_chemistry if col != "Fe"]
            other_sum = sum(row[col] if pd.notna(row[col]) else 0 
                          for col in other_elements if col in X_out.columns)
            X_out.loc[idx, "Fe"] = 100 - other_sum
            
        elif material in col_chemistry:
            other_elements = [col for col in col_chemistry if col != material]
            other_sum = sum(row[col] if pd.notna(row[col]) else 0 
                          for col in other_elements if col in X_out.columns)
            X_out.loc[idx, material] = 100 - other_sum
    
    return X_out


def diagnose_nan_patterns(df, target_col='UTS (MPa)', show_plot=True):
    """
    Comprehensive NaN diagnosis by feature category.
    
    Parameters:
        df: DataFrame with features
        target_col: Target column name
        show_plot: Whether to show visualization
    
    Returns:
        DataFrame with category statistics
    """
    print(f"\n{'='*80}")
    print("NaN DIAGNOSIS BY FEATURE CATEGORY")
    print(f"{'='*80}\n")
    
    # Define feature categories with automatic detection
    feature_categories = {
        'Target': [target_col],
        'Chemistry': ["Al", "B", "C", "Ca", "Ce", "Co", "Cr", "Cu", "Er", "La", "Mg", "Mn",
                      "Mo", "N", "Nb", "Ni", "O", "P", "S", "Si", "Sm", "Sn", "Ti", "Ta",
                      "V", "W", "Zn", "Zr", "Fe"],
        'Scale': [c for c in df.columns if 'um_per_pixel' in c or 'resolution' in c],
        'Morphology': [c for c in df.columns if c.startswith('morph_')],
        'GLCM': [c for c in df.columns if c.startswith('glcm_')],
        'LBP': [c for c in df.columns if c.startswith('lbp_')],
        'S2': [c for c in df.columns if c.startswith('s2_')],
        'Gram_conv1': [c for c in df.columns if c.startswith('conv1_')],
        'Gram_layer1': [c for c in df.columns if c.startswith('layer1_')],
        'Gram_layer2': [c for c in df.columns if c.startswith('layer2_')],
        'Gram_layer3': [c for c in df.columns if c.startswith('layer3_')],
        'Gram_layer4': [c for c in df.columns if c.startswith('layer4_')],
        'Gram_cross': [c for c in df.columns if c.startswith('cross_')],
        'Gram_global': [c for c in df.columns if c.startswith('global_') or c.startswith('multi_scale')],
    }
    
    category_stats = []
    
    for cat_name, cols in feature_categories.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
            
        cat_data = df[available]
        n_features = len(available)
        avg_nan_pct = cat_data.isna().mean().mean() * 100
        rows_all_nan = cat_data.isna().all(axis=1).sum()
        rows_any_nan = cat_data.isna().any(axis=1).sum()
        rows_complete = len(df) - rows_any_nan
        
        category_stats.append({
            'Category': cat_name,
            'N_Features': n_features,
            'Avg_NaN_%': avg_nan_pct,
            'Rows_All_NaN': rows_all_nan,
            'Rows_Any_NaN': rows_any_nan,
            'Rows_Complete': rows_complete
        })
        
        # Status indicator
        if avg_nan_pct < 5:
            status = "✓ GOOD"
        elif avg_nan_pct < 50:
            status = "⚠ FAIR"
        else:
            status = "❌ POOR"
        
        print(f"{status:12} {cat_name:20} Features: {n_features:3d}  |  Avg NaN: {avg_nan_pct:5.1f}%  |  "
              f"Complete rows: {rows_complete:4d}/{len(df)} ({100*rows_complete/len(df):5.1f}%)")
    
    stats_df = pd.DataFrame(category_stats).sort_values('Avg_NaN_%', ascending=False)
    
    # Visualization
    if show_plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Avg NaN by category
        colors = ['red' if x > 50 else 'orange' if x > 20 else 'yellow' if x > 5 else 'green' 
                 for x in stats_df['Avg_NaN_%']]
        axes[0].barh(stats_df['Category'], stats_df['Avg_NaN_%'], color=colors, alpha=0.7)
        axes[0].set_xlabel('Average NaN (%)', fontsize=11)
        axes[0].set_title('NaN Percentage by Category', fontsize=12, fontweight='bold')
        axes[0].axvline(50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='50%')
        axes[0].axvline(20, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='20%')
        axes[0].axvline(5, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='5%')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='x')
        
        # Plot 2: Complete rows by category
        axes[1].barh(stats_df['Category'], stats_df['Rows_Complete'], color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Number of Complete Rows', fontsize=11)
        axes[1].set_title('Complete Rows per Category', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    return stats_df

def diagnose_infinity_sources(df, show_top=20):
    """
    Identify which features have infinity values.
    Useful for debugging extraction code.
    
    Parameters:
        df: DataFrame to check
        show_top: Number of features to show
    
    Returns:
        DataFrame with infinity statistics
    """
    print(f"\n{'='*80}")
    print("INFINITY VALUE DIAGNOSIS")
    print(f"{'='*80}\n")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    inf_stats = []
    for col in num_cols:
        n_pos_inf = (df[col] == np.inf).sum()
        n_neg_inf = (df[col] == -np.inf).sum()
        n_total_inf = n_pos_inf + n_neg_inf
        
        if n_total_inf > 0:
            inf_stats.append({
                'Feature': col,
                'Pos_Inf': n_pos_inf,
                'Neg_Inf': n_neg_inf,
                'Total_Inf': n_total_inf,
                'Pct_Inf': (n_total_inf / len(df)) * 100
            })
    
    if not inf_stats:
        print("✓ No infinity values found!")
        return pd.DataFrame()
    
    inf_df = pd.DataFrame(inf_stats).sort_values('Total_Inf', ascending=False)
    
    print(f"Found {len(inf_df)} features with infinity values")
    print(f"Total infinity values: {inf_df['Total_Inf'].sum():,}\n")
    
    print(f"{'Feature':<50} {'Pos_Inf':>10} {'Neg_Inf':>10} {'Total':>10} {'%':>8}")
    print("-" * 90)
    
    for _, row in inf_df.head(show_top).iterrows():
        print(f"{row['Feature']:<50} {row['Pos_Inf']:>10.0f} {row['Neg_Inf']:>10.0f} "
              f"{row['Total_Inf']:>10.0f} {row['Pct_Inf']:>7.1f}%")
    
    if len(inf_df) > show_top:
        print(f"\n... and {len(inf_df) - show_top} more features")
    
    # Categorize infinity sources
    print(f"\n{'='*80}")
    print("INFINITY BY CATEGORY")
    print(f"{'='*80}\n")
    
    categories = {
        'Gram': inf_df[inf_df['Feature'].str.contains('conv|layer|cross_|global_|multi_scale', case=False)],
        'GLCM': inf_df[inf_df['Feature'].str.startswith('glcm_')],
        'S2': inf_df[inf_df['Feature'].str.startswith('s2_')],
        'Morphology': inf_df[inf_df['Feature'].str.startswith('morph_')],
        'LBP': inf_df[inf_df['Feature'].str.startswith('lbp_')],
        'Other': inf_df[~inf_df['Feature'].str.contains('conv|layer|cross_|global_|multi_scale|glcm_|s2_|morph_|lbp_', case=False)]
    }
    
    for cat, data in categories.items():
        if len(data) > 0:
            print(f"{cat:15}: {len(data):3d} features, {data['Total_Inf'].sum():6.0f} infinity values")
    
    return inf_df

def clean_multicollinearity(
    X,
    composition_cols,
    vif_threshold=10.0,
    pairwise_corr_threshold=0.9,
    comp_corr_threshold=0.95,
    protect_elements=("Fe",),
):
    """
    Generic multicollinearity reduction with:
      - automatic composite formation for strongly coupled composition pairs
      - VIF-based pruning for non-composition features.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix. 
    composition_cols : list of str
        Columns treated as composition (elements, composites). They are
        never dropped by the VIF pruning step.
    vif_threshold : float
        VIF cutoff for non-composition features.
    pairwise_corr_threshold : float
        |r| threshold for treating non-composition pairs as "very strongly
        correlated" when deciding which feature to drop.
    comp_corr_threshold : float
        |r| threshold for detecting strongly coupled composition pairs
        that will be merged into composites.
    protect_elements : tuple of str
        Composition columns that must never be removed or merged
        (e.g. ("Fe",) to always keep Fe explicit).

    Returns
    -------
    X_reduced : pd.DataFrame
        Feature matrix after multicollinearity reduction.
    removed_features : list of str
        Features removed or merged during the procedure.
    created_composites : list of str
        Names of composition composites that were created.
    """

    if "sample_id" in X.columns:
        X_work = X.drop(columns="sample_id").copy()
    else:
        X_work = X.copy()
    removed = []
    created_composites = []

    # ------------------------------------------------------------------
    # 1. Automatic composites for strongly coupled composition pairs
    # ------------------------------------------------------------------
    # Start from composition_cols, but keep only columns that currently exist
    comp_cols_present = [c for c in composition_cols if c in X_work.columns]
    
    # If there are fewer than 2 composition columns, nothing to do
    if len(comp_cols_present) > 1:
        comp_corr = X_work[comp_cols_present].corr().abs()
    
        # Find pairs of composition features with |r| > comp_corr_threshold
        high_pairs = np.where(
            (comp_corr > comp_corr_threshold) & (comp_corr < 1.0)
        )
        comp_pairs = sorted(
            {tuple(sorted((comp_corr.index[i], comp_corr.columns[j])))
             for i, j in zip(*high_pairs)}
        )
    
        for a, b in comp_pairs:
            # Skip if either element is protected (e.g. Fe)
            if a in protect_elements or b in protect_elements:
                continue
    
            # SAFETY: re-check that both columns still exist
            if a not in X_work.columns or b not in X_work.columns:
                continue
    
            new_name = f"{a}_plus_{b}"
            if new_name in X_work.columns:
                continue  # already created earlier
    
            # Create composite as sum; interpretable as total addition of a and b
            X_work[new_name] = X_work[a] + X_work[b]
            created_composites.append(new_name)
    
            # Drop originals (now encoded in the composite)
            for col in (a, b):
                if col in X_work.columns:
                    X_work = X_work.drop(columns=col)
                    removed.append(col)
    
            # Track the new composite as a composition column
            comp_cols_present.append(new_name)
    
    # Refresh composition_cols after any changes
    composition_cols = sorted(
        set(c for c in comp_cols_present if c in X_work.columns)
    )
    # ------------------------------------------------------------------
    # 2. Iterative VIF-based pruning for NON-composition features
    # ------------------------------------------------------------------
    def compute_vif(df):
        X_num = df.select_dtypes(include=["number"]).dropna()
        if X_num.shape[1] <= 1:
            return pd.DataFrame({"feature": X_num.columns, "VIF": [1.0]})
        vif_vals = [
            variance_inflation_factor(X_num.values, i)
            for i in range(X_num.shape[1])
        ]
        return pd.DataFrame({"feature": X_num.columns, "VIF": vif_vals})

    improved = True
    while improved:
        improved = False

        vif_df = compute_vif(X_work)
        # Exclude composition columns from removal
        vif_df = vif_df[~vif_df["feature"].isin(composition_cols)]
        vif_df = vif_df.sort_values("VIF", ascending=False)

        if vif_df.empty:
            break

        top_feat, top_vif = vif_df.iloc[0]["feature"], vif_df.iloc[0]["VIF"]

        if top_vif <= vif_threshold:
            break  # worst non-composition VIF acceptable

        # Identify strongly correlated partners of top_feat
        corr = X_work.corr(numeric_only=True)
        if top_feat not in corr.columns:
            break

        partners = corr[top_feat].abs().sort_values(ascending=False)
        partners = partners[
            (partners.index != top_feat)
            & (partners > pairwise_corr_threshold)
        ]

        if partners.empty:
            # No strong partner: drop the high-VIF feature itself
            drop = top_feat
        else:
            # Within the high-corr group, drop the feature that is
            # most redundant: highest mean abs correlation to all others.
            candidate_group = [top_feat] + partners.index.tolist()
            subcorr = corr.loc[candidate_group, candidate_group].abs()
            mean_corr = subcorr.mean(axis=1)

            # Never drop composition features
            mean_corr = mean_corr.drop(
                labels=[c for c in mean_corr.index if c in composition_cols],
                errors="ignore",
            )

            drop = mean_corr.sort_values(ascending=False).index[0]

        X_work = X_work.drop(columns=drop)
        removed.append(drop)
        improved = True

    if "sample_id" in X.columns:
        X_work["sample_id"] = X["sample_id"]

    return X_work, removed, created_composites

def clean_data_simple(df, target_col='UTS (MPa)', id_cols=None, 
                     impute_method='category', handle_inf='clip', verbose=True):
    """
    Simplified data cleaning with smart defaults.
    
    Parameters:
        df: Input DataFrame
        target_col: Target column name
        id_cols: List of ID/metadata columns to exclude
        impute_method: 'none' (drop NaN rows), 'simple' (median all), 
                      'category' (median by feature type) - RECOMMENDED
        handle_inf: How to handle infinity values:
                   'clip' - Replace with large finite values (default)
                   'nan' - Replace with NaN then impute
                   'drop_features' - Drop features with any infinity
        verbose: Print progress messages
    
    Returns:
        X, y: Cleaned features and target
    """
    if verbose:
        print(f"\n{'='*80}")
        print("DATA CLEANING PIPELINE")
        print(f"{'='*80}\n")
    
    if id_cols is None:
        id_cols = []
    
    # Define columns to exclude
    target_cols = ['YS (MPa)', 'UTS (MPa)', 'Uniform EL (%)', 'Total EL (%)',
                   'Area Reduction (%)', 'Absorbed Energy (J/cm²)', 'Hardness (HV)', 
                   'n2', "Young's modulus [GPa]", 'Young modulus [GPa]']
    
    chemistry_cols = ["Al", "B", "C", "Ca", "Ce", "Co", "Cr", "Cu", "Er", "La", "Mg", "Mn",
                     "Mo", "N", "Nb", "Ni", "O", "P", "S", "Si", "Sm", "Sn", "Ti", "Ta",
                     "V", "W", "Zn", "Zr", "Fe"]
    
    X = df.copy()
    y = df[target_col].copy()
    
    if verbose:
        print(f"Initial: {X.shape[0]} rows, {X.shape[1]} columns")
    
    # STEP 1: Drop rows with missing target
    valid_target = y.notna()
    X = X[valid_target]
    y = y[valid_target]
    
    if verbose:
        print(f"Step 1: Removed {(~valid_target).sum()} rows with missing target → {len(X)} rows")
    
    # STEP 2: Handle chemistry (fill with 0, calculate balance)
    chem_available = [c for c in chemistry_cols if c in X.columns]
    if chem_available:
        # if 'Material' in X.columns:      # balance material does not provide new info, according to abaltion study and the decision was made to drop it for sake of interpretability
        #     X = calculate_balance_composition(X, chem_available)
        X[chem_available] = X[chem_available].fillna(0)

        if verbose:
            print("Step 2: Chemistry features filled with 0 (absence)")
            
    if X['Material'].unique()[0]=='Steel' and 'Fe' in X.columns: # drop Fe for steels
        X = X.copy().drop(columns=["Fe"])   

    # STEP 3: Remove ID and target columns
    cols_to_drop = list(set(id_cols + target_cols + ['Material', 'Other']))
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
    
    if verbose:
        print(f"Step 3: Removed ID/target columns → {X.shape[1]} feature columns")

    # STEP 3b: Drop strictly constant features (including constant-zero elements)
    # This is the only structural feature drop; rare non-constant elements are kept.
    nunique = X.nunique(dropna=False)
    constant_features = nunique[nunique <= 1].index.tolist()
    if constant_features:
        X = X.drop(columns=constant_features)
        if verbose:
            print(f"Step 3b: Dropped {len(constant_features)} constant features → {X.shape[1]} features")
            print(str(constant_features))

    # STEP 4: Drop features with >80% NaN
    nan_pct = X.isna().mean()
    high_nan_features = nan_pct[nan_pct > 0.80].index.tolist()
    
    if high_nan_features:
        X = X.drop(columns=high_nan_features)
        if verbose:
            print(f"Step 4: Dropped {len(high_nan_features)} features with >80% NaN → {X.shape[1]} features")
            gram_dropped = len([c for c in high_nan_features if any(x in c for x in ['conv', 'layer', 'cross_', 'global_', 'multi_scale'])])
            if gram_dropped > 0:
                print(f"        ⚠ {gram_dropped} Gram features dropped (likely extraction failure)")
    else:
        if verbose:
            print("Step 4: No features >80% NaN ✓")

    # STEP 5: Drop duplicate/redundant features
    redundant = ['resolution_um', 'resolution_px']
    X = X.drop(columns=[c for c in redundant if c in X.columns], errors='ignore')
    
    # STEP 6: Critical feature check - drop rows missing um_per_pixel
    if 'um_per_pixel' in X.columns:
        missing_scale = X['um_per_pixel'].isna()
        if missing_scale.any():
            X = X[~missing_scale]
            y = y.loc[X.index]
            if verbose:
                print(f"Step 5: Dropped {missing_scale.sum()} rows missing um_per_pixel (critical)")
    
    # STEP 7: Handle infinity values BEFORE imputation (critical!)
    num_cols = X.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(X[num_cols].values)
    n_infs = inf_mask.sum()
    
    if n_infs > 0:
        if verbose:
            print(f"\n⚠️  Found {n_infs} infinity values before imputation!")
        
        cols_with_inf = []
        for col in num_cols:
            if np.isinf(X[col]).any():
                n_inf = np.isinf(X[col]).sum()
                cols_with_inf.append((col, n_inf))
        
        if verbose and len(cols_with_inf) <= 20:
            print(f"    Columns with infinity ({len(cols_with_inf)}):")
            for col, n in sorted(cols_with_inf, key=lambda x: x[1], reverse=True):
                print(f"      - {col}: {n} values")
        elif verbose:
            print(f"    Columns with infinity: {len(cols_with_inf)} (showing top 10)")
            for col, n in sorted(cols_with_inf, key=lambda x: x[1], reverse=True)[:10]:
                print(f"      - {col}: {n} values")
        
        if handle_inf == 'drop_features':
            drop_cols = [col for col, _ in cols_with_inf]
            X = X.drop(columns=drop_cols)
            if verbose:
                print(f"    ✓ Dropped {len(drop_cols)} features with infinity values")
        
        elif handle_inf == 'nan':
            for col, _ in cols_with_inf:
                X.loc[np.isinf(X[col]), col] = np.nan
            if verbose:
                print(f"    ✓ Replaced infinity with NaN in {len(cols_with_inf)} features (will impute)")
        
        else:  # handle_inf == 'clip' (default)
            for col in num_cols:
                if not np.isinf(X[col]).any():
                    continue
                finite_vals = X[col][np.isfinite(X[col])]
                if len(finite_vals) > 0:
                    col_max = finite_vals.max()
                    col_min = finite_vals.min()
                    col_std = finite_vals.std()
                    if col_max > 0:
                        pos_clip = min(col_max * 10, col_max + 5 * col_std)
                    else:
                        pos_clip = 1e10
                    if col_min < 0:
                        neg_clip = max(col_min * 10, col_min - 5 * col_std)
                    else:
                        neg_clip = -1e10
                else:
                    pos_clip = 1e10
                    neg_clip = -1e10
                X.loc[X[col] == np.inf, col] = pos_clip
                X.loc[X[col] == -np.inf, col] = neg_clip
            
            if verbose:
                print("    ✓ Clipped infinity values to finite bounds")
    
    # STEP 8: Imputation (now infinity-free)
    if impute_method == 'none':
        before = len(X)
        X = X.dropna()
        y = y.loc[X.index]
        dropped = before - len(X)
        if verbose:
            print(f"Step 6: IMPUTATION = None → Dropped {dropped} rows with any NaN → {len(X)} rows")
    
    elif impute_method == 'simple':
        num_cols = X.select_dtypes(include=[np.number]).columns
        cols_with_nan = [c for c in num_cols if X[c].isna().any()]
        if cols_with_nan:
            X[cols_with_nan] = SimpleImputer(strategy='median').fit_transform(X[cols_with_nan])
            if verbose:
                print(f"Step 6: IMPUTATION = Simple median → Imputed {len(cols_with_nan)} features")
        else:
            if verbose:
                print("Step 6: IMPUTATION = Simple median → No features need imputation ✓")
    
    elif impute_method == 'category':
        if verbose:
            print("Step 6: IMPUTATION = Category-aware median")
        
        feature_blocks = {
            'Morphology': [c for c in X.columns if c.startswith('morph_')],
            'GLCM': [c for c in X.columns if c.startswith('glcm_')],
            'LBP': [c for c in X.columns if c.startswith('lbp_')],
            'S2': [c for c in X.columns if c.startswith('s2_')],
            'Gram': [c for c in X.columns if c.startswith(('conv', 'layer', 'cross_', 'global_', 'multi_scale'))],
        }
        
        total_imputed = 0
        for block_name, cols in feature_blocks.items():
            if not cols:
                continue
            cols_with_nan = [c for c in cols if X[c].isna().any()]
            if cols_with_nan:
                X[cols_with_nan] = SimpleImputer(strategy='median').fit_transform(X[cols_with_nan])
                total_imputed += len(cols_with_nan)
                if verbose:
                    print(f"        → {block_name:12}: Imputed {len(cols_with_nan):3d} features")
        
        num_cols = X.select_dtypes(include=[np.number]).columns
        remaining_nan = [c for c in num_cols if X[c].isna().any()]
        if remaining_nan:
            X[remaining_nan] = SimpleImputer(strategy='median').fit_transform(X[remaining_nan])
            total_imputed += len(remaining_nan)
            if verbose:
                print(f"        → {'Other':12}: Imputed {len(remaining_nan):3d} features")
        
        if verbose:
            print(f"        Total imputed: {total_imputed} features")
    
    # STEP 9: Treading Multicollinearity
    X, _, _ = clean_multicollinearity(
        X,
        chem_available,
        vif_threshold=10.0,
        pairwise_corr_threshold=0.9,
        comp_corr_threshold=0.95,
        protect_elements=("Fe",),
    )
    
    # STEP 10: Final verification
    remaining_nan = X.isna().sum().sum()
    num_cols_final = X.select_dtypes(include=[np.number]).columns
    remaining_inf = np.isinf(X[num_cols_final].values).sum() if len(num_cols_final) > 0 else 0
    
    if verbose:
        print(f"\n{'='*80}")
        print("CLEANING SUMMARY")
        print(f"{'='*80}")
        print(f"Final shape:          {X.shape}")
        print(f"Samples retained:     {len(X)}/{len(df)} ({100*len(X)/len(df):.1f}%)")
        print(f"Features retained:    {X.shape[1]}")
        print(f"Remaining NaNs:       {remaining_nan}")
        print(f"Remaining infinities: {remaining_inf}")
        
        if remaining_nan == 0 and remaining_inf == 0:
            print("\n✓✓✓ DATA IS CLEAN AND ML-READY! ✓✓✓")
        else:
            if remaining_nan > 0:
                print(f"\n⚠️  WARNING: {remaining_nan} NaN values remain!")
            if remaining_inf > 0:
                print(f"⚠️  WARNING: {remaining_inf} infinity values remain!")
        print(f"{'='*80}\n")
    
    return X, y

def quick_clean(df, target_col='UTS (MPa)', id_cols=None, handle_inf='clip'):
    """
    Ultra-simple one-line cleaning for quick testing.
    Uses category-aware imputation (recommended).
    
    Parameters:
        df: Input DataFrame
        target_col: Target column name
        id_cols: List of ID columns (optional)
        handle_inf: 'clip', 'nan', or 'drop_features'
    
    Returns:
        X, y: Clean features and target, ready for ML
    """
    return clean_data_simple(df, target_col, id_cols, 
                            impute_method='category', handle_inf=handle_inf,
                            verbose=False)


# Convenience function - keep your original name for compatibility
def separate_XY(df, target_col, id_cols=None, imputation_strategy='category', 
               handle_inf='clip', verbose=False, microscopy_method=None):
    """
    Wrapper for backwards compatibility with original script.
    Maps old parameter names to new function and optionally filters by microscopy method.
    
    Parameters:
        df: Input DataFrame
        target_col: Target column name
        id_cols: List of ID columns
        imputation_strategy: 'none', 'simple', or 'category' (recommended)
        handle_inf: 'clip', 'nan', or 'drop_features'
        verbose: Print progress
        microscopy_method: list of microscopy methods considered ('OM', 'SEM', 'TEM'). Default to all ('None')
    
    Returns:
        X, y: Cleaned features and target
    """
    if microscopy_method != None:
        df_filtered = df[df['method'].isin(microscopy_method)]
    else:
        df_filtered = df.copy()
    
    return clean_data_simple(df_filtered, target_col, id_cols, 
                            impute_method=imputation_strategy, 
                            handle_inf=handle_inf, verbose=verbose)

def fit_cleaning_pipeline(
    df,
    target_col="UTS (MPa)",
    id_cols=None,
    impute_method="category",
    handle_inf="clip",
    vif_threshold=10.0,
    pairwise_corr_threshold=0.9,
    comp_corr_threshold=0.95,
    protect_elements=(),
    verbose=True
):
    if id_cols is None:
        id_cols = []

    target_cols = [
        "YS (MPa)", "UTS (MPa)", "Uniform EL (%)", "Total EL (%)",
        "Area Reduction (%)", "Absorbed Energy (J/cm²)", "Hardness (HV)",
        "n2", "Young's modulus [GPa]", "Young modulus [GPa]"
    ]

    chemistry_cols = [
        "Al", "B", "C", "Ca", "Ce", "Co", "Cr", "Cu", "Er", "La", "Mg", "Mn",
        "Mo", "N", "Nb", "Ni", "O", "P", "S", "Si", "Sm", "Sn", "Ti", "Ta",
        "V", "W", "Zn", "Zr", "Fe"
    ]

    X = df.copy()
    y = X[target_col].copy() if target_col in X.columns else None

    if y is not None:
        valid_target = y.notna()
        X = X.loc[valid_target].copy()
        y = y.loc[valid_target].copy()

    chem_available = [c for c in chemistry_cols if c in X.columns]
    if chem_available:
        X[chem_available] = X[chem_available].fillna(0)
        X = calculate_balance_composition(X, chem_available)

    cols_to_drop = list(set(id_cols + target_cols + ["Material", "Other"]))
    cols_to_drop = [c for c in cols_to_drop if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    constant_features = X.columns[X.nunique(dropna=False) <= 1].tolist()
    X = X.drop(columns=constant_features, errors="ignore")

    nan_pct = X.isna().mean()
    high_nan_features = nan_pct[nan_pct > 0.80].index.tolist()
    X = X.drop(columns=high_nan_features, errors="ignore")

    redundant = [c for c in ["resolution_um", "resolution_px"] if c in X.columns]
    X = X.drop(columns=redundant, errors="ignore")

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    inf_replacement = {}

    if handle_inf == "clip":
        for col in num_cols:
            if not np.isinf(X[col]).any():
                continue
            finite_vals = X.loc[np.isfinite(X[col]), col]
            if len(finite_vals) > 0:
                col_max = finite_vals.max()
                col_min = finite_vals.min()
                col_std = finite_vals.std()
                pos_clip = min(col_max * 10, col_max + 5 * col_std) if col_max > 0 else 1e10
                neg_clip = max(col_min * 10, col_min - 5 * col_std) if col_min < 0 else -1e10
            else:
                pos_clip = 1e10
                neg_clip = -1e10

            X.loc[X[col] == np.inf, col] = pos_clip
            X.loc[X[col] == -np.inf, col] = neg_clip
            inf_replacement[col] = {"pos_clip": pos_clip, "neg_clip": neg_clip}

    elif handle_inf == "nan":
        for col in num_cols:
            if np.isinf(X[col]).any():
                X.loc[np.isinf(X[col]), col] = np.nan

    elif handle_inf == "dropfeatures":
        inf_cols = [col for col in num_cols if np.isinf(X[col]).any()]
        X = X.drop(columns=inf_cols, errors="ignore")

    impute_values = {}
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            impute_values[col] = median_val

    chem_available = [c for c in chemistry_cols if c in X.columns]
    X_mc, removed_features, created_composites = clean_multicollinearity(
        X,
        composition_cols=chem_available,
        vif_threshold=vif_threshold,
        pairwise_corr_threshold=pairwise_corr_threshold,
        comp_corr_threshold=comp_corr_threshold,
        protect_elements=protect_elements
    )

    pipeline = {
        "target_col": target_col,
        "id_cols": id_cols,
        "constant_features": constant_features,
        "high_nan_features": high_nan_features,
        "redundant_features": redundant,
        "inf_replacement": inf_replacement,
        "impute_values": impute_values,
        "removed_features": removed_features,
        "created_composites": created_composites,
        "final_columns": X_mc.columns.tolist(),
        "chemistry_cols": chemistry_cols,
        "handle_inf": handle_inf
    }

    return X_mc, y, pipeline

def apply_cleaning_pipeline(df, pipeline, target_col=None, verbose=True):
    X = df.copy()

    if target_col is None:
        target_col = pipeline["target_col"]

    y = X[target_col].copy() if target_col in X.columns else None
    if y is not None:
        valid_target = y.notna()
        X = X.loc[valid_target].copy()
        y = y.loc[valid_target].copy()

    chemistry_cols = [c for c in pipeline["chemistry_cols"] if c in X.columns]
    if chemistry_cols:
        X[chemistry_cols] = X[chemistry_cols].fillna(0)

    if "Material" in X.columns and X["Material"].nunique() == 1:
        if X["Material"].iloc[0] == "Steel" and "Fe" in X.columns:
            X = X.drop(columns=["Fe"])

    target_cols = [
        "YS (MPa)", "UTS (MPa)", "Uniform EL (%)", "Total EL (%)",
        "Area Reduction (%)", "Absorbed Energy (J/cm²)", "Hardness (HV)",
        "n2", "Young's modulus [GPa]", "Young modulus [GPa]"
    ]

    cols_to_drop = list(set(pipeline["id_cols"] + target_cols + ["Material", "Other"]))
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors="ignore")

    X = X.drop(columns=[c for c in pipeline["constant_features"] if c in X.columns], errors="ignore")
    X = X.drop(columns=[c for c in pipeline["high_nan_features"] if c in X.columns], errors="ignore")
    X = X.drop(columns=[c for c in pipeline["redundant_features"] if c in X.columns], errors="ignore")

    if "um_per_pixel" in X.columns and X["um_per_pixel"].isna().any():
        missing_scale = X["um_per_pixel"].isna()
        X = X.loc[~missing_scale].copy()
        if y is not None:
            y = y.loc[X.index]

    if pipeline["handle_inf"] == "clip":
        for col, vals in pipeline["inf_replacement"].items():
            if col in X.columns:
                X.loc[X[col] == np.inf, col] = vals["pos_clip"]
                X.loc[X[col] == -np.inf, col] = vals["neg_clip"]
    elif pipeline["handle_inf"] == "nan":
        for col in X.select_dtypes(include=np.number).columns:
            X.loc[np.isinf(X[col]), col] = np.nan
    elif pipeline["handle_inf"] == "drop_features":
        inf_cols = [col for col in X.select_dtypes(include=np.number).columns if np.isinf(X[col]).any()]
        X = X.drop(columns=inf_cols, errors="ignore")

    for col, median_val in pipeline["impute_values"].items():
        if col in X.columns:
            X[col] = X[col].fillna(median_val)

    # Recreate training composites from names like "Cr_plus_Ni"
    for comp_name in pipeline["created_composites"]:
        if comp_name in X.columns:
            continue

        if "_plus_" not in comp_name:
            if verbose:
                print(f"Warning: cannot parse composite name {comp_name}")
            continue

        a, b = comp_name.split("_plus_", 1)

        a_exists = a in X.columns
        b_exists = b in X.columns

        if a_exists and b_exists:
            X[comp_name] = X[a] + X[b]
        elif a_exists and not b_exists:
            X[comp_name] = X[a]
        elif b_exists and not a_exists:
            X[comp_name] = X[b]
        else:
            X[comp_name] = 0.0
            if verbose:
                print(f"Warning: neither source column found for composite {comp_name}")

    X = X.drop(columns=[c for c in pipeline["removed_features"] if c in X.columns], errors="ignore")

    for col in pipeline["final_columns"]:
        if col not in X.columns:
            X[col] = 0.0

    extra_cols = [c for c in X.columns if c not in pipeline["final_columns"]]
    if extra_cols:
        X = X.drop(columns=extra_cols, errors="ignore")

    X = X[pipeline["final_columns"]]

    return X, y

if __name__ == "__main__":
    import os, sys
    module_path = str.rsplit(os.path.abspath(''), os.path.sep, 1)[0]
    sys.path.append(module_path)
    from feat_reducer import CustomComprehensiveReducer

    STAINLESS_CATS = ["Ferritic Stainless Steel", "Duplex Stainless Steel", "Austenitic Stainless Steel"]
    DP_CATS        = ["Conventional DP"]
    STUDY_CATEGORIES = STAINLESS_CATS + DP_CATS

    GROUPS = {
        "ss_sem": {"categories": STAINLESS_CATS, "methods": ["SEM: SE", "SEM: BSE"]},
        "dp_sem": {"categories": DP_CATS,        "methods": ["SEM: SE", "SEM: BSE"]},
        "ss_om":  {"categories": STAINLESS_CATS, "methods": ["OM"]},
    }

    id_columns = [
        'obs_id', 'ID_name',
        'Sample', 'Paper_ID', 'Material', 'Material_Subcategory', 'Category',
        'Classification_Confidence', 'DOI', 'DOI_Enhanced', 'subfolder',
        'image_file', 'image_name', 'image_number', 'resolution_um', 'method', 'Fig',
        'doi', 'doi_norm', 'doi_url', 'pdf_filename', 'extraction_date', 'image_name_norm',
        'cls_type', 'cls_confidence', 'morph_source', 'feat_morph_source',
        'um_per_pixel',
    ]

    # Load full descriptors and restrict to study population
    print("\nLoading data...")
    df_full = pd.read_csv(os.path.join(module_path, 'data', 'descriptors_full.csv'))
    print(f"Loaded: {df_full.shape[0]} samples, {df_full.shape[1]} columns")
    df_full = filter_by_subcategory(df_full, STUDY_CATEGORIES, drop_col=False)

    # STEP 1: Diagnose NaN patterns on the study population
    print("\n" + "="*80)
    print("STEP 1: DIAGNOSE NaN PATTERNS")
    print("="*80)
    diagnose_nan_patterns(df_full, target_col='UTS (MPa)', show_plot=True)

    xy_dir = os.path.join(module_path, 'data', 'Xy')
    os.makedirs(xy_dir, exist_ok=True)

    for group_name, cfg in GROUPS.items():
        print("\n" + "="*80)
        print(f"GROUP: {group_name.upper()}")
        print("="*80)

        # STEP 2: Feature reduction (cached), fitted on this group only
        reduced_path = os.path.join(module_path, 'data', f'descriptors_reduced_{group_name}.csv')
        if not os.path.exists(reduced_path):
            print(f"No reduced file found — running feat_reducer for '{group_name}'...")
            df_group = df_full[
                df_full['method'].isin(cfg['methods']) &
                df_full['Category'].isin(cfg['categories'])
            ].copy().reset_index(drop=True)
            reducer = CustomComprehensiveReducer(reduce_s2_aggressively=False)
            df_reduced, _ = reducer.reduce(df_group)
            df_reduced.to_csv(reduced_path, index=False)
            print(f"Saved: {reduced_path}")
        else:
            print(f"Loading cached: {reduced_path}")
            df_reduced = pd.read_csv(reduced_path)

        id_cols = [c for c in id_columns if c in df_reduced.columns]

        # STEP 3: Clean and generate X/y
        print("\n" + "="*80)
        print("STEP 3: CLEAN DATA")
        print("="*80)

        X_uts, y_uts = separate_XY(df_reduced, target_col='UTS (MPa)',    id_cols=id_cols, imputation_strategy='category')
        X_ys,  y_ys  = separate_XY(df_reduced, target_col='YS (MPa)',     id_cols=id_cols, imputation_strategy='category')
        X_el,  y_el  = separate_XY(df_reduced, target_col='Total EL (%)', id_cols=id_cols, imputation_strategy='category')

        X_uts.to_csv(os.path.join(xy_dir, f'X_{group_name}_uts.csv'), index=False)
        y_uts.to_csv(os.path.join(xy_dir, f'y_{group_name}_uts.csv'), index=False)
        X_ys.to_csv( os.path.join(xy_dir, f'X_{group_name}_ys.csv'),  index=False)
        y_ys.to_csv( os.path.join(xy_dir, f'y_{group_name}_ys.csv'),  index=False)
        X_el.to_csv( os.path.join(xy_dir, f'X_{group_name}_el.csv'),  index=False)
        y_el.to_csv( os.path.join(xy_dir, f'y_{group_name}_el.csv'),  index=False)

        print(f"{group_name}: UTS {X_uts.shape}, YS {X_ys.shape}, EL {X_el.shape}")