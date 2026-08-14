#!/usr/bin/env python3
"""
Smart Feature Reduction 
Reduces features by keeping interpretable features 
and combining similar ones by semantic similarity.

Author: Gerfried Millner
Version: 1.1.0
Date: 14.08.2026
Status: Deliver
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import argparse
import sys
import json

class SmartGramReducer:
    """
    Smart reduction: Keep interpretable + Combine similar features.
    """

    def __init__(self):
        self.gram_cols = None
        self.non_gram_cols = None
        self.feature_mapping = {}

        self.layers_ = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        self.kept_features_ = None

        self.basic_cols_ = None
        self.basic_imputer_ = None
        self.basic_scaler_ = None
        self.basic_pca_ = None
        self.basic_n_components_ = 0

        self.eigen_feature_defs_ = [
            ('eigen_max', 'gram_eigen_max_mean'),
            ('eigen_sum', 'gram_eigen_sum_mean'),
            ('effective_rank', 'gram_effective_rank_mean')
        ]

        self.norm_cols_ = None
        self.norm_imputer_ = None
        self.norm_scaler_ = None
        self.norm_pca_ = None
        self.norm_n_components_ = 0

        self.homogeneity_feature_defs_ = [
            ('entropy', 'gram_entropy_mean'),
            ('sparsity', 'gram_sparsity_mean')
        ]

        self.spatial_cols_ = None
        self.spatial_imputer_ = None
        self.spatial_scaler_ = None
        self.spatial_pca_ = None
        self.spatial_n_components_ = 0

        self.top_cross_ = None
        self.output_columns_ = None

    def identify_columns(self, df):
        self.gram_cols = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in
            ['conv1_', 'layer1_', 'layer2_', 'layer3_', 'layer4_', 'cross_', 'multi_scale', 'global_']
        )]
        self.non_gram_cols = [col for col in df.columns if col not in self.gram_cols]
        return len(self.gram_cols), len(self.non_gram_cols)

    def fit(self, df, verbose=True):
        if verbose:
            print("="*80)
            print("SMART GRAM FEATURE REDUCTION")
            print("="*80)

        n_gram, n_non_gram = self.identify_columns(df)

        if verbose:
            print(f"\nOriginal features:")
            print(f" Gram: {n_gram}")
            print(f" Non-Gram: {n_non_gram}")
            print(f" Total: {len(df.columns)}")

        self.feature_mapping = {}

        self.kept_features_ = []
        for layer in self.layers_:
            for stat in ['trace', 'condition_number']:
                col = f'{layer}_{stat}'
                if col in df.columns:
                    self.kept_features_.append(col)
                    self.feature_mapping[col] = [col]

        basic_stats = ['mean', 'std', 'skew', 'kurtosis', 'min', 'max']
        self.basic_cols_ = []
        for layer in self.layers_:
            for stat in basic_stats:
                col = f'{layer}_{stat}'
                if col in df.columns and col not in self.kept_features_:
                    self.basic_cols_.append(col)

        if len(self.basic_cols_) > 3:
            self.basic_imputer_ = SimpleImputer(strategy='median')
            X_basic = self.basic_imputer_.fit_transform(df[self.basic_cols_])

            self.basic_scaler_ = StandardScaler()
            X_basic_scaled = self.basic_scaler_.fit_transform(X_basic)

            self.basic_n_components_ = 3
            self.basic_pca_ = PCA(n_components=3, random_state=42)
            self.basic_pca_.fit(X_basic_scaled)

            for i in range(3):
                self.feature_mapping[f'gram_basic_pc{i+1}'] = self.basic_cols_

        self.norm_cols_ = []
        norm_stats = ['frobenius_norm', 'spectral_norm', 'log_determinant']
        for layer in self.layers_:
            for stat in norm_stats:
                col = f'{layer}_{stat}'
                if col in df.columns:
                    self.norm_cols_.append(col)

        if len(self.norm_cols_) > 2:
            self.norm_imputer_ = SimpleImputer(strategy='median')
            X_norms = self.norm_imputer_.fit_transform(df[self.norm_cols_])

            self.norm_scaler_ = StandardScaler()
            X_norms_scaled = self.norm_scaler_.fit_transform(X_norms)

            self.norm_n_components_ = 2
            self.norm_pca_ = PCA(n_components=2, random_state=42)
            self.norm_pca_.fit(X_norms_scaled)

            for i in range(2):
                self.feature_mapping[f'gram_norms_pc{i+1}'] = self.norm_cols_

        for old_name, new_name in self.eigen_feature_defs_:
            cols = [f'{layer}_{old_name}' for layer in self.layers_ if f'{layer}_{old_name}' in df.columns]
            if cols:
                self.feature_mapping[new_name] = cols

        for old_name, new_name in self.homogeneity_feature_defs_:
            cols = [f'{layer}_{old_name}' for layer in self.layers_ if f'{layer}_{old_name}' in df.columns]
            if cols:
                self.feature_mapping[new_name] = cols

        spatial_stats = ['diag_mean', 'offdiag_mean', 'diag_to_offdiag_ratio']
        self.spatial_cols_ = []
        for layer in self.layers_:
            for stat in spatial_stats:
                col = f'{layer}_{stat}'
                if col in df.columns:
                    self.spatial_cols_.append(col)

        if len(self.spatial_cols_) > 2:
            self.spatial_imputer_ = SimpleImputer(strategy='median')
            X_spatial = self.spatial_imputer_.fit_transform(df[self.spatial_cols_])

            self.spatial_scaler_ = StandardScaler()
            X_spatial_scaled = self.spatial_scaler_.fit_transform(X_spatial)

            self.spatial_n_components_ = 2
            self.spatial_pca_ = PCA(n_components=2, random_state=42)
            self.spatial_pca_.fit(X_spatial_scaled)

            for i in range(2):
                self.feature_mapping[f'gram_spatial_pc{i+1}'] = self.spatial_cols_

        cross_cols = [col for col in df.columns if col.startswith('cross_')]
        self.top_cross_ = []
        if len(cross_cols) > 0:
            cross_variance = df[cross_cols].var().sort_values(ascending=False)
            n_cross = min(3, len(cross_cols))
            self.top_cross_ = cross_variance.head(n_cross).index.tolist()
            for col in self.top_cross_:
                self.feature_mapping[col] = [col]

        transformed = self.transform(df, verbose=False)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, df, verbose=True):
        if self.gram_cols is None:
            raise RuntimeError("SmartGramReducer must be fitted before calling transform().")

        combined_features = pd.DataFrame(index=df.index)

        for feat in self.kept_features_:
            combined_features[feat] = df[feat] if feat in df.columns else 0.0

        if self.basic_pca_ is not None:
            X_basic = df.reindex(columns=self.basic_cols_)
            X_basic = self.basic_imputer_.transform(X_basic)
            X_basic_scaled = self.basic_scaler_.transform(X_basic)
            X_basic_pca = self.basic_pca_.transform(X_basic_scaled)
            for i in range(self.basic_n_components_):
                combined_features[f'gram_basic_pc{i+1}'] = X_basic_pca[:, i]

        for old_name, new_name in self.eigen_feature_defs_:
            cols = [f'{layer}_{old_name}' for layer in self.layers_ if f'{layer}_{old_name}' in self.feature_mapping.get(new_name, [])]
            if cols:
                combined_features[new_name] = df.reindex(columns=cols).mean(axis=1)

        if self.norm_pca_ is not None:
            X_norms = df.reindex(columns=self.norm_cols_)
            X_norms = self.norm_imputer_.transform(X_norms)
            X_norms_scaled = self.norm_scaler_.transform(X_norms)
            X_norms_pca = self.norm_pca_.transform(X_norms_scaled)
            for i in range(self.norm_n_components_):
                combined_features[f'gram_norms_pc{i+1}'] = X_norms_pca[:, i]

        for old_name, new_name in self.homogeneity_feature_defs_:
            cols = [f'{layer}_{old_name}' for layer in self.layers_ if f'{layer}_{old_name}' in self.feature_mapping.get(new_name, [])]
            if cols:
                combined_features[new_name] = df.reindex(columns=cols).mean(axis=1)

        if self.spatial_pca_ is not None:
            X_spatial = df.reindex(columns=self.spatial_cols_)
            X_spatial = self.spatial_imputer_.transform(X_spatial)
            X_spatial_scaled = self.spatial_scaler_.transform(X_spatial)
            X_spatial_pca = self.spatial_pca_.transform(X_spatial_scaled)
            for i in range(self.spatial_n_components_):
                combined_features[f'gram_spatial_pc{i+1}'] = X_spatial_pca[:, i]

        for col in self.top_cross_:
            combined_features[col] = df[col] if col in df.columns else 0.0

        non_gram_cols = [col for col in df.columns if not any(
            col.startswith(prefix) for prefix in
            ['conv1_', 'layer1_', 'layer2_', 'layer3_', 'layer4_', 'cross_', 'multi_scale', 'global_']
        )]

        df_reduced = pd.concat([df[non_gram_cols], combined_features], axis=1)

        if self.output_columns_ is not None:
            for col in self.output_columns_:
                if col not in df_reduced.columns:
                    df_reduced[col] = 0.0
            extra = [c for c in df_reduced.columns if c not in self.output_columns_]
            if extra:
                df_reduced = df_reduced.drop(columns=extra)
            df_reduced = df_reduced[self.output_columns_]

        return df_reduced

    def fit_transform(self, df, verbose=True):
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def reduce(self, df, verbose=True):
        df_reduced = self.fit_transform(df, verbose=verbose)

        n_gram = len([col for col in df.columns if any(
            col.startswith(prefix) for prefix in
            ['conv1_', 'layer1_', 'layer2_', 'layer3_', 'layer4_', 'cross_', 'multi_scale', 'global_']
        )])
        n_gram_final = len([col for col in df_reduced.columns if any(
            col.startswith(prefix) for prefix in
            ['conv1_', 'layer1_', 'layer2_', 'layer3_', 'layer4_', 'cross_', 'multi_scale', 'global_', 'gram_']
        )])

        info = {
            'domain_features': self.kept_features_,
            'combined_features': [c for c in df_reduced.columns if c.startswith('gram_') or c in self.top_cross_],
            'final_gram_count': n_gram_final,
            'original_gram_count': n_gram,
            'reduction_pct': (1 - n_gram_final / n_gram) * 100 if n_gram > 0 else 0,
            'feature_mapping': self.feature_mapping,
            'total_original': len(df.columns),
            'total_reduced': len(df_reduced.columns)
        }

        return df_reduced, info


def validate_reduction(df_original, df_reduced, target_col='UTS (MPa)', cv=5):
    """
    Compare model performance: original vs reduced.
    """
    print(f"\n{'='*80}")
    print("VALIDATION: Comparing Original vs Reduced")
    print(f"{'='*80}\n")
    
    results = {}
    
    for name, df in [('Original', df_original), ('Reduced', df_reduced)]:
        # Prepare data
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # Remove missing targets
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_clean = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_clean)
        
        # Test models
        models = {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42, 
                learning_rate=0.1
            ),
        }
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=cv, scoring='r2', n_jobs=-1
                )
                results[f'{name}_{model_name}'] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'n_features': X.shape[1],
                    'n_samples': len(X)
                }
            except Exception as e:
                print(f"Warning: {model_name} failed: {e}")
    
    # Print results
    print(f"{'Dataset':<12} {'Model':<20} {'Features':>10} {'R² (CV)':<20}")
    print("-"*80)
    
    for key, vals in results.items():
        dataset, model = key.split('_', 1)
        print(f"{dataset:<12} {model:<20} {vals['n_features']:>10} "
              f"{vals['mean_r2']:>6.3f} ± {vals['std_r2']:<6.3f}")
    
    # Performance change
    print(f"\n{'='*80}")
    print("PERFORMANCE CHANGE")
    print(f"{'='*80}\n")
    
    for model_name in ['ElasticNet', 'GradientBoosting']:
        if f'Original_{model_name}' in results and f'Reduced_{model_name}' in results:
            orig_r2 = results[f'Original_{model_name}']['mean_r2']
            red_r2 = results[f'Reduced_{model_name}']['mean_r2']
            change = (red_r2 - orig_r2) / abs(orig_r2) * 100 if orig_r2 != 0 else 0
            
            status = "✅" if abs(change) < 10 else "⚠️" if abs(change) < 20 else "❌"
            print(f"{model_name:<20} {orig_r2:>6.3f} → {red_r2:>6.3f} ({change:+.1f}%) {status}")
    
    print(f"\n{'='*80}")
    if all(abs((results[f'Reduced_{m}']['mean_r2'] - results[f'Original_{m}']['mean_r2']) / 
               results[f'Original_{m}']['mean_r2'] * 100) < 10 
           for m in ['ElasticNet', 'GradientBoosting'] 
           if f'Original_{m}' in results and results[f'Original_{m}']['mean_r2'] != 0):
        print("✅ EXCELLENT! R² loss < 10% - reduction successful!")
    else:
        print("⚠️  R² dropped >10% - consider keeping more features")
    print(f"{'='*80}\n")
    
    return results


def save_reduction_report(info, results, output_path):
    """Save a detailed reduction report as JSON."""
    report = {
        'reduction_summary': {
            'original_features': info['total_original'],
            'reduced_features': info['total_reduced'],
            'original_gram_features': info['original_gram_count'],
            'reduced_gram_features': info['final_gram_count'],
            'reduction_percentage': info['reduction_pct']
        },
        'feature_groups': {
            'domain_features': info['domain_features'],
            'combined_features': info['combined_features']
        },
        'feature_mapping': info['feature_mapping'],
        'validation_results': results if results else None
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Reduction report saved to: {output_path}")


class GLCMReducer:
    """
    Reducer for YOUR GLCM features (22 → 8-10 features)

    Same logic as before, but now supports fit/transform for validation-safe use.
    """

    def __init__(self):
        self.glcm_cols_ = None
        self.non_glcm_cols_ = None
        self.basic_features_ = ['glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity']
        self.aniso_cols_ = None
        self.homog_cols_ = None
        self.homog_aniso_cols_ = None
        self.contrast_cols_ = None

        self.aniso_imputer_ = None
        self.homog_imputer_ = None
        self.homog_aniso_imputer_ = None
        self.contrast_imputer_ = None
        self.contrast_scaler_ = None
        self.contrast_pca_ = None
        self.n_contrast_components_ = 0

        self.feature_mapping_ = {}
        self.output_columns_ = None

    def fit(self, df, verbose=True):
        self.glcm_cols_ = [col for col in df.columns if col.startswith('glcm_')]
        self.non_glcm_cols_ = [col for col in df.columns if col not in self.glcm_cols_]

        if len(self.glcm_cols_) == 0:
            self.output_columns_ = df.columns.tolist()
            return self

        if verbose:
            print(f"\n{'='*80}")
            print("GLCM FEATURE REDUCTION")
            print(f"{'='*80}")
            print(f"Found {len(self.glcm_cols_)} GLCM features\n")

        self.aniso_cols_ = [c for c in ['glcm_anisotropy_d1', 'glcm_anisotropy_d3', 'glcm_anisotropy_d5'] if c in df.columns]
        self.homog_cols_ = [c for c in ['glcm_homogeneity_d1', 'glcm_homogeneity_d3', 'glcm_homogeneity_d5'] if c in df.columns]
        self.homog_aniso_cols_ = [c for c in [
            'glcm_homogeneity_anisotropy_d1',
            'glcm_homogeneity_anisotropy_d3',
            'glcm_homogeneity_anisotropy_d5'
        ] if c in df.columns]
        self.contrast_cols_ = [c for c in [
            'glcm_max_contrast_d1', 'glcm_max_contrast_d3', 'glcm_max_contrast_d5',
            'glcm_min_contrast_d1', 'glcm_min_contrast_d3', 'glcm_min_contrast_d5'
        ] if c in df.columns]

        self.feature_mapping_ = {
            'basic': [c for c in self.basic_features_ if c in df.columns],
            'anisotropy': self.aniso_cols_,
            'homogeneity_multiscale': self.homog_cols_,
            'homogeneity_anisotropy': self.homog_aniso_cols_,
            'contrast_variations': self.contrast_cols_
        }

        if len(self.aniso_cols_) > 0:
            self.aniso_imputer_ = SimpleImputer(strategy='median')
            self.aniso_imputer_.fit(df[self.aniso_cols_])

        if len(self.homog_cols_) > 0:
            self.homog_imputer_ = SimpleImputer(strategy='median')
            self.homog_imputer_.fit(df[self.homog_cols_])

        if len(self.homog_aniso_cols_) > 0:
            self.homog_aniso_imputer_ = SimpleImputer(strategy='median')
            self.homog_aniso_imputer_.fit(df[self.homog_aniso_cols_])

        if len(self.contrast_cols_) >= 3:
            self.contrast_imputer_ = SimpleImputer(strategy='median')
            X_contrast = self.contrast_imputer_.fit_transform(df[self.contrast_cols_])

            self.contrast_scaler_ = StandardScaler()
            X_scaled = self.contrast_scaler_.fit_transform(X_contrast)

            self.n_contrast_components_ = min(2, len(self.contrast_cols_))
            self.contrast_pca_ = PCA(n_components=self.n_contrast_components_, random_state=42)
            self.contrast_pca_.fit(X_scaled)

        transformed = self.transform(df, verbose=False)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, df, verbose=True):
        if self.glcm_cols_ is None:
            raise RuntimeError("GLCMReducer must be fitted before calling transform().")

        if len(self.glcm_cols_) == 0:
            return df.copy()

        reduced_features = pd.DataFrame(index=df.index)

        for feat in self.basic_features_:
            if feat in df.columns:
                reduced_features[feat] = df[feat]
            else:
                reduced_features[feat] = 0.0

        if len(self.aniso_cols_) > 0:
            X_aniso = df.reindex(columns=self.aniso_cols_)
            X_aniso = self.aniso_imputer_.transform(X_aniso)
            reduced_features['glcm_anisotropy_mean'] = X_aniso.mean(axis=1)
            reduced_features['glcm_anisotropy_range'] = X_aniso.max(axis=1) - X_aniso.min(axis=1)

        if len(self.homog_cols_) > 0:
            X_homog = df.reindex(columns=self.homog_cols_)
            X_homog = self.homog_imputer_.transform(X_homog)
            reduced_features['glcm_homogeneity_multiscale_mean'] = X_homog.mean(axis=1)

        if len(self.homog_aniso_cols_) > 0:
            X_homog_aniso = df.reindex(columns=self.homog_aniso_cols_)
            X_homog_aniso = self.homog_aniso_imputer_.transform(X_homog_aniso)
            reduced_features['glcm_homogeneity_anisotropy_mean'] = X_homog_aniso.mean(axis=1)

        if self.contrast_pca_ is not None:
            X_contrast = df.reindex(columns=self.contrast_cols_)
            X_contrast = self.contrast_imputer_.transform(X_contrast)
            X_scaled = self.contrast_scaler_.transform(X_contrast)
            X_pca = self.contrast_pca_.transform(X_scaled)

            for i in range(self.n_contrast_components_):
                reduced_features[f'glcm_contrast_var_pc{i+1}'] = X_pca[:, i]

        non_glcm_cols = [col for col in df.columns if not col.startswith('glcm_')]
        df_reduced = pd.concat([df[non_glcm_cols], reduced_features], axis=1)

        if self.output_columns_ is not None:
            for col in self.output_columns_:
                if col not in df_reduced.columns:
                    df_reduced[col] = 0.0
            extra = [c for c in df_reduced.columns if c not in self.output_columns_]
            if extra:
                df_reduced = df_reduced.drop(columns=extra)
            df_reduced = df_reduced[self.output_columns_]

        return df_reduced

    def fit_transform(self, df, verbose=True):
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def reduce(self, df, verbose=True):
        df_reduced = self.fit_transform(df, verbose=verbose)

        n_original = len([c for c in df.columns if c.startswith('glcm_')])
        n_reduced = len([c for c in df_reduced.columns if c.startswith('glcm_')])

        info = {
            'n_original': n_original,
            'n_reduced': n_reduced,
            'reduction_pct': (1 - n_reduced / n_original) * 100 if n_original > 0 else 0,
            'feature_mapping': self.feature_mapping_
        }

        if verbose and n_original > 0:
            print(f"\nReduction: {n_original} → {n_reduced} features")
            print(f"Removed: {info['reduction_pct']:.1f}%")
            print(f"{'='*80}\n")

        return df_reduced, info

class LBPReducer:

    def __init__(self):
        self.lbp_cols_ = None
        self.non_lbp_cols_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.pca_ = None
        self.n_components_ = 0
        self.output_columns_ = None

    def fit(self, df, verbose=True):
        self.lbp_cols_ = [col for col in df.columns if col.startswith('lbp_')]
        self.non_lbp_cols_ = [col for col in df.columns if col not in self.lbp_cols_]

        if len(self.lbp_cols_) == 0:
            self.output_columns_ = df.columns.tolist()
            return self

        if verbose:
            print(f"\n{'='*80}")
            print("LBP FEATURE REDUCTION")
            print(f"{'='*80}")
            print(f"Found {len(self.lbp_cols_)} LBP histogram bins\n")

        self.imputer_ = SimpleImputer(strategy='median')
        X_lbp = self.imputer_.fit_transform(df[self.lbp_cols_])

        self.n_components_ = min(3, len(self.lbp_cols_))
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_lbp)

        self.pca_ = PCA(n_components=self.n_components_, random_state=42)
        self.pca_.fit(X_scaled)

        transformed = self.transform(df, verbose=False)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, df, verbose=True):
        if self.lbp_cols_ is None:
            raise RuntimeError("LBPReducer must be fitted before calling transform().")

        if len(self.lbp_cols_) == 0:
            return df.copy()

        reduced_features = pd.DataFrame(index=df.index)

        X_lbp = df.reindex(columns=self.lbp_cols_)
        X_lbp = self.imputer_.transform(X_lbp)

        X_scaled = self.scaler_.transform(X_lbp)
        X_pca = self.pca_.transform(X_scaled)

        for i in range(self.n_components_):
            reduced_features[f'lbp_pattern_pc{i+1}'] = X_pca[:, i]

        reduced_features['lbp_pattern_mean'] = X_lbp.mean(axis=1)
        reduced_features['lbp_pattern_std'] = X_lbp.std(axis=1)

        non_lbp_cols = [col for col in df.columns if not col.startswith('lbp_')]
        df_reduced = pd.concat([df[non_lbp_cols], reduced_features], axis=1)

        if self.output_columns_ is not None:
            for col in self.output_columns_:
                if col not in df_reduced.columns:
                    df_reduced[col] = 0.0
            extra = [c for c in df_reduced.columns if c not in self.output_columns_]
            if extra:
                df_reduced = df_reduced.drop(columns=extra)
            df_reduced = df_reduced[self.output_columns_]

        return df_reduced

    def fit_transform(self, df, verbose=True):
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def reduce(self, df, verbose=True):
        df_reduced = self.fit_transform(df, verbose=verbose)

        n_original = len([c for c in df.columns if c.startswith('lbp_')])
        n_reduced = len([c for c in df_reduced.columns if c.startswith('lbp_')])

        info = {
            'n_original': n_original,
            'n_reduced': n_reduced,
            'reduction_pct': (1 - n_reduced / n_original) * 100 if n_original > 0 else 0
        }

        if verbose and n_original > 0:
            print(f"\nReduction: {n_original} → {n_reduced} features")
            print(f"Removed: {info['reduction_pct']:.1f}%")
            print(f"{'='*80}\n")

        return df_reduced, info

class TwoPointCorrelationReducer:
    """
    Reducer for YOUR 2-Point Correlation features (11 → 8-9 features)
    """

    def __init__(self):
        self.s2_cols_ = None
        self.non_s2_cols_ = None
        self.aggressive_ = False
        self.structural_ = ['s2_correlation_length', 's2_baseline', 's2_slope_at_0', 's2_specific_surface']
        self.peaks_ = ['s2_first_peak_pos', 's2_first_peak_val']
        self.moment_cols_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.pca_ = None
        self.n_components_ = 0
        self.output_columns_ = None

    def fit(self, df, aggressive=False, verbose=True):
        self.aggressive_ = aggressive
        self.s2_cols_ = [col for col in df.columns if col.startswith('s2_')]
        self.non_s2_cols_ = [col for col in df.columns if col not in self.s2_cols_]

        if len(self.s2_cols_) == 0:
            self.output_columns_ = df.columns.tolist()
            return self

        if verbose:
            print(f"\n{'='*80}")
            print("2-POINT CORRELATION FEATURE REDUCTION")
            print(f"{'='*80}")
            print(f"Found {len(self.s2_cols_)} S2 features\n")

        if not aggressive:
            if verbose:
                print("ℹ️ Mode: MINIMAL REDUCTION (these features are mostly unique)")
                print(" Keeping all features - they provide complementary information")
                print(f" Tip: Use aggressive=True for more reduction\n")
            self.output_columns_ = df.columns.tolist()
            return self

        self.moment_cols_ = [c for c in [
            's2_correlation_integral', 's2_mean_excess', 's2_variance_excess',
            's2_skewness_excess', 's2_kurtosis_excess'
        ] if c in df.columns]

        if len(self.moment_cols_) >= 3:
            self.imputer_ = SimpleImputer(strategy='median')
            X_moments = self.imputer_.fit_transform(df[self.moment_cols_])

            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_moments)

            self.n_components_ = min(2, len(self.moment_cols_))
            self.pca_ = PCA(n_components=self.n_components_, random_state=42)
            self.pca_.fit(X_scaled)

        transformed = self.transform(df, verbose=False)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, df, verbose=True):
        if self.s2_cols_ is None:
            raise RuntimeError("TwoPointCorrelationReducer must be fitted before calling transform().")

        if len(self.s2_cols_) == 0:
            return df.copy()

        if not self.aggressive_:
            return df.copy()

        reduced_features = pd.DataFrame(index=df.index)

        for feat in self.structural_:
            if feat in df.columns:
                reduced_features[feat] = df[feat]
            else:
                reduced_features[feat] = 0.0

        for feat in self.peaks_:
            if feat in df.columns:
                reduced_features[feat] = df[feat]
            else:
                reduced_features[feat] = 0.0

        if self.pca_ is not None:
            X_moments = df.reindex(columns=self.moment_cols_)
            X_moments = self.imputer_.transform(X_moments)
            X_scaled = self.scaler_.transform(X_moments)
            X_pca = self.pca_.transform(X_scaled)

            for i in range(self.n_components_):
                reduced_features[f's2_moment_pc{i+1}'] = X_pca[:, i]

        non_s2_cols = [col for col in df.columns if not col.startswith('s2_')]
        df_reduced = pd.concat([df[non_s2_cols], reduced_features], axis=1)

        if self.output_columns_ is not None:
            for col in self.output_columns_:
                if col not in df_reduced.columns:
                    df_reduced[col] = 0.0
            extra = [c for c in df_reduced.columns if c not in self.output_columns_]
            if extra:
                df_reduced = df_reduced.drop(columns=extra)
            df_reduced = df_reduced[self.output_columns_]

        return df_reduced

    def fit_transform(self, df, aggressive=False, verbose=True):
        self.fit(df, aggressive=aggressive, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def reduce(self, df, aggressive=False, verbose=True):
        df_reduced = self.fit_transform(df, aggressive=aggressive, verbose=verbose)

        n_original = len([c for c in df.columns if c.startswith('s2_')])
        n_reduced = len([c for c in df_reduced.columns if c.startswith('s2_')])

        info = {
            'n_original': n_original,
            'n_reduced': n_reduced,
            'reduction_pct': (1 - n_reduced / n_original) * 100 if n_original > 0 else 0
        }

        if verbose and n_original > 0:
            print(f"\nReduction: {n_original} → {n_reduced} features")
            print(f"Removed: {info['reduction_pct']:.1f}%")
            print(f"{'='*80}\n")

        return df_reduced, info

class CustomComprehensiveReducer:
    """
    Comprehensive reducer for YOUR EXACT descriptor set.
    """

    def __init__(self, reduce_s2_aggressively=False):
        self.glcm_reducer = GLCMReducer()
        self.lbp_reducer = LBPReducer()
        self.s2_reducer = TwoPointCorrelationReducer()
        self.gram_reducer = SmartGramReducer()
        self.reduce_s2_aggressively = reduce_s2_aggressively
        self.report_ = None
        self.output_columns_ = None

    def fit(self, df, verbose=True):
        if verbose:
            print("\n" + "🚀"*40)
            print("CUSTOM COMPREHENSIVE REDUCTION")
            print("Your Exact Descriptor Set")
            print("🚀"*40 + "\n")
            print(f"Original: {df.shape[0]} samples × {df.shape[1]} features\n")

        df_reduced = df.copy()
        reduction_log = {}

        if verbose:
            print("STEP 1: GLCM Features")
        self.glcm_reducer.fit(df_reduced, verbose=verbose)
        df_reduced = self.glcm_reducer.transform(df_reduced, verbose=False)
        n0 = len([c for c in df.columns if c.startswith('glcm_')])
        n1 = len([c for c in df_reduced.columns if c.startswith('glcm_')])
        if n0 > 0:
            reduction_log['GLCM'] = {
                'n_original': n0,
                'n_reduced': n1,
                'reduction_pct': (1 - n1 / n0) * 100 if n0 > 0 else 0,
                'feature_mapping': self.glcm_reducer.feature_mapping_
            }

        if verbose:
            print("STEP 2: LBP Features")
        before_lbp = df_reduced.copy()
        self.lbp_reducer.fit(df_reduced, verbose=verbose)
        df_reduced = self.lbp_reducer.transform(df_reduced, verbose=False)
        n0 = len([c for c in before_lbp.columns if c.startswith('lbp_')])
        n1 = len([c for c in df_reduced.columns if c.startswith('lbp_')])
        if n0 > 0:
            reduction_log['LBP'] = {
                'n_original': n0,
                'n_reduced': n1,
                'reduction_pct': (1 - n1 / n0) * 100 if n0 > 0 else 0
            }

        if verbose:
            print("STEP 3: 2-Point Correlation Features")
        before_s2 = df_reduced.copy()
        self.s2_reducer.fit(df_reduced, aggressive=self.reduce_s2_aggressively, verbose=verbose)
        df_reduced = self.s2_reducer.transform(df_reduced, verbose=False)
        n0 = len([c for c in before_s2.columns if c.startswith('s2_')])
        n1 = len([c for c in df_reduced.columns if c.startswith('s2_')])
        if n0 > 0:
            reduction_log['2-Point Correlation'] = {
                'n_original': n0,
                'n_reduced': n1,
                'reduction_pct': (1 - n1 / n0) * 100 if n0 > 0 else 0
            }

        if verbose:
            print("STEP 4: Gram Matrix Features")
        before_gram = df_reduced.copy()
        self.gram_reducer.fit(df_reduced, verbose=verbose)
        df_reduced = self.gram_reducer.transform(df_reduced, verbose=False)

        n0 = len([c for c in before_gram.columns if any(
            c.startswith(prefix) for prefix in
            ['conv1_', 'layer1_', 'layer2_', 'layer3_', 'layer4_', 'cross_', 'multi_scale', 'global_']
        )])
        n1 = len([c for c in df_reduced.columns if c.startswith('gram_') or c in self.gram_reducer.top_cross_ or c in self.gram_reducer.kept_features_])
        if n0 > 0:
            reduction_log['Gram Matrix'] = {
                'n_original': n0,
                'n_reduced': n1,
                'reduction_pct': (1 - n1 / n0) * 100 if n0 > 0 else 0,
                'feature_mapping': self.gram_reducer.feature_mapping
            }

        self.output_columns_ = df_reduced.columns.tolist()
        self.report_ = {
            'reductions_applied': reduction_log,
            'original_shape': df.shape,
            'reduced_shape': df_reduced.shape
        }

        if verbose:
            self._print_summary(df, df_reduced, reduction_log)

        return self

    def transform(self, df, verbose=True):
        df_reduced = df.copy()
        df_reduced = self.glcm_reducer.transform(df_reduced, verbose=False)
        df_reduced = self.lbp_reducer.transform(df_reduced, verbose=False)
        df_reduced = self.s2_reducer.transform(df_reduced, verbose=False)
        df_reduced = self.gram_reducer.transform(df_reduced, verbose=False)

        if self.output_columns_ is not None:
            for col in self.output_columns_:
                if col not in df_reduced.columns:
                    df_reduced[col] = 0.0
            extra = [c for c in df_reduced.columns if c not in self.output_columns_]
            if extra:
                df_reduced = df_reduced.drop(columns=extra)
            df_reduced = df_reduced[self.output_columns_]

        return df_reduced

    def fit_transform(self, df, verbose=True):
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def reduce(self, df, verbose=True):
        df_reduced = self.fit_transform(df, verbose=verbose)
        return df_reduced, self.report_

    def _print_summary(self, df_original, df_reduced, reduction_log):
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80 + "\n")

        print("📊 Overall:")
        print(f" Original: {df_original.shape[1]} features")
        print(f" Reduced: {df_reduced.shape[1]} features")

        total_removed = df_original.shape[1] - df_reduced.shape[1]
        reduction_pct = (total_removed / df_original.shape[1]) * 100
        print(f" Removed: {total_removed} features ({reduction_pct:.1f}%)")

        print("\n📋 Reductions Applied:")
        for group_name, info in reduction_log.items():
            n_orig = info['n_original']
            n_red = info['n_reduced']
            pct = info['reduction_pct']
            print(f" {group_name:25s}: {n_orig:3d} → {n_red:3d} ({pct:5.1f}% reduction)")

        print("\n✅ Kept Unchanged (by design):")
        print(" Chemistry: 29 features")
        print(" Morphology: 5 features")
        print(" Scale (um_per_pixel): 1 feature")

        print("\n" + "="*80)
        print("✨ Reduction complete!")
        print("="*80 + "\n")

if __name__ == '__main__':
    import sys
    import os
    from clean_data import filter_by_subcategory

    module_path = str.rsplit(os.path.abspath(''), os.path.sep, 1)[0]
    print(f"Module path: {module_path}")
    sys.path.insert(0, module_path)

    STAINLESS_CATS   = ["Ferritic Stainless Steel", "Duplex Stainless Steel", "Austenitic Stainless Steel"]
    DP_CATS          = ["Conventional DP"]
    STUDY_CATEGORIES = STAINLESS_CATS + DP_CATS

    GROUPS = {
        "ss_sem": {"categories": STAINLESS_CATS, "methods": ["SEM: SE", "SEM: BSE"]},
        "dp_sem": {"categories": DP_CATS,        "methods": ["SEM: SE", "SEM: BSE"]},
        "ss_om":  {"categories": STAINLESS_CATS, "methods": ["OM"]},
    }

    df_full = pd.read_csv(os.path.join(module_path, "data", "descriptors_full.csv"))
    df_full = filter_by_subcategory(df_full, STUDY_CATEGORIES, drop_col=False)

    for group_name, cfg in GROUPS.items():
        df_group = df_full[
            df_full["method"].isin(cfg["methods"]) &
            df_full["Category"].isin(cfg["categories"])
        ].copy().reset_index(drop=True)

        print(f"\n{'='*60}\nProcessing: {group_name} ({len(df_group)} rows)\n{'='*60}")

        reducer = CustomComprehensiveReducer(reduce_s2_aggressively=False)
        df_reduced, report = reducer.reduce(df_group)

        out_path = os.path.join(module_path, 'data', f"descriptors_reduced_{group_name}.csv")
        df_reduced.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        
    
        
    
    
    
    
    
    
    
    
    