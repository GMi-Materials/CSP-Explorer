# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:52:29 2025

@author: gerfried.millner
"""

author = "Gerfried Millner"
version = "1.0.0"
date = "31.03.2026"
email = "gerfried.millner@ntu.edu.sg"
status = "Deliver"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from skimage import io, filters, measure, morphology, feature, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries
from scipy.spatial import KDTree, cKDTree
from numba import njit
from typing import Tuple
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import scipy.stats
from pathlib import Path

def _normalize_filename(s):
    if pd.isna(s):
        return None
    fname = os.path.basename(str(s)).strip().lower()
    return os.path.splitext(fname)[0]

def segment_image(image, min_size=50, mask=None):
    """
    Segment an SEM/TEM image using Otsu thresholding and connected-component labeling.

    Parameters:
        image (ndarray): 2D grayscale image
        min_size (int): minimum size (in pixels) of objects to keep
        mask (ndarray, optional): binary mask (True=ignore, False=keep)

    Returns:
        labeled_image (ndarray): image with labeled connected components
    """
    if mask is not None:
        image_masked = image.copy()
        image_masked[mask] = image_masked.min()
    else:
        image_masked = image
    
    if mask is not None and (~mask).sum() > 0:
        thresh = threshold_otsu(image_masked[~mask])
    else:
        thresh = threshold_otsu(image_masked)
    binary = image_masked > thresh
    
    if mask is not None:
        binary[mask] = False
    
    cleaned = morphology.remove_small_objects(binary, min_size)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    labeled_image = measure.label(cleaned)
    return labeled_image

def extract_morphology_descriptors(labeled_image, mask=None):
    """
    Extract area, equivalent diameter, aspect ratio, and eccentricity for each particle.

    Parameters:
        labeled_image (ndarray): labeled image from segment_image()
        mask (ndarray, optional): binary mask (True=ignore, False=keep)

    Returns:
        list of dicts: morphological descriptors per particle
    """
    regions = measure.regionprops(labeled_image)
    descriptors = []
    
    for r in regions:
        if mask is not None:
            coords = r.coords
            masked_pixels = np.sum(mask[coords[:, 0], coords[:, 1]])
            if masked_pixels > len(coords) * 0.5:
                continue
        descriptors.append({
            "area": r.area,
            "equivalent_diameter": r.equivalent_diameter,
            "aspect_ratio": (r.major_axis_length / (r.minor_axis_length + 1e-6)),
            "eccentricity": r.eccentricity
        })
    return descriptors

def extract_glcm_descriptors(image, distance=5, angle=0, levels=256, mask=None):
    """
    Extract GLCM texture features: contrast, energy, homogeneity.

    Parameters:
        image (ndarray): 2D grayscale image
        distance (int): pixel distance for GLCM
        angle (float): angle in radians for GLCM
        levels (int): number of gray levels
        mask (ndarray, optional): binary mask (True=ignore, False=keep)

    Returns:
        dict: GLCM texture descriptors
    """
    if mask is not None:
        image_masked = image.copy()
        image_masked[mask] = 0
        roi_mask = ~mask
    else:
        image_masked = image
        roi_mask = None
    
    if roi_mask is not None and hasattr(graycomatrix, 'mask'):
        try:
            glcm = graycomatrix(image_masked, distances=[distance], angles=[angle], 
                              levels=levels, symmetric=True, normed=True, mask=roi_mask)
        except TypeError:
            glcm = graycomatrix(image_masked, distances=[distance], angles=[angle], 
                              levels=levels, symmetric=True, normed=True)
    else:
        glcm = graycomatrix(image_masked, distances=[distance], angles=[angle], 
                          levels=levels, symmetric=True, normed=True)
    
    result = {
        "glcm_contrast": graycoprops(glcm, 'contrast')[0, 0],
        "glcm_homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
        "glcm_energy": graycoprops(glcm, 'energy')[0, 0],
    }
    
    try:
        angles_multi = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        distances_multi = [1, 3, 5]
        
        for dist in distances_multi:
            contrasts_by_angle = []
            homogeneities_by_angle = []
            
            for ang in angles_multi:
                glcm_ang = graycomatrix(image_masked, distances=[dist], angles=[ang], 
                                       levels=levels, symmetric=True, normed=True)
                contrasts_by_angle.append(graycoprops(glcm_ang, 'contrast')[0, 0])
                homogeneities_by_angle.append(graycoprops(glcm_ang, 'homogeneity')[0, 0])
            
            mean_contrast = np.mean(contrasts_by_angle)
            max_contrast = np.max(contrasts_by_angle)
            min_contrast = np.min(contrasts_by_angle)
            
            if mean_contrast > 1e-4:
                # Use coefficient of variation instead of raw ratio
                cv = np.std(contrasts_by_angle) / (mean_contrast + 1e-10)
                result[f'glcm_anisotropy_d{dist}'] = float(np.clip(cv, 0.0, 10.0))
            else:
                result[f'glcm_anisotropy_d{dist}'] = 0.0
            result[f'glcm_homogeneity_anisotropy_d{dist}'] = np.std(homogeneities_by_angle)
            result[f'glcm_max_contrast_d{dist}'] = max_contrast
            result[f'glcm_min_contrast_d{dist}'] = min_contrast
        
        result['glcm_anisotropy_index'] = result['glcm_anisotropy_d5']
        result['glcm_homogeneity_anisotropy'] = result['glcm_homogeneity_anisotropy_d5']
        
        d5 = result['glcm_anisotropy_d5']
        d1 = result['glcm_anisotropy_d1']
        
        if d5 > 1e-4:  # More conservative threshold
            ratio = d1 / d5
            # Clip to prevent extreme values (range: 0.01 to 100)
            result['glcm_anisotropy_ratio'] = float(np.clip(ratio, 0.01, 100.0))
        else:
            # If d5 is essentially zero, default to 1 (isotropic)
            result['glcm_anisotropy_ratio'] = 1.0
        
    except Exception as e:
        for dist in [1, 3, 5]:
            result[f'glcm_anisotropy_d{dist}'] = np.nan
            result[f'glcm_homogeneity_anisotropy_d{dist}'] = np.nan
            result[f'glcm_max_contrast_d{dist}'] = np.nan
            result[f'glcm_min_contrast_d{dist}'] = np.nan
        result['glcm_anisotropy_index'] = np.nan
        result['glcm_homogeneity_anisotropy'] = np.nan
        result['glcm_anisotropy_ratio'] = np.nan
    
    try:
        distances_multi = [1, 3, 5]
        homogeneities_by_dist = []
        
        for dist in distances_multi:
            glcm_dist = graycomatrix(image_masked, distances=[dist], angles=[angle], 
                                    levels=levels, symmetric=True, normed=True)
            homogeneities_by_dist.append(graycoprops(glcm_dist, 'homogeneity')[0, 0])
        
        result['glcm_homogeneity_d1'] = homogeneities_by_dist[0]
        result['glcm_homogeneity_d3'] = homogeneities_by_dist[1]
        result['glcm_homogeneity_d5'] = homogeneities_by_dist[2]
        
    except:
        result['glcm_homogeneity_d1'] = np.nan
        result['glcm_homogeneity_d3'] = np.nan
        result['glcm_homogeneity_d5'] = np.nan
    
    try:
        result['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    except:
        result['glcm_correlation'] = np.nan
    
    return result

def extract_lbp_histogram(image, P=8, R=1, mask=None):
    """
    Extract Local Binary Pattern histogram with robust normalization.
    """
    lbp = local_binary_pattern(image, P=P, R=R, method='uniform')
    
    if mask is not None:
        lbp_valid = lbp[~mask]
        if len(lbp_valid) == 0:
            return np.zeros(P + 2)
    else:
        lbp_valid = lbp.flatten()
    
    hist, _ = np.histogram(lbp_valid, bins=np.arange(0, P + 3), range=(0, P + 2))
    
    # Robust normalization
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist.astype(np.float64) / hist_sum
    else:
        hist = np.ones(P + 2) / (P + 2)  # Uniform if empty
    
    return hist

def plot_segmentation(image, labeled_image, mask=None):
    """
    Plot original image and segmented image side by side.
    
    Parameters:
        image (ndarray): original grayscale image
        labeled_image (ndarray): labeled segmentation result
        mask (ndarray, optional): binary mask to overlay
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image, cmap='gray')
    if mask is not None:
        masked_overlay = np.zeros((*image.shape, 4))
        masked_overlay[mask] = [1, 0, 0, 0.3]
        ax[0].imshow(masked_overlay)
    ax[0].set_title("Original Image" + (" (with mask overlay)" if mask is not None else ""))
    
    ax[1].imshow(labeled_image, cmap='nipy_spectral')
    ax[1].set_title("Segmented Particles")
    
    for a in ax: 
        a.axis('off')
    plt.show()

##############################################################################
#################### HYBRID ADAPTIVE PREPROCESSING ###########################
##############################################################################

def classify_with_confidence(image_gray, mask=None, filename="", subfolder=""):
    """
    Classify SEM image type with confidence score for adaptive preprocessing.
    
    Parameters:
        image_gray (ndarray): Grayscale uint8 image
        mask (ndarray, optional): Binary mask (True=ignore, False=keep)
        filename (str): Image filename (for metadata-based rules)
        subfolder (str): Subfolder name (for metadata-based rules)
    
    Returns:
        img_type (str): Classified image type
        confidence (float): Confidence score (0-1)
        features (dict): Classification features used
    """
    if mask is not None:
        img_masked = image_gray.copy()
        img_masked[mask] = 0
        if mask.sum() >= 0.95 * mask.size:  # Almost entirely masked
            return 'unknown', 0.0, {}
        analysis_region = img_masked[~mask]
    else:
        analysis_region = image_gray.flatten()
    
    if np.std(analysis_region) < 1.0:
        return 'uniform', 1.0, {'std': np.std(analysis_region)}
    
    imgf = (image_gray.astype(np.float32) / 255.0)
    edges = filters.sobel(imgf)
    if mask is not None:
        edges[mask] = 0.0
    thr_e = np.percentile(edges, 90)
    edge_density = float((edges > thr_e).mean())
    
    glcm = graycomatrix(image_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    gy, gx = np.gradient(image_gray.astype(float))
    Jxx = np.sum(gx * gx)
    Jyy = np.sum(gy * gy)
    Jxy = np.sum(gx * gy)
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    lambda1 = (trace + np.sqrt(trace**2 - 4*det + 1e-10)) / 2
    lambda2 = (trace - np.sqrt(trace**2 - 4*det + 1e-10)) / 2
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
    
    hist, _ = np.histogram(analysis_region, bins=256, range=(0, 256))
    hist_peaks = len([i for i in range(1, len(hist)-1) 
                     if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > hist.max()*0.05])
    
    confidence = 1.0
    img_type = None
    
    if 'fracture' in filename.lower() or 'Xu2014' in subfolder:
        img_type = 'fracture'
        confidence = 0.95
    elif coherence > 0.40:
        img_type = 'directional'
        confidence = 0.90
    elif edge_density > 0.50 and coherence < 0.12:
        img_type = 'fracture'
        confidence = 0.85
    elif edge_density < 0.25 and contrast > 1000:
        img_type = 'coarse_grained'
        confidence = 0.85
    elif coherence > 0.25:
        img_type = 'directional'
        confidence = 0.60
    elif edge_density > 0.45 and hist_peaks >= 3:
        img_type = 'multiphase'
        confidence = 0.65
    elif edge_density > 0.40 and coherence < 0.15:
        img_type = 'fracture'
        confidence = 0.60
    elif contrast > 2500:
        img_type = 'coarse_grained'
        confidence = 0.55
    else:
        img_type = 'multiphase'
        confidence = 0.40
    
    features = {
        'coherence': coherence,
        'edge_density': edge_density,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'hist_peaks': hist_peaks
    }
    
    return img_type, confidence, features

def analyze_SEM_images(folder_path, output_csv=None,
                      min_size=50, glcm_distance=5, glcm_angle=0,
                      lbp_P=8, lbp_R=1, mask_folder=None, mask_suffix="_mask"):
    """
    Analyze all SEM images in a folder and extract morphology, GLCM, and LBP descriptors.

    Parameters:
        folder_path (str): Path to the folder containing images (tif, png, jpg)
        output_csv (str): Path to the CSV file to save the results, if None, results are not saved
        min_size, glcm_distance, glcm_angle, lbp_P, lbp_R: parameters passed to the functions
        mask_folder (str, optional): Path to folder containing mask files (.npy)
        mask_suffix (str): Suffix for mask files (default "_mask")

    Returns:
        results (list of dicts): extracted features for each image
    """
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]

    results = []

    for filename in tqdm(image_files, desc="Analyzing images", unit="file"):
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)

        mask = None
        if mask_folder is not None:
            base_name, ext = os.path.splitext(filename)
            mask_name = f"{base_name}{mask_suffix}.npy"
            mask_path = os.path.join(mask_folder, mask_name)
            
            if os.path.exists(mask_path):
                try:
                    mask = np.load(mask_path)
                    if mask.dtype != bool:
                        mask = mask.astype(bool)
                except Exception as e:
                    print(f"[WARNING] Failed to load mask {mask_name}: {e}")
            else:
                print(f"[WARNING] Mask file not found: {mask_name}")

        if image.ndim == 3:
            image = rgb2gray(image)
        image = img_as_ubyte(image)

        labeled = segment_image(image, min_size=min_size, mask=mask)

        morph = extract_morphology_descriptors(labeled, mask=mask)
        if len(morph) > 0:
            morph_avg = {
                "morph_area_mean": np.mean([m["area"] for m in morph]),
                "morph_diameter_mean": np.mean([m["equivalent_diameter"] for m in morph]),
                "morph_aspect_ratio_mean": np.mean([m["aspect_ratio"] for m in morph]),
                "morph_eccentricity_mean": np.mean([m["eccentricity"] for m in morph]),
            }
        else:
            morph_avg = {"morph_area_mean": np.nan, "morph_diameter_mean": np.nan,
                         "morph_aspect_ratio_mean": np.nan, "morph_eccentricity_mean": np.nan}

        glcm_feats = extract_glcm_descriptors(image,
                                              distance=glcm_distance,
                                              angle=glcm_angle,
                                              mask=mask)

        lbp_hist = extract_lbp_histogram(image, P=lbp_P, R=lbp_R, mask=mask)

        result = {"image_filename": filename}
        result.update(morph_avg)
        result.update(glcm_feats)
        for i, val in enumerate(lbp_hist):
            result[f"lbp_p{lbp_P}r{lbp_R}_bin_{i}"] = val

        results.append(result)
        
    if output_csv is not None:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Analysis complete! Results saved to: {output_csv}")
        
    return results


#################### 2-points-correlation ###########################

def get_feature_coordinates(image: np.ndarray, 
                          use_boundaries: bool = True,
                          mask: np.ndarray = None) -> np.ndarray:
    """
    Extract feature coordinates from binary or labeled image.
    
    Args:
        image: Input image (binary or labeled)
        use_boundaries: Whether to extract boundaries (True) or all feature pixels (False)
        mask: Optional mask (True=ignore, False=keep)
    
    Returns:
        Array of (y,x) coordinates
    """
    if use_boundaries and (image.dtype == bool or np.issubdtype(image.dtype, np.integer)):
        boundaries = find_boundaries(image.astype(np.uint8), mode='outer')
        coords = np.column_stack(np.where(boundaries))
    else:
        coords = np.column_stack(np.where(image > 0))

    if mask is not None:
        valid_mask = ~mask
        valid_coords = []
        for coord in coords:
            y, x = coord
            if valid_mask[y, x]:
                valid_coords.append(coord)
        coords = np.array(valid_coords) if valid_coords else np.array([]).reshape(0, 2)
    
    return coords

def calculate_correlation(
    coords: np.ndarray,
    max_distance: float = 200.0,
    num_bins: int = 50,
    batch_size: int = 2000,
    show_progress: bool = True,
    mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrected and optimized n-point correlation function.
    
    Args:
        coords: Array of (y,x) coordinates
        max_distance: Maximum correlation distance
        num_bins: Number of distance bins
        batch_size: Batch size for processing
        show_progress: Show progress bar
        mask: Optional mask (True=ignore, False=keep) - coords should already be filtered
    """
    if len(coords) < 2:
        return np.array([]), np.array([])

    coords = coords.astype(np.float32)
    tree = KDTree(coords)
    bins = np.linspace(0, max_distance, num_bins + 1, dtype=np.float32)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist = np.zeros(num_bins, dtype=np.uint64)

    @njit(fastmath=True)
    def _fast_histogram(dists, bins, hist):
        for d in dists:
            for i in range(len(bins)-1):
                if bins[i] <= d < bins[i+1]:
                    hist[i] += 1
                    break
        return hist

    n_points = len(coords)
    n_batches = int(np.ceil(n_points / batch_size))
    pbar = tqdm(total=n_batches, desc="Processing", disable=not show_progress)

    try:
        for i in range(0, n_points, batch_size):
            batch = coords[i:i + batch_size]
            indices = tree.query_ball_point(batch, max_distance)
            dists = []
            for j, neighbors in enumerate(indices):
                if len(neighbors) > 0:
                    diff = coords[neighbors] - batch[j]
                    sq_dists = np.sum(diff**2, axis=1)
                    valid_dists = np.sqrt(sq_dists[sq_dists > 0])
                    dists.extend(valid_dists)
            
            if len(dists) > 0:
                hist = _fast_histogram(np.array(dists), bins, hist)
            
            pbar.update(1)
    finally:
        pbar.close()

    annulus_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    correlation = hist / (annulus_areas + 1e-10)
    if np.max(correlation) > 0:
        correlation /= np.max(correlation)

    return bin_centers, correlation

def subsampled_correlation(coords, max_distance=200, num_bins=50, 
                          subsample_frac=0.1, min_samples=1000, 
                          n_neighbors=30, random_state=None,
                          mask=None):
    """
    Compute 2-point correlation with stratified subsampling.
    """
    rng = np.random.RandomState(random_state)
    N = len(coords)
    sample_size = max(min_samples, int(N * subsample_frac))
    
    if N < 2:
        return np.array([]), np.array([])

    if N > sample_size:
        x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 10)
        y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 10)

        stratified_indices = []
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                in_cell = ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i + 1]) &
                           (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j + 1]))
                cell_indices = np.where(in_cell)[0]
                if cell_indices.size:
                    n_cell = max(1, cell_indices.size * sample_size // N)
                    stratified_indices.extend(
                        rng.choice(cell_indices, min(n_cell, cell_indices.size), replace=False)
                    )

        idx = np.unique(np.asarray(stratified_indices, dtype=int))
        if idx.size < sample_size:
            rest = np.setdiff1d(np.arange(N), idx)
            need = sample_size - idx.size
            if rest.size:
                idx = np.concatenate([idx, rng.choice(rest, min(need, rest.size), replace=False)])

        if idx.size < 2:
            idx = rng.choice(np.arange(N), min(sample_size, N), replace=False)

        coords = coords[idx[:sample_size]]
    if len(coords) < 2:
        return np.array([]), np.array([])    
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist = np.zeros(num_bins)
    
    tree = cKDTree(coords)
    for i in tqdm(range(len(coords)), desc="Calculating correlations"):
        neighbor_indices = tree.query_ball_point(coords[i], max_distance)
        if len(neighbor_indices) > 1:
            distances = np.linalg.norm(coords[neighbor_indices] - coords[i], axis=1)
            valid_dists = distances[distances > 0]
            hist += np.histogram(valid_dists, bins=bins)[0]
    
    annulus_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    correlation = hist / (annulus_areas * len(coords) + 1e-12)
    correlation = correlation / np.max(correlation) if np.max(correlation) > 0 else np.zeros_like(bin_centers)

    return bin_centers, correlation

class S2Descriptors:
    def __init__(self, r, S2):
        """
        r   : numpy array of distances
        S2  : numpy array of normalized 2-point correlation values (S2(0)=1)
        """
        self.r = np.asarray(r)
        self.S2 = np.asarray(S2)

    def baseline(self):
        n = max(5, int(0.05 * len(self.S2)))
        return np.mean(self.S2[-n:])

    def excess_S2(self):
        base = self.baseline()
        return self.S2 - base

    def baseline_value(self):
        return self.baseline()
    
    def correlation_integral(self):
        return np.trapezoid(self.excess_S2(), self.r)
    
    def correlation_length(self):
        """Correlation length as a characteristic decay distance (robust to zero-crossings)."""
        excess = self.excess_S2()
        r = self.r
        if excess is None or r is None or len(r) == 0:
            return np.nan
    
        # Prevent cancellation if excess crosses 0
        w = np.maximum(excess, 0.0)
        if not np.any(w > 0):
            w = np.abs(excess)
    
        denom = np.trapezoid(w, r)
        if not np.isfinite(denom) or denom <= 1e-12:
            return np.nan
    
        xi = np.trapezoid(r * w, r) / denom
    
        rmax = float(np.nanmax(r))
        if not np.isfinite(xi) or xi <= 0 or (np.isfinite(rmax) and xi > rmax):
            return np.nan
    
        return float(xi)

    def slope_at_origin(self):
        """
        Robust slope estimation using linear fit of first N points.
        Prevents noise-induced positive slopes.
        """
        # Use enough points to average out noise, but stay near origin
        n_fit = min(10, len(self.r) // 4)  # Use first 25% of points, max 10
        
        if n_fit < 3:
            return np.nan  # Not enough points for reliable fit
        
        # Weighted linear fit (give more weight to points near origin)
        weights = np.exp(-self.r[:n_fit] / self.r[n_fit])  # Exponential decay weights
        
        try:
            coeffs = np.polyfit(self.r[:n_fit], self.S2[:n_fit], deg=1, w=weights)
            slope = coeffs[0]
            
            # Safety check: S2 MUST decay (negative slope)
            if slope >= 0:
                # This indicates numerical issue or insufficient sampling
                return np.nan
            
            return slope
            
        except:
            return np.nan

    def specific_surface(self):
        return -4 * self.slope_at_origin()

    def first_peak_properties(self):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self.S2)
        if len(peaks) > 0:
            idx = peaks[0]
            return self.r[idx], self.S2[idx]
        else:
            return np.nan, np.nan

    def moments(self):
        """Statistical moments with division-by-zero protection."""
        excess = self.excess_S2()
        # Check for valid data
        if len(excess) < 2 or np.std(excess) < 1e-10:
            return 0.0, 0.0, 0.0, 0.0
        mean = np.mean(excess)
        var = np.var(excess)
        std = np.std(excess)
        
        # Protected skewness and kurtosis
        skew = np.mean((excess - mean)**3) / (std**3 + 1e-12)
        kurt = np.mean((excess - mean)**4) / (var**2 + 1e-12)
        
        # Clip to reasonable ranges
        skew = np.clip(skew, -100, 100)
        kurt = np.clip(kurt, -100, 100)
        
        return mean, var, skew, kurt

    def feature_vector(self):
        peak_r, peak_val = self.first_peak_properties()
        mean, var, skew, kurt = self.moments()
        return {
            "s2_baseline": self.baseline_value(),
            "s2_correlation_integral": self.correlation_integral(),
            "s2_correlation_length": self.correlation_length(),
            "s2_slope_at_0": self.slope_at_origin(),
            "s2_specific_surface": self.specific_surface(),
            "s2_first_peak_pos": peak_r,
            "s2_first_peak_val": peak_val,
            "s2_mean_excess": mean,
            "s2_variance_excess": var,
            "s2_skewness_excess": skew,
            "s2_kurtosis_excess": kurt,
        }
    
    def plot_descriptor(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(self.r, self.S2, label=r"$S_2(r)$", lw=2)

        base = self.baseline()
        corr_len = self.correlation_length()
        ax.axhline(base, color='r', linestyle='--', label='Baseline', lw=1.5)
        ax.axvline(corr_len, color='g', linestyle=':', label=r'Correlation length $\xi$', lw=1.5)

        ax.set_xlabel("Distance r")
        ax.set_ylabel("S2(r)")
        ax.set_title("2-point correlation function descriptors")
        ax.legend()

        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.grid(True, which='major', linestyle='--', color='gray', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', color='gray', alpha=0.5)

        return ax


def calc_2p_correlation(image: np.ndarray, 
                        max_distance: float = 200, 
                        subsample_frac: float = 0.2, 
                        min_samples: int = 5000, 
                        random_state: int = 42,
                        mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 2-point correlation fucntion for a given image, using a subsample of the whole image.
    """
    binary = image > threshold_otsu(image)
    coords = get_feature_coordinates(binary, use_boundaries=True, mask=mask)
    distances, correlation = subsampled_correlation(
        coords=coords,
        max_distance=max_distance,
        subsample_frac=subsample_frac,
        min_samples=min_samples,
        random_state=random_state,
        mask=mask
    )
    return distances, correlation

def descriptor_2p(image: np.ndarray, 
                        max_distance: float = 200, 
                        subsample_frac: float = 0.2, 
                        min_samples: int = 5000, 
                        random_state: int = 42,
                        mask: np.ndarray = None) -> dict:
    """
    Calculate 2-point correlation descriptors for a given image, using a subsample of the whole image.
    """
    distances, correlation = calc_2p_correlation(
        image=image,
        max_distance=max_distance,
        subsample_frac=subsample_frac,
        min_samples=min_samples,
        random_state=random_state,
        mask=mask
    )
    desc = S2Descriptors(distances, correlation)
    return desc, desc.feature_vector()

class GramDescriptorExtractor:
    """
    Extracts Gram matrix-based descriptors from images using a pre-trained ResNet model.
    
    Enhanced version includes:
    - Within-layer Gram statistics (38 features per layer)
    - Cross-layer correlations (multi-scale interactions)
    - Eigenvalue decomposition (principal texture patterns)
    - Global multi-scale aggregation
    
    Total features: ~215-240 depending on configuration
    """

    def __init__(self, device=None, image_size=512, layers=None):
        """
        Initializes the descriptor extractor with ResNet-50.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load ResNet-50
        resnet = models.resnet50(pretrained=True).to(self.device).eval()
        
        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.selected_layers = layers or {
            '0': 'conv1',      # Initial convolution
            '4': 'layer1',     # First residual block
            '5': 'layer2',     # Second residual block  
            '6': 'layer3',     # Third residual block
            '7': 'layer4'      # Fourth residual block
        }
    
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_image_array(self, img_array, mask=None):
        """
        Converts a raw image array (HWC, RGB or grayscale) to a preprocessed tensor.
        """
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)

        if mask is not None:
            img_array = img_array.copy()
            img_array[mask] = 0

        image = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def load_image(self, img_path, mask=None):
        """
        Loads and preprocesses an image from disk.
        """
        image = Image.open(img_path).convert('RGB')
        img_array = np.array(image)
        
        if mask is not None:
            img_array = img_array.copy()
            img_array[mask] = 0
            image = Image.fromarray(img_array)
        
        return self.transform(image).unsqueeze(0).to(self.device)

    def get_features(self, image, mask=None):
        """
        Extracts intermediate CNN features from the image.
        """
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features

    def gram_matrix(self, tensor, mask=None):
        """
        Computes the Gram matrix from a feature tensor.
        """
        b, c, h, w = tensor.size()
        
        if mask is not None:
            mask_resized = torch.as_tensor(mask, device=tensor.device, dtype=torch.float32)
            mask_resized = torch.nn.functional.interpolate(
                mask_resized.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='nearest'
            ).squeeze(0).squeeze(0).bool()
            mask_expanded = mask_resized.unsqueeze(0).expand(c, -1, -1)
            tensor_masked = tensor.squeeze(0).clone()
            tensor_masked[mask_expanded] = 0
            features = tensor_masked.view(c, h * w)
            valid_pixels = (~mask_resized).sum().float()
            normalization = c * valid_pixels if valid_pixels > 0 else c * h * w
        else:
            features = tensor.view(c, h * w)
            normalization = c * h * w
        
        G = torch.mm(features, features.t())
        return G / normalization

    def summarize_gram_matrix(self, gram, layer_name="", mask=None, top_k=5):
        """
        Comprehensive Gram matrix summary with maximum interpretability.
        """
        diag = np.diag(gram)
        triu_idx = np.triu_indices_from(gram, k=1)
        off_diag = gram[triu_idx]
        flat = gram.flatten()
        
        features = {}
        
        features.update({
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "skew": float(scipy.stats.skew(flat)),
            "kurtosis": float(scipy.stats.kurtosis(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        })
        
        quantiles = np.percentile(flat, [25, 50, 75])
        features.update({
            "q25": float(quantiles[0]),
            "q50_median": float(quantiles[1]),
            "q75": float(quantiles[2]),
            "iqr": float(quantiles[2] - quantiles[0]),
        })
        
        features.update({
            "diag_mean": float(np.mean(diag)),
            "diag_std": float(np.std(diag)),
            "diag_max": float(np.max(diag)),
            "diag_min": float(np.min(diag)),
        })
        
        features.update({
            "offdiag_mean": float(np.mean(off_diag)),
            "offdiag_std": float(np.std(off_diag)),
            "offdiag_max": float(np.max(off_diag)),
            "offdiag_min": float(np.min(off_diag)),
        })
        
        features.update({
            "trace": float(np.trace(gram)),
            "frobenius_norm": float(np.linalg.norm(gram, 'fro')),
            "spectral_norm": float(np.linalg.norm(gram, 2)),
            "determinant": float(np.linalg.det(gram)),
        })
        
        try:
            sign, logdet = np.linalg.slogdet(gram)
            
            if sign <= 0:
                # Not positive definite (numerical issue)
                features["log_determinant"] = -100.0
            elif np.isinf(logdet):
                # Explicitly infinite
                features["log_determinant"] = -100.0 if logdet < 0 else 100.0
            elif logdet < -100 or logdet > 100:
                # Clip to reasonable range
                features["log_determinant"] = float(np.clip(logdet, -100, 100))
            else:
                features["log_determinant"] = float(logdet)
        except:
            features["log_determinant"] = -100.0
        
        flat_pos = np.abs(flat) + 1e-12
        flat_normalized = flat_pos / np.sum(flat_pos)
        
        features.update({
            "entropy": float(scipy.stats.entropy(flat_normalized)),
            "gini": float(self._gini_coefficient(flat)),
        })
        
        threshold = 1e-6 * np.max(np.abs(flat))
        n_sparse = np.sum(np.abs(flat) < threshold)
        
        features.update({
            "sparsity": float(n_sparse / len(flat)),
            "offdiag_sparsity": float(np.sum(np.abs(off_diag) < threshold) / len(off_diag)),
            "l1_l2_ratio": float(np.linalg.norm(flat, 1) / (np.linalg.norm(flat, 2) + 1e-12)),
        })
        
        diag_mean = np.mean(diag)
        offdiag_mean = np.mean(off_diag)
        std_val = np.std(flat)
        
        # Diagonal to off-diagonal ratio
        if abs(offdiag_mean) > 1e-6:
            diag_offdiag = diag_mean / offdiag_mean
            features["diag_to_offdiag_ratio"] = float(np.clip(diag_offdiag, -1000, 1000))
        else:
            features["diag_to_offdiag_ratio"] = 1.0
        
        # Range to std ratio
        if std_val > 1e-6:
            range_std = (np.max(flat) - np.min(flat)) / std_val
            features["range_to_std_ratio"] = float(np.clip(range_std, 0, 1000))
        else:
            features["range_to_std_ratio"] = 0.0
        
        try:
            eigenvalues = np.linalg.eigvalsh(gram)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
            
            for i in range(min(top_k, len(eigenvalues))):
                features[f"eigen_{i+1}"] = float(eigenvalues[i])
            
            features.update({
                "eigen_sum": float(np.sum(eigenvalues_pos)),
                "eigen_max": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
                "eigen_min": float(eigenvalues_pos[-1]) if len(eigenvalues_pos) > 0 else 0.0,
            })
            
            if len(eigenvalues_pos) > 1:
                max_eigen = eigenvalues[0]
                min_eigen = eigenvalues_pos[-1]
                
                # Use relative tolerance based on max eigenvalue
                min_eigen_safe = max(min_eigen, max_eigen * 1e-10)
                
                # Calculate condition number
                condition_number = max_eigen / min_eigen_safe
                
                # Cap at 1e15 (standard upper bound)
                condition_number = min(condition_number, 1e15)
                
                features["condition_number"] = float(condition_number)
                
                # Eigenvalue entropy and effective rank
                normalized_eigs = eigenvalues_pos / (np.sum(eigenvalues_pos) + 1e-12)
                features.update({
                    "eigen_entropy": float(scipy.stats.entropy(normalized_eigs)),
                    "effective_rank": float(np.exp(scipy.stats.entropy(normalized_eigs))),
                })
                
            elif len(eigenvalues_pos) == 1:
                # Only one significant eigenvalue - rank 1 matrix
                features["condition_number"] = 1.0
                features["eigen_entropy"] = 0.0
                features["effective_rank"] = 1.0
            else:
                # No positive eigenvalues - degenerate
                features["condition_number"] = np.nan
                features["eigen_entropy"] = np.nan
                features["effective_rank"] = np.nan
            
        except np.linalg.LinAlgError as e:
            for i in range(top_k):
                features[f"eigen_{i+1}"] = np.nan
            features.update({
                "eigen_sum": np.nan,
                "eigen_max": np.nan,
                "eigen_min": np.nan,
                "eigen_entropy": np.nan,
                "effective_rank": np.nan,
                "condition_number": np.nan,
            })
        
        return features

    def _gini_coefficient(self, x):
        """
        Calculate Gini coefficient for texture inequality.
        """
        x = np.sort(np.abs(x))
        n = len(x)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n)

    def compute_cross_layer_features(self, features_dict, mask=None):
        """
        Compute cross-layer correlations to capture multi-scale interactions.
        """
        cross_features = {}
        layer_names = list(features_dict.keys())
        
        grams = {}
        for layer in layer_names:
            grams[layer] = self.gram_matrix(features_dict[layer], mask=mask).detach().cpu().numpy()
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i < j:
                    G1 = grams[layer1]
                    G2 = grams[layer2]
                    
                    if G1.shape != G2.shape:
                        min_size = min(G1.shape[0], G2.shape[0])
                        G1_crop = G1[:min_size, :min_size]
                        G2_crop = G2[:min_size, :min_size]
                    else:
                        G1_crop = G1
                        G2_crop = G2
                    
                    cross_corr = np.sum(G1_crop * G2_crop) / (
                        np.linalg.norm(G1_crop, 'fro') * np.linalg.norm(G2_crop, 'fro') + 1e-12
                    )
                    
                    cos_sim = np.sum(G1_crop.flatten() * G2_crop.flatten()) / (
                        np.linalg.norm(G1_crop.flatten()) * np.linalg.norm(G2_crop.flatten()) + 1e-12
                    )
                    
                    feature_name = f"cross_{layer1}_{layer2}"
                    cross_features[f"{feature_name}_frobenius_corr"] = float(cross_corr)
                    cross_features[f"{feature_name}_cosine_sim"] = float(cos_sim)
        
        return cross_features

    def extract(self, img_array, mask=None):
        """
        Extracts Gram matrix descriptors from a raw image array.
        """
        image = self._preprocess_image_array(img_array, mask=mask)
        features = self.get_features(image, mask=mask)
        grams = {layer: self.gram_matrix(features[layer], mask=mask).detach().cpu().numpy()
                 for layer in features}
        return grams

    def extract_flat(self, img_array, mask=None):
        """
        Extracts a flattened Gram matrix vector from a raw image array.
        """
        image = self._preprocess_image_array(img_array, mask=mask)
        features = self.get_features(image, mask=mask)
        grams = [self.gram_matrix(features[layer], mask=mask).view(-1) for layer in features]
        return torch.cat(grams).detach().cpu().numpy()

    def extract_dataframe(self, img_array, image_id="", mask=None, 
                         top_k_eigen=5, include_cross_layer=True):
        """
        Extracts comprehensive Gram matrix summaries from a raw image array.
        """
        image = self._preprocess_image_array(img_array, mask=mask)
        features_dict = self.get_features(image, mask=mask)
        summary_features = {}
        
        for layer in features_dict:
            G = self.gram_matrix(features_dict[layer], mask=mask).detach().cpu().numpy()
            stats = self.summarize_gram_matrix(
                G, 
                layer_name=layer, 
                mask=mask, 
                top_k=top_k_eigen
            )
            for stat_name, value in stats.items():
                summary_features[f"{layer}_{stat_name}"] = value
        
        if include_cross_layer:
            cross_features = self.compute_cross_layer_features(features_dict, mask=mask)
            summary_features.update(cross_features)
        
        all_traces = [summary_features[f"{layer}_trace"] 
                      for layer in features_dict if f"{layer}_trace" in summary_features]
        all_entropies = [summary_features[f"{layer}_entropy"] 
                         for layer in features_dict if f"{layer}_entropy" in summary_features]
        all_diag_means = [summary_features[f"{layer}_diag_mean"] 
                          for layer in features_dict if f"{layer}_diag_mean" in summary_features]
        
        summary_features.update({
            "global_trace_mean": float(np.mean(all_traces)),
            "global_trace_std": float(np.std(all_traces)),
            "global_entropy_mean": float(np.mean(all_entropies)),
            "global_texture_strength": float(np.mean(all_diag_means)),
            "multi_scale_consistency": float(1.0 / (np.std(all_diag_means) + 1e-12)),
        })
        
        df = pd.DataFrame([summary_features])
        df.insert(0, "id", image_id)
        return df

    def from_folder(self, folder_path, extensions=('.jpg', '.jpeg', '.png', '.tif', '.bmp'), 
                   mask=None, top_k_eigen=5, include_cross_layer=True):
        """
        Processes all image files in a folder and returns a descriptor DataFrame.
        """
        from skimage import io

        image_files = [os.path.join(folder_path, f)
                       for f in os.listdir(folder_path)
                       if f.lower().endswith(extensions)]

        rows = []
        for img_path in tqdm(image_files, desc="Extracting Gram descriptors"):
            try:
                img_array = io.imread(img_path)
                df_row = self.extract_dataframe(
                    img_array, 
                    image_id=os.path.basename(img_path), 
                    mask=mask,
                    top_k_eigen=top_k_eigen,
                    include_cross_layer=include_cross_layer
                )
                rows.append(df_row)
            except Exception as e:
                print(f"⚠️ Skipped {img_path}: {e}")

        if rows:
            return pd.concat(rows, ignore_index=True)
        else:
            return pd.DataFrame()

##############################################################################

def extract_all_descriptors(
    image: np.ndarray,
    filename: str = "unknown",
    subfolder: str = "",
    gram_extractor=None,
    min_size: int = 50,
    glcm_distance: int = 5,
    glcm_angle: float = 0,
    lbp_P: int = 8,
    lbp_R: int = 1,
    max_distance: float = 200,
    subsample_frac: float = 0.2,
    min_samples: int = 5000,
    random_state: int = 42,
    verbose: bool = False,
    mask: np.ndarray = None,
    use_adaptive: bool = True,
    confidence_threshold: float = 0.55
) -> dict:
    """
    Extracts all descriptor types from a single SEM image.
    """
    result = {"image_filename": filename, "subfolder": subfolder}

    if image.ndim == 3 and image.shape[2] == 4:
        if verbose:
            print(f"[INFO] Converting RGBA to RGB for {filename}")
        image = image[:, :, :3]
    
    if image.ndim == 3:
        image_gray = rgb2gray(image)
    else:
        image_gray = image
    image_gray = img_as_ubyte(image_gray)

    if use_adaptive:
        img_type, confidence, class_features = classify_with_confidence(
            image_gray, mask, filename, subfolder
        )
        
        result['cls_type'] = img_type
        result['cls_confidence'] = confidence
        
        if verbose:
            print(f"[INFO] Classified as {img_type} (confidence: {confidence:.2f})")
        
        if confidence >= confidence_threshold:
            if verbose:
                print("[INFO] Using adaptive segmentation method")
            params_map = {
                'fracture': {'min_size': 200, 'invert': False},
                'directional': {'min_size': 100, 'invert': False},
                'multiphase': {'min_size': 50, 'invert': False},
                'coarse_grained': {'min_size': 300, 'invert': True},
                'fine_grained': {'min_size': 30, 'invert': False},
                'uniform': {'min_size': 50, 'invert': False},
                'unknown': {'min_size': 50, 'invert': False},
            }
            params = params_map.get(img_type, {'min_size': 50, 'invert': False})
            
            if mask is not None and (~mask).sum() > 0:
                thresh = threshold_otsu(image_gray[~mask])
            else:
                thresh = threshold_otsu(image_gray)
            binary = image_gray < thresh if params['invert'] else image_gray > thresh
            
            if mask is not None:
                binary[mask] = False
            
            cleaned = morphology.remove_small_objects(binary, params['min_size'])
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=params['min_size'])
            labeled = measure.label(cleaned)
            morph = extract_morphology_descriptors(labeled, mask=mask)
            
            if len(morph) > 0:
                areas = [m["area"] for m in morph]
                diameters = [m["equivalent_diameter"] for m in morph]
                aspects = [m["aspect_ratio"] for m in morph]
                eccentricities = [m["eccentricity"] for m in morph]
                
                result.update({
                    "morph_area_mean": np.nanmedian(areas),
                    "morph_diameter_mean": np.nanmedian(diameters),
                    "morph_aspect_ratio_mean": np.nanmedian(aspects),
                    "morph_eccentricity_mean": np.nanmedian(eccentricities),
                    "morph_n_particles": len(morph),
                    "morph_method_used": "adaptive",
                })

            else:
                result.update({
                    "morph_area_mean": np.nan,
                    "morph_diameter_mean": np.nan,
                    "morph_aspect_ratio_mean": np.nan,
                    "morph_eccentricity_mean": np.nan,
                    "morph_n_particles": 0,
                    "morph_method_used": "adaptive"
                })
        
        else:
            if verbose:
                print("[INFO] Using ensemble segmentation methods")
            
            methods = {
                'standard': {'min_size': 50, 'invert': False},
                'fine': {'min_size': 30, 'invert': False},
                'inverse': {'min_size': 100, 'invert': True}
            }
            
            candidates = []
            def _coverage(b):
                return float(np.mean(b)) if b is not None else 0.0

            for method_name, params in methods.items():
                if mask is not None and (~mask).sum() > 0:
                    t = threshold_otsu(image_gray[~mask])
                else:
                    t = threshold_otsu(image_gray)

                binary = image_gray < t if params['invert'] else image_gray > t
                if mask is not None:
                    binary[mask] = False
                
                cleaned = morphology.remove_small_objects(binary, params['min_size'])
                cleaned = morphology.remove_small_holes(cleaned, area_threshold=params['min_size'])
                labeled = measure.label(cleaned)
                morph = extract_morphology_descriptors(labeled, mask=mask)
                
                if len(morph) > 0:
                    mstats = {
                        "morph_n_particles": len(morph),
                        "morph_area_mean": np.mean([m["area"] for m in morph]),
                        "morph_diameter_mean": np.mean([m["equivalent_diameter"] for m in morph]),
                        "morph_aspect_ratio_mean": np.mean([m["aspect_ratio"] for m in morph]),
                        "morph_eccentricity_mean": np.mean([m["eccentricity"] for m in morph]),
                        "_coverage": _coverage(cleaned)
                    }
                else:
                    mstats = {
                        "morph_n_particles": 0,
                        "morph_area_mean": np.nan,
                        "morph_diameter_mean": np.nan,
                        "morph_aspect_ratio_mean": np.nan,
                        "morph_eccentricity_mean": np.nan,
                        "_coverage": _coverage(cleaned)
                    }

                for k, v in mstats.items():
                    if not k.startswith("_"):
                        result[f"morph_{method_name}_{k.replace('morph_','')}"] = v

                cov = mstats["_coverage"]
                cov_penalty = 0.0
                if cov < 0.02: cov_penalty = (0.02 - cov) * 50
                elif cov > 0.60: cov_penalty = (cov - 0.60) * 50
                score = (mstats["morph_n_particles"] > 0) * 10 + mstats["morph_n_particles"] - cov_penalty
                candidates.append((score, method_name, mstats))

            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0] if candidates else None

            if (best is None) or (best[2]["morph_n_particles"] == 0):
                try:
                    from skimage.filters import threshold_multiotsu
                    if mask is not None and (~mask).sum() > 0:
                        t1, t2 = threshold_multiotsu(image_gray[~mask], classes=3)
                    else:
                        t1, t2 = threshold_multiotsu(image_gray, classes=3)
                    binary_hi = image_gray > t2
                    binary_lo = image_gray < t1
                    for binary in (binary_hi, binary_lo):
                        if mask is not None:
                            binary[mask] = False
                        cleaned = morphology.remove_small_objects(binary, 50)
                        cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
                        labeled = measure.label(cleaned)
                        morph = extract_morphology_descriptors(labeled, mask=mask)
                        if len(morph) > 0:
                            mstats = {
                                "morph_n_particles": len(morph),
                                "morph_area_mean": np.mean([m["area"] for m in morph]),
                                "morph_diameter_mean": np.mean([m["equivalent_diameter"] for m in morph]),
                                "morph_aspect_ratio_mean": np.mean([m["aspect_ratio"] for m in morph]),
                                "morph_eccentricity_mean": np.mean([m["eccentricity"] for m in morph]),
                            }
                            best = (9.9, "multiotsu", mstats)
                            break
                except Exception as _:
                    pass

            if best is not None:
                _, best_name, m = best
                result["morph_method_used"] = best_name
                result["morph_n_particles"] = m["morph_n_particles"]
                result["morph_area_mean"] = m["morph_area_mean"]
                result["morph_diameter_mean"] = m["morph_diameter_mean"]
                result["morph_aspect_ratio_mean"] = m["morph_aspect_ratio_mean"]
                result["morph_eccentricity_mean"] = m["morph_eccentricity_mean"]
            else:
                result["morph_method_used"] = "none"
                result["morph_n_particles"] = 0
    
    else:
        labeled = segment_image(image_gray, min_size=min_size, mask=mask)
        morph = extract_morphology_descriptors(labeled, mask=mask)
        
        if len(morph) > 0:
            result.update({
                "morph_area_mean": np.mean([m["area"] for m in morph]),
                "morph_diameter_mean": np.mean([m["equivalent_diameter"] for m in morph]),
                "morph_aspect_ratio_mean": np.mean([m["aspect_ratio"] for m in morph]),
                "morph_eccentricity_mean": np.mean([m["eccentricity"] for m in morph]),
            })
        else:
            result.update({
                "morph_area_mean": np.nan,
                "morph_diameter_mean": np.nan,
                "morph_aspect_ratio_mean": np.nan,
                "morph_eccentricity_mean": np.nan
            })

    result.update(
        extract_glcm_descriptors(image_gray, distance=glcm_distance, angle=glcm_angle, mask=mask)
    )

    lbp_hist = extract_lbp_histogram(image_gray, P=lbp_P, R=lbp_R, mask=mask)
    for i, val in enumerate(lbp_hist):
        result[f"lbp_p{lbp_P}r{lbp_R}_bin_{i}"] = val

    try:
        desc, features = descriptor_2p(
            image=image_gray,
            max_distance=max_distance,
            subsample_frac=subsample_frac,
            min_samples=min_samples,
            random_state=random_state,
            mask=mask
        )
        result.update(features)
    except Exception as e:
        if verbose:
            print(f"[WARNING] 2-point correlation failed: {e}")
        for k in [
            "s2_baseline", "s2_correlation_integral", "s2_correlation_length", "s2_slope_at_0",
            "s2_specific_surface", "s2_first_peak_pos", "s2_first_peak_val",
            "s2_mean_excess", "s2_variance_excess", "s2_skewness_excess", "s2_kurtosis_excess"
        ]:
            result[k] = np.nan

    if gram_extractor is not None:
        try:
            gram_df = gram_extractor.extract_dataframe(
                image, 
                image_id=filename, 
                mask=mask,
                top_k_eigen=5,
                include_cross_layer=True
            )
            result.update(gram_df.drop(columns=["id"]).iloc[0].to_dict())
        except Exception as e:
            if verbose:
                print(f"[WARNING] Gram descriptor failed: {e}")
            stat_names = [
                "mean", "std", "skew", "kurtosis", "min", "max",
                "q25", "q50_median", "q75", "iqr",
                "diag_mean", "diag_std", "diag_max", "diag_min",
                "offdiag_mean", "offdiag_std", "offdiag_max", "offdiag_min",
                "trace", "frobenius_norm", "spectral_norm", "determinant", "log_determinant",
                "entropy", "gini",
                "sparsity", "offdiag_sparsity", "l1_l2_ratio",
                "diag_to_offdiag_ratio", "range_to_std_ratio",
                "eigen_1", "eigen_2", "eigen_3", "eigen_4", "eigen_5",
                "eigen_sum", "eigen_max", "eigen_min", "eigen_entropy", 
                "effective_rank", "condition_number"
            ]
            
            for layer in gram_extractor.selected_layers.values():
                for stat in stat_names:
                    result[f"{layer}_{stat}"] = np.nan
            
            layer_names = list(gram_extractor.selected_layers.values())
            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names):
                    if i < j:
                        result[f"cross_{layer1}_{layer2}_frobenius_corr"] = np.nan
                        result[f"cross_{layer1}_{layer2}_cosine_sim"] = np.nan
            
            result["global_trace_mean"] = np.nan
            result["global_trace_std"] = np.nan
            result["global_entropy_mean"] = np.nan
            result["global_texture_strength"] = np.nan
            result["multi_scale_consistency"] = np.nan

    return result


def process_image_folder(folder_path: str, output_csv: str = None, 
                         mask_folder: str = None, mask_suffix: str = "_mask",
                         use_gram: bool = True, 
                         gram_device: str = None,
                         **kwargs) -> pd.DataFrame:
    """
    Process all images in a folder using extract_all_descriptors().
    
    Parameters:
        folder_path: Path to folder containing images
        output_csv: Optional path to save results
        mask_folder: Subfolder name containing mask files
        mask_suffix: Suffix for mask filenames
        use_gram: Whether to extract Gram matrix features (default: True)
        gram_device: Device for Gram extraction ('cuda', 'cpu', or None for auto)
        **kwargs: Additional arguments passed to extract_all_descriptors
    
    Returns:
        DataFrame with all extracted features
    """
    # Create Gram extractor if requested
    gram_extractor = None
    if use_gram:
        try:
            device = gram_device or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[INFO] Initializing Gram matrix extractor on device: {device}")
            gram_extractor = GramDescriptorExtractor(device=device)
            print(f"[INFO] Gram matrix extractor ready - will extract ~240 features per image")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Gram extractor: {e}")
            print(f"[WARNING] Continuing without Gram features")
    else:
        print("[INFO] Gram matrix extraction disabled")
    
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    
    print(f"[INFO] Found {len(files)} images to process")

    features = []
    for f in tqdm(files, desc="Processing all descriptors"):
        image_path = os.path.join(folder_path, f)
        
        mask = None
        if mask_folder is not None:
            full_mask_folder = os.path.join(folder_path, mask_folder)
            
            base_name, ext = os.path.splitext(f)
            mask_name = f"{base_name}{mask_suffix}.npy"
            mask_path = os.path.join(full_mask_folder, mask_name)
            
            if os.path.exists(mask_path):
                try:
                    mask = np.load(mask_path)
                    if mask.dtype != bool:
                        mask = mask.astype(bool)
                    print(f"[INFO] Loaded mask for {f}")
                except Exception as e:
                    print(f"[WARNING] Failed to load mask {mask_name}: {e}")
            else:
                print(f"[WARNING] Mask file not found: {mask_path}")
        
        try:
            image = io.imread(image_path)
            
            if image.ndim == 3 and image.shape[2] == 4:
                print(f"[INFO] Converting RGBA to RGB for {f}")
                image = image[:, :, :3]
            
            # CRITICAL FIX: Pass gram_extractor to the extraction function
            feature_dict = extract_all_descriptors(
                image, 
                filename=f, 
                mask=mask,
                gram_extractor=gram_extractor,  # <-- THIS IS THE FIX
                **kwargs
            )
            features.append(feature_dict)
        except Exception as e:
            print(f"[ERROR] Failed to process {f}: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(features)
    
    # Report feature counts
    print(f"\n[INFO] Extraction complete!")
    print(f"[INFO] Total images processed: {len(df)}")
    print(f"[INFO] Total features extracted: {len(df.columns)}")
    
    # Count feature types
    chemistry_cols = [c for c in df.columns if c in ['Al', 'B', 'C', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'La', 'Mg', 'Mn', 'Mo', 'N', 'Nb', 'Ni', 'O', 'P', 'S', 'Si', 'Sm', 'Sn', 'Ti', 'Ta', 'V', 'W', 'Zn', 'Zr', 'Fe']]
    morph_cols = [c for c in df.columns if c.startswith('morph_')]
    glcm_cols = [c for c in df.columns if c.startswith('glcm_')]
    lbp_cols = [c for c in df.columns if c.startswith('lbp_')]
    s2_cols = [c for c in df.columns if c.startswith('s2_')]
    gram_cols = [c for c in df.columns if 'conv' in c or 'cross_' in c or c.startswith('global_')]
    
    print(f"[INFO] Feature breakdown:")
    print(f"  - Chemistry: {len(chemistry_cols)}")
    print(f"  - Morphology: {len(morph_cols)}")
    print(f"  - GLCM: {len(glcm_cols)}")
    print(f"  - LBP: {len(lbp_cols)}")
    print(f"  - S2 Correlation: {len(s2_cols)}")
    print(f"  - Gram Matrix: {len(gram_cols)}")
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Results saved to {output_csv}")
    
    return df

def process_subfolders_with_doi(
    root_folder: str, 
    output_csv: str = None, 
    doi_filename: str = "doi_info.txt", 
    subfolder_filter: list = None,
    use_gram: bool = True,  # <-- NEW PARAMETER
    gram_device: str = None,  # <-- NEW PARAMETER
    **kwargs
) -> pd.DataFrame:
    """
    Process all subfolders in a root folder, extracting image descriptors and DOI information.
    Each subfolder should contain images and a doi_info.txt file.
    
    Parameters:
        root_folder: Path to root folder containing subfolders
        output_csv: Optional path to save combined results
        doi_filename: Name of DOI info file in each subfolder
        subfolder_filter: Optional list of subfolder names to process
        use_gram: Whether to extract Gram matrix features (default: True)
        gram_device: Device for Gram extraction ('cuda', 'cpu', or None for auto)
        **kwargs: Additional arguments passed to process_image_folder
    
    Returns:
        Combined DataFrame with all features from all subfolders
    """
    root_path = Path(root_folder)
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_folder}")
    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {root_folder}")
    
    all_subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if subfolder_filter is not None:
        subfolder_filter_set = set(subfolder_filter)
        subfolders = [f for f in all_subfolders if f.name in subfolder_filter_set]
        
        found_names = {f.name for f in subfolders}
        missing = subfolder_filter_set - found_names
        if missing:
            print(f"[WARNING] Requested subfolders not found: {missing}")
    else:
        subfolders = all_subfolders
    
    if not subfolders:
        raise ValueError(f"No subfolders found in {root_folder}" + 
                        (" matching the filter" if subfolder_filter else ""))
    
    print(f"[INFO] Found {len(subfolders)} subfolders to process" +
          (f" (filtered from {len(all_subfolders)})" if subfolder_filter else ""))
    
    all_features = []
    processed_count = 0
    failed_folders = []
    
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        try:
            doi_info = extract_doi_info(subfolder, doi_filename)
            
            # CRITICAL FIX: Pass use_gram and gram_device parameters
            df_subfolder = process_image_folder(
                str(subfolder),
                use_gram=use_gram,  # <-- THIS IS THE FIX
                gram_device=gram_device,  # <-- THIS IS THE FIX
                **kwargs
            )
            
            if df_subfolder.empty:
                print(f"[WARNING] No features extracted from {subfolder.name}")
                continue
            
            for key, value in doi_info.items():
                df_subfolder[key] = value
            
            df_subfolder['subfolder'] = subfolder.name
            df_subfolder["id_name"] = df_subfolder['subfolder'] + "_" + df_subfolder['image_filename'].apply(_normalize_filename)
                        
            all_features.append(df_subfolder)
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Failed to process subfolder {subfolder.name}: {e}"
            failed_folders.append((subfolder.name, str(e)))
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
    
    if not all_features:
        raise ValueError("No subfolders were successfully processed")
    
    combined_df = pd.concat(all_features, ignore_index=True)
    
    print(f"\n[INFO] Successfully processed {processed_count} subfolders")
    print(f"[INFO] Total images processed: {len(combined_df)}")
    print(f"[INFO] Total features per image: {len(combined_df.columns)}")
    
    if failed_folders:
        print(f"[WARNING] Failed to process {len(failed_folders)} subfolders:")
        for folder_name, error in failed_folders:
            print(f"  - {folder_name}: {error}")
    
    if output_csv:
        combined_df.to_csv(output_csv, index=False)
        print(f"[INFO] Combined results saved to {output_csv}")
    
    return combined_df


def extract_doi_info(subfolder_path: Path, doi_filename: str = "doi_info.txt") -> dict:
    """
    Extract DOI information from a text file in the subfolder.
    """
    
    doi_file_path = subfolder_path / doi_filename
    
    if not doi_file_path.exists():
        raise FileNotFoundError(f"DOI file not found: {doi_file_path}")
    
    doi_info = {}
    
    try:
        with open(doi_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('PDF:'):
                    doi_info['pdf_filename'] = line.replace('PDF:', '').strip()
                elif line.startswith('DOI:') and not line.startswith('DOI URL:'):
                    doi_info['doi'] = line.replace('DOI:', '').strip()
                elif line.startswith('DOI URL:'):
                    doi_info['doi_url'] = line.replace('DOI URL:', '').strip()
                elif line.startswith('Extraction Date:'):
                    doi_info['extraction_date'] = line.replace('Extraction Date:', '').strip()
                else:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        doi_info[key] = value.strip()
        
        if 'doi' not in doi_info and 'doi_url' not in doi_info:
            raise ValueError("No DOI information found in file")
            
        return doi_info
        
    except Exception as e:
        raise ValueError(f"Failed to parse DOI file {doi_file_path}: {e}")


def get_subfolder_summary(root_folder: str, doi_filename: str = "doi_info.txt", mask_suffix: str = "_mask") -> pd.DataFrame:
    """
    Get a summary of all subfolders and their DOI information without processing images.
    Useful for checking folder structure before full processing.
    """
    
    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    summary_data = []
    
    for subfolder in subfolders:
        summary_row = {'subfolder': subfolder.name}
        
        image_files = [f for f in subfolder.iterdir() 
                      if f.is_file() and f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']]
        summary_row['image_count'] = len(image_files)
        
        mask_folder = subfolder / "masks"
        summary_row['mask_folder_exists'] = mask_folder.exists()
        
        if mask_folder.exists():
            mask_files = [f for f in mask_folder.iterdir() if f.suffix == '.npy']
            summary_row['mask_count'] = len(mask_files)
            
            matched_masks = 0
            for img_file in image_files:
                expected_mask = mask_folder / f"{img_file.stem}{mask_suffix}.npy"
                if expected_mask.exists():
                    matched_masks += 1
            
            summary_row['matched_masks'] = matched_masks
            summary_row['mask_status'] = f"{matched_masks}/{len(image_files)} matched"
        else:
            summary_row['mask_count'] = 0
            summary_row['matched_masks'] = 0
            summary_row['mask_status'] = 'No mask folder'
        
        try:
            doi_info = extract_doi_info(subfolder, doi_filename)
            summary_row.update(doi_info)
            summary_row['doi_status'] = 'OK'
        except Exception as e:
            summary_row['doi_status'] = f'ERROR: {str(e)}'
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

def debug_mask_paths(folder_path: str, mask_folder: str = "masks", mask_suffix: str = "_mask"):
    """
    Debug function to check mask path resolution
    """
    print(f"[DEBUG] Checking folder: {folder_path}")
    
    full_mask_folder = os.path.join(folder_path, mask_folder)
    print(f"[DEBUG] Looking for masks in: {full_mask_folder}")
    print(f"[DEBUG] Mask folder exists: {os.path.exists(full_mask_folder)}")
    
    if os.path.exists(full_mask_folder):
        mask_files = os.listdir(full_mask_folder)
        print(f"[DEBUG] Found {len(mask_files)} files in mask folder:")
        for mf in mask_files[:5]:
            print(f"  - {mf}")
    
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    print(f"[DEBUG] Found {len(image_files)} image files:")
    for img in image_files[:3]:
        print(f"  - {img}")
        
        base_name, ext = os.path.splitext(img)
        expected_mask = f"{base_name}{mask_suffix}.npy"
        expected_path = os.path.join(full_mask_folder, expected_mask)
        exists = os.path.exists(expected_path)
        print(f"    Expected mask: {expected_mask} -> Exists: {exists}")

if __name__ == "__main__":
    import os,sys
    module_path = str.rsplit(os.path.abspath(''),os.path.sep,1)[0]
    sys.path.append(module_path)
    print(module_path)
    
    from data_processing.combine_data import load_literature_data
    from data_processing.clean_data import filter_by_subcategory
    df_literature = load_literature_data(os.path.join(module_path, "data", "literature_data.xlsx"), sheet_name='papers')
    df_filter = filter_by_subcategory(df_literature, ["Ferritic Stainless Steel", "Duplex Stainless Steel", 'Austenitic Stainless Steel',"Conventional DP"])
    subfolder_list = list(df_filter['subfolder'].unique())
    
    path_images_folder = os.path.join(module_path, "data", "processed_images")
    
    # # Get summary first
    summary = get_subfolder_summary(path_images_folder)
    print(summary[['subfolder', 'mask_status','doi_status']])
    
    df = process_subfolders_with_doi(
        root_folder=path_images_folder,
        output_csv=os.path.join(module_path, "results", "descriptors_new.csv"),
        mask_folder='masks',  
        mask_suffix='_mask',
        subfolder_filter=subfolder_list,
        use_gram=True,  
        gram_device=None  # <-- Auto-detect (will use CUDA if available)
    )
    
    print(f"Processing completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    




