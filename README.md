# Composition–Structure–Property Mapping from OM and SEM Micrographs in Steels

This repository contains the code and data to reproduce the experiments from:
**“Explainable Composition–Structure–Property Modelling of Steels Using Micrographs and Transfer Learning”** It provides a
microstructure- and chemistry-aware pipeline for predicting tensile properties
from optical microscopy (OM) and scanning electron microscopy (SEM) images,
together with compositional data, with emphasis on explainability and
transferability across metallic systems.

## Repository structure

- `data/`  
  Contains all input data required for the experiments: the full and reduced descriptor tables (`descriptors_full.csv`, `descriptors_reduced.csv`, `descriptors_reduced_report.json`), target-specific X/y source tables in `data/Xy/`, curated literature datasets, the validation dataset and micrographs.

- `data_processing/`  
  Scripts for dataset curation and preprocessing (see "Preprocessing workflow"):
  - `feat_reducer.py`: deterministic, target-independent descriptor reduction (`descriptors_full.csv` to `descriptors_reduced.csv`).
  - `clean_data.py`: cohort selection, export of X/y source tables, and the fold-safe cleaning pipeline (`fit_cleaning_pipeline`, `apply_cleaning_pipeline`, `prepare_clean_fold`).

- `image_processing/`  
  Scripts to extract microstructure and texture descriptors from SEM and OM micrographs. The resulting descriptor tables are saved under `data/` and are used as inputs for the preprocessing and machine‑learning scripts.

- `machine_learning/`  
  Core modelling and analysis scripts:
  - `models_paper.py`: custom scaler, metric computation, nested group‑aware cross‑validation with fold‑local cleaning, Optuna‑based hyperparameter tuning for XGBoost, and `train_final_xgb_full`.
  - `shap_script.py`: SHAP analysis pipeline that fits a final XGBoost model on the full dataset and computes SHAP values and plots for interpretability.
  - `transfer_learning_curve.py`: instance‑reweighted transfer‑learning experiments from stainless steel (source) to dual‑phase steel (target), including learning curves as a function of available target data.

- `results/`  
  Stores generated outputs such as trained models, cross‑validation metrics, out‑of‑fold predictions, SHAP values, and transfer‑learning curves in `.csv` and `.pkl` formats.

- `environment.yaml`  
  Conda environment specification listing all dependencies required to run the full pipeline (Python 3.12, XGBoost, SHAP, etc.).

## Installation

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yaml
conda activate csp
```

### 2. Main dependencies

The environment uses Python 3.12 and the following key packages (see `environment.yaml` for the full list):

- numpy, pandas, scipy
- scikit-learn, statsmodels
- scikit-image, py-opencv, pillow
- matplotlib, tqdm, numba
- pytorch, torchvision
- optuna
- xgboost==3.0.2 (via pip)
- shap==0.47.2 (via pip)

## Data

The `data/` directory contains:

- `descriptors_full.csv`: all extracted descriptors, including five candidate segmentation methods for morphology and the complete Gram‑matrix descriptor family.
- `descriptors_reduced.csv`: output of `feat_reducer.py` (canonical morphology block and fixed 30‑column Gram panel).
- **Target‑specific source tables** in `data/Xy/`: `X_<cohort>_<target>.csv` and `y_<cohort>_<target>.csv`, where `<cohort>` is `ss_sem`, `ss_om` or `dp_sem`, and `<target>` is `uts` (UTS, MPa), `ys` (YS, MPa) or `el` (Total EL, %).
- Curated **literature metadata** (`literature_data.xlsx`) including alloy composition, processing information, and references to the original SEM figures via DOI and figure number.

Cohort definitions (fixed in `clean_data.py`):

- `ss_sem`: Ferritic, Duplex and Austenitic Stainless Steel; methods `SEM: SE`, `SEM: BSE`.
- `ss_om`: Ferritic, Duplex and Austenitic Stainless Steel; method `OM`.
- `dp_sem`: Conventional DP; methods `SEM: SE`, `SEM: BSE`.

The exported X files are source tables, not final machine‑learning matrices. They retain raw predictor values and `sample_id` (required for grouped cross‑validation). All data‑derived cleaning is performed inside each cross‑validation fold (see below).

Raw SEM micrographs **are not redistributed** in this repository due to copyright restrictions.
However, each micrograph used in this work is documented in the literature metadata by DOI and figure identifier, so that interested users can obtain the original images from the respective publishers.

## Preprocessing workflow

```text
descriptors_full.csv
  -> feat_reducer.py -> descriptors_reduced.csv (+ descriptors_reduced_report.json)
  -> clean_data.py   -> data/Xy/X_<cohort>_<target>.csv, y_<cohort>_<target>.csv
  -> models_paper.py -> split by sample_id, then prepare_clean_fold() per fold
```

### 1. Deterministic descriptor reduction (`feat_reducer.py`)

The reduction uses no targets and fits no data‑derived parameters.

- **Morphology collapse.** The five candidate segmentations (`standard`, `fine`, `inverse`, `adaptive`, `multiotsu`) are replaced by one canonical block (`morph_n_particles`, `morph_area_mean`, `morph_diameter_mean`, `morph_aspect_ratio_mean`, `morph_eccentricity_mean`). For each row, the method with the highest particle count is selected; ties go to the earlier method in the order above. The selected method is recorded in `morph_source`. Rows without a segmentation of at least one particle receive `morph_n_particles = 0`, NaN shape metrics and `morph_source = "none"`. No rows are dropped.
- **Gram panel reduction.** The complete Gram‑derived family is replaced by a fixed panel of 30 descriptors: five sources (`conv1`, `layer1`, `layer2`, `layer3`, `layer4`) × six statistics (`trace`, `frobenius_norm`, `spectral_norm`, `condition_number`, `eigen_entropy`, `effective_rank`). An error is raised if any of these columns is missing.
- All other columns are passed through unchanged (verified internally). A JSON audit report records column counts, retained and discarded Gram columns, and morphology source counts.

### 2. Cohort export (`clean_data.py`)

Running `clean_data.py` prints missingness and infinity diagnostics (for inspection only, not for feature selection) and writes the `X_*` and `y_*` source tables for each cohort and target. Rows with a missing target value are excluded from the corresponding table.

### 3. Fold‑safe cleaning (`prepare_clean_fold`)

Cleaning is fitted on each training partition only (`fit_cleaning_pipeline`) and applied unchanged to the paired validation partition (`apply_cleaning_pipeline`).

Fixed operations:

- Rows with a missing target are dropped.
- Every row must report at least one non‑Fe element; otherwise an error is raised.
- Missing chemistry values are set to 0.0 (database convention).
- Identifiers, metadata (`Material`, `Category`, `method`, `morph_source`, etc.), all target columns, `Fe`, `Other` and non‑numeric columns are removed.

Training‑fold‑fitted operations, in order:

1. Constant features (zero or one distinct finite value) are removed.
2. Image descriptors with more than 80 % missing (or non‑finite) values are removed; rows with less than 80 % of the retained image descriptors observed are removed.
3. Canonical morphology features with more than 20 % missing values are removed; rows are retained to avoid selection bias from failed segmentations.
4. Rare‑support chemistry features (non‑zero in fewer than 5 unique `sample_id` groups) are removed.
5. Any remaining feature with more than 80 % missing values is removed.
6. Infinite values are handled with `handle_inf` (`clip` by default, with limits learned from the training fold; alternatives: `nan`, `drop_features`).
7. Remaining missing values are imputed with training‑fold medians.

Columns absent from the validation or query data are filled with training medians and a warning is printed. The settings used in `models_paper.py` are `missing_feature_threshold=0.80`, `handle_inf="clip"` and `drop_fe=True`.

```python
from data_processing.clean_data import prepare_clean_fold

X_tr, y_tr, X_va, y_va, cleaner = prepare_clean_fold(
    dataframe_train=df_train,
    dataframe_validation=df_valid,
    target_col="UTS (MPa)",
    missing_feature_threshold=0.80,
    handle_inf="clip",
    drop_fe=True,
)
```

## Usage

> The commands below assume that you run each script from within its own subdirectory (the scripts add the parent of the working directory to `sys.path`), or that the repository root is on your `PYTHONPATH`, so that `data_processing.clean_data` can be imported.

### 1. Preprocessing

```bash
cd data_processing
python feat_reducer.py
python clean_data.py
```

### 2. Main stainless‑steel models

The main models for each property and cohort are implemented in `models_paper.py` via `run_paper_models` and XGBoost + Optuna objectives.

```bash
cd machine_learning
python models_paper.py
```

Inside `models_paper.py`, the script:

- Loads `X_<cohort>_<target>.csv` and `y_<cohort>_<target>.csv` from `data/Xy/`. The `__main__` block currently iterates over `ss_sem` and `ss_om` and the targets `uts`, `ys`, `el`; add `dp_sem` to `GROUPS` to run dual‑phase steels.
- Runs nested cross‑validation (5 outer, 3 inner folds) with `GroupKFold` on `sample_id`.
- Fits the cleaning pipeline on each outer training partition only. Inner‑fold cleaning is precomputed once per outer fold and reused across Optuna trials, which is valid because cleaning does not depend on the hyperparameters.
- Tunes XGBoost hyperparameters with Optuna (default 100 trials, random state 2026).
- Retains out‑of‑fold predictions only for validation rows that pass the fitted row‑completeness rule.
- Saves `<cohort>_<target>_summary.csv`, `<cohort>_<target>_oof.csv` and `<cohort>_<target>_nested_cv.pkl` to `results/ML/<timestamp>/`.

### 3. Transfer learning from stainless to dual‑phase steels

The script `transfer_learning_curve.py` performs instance‑reweighted transfer‑learning experiments from stainless steels (SS) to dual‑phase steels (DP).

```bash
cd machine_learning
python transfer_learning_curve.py
```

This script:

- Loads aligned SS and DP feature matrices and targets from `data/Xy/`.
- Tunes XGBoost hyperparameters on the source (SS) data using inner K‑fold cross‑validation.
- Runs repeated subsampling experiments over a range of DP sample sizes `N_VALUES` (default: 20–200) and evaluates:
  - A DP‑only baseline model.
  - An instance‑reweighted SS+DP pooled model.
- Stores per‑run metrics and summary statistics (means, standard errors, 95 % confidence intervals) in `results/transfer_learning/<timestamp>/` as `.csv` files.

You can adjust `N_VALUES`, `N_REPS`, and other settings at the bottom of `transfer_learning_curve.py` before running.

### 4. SHAP‑based explainability

The SHAP analysis is implemented in `shap_script.py` and leverages the final XGBoost models trained via `train_final_xgb_full` in `models_paper.py`.

```bash
cd machine_learning
python shap_script.py
```

Within `shap_script.py`, the function `fit_full_xgb_and_compute_shap`:

1. Calls `train_final_xgb_full` to tune XGBoost hyperparameters with group‑aware inner cross‑validation and fit a final model and cleaning pipeline on the full dataset.
2. Computes SHAP values using `shap.TreeExplainer` with an interventional background.
3. Saves:
   - SHAP arrays and explainer objects (`shap_*.pkl`).
   - Summary tables with mean absolute SHAP values and cumulative importance (`shapsummary_*.csv`).
   - Optionally, beeswarm and bar plots for the top‑ranked features.

### 5. Descriptor extraction from OM and SEM micrographs

The script `image_processing/extract_descriptors.py` implements the
descriptor-extraction pipeline used in the paper. It reads optical microscopy
(OM) or scanning electron microscopy (SEM) images from a user-specified
directory, applies preprocessing (for example grayscale conversion, resizing,
and masking), and computes microstructure and texture descriptors. These
include morphology descriptors, gray-level co-occurrence matrix (GLCM)
statistics, local binary patterns, two-point spatial correlations, and optional
CNN Gram-matrix features. The resulting table corresponds to
`data/descriptors_full.csv`; run `feat_reducer.py` afterwards.

### 6. Predict properties from new micrographs

`predict_from_micrograph.py` provides an end-to-end workflow to predict tensile
properties from prepared optical-microscopy (OM) or SEM micrographs. For each
image, the script extracts the same descriptor families used in the modelling
pipeline, combines them with composition and imaging metadata, applies the same
reduction and cleaning, trains an XGBoost model on a filtered literature
dataset, and predicts:

- Ultimate tensile strength, `UTS (MPa)`
- Yield strength, `YS (MPa)`
- Total elongation, `Total EL (%)`

The default batch workflow processes all `.png` files in an image folder. A
single-image function is also available for programmatic use.

#### Required input files

For each image, provide:

1. A prepared micrograph in `.png` format. The current batch implementation
   searches the selected folder for `*.png` images.
2. A matching exclusion mask stored as:

   ```text
   <image_folder>/masks/<image_stem>_mask.npy
   ```

   The mask must be generated from the same image using the repository masking
   workflow (`doi_and_mask.py` / equivalent masking script). In the mask,
   `True` pixels are excluded from descriptor extraction.
3. A validation metadata table, by default:

   ```text
   data/validation/csp_validation.xlsx
   ```

   This table provides the metadata associated with each image, including image
   name, imaging modality, physical scale (`um_per_pixel`), material category,
   and chemical composition.
4. A literature descriptor dataset used to fit the model, by default:

   ```text
   data/descriptors_full.csv
   ```

#### Expected folder layout

```text
data/
└── validation/
    ├── csp_validation.xlsx
    └── processed_images/
        ├── OM/
        │   ├── sample_01.png
        │   └── masks/
        │       └── sample_01_mask.npy
        └── SEM/
            ├── sample_02.png
            └── masks/
                └── sample_02_mask.npy
```

The image stem must match the corresponding `image_name` in
`csp_validation.xlsx`. For example, `sample_02.png` requires a metadata entry
with `image_name = sample_02` and a mask named `sample_02_mask.npy`.

#### Batch prediction

From the repository root, run:

```bash
python predict_from_micrograph.py
```

The default `__main__` block processes the `OM` and `SEM` folders under:

```text
data/validation/processed_images/
```

For each modality, the script:

1. Filters the training dataset by imaging modality and selected material
   categories.
2. Extracts morphology, GLCM texture, local-binary-pattern, two-point
   correlation, and optionally CNN Gram-matrix descriptors.
3. Converts pixel-based descriptors into physical units using
   `um_per_pixel`.
4. Applies the deterministic reduction (canonical morphology block and fixed
   30-column Gram panel) identically to training and query data, then fits the
   cleaning pipeline (`fit_cleaning_pipeline`) on the filtered training dataset
   only.
5. Applies the fitted pipeline to every query image (`apply_cleaning_pipeline`),
   aligning it to the training-feature schema. Absent columns are filled with
   training medians and a warning is printed.
6. Fits one XGBoost model for each target property and generates predictions.

By default, SEM training data combine `SEM: SE` and `SEM: BSE` records; OM data
are processed separately.

#### Configuration

Edit the configuration block in `predict_from_micrograph.py` before execution:

```python
modalities = ["OM", "SEM"]

config = QueryConfig(
    image_folder=image_folder,
    image_modality=modality,
    material_categories=[
        "Ferritic Stainless Steel",
        "Duplex Stainless Steel",
        "Austenitic Stainless Steel",
    ],
    training_dataset_csv=os.path.join(
        MODULE_PATH, "data", "descriptors_full.csv"
    ),
    use_gram=True,
    gram_device=None,
    n_trials=100,
    n_splits_inner=5,
)
```

For a shorter test run, reduce the hyperparameter-optimisation settings:

```python
n_trials = 20
n_splits_inner = 3
```

This reduces runtime but should not be used for the final reported model results.

#### Outputs

Each run creates timestamped output directories under
`data/validation/processed_images/<modality>/predictions/` and
`data/validation/predictions/`. The outputs include:

- `query_descriptors_batch.csv`: extracted descriptors and metadata for the
  processed query images.
- `X_train_<target>.csv` and `y_train_<target>.csv`: training matrices used
  for each target.
- `X_query_batch_<target>.csv`: final aligned query feature matrix.
- `xgb_<modality>_<target>_model.pkl`: fitted XGBoost prediction model.
- `xgb_<modality>_<target>_study.pkl`: Optuna hyperparameter-optimisation
  study.
- `xgb_<modality>_<target>_predictions.csv`: predicted values for the query
  images.
- `all_xgb_validation_results.pkl`: bundle containing all modality- and
  property-specific models and predictions.

#### Single-image use from Python

For a single image that is already represented in `csp_validation.xlsx`, use
`run_single_image`:

```python
import os
import pandas as pd

from predict_from_micrograph import QueryConfig, run_single_image

df_validation = pd.read_excel(
    "data/validation/csp_validation.xlsx"
)

config = QueryConfig(
    image_folder="data/validation/processed_images/SEM",
    image_modality="SEM",
    material_categories=[
        "Ferritic Stainless Steel",
        "Duplex Stainless Steel",
        "Austenitic Stainless Steel",
    ],
    training_dataset_csv="data/descriptors_full.csv",
    use_gram=True,
    n_trials=100,
    n_splits_inner=5,
)

results = run_single_image(
    name="sample_02",
    image_folder=config.image_folder,
    df_val=df_validation,
    config=config,
    targets=["UTS (MPa)", "YS (MPa)", "Total EL (%)"],
    save_bundle=True,
)

print({target: item["y_pred"] for target, item in results.items()})
```

`name` is the image stem only: use `sample_02`, not `sample_02.png`.

#### Imaging modalities

The prediction workflow supports optical microscopy (`OM`) and scanning electron
microscopy (`SEM`). OM images are processed and modelled separately. Selecting
`SEM` pools the `SEM: SE` and `SEM: BSE` records in the training dataset. The
query-image modality must be consistent with the selected training subset.

#### Scope and limitations

The script is currently configured for validation data whose image-level
metadata, chemical composition, and physical scale are available in
`csp_validation.xlsx`. In the current workflow, the metadata table—not manual
arguments—is the authoritative source for image modality, `um_per_pixel`,
material category, and composition.

Predictions should be interpreted only within the applicability domain of the
selected training subset. In particular, differences in steel class,
microstructure, specimen preparation, image contrast, magnification, pixel
scale, SEM detector mode, and unrepresented compositional ranges can produce
out-of-distribution inputs. The predictions support screening and hypothesis
generation and do not replace experimental mechanical testing.

## Reproducibility and configuration

- Experiments are controlled by parameters defined inside the scripts (such as number of Optuna trials, cross‑validation folds, and cleaning thresholds) and, where present, configuration files referenced from those scripts.
- Data are split by `sample_id` first (`GroupKFold`); all data‑derived cleaning (feature removal, row filtering, clipping limits, imputation medians) and feature scaling are then fitted on the training partition only. The deterministic reduction in `feat_reducer.py` is target‑independent and fits no statistics, so it may precede splitting.
- Cleaning thresholds (0.80 feature missingness, 0.80 row completeness, 0.20 morphology missingness, 5 groups for rare‑support chemistry) are parameters of `clean_data.py` and should be reported with any results.
- Random seeds (for example `random_state=2026`, `random_state=42`) are explicitly specified to support reproducibility.
- `descriptors_reduced_report.json` documents the reduction applied to `descriptors_full.csv`.
- Script versions: `clean_data.py` and `feat_reducer.py` v1.3.0 (12.10.2026); `models_paper.py` v1.1.0 (08.10.2026).

## Citation

If you use this code or data in your own work, please cite:

Millner, G., Wang, T.-W., Shen, S., Li, Y., & Chen, E. Y.-S. (2026). Explainable composition–structure–property modelling of steels using micrographs and transfer learning. Materials Today Communications, 116478. https://doi.org/10.1016/j.mtcomm.2026.116478
