# Composition–Structure–Property Mapping from OM and SEM Micrographs in Steels

This repository contains the code and data to reproduce the experiments from:
**“A Transferable, Explainable AI Framework for Composition–Structure–Property Mapping from Steel Micrographs.”** It provides a
microstructure- and chemistry-aware pipeline for predicting tensile properties
from optical microscopy (OM) and scanning electron microscopy (SEM) images,
together with compositional data, with emphasis on explainability and
transferability across metallic systems.

## Repository structure

- `data/`  
  Contains all input data required for the experiments, including feature matrices and labels (X/y files) as well as curated literature datasets for stainless steels and dual‑phase steels (for example `data/Xy/X_ss_ys.csv`, `data/Xy/ys_ss.csv`, `data/Xy/X_dp_ys.csv`, `data/Xy/ys_dp.csv`). Additionaly, the validation dataset and micrographs can be found here.

- `data_processing/`  
  Scripts for dataset curation and preprocessing, including cleaning, combining image-based descriptors from SEM micrographs with literature metadata, and treating multicollinearity in the feature space.

- `image_processing/`  
  Scripts to extract microstructure and texture descriptors from SEM micrographs. The resulting descriptor tables are saved under `data/` and are used as inputs for the machine‑learning scripts.

- `machine_learning/`  
  Core modelling and analysis scripts:
  - `models_paper.py`: reusable utilities (custom scaler, metric computation), Optuna‑based hyperparameter tuning for XGBoost, and functions to run the main models for stainless and dual‑phase steels.
  - `shap_script.py`: SHAP analysis pipeline that fits a final XGBoost model on the full dataset and computes SHAP values and plots for interpretability.
  - `transfer_learning_curve.py`: instance‑reweighted transfer‑learning experiments from stainless steel (source) to dual‑phase steel (target), including learning curves as a function of available target data.

- `results/`  
  Stores generated outputs such as trained models, cross‑validation metrics, SHAP values, and transfer‑learning curves in `.csv` and `.pkl` formats.

- `environment.yaml`  
  Conda environment specification listing all dependencies required to run the full pipeline (Python 3.12, XGBoost, SHAP, etc.).

## Installation

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yaml
conda activate csp
```

### 2. Main dependencies

The environment uses Python 3.12 and the following key packages (see `environment.yml` for the full list):

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

- Precomputed **descriptor matrices** derived from SEM micrographs (for example `data/Xy/X_ss_ys.csv`, `data/Xy/X_dp_ys.csv`).  
- **Target property vectors**: yield strength (`ys_*`), ultimate tensile strength (`uts_*`), and elongation at fracture (`el_*`).  
- Curated **literature metadata** (`literature_data.xlsx`) including alloy composition, processing information, and references to the original SEM figures via DOI and figure number.

Raw SEM micrographs **are not redistributed** in this repository due to copyright restrictions.  
However, each micrograph used in this work is documented in the literature metadata by DOI and figure identifier, so that interested users can obtain the original images from the respective publishers.

## Usage

> The commands below assume that the repository root is on your `PYTHONPATH`, so that `models_paper.py` and related modules can be imported.

### 1. Main stainless and dual‑phase models

The main models for each property and alloy system are implemented in `models_paper.py` via helper functions such as `run_paper_models` and XGBoost + Optuna objectives.  
A typical pattern is:

```bash
cd path/to/repo

python -m machine_learning.models_paper
```

Inside `models_paper.py`, the script:

- Loads stainless and dual‑phase X/y files from `data/Xy/` (for example `X_ss_uts.csv`, `uts_ss.csv`, `X_dp_uts.csv`, `uts_dp.csv`).  
- Runs repeated cross‑validation with group‑aware splits where appropriate (for example `GroupKFold` on `sample_id`).  
- Tunes XGBoost hyperparameters with Optuna and saves summary statistics and raw results to `results/` (for example `ss_sem_uts_summary.csv`, `dp_sem_uts.pkl`).

### 2. Transfer learning from stainless to dual‑phase steels

The script `transfer_learning_curve.py` performs instance‑reweighted transfer‑learning experiments from stainless steels (SS) to dual‑phase steels (DP).

```bash
cd path/to/repo

python machine_learning/transfer_learning_curve.py
```

This script:

- Loads aligned SS and DP feature matrices and targets from `data/Xy/` (`X_ss_{prop}.csv`, `X_dp_{prop}.csv`, `{prop}_ss.csv`, `{prop}_dp.csv`).  
- Tunes XGBoost hyperparameters on the source (SS) data using inner K‑fold cross‑validation.  
- Runs repeated subsampling experiments over a range of DP sample sizes `N_VALUES` (default: 20–200) and evaluates:
  - A DP‑only baseline model.  
  - An instance‑reweighted SS+DP pooled model.  
- Stores per‑run metrics and summary statistics (means, standard errors, 95 % confidence intervals) in `results/transfer_learning/<timestamp>/` as `.csv` files.

You can adjust `N_VALUES`, `N_REPS`, and other settings at the bottom of `transfer_learning_curve.py` before running.

### 3. SHAP‑based explainability

The SHAP analysis is implemented in `shap_script.py` and leverages the final XGBoost models trained via `train_final_xgb_full` in `models_paper.py`.

A typical workflow for a given dataset (for example stainless steel yield strength) is:

```bash
cd path/to/repo

python machine_learning/shap_script.py
```

Within `shap_script.py`, the function `fit_full_xgb_and_compute_shap`:

1. Calls `train_final_xgb_full` to tune XGBoost hyperparameters with inner cross‑validation and fit a final pipeline (scaler + XGBRegressor) on the full dataset.  
2. Extracts the scaled feature matrix from the fitted pipeline and computes SHAP values using `shap.TreeExplainer` with an interventional background.  
3. Saves:
   - SHAP arrays and explainer objects (`shap_*.pkl`).  
   - Summary tables with mean absolute SHAP values and cumulative importance (`shapsummary_*.csv`).  
   - Optionally, beeswarm and bar plots for the top‑ranked features.

The outputs are written to a user‑specified directory (for example under `results/shap/`) controlled via arguments or variables in the script.


### 4. Descriptor extraction from OM and SEM micrographs

The script `image_processing/extract_descriptors.py` implements the
descriptor-extraction pipeline used in the paper. It reads optical microscopy
(OM) or scanning electron microscopy (SEM) images from a user-specified
directory, applies preprocessing (for example grayscale conversion, resizing,
and masking), and computes microstructure and texture descriptors. These
include morphology descriptors, gray-level co-occurrence matrix (GLCM)
statistics, local binary patterns, two-point spatial correlations, and optional
CNN Gram-matrix features. The resulting descriptor table is suitable for the
machine-learning workflow.

### 5. Predict properties from new micrographs

`predict_from_micrograph.py` provides an end-to-end workflow to predict tensile
properties from prepared optical-microscopy (OM) or SEM micrographs. For each
image, the script extracts the same descriptor families used in the modelling
pipeline, combines them with composition and imaging metadata, preprocesses the
resulting feature vector, trains an XGBoost model on a filtered literature
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
4. Fits the feature-reduction and cleaning steps using the filtered training
   dataset only.
5. Aligns every query image to the resulting training-feature schema.
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

- Experiments are controlled by parameters defined inside the scripts (such as number of Optuna trials, cross‑validation folds, tolerance thresholds) and, where present, configuration files referenced from those scripts.  
- Group‑aware cross‑validation is used when a `sample_id` column is available to avoid information leakage between micrographs from the same sample.  
- Random seeds (for example `rands=2026`, `base_seed=42`) are explicitly specified to support reproducibility.

## Citation

If you use this code or data in your own work, please cite:

> Author(s), “A Transferable, Explainable AI Framework for Composition–Structure–Property Mapping from Steel Micrographs”, Journal, Year, Volume, Pages, DOI.
(Replace with the final bibliographic information when available.)
