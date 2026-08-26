# A cohort-adaptive multi-omics integration framework for predicting immune checkpoint blockade efficacy

## Overview

**ICIMTL** is a cohort-adaptive multi-task learning framework for predicting objective response to immune checkpoint blockade (ICB). It jointly learns from heterogeneous immunotherapy cohorts while retaining clinical and genomic variables that are available only in individual cohorts.

The framework integrates:

* RNA-seq representations learned by a TCGA-pretrained autoencoder;
* cohort-shared covariates, including tumor mutational burden (TMB) and sex;
* cohort-specific clinical and genomic features;
* a Multi-gate Mixture-of-Experts (MMoE) backbone for cross-cohort knowledge sharing; and
* cohort-specific gated prediction heads for adaptive response prediction.

The current implementation jointly analyzes four independent ICB cohorts: **DavidA**, **DavidLiu**, **Ravi**, and **IMvigor210**. Model performance is evaluated using five-fold stratified cross-validation and compared with Logistic Regression, XGBoost, and a neural network baseline.

## Features

* **Cohort-adaptive multi-task learning**: models each ICB cohort as a related task while preserving cohort-specific characteristics.
* **Multi-omics integration**: combines RNA-seq representations with clinical and genomic variables.
* **TCGA-based representation learning**: uses a pretrained autoencoder to transform high-dimensional RNA-seq profiles into compact latent representations.
* **Shared and cohort-specific features**: separates variables shared across cohorts from features available only in individual cohorts.
* **MMoE architecture**: uses multiple shared experts and cohort-specific gates for cross-cohort knowledge transfer.
* **Adaptive prediction heads**: combines shared representations with cohort-specific clinical/genomic representations using learnable gating.
* **Transfer learning**: initializes compatible shared components using TCGA-pretrained parameters.
* **Five-fold cross-validation**: evaluates performance using consistent stratified folds.
* **Baseline comparison**: compares ICIMTL with Logistic Regression, XGBoost, and Multi-Layer Perceptron models.
* **Multiple evaluation metrics**: computes AUROC, AUPRC, F1 score, accuracy, sensitivity, specificity, PPV, and NPV.
* **Prediction export**: saves cross-validation predictions and trained ICIMTL models for downstream analysis.

## Repository Structure

```text
ICIMTL/
│
├── README.md
├── requirements.txt
│
├── ICIMTL/
│   ├── compare_with_baselinemodel.py   # Main training, cross-validation, and comparison script
│   ├── ICIMTL_model.py                 # ICIMTL model architecture
│   ├── ICIdataset.py                   # Data processing and PyTorch dataset
│   ├── autoencoder.py                  # TCGA-pretrained RNA autoencoder architecture
│   ├── DNN.py                          # Basic DNN modules
│   ├── utils1.py                       # Training and evaluation utilities
│   └── config.py                       # Model hyperparameters
│
├── icidata/
│   ├── clinical/
│   │   ├── DavidA_clinical.csv
│   │   ├── DavidLiu_clinical.csv
│   │   ├── Ravi_clinical.csv
│   │   └── IMvigor210_clinical.csv
│   │
│   └── rna/
│       ├── DavidA_rna.csv
│       ├── DavidLiu_rna.csv
│       ├── Ravi_rna.csv
│       └── IMvigor210_rna.csv
│
├── TCGA_pretraining/
│   ├── encoder.pth
│   ├── shared_encoder.pth
│   ├── selected_genes.json
│   └── std_scaler.joblib
│
└── output1/
    ├── MTL_val_predictions.xlsx
    ├── LogisticRegression_val_predictions.xlsx
    ├── XGBoost_val_predictions.xlsx
    ├── NeuralNetwork_val_predictions.xlsx
    └── best_model_fold_*.pth
```

## Installation

### Prerequisites

* **Python 3.10**
* **pip** package manager
* Recommended: use a virtual environment such as `venv` or `conda`
* Optional: CUDA-capable GPU for faster model training

The experiments were developed with **Python 3.10** and **PyTorch 2.3.1**. CPU execution is supported but may be slower.

### Dependencies

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```text
torch
numpy
pandas
scikit-learn
xgboost
matplotlib
joblib
openpyxl
```

For closest agreement with the reported software environment, use:

```text
Python          3.10
PyTorch         2.3.1
NumPy           1.26.4
pandas          2.2.2
scikit-learn    1.7.1
XGBoost         2.1.1
```

### Installation Steps

1. **Clone the Repository**

```bash
git clone https://github.com/WuDanni701/ICIMTL.git
cd ICIMTL
```

2. **Set Up a Virtual Environment (Optional but Recommended)**

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. **Upgrade pip and Install Dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The processed ICB datasets used in the experiments are provided in:

```text
icidata/
```

The directory contains two types of input data:

```text
icidata/
├── clinical/
└── rna/
```

### Clinical Data

Clinical and genomic features are stored in:

```text
icidata/clinical/
```

including:

```text
DavidA_clinical.csv
DavidLiu_clinical.csv
Ravi_clinical.csv
IMvigor210_clinical.csv
```

The clinical datasets contain the objective response labels together with cohort-shared and cohort-specific clinical/genomic variables.

Shared covariates used by the current model include:

```text
Sex_F
Sex_M
TMB
```

Cohort-specific variables differ between studies and include features such as age, tumor purity, neoantigen burden, clonal/subclonal TMB, genomic alterations, ECOG status, smoking-related variables, immune phenotype, and other cohort-specific measurements.

### RNA-seq Data

RNA expression matrices are stored in:

```text
icidata/rna/
```

including:

```text
DavidA_rna.csv
DavidLiu_rna.csv
Ravi_rna.csv
IMvigor210_rna.csv
```

Clinical and RNA-seq samples are matched by patient identifiers during data loading.

The four datasets are mapped to four prediction tasks:

| Task ID | Cohort     |
| ------- | ---------- |
| 0       | DavidA     |
| 1       | DavidLiu   |
| 2       | Ravi       |
| 3       | IMvigor210 |

### TCGA Pretrained Components

Pretrained components used for RNA representation learning and transfer learning are provided in:

```text
TCGA_pretraining/
```

The directory contains:

```text
encoder.pth
shared_encoder.pth
selected_genes.json
std_scaler.joblib
```

No additional TCGA pretraining is required to run the main experiment.

## Code Explanation

The project is organized into several modules, each responsible for a different part of data processing, model construction, training, and evaluation.

### 1. Data Preparation

#### `ICIProcessor`

Defined in `ICIMTL/ICIdataset.py`.

`ICIProcessor` loads RNA-seq and clinical data from the four cohorts, matches samples by patient ID, extracts shared features and cohort-specific features, and assigns task IDs.

```python
class ICIProcessor:
    def load_all(self):
        # Load and match RNA-seq and clinical data
        # Construct shared and cohort-specific features
        # Combine all four cohorts
        return (
            all_rna,
            all_shared,
            all_specific,
            all_response,
            all_task_ids,
            OSs,
            OSEs,
            TMB_origin
        )
```

#### `ICIDataset`

Converts processed features and labels into PyTorch tensors used during model training.

```python
class ICIDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "rna": ...,
            "shared": ...,
            "specific": ...,
            "label": ...,
            "task_id": ...
        }
```

#### `load_tcga_artifacts`

Defined in `ICIMTL/compare_with_baselinemodel.py`.

Loads the selected genes, pretrained expression scalers, autoencoder parameters, and pretrained shared encoder.

```python
def load_tcga_artifacts(output_dir_tcga):
    # Load selected_genes.json
    # Load std_scaler.joblib
    # Load encoder.pth
    # Locate shared_encoder.pth
    return selected_genes, ss, enc_state, se_path
```

#### `prepare_ici_data`

Applies cohort-specific RNA scaling, selects TCGA-derived genes, and generates latent RNA representations using the pretrained autoencoder.

```python
def prepare_ici_data(
    rna_paths,
    cohort_paths,
    shared_cols,
    specific_cols,
    autoencoder,
    device,
    tcga_scaler,
    selected_genes,
    batch_size=50
):
    # Load ICB cohorts
    # Scale RNA expression
    # Select pretrained genes
    # Generate latent RNA representations
    return (
        encoded_rna_df,
        all_shared,
        all_specific,
        all_response,
        all_task_ids,
        OSs,
        OSEs,
        TMB_origin
    )
```

#### `scale_fold`

Performs fold-specific scaling of continuous clinical features using only the training samples to avoid information leakage.

```python
def scale_fold(
    shared_feats,
    specific_feats,
    task_ids,
    train_idx,
    val_idx,
    shared_continuous_cols,
    task_specific_continuous_cols
):
    # Fit RobustScaler on training samples
    # Apply the fitted transformations to validation samples
    return shared_scaled, specific_scaled, fitted_scalers
```

### 2. RNA Representation Learning

#### `Autoencoder`

Defined in `ICIMTL/autoencoder.py`.

The autoencoder provides the pretrained encoder used to transform high-dimensional RNA-seq profiles into latent transcriptomic representations.

```python
class Autoencoder(nn.Module):
    def forward(self, x):
        # Encode RNA expression
        # Reconstruct expression distribution
        return u, logvar, z
```

During the ICB experiment, the pretrained encoder is used to obtain the latent representation `z`.

### 3. ICIMTL Model

The main multi-task learning architecture is defined in `ICIMTL/ICIMTL_model.py`.

#### `sharedEncoder`

Implements the MMoE backbone. Multiple experts learn shared representations, while task-specific gates determine how expert outputs are combined for individual cohorts.

```python
class sharedEncoder(nn.Module):
    def forward(self, dnn_input):
        # Compute outputs from shared experts
        # Compute task-specific gate weights
        # Combine expert representations
        return mmoe_outs, total_loss
```

#### `TaskSpecificHead`

Encodes cohort-specific features and adaptively combines them with the shared MMoE representation.

```python
class TaskSpecificHead(nn.Module):
    def forward(self, shared_feat, specific_feat):
        # Encode shared representation
        # Encode cohort-specific representation
        # Learn adaptive fusion weight
        # Predict ICB response
        return logit
```

#### `ICIMTL`

Combines the MMoE shared encoder with four cohort-specific prediction heads.

```python
class ICIMTL(nn.Module):
    def forward(
        self,
        shared_input,
        spc_inputs,
        task_ids,
        task_specific_slices
    ):
        # Generate shared task representations
        # Select cohort-specific features
        # Apply corresponding task head
        return logits_list, total_loss
```

### 4. Model Training

Training utilities are implemented in `ICIMTL/utils1.py`.

#### `ICITransferTrainer`

Controls optimization of the shared and cohort-specific components.

```python
class ICITransferTrainer:
    def train_epoch(self, loader):
        # Forward propagation
        # Compute task-specific weighted losses
        # Add regularization
        # Backpropagation and parameter update
        return train_loss, train_metrics
```

When `freeze_shared=True`, the shared expert parameters use a learning rate of zero, while compatible shared gating and cohort-specific components can still be optimized.

#### `update_pos_weights_from_loader`

Computes cohort-specific class weights to account for imbalance between responders and non-responders.

```python
def update_pos_weights_from_loader(self, loader):
    # Count positive and negative samples per task
    # Update BCEWithLogitsLoss weights
```

#### `evaluate`

Evaluates the model independently for each cohort.

```python
def evaluate(self, loader, return_loss=False):
    # Generate probabilities
    # Collect labels by task
    # Compute cohort-specific metrics
    return metrics
```

#### `EarlyStopping`

Monitors validation performance and stops training when model performance no longer improves.

```python
class EarlyStopping:
    def step(self, metrics, epoch):
        # Track the best validation metric
        # Update early-stopping counter
        return improved
```

### 5. Five-Fold Cross-Validation and Baseline Comparison

#### `five_fold_cross_validation`

Defined in `ICIMTL/compare_with_baselinemodel.py`.

This is the main function for reproducing the experiment.

```python
def five_fold_cross_validation(freeze_shared=False):
    # Load ICB data and TCGA pretrained components
    # Construct five stratified folds
    # Perform fold-specific preprocessing
    # Train ICIMTL
    # Train Logistic Regression
    # Train XGBoost
    # Train Neural Network baseline
    # Evaluate all models
    # Save predictions and trained models
    # Plot cross-validated ROC curves
```

The folds are constructed using:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)
```

Stratification considers both the cohort/task assignment and the binary response label.

The current entry point uses:

```python
five_fold_cross_validation(freeze_shared=True)
```

With `freeze_shared=True`, compatible parameters from:

```text
TCGA_pretraining/shared_encoder.pth
```

are transferred to the ICIMTL shared encoder before cross-validation training.

### 6. Performance Evaluation

#### `compute_all_metrics`

Computes classification performance from observed response labels and predicted probabilities.

```python
def compute_all_metrics(y_true, y_prob, threshold=None):
    # AUROC
    # AUPRC
    # F1 score
    # Accuracy
    # Sensitivity
    # Specificity
    # PPV
    # NPV
    return metrics
```

#### `AUC_calculator`

Calculates AUROC and identifies a ROC-based classification threshold when a threshold is not explicitly supplied.

```python
def AUC_calculator(y, y_pred):
    # Calculate ROC curve and AUROC
    # Identify the optimal threshold
    return auroc, threshold
```

### 7. Result Saving

Prediction results are exported inside `five_fold_cross_validation()`.

The validation predictions for each model are saved to:

```text
output1/MTL_val_predictions.xlsx
output1/LogisticRegression_val_predictions.xlsx
output1/XGBoost_val_predictions.xlsx
output1/NeuralNetwork_val_predictions.xlsx
```

Each file contains information including:

```text
fold
cohort
patientID
response
prob
OS
OS_event
TMB
```

The best ICIMTL model from each fold is also saved as:

```text
output1/best_model_fold_0.pth
output1/best_model_fold_1.pth
output1/best_model_fold_2.pth
output1/best_model_fold_3.pth
output1/best_model_fold_4.pth
```

### 8. Visualization

#### `plot_cv_roc_curves`

Generates cross-validated ROC curves for ICIMTL and the three baseline models for each cohort.

```python
def plot_cv_roc_curves(
    models_metrics_dict,
    task_id,
    task_name
):
    # Aggregate ROC curves across folds
    # Calculate mean AUROC
    # Plot ICIMTL and baseline ROC curves
    plt.show()
```

The plotted models include:

```text
ICIMTL
Logistic Regression
XGBoost
Neural Network
```

## Example Run

The complete experiment can be reproduced using the main comparison script.

From the repository root:

```bash
python ICIMTL/compare_with_baselinemodel.py
```

The script performs five-fold cross-validation, trains ICIMTL and the three baseline models, evaluates cohort-specific predictive performance, generates ROC curves, and saves validation predictions and trained ICIMTL models to `output1/`.

> **GPU selection:** the current script selects `cuda:7` whenever CUDA is available. If GPU 7 is not available on your machine, change the device definition in `ICIMTL/compare_with_baselinemodel.py` to an available device such as `cuda:0`, or use `cpu`.

> **Working directory:** run the script from the repository root because the current implementation uses relative paths to `icidata/`, `TCGA_pretraining/`, and `output1/`.
