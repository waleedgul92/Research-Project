# Research-Project


- This repository contains the centralized machine learning baseline for a Master's dissertation 
---

## Repository Structure

| Directory / File | Description |
|---|---|
| **`data_pipeline/`** | **Data Engineering Engine** |
| ├── `dataset.py` | Handles secure Kaggle API downloads and memory-safe train/test splitting. |
| ├── `preprocess.py` | Applies MNAR thresholds, label encoding, and feature selection. |
| ├── `eda.py` | Generates domain-specific exploratory data visualizations. |
| └── `fl_data_splitter.py`| Partitions data into IID slices for Federated bank nodes. |
| **`federated_core/`** | **Model & Network Engine** |
| ├── `model.py` | Optimized XGBoost training via Stratified 5-Fold Cross-Validation. |
| ├── `fl_client.py` | Standard logic for isolated local bank node training. |
| ├── `dp_fed_client.py` | Injects Laplacian noise into local SHAP matrices ($\epsilon$-DP). |
| └── `fl_server.py` | Central server aggregation for global ensemble evaluation and XAI. |
| **`evaluation/`** | **Security & Utility Metrics** |
| ├── `compare_all_xai.py` | Calculates MAE, Top-10 Overlap, and Spearman Rank for utility. |
| └── `membership_inference.py`| Executes the Random Forest Membership Inference Attack (MIA). |
| **`Execution Scripts`** | **Master Controls (Root Directory)** |
| ├── `config.json` | Centralized hyperparameters, privacy budgets ($\epsilon$), and dataset limits. |
| ├── `run_baseline.py` | Trains centralized model and extracts ground-truth SHAP values. |
| ├── `run_federated.py` | Simulates standard Federated Learning across nodes. |
| └── `run_dp_federated.py`| Simulates Privacy-Preserving Federated Learning with DP. |

---

## Prerequisites & Environment Setup

### 1. System Requirements

Due to the memory-intensive nature of the dataset and the TreeSHAP explainer, machines with 8 GB of RAM running WSL (Windows Subsystem for Linux) must allocate sufficient swap space.

If using WSL, ensure your `~/.wslconfig` file in Windows is configured as follows **before starting**:

```ini
[wsl2]
memory=4GB
swap=16GB
```

### 2. Virtual Environment Setup

Clone the repository and set up your isolated Python environment:

```bash
python3 -m venv dissertation_env
source dissertation_env/bin/activate
pip install pandas scikit-learn xgboost shap matplotlib seaborn kaggle
```

### 3. Kaggle API Authentication

The `dataset.py` script automatically downloads the 1.35 GB dataset directly from Kaggle. You must provide your own API credentials.

1. Log into Kaggle and generate a `kaggle.json` API token from your account settings.
2. Create a hidden `.kaggle` folder inside the `code` directory.
3. Place your `kaggle.json` file inside that `.kaggle` folder.

```
code/
└── .kaggle/
    └── kaggle.json
```

---

## Execution Pipeline

Run the scripts in the following **exact order** from inside the `code` directory to replicate the pipeline.

### Step 1: Download Data & Verify Splits

```bash
cd code
python dataset.py
```

### Step 2: Generate Exploratory Data Analysis

This will populate the `Figures` directory with visualizations regarding class imbalance and sparse feature distributions.

```bash
python eda.py
```

### Step 3: Train the Baseline Model

This executes the 5-Fold Stratified Cross-Validation, handles the severe class imbalance, and saves the final JSON model to the `Models` directory.

```bash
python model.py
```

### Step 4: Generate Explainability (TreeSHAP)

This extracts the baseline feature importances and saves the summary plot to the `Figures` directory.

```bash
python shap_analysis.py
```

---

## Output Artefacts

| Directory | Contents |
|---|---|
| `Figures/` | EDA plots, class imbalance charts, TreeSHAP summary plot |
| `Models/` | Trained XGBoost model saved as `.json` |

---

## Citation

If you reference this work, please cite the associated Master's dissertation accordingly.
