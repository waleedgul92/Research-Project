# Mitigating Explanation Leakage in Financial Fraud Detection Systems



This repository contains the complete experimental framework and codebase for a dissertation investigating **privacy vulnerabilities in Explainable AI (XAI)**. It implements a centralised XGBoost baseline, a standard Federated Learning (FL) architecture, and a novel **DP-FedSHAP** mechanism designed to mitigate Membership Inference Attacks (MIAs) against post-hoc TreeSHAP explanation vectors.

---

## Key Results (Ablation Study)

Privacy leakage is measured via **Membership Inference Attack (MIA) accuracy**, where **50% represents a random guess**. The table below highlights the optimal configuration identified during the study.

| Architecture | MIA Accuracy | Top-10 Feature Overlap | Spearman Rank | Privacy Leakage (L) |
|---|---|---|---|---|
| Centralised Baseline | 61.67% | 100.0% | 1.000 | 11.67% |
| Standard Federated (FL) | 51.58% | 100.0% | 0.980 | 1.58% |
| **DP-FedSHAP (ε = 1.2)** | **47.50%** | **80.0%** | **0.793** | **Formal Guarantee** |

By applying **dynamic 95th-percentile clipping** and **Laplacian noise** at ε = 1.2, DP-FedSHAP successfully drives MIA accuracy **below the 50% random-guess threshold**. This provides formal differential privacy guarantees while retaining 80% of the most critical fraud-indicating features, satisfying both GDPR privacy constraints and regulatory transparency requirements.

---

## Methodology Highlights

**Dataset Setup** — Utilises the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (10,000-record stratified sample), strictly preserving the highly imbalanced 3.5% real-world fraud prevalence.

**Preprocessing** — Features with >70% missing values are dropped. The pipeline applies training-set-only median imputation, label encoding, and dimensionality reduction down to the top 40 features via XGBoost embedded feature selection.

**Federated Architecture** — Simulates Horizontal Federated Learning (HFL) across 3 distinct institutional banking nodes, exchanging mathematical model weights instead of raw customer records.

**Threat Model** — Assumes a maximally sophisticated adversary equipped with the same algorithmic tools as the defender, deploying an XGBoost shadow model to exploit variance, L2 norm, and maximum absolute values of the post-hoc SHAP vectors.

---

## Repository Structure

```
.
├── config.json                  # Centralised hyperparameters, privacy budget (ε=1.2), and dataset limits
├── run_baseline.py              # Trains the intentionally overfit centralised model and extracts baseline SHAP values
├── run_federated.py             # Simulates standard Federated Learning across three institutional nodes
├── run_dp_federated.py          # Orchestrates the full DP-FedSHAP pipeline and averages privatised SHAP matrices
│
├── data_pipeline/               # Data Engineering Engine
│   ├── dataset.py               # Handles Kaggle API downloads, dataset merging, and train/test splitting
│   ├── preprocess.py            # Applies missing-value thresholds, median imputation, label encoding, feature selection
│   ├── eda.py                   # Generates domain-specific exploratory data visualisations
│   └── fl_data_splitter.py      # Partitions data via stratified sampling into exclusive slices for three bank nodes
│
├── federated_core/              # Model & Network Engine
│   ├── model.py                 # Optimised XGBoost training via multi-objective Optuna hyperparameter tuning
│   ├── fl_client.py             # Logic for isolated local bank node training across the horizontal federation
│   ├── dp_fed_client.py         # Implements dynamic 95th-percentile clipping and Laplacian noise injection
│   └── fl_server.py             # Central server aggregation executing weighted FedAvg across the local models
│
├── evaluation/                  # Security & Utility Metrics
│   ├── compare_all_xai.py       # Calculates MAE, Top-10 Feature Overlap, and Spearman Rank
│   └── membership_inference.py  # Executes the XGBoost-based shadow model Membership Inference Attack (MIA)
│
├── Figures/                     # EDA plots, class imbalance charts, and TreeSHAP summary visualisations
├── Models/                      # Trained XGBoost models (.json) and processed .npy SHAP arrays
└── Data/                        # Downloaded Kaggle datasets, preprocessed feature sets, and federated splits
```

---

## Prerequisites & Environment Setup

### 1. System Requirements

Due to the memory-intensive nature of the IEEE-CIS dataset and the TreeSHAP explainer, machines with 8 GB of RAM running WSL (Windows Subsystem for Linux) must allocate sufficient swap space.

Configure your `~/.wslconfig` file in Windows **before starting**:

```ini
[wsl2]
memory=4GB
swap=16GB
```

### 2. Virtual Environment Setup

The framework is implemented in **Python 3.12**. Clone the repository and set up an isolated environment:

```bash
python3.12 -m venv dissertation_env
source dissertation_env/bin/activate
pip install pandas scikit-learn xgboost>=2.0 shap>=0.44 optuna>=3.6 numpy matplotlib seaborn kaggle
```

### 3. Kaggle API Authentication

The `dataset.py` script automatically downloads the IEEE-CIS Fraud Detection dataset directly from Kaggle. You must provide your own API credentials.

1. Log into [Kaggle](https://www.kaggle.com) and generate a `kaggle.json` API token from your account settings.
2. Create a hidden `.kaggle` folder inside the `code/` directory.
3. Place your `kaggle.json` file inside that folder.

```
code/
└── .kaggle/
    └── kaggle.json
```

---

## Execution Pipeline

Run the following scripts **in order** from inside the `code/` directory to replicate the full pipeline and three-way ablation study.

### Step 1 — Download & Preprocess Data
```bash
cd code
python data_pipeline/dataset.py
```

### Step 2 — Generate Exploratory Data Analysis
```bash
python data_pipeline/eda.py
```

### Step 3 — Train Centralised Baseline Model
```bash
python run_baseline.py
```

### Step 4 — Execute Standard Federated Learning
```bash
python run_federated.py
```

### Step 5 — Execute Privacy-Preserving DP-FedSHAP
```bash
python run_dp_federated.py
```

### Step 6 — Evaluate Security & Utility Outcomes
```bash
python evaluation/membership_inference.py
python evaluation/compare_all_xai.py
```

---

## Output Artefacts

| Directory | Contents |
|---|---|
| `Figures/` | EDA plots, class imbalance charts, and TreeSHAP summary visualisations |
| `Models/` | Trained XGBoost models (`.json`) and processed `.npy` arrays — baseline, standard FL, and `shap_differential_privacy.npy` |
| `Data/` | Downloaded Kaggle datasets, preprocessed feature sets, and federated training splits |

---

## Citation

If you reference this work, please cite the associated Master's dissertation:

```
Gul, M. W. (2025). Mitigating Explanation Leakage in Financial Fraud Detection Systems.
London Metropolitan University.
```
