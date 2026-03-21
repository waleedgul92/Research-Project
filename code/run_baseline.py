import os
import logging
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from data_pipeline.dataset import load_dataset, process_data
from data_pipeline.preprocess import preprocess_features
from federated_core.model import train_cv_xgboost

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_shap_plots(model, X_train, X_test):
    logging.info("Preparing data samples for SHAP extraction...")
    X_test_sample = X_test.sample(n=2000, random_state=42)
    X_train_sample = X_train.sample(n=2000, random_state=42)
    
    logging.info("Initializing TreeSHAP and calculating values...")
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test_sample)
    np.save(os.path.join(MODELS_DIR, 'shap_baseline.npy'), shap_values)
    
    train_shap_values = explainer.shap_values(X_train_sample)
    np.save(os.path.join(MODELS_DIR, 'train_shap_baseline.npy'), train_shap_values)
    
    logging.info("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_baseline.png'))
    plt.close()
    
    logging.info("Baseline SHAP extraction complete and files saved successfully.")

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, _ = process_data(df_trans, df_id)
    
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    logging.info("Calling train_cv_xgboost from federated_core.model...")
    final_model = train_cv_xgboost(X_train, y_train)
    
    generate_shap_plots(final_model, X_train, X_test)