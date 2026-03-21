import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_federated_ensemble(models, X_test, y_test):
    logging.info("Central Server: Aggregating local models for global evaluation...")
    
    ensemble_preds = np.zeros(len(X_test))
    
    for model in models:
        ensemble_preds += model.predict_proba(X_test)[:, 1]
        
    ensemble_preds /= len(models)
    
    global_auprc = average_precision_score(y_test, ensemble_preds)
    logging.info(f"============================================================")
    logging.info(f"FEDERATED ENSEMBLE GLOBAL AUPRC SCORE: {global_auprc:.4f}")
    logging.info(f"============================================================")
    
    return global_auprc

def extract_federated_shap(models, X_test, X_train):
    logging.info("Central Server: Initiating Global Federated XAI Extraction...")
    
    X_test_sample = X_test.sample(n=2000, random_state=42)
    X_train_sample = X_train.sample(n=2000, random_state=42)
    
    all_test_shap_values = []
    all_train_shap_values = []
    
    for i, model in enumerate(models):
        logging.info(f"Calculating Shapley values for Bank Node {i+1}...")
        explainer = shap.TreeExplainer(model)
        
        test_shap_values = explainer.shap_values(X_test_sample)
        all_test_shap_values.append(test_shap_values)
        
        train_shap_values = explainer.shap_values(X_train_sample)
        all_train_shap_values.append(train_shap_values)
        
    logging.info("Averaging local Shapley matrices into Global Explanations...")
    global_test_shap_values = np.mean(all_test_shap_values, axis=0)
    global_train_shap_values = np.mean(all_train_shap_values, axis=0)
    
    os.makedirs('../Models', exist_ok=True)
    np.save('../Models/shap_federated.npy', global_test_shap_values)
    np.save('../Models/train_shap_federated.npy', global_train_shap_values)
    logging.info("Federated SHAP matrices saved to ../Models/")
    
    os.makedirs('../Figures', exist_ok=True)
    
    logging.info("Rendering Federated SHAP Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(global_test_shap_values, X_test_sample, show=False)
    plt.title("Federated Learning Global SHAP Summary (Ensemble)")
    plt.tight_layout()
    plt.savefig('../Figures/shap_summary_federated.png')
    plt.close()
    
    logging.info("Federated SHAP extraction complete and saved successfully.")