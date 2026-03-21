import os
import logging
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from data_pipeline.dataset import load_dataset, process_data, load_config
from data_pipeline.preprocess import preprocess_features
from federated_core.dp_fed_client import get_dp_shap_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_dp_simulation():
    logging.info("--- STARTING OPTIMIZED DP FEDERATED SIMULATION ---")
    
    config = load_config()
    num_nodes = config.get('fl_num_nodes', 3)
    epsilon = config.get('dp_epsilon', 5.0) 
    
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, _ = process_data(df_trans, df_id)
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    X_test_sample = X_test.sample(n=2000, random_state=42)
    X_train_sample = X_train.sample(n=2000, random_state=42)
    
    all_noisy_shap_values = []
    all_noisy_train_shap_values = []
    
    for i in range(1, num_nodes + 1):
        model_path = f'../Models/bank_node_{i}_model.json'
        if os.path.exists(model_path):
            logging.info(f"Loading Bank Node {i} and applying Dynamic Clipping DP...")
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            noisy_shap = get_dp_shap_values(model, X_test_sample, epsilon, percentile=95)
            all_noisy_shap_values.append(noisy_shap)
            
            noisy_train_shap = get_dp_shap_values(model, X_train_sample, epsilon, percentile=95)
            all_noisy_train_shap_values.append(noisy_train_shap)
        else:
            logging.error(f"Missing model for Bank Node {i}.")
            return

    logging.info("Averaging NOISY local Shapley matrices at Central Server...")
    global_dp_shap_values = np.mean(all_noisy_shap_values, axis=0)
    global_dp_train_shap_values = np.mean(all_noisy_train_shap_values, axis=0)
    
    os.makedirs('../Models', exist_ok=True)
    np.save('../Models/shap_differential_privacy.npy', global_dp_shap_values)
    np.save('../Models/train_shap_differential_privacy.npy', global_dp_train_shap_values)
    
    os.makedirs('../Figures', exist_ok=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(global_dp_shap_values, X_test_sample, show=False)
    plt.title(f"Optimized DP-SHAP Summary (Epsilon={epsilon}, Clip=95th Percentile)")
    plt.tight_layout()
    plt.savefig('../Figures/shap_summary_dp_federated.png')
    plt.close()

if __name__ == "__main__":
    run_dp_simulation()