import os
import logging
from dataset import load_dataset, process_data, load_config
from preprocess import preprocess_features
from fl_data_splitter import create_iid_partitions
from fl_client import train_local_model
from fl_server import evaluate_federated_ensemble, extract_federated_shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_federated_simulation():
    logging.info("--- STARTING HORIZONTAL FEDERATED LEARNING SIMULATION ---")
    
    config = load_config()
    num_nodes = config.get('fl_num_nodes', 3)
    
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    partitions = create_iid_partitions(X_train, y_train, num_nodes=num_nodes)
    
    os.makedirs('../Models', exist_ok=True)
    federated_models = []
    
    for i, (X_local, y_local) in enumerate(partitions):
        node_id = i + 1
        model = train_local_model(node_id, X_local, y_local)
        
        model_path = f'../Models/bank_node_{node_id}_model.json'
        model.save_model(model_path)
        logging.info(f"[Bank Node {node_id}] Model securely saved to disk: {model_path}")
        
        federated_models.append(model)
        
    evaluate_federated_ensemble(federated_models, X_test, y_test)
    extract_federated_shap(federated_models, X_test)
    
    logging.info("--- FEDERATED LEARNING SIMULATION SUCCESSFULLY COMPLETED ---")

if __name__ == "__main__":
    run_federated_simulation()