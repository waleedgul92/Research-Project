import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_iid_partitions(X_train, y_train, num_nodes=3):
    logging.info(f"Initiating Stratified IID Data Split across {num_nodes} Simulated Bank Nodes...")
    
    skf = StratifiedKFold(n_splits=num_nodes, shuffle=True, random_state=42)
    
    partitions = []
    node_id = 1
    
    for _, node_idx in skf.split(X_train, y_train):
        X_local = X_train.iloc[node_idx]
        y_local = y_train.iloc[node_idx]
        
        fraud_ratio = y_local.mean()
        logging.info(f"Bank Node {node_id} Provisioned | Rows: {len(X_local)} | Fraud Ratio: {fraud_ratio:.4f}")
        
        partitions.append((X_local, y_local))
        node_id += 1
        
    logging.info("IID Data Fragmentation Complete. All nodes contain isolated, non-overlapping data.")
    return partitions