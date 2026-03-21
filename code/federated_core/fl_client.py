import logging
import xgboost as xgb
from data_pipeline.dataset import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_local_model(node_id, X_local, y_local):
    logging.info(f"[Bank Node {node_id}] Initializing local training sequence...")
    
    config = load_config()
    n_estimators = config.get('xgb_n_estimators', 100)
    max_depth = config.get('xgb_max_depth', 6)
    learning_rate = config.get('xgb_learning_rate', 0.1)
    subsample = config.get('xgb_subsample', 1.0)
    colsample_bytree = config.get('xgb_colsample_bytree', 1.0)
    min_child_weight = config.get('xgb_min_child_weight', 1)
    
    scale_weight = (len(y_local) - sum(y_local)) / sum(y_local)
    logging.info(f"[Bank Node {node_id}] Local scale_pos_weight: {scale_weight:.4f}")
    
    local_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_weight,
        random_state=42 + node_id,
        n_jobs=-1,
        tree_method='hist'
    )
    
    local_model.fit(X_local, y_local)
    logging.info(f"[Bank Node {node_id}] Local model training complete.")
    
    return local_model