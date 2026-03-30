import logging
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
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
    cv_splits = config.get('cv_splits', 5)
    
    scale_weight = (len(y_local) - sum(y_local)) / sum(y_local)
    logging.info(f"[Bank Node {node_id}] Local scale_pos_weight: {scale_weight:.4f}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42 + node_id)
    
    train_scores = []
    val_scores = []

    logging.info(f"[Bank Node {node_id}] Initiating {cv_splits}-Fold Stratified Cross-Validation...")

    X_np = X_local.values if hasattr(X_local, 'values') else np.array(X_local)
    y_np = y_local.values if hasattr(y_local, 'values') else np.array(y_local)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_np, y_np), 1):
        X_tr, y_tr = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        model = xgb.XGBClassifier(
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
        
        model.fit(X_tr, y_tr)
        
        train_preds = model.predict_proba(X_tr)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        
        train_aucpr = average_precision_score(y_tr, train_preds)
        val_aucpr = average_precision_score(y_val, val_preds)
        
        train_scores.append(train_aucpr)
        val_scores.append(val_aucpr)
        
        logging.info(f"[Bank Node {node_id}] Fold {fold} - Train AUPRC: {train_aucpr:.4f} | Val AUPRC: {val_aucpr:.4f} | Gap: {train_aucpr - val_aucpr:.4f}")

    logging.info(f"[Bank Node {node_id}] --- XGBoost {cv_splits}-Fold CV Results ---")
    logging.info(f"[Bank Node {node_id}] Mean Train AUPRC: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
    logging.info(f"[Bank Node {node_id}] Mean Val AUPRC:   {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")
    logging.info(f"[Bank Node {node_id}] Mean Gap:         {np.mean(train_scores) - np.mean(val_scores):.4f}")

    logging.info(f"[Bank Node {node_id}] Fitting final local XGBoost model on complete local dataset...")
    
    final_model = xgb.XGBClassifier(
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
    final_model.fit(X_np, y_np)
    
    logging.info(f"[Bank Node {node_id}] Local model training complete.")
    
    return final_model