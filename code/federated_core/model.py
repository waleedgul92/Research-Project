import os
import logging
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.metrics import average_precision_score
from data_pipeline.dataset import load_dataset, process_data, load_config
from data_pipeline.preprocess import preprocess_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_cv_xgboost(X_train, y_train):
    logging.info("Loading configuration parameters for XGBoost model...")
    config = load_config()
    
    n_estimators = config.get('xgb_n_estimators', 100)
    max_depth = config.get('xgb_max_depth', 6)
    learning_rate = config.get('xgb_learning_rate', 0.1)
    subsample = config.get('xgb_subsample', 1.0)
    colsample_bytree = config.get('xgb_colsample_bytree', 1.0)
    min_child_weight = config.get('xgb_min_child_weight', 1)
    cv_splits = config.get('cv_splits', 5)

    logging.info(f"Hyperparameters loaded - Estimators: {n_estimators}, Max Depth: {max_depth}, LR: {learning_rate}")
    logging.info(f"Regularization parameters - Subsample: {subsample}, Colsample: {colsample_bytree}, Min Child Weight: {min_child_weight}")

    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    logging.info(f"Calculated scale_pos_weight to handle class imbalance: {scale_weight:.4f}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    train_scores = []
    val_scores = []

    logging.info(f"Initiating {cv_splits}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            scale_pos_weight=scale_weight,
            random_state=42,
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
        
        logging.info(f"Fold {fold} - Train AUPRC: {train_aucpr:.4f} | Val AUPRC: {val_aucpr:.4f} | Gap: {train_aucpr - val_aucpr:.4f}")

    logging.info(f"--- XGBoost {cv_splits}-Fold Cross-Validation Results ---")
    logging.info(f"Mean Train AUPRC: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
    logging.info(f"Mean Val AUPRC:   {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")
    logging.info(f"Mean Gap:         {np.mean(train_scores) - np.mean(val_scores):.4f}")
    
    logging.info("Fitting final XGBoost model on the complete training dataset...")
    
    final_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    final_model.fit(X_train, y_train)
    
    os.makedirs('../Models', exist_ok=True)
    final_model.save_model('../Models/baseline_xgboost.json')
    logging.info("Final model saved successfully to ../Models/baseline_xgboost.json")
    
    return final_model


if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    final_model = train_cv_xgboost(X_train, y_train)