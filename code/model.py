import os
import logging
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from dataset import load_dataset, process_data, load_config
from preprocess import preprocess_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_cv_xgboost(X_train, y_train):
    config = load_config()
    n_estimators = config.get('xgb_n_estimators', 100)
    max_depth = config.get('xgb_max_depth', 6)
    learning_rate = config.get('xgb_learning_rate', 0.1)
    cv_splits = config.get('cv_splits', 5)

    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring='average_precision',
        n_jobs=2
    )
    
    logging.info(f"--- XGBoost {cv_splits}-Fold Cross-Validation ---")
    logging.info(f"AUPRC Scores for each fold: {np.round(cv_scores, 4)}")
    logging.info(f"Mean AUPRC: {np.mean(cv_scores):.4f}")
    logging.info(f"Standard Deviation: {np.std(cv_scores):.4f}")
    
    model.fit(X_train, y_train)
    
    os.makedirs('../Models', exist_ok=True)
    model.save_model('../Models/baseline_xgboost.json')
    logging.info("Model saved successfully to ../Models/baseline_xgboost.json")
    
    return model

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw)
    
    final_model = train_cv_xgboost(X_train, y_train)