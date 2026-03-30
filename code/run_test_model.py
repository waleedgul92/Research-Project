import os
import logging
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from data_pipeline.dataset import load_dataset, process_data
from data_pipeline.preprocess import preprocess_features
from federated_core.test_model import train_xgboost

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test= process_data(df_trans, df_id)
    
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    logging.info("Calling train_cv_xgboost from federated_core.model...")
    study = train_xgboost(X_train, y_train, X_test, y_test)

