import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset import load_dataset, process_data, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_features(X_train, X_test):
    config = load_config()
    drop_threshold = config.get('drop_threshold', 70.0)
    
    missing_pct = (X_train.isnull().sum() / len(X_train)) * 100
    cols_to_drop_missing = missing_pct[missing_pct > drop_threshold].index
    
    X_train = X_train.drop(columns=cols_to_drop_missing)
    X_test = X_test.drop(columns=cols_to_drop_missing)
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    variances = X_train[num_cols].var()
    constant_cols = variances[variances == 0].index.tolist()
    
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)
    logging.info(f"Dropped {len(constant_cols)} constant/zero-variance features.")
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    corr_matrix = X_train[num_cols].sample(n=min(10000, len(X_train)), random_state=42).corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    X_train = X_train.drop(columns=to_drop_corr)
    X_test = X_test.drop(columns=to_drop_corr)
    logging.info(f"Dropped {len(to_drop_corr)} highly correlated features.")
    
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing_Value').astype(str)
        X_test[col] = X_test[col].fillna('Missing_Value').astype(str)
        
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
        
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        
    return X_train, X_test

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    
    X_train_clean, X_test_clean = preprocess_features(X_train_raw, X_test_raw)
    
    logging.info(f"Final Training Data Shape: {X_train_clean.shape}")
    logging.info(f"Final Testing Data Shape: {X_test_clean.shape}")