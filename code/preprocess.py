import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset import load_dataset, process_data, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_features(X_train, X_test):
    """
    Reads the drop threshold from config, drops sparse features, fills missing 
    categorical values, and applies label encoding to all object-type columns.
    """
    config = load_config()
    drop_threshold = config.get('drop_threshold', 70.0)
    
    missing_pct = (X_train.isnull().sum() / len(X_train)) * 100
    cols_to_drop = missing_pct[missing_pct > drop_threshold].index
    
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    
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