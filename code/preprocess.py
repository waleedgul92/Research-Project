import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from dataset import load_dataset, process_data, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_features(X_train, X_test, y_train):
    config = load_config()
    drop_threshold = config.get('drop_threshold', 70.0)
    top_k_features = config.get('top_k_features', 40)
    
    initial_feature_count = X_train.shape[1]
    logging.info(f"Starting feature count: {initial_feature_count}")
    
    missing_pct = (X_train.isnull().sum() / len(X_train)) * 100
    cols_to_drop_missing = missing_pct[missing_pct > drop_threshold].index
    
    X_train = X_train.drop(columns=cols_to_drop_missing)
    X_test = X_test.drop(columns=cols_to_drop_missing)
    logging.info(f"Applied {drop_threshold}% missing data threshold. Dropped {len(cols_to_drop_missing)} features.")
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    variances = X_train[num_cols].var()
    constant_cols = variances[variances == 0].index.tolist()
    
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)
    logging.info(f"Variance thresholding complete. Dropped {len(constant_cols)} constant/zero-variance features.")
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logging.info("Calculating Spearman correlation matrix for multicollinearity reduction...")
    corr_matrix = X_train[num_cols].sample(n=min(10000, len(X_train)), random_state=42).corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    X_train = X_train.drop(columns=to_drop_corr)
    X_test = X_test.drop(columns=to_drop_corr)
    logging.info(f"Applied 0.90 Spearman threshold. Dropped {len(to_drop_corr)} highly correlated features.")
    
    cat_cols = X_train.select_dtypes(include=['object']).columns
    logging.info(f"Initiating Label Encoding for {len(cat_cols)} categorical features...")
    
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing_Value').astype(str)
        X_test[col] = X_test[col].fillna('Missing_Value').astype(str)
        
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
        
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        
    logging.info("Label Encoding complete.")
    
    logging.info("Initiating Stage 2: Embedded Feature Selection via XGBoost Information Gain...")
    selection_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        random_state=42, 
        tree_method='hist',
        n_jobs=-1
    )
    selection_model.fit(X_train, y_train)
    
    importance_series = pd.Series(selection_model.feature_importances_, index=X_train.columns)
    top_features = importance_series.sort_values(ascending=False).head(top_k_features).index.tolist()
    
    dropped_embedded = X_train.shape[1] - len(top_features)
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    logging.info(f"Embedded Selection complete. Dropped {dropped_embedded} features with low information gain.")
    logging.info(f"Top 5 retained features: {top_features[:5]}")
    logging.info(f"Preprocessing finished. Features successfully reduced from {initial_feature_count} to {X_train.shape[1]}.")
    
    return X_train, X_test

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    
    X_train_clean, X_test_clean = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    logging.info(f"Final Training Data Shape: {X_train_clean.shape}")
    logging.info(f"Final Testing Data Shape: {X_test_clean.shape}")