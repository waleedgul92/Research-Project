import os
import zipfile
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATASET_DIR = os.path.join(PARENT_DIR, 'Dataset')
KAGGLE_DIR = os.path.join(PARENT_DIR, '.kaggle')

os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_DIR
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def download_dataset():
    if os.path.exists(os.path.join(DATASET_DIR, 'train_transaction.csv')):
        logging.info(f"Dataset already exists at {DATASET_DIR}. Skipping download.")
        return
        
    os.makedirs(DATASET_DIR, exist_ok=True)
        
    from kaggle.api.kaggle_api_extended import KaggleApi

    logging.info("Authenticating Kaggle API for dataset download...")
    api = KaggleApi()
    api.authenticate()

    logging.info("Downloading IEEE-CIS Fraud Detection dataset. This may take several minutes...")
    api.competition_download_files('ieee-fraud-detection', path=DATASET_DIR)

    zip_path = os.path.join(DATASET_DIR, 'ieee-fraud-detection.zip')

    logging.info(f"Extracting zip archive from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    logging.info("Cleaning up zip archive...")
    os.remove(zip_path)
    
    logging.info("Dataset download and extraction successfully completed.")

def load_dataset():
    config = load_config()
    nrows_limit = config['nrows']
    
    logging.info(f"Loading full transaction dataset. Target row limit post-RUS: {nrows_limit}")
    df_trans_full = pd.read_csv(os.path.join(DATASET_DIR, 'train_transaction.csv'))
    df_id = pd.read_csv(os.path.join(DATASET_DIR, 'train_identity.csv'))
    
    logging.info(f"Original transaction dataset shape loaded: {df_trans_full.shape}")
    logging.info(f"Original identity dataset shape loaded: {df_id.shape}")
    
    fraud_cases = df_trans_full[df_trans_full['isFraud'] == 1]
    legit_cases = df_trans_full[df_trans_full['isFraud'] == 0]
    
    logging.info(f"Class distribution before RUS - Fraud: {len(fraud_cases)}, Legit: {len(legit_cases)}")
    
    n_fraud = min(len(fraud_cases), int(nrows_limit / 2))
    n_legit = nrows_limit - n_fraud
    
    logging.info(f"Executing Random Undersampling. Sampling {n_fraud} fraud cases and {n_legit} legit cases.")
    
    fraud_sampled = fraud_cases.sample(n=n_fraud, random_state=42)
    legit_sampled = legit_cases.sample(n=n_legit, random_state=42)
    
    df_trans = pd.concat([fraud_sampled, legit_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"Transaction data shape after Random Undersampling: {df_trans.shape}")
    logging.info(f"Identity data shape remains: {df_id.shape}")
    logging.info(f"Final Fraud ratio in sampled data: {df_trans['isFraud'].mean():.4f}")
    
    return df_trans, df_id

def process_data(df_trans, df_id):
    logging.info("Merging transaction and identity dataframes on TransactionID...")
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    logging.info(f"Merged dataframe shape: {df.shape}")
    
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])
    
    logging.info("Initiating 80/20 stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logging.info(f"Stratified split complete. Training features shape: {X_train.shape}")
    logging.info(f"Stratified split complete. Testing features shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    download_dataset()
    df_trans, df_id = load_dataset()
    X_train, X_test, y_train, y_test = process_data(df_trans, df_id)