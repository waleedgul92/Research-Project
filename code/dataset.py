import os
import zipfile
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """
    Loads the configuration variables from the config.json file.
    """
    with open('config.json', 'r') as f:
        return json.load(f)

def download_dataset():
    """
    Authenticates with the Kaggle API, downloads the IEEE-CIS Fraud Detection 
    dataset if not already present, and extracts it into the Dataset directory.
    """
    if os.path.exists(os.path.join('../Dataset', 'train_transaction.csv')):
        logging.info("Dataset already downloaded and extracted. Skipping.")
        return
        
    from kaggle.api.kaggle_api_extended import KaggleApi

    logging.info("Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    logging.info("Downloading IEEE-CIS Fraud Detection dataset (this may take a few minutes)...")
    api.competition_download_files('ieee-fraud-detection', path='.')

    logging.info("Extracting files into Dataset folder...")
    with zipfile.ZipFile('ieee-fraud-detection.zip', 'r') as zip_ref:
        zip_ref.extractall('Dataset')

    logging.info("Cleaning up zip file...")
    os.remove('ieee-fraud-detection.zip')
    
    logging.info("Download and extraction complete!")

def load_dataset():
    """
    Reads the transaction and identity CSV files based on the row limit 
    specified in the configuration and returns them as pandas DataFrames.
    """
    config = load_config()
    nrows_limit = config['nrows']
    
    df_trans = pd.read_csv('../Dataset/train_transaction.csv', nrows=nrows_limit)
    df_id = pd.read_csv('../Dataset/train_identity.csv')
    
    logging.info(f"Transaction data shape: {df_trans.shape}")
    logging.info(f"Identity data shape: {df_id.shape}")
    
    return df_trans, df_id


# def load_dataset_balanced():
#     """
#     Loads the full datasets and applies Random Undersampling (RUS) to balance 
#     the classes before returning the specified number of rows.
#     """
#     config = load_config()
#     nrows_limit = config['nrows']
    
#     logging.info("Loading full transaction dataset to perform Random Undersampling...")
#     df_trans_full = pd.read_csv('../Dataset/train_transaction.csv')
#     df_id = pd.read_csv('../Dataset/train_identity.csv')
    
#     fraud_cases = df_trans_full[df_trans_full['isFraud'] == 1]
#     legit_cases = df_trans_full[df_trans_full['isFraud'] == 0]
    
#     n_fraud = min(len(fraud_cases), int(nrows_limit / 2))
#     n_legit = nrows_limit - n_fraud
    
#     fraud_sampled = fraud_cases.sample(n=n_fraud, random_state=42)
#     legit_sampled = legit_cases.sample(n=n_legit, random_state=42)
    
#     df_trans = pd.concat([fraud_sampled, legit_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
#     logging.info(f"Transaction data shape after RUS: {df_trans.shape}")
#     logging.info(f"Identity data shape: {df_id.shape}")
#     logging.info(f"Fraud ratio in sampled data: {df_trans['isFraud'].mean():.4f}")
    
#     return df_trans, df_id

def process_data(df_trans, df_id):
    """
    Merges transaction and identity dataframes, isolates the target variable, 
    and applies an 80/20 stratified train-test split.
    """
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    download_dataset()
    df_trans, df_id = load_dataset()
    X_train, X_test, y_train, y_test = process_data(df_trans, df_id)