import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_domain_eda(df_trans):
    """
    Conducts domain-specific exploratory data analysis, calculates missing value 
    percentages, drops extremely sparse features, and generates visualization charts.
    """
    os.makedirs('../Figures', exist_ok=True)
    
    missing_percentage = (df_trans.isnull().sum() / len(df_trans)) * 100
    
    threshold = 50.0
    cols_to_drop = missing_percentage[missing_percentage > threshold].sort_values(ascending=False)
    
    num_dropped = len(cols_to_drop)
    top_15_dropped = cols_to_drop.head(15)
    
    logging.info(f"Total columns dropped (>{threshold}% null): {num_dropped}")
    logging.info(f"\nTop 15 dropped columns and their missing percentages:\n{top_15_dropped}")
    
    df_filtered = df_trans.drop(columns=cols_to_drop.index)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_15_dropped.index, y=top_15_dropped.values, hue=top_15_dropped.index, palette="Reds_r", legend=False)
    plt.title('Top 15 Dropped Features by Missing Percentage')
    plt.ylabel('Missing Percentage (%)')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../Figures/top_15_dropped_features.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_filtered, x='isFraud', hue='isFraud', palette='Set2', legend=False)
    plt.title('Fraud Class Imbalance (isFraud)')
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/fraud_class_imbalance.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x='isFraud', y='TransactionAmt', hue='isFraud', palette='Set3', legend=False)
    plt.yscale('log')
    plt.title('Transaction Amount by Fraud Status (Log Scale)')
    plt.ylabel('Transaction Amount (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/transaction_amt_vs_fraud.png')
    plt.close()

    if 'ProductCD' in df_filtered.columns:
        productcd_fraud = df_filtered.groupby(['ProductCD', 'isFraud']).size().unstack(fill_value=0)
        productcd_fraud.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title('Transaction Product Code (ProductCD) by Fraud Status')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../Figures/productcd_vs_fraud.png')
        plt.close()

    return df_filtered

if __name__ == "__main__":
    df_trans, _ = load_dataset()
    df_clean = perform_domain_eda(df_trans)