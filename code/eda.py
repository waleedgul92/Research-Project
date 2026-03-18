import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_comprehensive_eda(df_trans):
    """
    Conducts domain-specific and high-dimensional exploratory data analysis.
    Generates missing value charts, domain boxplots, correlation tornados, 
    and multicollinearity heatmaps to justify the thesis preprocessing pipeline.
    """
    os.makedirs('../Figures', exist_ok=True)
    
    # =====================================================================
    # PART 1: MISSING DATA & BASIC DOMAIN EDA (From your original code)
    # =====================================================================
    logging.info("Generating Missing Data and Domain EDA plots...")
    
    missing_percentage = (df_trans.isnull().sum() / len(df_trans)) * 100
    threshold = 70.0 # Updated to your academically defended 70% threshold
    cols_to_drop = missing_percentage[missing_percentage > threshold].sort_values(ascending=False)
    
    top_15_dropped = cols_to_drop.head(15)
    df_filtered = df_trans.drop(columns=cols_to_drop.index)
    
    # 1. Missing Data Barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_15_dropped.index, y=top_15_dropped.values, palette="Reds_r")
    plt.title('Top 15 Dropped Features by Missing Percentage (>70%)')
    plt.ylabel('Missing Percentage (%)')
    plt.xlabel('Feature Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../Figures/1_missing_data_drops.png')
    plt.close()

    # 2. Fraud Class Imbalance
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_filtered, x='isFraud', palette='Set2')
    plt.title('Log-Scaled Class Imbalance (isFraud)')
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/2_class_imbalance.png')
    plt.close()

    # 3. Transaction Amount Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x='isFraud', y='TransactionAmt', palette='Set3')
    plt.yscale('log')
    plt.title('Transaction Amount by Fraud Status (Log Scale)')
    plt.ylabel('Transaction Amount (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/3_transaction_amt_boxplot.png')
    plt.close()

    # 4. ProductCD Breakdown
    if 'ProductCD' in df_filtered.columns:
        productcd_fraud = df_filtered.groupby(['ProductCD', 'isFraud']).size().unstack(fill_value=0)
        productcd_fraud.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title('Transaction Product Code (ProductCD) by Fraud Status')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../Figures/4_productcd_breakdown.png')
        plt.close()

    # =====================================================================
    # PART 2: ADVANCED HIGH-DIMENSIONAL EDA (For the thesis defense)
    # =====================================================================
    logging.info("Generating Advanced Correlation and Separability plots...")
    
    # Get numeric columns, excluding ID and datetime
    numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in ['TransactionID', 'TransactionDT']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # 5. Target Correlation Tornado Chart
    target_corr = df_filtered[numeric_cols].corr()['isFraud'].drop('isFraud')
    top_pos = target_corr.nlargest(10)
    top_neg = target_corr.nsmallest(10)
    top_features_corr = pd.concat([top_pos, top_neg]).sort_values()
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_features_corr.values]
    top_features_corr.plot(kind='barh', color=colors)
    plt.title('Top 20 Features Correlated with Fraud')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('../Figures/5_target_correlation_tornado.png')
    plt.close()

    # 6. Restricted Spearman Multicollinearity Heatmap
    top_feature_names = top_features_corr.index.tolist()
    spearman_corr_matrix = df_filtered[top_feature_names].corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr_matrix, cmap='coolwarm', center=0, annot=False, square=True, linewidths=0.5)
    plt.title('Spearman Multicollinearity Heatmap (Top 20 Predictive Features)')
    plt.tight_layout()
    plt.savefig('../Figures/6_restricted_multicollinearity_heatmap.png')
    plt.close()

    # 7. KDE Class Separability (TransactionAmt)
    if 'TransactionAmt' in df_filtered.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_filtered[df_filtered['isFraud'] == 0], x='TransactionAmt', log_scale=True, fill=True, label='Legitimate (0)', alpha=0.4)
        sns.kdeplot(data=df_filtered[df_filtered['isFraud'] == 1], x='TransactionAmt', log_scale=True, fill=True, label='Fraudulent (1)', alpha=0.4)
        plt.title('Class Separability: Probability Distribution of Transaction Amount')
        plt.xlabel('Transaction Amount (Log Scale)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../Figures/7_kde_separability.png')
        plt.close()

    logging.info("All EDA graphs generated successfully in the '../Figures' folder.")
    return df_filtered

if __name__ == "__main__":
    # Load dataset (make sure dataset.py contains your Random Undersampling logic)
    df_trans, _ = load_dataset()
    df_clean = perform_comprehensive_eda(df_trans)