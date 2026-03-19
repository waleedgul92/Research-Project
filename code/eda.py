import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
import altair as alt
import warnings
from dataset import load_dataset

warnings.simplefilter("ignore")
plt.style.use('ggplot')

def setup_directories():
    os.makedirs('../Figures', exist_ok=True)

def plot_missing_data(df):
    data_null = (df.isnull().sum() / len(df)) * 100
    data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:50]
    
    plt.figure(figsize=(20, 8))
    sns.barplot(x=data_null.index, y=data_null.values, palette="Reds_r")
    plt.xticks(rotation=90)
    plt.title('Top 50 Features by Missing Data Percentage', fontsize=18)
    plt.ylabel('Missing Rate (%)', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('../Figures/1_missing_data_rates.png')
    plt.close()

def plot_train_test_time_split(train, test):
    plt.figure(figsize=(12, 6))
    plt.hist(train['TransactionDT'], bins=50, label='Train', alpha=0.7, color='blue')
    plt.hist(test['TransactionDT'], bins=50, label='Test', alpha=0.7, color='orange')
    plt.legend()
    plt.title('Transaction Dates: Train vs Test Chronological Split')
    plt.xlabel('TransactionDT (Time Delta)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('../Figures/2_time_split.png')
    plt.close()

def plot_class_imbalance(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='isFraud', palette='Set2')
    plt.title('Fraud Class Imbalance (Log Scale)')
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/3_class_imbalance.png')
    plt.close()

def plot_transaction_amt_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='isFraud', y='TransactionAmt', palette='Set3')
    plt.yscale('log')
    plt.title('Transaction Amount by Fraud Status (Log Scale)')
    plt.ylabel('Transaction Amount (Log Scale)')
    plt.tight_layout()
    plt.savefig('../Figures/4_transaction_amt_boxplot.png')
    plt.close()

def plot_categorical_stacked_bar(df, col_name='ProductCD'):
    if col_name in df.columns:
        cat_fraud = df.groupby([col_name, 'isFraud']).size().unstack(fill_value=0)
        cat_fraud.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title(f'Transaction Count by {col_name} and Fraud Status')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'../Figures/5_{col_name}_vs_fraud.png')
        plt.close()

def plot_altair_high_cardinality(df, col_name, top_n=40):
    feature_count = df[col_name].value_counts(dropna=False)[:top_n].reset_index()
    feature_count.columns = [col_name, 'count']
    
    chart = alt.Chart(feature_count).mark_bar().encode(
        x=alt.X(f"{col_name}:N", sort='-y', title=col_name),
        y=alt.Y('count:Q', title='Count'),
        tooltip=[col_name, 'count']
    ).properties(title=f"Top {top_n} Counts of {col_name}", width=800, height=400)
    
    return chart

def plot_dimensionality_reduction_clusters(X_train, y_train, n_components=2, sample_size=10000):
    print(f"Running Dimensionality Reduction on a sample of {sample_size} rows...")
    
    X_sample = X_train.sample(n=sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]
    
    X_sample_filled = X_sample.fillna(-999)

    X_tsne = TSNE(n_components=n_components, random_state=42).fit_transform(X_sample_filled.values)
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X_sample_filled.values)
    X_svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42).fit_transform(X_sample_filled.values)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    f.suptitle('Clusters using Dimensionality Reduction', fontsize=18)
    
    blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
    red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

    ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y_sample == 0), cmap='coolwarm', label='No Fraud', alpha=0.5)
    ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y_sample == 1), cmap='coolwarm', label='Fraud', alpha=0.5)
    ax1.set_title('t-SNE', fontsize=14)
    ax1.grid(True)
    ax1.legend(handles=[blue_patch, red_patch])

    ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y_sample == 0), cmap='coolwarm', label='No Fraud', alpha=0.5)
    ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y_sample == 1), cmap='coolwarm', label='Fraud', alpha=0.5)
    ax2.set_title('PCA', fontsize=14)
    ax2.grid(True)
    ax2.legend(handles=[blue_patch, red_patch])

    ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y_sample == 0), cmap='coolwarm', label='No Fraud', alpha=0.5)
    ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y_sample == 1), cmap='coolwarm', label='Fraud', alpha=0.5)
    ax3.set_title('Truncated SVD', fontsize=14)
    ax3.grid(True)
    ax3.legend(handles=[blue_patch, red_patch])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('../Figures/6_dimensionality_reduction_clusters.png')
    plt.close()
    print("Dimensionality reduction plots saved.")

if __name__ == "__main__":
    setup_directories()
    df_trans, df_id = load_dataset()
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    train, test = train_test_split(df, test_size=0.2, stratify=df['isFraud'], random_state=42)
    
    plot_missing_data(train)
    plot_train_test_time_split(train, test)
    plot_class_imbalance(train)
    plot_transaction_amt_boxplot(train)
    plot_categorical_stacked_bar(train, col_name='ProductCD')
    
    X_train = train.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])
    X_train_numeric = X_train.select_dtypes(include=['int64', 'float64'])
    y_train = train['isFraud']
    plot_dimensionality_reduction_clusters(X_train_numeric, y_train)