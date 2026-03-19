import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

def calculate_metrics(base_shap, comp_shap, features, top_k=10):
    imp_base = np.abs(base_shap).mean(axis=0)
    imp_comp = np.abs(comp_shap).mean(axis=0)

    df_base = pd.DataFrame({'feature': features, 'importance': imp_base}).sort_values(by='importance', ascending=False).reset_index(drop=True)
    df_comp = pd.DataFrame({'feature': features, 'importance': imp_comp}).sort_values(by='importance', ascending=False).reset_index(drop=True)

    top_base_set = set(df_base['feature'].head(top_k))
    top_comp_set = set(df_comp['feature'].head(top_k))
    overlap = (len(top_base_set.intersection(top_comp_set)) / top_k) * 100

    df_comp_aligned = df_comp.set_index('feature').loc[df_base['feature']].reset_index()
    rho, _ = spearmanr(df_base.index.values, df_comp_aligned.index.values)

    mae = mean_absolute_error(base_shap.flatten(), comp_shap.flatten())

    return overlap, rho, mae, df_base, df_comp

def run_master_comparison():
    features = [f"Feature_{i}" for i in range(40)]
    
    shap_base = np.load('../Models/shap_baseline.npy')
    shap_fl = np.load('../Models/shap_federated.npy')
    shap_dp = np.load('../Models/shap_differential_privacy.npy')

    overlap_fl, rho_fl, mae_fl, df_base, df_fl = calculate_metrics(shap_base, shap_fl, features)
    
    overlap_dp, rho_dp, mae_dp, _, df_dp = calculate_metrics(shap_base, shap_dp, features)

    print("==================================================")
    print("      MASTER XAI DEGRADATION RESULTS              ")
    print("==================================================")
    print("[1] Federated Learning (No Noise)")
    print(f"    Top-10 Overlap:   {overlap_fl:.1f}%")
    print(f"    Spearman Rank:    {rho_fl:.4f}")
    print(f"    SHAP Matrix MAE:  {mae_fl:.6f}\n")
    
    print("[2] FL + Differential Privacy (Laplacian Noise)")
    print(f"    Top-10 Overlap:   {overlap_dp:.1f}%")
    print(f"    Spearman Rank:    {rho_dp:.4f}")
    print(f"    SHAP Matrix MAE:  {mae_dp:.6f}")
    print("==================================================")

    top_10_features = df_base['feature'].head(10).tolist()
    
    base_vals = df_base.set_index('feature').loc[top_10_features]['importance'].values
    fl_vals = df_fl.set_index('feature').loc[top_10_features]['importance'].values
    dp_vals = df_dp.set_index('feature').loc[top_10_features]['importance'].values

    x = np.arange(len(top_10_features))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, base_vals, width, label='Centralized Baseline', color='#1f77b4')
    plt.bar(x, fl_vals, width, label='Federated Ensemble', color='#ff7f0e')
    plt.bar(x + width, dp_vals, width, label='Differential Privacy (DP)', color='#2ca02c')

    plt.xlabel('Top 10 Features (Baseline Order)', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value (Impact)', fontsize=12)
    plt.title('Explainability Degradation Across Architectures', fontsize=16)
    plt.xticks(x, top_10_features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Figures/7_xai_degradation_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_master_comparison()