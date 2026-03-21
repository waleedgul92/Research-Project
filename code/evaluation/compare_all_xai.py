import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

def calculate_metrics(base_shap, comp_shap, features, top_k=10):
    # Calculate mean absolute importance
    imp_base = np.abs(base_shap).mean(axis=0)
    imp_comp = np.abs(comp_shap).mean(axis=0)

    # Map features to their importance scores
    base_dict = {f: imp_base[i] for i, f in enumerate(features)}
    comp_dict = {f: imp_comp[i] for i, f in enumerate(features)}

    # Sort features from most important to least important
    base_sorted = sorted(features, key=lambda x: base_dict[x], reverse=True)
    comp_sorted = sorted(features, key=lambda x: comp_dict[x], reverse=True)

    # 1. Top-K Overlap
    top_base = set(base_sorted[:top_k])
    top_comp = set(comp_sorted[:top_k])
    overlap = (len(top_base.intersection(top_comp)) / top_k) * 100.0

    # 2. Spearman Rank Correlation (Bug-Free)
    # We find the exact rank (0 to 39) of each feature in both lists
    rank_base = [base_sorted.index(f) for f in features]
    rank_comp = [comp_sorted.index(f) for f in features]
    rho, _ = spearmanr(rank_base, rank_comp)

    # 3. SHAP Matrix MAE
    mae = mean_absolute_error(base_shap.flatten(), comp_shap.flatten())

    return overlap, rho, mae, base_sorted, base_dict, comp_dict

def run_master_comparison():
    # Placeholder names for your 40 features
    features = [f"Feature_{i}" for i in range(40)]
    
    # Load matrices
    shap_base = np.load('../Models/shap_baseline.npy')
    shap_fl = np.load('../Models/shap_federated.npy')
    shap_dp = np.load('../Models/shap_differential_privacy.npy')

    # Calculate metrics
    overlap_fl, rho_fl, mae_fl, base_sorted, base_dict, fl_dict = calculate_metrics(shap_base, shap_fl, features)
    overlap_dp, rho_dp, mae_dp, _, _, dp_dict = calculate_metrics(shap_base, shap_dp, features)

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

    # --- Plotting the Bar Chart ---
    top_10_features = base_sorted[:10]
    
    base_vals = [base_dict[f] for f in top_10_features]
    fl_vals = [fl_dict[f] for f in top_10_features]
    dp_vals = [dp_dict[f] for f in top_10_features]

    x = np.arange(len(top_10_features))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, base_vals, width, label='Centralized Baseline', color='#1f77b4')
    plt.bar(x, fl_vals, width, label='Federated Ensemble (No Noise)', color='#ff7f0e')
    plt.bar(x + width, dp_vals, width, label='FL + Differential Privacy', color='#2ca02c')

    plt.xlabel('Top 10 Features (Baseline Rank Order)', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value (Impact)', fontsize=12)
    plt.title('Explainability Degradation Across Privacy Architectures', fontsize=16)
    plt.xticks(x, top_10_features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Figures/7_xai_degradation_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_master_comparison()