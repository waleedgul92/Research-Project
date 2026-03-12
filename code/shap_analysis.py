import logging
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from dataset import load_dataset, process_data
from preprocess import preprocess_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_shap_plots(model_path, X_test):
    logging.info(f"Loading trained XGBoost model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    X_test_sample = X_test.sample(n=2000, random_state=42)
    
    logging.info("Initializing TreeSHAP and calculating values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    logging.info("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig('../Figures/shap_summary_baseline.png')
    plt.close()
    
    logging.info("SHAP summary plot saved to ../Figures/shap_summary_baseline.png")

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, _, _ = process_data(df_trans, df_id)
    
    _, X_test = preprocess_features(X_train_raw, X_test_raw)
    
    generate_shap_plots('../Models/baseline_xgboost.json', X_test)