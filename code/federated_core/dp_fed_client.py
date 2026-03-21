import numpy as np
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dp_shap_values(model, X_sample, epsilon, percentile=95):
    explainer = shap.TreeExplainer(model)
    exact_shap = explainer.shap_values(X_sample)
    
    dynamic_threshold = np.percentile(np.abs(exact_shap), percentile)
    
    clipped_shap = np.clip(exact_shap, -dynamic_threshold, dynamic_threshold)
    
    sensitivity = 2 * dynamic_threshold
    scale = sensitivity / epsilon
    
    noise = np.random.laplace(loc=0.0, scale=scale, size=clipped_shap.shape)
    noisy_shap = clipped_shap + noise
    
    return noisy_shap