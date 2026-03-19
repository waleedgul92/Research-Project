import numpy as np

def save_shap_matrix(shap_values, scenario_name):
    file_path = f'../Models/shap_{scenario_name}.npy'
    np.save(file_path, shap_values)

def load_shap_matrix(scenario_name):
    file_path = f'../Models/shap_{scenario_name}.npy'
    shap_values = np.load(file_path)
    return shap_values