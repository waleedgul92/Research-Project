import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..' , '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_membership_inference(scenario_name, train_shap_path, test_shap_path):
    logging.info(f"--- Attacking {scenario_name} ---")
    
    train_shap = np.load(train_shap_path)
    test_shap = np.load(test_shap_path)

    train_labels = np.ones(train_shap.shape[0])
    test_labels = np.zeros(test_shap.shape[0])

    X = np.vstack((train_shap, test_shap))
    y = np.concatenate((train_labels, test_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
    attack_model.fit(X_train, y_train)

    predictions = attack_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f">>> {scenario_name} MIA Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy

if __name__ == "__main__":
    run_membership_inference(
        "Centralized Baseline",
        os.path.join(MODELS_DIR, "train_shap_baseline.npy"),
        os.path.join(MODELS_DIR, "shap_baseline.npy")
    )
    
    run_membership_inference(
        "Standard Federated Learning",
        os.path.join(MODELS_DIR, "train_shap_federated.npy"),
        os.path.join(MODELS_DIR, "shap_federated.npy")
    )
    
    run_membership_inference(
        "Differential Privacy FL",
        os.path.join(MODELS_DIR, "train_shap_differential_privacy.npy"),
        os.path.join(MODELS_DIR, "shap_differential_privacy.npy")
    )