import os
import logging
import numpy as np
import xgboost as xgb
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

    train_norms = np.linalg.norm(train_shap, axis=1, keepdims=True)
    test_norms = np.linalg.norm(test_shap, axis=1, keepdims=True)
    
    train_var = np.var(train_shap, axis=1, keepdims=True)
    test_var = np.var(test_shap, axis=1, keepdims=True)
    
    train_max = np.max(np.abs(train_shap), axis=1, keepdims=True)
    test_max = np.max(np.abs(test_shap), axis=1, keepdims=True)

    train_features = np.hstack((train_shap, train_norms, train_var, train_max))
    test_features = np.hstack((test_shap, test_norms, test_var, test_max))

    train_labels = np.ones(train_features.shape[0])
    test_labels = np.zeros(test_features.shape[0])

    X = np.vstack((train_features, test_features))
    y = np.concatenate((train_labels, test_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    attack_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
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