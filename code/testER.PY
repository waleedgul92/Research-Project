import os
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation.membership_inference import run_membership_inference
from evaluation.compare_all_xai import calculate_metrics

MODELS_DIR = '../Models'

def apply_static_dp(exact_shap, epsilon, clip_threshold):
    clipped_shap = np.clip(exact_shap, -clip_threshold, clip_threshold)
    sensitivity = 2 * clip_threshold
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=clipped_shap.shape)
    return clipped_shap + noise

def objective(trial):
    epsilon = trial.suggest_float("dp_epsilon", 0.1, 3.0)
    clip_threshold = trial.suggest_float("dp_clip_threshold", 0.1, 2.0)

    shap_fl_train = np.load(os.path.join(MODELS_DIR, "train_shap_federated.npy"))
    shap_fl_test = np.load(os.path.join(MODELS_DIR, "shap_federated.npy"))
    shap_baseline = np.load(os.path.join(MODELS_DIR, "shap_baseline.npy"))
    
    shap_dp_train = apply_static_dp(shap_fl_train, epsilon, clip_threshold)
    shap_dp_test = apply_static_dp(shap_fl_test, epsilon, clip_threshold)

    temp_train_path = os.path.join(MODELS_DIR, "optuna_train_static.npy")
    temp_test_path = os.path.join(MODELS_DIR, "optuna_test_static.npy")
    
    np.save(temp_train_path, shap_dp_train)
    np.save(temp_test_path, shap_dp_test)

    mia_acc = run_membership_inference(
        f"Trial_{trial.number}",
        temp_train_path,
        temp_test_path
    )

    features = [f"Feature_{i}" for i in range(40)]
    overlap, _, _, _, _, _ = calculate_metrics(shap_baseline, shap_dp_test, features)

    privacy_objective = abs(mia_acc - 50.0)
    utility_objective = 100.0 - overlap

    return privacy_objective, utility_objective

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=50)

    print("\n=== STATIC PARETO FRONT: BEST COMPROMISES ===")
    for trial in study.best_trials:
        p = trial.params
        print(f"Epsilon: {p['dp_epsilon']:.2f} | Threshold: {p['dp_clip_threshold']:.2f}")
        print(f"  -> Privacy Gap: {trial.values[0]:.2f}% | Overlap: {100 - trial.values[1]:.1f}%\n")