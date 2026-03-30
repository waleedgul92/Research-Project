import os
import logging
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from data_pipeline.dataset import load_dataset, process_data, load_config
from data_pipeline.preprocess import preprocess_features
from sklearn.metrics import roc_auc_score
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_TARGET = 0.95
TEST_TARGET  = 0.75
MIN_GAP      = 0.15

def train_xgboost(X_train, y_train, X_test=None, y_test=None,n_trials=50):
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    def objective(trial):
        params = {
            'max_depth':        trial.suggest_int  ('max_depth',        2,    8),
            'n_estimators':     trial.suggest_int  ('n_estimators',     50,   400),
            'learning_rate':    trial.suggest_float('learning_rate',    0.05, 0.5),
            'subsample':        trial.suggest_float('subsample',        0.2,  0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2,  0.9),
            'reg_lambda':       trial.suggest_float('reg_lambda',       0.0,  2.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',        0.0,  1.0),
        }

        clf = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_weight,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        test_auc  = roc_auc_score(y_test,  clf.predict_proba(X_test)[:, 1])
        gap       = train_auc - test_auc

        train_penalty = abs(train_auc - TRAIN_TARGET)
        test_penalty  = abs(test_auc  - TEST_TARGET)
        gap_penalty   = max(0, MIN_GAP - gap)

        loss = train_penalty + test_penalty + gap_penalty

        logging.info(
            f"Trial {trial.number:03d} | Train: {train_auc:.4f} | Test: {test_auc:.4f} | "
            f"Gap: {gap:.4f} | Loss: {loss:.4f} | "
            f"depth={params['max_depth']} n={params['n_estimators']} "
            f"lr={params['learning_rate']:.3f} sub={params['subsample']:.2f} "
            f"col={params['colsample_bytree']:.2f} lam={params['reg_lambda']:.2f}"
        )
        return loss
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logging.info("=== BEST PARAMETERS FOUND ===")
    for k, v in best.items():
        logging.info(f"  {k}: {v}")
    logging.info(f"  Best loss: {study.best_value:.4f}")

    return study

if __name__ == "__main__":
    df_trans, df_id = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = process_data(df_trans, df_id)
    
    X_train, X_test = preprocess_features(X_train_raw, X_test_raw, y_train)
    
    final_model = train_xgboost(X_train, y_train,X_test, y_test)