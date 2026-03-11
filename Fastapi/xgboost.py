import yaml
import logging
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)

def load_parameters(file_path: str) -> Dict[str, Any]:
    """
    Loads configuration parameters from a YAML file.

    Args:
        file_path (str): The path to the params.yaml file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded parameters.
    """
    logger.info(f"Loading parameters from {file_path}")
    with open(file_path, "r") as file:
        params: Dict[str, Any] = yaml.safe_load(file)
    return params

def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the feature matrix and target vector for the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The feature matrix X and target vector y.
    """
    logger.info("Loading dataset from sklearn")
    dataset = load_breast_cancer()
    X: np.ndarray = dataset.data
    y: np.ndarray = dataset.target
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing subsets.

    Args:
        X (np.ndarray): The full feature matrix.
        y (np.ndarray): The full target vector.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test.
    """
    logger.info(f"Splitting dataset into train and test sets with test_size={test_size}")
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> xgb.XGBClassifier:
    """
    Initializes and trains an XGBoost classifier.

    Args:
        X_train (np.ndarray): The training feature matrix.
        y_train (np.ndarray): The training target vector.
        random_state (int): The seed used by the random number generator.

    Returns:
        xgb.XGBClassifier: The fitted XGBoost model instance.
    """
    logger.info("Initializing and training the XGBoost model")
    model: xgb.XGBClassifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost model training completed")
    return model

def save_xgboost_model(model: xgb.XGBClassifier, output_path: str) -> None:
    """
    Saves the trained XGBoost model to the specified file path.

    Args:
        model (xgb.XGBClassifier): The trained XGBoost model instance.
        output_path (str): The destination file path for the saved model.
    """
    logger.info(f"Saving the trained model to {output_path}")
    model.save_model(output_path)
    logger.info("Model saved successfully")

if __name__ == "__main__":
    params_file: str = "params.yaml"
    model_output_file: str = "model.json"
    
    parameters: Dict[str, Any] = load_parameters(params_file)
    seed: int = parameters.get("random_state", 42)
    test_ratio: float = parameters.get("test_size", 0.2)

    X_full: np.ndarray
    y_full: np.ndarray
    X_full, y_full = load_dataset()

    X_tr: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    X_tr, X_te, y_tr, y_te = split_data(X_full, y_full, test_size=test_ratio, random_state=seed)

    trained_model: xgb.XGBClassifier = train_xgboost(X_tr, y_tr, random_state=seed)

    save_xgboost_model(trained_model, model_output_file)