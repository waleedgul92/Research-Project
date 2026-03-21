import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_membership_inference(train_shap_path, test_shap_path):
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

    print("MIA Accuracy: " + str(round(accuracy * 100, 2)) + "%")
    return accuracy

if __name__ == "__main__":
    train_path = "../Models/shap_federated.npy"
    test_path = "../Models/differential_privacy.npy"
    
    run_membership_inference(train_path, test_path)