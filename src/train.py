import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow

# Load dataset
def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train and evaluate model
def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    mlflow.log_metric("accuracy", accuracy)
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    mlflow.set_experiment("Iris Model Training")
    with mlflow.start_run():
        train_and_evaluate()
