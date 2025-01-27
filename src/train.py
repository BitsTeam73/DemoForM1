from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


def train_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

     # Save the model
    os.makedirs("models", exist_ok=True)  # Ensure the directory exists
    joblib.dump(clf, "models/model.pkl")


if __name__ == "__main__":
    train_model()
