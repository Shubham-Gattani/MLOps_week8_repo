import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Load the Iris dataset
iris = pd.read_csv("./iris.csv")
X, y = iris.drop(columns=["species"]), iris["species"]

mlflow.set_tracking_uri("http://127.0.0.1:8100")

# Setting experiment name for mlflow

mlflow.set_experiment("IRIS Data Poisoning test1")

# Function to poison a fraction of the dataset
def poison_data(X, poison_fraction):
    X_poisoned = X.copy()
    n_samples = int(len(X) * poison_fraction)
    poisoned_indices = np.random.choice(len(X), n_samples, replace=False)
    
    # Create random noise
    random_noise = np.random.uniform(
        X.values.min(), X.values.max(), (n_samples, X.shape[1])
    )
    
    # Assign noise using iloc
    X_poisoned.iloc[poisoned_indices] = random_noise
    return X_poisoned
"""
Example for one poisoned row it may generate:
[6.9, 1.2, 4.7, 0.33]

For another poisoned row:
[2.1, 5.6, 0.9, 3.4]
"""


# Function to train, evaluate and log with MLflow
def train_and_log(poison_fraction):
    X_poisoned = poison_data(X, poison_fraction)
    X_train, X_test, y_train, y_test = train_test_split(
        X_poisoned, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    signature = infer_signature(X_train, clf.predict(X_train))

    # Start MLflow run
    with mlflow.start_run(run_name=f"Iris_Poison_{int(poison_fraction*100)}"):
        mlflow.log_param("poison_fraction", poison_fraction)
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"]
        })
        mlflow.sklearn.log_model(
            sk_model = clf, 
            name = "w8_model",
            registered_model_name = "poisoned-IRIS-Classifier-week-8",
            signature = signature,
        )
        print(f"\n=== POISON LEVEL: {int(poison_fraction*100)}% ===")
        print(classification_report(y_test, y_pred))

# Define poison levels
poison_levels = [0.0, 0.05, 0.10, 0.50]

# Run for all levels
if __name__=="__main__"
    for level in poison_levels:
        train_and_log(level)
