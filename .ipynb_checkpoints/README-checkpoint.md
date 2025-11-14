# 📌 **Code Description: Iris Data Poisoning Experiment with MLflow Tracking**

This project demonstrates how data poisoning affects machine-learning model performance using the Iris dataset. The script simulates controlled data corruption, trains a classifier on both clean and poisoned data, and logs all results to MLflow for comparison and analysis.

---

## ✅ **1. Dataset Loading**

The script loads the Iris dataset from a local CSV file and separates it into:

* **Features (X)** → sepal length, sepal width, petal length, petal width
* **Labels (y)** → species name

MLflow is configured to use a local tracking server at:

```
http://127.0.0.1:8100
```

---

## ✅ **2. Experiment Setup**

An MLflow experiment called **“IRIS Data Poisoning test1”** is created.
Each poisoning level will be logged as a separate MLflow run.

---

## ✅ **3. Data Poisoning Mechanism**

A custom function `poison_data()` corrupts a chosen percentage of rows in the dataset.
Steps:

1. Randomly select a fraction of samples based on `poison_fraction`
2. Generate random noise values within the global min/max range of the original dataset
3. Replace all feature values in those selected rows with random noise
4. Labels are *not* changed, creating mismatched and misleading training examples

This simulates a realistic poisoning attack.

---

## ✅ **4. Training and Evaluation Workflow**

The function `train_and_log()` performs the following:

### **A. Poison the dataset**

* Apply the selected poisoning level
* Split into training and test sets (80/20)

### **B. Train the model**

* A **DecisionTreeClassifier** (depth=3, random_state=1) is trained
* Predictions are generated on the test set

### **C. Evaluate performance**

A classification report is computed, including:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score

### **D. Log everything to MLflow**

MLflow logs:

* Poison fraction (param)
* Accuracy and other metrics
* The trained model (saved + registered)
* Model signature inferred from the training data

The model is logged under:

```
registered_model_name = "poisoned-IRIS-Classifier-week-8th"
```

Each run is named based on the poisoning percentage, e.g., `Iris_Poison_5`.

---

## ✅ **5. Multiple Poisoning Levels**

The experiment automatically tests multiple poisoning percentages:

```
poison_levels = [0.0, 0.05, 0.10, 0.50]
```

For each level:

* The dataset is partially corrupted
* The model is trained
* Metrics are logged
* A classification report is printed

This allows you to compare how performance degrades as data quality drops.

---

## 📊 **6. Purpose of This Experiment**

This code helps you:

* Understand how sensitive ML models are to corrupted data
* Visualize the drop in performance as poisoning increases
* Track and compare runs using MLflow
* Test robustness of ML pipelines
* Demonstrate the impact of adversarial data attacks