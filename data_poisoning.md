# ✅ **1. How to Mitigate Data Poisoning Attacks**

Poisoning attacks manipulate training data so the model learns incorrect patterns. Defenses fall into four categories:

---

## **A. Prevention (Stop poisoned data from entering the pipeline)**

### **1. Data provenance / logging**

Track where data came from:

* Source identity
* Collection timestamp
* Versioning (e.g., DVC, Delta Lake)
* Audit logs

This makes malicious inserts detectable and reversible.

### **2. Access control**

Limit who can modify data folders, buckets, or pipelines.
Enforce:

* Read-only datasets
* RBAC for dataset owners
* Code reviews for data ingestion changes

### **3. Automated schema & statistical checks**

Before training, enforce rules such as:

* Feature ranges
* Distribution checks
* Missing value limits
* Duplicate detection

These catch random-noise or out-of-range poisoning attempts early.

---

## **B. Detection (Find poisoned records before training)**

### **1. Outlier detection**

Use models like:

* Isolation Forest
* Local Outlier Factor (LOF)
* PCA reconstruction error

Poisoned rows (filled with noise) stand out strongly.

### **2. Influence functions (impact analysis)**

Detect data points that strongly change the model's predictions.

### **3. Clustering anomalies**

Cluster your dataset. Points far from all clusters often represent poisoned samples.

### **4. Train-two-models technique**

Train 1 model on the full dataset, and another on a cleaned/filtered dataset → compare predictions.
Big divergence may indicate poisoning.

---

## **C. Mitigation During Training**

### **1. Robust loss functions**

Use losses that are less sensitive to mislabeled or corrupted inputs:

* Huber loss
* Label-smoothing
* Trimmed loss (ignore the top X% highest-loss samples)

### **2. Model ensembling**

Poisoned data affects one model more than an ensemble.
Averaging multiple models reduces impact.

### **3. Adversarial training**

Train the model to be tolerant to input perturbations.

---

## **D. Post-training Monitoring**

### **1. Drift detection**

Poisoning often creates unexpected shifts in:

* feature distributions
* error distributions
* prediction entropy

### **2. Canary deployments**

Deploy to a small subset first → if performance drops, rollback.

---

# ✅ **2. How Data Quantity Requirements Evolve When Data Quality Drops**

Think of ML training as a signal-vs-noise problem.

* **Signal = clean data**
* **Noise = poisoned/corrupted data**

When noise increases, you need *more overall data* to recover the original amount of signal.

---

## **A. Effective dataset size = clean_data_fraction × total_size**

Example:
If poisoning fraction = 20% and you have 1000 rows:

Clean rows = 800 ← model learns only from these

To get the same “learning power” as 1000 clean rows…

You need:

```
Required_data = 1000 / 0.8 = 1250 rows
```

Meaning: **you must add 250 more rows** to compensate.

As poisoning increases, this gets worse:

| Poisoned % | Clean Fraction | You Keep | You Lose | Required Extra Data |
| ---------- | -------------- | -------- | -------- | ------------------- |
| 5%         | 0.95           | 95%      | 5%       | +5.2%               |
| 10%        | 0.90           | 90%      | 10%      | +11.1%              |
| 20%        | 0.80           | 80%      | 20%      | +25%                |
| 50%        | 0.50           | 50%      | 50%      | +100%               |

If half your dataset is poisoned, you need **double the size** to regain the same clean-data volume.

---

## **B. Model sample complexity increases**

Low-quality data increases:

* variance
* overfitting risk
* instability in decision boundaries

To stabilize training, you need:

* **more samples**, and
* **more regularization**

---

## **C. Bigger models are more sensitive to noise**

Large models (RF, NN, LLM-fine-tuning) need more clean data.
When dataset is partially poisoned, the learning curve flattens faster → diminishing returns.

---

# 🎯 **Simple Summary**

### **How poisoning is mitigated**

* Clean the data (detect/remove outliers)
* Control ingestion (logging, validation, permissions)
* Use robust training techniques
* Continuously monitor the model

### **How data quantity evolves**

If X% of your data is poisoned, your effective dataset shrinks to (1−X).
To recover performance, you must increase total data by **1 / (1−X)**.
