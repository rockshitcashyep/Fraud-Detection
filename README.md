# 💳 Credit Card Fraud Detection

A comprehensive Machine Learning pipeline designed to detect fraudulent transactions in a highly imbalanced dataset (0.17% fraud rate). This project contrasts **Supervised Learning** (with SMOTE) against **Unsupervised Anomaly Detection** techniques.

## 📌 Project Overview
The goal of this project is to identify fraudulent credit card transactions. Because fraud is extremely rare, standard accuracy metrics are misleading (a model predicting "No Fraud" for every case achieves 99.8% accuracy but fails its purpose).

This pipeline solves the class imbalance problem using three distinct approaches:
1.  **Baseline:** Logistic Regression (Standard).
2.  **Supervised (Balanced):** Random Forest Classifier + SMOTE (Synthetic Minority Over-sampling Technique).
3.  **Unsupervised:** Isolation Forest (Anomaly Detection).

## 🛠️ Tech Stack
* **Python 3.x**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Modeling & Evaluation)
* **Imbalanced-Learn** (SMOTE)
* **Seaborn & Matplotlib** (Visualization)

## 📊 Key Techniques & Methodology

### 1. Data Preprocessing
* **Robust Scaling:** Used `RobustScaler` instead of Standard Scaler to reduce the impact of extreme outliers in the `Amount` feature.
* **Stratified Splitting:** Maintained the 0.17% fraud ratio in both Training and Test sets to ensure realistic evaluation.

### 2. Handling Imbalance (Supervised Approach)
* **Problem:** The original dataset had ~280,000 legitimate transactions and only ~490 fraud cases.
* **Solution:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the **Training Data only**.
    * *Logic:* Synthesized new fraud examples by interpolating between existing ones.
    * *Result:* Trained a **Random Forest Classifier** on a balanced 50/50 dataset.

### 3. Anomaly Detection (Unsupervised Approach)
* **Algorithm:** **Isolation Forest**.
* **Logic:** Instead of learning "what is fraud," the model learns "what is normal." It isolates observations by randomly selecting a feature and randomly selecting a split value. Anomalies (fraud) are isolated faster (fewer splits) than normal observations.
* **Metric:** Evaluated using the **Precision-Recall Curve** rather than ROC, as ROC can be overly optimistic for imbalanced data.

## 📈 Results Summary
| Model | Approach | Key Observation |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline | High Accuracy, but poor Recall (missed many fraud cases). |
| **Random Forest + SMOTE** | Supervised | **Best Performance.** High Recall and improved Precision. Successfully modeled the minority class. |
| **Isolation Forest** | Unsupervised | Lower F1 Score (expected), but demonstrated utility in detecting fraud without requiring labeled training data. |


## ⚠️ Dataset Note
The dataset (`creditcard.csv`) is not included in this repo due to size limits (~150MB). It must be downloaded locally.