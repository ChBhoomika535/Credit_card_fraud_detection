# 💳 Credit Card Fraud Detection using Random Forest

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using **machine learning** techniques.  
Due to the highly **imbalanced nature** of fraud data, traditional accuracy metrics are misleading, so this project emphasizes **precision, recall, and F1-score**.

The task demonstrates:
- Handling imbalanced datasets
- Applying ensemble learning (Random Forest)
- Comparing baseline vs advanced models
- Model evaluation and feature importance analysis

---

## 🗂 Dataset
**Source:** Kaggle – Credit Card Fraud Detection  
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

**Details:**
- Transactions: 284,807  
- Fraud cases: 492  
- Features: PCA-transformed (`V1`–`V28`), `Time`, `Amount`  
- Target:
  - `0` → Non-Fraud  
  - `1` → Fraud  

---

## 🛠 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

---

## 🧪 Approach
1. Loaded dataset and analyzed class imbalance
2. Separated features and target variable
3. Used **stratified train-test split** to preserve fraud ratio
4. Trained **Logistic Regression** as a baseline model
5. Trained **Random Forest Classifier**
6. Evaluated models using:
   - Precision
   - Recall
   - F1-score
7. Visualized **feature importances**
8. Saved trained model for reuse

---

## 📊 Model Evaluation
Accuracy is avoided due to heavy class imbalance.  
Key evaluation metrics used:
- Precision
- Recall
- F1-score
- Confusion Matrix

**Random Forest outperformed Logistic Regression**, especially in recall for fraud detection.

---

## 📈 Feature Importance
Random Forest provides feature importance scores, helping identify key variables that contribute most to fraud detection.

A bar chart of the **top 15 important features** is included in the notebook.

---

## 💾 Saved Models
- `random_forest_fraud_model.pkl` – Trained Random Forest model
- `scaler.pkl` – StandardScaler used for feature scaling


