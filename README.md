# Breast Cancer Tumor Classification Using Machine Learning

## Overview  
This project focuses on optimizing binary classification of breast tumors (malignant vs. benign) using machine learning techniques on the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. It explores and compares Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost algorithms. Key enhancements include handling class imbalance with SMOTE, feature selection with Recursive Feature Elimination (RFE), and threshold tuning for improved precision-recall balance. The best-performing model, XGBoost, is deployed as an interactive web application using Streamlit.

---

## Features

- **Data preprocessing:** Standardization of features and label encoding.
- **Feature selection:** Recursive Feature Elimination (RFE) to reduce dimensionality and remove redundant features.
- **Imbalance handling:** Synthetic Minority Oversampling Technique (SMOTE) to balance malignant and benign classes.
- **Classification models:** Logistic Regression, SVM (with RBF kernel), Random Forest, and XGBoost.
- **Threshold tuning:** Custom classification thresholds for optimized clinical relevance.
- **Evaluation:** Metrics including accuracy, precision, recall, and F1-score.
- **Deployment:** Interactive Streamlit app for real-time tumor classification based on user input.

---

## Dataset

- **Source:** Wisconsin Breast Cancer Diagnostic (WBCD) dataset from the UCI Machine Learning Repository.
- **Samples:** 569 instances.
- **Features:** 30 numerical features describing cell nuclei characteristics.
- **Target:** Binary classification — Malignant (1) or Benign (0).
- **Imbalance:** Benign cases significantly outnumber malignant cases.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
