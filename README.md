# Breast Cancer Tumor Classification Using Machine Learning

## 🧠 Overview
This project focuses on optimizing binary classification of breast tumors (malignant vs. benign) using machine learning techniques on the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. It explores and compares Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost algorithms.

Key enhancements include:

- Handling class imbalance with SMOTE
- Feature selection with Recursive Feature Elimination (RFE)
- Threshold tuning for improved precision-recall balance
- The best-performing model (XGBoost) is deployed as an interactive web application using Streamlit.

  
## 🚀 Features
- 📊 **Data Preprocessing:** Standardization and label encoding
- 🔍 **Feature Selection:** RFE to reduce dimensionality
- ⚖️ **Imbalance Handling:** SMOTE for oversampling minority class
- 🤖 **Classification Models:** Logistic Regression, SVM (RBF), Random Forest, XGBoost
- 🎯 **Threshold Tuning:** For clinical relevance and balance
- 📈 **Evaluation:** Accuracy, Precision, Recall, F1-Score
- 🌐 **Deployment:** Real-time predictions with Streamlit app


## 📂 Dataset
- **Source:** UCI Machine Learning Repository – WBCD  
- **Samples:** 569  
- **Features:** 30 numerical features describing cell nuclei characteristics  
- **Target:** Binary classification — Malignant (1) or Benign (0)  
- **Note:** The dataset is imbalanced, with more benign cases.


## 💻 Installation
```bash
git clone https://github.com/GARVS0205/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt


🚀 Usage
To run the Streamlit app locally, execute:
streamlit run app.py
This will launch the web app in your default browser where you can input data and get tumor classification predictions in real time.


📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.


📞 Contact
If you have any questions or want to collaborate, feel free to reach out:
GitHub: https://github.com/GARVS0205
Email: garvsachdeva@gmail.com


Thank you for checking out my project! 😊



