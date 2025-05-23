import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Set Streamlit page config
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# ✅ Get current directory and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "breast_cancer_model.pkl")
features_path = os.path.join(current_dir, "selected_features.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")
dataset_path = os.path.join(current_dir, "Cancer Dataset.csv")

# ✅ Check for required files
if not all(os.path.exists(path) for path in [model_path, features_path, scaler_path, dataset_path]):
    st.error("❌ Missing model, feature list, scaler or dataset file.")
    st.stop()

# ✅ Load model, features, scaler, dataset
model = joblib.load(model_path)
selected_features = joblib.load(features_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(dataset_path)

# ✅ Simulate test set for random sampling
_, df_test = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# 🧠 App Title
st.title("🩺 Breast Cancer Prediction App")

st.write("""
### Enter patient details below:
Choose whether you want to enter values *manually* or *select from the dataset*. You can also auto-fill a random sample.
""")

# ✅ Global options
input_mode = st.radio("Choose input method for all features:", ["Manual Entry", "Select from Dataset"], horizontal=True)
use_random = st.checkbox("🔁 Auto-fill with random test sample")

# ✅ Get random row if selected
if use_random:
    random_row = df_test[selected_features].sample(n=1, random_state=np.random.randint(0, 1000)).iloc[0]
    st.info("✅ Random sample loaded! You can still edit the values.")
else:
    random_row = pd.Series([None] * len(selected_features), index=selected_features)

# 🔢 Input fields
user_inputs = []
st.markdown("### 🧾 Input Feature Values")

for feature in selected_features:
    default_val = (
        float(random_row[feature]) if use_random and not pd.isnull(random_row[feature]) else 0.0
    )

    if input_mode == "Manual Entry":
        value = st.number_input(
            f"{feature}",
            value=default_val,
            step=0.01,
            format="%.4f"
        )
    else:  # Select from dataset
        options = sorted(df[feature].dropna().unique())
        if default_val not in options:
            default_val = options[0]
        value = st.selectbox(
            f"{feature}",
            options,
            index=options.index(default_val),
            key=f"select_{feature}"
        )

    user_inputs.append(value)

# ✅ Convert and scale input
user_inputs = np.array(user_inputs).reshape(1, -1)
user_inputs_scaled = scaler.transform(user_inputs)

# ✅ Threshold for classification
threshold = 0.4

# ✅ Predict
if st.button("🔍 Predict"):
    probability = model.predict_proba(user_inputs_scaled)[:, 1][0]
    prediction = int(probability >= threshold)

    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error("🔴 The tumor is *Malignant (Cancerous)*.")
    else:
        st.success("🟢 The tumor is *Benign (Non-Cancerous)*.")


# Footer
st.write("---")
st.markdown("💡 This app is for educational/demo purposes only. Always consult a doctor for medical advice.")