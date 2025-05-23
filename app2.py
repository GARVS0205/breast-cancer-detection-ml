import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import random
from sklearn.datasets import load_breast_cancer

# ✅ 1. Set up Streamlit app config
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# ✅ 2. Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "breast_cancer_model.pkl")
features_path = os.path.join(current_dir, "selected_features.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

# ✅ 3. Load model, features, and scaler
if not all(os.path.exists(p) for p in [model_path, features_path, scaler_path]):
    st.error("❌ Missing model, features, or scaler file!")
    st.stop()

model = joblib.load(model_path)
selected_features = joblib.load(features_path)
scaler = joblib.load(scaler_path)

# ✅ 4. Load dataset and prepare feature values
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df = df[selected_features]  # Keep only selected features

# ✅ 5. Title and Description
st.title("🩺 Breast Cancer Prediction App")

st.write("### Select values for each feature or fill randomly to predict tumor classification.")

# ✅ 6. State to store selected values
user_inputs = {}
rounded_feature_options = {}

# ✅ 7. Prepare dropdown options per feature
for feature in selected_features:
    unique_vals = sorted(df[feature].unique())
    rounded_vals = sorted(set(round(v, 2) for v in unique_vals))
    options = rounded_vals[:50] if len(rounded_vals) > 50 else rounded_vals
    rounded_feature_options[feature] = options

# ✅ 8. Add button to randomly fill values
if st.button("🎲 Randomly Fill Sample Values"):
    for feature in selected_features:
        user_inputs[feature] = random.choice(rounded_feature_options[feature])
else:
    for feature in selected_features:
        user_inputs[feature] = None

# ✅ 9. Dropdowns (with random defaults if available)
user_inputs_final = []
for feature in selected_features:
    default_value = user_inputs[feature]
    selected_val = st.selectbox(
        f"{feature}",
        rounded_feature_options[feature],
        index=rounded_feature_options[feature].index(default_value) if default_value in rounded_feature_options[feature] else 0
    )
    user_inputs_final.append(selected_val)

# ✅ 10. Scale inputs
user_inputs_array = np.array(user_inputs_final).reshape(1, -1)
user_inputs_scaled = scaler.transform(user_inputs_array)

# ✅ 11. Predict
threshold = 0.4

if st.button("🔍 Predict"):
    prob = model.predict_proba(user_inputs_scaled)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("🔴 The tumor is **Malignant (Cancerous)**.")
    else:
        st.success("🟢 The tumor is **Benign (Non-Cancerous)**.")

    st.write(f"📊 **Confidence Score:** {prob:.2%}")

st.write("💡 *Note: This is a machine learning prediction. Always consult a doctor for a professional diagnosis.*")
