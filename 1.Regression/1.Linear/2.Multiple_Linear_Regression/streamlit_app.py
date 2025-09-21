# streamlit_app.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load the trained MLR model
# -------------------------------
MODEL_PATH = os.path.join("models", "multiple_linear_regression_model.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file not found at {MODEL_PATH}! Please train and save the model first.")
    st.stop()

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Profit Prediction App", page_icon="📊", layout="centered")

# -------------------------------
# App Title & Description
# -------------------------------
st.title("💼 Multiple Linear Regression Predictor")
st.markdown(
    """
    Predict **Company Profit** based on investment in:  
    - Digital Marketing  
    - Promotion  
    - Research  
    - State (Bangalore, Chennai, Hyderabad)  
    Enter the values below and click **Predict Profit**.
    """
)

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("📝 Enter Investment Details")

digital_marketing = st.number_input(
    "💻 Digital Marketing Spend", min_value=0.0, value=10000.0, format="%.2f"
)
promotion = st.number_input(
    "📢 Promotion Spend", min_value=0.0, value=5000.0, format="%.2f"
)
research = st.number_input(
    "🔬 Research Spend", min_value=0.0, value=2000.0, format="%.2f"
)
state = st.selectbox("📍 State", ["Bangalore", "Chennai", "Hyderabad"])

# -------------------------------
# Preprocess Inputs (One-Hot Encoding)
# -------------------------------
state_features = {
    "Bangalore": [1, 0, 0],
    "Chennai": [0, 1, 0],
    "Hyderabad": [0, 0, 1]
}

# Column names must match training data
feature_names = [
    "DigitalMarketing", "Promotion", "Research",
    "State_Bangalore", "State_Chennai", "State_Hyderabad"
]

input_df = pd.DataFrame(
    [[digital_marketing, promotion, research] + state_features[state]],
    columns=feature_names
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Profit"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Profit: **₹{prediction:,.2f}**")

    # -------------------------------
    # Optional Visualization
    # -------------------------------
    st.subheader("📈 Investment Breakdown")
    fig, ax = plt.subplots()
    categories = ["Digital Marketing", "Promotion", "Research"]
    values = [digital_marketing, promotion, research]
    ax.bar(categories, values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylabel("Investment Amount")
    ax.set_title("Investment Distribution")
    st.pyplot(fig)

# -------------------------------
# Model Info Section
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ About the Model")
st.write(
    """
    - **Multiple Linear Regression** model trained on company investment dataset.  
    - Features: Digital Marketing, Promotion, Research, State (One-Hot Encoded)  
    - Predicts company **Profit** based on these inputs.  

    **~ Mubasshir Ahmed**
    """
)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🔗 [GitHub Repository](https://github.com/mubasshirahmed-3712) | ✨ Built by *Mubasshir Ahmed*")
