import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model bundle (scalers + model)
with open("models/svr_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

scaler_X = model_bundle["scaler_X"]
scaler_y = model_bundle["scaler_y"]
svr_model = model_bundle["svr_model"]

# Load dataset (for visualization only)
df = pd.read_csv("data/emp_sal.csv")

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="SVR Regression App", page_icon="🤖", layout="wide")

st.title("💼 Salary Prediction using SVR Regression")
st.markdown("This app predicts **Employee Salary** based on Position Level using Support Vector Regression (SVR).")

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["🔍 Prediction", "ℹ️ About"])

# -------------------- PREDICTION PAGE --------------------
if menu == "🔍 Prediction":
    st.subheader("📊 Predict Employee Salary")

    # User input
    position_level = st.number_input("Enter Position Level:", min_value=1.0, max_value=10.0, step=0.5, value=6.5)

    # Preprocess input
    X_new = np.array([[position_level]])
    X_scaled = scaler_X.transform(X_new)
    y_pred_scaled = svr_model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # Show prediction
    st.success(f"💰 Predicted Salary for Level {position_level}: **{y_pred[0,0]:,.2f}**")

    # Visualization
    st.subheader("📉 Model Visualization")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df.iloc[:,1], df.iloc[:,2], color="blue", label="Actual Data")
    ax.scatter(position_level, y_pred, color="red", s=100, label="Predicted Point")
    ax.set_xlabel("Position Level")
    ax.set_ylabel("Salary")
    ax.set_title("SVR Regression Fit")
    ax.legend()
    st.pyplot(fig)

# -------------------- ABOUT PAGE --------------------
elif menu == "ℹ️ About":
    st.subheader("ℹ️ About this App")
    st.write("""
    - **Algorithm:** Support Vector Regression (SVR)  
    - **Use case:** Predicting salary from employee position level  
    - **Dataset:** Employee Salary (emp_sal.csv)  
    - Scaling applied: StandardScaler (for both X and y)  
    - Saved Model: `svr_model.pkl`  

    👉 Developed as part of **FSDS Regression Module**.  
    """)

    st.markdown("---")
    st.markdown("👨‍💻 *Developed by Mubasshir Ahmed*")
