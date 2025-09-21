import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# ---------------------------
# Load saved KNN bundle
# ---------------------------
with open("models/knn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

scaler_X = bundle["scaler_X"]
scaler_y = bundle["scaler_y"]
knn_model = bundle["knn_model"]

# Load dataset for visualization
df = pd.read_csv("data/emp_sal.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="KNN Regression Salary Predictor", page_icon="💼", layout="centered")

st.title("💼 KNN Regression Salary Predictor")
st.write("Predict employee salary based on **Position Level** using a trained **KNN Regressor**.")

# Sidebar inputs
st.sidebar.header("⚙️ Settings")

position_level = st.sidebar.slider("Select Position Level", float(X.min()), float(X.max()), 6.5, 0.5)

# Option to override k
k_override = st.sidebar.number_input("Neighbors (k)", min_value=1, max_value=10, value=knn_model.n_neighbors, step=1)

# ---------------------------
# Prediction
# ---------------------------
# If user changes k, retrain temporarily with new k
if k_override != knn_model.n_neighbors:
    temp_model = KNeighborsRegressor(n_neighbors=k_override, weights="uniform")
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1,1)).ravel()
    temp_model.fit(X_scaled, y_scaled)
    model_used = temp_model
    st.sidebar.info(f"🔄 Using custom k = {k_override}")
else:
    model_used = knn_model

# Prepare input and predict
X_input_scaled = scaler_X.transform(np.array([[position_level]]))
y_pred_scaled = model_used.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))[0,0]

# Show result
st.subheader("💰 Predicted Salary")
st.success(f"Predicted Salary for Level {position_level:.1f}: **{y_pred:,.2f}**")

# ---------------------------
# Visualization
# ---------------------------
st.subheader("📈 Model Visualization")

# Smooth curve
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
X_grid_scaled = scaler_X.transform(X_grid)
y_grid_pred_scaled = model_used.predict(X_grid_scaled)
y_grid_pred = scaler_y.inverse_transform(y_grid_pred_scaled.reshape(-1,1))

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X, y, color='red', label='Actual data')
ax.plot(X_grid, y_grid_pred, color='blue', label=f'KNN Predictions (k={model_used.n_neighbors})')
ax.scatter(position_level, y_pred, color='green', s=100, label=f'Prediction @ {position_level:.1f}')
ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.set_title("KNN Regression Salary Prediction")
ax.legend()
st.pyplot(fig)

# ---------------------------
# About
# ---------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
    - **Model:** K-Nearest Neighbors (KNN) Regressor  
    - **Dataset:** Employee Position Levels vs Salary  
    - **Features:**  
      - Interactive prediction for any Position Level  
      - Option to change number of neighbors (k)  
      - Visualization of actual vs predicted salaries  
    - Built with **Streamlit + scikit-learn**
    """)
