import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Load saved Random Forest model
# ---------------------------
rf_model = joblib.load("model/rf_model.pkl")

# Load dataset for visualization
df = pd.read_csv("data/emp_sal.csv")
X = df.iloc[:, 1:2].values  # Position Level
y = df.iloc[:, 2].values    # Salary

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="🌲 Mubasshir's Random Forest Salary Predictor", page_icon="🌲", layout="centered")

st.title("🌲 Random Forest Regression Salary Predictor")
st.write("Predict employee salary based on **Position Level** using a trained **Random Forest Regressor**.")

# Sidebar inputs
st.sidebar.header("⚙️ Settings")

position_level = st.sidebar.slider("Select Position Level", float(X.min()), float(X.max()), 6.5, 0.5)

# Option to override n_estimators
n_override = st.sidebar.number_input("Number of Trees (n_estimators)", min_value=1, value=rf_model.n_estimators, step=1)

# ---------------------------
# Prediction
# ---------------------------
# If user changes n_estimators, retrain temporarily with new trees
if n_override != rf_model.n_estimators:
    from sklearn.ensemble import RandomForestRegressor
    temp_model = RandomForestRegressor(n_estimators=n_override, random_state=0)
    temp_model.fit(X, y)
    model_used = temp_model
    st.sidebar.info(f"🔄 Using custom n_estimators = {n_override}")
else:
    model_used = rf_model

# Prepare input and predict
y_pred = model_used.predict(np.array([[position_level]]))[0]

# Show result
st.subheader("💰 Predicted Salary")
st.success(f"Predicted Salary for Level {position_level:.1f}: **{y_pred:,.2f}**")

# ---------------------------
# Visualization
# ---------------------------
st.subheader("📈 Model Visualization")

# Smooth curve
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_grid_pred = model_used.predict(X_grid)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color='red', label='Actual data')
ax.plot(X_grid, y_grid_pred, color='blue', label=f'Random Forest Predictions (n={model_used.n_estimators})')
ax.scatter(position_level, y_pred, color='green', s=100, label=f'Prediction @ {position_level:.1f}')
ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.set_title("Random Forest Regression Salary Prediction")
ax.legend()
st.pyplot(fig)

# ---------------------------
# About
# ---------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
    - **Model:** Random Forest Regressor  
    - **Dataset:** Employee Position Levels vs Salary  
    - **Features:**  
      - Interactive prediction for any Position Level  
      - Option to change number of trees (n_estimators)  
      - Visualization of actual vs predicted salaries  
    - Built with **Streamlit + scikit-learn**  
    - 🏷️ Brand: *Mubasshir AI Labs*  
    """)
