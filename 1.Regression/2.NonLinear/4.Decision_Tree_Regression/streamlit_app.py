import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# Load saved Decision Tree model
# ---------------------------
model = joblib.load("models/decision_tree_model.pkl")

# Load dataset
df = pd.read_csv("data/emp_sal.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Decision Tree Salary Predictor - Mubasshir", 
                   page_icon="🌳", layout="centered")

st.title("🌳 Decision Tree Regression Salary Predictor")
st.markdown("### Built with ❤️ by **Mubasshir Ahmed | FSDS**")

st.write("Predict employee salary based on **Position Level** using a trained **Decision Tree Regressor**.")

# Sidebar inputs
st.sidebar.header("⚙️ Settings")

position_level = st.sidebar.slider("Select Position Level", float(X.min()), float(X.max()), 6.5, 0.5)

# Option to override depth
depth_override = st.sidebar.number_input("Tree Depth (max_depth)", min_value=1, max_value=10, value=None, step=1)

# Option to override random_state
rs_override = st.sidebar.number_input("Random State", min_value=0, max_value=50, value=0, step=1)

# ---------------------------
# Prediction
# ---------------------------
if depth_override:
    temp_model = DecisionTreeRegressor(max_depth=depth_override, random_state=rs_override)
    temp_model.fit(X, y)
    model_used = temp_model
    st.sidebar.info(f"🔄 Using custom Decision Tree (depth={depth_override}, random_state={rs_override})")
else:
    model_used = model

# Prediction
y_pred = model_used.predict([[position_level]])[0]

# Show result
st.subheader("💰 Predicted Salary")
st.success(f"Predicted Salary for Level {position_level:.1f}: **{y_pred:,.2f}**")

# ---------------------------
# Visualization
# ---------------------------
st.subheader("📈 Model Visualization")

X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
y_grid_pred = model_used.predict(X_grid)

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X, y, color='red', label='Actual data')
ax.plot(X_grid, y_grid_pred, color='blue', label='Decision Tree Predictions')
ax.scatter(position_level, y_pred, color='green', s=100, label=f'Prediction @ {position_level:.1f}')
ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.set_title("Decision Tree Regression Salary Prediction")
ax.legend()
st.pyplot(fig)

# ---------------------------
# About
# ---------------------------
with st.expander("ℹ️ About this App"):
    st.write(f"""
    - **Model:** Decision Tree Regressor  
    - **Dataset:** Employee Position Levels vs Salary  
    - **Features:**  
      - Predicts salary for any Position Level  
      - Option to adjust Decision Tree depth and random state  
      - Visualization of actual vs predicted salaries  
    - **Branding:** Custom-built by *Mubasshir Ahmed (FSDS Portfolio Project)*  
    """)
