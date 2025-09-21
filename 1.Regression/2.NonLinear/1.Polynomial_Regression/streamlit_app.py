import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# Load dataset
df = pd.read_csv("data/emp_sal.csv")
X = df.iloc[:, 1:2].values   # keep 2D
y = df.iloc[:, 2].values

# ----------------------------------------------------
# Streamlit UI
st.title("📈 Polynomial Regression Salary Predictor")

st.sidebar.header("Model Controls")

# Dynamic polynomial degree slider
degree = st.sidebar.slider("Select Polynomial Degree", min_value=1, max_value=10, value=5)

# Train Linear Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train Polynomial Model (dynamic degree)
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# ----------------------------------------------------
# Predictions
position_level = st.sidebar.number_input("Enter Position Level", min_value=float(min(X)), max_value=float(max(X)), value=6.5, step=0.5)
linear_pred = lin_reg.predict([[position_level]])[0]
poly_pred = poly_reg.predict(poly.transform([[position_level]]))[0]

st.subheader("🔮 Predictions")
st.write(f"**Linear Regression Prediction:** {linear_pred:,.2f}")
st.write(f"**Polynomial Regression (Degree {degree}) Prediction:** {poly_pred:,.2f}")

# ----------------------------------------------------
# Model Performance
st.subheader("📊 Model Performance")
lin_r2 = r2_score(y, lin_reg.predict(X))
poly_r2 = r2_score(y, poly_reg.predict(X_poly))
st.write(f"Linear Regression R²: {lin_r2:.4f}")
st.write(f"Polynomial Regression R² (Degree {degree}): {poly_r2:.4f}")

# ----------------------------------------------------
# Visualization
st.subheader("📉 Regression Visualization")
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X, y, color='red', label="Actual Data")

# Linear line
ax.plot(X, lin_reg.predict(X), color='green', label="Linear Regression")

# Polynomial curve
x_grid = np.arange(min(X), max(X)+0.1, 0.1).reshape(-1,1)
ax.plot(x_grid, poly_reg.predict(poly.transform(x_grid)), color='blue', label=f"Polynomial (Degree {degree})")

ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.legend()
st.pyplot(fig)
