import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r'C:\Users\MUBASSHIR\OneDrive\Desktop\FSDS-Class-Files\25-8-2025_ML\emp_sal.csv')

# Features (Position Level) & Target (Salary)
X = df.iloc[:, 1:2].values   # keep 2D
y = df.iloc[:, 2].values

# ----------------------------------------------------
# Function: Visualization helper
def plot_regression(x, y, model, title, poly=None):
    plt.scatter(x, y, color='red')
    
    if poly:  # Polynomial regression curve
        x_grid = np.arange(min(x), max(x)+0.1, 0.1).reshape(-1,1)
        plt.plot(x_grid, model.predict(poly.transform(x_grid)), color='blue')
    else:     # Linear regression line
        plt.plot(x, model.predict(x), color='blue')
        
    plt.title(title)
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()

# ----------------------------------------------------
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression (degree = 5)
degree = 5
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# ----------------------------------------------------
# Visualizations
plot_regression(X, y, lin_reg, "Linear Regression")
plot_regression(X, y, poly_reg, "Polynomial Regression", poly=poly)

# ----------------------------------------------------
# Predictions at Position Level = 6.5
linear_pred = lin_reg.predict([[6.5]])
poly_pred = poly_reg.predict(poly.transform([[6.5]]))

print(f"Linear Prediction at 6.5: {linear_pred[0]:.2f}")
print(f"Polynomial Prediction at 6.5: {poly_pred[0]:.2f}")

# ----------------------------------------------------
# R² Scores (model evaluation)
print("Linear Regression R²:", r2_score(y, lin_reg.predict(X)))
print("Polynomial Regression R²:", r2_score(y, poly_reg.predict(X_poly)))

# ----------------------------------------------------
# Save everything into ONE pickle
import os
import pickle

# ----------------------------------------------------
# Save models
os.makedirs("../models", exist_ok=True)   # ✅ create models folder if not exists
model_path = "../models/polynomial_all_models.pkl"

with open(model_path, "wb") as f:
    pickle.dump({
        "linear_model": lin_reg,
        "polynomial_model": poly_reg,
        "poly_transformer": poly,
        "degree": 5
    }, f)

print(f"✅ Models saved successfully at {model_path}")

