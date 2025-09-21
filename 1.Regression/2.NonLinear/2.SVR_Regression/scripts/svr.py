import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r'C:\Users\MUBASSHIR\OneDrive\Desktop\FSDS-Class-Files\25-8-2025_ML\emp_sal.csv')

# Features (Position Level) & Target (Salary)
X = df.iloc[:, 1:2].values   # keep 2D
y = df.iloc[:, 2].values

#svm model
from sklearn.svm import SVR
svr_regression = SVR(kernel='poly', degree=5, gamma='auto')
svr_regressor = svr_regression.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("SVR Model Prediction for 6.5:", svr_model_pred)

#knn model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=3, weights='uniform')
knn_reg_model.fit(X, y)

knn_model_pred = knn_reg_model.predict([[6.5]])
print("KNN Model Prediction for 6.5:", knn_model_pred)

