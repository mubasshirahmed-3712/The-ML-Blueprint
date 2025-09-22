# 📊 Regression Algorithms

Regression is one of the **fundamental supervised learning techniques** in Machine Learning.  
It focuses on modeling the relationship between **independent variables (features)** and a **dependent variable (target)**.  
The main goal is to **predict continuous outcomes** such as prices, sales, or scores.

---

## 🚀 What’s Inside this Folder?
This section of the repository contains end-to-end implementations of **Regression algorithms** including:

1. **Linear Regression** (Simple & Multiple)  
2. **Polynomial Regression**  
3. **Decision Tree Regression**  
4. **Random Forest Regression**  
5. **Support Vector Regression (SVR)**  
6. **Other Non-linear regressors**  

Each implementation covers:  
✔️ Dataset exploration & preprocessing  
✔️ Model training and evaluation  
✔️ Visualization of predictions  
✔️ Exporting trained models (`.pkl`)  
✔️ Deployment-ready Streamlit apps  

---

## 🧠 Understanding Regression

### 🔹 Linear Regression
- Assumes a **straight-line relationship** between input variables and the output.  
- Simple yet powerful baseline model.  
- Example: Predicting house prices based on size.

**Formula:**  
\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
\]

---

### 🔹 Nonlinear Regression
- Handles situations where data shows **curved or complex patterns**.  
- Models like **Polynomial, Decision Tree, Random Forest, SVR** can capture these relationships.  
- Example: Growth of bacteria population over time.

---

## 📂 Folder Structure

```
1.Regression/
├── 1.Linear/                # Linear regression models
│   ├── notebooks/           # Jupyter notebooks with step-by-step code
│   ├── data/                # Datasets used
│   ├── models/              # Saved trained models (.pkl)
│   └── app/                 # Streamlit apps for deployment
│
├── 2.NonLinear/             # Non-linear regression models
│   ├── Polynomial/
│   ├── DecisionTree/
│   ├── RandomForest/
│   ├── SVR/
│   └── ...
│
└── README.md                # ← You are here
```

---

## 📊 Evaluation Metrics
The performance of regression models is measured using:

- **Mean Absolute Error (MAE)** → Average absolute difference between predicted & actual values.  
- **Mean Squared Error (MSE)** → Penalizes larger errors more strongly.  
- **R² Score (Coefficient of Determination)** → Explains how much variance in target is captured by the model.

---

## 🎯 Learning Outcomes
By working through these notebooks and apps, you will:  
✅ Understand the **theory behind regression models**  
✅ Gain hands-on experience with **data preprocessing & feature engineering**  
✅ Learn how to **evaluate regression performance**  
✅ Build and deploy regression models with **Streamlit**  

---

## 📌 Next Steps
Once comfortable with Regression, proceed to:  
➡️ **Classification** (for predicting categories)  
➡️ **Clustering** (for grouping unlabeled data)  

---

🔥 *Regression is your foundation — master it, and you unlock the doors to advanced ML modeling!*  
