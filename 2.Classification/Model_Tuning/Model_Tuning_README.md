# 🔹 Model Tuning in Machine Learning (Classification Focus)

When we train classification models (Logistic Regression, KNN, SVM, Random Forest, etc.), their performance depends on three things:  

1. **How we split & evaluate data** (Train/Test, Cross Validation)  
2. **Which hyperparameters we choose** (C in SVM, k in KNN, max depth in trees)  
3. **Which metrics we optimize** (Accuracy? F1? AUC?)  

👉 **Model Tuning** = systematically improving these three steps.  
This folder demonstrates model tuning using **Support Vector Machine (SVM)** on the `Social_Network_Ads.csv` dataset.  

---

## 📂 Folder Structure

```
8.Model_Tuning/
├─ data/
│  └─ Social_Network_Ads.csv
└─ notebooks/
   ├─ 1_SVM_with_CrossValidation.ipynb
   ├─ 2_SVM_with_GridSearchCV.ipynb
   ├─ 3_SVM_with_RandomSearchCV.ipynb
   ├─ 4_SVM_with_ROC_AUC.ipynb
   └─ 5_Final_Summary.ipynb
```

---

## 🔹 1. Cross Validation (CV)

### ❓ Problem:
A simple train-test split is unstable. Accuracy depends on random split → model may look better/worse than it actually is.  

### ✅ Solution: **K-Fold Cross Validation**  
- Split dataset into **k folds** (e.g., k=5).  
- Train on k-1 folds, test on remaining fold.  
- Repeat for all folds.  
- Take average performance → more reliable.  

⚡ **Syntax**:

```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print("Mean Accuracy:", accuracies.mean())
print("Std Deviation:", accuracies.std())
```

---

## 🔹 2. Grid Search

### ❓ Problem:
Algorithms have hyperparameters (not learned automatically). Example:  
- SVM → `C`, `kernel`, `gamma`  
- KNN → number of neighbors `k`  
- Decision Trees → `max_depth`, `min_samples_split`  

Choosing manually = guesswork.  

### ✅ Solution: **GridSearchCV**  
- Define a **grid (dictionary)** of possible hyperparameter values.  
- Train models for **all combinations**.  
- Evaluate with CV.  
- Select the **best parameters**.  

⚡ **Syntax**:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": [0.1, 0.5, 1.0]
}

grid = GridSearchCV(SVC(), param_grid, cv=10, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
```

---

## 🔹 3. Random Search

### ❓ Problem:
GridSearch tests *all* combinations → slow when search space is large.  

### ✅ Solution: **RandomizedSearchCV**  
- Randomly sample combinations for fixed number of iterations (`n_iter`).  
- Much faster.  
- Often finds near-optimal solution.  

⚡ **Syntax**:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {
    "C": [0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "rbf"],
    "gamma": uniform(0.1, 1.0)
}

random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=20, cv=10, scoring="accuracy", random_state=0
)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
```

---

## 🔹 4. Performance Metrics Beyond Accuracy

Accuracy alone can be misleading (e.g., in imbalanced data).  

### 📌 Confusion Matrix  

|               | Predicted Positive | Predicted Negative |
|---------------|-------------------|-------------------|
| **Actual Pos** | True Positive (TP) | False Negative (FN) |
| **Actual Neg** | False Positive (FP) | True Negative (TN) |

---

### 📌 Precision & Recall  

- **Precision** = TP / (TP + FP)  
👉 Out of predicted positives, how many are correct?  

- **Recall (Sensitivity/TPR)** = TP / (TP + FN)  
👉 Out of actual positives, how many did we catch?  

---

### 📌 F1 Score  

- Harmonic mean of Precision & Recall.  
- Good for imbalanced datasets.  

```python
from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

---

### 📌 ROC & AUC  

- **ROC Curve** = plots TPR (Recall) vs FPR at different thresholds.  
- **AUC** = probability model ranks a random positive higher than random negative.  
  - Perfect model → AUC = 1.0  
  - Random guessing → AUC = 0.5  

⚡ **Syntax**:

```python
from sklearn.metrics import roc_curve, roc_auc_score
y_prob = classifier.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("ROC AUC:", auc)
```

📊 Example ROC Curve:  

![ROC Curve Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png)  

---

## 🔹 5. Practical Tuning Workflow

1. Split data → Train/Test  
2. Use **K-Fold CV** on training data  
3. Use **GridSearchCV or RandomizedSearchCV** for hyperparameter tuning  
4. Evaluate final tuned model on Test Set  
5. Report metrics → Confusion Matrix, Precision, Recall, F1, ROC–AUC  

---

## ✅ Final Takeaways

- **Cross Validation** → reliable performance estimate  
- **Grid Search** → systematic tuning (but slow)  
- **Random Search** → faster, near-optimal  
- **Metrics** → use Precision, Recall, F1, ROC–AUC (not just Accuracy)  
- **SVM Tuning Example** shows how all these fit together  

👉 Without tuning & proper metrics, you risk deploying a model that looks good on a test split but fails in real-world data.  

---

📌 This README is not just project documentation — it’s also a **study guide** for anyone learning model tuning in classification.  
