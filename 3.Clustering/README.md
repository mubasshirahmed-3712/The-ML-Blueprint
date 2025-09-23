
# 🤖 Clustering Algorithms in Machine Learning

> *"Clustering is the art of discovering structure in data without labels."*  
This module explores **unsupervised learning techniques** where the goal is to group similar data points into clusters. Unlike supervised models, clustering does not rely on predefined outputs but instead uncovers hidden patterns.

---

## 📂 Folder Structure
```
3.Clustering/
│── 1.KMeans/
│   ├── concepts/              # Visual explanations of KMeans steps
│   ├── data/                  # Mall_Customers dataset
│   └── notebooks/             # Jupyter notebooks (basic + formatted)
│
│── 2.Hierarchical/
│   ├── concepts/              # Dendrogram and hierarchy visuals
│   ├── data/                  # Mall_Customers dataset
│   └── notebooks/             # Jupyter notebooks
│
│── 3.DBScan/
│   ├── concepts/              # Density-based clustering visuals
│   └── notebooks/             # Jupyter notebooks
```

---

## 🧭 Introduction to Clustering
Clustering is an **unsupervised learning task** where data points are grouped based on similarity.  
It is widely used in:
- 📊 Customer segmentation  
- 🧬 Gene sequence analysis  
- 🛍️ Market basket analysis  
- 📈 Anomaly detection  

---

## 1️⃣ KMeans Clustering

### 🔹 Concept
KMeans partitions the dataset into **K distinct clusters** based on the nearest mean (centroid).  

- **Steps:**
  1. Choose the number of clusters (K).  
  2. Initialize centroids randomly.  
  3. Assign points to the nearest centroid.  
  4. Update centroids as mean of points.  
  5. Repeat until convergence.  

- **Math Formula:**  
  Minimize the objective function:  
  J = Σ (||x - μ||²) over all clusters

### 🔹 In This Folder
- **Concepts:** Step-by-step visuals of centroid updates.  
- **Data:** Mall_Customers.csv dataset for clustering customer profiles.  
- **Notebooks:** `KMeans_Formatted.ipynb` explains the workflow clearly.  

### 🔹 Learning Outcome
- Understand how **centroids move** with iterations.  
- Learn how to choose **optimal K using the Elbow Method & Silhouette Score**.  

---

## 2️⃣ Hierarchical Clustering

### 🔹 Concept
Hierarchical clustering builds a **tree of clusters (dendrogram)**.  
Two main approaches:
- **Agglomerative (bottom-up)**: Merge clusters iteratively.  
- **Divisive (top-down)**: Split clusters recursively.  

- **Distance Metrics Used:**
  - Euclidean, Manhattan, Cosine, etc.  

- **Linkage Methods:**
  - Single, Complete, Average, Ward’s Method.  

### 🔹 In This Folder
- **Concepts:** Visuals of dendrograms and merges.  
- **Data:** Mall_Customers.csv for hierarchical clustering.  
- **Notebooks:** `hierarchy.ipynb` demonstrates Agglomerative clustering.  

### 🔹 Learning Outcome
- Learn how to **interpret dendrograms**.  
- Understand **linkage differences** and how they affect results.  

---

## 3️⃣ DBSCAN (Density-Based Spatial Clustering)

### 🔹 Concept
DBSCAN groups points that are **closely packed together** and labels points in sparse regions as **outliers/noise**.  

- **Parameters:**
  - `eps`: Maximum distance between two samples for them to be neighbors.  
  - `minPts`: Minimum number of points to form a dense cluster.  

- **Strengths:**  
  - Can find arbitrarily shaped clusters.  
  - Identifies noise/outliers naturally.  

- **Weaknesses:**  
  - Sensitive to parameter choice (`eps`, `minPts`).  

### 🔹 In This Folder
- **Concepts:** Illustrations of density and clusters.  
- **Notebooks:** `DB_SCAN CLUSTERING ALGORITHM.ipynb`.  

### 🔹 Learning Outcome
- Understand how density affects cluster formation.  
- Learn to detect anomalies in datasets using DBSCAN.  

---

## 🚀 Workflow for Clustering
1. Load dataset (`Mall_Customers.csv`).  
2. Preprocess (scaling if necessary).  
3. Apply algorithm (KMeans, Hierarchical, or DBSCAN).  
4. Visualize clusters in 2D/3D.  
5. Evaluate using metrics (Silhouette Score, Davies-Bouldin Index).  

---

## 🎯 Final Takeaways
- **KMeans** → Fast, scalable, but needs K beforehand.  
- **Hierarchical** → Provides hierarchy & dendrograms but less scalable.  
- **DBSCAN** → Great for irregular clusters & outlier detection.  

📌 This folder equips you with **theory + implementation + visualization** of clustering algorithms in Python.
