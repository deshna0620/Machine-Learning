# PCA and K-Means Clustering on Wine Dataset

## Overview
This project applies **Principal Component Analysis (PCA)** to reduce the Wine dataset to 2 dimensions, followed by **K-Means Clustering** to identify natural groups in the data. The clustering results are compared with the true class labels using the **Adjusted Rand Index (ARI)** and visual inspection.

## Objective
- Perform dimensionality reduction using **PCA**.
- Apply **K-Means clustering** on PCA-reduced data.
- Visualize the clustered data and compare it with true wine classes.
- Evaluate clustering performance using the **Adjusted Rand Index (ARI)**.

## Dataset
**Source:** `sklearn.datasets.load_wine()`  
**Features:** 13 numerical features describing various chemical properties of wine samples.  
**Target Classes:** 3 wine cultivars.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Steps Performed

1. **Data Loading:**
```python
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target
```

2. **Feature Scaling:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

3. **Dimensionality Reduction using PCA:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

4. **K-Means Clustering on PCA-Reduced Data:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
```

5. **Visualization (Subplots):**
```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# KMeans Clusters
scatter1 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
axs[0].set_title('K-Means Clustering on PCA-Reduced Wine Data')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
legend1 = axs[0].legend(*scatter1.legend_elements(), title="Cluster")
axs[0].add_artist(legend1)

# True Labels
scatter2 = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='jet', s=50)
axs[1].set_title('Original Wine Classes in PCA Space')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')
legend2 = axs[1].legend(*scatter2.legend_elements(), title="Class")
axs[1].add_artist(legend2)

plt.tight_layout()
plt.savefig('wine_pca_kmeans_17-06-2025.png')
plt.show()
```

6. **Clustering Evaluation using Adjusted Rand Index:**
```python
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(y, clusters)
print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
```

## Output Files
- **`wine_pca_kmeans_17-06-2025.png`**: Contains:
  1. K-Means cluster visualization on PCA-reduced data.
  2. Original wine class visualization on PCA-reduced data.

## Evaluation Metric
- **Adjusted Rand Index (ARI):** Measures similarity between predicted clusters and true classes.
- ARI Score Example Output:
```
Adjusted Rand Index (ARI): 0.897
```

## Requirements
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Notes
- PCA reduces the dataset from 13 dimensions to 2 for visualization.
- KMeans assumes **3 clusters** based on the known number of wine classes.
- ARI indicates the closeness of clustering to the actual classes.
