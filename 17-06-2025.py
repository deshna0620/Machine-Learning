# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Load Wine Dataset
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (2 Components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply KMeans Clustering on PCA-Reduced Data
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Create Subplots for KMeans Clusters and True Labels
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: KMeans Cluster Assignments
scatter1 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
axs[0].set_title('K-Means Clustering on PCA-Reduced Wine Data')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
legend1 = axs[0].legend(*scatter1.legend_elements(), title="Cluster")
axs[0].add_artist(legend1)
axs[0].grid(True)

# Plot 2: True Class Labels
scatter2 = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='jet', s=50)
axs[1].set_title('Original Wine Classes in PCA Space')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')
legend2 = axs[1].legend(*scatter2.legend_elements(), title="Class")
axs[1].add_artist(legend2)
axs[1].grid(True)

plt.tight_layout()
plt.savefig('wine_pca_kmeans_17-06-2025.png')
plt.show()

# Compute Adjusted Rand Index to compare clusters with true labels
ari_score = adjusted_rand_score(y, clusters)
print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
