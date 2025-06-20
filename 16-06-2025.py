# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load Iris Dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method (WCSS)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Silhouette Scores (for k=2 to 10)
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Apply KMeans with Optimal k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot Elbow, Silhouette, and Cluster Visualization in Subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Elbow Method
axs[0].plot(range(1, 11), wcss, marker='o', color='blue')
axs[0].set_title('Elbow Method')
axs[0].set_xlabel('Number of Clusters (k)')
axs[0].set_ylabel('WCSS')
axs[0].grid(True)

# Subplot 2: Silhouette Score
axs[1].plot(k_range, silhouette_scores, marker='s', color='green')
axs[1].set_title('Silhouette Scores')
axs[1].set_xlabel('Number of Clusters (k)')
axs[1].set_ylabel('Silhouette Score')
axs[1].grid(True)

# Subplot 3: PCA Cluster Visualization
scatter = axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
axs[2].set_title('K-Means Clustering (k=3)')
axs[2].set_xlabel('PCA 1')
axs[2].set_ylabel('PCA 2')
legend1 = axs[2].legend(*scatter.legend_elements(), title="Clusters")
axs[2].add_artist(legend1)
axs[2].grid(True)

plt.tight_layout()
plt.savefig('iris_kmeans_plots_16-06-2025.png')
plt.show()

# Final Silhouette Score for selected k
final_score = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score for k={optimal_k}: {final_score:.3f}')
