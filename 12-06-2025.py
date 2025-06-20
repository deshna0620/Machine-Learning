# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = pd.read_csv('C:/Users/Neeraj kumar gothi/Desktop/Machine Learning/Mall_Customers.csv')

# Select Features: Age, Annual Income (k$), Spending Score (1-100)
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to Determine Optimal Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Determine optimal clusters (elbow — let's assume 4 for this example)
optimal_clusters = 4

# Apply K-Means with Optimal Clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# Create Subplots: Elbow Method + Cluster Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Elbow Method
axs[0].plot(range(1, 11), wcss, marker='o', color='blue')
axs[0].set_title('Elbow Method')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('WCSS')
axs[0].grid(True)

# Subplot 2: Customer Segments
scatter = axs[1].scatter(X_scaled[:, 1], X_scaled[:, 2], c=clusters, cmap='viridis')
axs[1].set_title('Customer Segments')
axs[1].set_xlabel('Annual Income (scaled)')
axs[1].set_ylabel('Spending Score (scaled)')
legend1 = axs[1].legend(*scatter.legend_elements(), title="Clusters")
axs[1].add_artist(legend1)
axs[1].grid(True)

plt.tight_layout()
plt.savefig('kmeans_plots_12-06-2025.png')
plt.show()

# Brief Cluster Interpretation (Print)
print("Cluster Summary:")
print(data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
