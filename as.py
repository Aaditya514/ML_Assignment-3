import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\KIIT\Downloads\kmeans - kmeans_blobs (1).csv" 
df = pd.read_csv(file_path)

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

data = df.values[:, :2]  
data = normalize_data(data)

def k_means(data, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    centroids = data[np.random.choice(len(data), k, replace=False)]  
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

labels_k2, centroids_k2 = k_means(data, k=2)
labels_k3, centroids_k3 = k_means(data, k=3)

def plot_clusters(data, labels, centroids, k, title):
    plt.figure(figsize=(6, 5))
    for i in range(k):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f"Cluster {i+1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()

plot_clusters(data, labels_k2, centroids_k2, k=2, title="K-Means Clustering (k=2)")
plot_clusters(data, labels_k3, centroids_k3, k=3, title="K-Means Clustering (k=3)")
