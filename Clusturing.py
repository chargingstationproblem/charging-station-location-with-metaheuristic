import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import math




def ml_clusture_func(decoded_vector,node_coordinates):
    # Generate sample data
    n_samples = len(node_coordinates)
    n_features = 2
    X=(node_coordinates)
    #X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)

    # K-Means clustering
    n_clusters = int(decoded_vector[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    

    # distances now contains the rectilinear distances between each cluster center and each node coordinate
    distances=distance_func(cluster_centers,node_coordinates)
    return cluster_centers,distances



def distance_func(cluster_centers,node_coordinates):
    distances = np.zeros((len(cluster_centers), len(node_coordinates)))
    for i, center in enumerate(cluster_centers):
        for j, node in enumerate(node_coordinates):
            distances[i, j] = haversine(center[0], center[1], node[0], node[1])
            # (np.abs(center[0] - node[0]) + np.abs(center[1] - node[1])*1000)

    return distances




def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c  # Distance in kilometers

    return distance
