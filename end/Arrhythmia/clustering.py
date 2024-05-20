import numpy as np
import random

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def hierarchical_clustering(data,cl_num):
    num_samples = len(data)
    distances = np.zeros((num_samples, num_samples))
    np.fill_diagonal(distances, np.inf)

    clusters = [[i] for i in range(num_samples)]

    while len(clusters) > cl_num:
        min_distance = np.inf
        merge_indices = (0, 0)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = clusters[i]
                cluster_j = clusters[j]

                for index_i in cluster_i:
                    for index_j in cluster_j:
                        distance = euclidean_distance(data[index_i], data[index_j])
                        if distance < min_distance:
                            min_distance = distance
                            merge_indices = (i, j)

        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]

        for k in range(len(clusters)):
            if k != i:
                cluster_k = clusters[k]
                for index_i in clusters[i]:
                    for index_k in cluster_k:
                        distances[index_i, index_k] = euclidean_distance(data[index_i], data[index_k])
                        distances[index_k, index_i] = distances[index_i, index_k]

    return clusters

# Example usage


cluster_num = int(input("Enter the number of clusters: "))
data = np.array([random.randint(0,100) for x in range(20)])
print(data)
cl = hierarchical_clustering(data,cluster_num)
print(cl)