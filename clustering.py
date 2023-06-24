from sklearn.cluster import KMeans
from get_exp_data import Experiment
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def best_k_kmeans(sentences_vectors):
    np.random.seed(42)
    k_values = range(2, 15)
    wcss = []
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sentences_vectors)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(sentences_vectors, labels))

    # Plot the WCSS values
    plt.plot(k_values, wcss, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

    # Find the best value of K
    diff = []
    for i in range(1, len(wcss)):
        diff.append(wcss[i] - wcss[i - 1])

    best_k_wscc = diff.index(max(diff)) + 2

    # Plot the silhouette scores
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()

    best_k_silo = k_values[silhouette_scores.index(max(silhouette_scores))]

    return best_k_wscc

def run_kmeans(exp: Experiment, vectors, k: int=None):
    if not k:
        k = best_k_kmeans(vectors)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(vectors)
    clusters_numbers = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    cluster_to_categories = {}
    # for each cluster creates dict of {category: count}
    for idx, cluster_num in enumerate(clusters_numbers):
        category_name = exp.categories_all_vectors[idx]
        if cluster_num not in cluster_to_categories.keys():
            cluster_to_categories[cluster_num] = {}  # new count diict
        if category_name not in cluster_to_categories[cluster_num].keys():
            cluster_to_categories[cluster_num][category_name] = 1  # add category
        else:
            cluster_to_categories[cluster_num][category_name] += 1  # increase count

    category_to_cluster = {}
    for category in exp.categories_names:
        chosen_cluster = -1
        max_appearance = 0
        for cluster_num in range(k):
            if category not in cluster_to_categories[cluster_num].keys():
                pass
            else:
                if cluster_to_categories[cluster_num][category] > max_appearance:
                    max_appearance = cluster_to_categories[cluster_num][category]
                    chosen_cluster = cluster_num

        category_to_cluster[category] = chosen_cluster

    cluster_nums_of_all_vectors = [
        category_to_cluster[category] for category in exp.categories_all_vectors
    ]

    return cluster_nums_of_all_vectors, category_to_cluster
