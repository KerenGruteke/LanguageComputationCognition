from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from get_exp_data import Experiment

RESULTS_PATH = Path("results")


def best_k_kmeans(vectors, vector_type):
    np.random.seed(42)
    k_values = range(2, 15)
    wcss = []
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(vectors, labels))

    # Plot the WCSS values
    plt.plot(k_values, wcss, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    # Find the best value of K
    diff = []
    for i in range(1, len(wcss)):
        diff.append(wcss[i] - wcss[i - 1])

    best_k_wscc = diff.index(max(diff)) + 2

    plt.savefig(RESULTS_PATH / f"{vector_type} Elbow Method best_k={best_k_wscc}.jpg")
    plt.clf()

    # Plot the silhouette scores
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")

    best_k_silo = k_values[silhouette_scores.index(max(silhouette_scores))]
    plt.savefig(
        RESULTS_PATH / f"{vector_type} Silhouette Method best_k={best_k_silo}.jpg"
    )

    print(f"{best_k_wscc=}")
    return best_k_wscc


def run_kmeans(exp: Experiment, vectors, vector_type: str, k: int = None):
    if not k:
        k = best_k_kmeans(vectors=vectors, vector_type=vector_type)
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
