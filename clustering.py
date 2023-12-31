from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing  # to normalise existing X
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from get_exp_data import Experiment
from rank_based_accuracy_functions import cosine_similarity

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
    plt.clf()

    print(f"{best_k_wscc=}")
    return best_k_wscc


def run_kmeans(
    exp: Experiment, avg_categories: bool, vectors, vector_type: str, k: int = None
):
    if avg_categories:
        categories_names = exp.categories_names
    else:
        categories_names = exp.categories_all_vectors

    if not k:
        k = best_k_kmeans(vectors=vectors, vector_type=vector_type)
    kmeans = KMeans(n_clusters=k)
    norm_vectors = preprocessing.normalize(vectors)
    kmeans.fit(norm_vectors)
    clusters_numbers = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    cluster_to_categories = {}
    # for each cluster creates dict of {category: count}
    for idx, cluster_num in enumerate(clusters_numbers):
        category_name = categories_names[idx]
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

    return cluster_nums_of_all_vectors, category_to_cluster, k


def calculate_within_similatiry(cluter_to_vecs: dict, cluter_to_names: dict):
    mean_within = {}
    median_within = {}
    similarity_values = {}

    # Iterate over each cluster in the dictionary
    for cluster_num, vectors in cluter_to_vecs.items():
        cluster_size = len(vectors)
        if cluster_size <= 1:
            mean_within[cluster_num] = 1
            median_within[cluster_num] = 1
            continue

        # Calculate the pairwise distances between vectors in the cluster using cosine similarity
        similarity_list = []
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                similarity = cosine_similarity(vectors[i], vectors[j])
                similarity_list.append(similarity)

        # Calculate the sum of pairwise distances
        mean_within_dist = np.mean(similarity_list)
        median_within_dist = np.median(similarity_list)
        print(mean_within_dist, median_within_dist)

        mean_within[cluster_num] = mean_within_dist
        median_within[cluster_num] = median_within_dist
        similarity_values[cluster_num] = similarity_list
    return mean_within, median_within, similarity_values


def calculate_between_similarity(cluter_to_vecs: dict, cluter_to_names: dict):
    # Compute the pairwise distances between centroids
    cluster_nums = list(cluter_to_vecs.keys())
    num_clusters = len(cluster_nums)
    similarity_list = []
    for i in range(num_clusters):
        print(f"until cluster {i}: mean_between={np.mean(similarity_list)}")
        for j in range(i + 1, num_clusters):
            cluster_i = cluter_to_vecs[i]
            cluter_i_name = cluter_to_names[i]
            cluster_j = cluter_to_vecs[j]
            cluter_j_name = cluter_to_names[j]
            for vec_i, name_i in zip(cluster_i, cluter_i_name):
                for vec_j, name_j in zip(cluster_j, cluter_j_name):
                    # Calculate the pairwise distances between vectors in the cluster using cosine similarity
                    similarity = cosine_similarity(vec_i, vec_j)
                    similarity_list.append(similarity)
                    print(f"sim {name_i}, {name_j} = {similarity}")

    # Calculate the average pairwise distance between centroids
    mean_between = np.mean(similarity_list)
    median_between = np.mean(similarity_list)
    print(mean_between, median_between)
    return mean_between, median_between, similarity_list


def calculate_between_centorids_similarity(cluster_dict):

    # Calculate the centroid for each cluster
    centroids = {}
    for cluster_num, vectors in cluster_dict.items():
        centroid = np.mean(vectors, axis=0)
        centroids[cluster_num] = centroid

    # Compute the pairwise distances between centroids
    cluster_nums = list(cluster_dict.keys())
    num_clusters = len(cluster_nums)
    similarity_list = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            centroid_i = centroids[cluster_nums[i]]
            centroid_j = centroids[cluster_nums[j]]
            similarity = cosine_similarity(centroid_i, centroid_j)
            similarity_list.append(similarity)

    # Calculate the average pairwise distance between centroids
    mean_between = np.mean(similarity_list)
    median_between = np.mean(similarity_list)
    print(mean_between, median_between)
    return mean_between, median_between, similarity_list


def create_cluster_to_vecs(category_to_cluster, vectors, categories_names, k: int):
    if type(vectors[0]) == list:
        vectors = [np.array(v) for v in vectors]

    vec_cluster_num_list = []
    for cat in categories_names:
        vec_cluster_num_list.append(category_to_cluster[cat])

    cluster_to_vecs = {}
    for cluster_num in range(k):
        cluster_to_vecs[cluster_num] = []
    for cluster, vec in zip(vec_cluster_num_list, vectors):
        cluster_to_vecs[cluster].append(vec)

    cluter_to_names = {}
    for cluster_num in range(k):
        cluter_to_names[cluster_num] = []
    for cluster, name in zip(vec_cluster_num_list, categories_names):
        cluter_to_names[cluster].append(name)

    return cluster_to_vecs, cluter_to_names


def plot_similarity_analysis(
    within_distances: dict,
    between_distance: float,
    y_axis_label: str,
    vector_type_for_clustring: str,
    vector_type_for_analyzing: str,
    k: int,
    mean_all: float = None,
    median_all: float = None,
):
    x_axis_label = "cluster num"
    title = f"{y_axis_label} within and between clusters of {vector_type_for_analyzing} vectors"
    distances = list(within_distances.values())
    distances.append(between_distance)
    clusters_list = list(within_distances.keys())
    clusters_list.append("between")
    clusters_list_str = [str(i) for i in clusters_list]

    colors = ["#219DE8"] * (len(clusters_list_str) - 1) + ["#ED541F"]
    plt.bar(clusters_list_str, distances, color=colors)
    if min(distances) > 0.5:
        plt.ylim(0.5, plt.ylim()[1])
    if mean_all:
        plt.axhline(mean_all, color="r", linestyle="--", label="Mean")
        plt.text(
            0.5, mean_all, f"Mean: {mean_all:.2f}", color="r", ha="center", va="bottom"
        )
    if median_all:
        plt.axhline(median_all, color="r", linestyle="--", label="Median")
        plt.text(
            5,
            median_all,
            f"Median: {median_all:.2f}",
            color="r",
            ha="center",
            va="bottom",
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.savefig(
        RESULTS_PATH
        / f"{y_axis_label} {vector_type_for_analyzing}. clustring by {vector_type_for_clustring} k={k}.jpg"
    )
    plt.clf()
