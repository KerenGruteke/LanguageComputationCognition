from sklearn.cluster import KMeans

from get_exp_data import Experiment


def run_kmeans(exp: Experiment, vectors, k: int):
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
