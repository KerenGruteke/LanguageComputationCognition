from sklearn.cluster import KMeans

from get_exp_data import Experiment


def run_kmeans(exp: Experiment, vectors, k: int):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(vectors)
    cluster_number = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    clusters_dict = {}
    for idx, point in enumerate(cluster_number):
        category_name = exp.categories_all_vectors[idx]
        if point not in clusters_dict.keys():
            clusters_dict[point] = {}
        if category_name not in clusters_dict[point].keys():
            clusters_dict[point][category_name] = 1
        else:
            clusters_dict[point][category_name] += 1

    category_to_cluster = {}
    for category in exp.categories_names:
        chosen_cluster = -1
        max_appearance = 0
        for point in range(k):
            if category not in clusters_dict[point].keys():
                pass
            else:
                if clusters_dict[point][category] > max_appearance:
                    max_appearance = clusters_dict[point][category]
                    chosen_cluster = point

        category_to_cluster[category] = chosen_cluster

    clusters_per_vec = [
        category_to_cluster[category] for category in exp.categories_all_vectors
    ]

    avg_vectors_per_category = {}
    for idx, vec in enumerate(vectors):
        name = exp.categories_all_vectors[idx]
        if name not in avg_vectors_per_category.keys():
            avg_vectors_per_category[name] = [vec, 1]
        else:
            avg_vectors_per_category[name][0] += vec
            avg_vectors_per_category[name][1] += 1

    avg_vectors_per_category_list = []
    for category, arr_values in avg_vectors_per_category.items():
        avg = [x / arr_values[1] for x in arr_values[0]]
        avg_vectors_per_category_list.append(avg)

    return (
        clusters_per_vec,
        avg_vectors_per_category_list,
        category_to_cluster,
    )
