from sklearn.cluster import KMeans


def run_kmeans(exp_3, sentences_vectors, categories_all_vectors, exp_names, K):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(sentences_vectors)
    cluster_number = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    clusters_dict = {}
    for idx, point in enumerate(cluster_number):
        category_name = categories_all_vectors[idx]
        if point not in clusters_dict.keys():
            clusters_dict[point] = {}
        if category_name not in clusters_dict[point].keys():
            clusters_dict[point][category_name] = 1
        else:
            clusters_dict[point][category_name] += 1

    category_to_cluster = {}
    for category in exp_names:
        chosen_cluster = -1
        max_appearance = 0
        for point in range(K):
            if category not in clusters_dict[point].keys():
                pass
            else:
                if clusters_dict[point][category] > max_appearance:
                    max_appearance = clusters_dict[point][category]
                    chosen_cluster = point

        category_to_cluster[category] = chosen_cluster

    clusters_per_vec = [
        category_to_cluster[category] for category in categories_all_vectors
    ]

    avg_vectors_per_category = {}
    for idx, vec in enumerate(exp_3.vectors):
        name = categories_all_vectors[idx]
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
        categories_all_vectors,
        clusters_per_vec,
        avg_vectors_per_category_list,
        category_to_cluster,
    )
