import random

import matplotlib.pyplot as plt
import numpy as np

from clustering import (
    RESULTS_PATH,
    calculate_between_centorids_similarity,
    calculate_between_similarity,
    calculate_within_similatiry,
    create_fmri_to_cluster,
    plot_similarity_analysis,
    run_kmeans,
)
from get_exp_data import Experiment
from reduce_dimension_and_plot import (
    plot_reduced_vectors_with_labels,
    reduce_dimension_and_plot,
)

# -----------------------------------------------------------------------------------------------------------------


def run_clustering_before_reduction(
    exp: Experiment, vector_type: str, k: int, method: str
):
    before_after = "before"
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)

    # --- regular ---
    cluster_nums_of_all_vectors, category_to_cluster, k = run_kmeans(
        exp=exp, avg_categories=False, vectors=vectors, vector_type=vector_type, k=k
    )
    X_reduced = reduce_dimension_and_plot(
        method=method,
        vectors_matrix=vectors,
    )
    plot_reduced_vectors_with_labels(
        method=method,
        vector_type=vector_type,
        k=k,
        labels=cluster_nums_of_all_vectors,
        names=exp.categories_all_vectors,
        X_reduced=X_reduced,
        plot_names=False,
        before_after=before_after,
    )

    # --- avg ---
    vector_type = f"{vector_type}_avg"
    cluster_nums_of_all_vectors, category_to_cluster, k = run_kmeans(
        exp=exp,
        avg_categories=True,
        vectors=avg_vectors_per_category,
        vector_type=vector_type,
        k=k,
    )
    avg_X_reduced = reduce_dimension_and_plot(
        method=method,
        vectors_matrix=np.array(avg_vectors_per_category),
    )
    plot_reduced_vectors_with_labels(
        method=method,
        vector_type=vector_type,
        k=k,
        labels=list(category_to_cluster.values()),
        names=exp.categories_names,
        X_reduced=avg_X_reduced,
        plot_names=True,
        before_after=before_after,
    )


def run_clustering_after_reduction(
    exp: Experiment, vector_type: str, k: int, method: str
):
    before_after = "after"
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)

    # --- regular ---
    X_reduced = reduce_dimension_and_plot(
        method=method,
        vectors_matrix=vectors,
    )
    cluster_nums_of_all_vectors, category_to_cluster, k = run_kmeans(
        exp=exp,
        avg_categories=False,
        vectors=X_reduced,
        vector_type=f"reduced_{vector_type}",
        k=k,
    )
    print("ploting...")
    plot_reduced_vectors_with_labels(
        method=method,
        vector_type=vector_type,
        k=k,
        labels=cluster_nums_of_all_vectors,
        names=exp.categories_all_vectors,
        X_reduced=X_reduced,
        plot_names=False,
        before_after=before_after,
    )

    # --- avg ---
    vector_type = f"{vector_type}_avg"
    avg_X_reduced = reduce_dimension_and_plot(
        method=method,
        vectors_matrix=np.array(avg_vectors_per_category),
    )

    cluster_nums_of_all_vectors, category_to_cluster, k = run_kmeans(
        exp=exp,
        avg_categories=True,
        vectors=avg_X_reduced,
        vector_type=f"reduced_{vector_type}_avg",
        k=k,
    )

    print("ploting...")
    plot_reduced_vectors_with_labels(
        method=method,
        vector_type=vector_type,
        k=k,
        labels=list(category_to_cluster.values()),
        names=exp.categories_names,
        X_reduced=avg_X_reduced,
        plot_names=True,
        before_after=before_after,
    )


def analyze_clusters_distances(exp: Experiment, vector_type: str, k: int):
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)
    # --- avg ---
    vector_type = f"{vector_type}_avg"
    cluster_nums_of_all_vectors, category_to_cluster, k = run_kmeans(
        exp=exp,
        avg_categories=True,
        vectors=avg_vectors_per_category,
        vector_type=vector_type,
        k=k,
    )
    fmri_to_clusters = create_fmri_to_cluster(
        category_to_cluster=category_to_cluster, exp=exp
    )
    # calc similarity
    mean_within, median_within, similarity_values = calculate_within_similatiry(
        fmri_to_clusters
    )
    mean_between, median_between, similarity_list = calculate_between_similarity(
        fmri_to_clusters
    )
    # plot
    plot_similarity_analysis(
        mean_within,
        mean_between,
        y_axis_label="mean cosine similarity",
        vector_type=vector_type,
        k=k,
    )
    plot_similarity_analysis(
        median_within,
        median_between,
        y_axis_label="median cosine similarity",
        vector_type=vector_type,
        k=k,
    )


if __name__ == "__main__":
    random.seed(42)
    # exp_1 = Experiment(exp_num=1)
    # exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=False)

    # explore k meands and reducing dimensions
    # for method in ["TSNE", "UMAP"]:
    #     for k in [None, 5, 8, 10]:
    #         # run_clustering_before_reduction(
    #         #     exp=exp_3, vector_type="Glove", k=k, method=method
    #         # )
    #         # run_clustering_after_reduction(
    #         #     exp=exp_3, vector_type="Glove", k=k, method=method
    #         # )
    #         run_clustering_before_reduction(
    #             exp=exp_3, vector_type="BERT", k=k, method=method
    #         )
    #         # run_clustering_before_reduction(
    #         #     exp=exp_3,
    #         #     vector_type="fMRI",
    #         #     k=k,
    #         #     method=method
    #         # )

    # analyze_clusters_distances
    analyze_clusters_distances(exp=exp_3, vector_type="Glove", k=5)
