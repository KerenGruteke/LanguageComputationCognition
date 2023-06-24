import random

import numpy as np

from clustering import run_kmeans
from get_exp_data import Experiment
from reduce_dimension_and_plot import (
    plot_reduced_vectors_with_labels,
    reduce_dimension_and_plot,
)

# -----------------------------------------------------------------------------------------------------------------


def run_clustering_before_reduction(exp: Experiment, vector_type: str, k: int):
    before_after = "before"
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)

    method = "PCA"
    # method = "TSNE"
    # method = "UMAP"

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


def run_clustering_after_reduction(exp: Experiment, vector_type: str, k: int):
    before_after = "after"
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)

    method = "PCA"
    # method = "TSNE"
    # method = "UMAP"

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


if __name__ == "__main__":
    random.seed(42)
    # exp_1 = Experiment(exp_num=1)
    # exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=False)

    # None, 2, 5,
    for k in [6]:
        run_clustering_before_reduction(
            exp=exp_3,
            vector_type="Glove",
            k=k,
        )
        run_clustering_after_reduction(
            exp=exp_3,
            vector_type="Glove",
            k=k,
        )
        # run_clustering_before_reduction(
        #     exp=exp_3,
        #     vector_type="BERT",
        #     k=k,
        # )
        # run_clustering_before_reduction(
        #     exp=exp_3,
        #     vector_type="fMRI",
        #     k=k,
        # )
