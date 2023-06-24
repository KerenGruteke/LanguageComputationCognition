import numpy as np

from clustering import run_kmeans
from get_exp_data import Experiment
from reduce_dimension_and_plot import reduce_dimension_and_plot

# -----------------------------------------------------------------------------------------------------------------


def run_all(exp: Experiment, vector_type: str, k: int):
    vectors, avg_vectors_per_category = exp.get_vectors_by_type(vector_type=vector_type)

    cluster_nums_of_all_vectors, category_to_cluster = run_kmeans(
        exp=exp, vectors=vectors, k=k
    )

    method = "PCA"
    # method = "TSNE"
    # method = "UMAP"

    reduce_dimension_and_plot(
        method=method,
        vectors_metrix=vectors,
        names=exp.categories_all_vectors,
        labels=cluster_nums_of_all_vectors,
        vector_type=vector_type,
        k=k,
        plot_names=False,
    )
    reduce_dimension_and_plot(
        method=method,
        vectors_metrix=np.array(avg_vectors_per_category),
        names=exp.categories_names,
        labels=list(category_to_cluster.values()),
        vector_type=f"{vector_type}_avg",
        k=k,
        plot_names=True,
    )


if __name__ == "__main__":
    k = 2
    # exp_1 = Experiment(exp_num=1)
    # exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=False)

    run_all(
        exp=exp_3,
        vector_type="Glove",
        k=k,
    )
    # run_all(
    #     exp=exp_3,
    #     vector_type="BERT",
    #     k=k,
    # )
    run_all(
        exp=exp_3,
        vector_type="fMRI",
        k=k,
    )
