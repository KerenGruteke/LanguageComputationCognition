import numpy as np

from clustering import run_kmeans
from get_exp_data import Experiment
from reduce_dimension_and_plot import reduce_dimension_and_plot

# create a list that every entry represents the category of the sentence by order


def vector_to_category(vectors):
    vec_to_category = []
    for sent_idx, sent in enumerate(vectors):
        sentence_idx = exp_3.labelsSentences[sent_idx - 1][0]
        passage_idx = exp_3.labelsPassageForEachSentence[sentence_idx - 1][0]
        category_idx = exp_3.labelsPassageCategory[passage_idx - 1][0]
        category_name = exp_3.keyPassageCategory[0][category_idx - 1][0]
        vec_to_category.append(category_name)
    return vec_to_category


# -----------------------------------------------------------------------------------------------------------------


def run_all(vectors, exp_names, k: int, vector_type: str):
    (
        categories_all_vectors,
        clusters_per_vec,
        avg_vectors_per_category_list,
        category_to_cluster,
    ) = run_kmeans(vectors, exp_names, k=k)

    method = "PCA"
    # method = "TSNE"
    # method = "UMAP"

    reduce_dimension_and_plot(
        method=method,
        vectors_metrix=vectors,
        names=categories_all_vectors,
        labels=clusters_per_vec,
        vector_type=vector_type,
        k=k,
        plot_names=False,
    )
    reduce_dimension_and_plot(
        method=method,
        vectors_metrix=np.array(avg_vectors_per_category_list),
        names=exp_names,
        labels=list(category_to_cluster.values()),
        vector_type=f"{vector_type}_avg",
        k=k,
        plot_names=True,
    )


if __name__ == "__main__":
    k = 2
    exp_1 = Experiment(exp_num=1)
    exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=True)

    # vec of 384 categories
    categories_all_vectors = vector_to_category(exp_3.glove_vectors)

    run_all(
        vectors=exp_3.glove_vectors,
        exp_names=exp_3.categories_names,
        k=k,
        vector_type="Glove",
    )
    run_all(
        vectors=exp_3.bert_vectors,
        exp_names=exp_3.categories_names,
        k=k,
        vector_type="BERT",
    )
    run_all(
        vectors=exp_3.fmri_data,
        exp_names=exp_3.categories_names,
        k=k,
        vector_type="fMRI",
    )
