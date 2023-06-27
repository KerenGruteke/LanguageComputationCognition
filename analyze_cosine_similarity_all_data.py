import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from clustering import RESULTS_PATH
from get_exp_data import Experiment
from rank_based_accuracy_functions import cosine_similarity


def plot_rand_sim_values(vectors, vec_type: str, n: int = None):
    if type(vectors) == np.ndarray:
        if vectors.shape[1] > 1:
            vectors = list(vectors)
    if n:
        rand_vectors = copy.deepcopy(vectors)
        random.shuffle(rand_vectors)
        rand_vectors = rand_vectors[:n]
    else:
        rand_vectors = vectors
        n = len(rand_vectors)

    if type(rand_vectors[0]) == list:
        rand_vectors = [np.array(v) for v in rand_vectors]

    similarity_list = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = cosine_similarity(rand_vectors[i], rand_vectors[j])
            similarity_list.append(similarity)

    plt.hist(similarity_list)
    plt.xlabel("cosine similarity score")
    plt.ylabel("amount of vectors pairs")
    plt.title(f"cosine similarity of {vec_type} vectors pairs")
    plt.savefig(RESULTS_PATH / f"sim hist {vec_type}.jpg")
    plt.clf()

    print(vec_type)
    print(len(similarity_list))
    print(np.mean(similarity_list), np.median(similarity_list))
    return np.mean(similarity_list), np.median(similarity_list)


if __name__ == "__main__":
    random.seed(42)
    # exp_1 = Experiment(exp_num=1)
    # exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=False)
    plot_rand_sim_values(exp_3.glove_vectors, vec_type="glove")
    plot_rand_sim_values(exp_3.avg_glove_vectors_by_category, vec_type="glove avg")
    plot_rand_sim_values(exp_3.fmri_data, vec_type="fmri")
