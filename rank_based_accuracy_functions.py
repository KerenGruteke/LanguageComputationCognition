from typing import List

import numpy as np

from get_exp_data import Experiment


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    similarity = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return similarity


def get_rank_exp(pred_vec, vec_index, exp_vectors):
    similarity_dict = {}
    for j, vec in enumerate(exp_vectors):
        sim = cosine_similarity(pred_vec, vec)
        similarity_dict[j] = sim

    sorted_sim = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (vec_index_in_dic, sim) in enumerate(sorted_sim):
        if vec_index_in_dic == vec_index:
            return rank + 1


def rank_based_accuracy_exp(fmri_data, exp_vectors, train_M, exp_dict: Experiment):
    pred_vectors = np.matmul(fmri_data, train_M)
    rank_list = []
    poor_rank_idx = []
    high_rank_idx = []
    extremely_high_idx = []
    for vec_index, pred_vec in enumerate(pred_vectors):
        rank = get_rank_exp(pred_vec, vec_index, exp_vectors)
        rank_list.append(rank)
        if rank >= exp_dict.poor_rank_threshold:
            poor_rank_idx.append(vec_index)
        if rank <= exp_dict.high_rank_threshold and rank >= exp_dict.extremely_rank_threshold:
            high_rank_idx.append(vec_index)
        if rank <= exp_dict.extremely_rank_threshold:
            extremely_high_idx.append(vec_index)

    accuracy = np.mean(rank_list)
    return accuracy, poor_rank_idx, high_rank_idx, extremely_high_idx


def get_best_worse_topics(special_idx_exp: List, exp: Experiment):
    categories = {}
    for sent_idx in special_idx_exp:
        sentence_idx = exp.labelsSentences[sent_idx - 1][0]
        passage_idx = exp.labelsPassageForEachSentence[sentence_idx - 1][0]
        category_idx = exp.labelsPassageCategory[passage_idx - 1][0]
        category_name = exp.keyPassageCategory[0][category_idx - 1][0]
        if category_name not in categories.keys():
            categories[category_name] = 1
        else:
            categories[category_name] += 1
    return list(categories.keys())
