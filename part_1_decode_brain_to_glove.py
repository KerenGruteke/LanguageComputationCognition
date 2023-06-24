import random

import numpy as np

from get_exp_data import Experiment
from learn_decoder import learn_decoder
from rank_based_accuracy_functions import get_best_worse_topics, rank_based_accuracy_exp


def decode_brain_to_glove():
    random.seed(42)
    exp_1 = Experiment(exp_num=1)
    exp_2 = Experiment(exp_num=2)
    exp_3 = Experiment(exp_num=3)

    # train on exp 1 data
    train_M = learn_decoder(exp_1.fmri_data, exp_1.glove_vectors)
    print(f"{train_M.shape=}")

    for exp in [exp_2, exp_3]:
        print(f"--------- EXP_{exp.exp_num} ----------")
        accuracy, poor_rank, high_rank, extremely_high_rank = rank_based_accuracy_exp(
            fmri_data=exp.fmri_data,
            exp_vectors=exp.glove_vectors,
            train_M=train_M,
            exp_dict=exp
        )
        print(f"avg accuracy over all data: {round(np.mean(accuracy), 3)}")

        print(
            f"categories with rank < {exp.extremely_rank_threshold}: {get_best_worse_topics(extremely_high_rank, exp)}"
        )
        print(f"categories with rank between {exp.high_rank_threshold} - {exp.extremely_rank_threshold}: {get_best_worse_topics(high_rank, exp)}")
        print(f"categories with rank > {exp.poor_rank_threshold}: {get_best_worse_topics(poor_rank, exp)}")


if __name__ == "__main__":
    decode_brain_to_glove()
