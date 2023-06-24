import random

import numpy as np

from get_exp_data import Experiment
from learn_decoder import learn_decoder
from rank_based_accuracy_functions import rank_based_accuracy_exp

# from sklearn.model_selection import train_test_split


def decode_brain_to_BERT():
    random.seed(42)
    exp_2 = Experiment(exp_num=2, get_bert_decoding=True)
    exp_3 = Experiment(exp_num=3, get_bert_decoding=True)

    # Calculate the number of rows for the train and test sets
    train_rows = int(0.7 * exp_2.sentences_bert_vecs.shape[0])

    # Split into train sets
    BERT_train_exp2 = exp_2.sentences_bert_vecs[:train_rows, :]
    Fmridata_train_exp2 = exp_2.fmri_data[:train_rows, :]
    glove_train_exp2 = exp_2.glove_vectors[:train_rows, :]

    # Split into test sets
    BERT_test_exp2 = exp_2.sentences_bert_vecs[train_rows:, :]
    Fmridata_test_exp2 = exp_2.fmri_data[train_rows:, :]
    glove_test_exp2 = exp_2.glove_vectors[train_rows:, :]
    # --------------------------------------------------------------------------------------------- exp2 train and test

    BERT_decoder_train = learn_decoder(exp_2.fmri_data, BERT_train_exp2)
    print("------ part 2 - BERT representation score in exp2- train and test----------")
    accuracy, poor_rank, high_rank, extremely_high_rank = rank_based_accuracy_exp(
        fmri_data=exp_2.fmri_data,
        exp_vectors=BERT_test_exp2,
        train_M=BERT_decoder_train,
        exp_dict=exp_2,
    )

    print(f"avg accuracy over all data: {round(np.mean(accuracy), 3)}")

    glove_decoder_train = learn_decoder(Fmridata_train_exp2, glove_train_exp2)
    print(
        "------ part 2 - paper representation score in exp2- train and test----------"
    )
    accuracy, poor_rank, high_rank, extremely_high_rank = rank_based_accuracy_exp(
        fmri_data=Fmridata_test_exp2,
        exp_vectors=glove_test_exp2,
        train_M=glove_decoder_train,
        exp_dict=exp_2,
    )
    print(f"avg accuracy over all data: {round(np.mean(accuracy), 3)}")
    # ------------------------------------------------------------------------------------------------------------- exp3
    BERT_decoder = learn_decoder(exp_2.fmri_data, exp_2.sentences_bert_vecs)
    print("------ part 2 - BERT representation score in ex3----------")
    accuracy, poor_rank, high_rank, extremely_high_rank = rank_based_accuracy_exp(
        fmri_data=exp_3.fmri_data,
        exp_vectors=exp_3.sentences_bert_vecs,
        train_M=BERT_decoder,
        exp_dict=exp_3,
    )
    print(f"avg accuracy over all data: {round(np.mean(accuracy), 3)}")

    train_M_exp2 = learn_decoder(exp_2.fmri_data, exp_2.glove_vectors)
    print("------ part 2 - paper representation score in ex3----------")
    accuracy, poor_rank, high_rank, extremely_high_rank = rank_based_accuracy_exp(
        fmri_data=exp_3.fmri_data,
        exp_vectors=exp_3.glove_vectors,
        train_M=train_M_exp2,
        exp_dict=exp_3,
    )
    print(f"avg accuracy over all data: {round(np.mean(accuracy), 3)}")

    # TODO: add analysis of best worse topics ?


if __name__ == "__main__":
    decode_brain_to_BERT()
