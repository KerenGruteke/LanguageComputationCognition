import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from learn_decoder import *
import sklearn.linear_model
import random
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


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


def rank_based_accuracy_exp(fmri_data, exp_vectors, train_M, exp=True):
    pred_vectors = np.matmul(fmri_data, train_M)
    rank_list = []
    poor_rank_idx = []
    high_rank_idx = []
    extremely_high_idx = []
    for vec_index, pred_vec in enumerate(pred_vectors):
        rank = get_rank_exp(pred_vec, vec_index, exp_vectors)
        rank_list.append(rank)
        if exp and rank >= 360:
            poor_rank_idx.append(vec_index)
        if (not exp) and rank >= 220:
            poor_rank_idx.append(vec_index)
        if rank <= 15:
            high_rank_idx.append(vec_index)
        if rank <= 5:
            extremely_high_idx.append(vec_index)

    accuracy = np.mean(rank_list)
    return accuracy, poor_rank_idx, high_rank_idx, extremely_high_idx


def best_worse_topics(special_idx_exp, labelsSentences_2, labelsPassageForEachSentence_2, labelsPassageCategory_2,
                      keyPassageCategory_2):
    categories = {}
    for sent_idx in special_idx_exp:
        sentence_idx = labelsSentences_2[sent_idx - 1][0]
        passage_idx = labelsPassageForEachSentence_2[sentence_idx - 1][0]
        category_idx = labelsPassageCategory_2[passage_idx - 1][0]
        category_name = keyPassageCategory_2[0][category_idx - 1][0]
        if category_name not in categories.keys():
            categories[category_name] = 1
        else:
            categories[category_name] += 1
    return categories


class Experiment():
    def __init__(self, pkl_path, vec_path, sentence_path):
        with open(pkl_path, 'rb') as f:
            exp_data = pickle.load(f)
        self.keyPassageCategory = exp_data["keyPassageCategory"]
        self.labelsPassageCategory = exp_data["labelsPassageCategory"]
        self.labelsPassageForEachSentence = exp_data["labelsPassageForEachSentence"]
        self.labelsSentences = exp_data["labelsSentences"]
        self.keySentences = exp_data["keySentences"]
        self.keyPassages = exp_data["keyPassages"]

        self.Fmridata = exp_data["Fmridata"]
        self.vectors = read_matrix(vec_path, sep=" ")
        with open(sentence_path, 'r') as file:
            sentences = file.readlines()
        sentences_exp = [sentence.strip() for sentence in sentences]
        self.bert_representations = extract_sentence_representation(sentences_exp)


def run():
    exp_2 = Experiment('EXP2.pkl', "vectors_384sentences.GV42B300.average.txt", "stimuli_384sentences.txt")
    exp_3 = Experiment('EXP3.pkl', "vectors_243sentences.GV42B300.average.txt", "stimuli_243sentences.txt")

    # data experiment 1
    vectors_exp1 = read_matrix("vectors_180concepts.GV42B300.txt", sep=" ")
    concepts_exp1 = np.genfromtxt('stimuli_180concepts.txt', dtype=np.dtype('U'))  # The names of the 180 concepts
    Fmridata_exp1 = read_matrix("modified_file.csv", sep=",")

    # running
    random.seed(42)
    train_M = learn_decoder(Fmridata_exp1, vectors_exp1)
    print(train_M.shape)
    print("---------EXP_2----------")
    accuracy_exp2, poor_rank_idx_exp2, high_rank_idx_exp2, extremely_high_idx_exp2 = \
        rank_based_accuracy_exp(exp_2.Fmridata, exp_2.vectors, train_M)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp2), 3)}")

    best_categories_exp2 = best_worse_topics(extremely_high_idx_exp2, exp_2.labelsSentences,
                                             exp_2.labelsPassageForEachSentence,
                                             exp_2.labelsPassageCategory, exp_2.keyPassageCategory)
    mid_categories_exp2 = best_worse_topics(high_rank_idx_exp2, exp_2.labelsSentences,
                                            exp_2.labelsPassageForEachSentence,
                                            exp_2.labelsPassageCategory, exp_2.keyPassageCategory)
    worse_categories_exp2 = best_worse_topics(poor_rank_idx_exp2, exp_2.labelsSentences,
                                              exp_2.labelsPassageForEachSentence,
                                              exp_2.labelsPassageCategory, exp_2.keyPassageCategory)
    print(f'best_categories for exp 2: {best_categories_exp2}')
    print(f'mid_categories for exp 2: {mid_categories_exp2}')
    print(f'worse_categories for exp 2: {worse_categories_exp2}')

    print("---------EXP_3----------")
    accuracy_exp3, poor_rank_idx_exp3, high_rank_idx_exp3, extremely_high_idx_exp3 = \
        rank_based_accuracy_exp(exp_3.Fmridata, exp_3.vectors, train_M, False)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp3), 3)}")

    best_categories_exp3 = best_worse_topics(extremely_high_idx_exp3, exp_3.labelsSentences,
                                             exp_3.labelsPassageForEachSentence,
                                             exp_3.labelsPassageCategory, exp_3.keyPassageCategory)
    mid_categories_exp3 = best_worse_topics(high_rank_idx_exp3, exp_3.labelsSentences,
                                            exp_3.labelsPassageForEachSentence,
                                            exp_3.labelsPassageCategory, exp_3.keyPassageCategory)
    worse_categories_exp3 = best_worse_topics(poor_rank_idx_exp3, exp_3.labelsSentences,
                                              exp_3.labelsPassageForEachSentence,
                                              exp_3.labelsPassageCategory, exp_3.keyPassageCategory)
    print(f'best_categories for exp 3: {best_categories_exp3}')
    print(f'mid_categories for exp 3: {mid_categories_exp3}')
    print(f'worse_categories for exp 3: {worse_categories_exp3}')

    # part 2


class SentenceRepresentation(nn.Module):
    def __init__(self):
        super(SentenceRepresentation, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 300)  # Adjust the input and output size as needed

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        sentence_representation = self.linear(sentence_representation)
        return sentence_representation


def extract_sentence_representation(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceRepresentation()
    representations = []

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones(input_ids.shape)

        with torch.no_grad():
            representation = model(input_ids, attention_mask)

        representations.append(representation)

    return torch.cat(representations, dim=0)


def run_part_2():
    with open('EXP2.pkl', 'rb') as f:
        exp2_data = pickle.load(f)
    Fmridata_exp2 = exp2_data["Fmridata"]
    vectors_exp2 = read_matrix("vectors_384sentences.GV42B300.average.txt", sep=" ")
    train_M_p2 = learn_decoder(Fmridata_exp2, vectors_exp2)
    with open('EXP3.pkl', 'rb') as f:
        exp3_data = pickle.load(f)
    Fmridata_exp3 = exp3_data["Fmridata"]
    vectors_exp3 = read_matrix("vectors_243sentences.GV42B300.average.txt", sep=" ")

    # Read sentences from file
    with open("stimuli_384sentences.txt", 'r') as file:
        sentences = file.readlines()
    sentences_exp2 = [sentence.strip() for sentence in sentences]

    with open("stimuli_243sentences.txt", 'r') as file:
        sentences = file.readlines()
    sentences_exp3 = [sentence.strip() for sentence in sentences]

    BERT_representations_exp2 = extract_sentence_representation(sentences_exp2)
    BERT_representations_exp3 = extract_sentence_representation(sentences_exp3)

    # Calculate the number of rows for the train and test sets
    train_rows = int(0.7 * BERT_representations_exp2.shape[0])

    # Split into train sets
    BERT_train2 = BERT_representations_exp2[:train_rows, :]
    Fmridata_train_exp2 = Fmridata_exp2[:train_rows, :]
    vectors_train_exp2 = vectors_exp2[:train_rows, :]

    # Split into teat sets
    BERT_test2 = BERT_representations_exp2[train_rows:, :]
    Fmridata_test_exp2 = Fmridata_exp2[train_rows:, :]
    vectors_test_exp2 = vectors_exp2[train_rows:, :]
    # ---------------------------------------------------------------------------------------------exp2 train and test
    BERT_decoder_train = learn_decoder(Fmridata_train_exp2, BERT_train2)
    print("------ part 2 - BERT representation score in ex2- train and test----------")
    accuracy_exp2, poor_rank_idx_exp2, high_rank_idx_exp2, extremely_high_idx_exp2 = \
        rank_based_accuracy_exp(Fmridata_test_exp2, BERT_test2, BERT_decoder_train, False)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp2), 3)}")

    vectors_decoder_train = learn_decoder(Fmridata_train_exp2, vectors_train_exp2)
    print("------ part 2 - paper representation score in ex2- train and test----------")
    accuracy_exp2, poor_rank_idx_exp2, high_rank_idx_exp2, extremely_high_idx_exp2 = \
        rank_based_accuracy_exp(Fmridata_test_exp2, vectors_test_exp2, vectors_decoder_train, False)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp2), 3)}")
    # -------------------------------------------------------------------------------------------------------------exp3
    BERT_decoder = learn_decoder(Fmridata_exp2, BERT_representations_exp2)
    print("------ part 2 - BERT representation score in ex3----------")
    accuracy_exp2, poor_rank_idx_exp2, high_rank_idx_exp2, extremely_high_idx_exp2 = \
        rank_based_accuracy_exp(Fmridata_exp3, BERT_representations_exp3, BERT_decoder, False)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp2), 3)}")

    train_M_exp2 = learn_decoder(Fmridata_exp2, vectors_exp2)
    print("------ part 2 - paper representation score in ex3----------")
    accuracy_exp2, poor_rank_idx_exp2, high_rank_idx_exp2, extremely_high_idx_exp2 = \
        rank_based_accuracy_exp(Fmridata_exp3, vectors_exp3, train_M_exp2, False)
    print(f"avg accuracy over all data: {round(np.mean(accuracy_exp2), 3)}")


def run_part_3():
    exp_2 = Experiment('EXP2.pkl', "vectors_384sentences.GV42B300.average.txt")
    exp_3 = Experiment('EXP3.pkl', "vectors_243sentences.GV42B300.average.txt")
    X_train, X_test, y_train, y_test = train_test_split(exp_3.vectors, exp_3.Fmridata, test_size=0.2, random_state=42)

    r2_scores = []
    for voxel in range(10):
        model = LinearRegression()
        y_train_voxel = y_train[:, voxel]
        model.fit(X_train, y_train_voxel)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test[:, voxel], y_pred)
        r2_scores.append(r2)

    for voxel, r2 in enumerate(r2_scores):
        print(f"Voxel {voxel + 1}: R2 score = {r2}")


if __name__ == '__main__':
    run()
    run_part_2()
    run_part_3()
