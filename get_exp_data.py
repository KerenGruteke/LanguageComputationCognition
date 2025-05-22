import copy
import pickle
from pathlib import Path

import numpy as np

from BERT_functions import extract_sentence_representation
from learn_decoder import read_matrix

EXP_1_PARAMS = {
    "glove_vectors_path": "vectors_180concepts.GV42B300.txt",
    "stimuli_text_path": "stimuli_180concepts.txt",
    "poor_rank_threshold": 160,
    "high_rank_threshold": 15,
    "extremely_rank_threshold": 5,
}

EXP_2_PARAMS = {
    "pickle_path": "EXP2.pkl",
    "glove_vectors_path": "vectors_384sentences.GV42B300.average.txt",
    "stimuli_text_path": "stimuli_384sentences.txt",
    "poor_rank_threshold": 360,
    "high_rank_threshold": 15,
    "extremely_rank_threshold": 5,
}

EXP_3_PARAMS = {
    "pickle_path": "EXP3.pkl",
    "glove_vectors_path": "vectors_243sentences.GV42B300.average.txt",
    "stimuli_text_path": "stimuli_243sentences.txt",
    "poor_rank_threshold": 220,
    "high_rank_threshold": 15,
    "extremely_rank_threshold": 5,
}

EXP_PARAMS_BY_EXP_NUM = {1: EXP_1_PARAMS, 2: EXP_2_PARAMS, 3: EXP_3_PARAMS}

DATA_PATH = Path("data")


class Experiment:
    def __init__(self, exp_num, get_bert_decoding=False):
        self.exp_num = exp_num
        exp_params = EXP_PARAMS_BY_EXP_NUM[exp_num]

        self.poor_rank_threshold = exp_params["poor_rank_threshold"]
        self.high_rank_threshold = exp_params["high_rank_threshold"]
        self.extremely_rank_threshold = exp_params["extremely_rank_threshold"]

        glove_vectors_path = DATA_PATH / exp_params["glove_vectors_path"]
        stimuli_text_path = DATA_PATH / exp_params["stimuli_text_path"]

        if exp_num == 2 or exp_num == 3:
            pickle_path = DATA_PATH / exp_params["pickle_path"]
            with open(pickle_path, "rb") as f:
                exp_data = pickle.load(f)

            self.keyPassageCategory = exp_data["keyPassageCategory"]
            self.labelsPassageCategory = exp_data["labelsPassageCategory"]
            self.labelsPassageForEachSentence = exp_data["labelsPassageForEachSentence"]
            self.labelsSentences = exp_data["labelsSentences"]
            self.keySentences = exp_data["keySentences"]
            self.keyPassages = exp_data["keyPassages"]
            self.fmri_data = exp_data["Fmridata"]

            self.glove_vectors = read_matrix(glove_vectors_path, sep=" ")
            self.categories_names = [arr[0] for arr in self.keyPassageCategory[0]]
            self.categories_all_vectors = self.vector_to_category()
            self.avg_glove_vectors_by_category = self.get_avg_vectors_per_category(
                self.glove_vectors.copy()
            )
            self.avg_fmri_vectors_by_category = self.get_avg_vectors_per_category(
                self.fmri_data.copy()
            )

            with open(stimuli_text_path, "r") as file:
                stimuli_text = file.readlines()
            self.stimuli_text = stimuli_text

        elif exp_num == 1:
            self.stimuli_text = np.genfromtxt(stimuli_text_path, dtype=np.dtype("U"))
            self.glove_vectors = read_matrix(glove_vectors_path, sep=" ")
            self.fmri_data = read_matrix(DATA_PATH / "neuralData_for_EXP1.csv", sep=",", fmri_exp_1=True)

        if get_bert_decoding:
            sentences_exp = [sentence.strip() for sentence in stimuli_text]
            self.bert_vectors = extract_sentence_representation(sentences_exp)
            self.avg_bert_vectors_by_category = self.get_avg_vectors_per_category(
                self.bert_vectors.clone().detach()
            )

    def vector_to_category(self):
        vec_to_category = []
        for sent_idx, sent in enumerate(self.glove_vectors):
            sentence_idx = self.labelsSentences[sent_idx - 1][0]
            passage_idx = self.labelsPassageForEachSentence[sentence_idx - 1][0]
            category_idx = self.labelsPassageCategory[passage_idx - 1][0]
            category_name = self.keyPassageCategory[0][category_idx - 1][0]
            vec_to_category.append(category_name)
        return vec_to_category

    def get_avg_vectors_per_category(self, input_vectors):
        new_vectors = copy.deepcopy(input_vectors)
        avg_vectors_per_category = {}
        for category_name in self.categories_names:
            avg_vectors_per_category[category_name] = 0

        for idx, vec in enumerate(new_vectors):
            name = self.categories_all_vectors[idx]
            if avg_vectors_per_category[name] == 0:
                avg_vectors_per_category[name] = [vec, 1]
            else:
                avg_vectors_per_category[name][0] += vec
                avg_vectors_per_category[name][1] += 1

        avg_vectors_per_category_list = []
        for category_values in avg_vectors_per_category.values():
            avg = [x / category_values[1] for x in category_values[0]]
            avg_vectors_per_category_list.append(avg)

        return avg_vectors_per_category_list

    def get_vectors_by_type(self, vector_type: str):
        if vector_type == "Glove":
            vectors = self.glove_vectors
            avg_vectors_per_category = self.avg_glove_vectors_by_category
        elif vector_type == "BERT":
            vectors = self.bert_vectors
            avg_vectors_per_category = self.avg_bert_vectors_by_category
        elif vector_type == "fMRI":
            vectors = self.fmri_data
            avg_vectors_per_category = self.avg_fmri_vectors_by_category

        return vectors, avg_vectors_per_category
