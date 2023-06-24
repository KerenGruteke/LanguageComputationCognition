import pickle
from pathlib import Path

import numpy as np

from BERT_functions import extract_sentence_representation
from learn_decoder import read_matrix

EXP_1_PARAMS = {
    "glove_vectors_path": "vectors_180concepts.GV42B300.txt",
    "stimuli_text_path": "stimuli_180concepts.txt",
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

EXP_PARAMS_BY_EXP_NUM = {2: EXP_2_PARAMS, 3: EXP_3_PARAMS}

DATA_PATH = Path("~").expanduser() / "data"


class Experiment:
    def __init__(self, exp_num, get_bert_decoding=False):
        self.exp_num = exp_num
        exp_params = EXP_PARAMS_BY_EXP_NUM[exp_num]

        self.poor_rank_threshold = exp_params["poor_rank_threshold"]
        self.high_rank_threshold = exp_params["high_rank_threshold"]
        self.extremely_rank_threshold = exp_params["extremely_rank_threshold"]

        pickle_path = DATA_PATH / exp_params["pickle_path"]
        glove_vectors_path = DATA_PATH / exp_params["glove_vectors_path"]
        stimuli_text_path = DATA_PATH / exp_params["stimuli_text_path"]

        with open(pickle_path, "rb") as f:
            exp_data = pickle.load(f)

        if exp_num == 2 or exp_num == 3:
            self.keyPassageCategory = exp_data["keyPassageCategory"]
            self.labelsPassageCategory = exp_data["labelsPassageCategory"]
            self.labelsPassageForEachSentence = exp_data["labelsPassageForEachSentence"]
            self.labelsSentences = exp_data["labelsSentences"]
            self.keySentences = exp_data["keySentences"]
            self.keyPassages = exp_data["keyPassages"]
            self.fmri_data = exp_data["Fmridata"]

            self.categories_names = [arr[0] for arr in self.keyPassageCategory[0]]
            self.glove_vectors = read_matrix(glove_vectors_path, sep=" ")

            with open(stimuli_text_path, "r") as file:
                stimuli_text = file.readlines()
            self.stimuli_text = stimuli_text

        elif exp_num == 1:
            self.stimuli_text = np.genfromtxt(stimuli_text_path, dtype=np.dtype("U"))
            self.glove_vectors = read_matrix(glove_vectors_path, sep=" ")
            # replace FMRI data because have bug in original file
            self.fmri_data = read_matrix("modified_file.csv", sep=",")

        if get_bert_decoding:
            sentences_exp = [sentence.strip() for sentence in stimuli_text]
            self.sentences_bert_vecs = extract_sentence_representation(sentences_exp)
