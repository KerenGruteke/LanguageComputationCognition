import BERT_functions as BERT
from get_exp_data import Experiment
import learn_decoder as decoder
import numpy as np 

def calc_P_matrix(X):
    C = np.matmul(X.T, X)
    C_inverse = np.linalg.inv(C)
    P = np.matmul(C_inverse, X.T)
    return P


def linear_regression_model(fMRI_data, embeddings):
    my_array = np.array(embeddings)
    n_embeddings = my_array.shape[0]
    print(n_embeddings)
    X = np.c_[np.ones(n_embeddings), embeddings]
    P = calc_P_matrix(X)

    R_squared_scores = []
    n_voxels = range(fMRI_data.shape[1])
    for voxel_id in n_voxels:
        y = fMRI_data[: ,voxel_id]
        beta_hat = np.matmul(P, y)
        y_hat = np.matmul(X, beta_hat)
        y_mean = np.mean(y)

        # calc R_squared
        SSR = 0
        SSres = 0
        for i in range(len(y_hat)):
            SSR += (y_hat[i] - y_mean) ** 2
            SSres += (y[i] - y_hat[i]) ** 2
        SST = SSR + SSres
        R_squared = SSR / SST
        R_squared_scores.append(R_squared)

        # print(f"{SSR=}")
        # print(f"{SST=}")
        # print(f"{SSres=}")
        # print(f"{R_squared=}")
    return R_squared_scores


# # def encode_brain_vectors():
# #     # BERT
# #     # Read sentences from file
# #     # file_path = "C:\Users\Hadar\Desktop\NLP_cognition_new\DATA\stimuli_243sentences.txt"
# #     # with open(file_path, 'r') as file:
# #     #     sentences = file.readlines()
# #     # sentences = [sentence.strip() for sentence in sentences]
# #     # representations = BERT.extract_sentence_representation(sentences)
# file_path = 'DATA/stimuli_243sentences.txt'
# with open(file_path, 'r') as file:
#     # Read the contents of the file
#     contents = file.read()

exp_3 = Experiment(exp_num=3)
vectors_exp3 = exp_3.stimuli_text
vectors_exp3 = decoder.read_matrix(vectors_exp3, sep=" ")
Fmridata = exp_3.fmri_data
x = linear_regression_model(Fmridata, vectors_exp3)

