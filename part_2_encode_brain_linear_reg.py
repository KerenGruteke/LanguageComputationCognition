from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from get_exp_data import Experiment

RESULTS_PATH = Path("results")


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
        y = fMRI_data[:, voxel_id]
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

    return R_squared_scores


def p_value(r2_data, k, n):
    p_vals = []
    # Calculate the degrees of freedom for the F-test
    df_model = k - 1
    df_residual = n - k
    # df_total = n - 1
    for r2 in r2_data:
        # Calculate the F-statistic
        f_statistic = (r2 / df_model) / ((1 - r2) / df_residual)

        # Calculate the p-value
        p_value = 1 - stats.f.cdf(f_statistic, df_model, df_residual)
        p_vals.append(p_value)
        # print("R^2:", r2)
        # print("F-Statistic:", f_statistic)

    return p_vals


def plot_R_squared(R_squared_scores, vector_type: str):
    plt.hist(R_squared_scores, bins=10, edgecolor="black")
    plt.title(f"{vector_type} Linear Regression R-squared values")
    plt.xlabel("R-squared")
    plt.ylabel("Count")
    plt.savefig(RESULTS_PATH / f"{vector_type} Linear Regression R-squared values.jpg")
    plt.clf()


def plot_p_values(p_vals, vector_type: str):
    plt.hist(p_vals, bins=10, edgecolor="black")
    plt.title(f"{vector_type} Linear Regression p values")
    plt.xlabel("p_value")
    plt.ylabel("Count")
    plt.savefig(RESULTS_PATH / f"{vector_type} Linear Regression p values.jpg")
    plt.clf()


def run_and_analyze(language_vectors, fmri_vectors, vector_type: str):
    n = fmri_vectors.shape[1]  # Number of observations -> 243
    k = language_vectors.shape[1]  # Number of independent variables -> 300

    R_squared_scores = linear_regression_model(
        fMRI_data=fmri_vectors, embeddings=language_vectors
    )
    plot_R_squared(R_squared_scores=R_squared_scores, vector_type=vector_type)
    p_vals = p_value(R_squared_scores=R_squared_scores, k=k, n=n)
    plot_p_values(p_vals=p_vals, vector_type=vector_type)


if __name__ == "__main__":
    exp_3 = Experiment(exp_num=3, get_bert_decoding=True)
    run_and_analyze(
        language_vectors=exp_3.glove_vectors,
        fmri_vectors=exp_3.fmri_data,
        vector_type="Glove",
    )
    run_and_analyze(
        language_vectors=exp_3.bert_vectors,
        fmri_vectors=exp_3.fmri_data,
        vector_type="BERT",
    )
