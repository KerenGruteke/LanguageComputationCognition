import numpy as np


def calc_P_matrix(X):
    C = np.matmul(X.T, X)
    C_inverse = np.linalg.inv(C)
    P = np.matmul(C_inverse, X.T)
    return P


def linear_regression_model(fMRI_data, embeddings):
    n_embeddings = embeddings.shape[0]
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

        print(f"{SSR=}")
        print(f"{SST=}")
        print(f"{SSres=}")
        print(f"{R_squared=}")
    return R_squared
