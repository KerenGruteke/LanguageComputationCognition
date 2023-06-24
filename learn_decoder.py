#!/usr/bin/env python
""" learn_decoder """

import numpy as np
import sklearn.linear_model


def read_matrix(filename, sep=","):
    lines = []
    with open(filename) as infile:
        for line in infile:
            lines.append(list(map(float, line.strip().split(sep))))
    return np.array(lines)


def learn_decoder(data, vectors):
    """Given data (a CxV matrix of V voxel activations per C concepts)
    and vectors (a CxD matrix of D semantic dimensions per C concepts)
    find a matrix M such that the dot product of M and a V-dimensional
    data vector gives a D-dimensional decoded semantic vector.

    The matrix M is learned using ridge regression:
    https://en.wikipedia.org/wiki/Tikhonov_regularization
    """
    ridge = sklearn.linear_model.RidgeCV(
        alphas=[
            1,
            10,
            0.01,
            100,
            0.001,
            1000,
            0.0001,
            10000,
            0.00001,
            100000,
            0.000001,
            1000000,
        ],
        fit_intercept=False,
    )
    ridge.fit(data, vectors)
    return ridge.coef_.T
