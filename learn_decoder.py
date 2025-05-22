#!/usr/bin/env python
""" learn_decoder """

import numpy as np
import sklearn.linear_model


def read_matrix(filename, sep=",", fmri_exp_1=False):
    # if fmri_exp_1 then the first line is a header
    # else no header
    if fmri_exp_1:
        with open(filename) as infile:
            lines = infile.readlines()[1:]
    else:
        with open(filename) as infile:
            lines = infile.readlines()
            
    clean_lines = []
    for line in lines:
        clean_lines.append(list(map(float, line.strip().split(sep))))
    
    # if fmri_exp_1 then the first element of each line is the index
    # remove the first element of each line
    if fmri_exp_1:
        clean_lines = [line[1:] for line in clean_lines]
    
    return np.array(clean_lines)


def learn_decoder(data, vectors):
    """Given data (a CxV matrix of V voxel activations per C concepts)
    and vectors (a CxD matrix of D semantic dimensions per C concepts)
    find a matrix M such that the dot product of M and a V-dimensional
    data vector gives a D-dimensional decoded semantic vector.
    data X M = vectors
    where data is a CxV matrix, vectors is a CxD matrix, and M is a VxD matrix.

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
