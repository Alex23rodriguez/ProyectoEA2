import numpy as np
from numpy import ones
from scipy import matrix


def y_hat(df, explain, ans):
    X = matrix(df[explain])
    if type(explain) is str:
        X = X.T
    X = np.concatenate([matrix(ones(len(X))).T, X], axis=1)
    Y = matrix(df[ans]).T
    return X*((X.T*X)**-1) * X.T * Y


def b_hat(df, explain, ans):
    X = matrix(df[explain])
    if type(explain) is str:
        X = X.T
    X = np.concatenate([matrix(ones(len(X))).T, X], axis=1)
    Y = matrix(df[ans]).T
    return (X.T*X)**-1 * X.T * Y
