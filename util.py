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


def get_H(df, explain, ans):
    X = matrix(df[explain])
    if type(explain) is str:
        X = X.T
    X = np.concatenate([matrix(ones(len(X))).T, X], axis=1)
    return X*((X.T*X)**-1)*X.T

def get_sigma_hat(df, explain, ans):
    X = matrix(df[explain])
    if type(explain) is str:
        X = X.T
    X = np.concatenate([matrix(ones(len(X))).T, X], axis=1)
    Y = matrix(df[ans]).T
    n,p = np.shape(matrix(df[explain]))
    return (1/(n-p))*(Y.T*(np.identity(n) - get_H(df, explain, ans))*Y)**(1/2)

