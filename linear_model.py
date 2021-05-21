from numpy.core.numeric import identity
from scipy import matrix
import numpy as np


class LM:
    def __init__(self, df, y_col):
        self.df = df.copy()
        self.y_col = y_col
        self.y = matrix(df[y_col]).T

        self.cols = list(filter(lambda x: x != y_col, df.columns))
        self.n = len(df)

        # set self.explain, self.p, self.H and self.X
        self.update(self.cols.copy())

    def predict(self, x):
        pass

    def uptade(self, explain):
        assert all([x in self.cols for x in explain])
        self.explain = explain
        self.p = len(self.explain)
        self.X = matrix(self.df[self.explain])
        self.X = np.concatenate(
            [matrix(np.ones(len(self.X))).T, self.X], axis=1)

        self.H = self.X*((self.X.T*self.X)**-1)*self.X.T
        self.beta_hat = self._get_beta_hat()
        self.sig_sq_hat = self._get_sig_sq_hat()

    def _get_beta_hat(self):
        return (self.X.T*self.X)**-1 * self.X.T * self.y

    def _get_y_hat(self):
        return self.X*((self.X.T*self.X)**-1) * self.X.T * self.y

    def _get_sig_sq_hat(self):
        return 1/(self.n-self.p) * self.y.T * (np.identity(len(self.H)) - self.H) * self.y

    def _get_sigma_sq_hat(self):
        return 1/(self.n-self.p)*(self.y.T*(np.identity(self.n) - self.H)) * self.y
