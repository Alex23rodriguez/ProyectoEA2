from numpy.core.numeric import identity
from scipy import matrix, stats
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

    def update(self, explain):
        assert all([x in self.cols for x in explain])
        self.explain = explain
        self.p = len(self.explain)
        self.X = matrix(self.df[self.explain])
        self.X = np.concatenate(
            [matrix(np.ones(len(self.X))).T, self.X], axis=1)

        self.H = self.X*((self.X.T*self.X)**-1)*self.X.T
        self.beta_hat = self._get_beta_hat()
        self.var_beta_hat = self._get_var_beta_hat()
        self.sigma_sq_hat = self._get_sigma_sq_hat()

    def _get_beta_hat(self):
        return (self.X.T*self.X)**-1 * self.X.T * self.y

    def _get_var_beta_hat(self):
        return self.sigma_sq_hat * (self.X.T*self.X)**-1

    def _get_y_hat(self):
        return self.X*((self.X.T*self.X)**-1) * self.X.T * self.y

    def _get_sigma_sq_hat(self):
        return 1/(self.n-self.p)*(self.y.T*(np.identity(self.n) - self.H)) * self.y

    def _get_cme(self):
        return (1/(self.n - self.p))*(self.y.T*(np.identity(self.n) - self.H)*self.y)

    def _get_sce(self):
        return (self.y.T*(np.identity(self.n) - self.H)*self.y)

    def _get_scr(self):
        return self.y.T*(self.H - (1/self.n)*np.ones((self.n, self.n)))*self.y

    def _get_syy(self):
        return self.y.T*(np.identity(self.n) - (1/self.n)*np.ones((self.n, self.n)))*self.y

    def _get_cmr(self):
        return (1/(self.p - 1))*self.y.T*(self.H - (1/self.n)*np.ones((self.n, self.n)))*self.y

    def _get_f(self):
        return ((1/(self.p - 1))*self.y.T*(self.H - (1/self.n)*np.ones((self.n, self.n)))*self.y)/((1/(self.n - self.p))*(self.y.T*(np.identity(self.n) - self.H)*self.y))

    def _get_R(self):
        return self._get_scr()/self._get_syy()

    def IC_beta_hat(self, j, alpha):
        tval = stats.t.ppf(alpha/2, self.n-self.p)
        k = tval * (self.sigma_sq_hat * self.var_beta_hat[j][j])**0.5
        return self.beta_hat[j] - k, self.beta_hat[j] + k


        
