from os import stat
from numpy.core.numeric import identity
from pandas.core.frame import DataFrame
from scipy import matrix, stats
import numpy as np
import pandas as pd


class LM:
    def __init__(self, df, y_col):
        self.df = df.copy()
        self.y_col = y_col
        self.y = matrix(df[y_col]).T

        self.cols = list(filter(lambda x: x != y_col, df.columns))
        self.n = len(df)

        # set self.explain, self.p, self.H and self.X
        self.update(self.cols.copy())

    @staticmethod
    def from_matrix(X, y):
        ydf = pd.DataFrame({'y': y})
        xdf = pd.DataFrame(X)
        return LM(ydf.join(xdf), 'y')

    def predict(self, xf):
        return xf.T*self.beta_hat

    def get_residuals(self):
        return self.y - self.X*self.beta_hat

    def residual_summary(self):
        r = sorted([x[0, 0] for x in self.get_residuals()])
        n = len(r)
        median = r[int((n-1)/2)] if n % 2 else (r[int(n/2)]+r[int(n/2)-1]) / 2
        print(f"sum: {sum(r)}, mean: {sum(r)/n}")
        return (r[0], median, r[-1])

    def update(self, explain):
        assert all([x in self.cols for x in explain])
        self.explain = explain
        self.p = len(self.explain) + 1
        self.X = matrix(self.df[self.explain])
        self.X = np.concatenate(
            [matrix(np.ones(len(self.X))).T, self.X], axis=1)

        self.H = self.X*((self.X.T*self.X)**-1)*self.X.T
        self.beta_hat = self._get_beta_hat()
        self.sigma_sq_hat = self._get_sigma_sq_hat()
        self.residual_std_error = self.sigma_sq_hat**0.5
        self.var_beta_hat = self._get_var_beta_hat()

    def _get_beta_hat(self):
        return (self.X.T*self.X)**-1 * self.X.T * self.y

    def _get_var_beta_hat(self):
        return self.sigma_sq_hat * (self.X.T*self.X)**-1

    def _get_y_hat(self):
        return self.X*((self.X.T*self.X)**-1) * self.X.T * self.y

    def _get_sigma_sq_hat(self):
        return 1/(self.n-self.p)*(self.y.T*(np.identity(self.n) - self.H) * self.y)[0, 0]

    def _get_syy(self):
        return (self.y.T*(np.identity(self.n) - (1/self.n)*np.ones((self.n, self.n)))*self.y)[0, 0]

    def _get_sce(self):
        return (self.y.T*(np.identity(self.n) - self.H)*self.y)[0, 0]

    def _get_scr(self):
        return (self.y.T*(self.H - (1/self.n)*np.ones((self.n, self.n)))*self.y)[0, 0]

    def _get_cme(self):
        return self._get_sce()/(self.n - self.p)

    def _get_cmr(self):
        return self._get_scr()/(self.p - 1)

    def _get_f(self):
        return self._get_cmr() / self._get_cme()

    def _get_f_statistic(self):
        gl1 = self.p-1
        gl2 = self.n - self.p
        fval = self._get_f()
        pval = 1-stats.f.cdf(fval, gl1, gl2)
        return fval, gl1, gl2, pval

    def _get_R(self):
        return self._get_scr()/self._get_syy()

    def _get_R_ajustado(self):
        return 1 - (self._get_cme()*(self.n-1)/self._get_syy())

    def IC_beta_hat(self, j, alpha):
        tval = stats.t.ppf(alpha/2, self.n-self.p)
        k = tval * (self.sigma_sq_hat * self.var_beta_hat[j, j])**0.5
        print(self.beta_hat[j, 0], tval,
              self.sigma_sq_hat, self.var_beta_hat[j, j])
        return self.beta_hat[j, 0] - k, self.beta_hat[j, 0] + k

    def anova(self):
        fval, gl1, gl2, pval = self._get_f_statistic()
        df = pd.DataFrame({
            'Suma de cuadrados': [self._get_scr(), self._get_sce(), self._get_syy()],
            'Grados de libertad': [gl1, gl2, self.n-1],
            'Cuadrados medios': [self._get_cmr(), self._get_cme(), None],
            'F': [fval, None, None],
            'p-value': [pval, None, None]
        })
        df.index = ['Regresión', 'Residual', 'Total']
        return df

    def summary(self):
        estimate = [b[0, 0] for b in self.beta_hat]
        std_error = [self.var_beta_hat[i, i]**0.5 for i in range(self.p)]
        tvals = [e/s for e, s in zip(estimate, std_error)]
        pr = [2*stats.t.cdf(-abs(x), self.n-self.p) for x in tvals]

        df = pd.DataFrame({
            'Estimate': estimate,
            'Std. Error': std_error,
            't value': tvals,
            'Pr(>|t|)': pr
        })
        df.index = ['(Intercept)'] + self.explain
        print(df)
        print('\n\n')

        print(
            f'Residual standard error: {self.residual_std_error} on {self.n-self.p} degrees of freedom')
        print(
            f'Multiple R-squared: {self._get_R()},  Adjusted R-squared: {self._get_R_ajustado()}')
        fval, gl1, gl2, pval = self._get_f_statistic()
        print(f'F-statistic: {fval} on {gl1} and {gl2} DF, p-value: {pval}')

    def inicial(self):
        ydf = pd.DataFrame(self.df[self.y_col])
        xdf = pd.DataFrame(self.df[self.cols])
        _ = pd.plotting.scatter_matrix(xdf, c=ydf['decibels'])
        _ = pd.plotting.scatter_matrix(self.df)
