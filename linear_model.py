from os import stat
from numpy.core.numeric import identity
from pandas.core.frame import DataFrame
from scipy import matrix, stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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

    def residual_summary(self, plot=True):
        res = sorted([x[0, 0] for x in self.get_residuals()])
        n = len(res)
        median = res[int((n-1)/2)] if n % 2 else (res[int(n/2)
                                                      ]+res[int(n/2)-1]) / 2
        print(f"sum: {sum(res)}, mean: {sum(res)/n}")
        if plot:
            _, ax = plt.subplots()
            ax.boxplot(res)
            ax.set_title('Residuals')
            plt.show()

            _, ax = plt.subplots()
            ax.scatter(self._get_y_hat().getA1(), res)
            ax.set_ylabel('Residuals')
            ax.set_xlabel(self.y_col)
            ax.set_title(f'Estimated {self.y_col} vs. Residuals')
            ax.grid(True)
            plt.show()

        return (res[0], median, res[-1])

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
        self.Syy = self._get_syy()
        self.SCR = self._get_scr()
        self.SCE = self._get_sce()
        self.CMR = self._get_cmr()
        self.CME = self._get_cme()

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
        return self.SCR/self.Syy

    def _get_R_ajustado(self):
        return 1 - (self.CME*(self.n-1)/self.Syy)

    def IC_beta_hat(self, j, alpha):
        tval = abs(stats.t.ppf(alpha/2, self.n-self.p))
        k = tval * (self.sigma_sq_hat * self.var_beta_hat[j, j])**0.5
        print(self.beta_hat[j, 0], tval,
              self.sigma_sq_hat, self.var_beta_hat[j, j])
        return self.beta_hat[j, 0] - k, self.beta_hat[j, 0] + k

    def anova(self):
        fval, gl1, gl2, pval = self._get_f_statistic()
        df = pd.DataFrame({
            'Suma de cuadrados': [self.SCR, self.SCE, self.Syy],
            'Grados de libertad': [gl1, gl2, self.n-1],
            'Cuadrados medios': [self.CMR, self.CME, ''],
            'F': [fval, '', ''],
            'p-value': [pval, '', '']
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
        xdf = pd.DataFrame(self.df[self.cols])
        _ = pd.plotting.scatter_matrix(xdf, c=self.df[self.y_col])
        _ = pd.plotting.scatter_matrix(self.df)

    def scatter(self, fit=True):
        for col in self.cols:
            fig, ax = plt.subplots()
            x, y = self.df[col], self.df[self.y_col]
            ax.scatter(x, y)
            ax.set_title(f'{col} vs. {self.y_col}')
            ax.set_xlabel(col)
            ax.set_ylabel(self.y_col)
            ax.grid(True)
            if fit:
                fn = np.poly1d(np.polyfit(x, y, 1))
                plt.plot([min(x), max(x)], fn([min(x), max(x)]), 'g--')
            plt.show()
