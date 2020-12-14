from itertools import count
import numpy as np
from numpy import ndarray
import scipy.linalg as linalg
from typing import Union, Tuple
import warnings

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge

from ..common.lasso import lasso_hetero, soft_threshold
from ..common.object import Numeric, Learner


class RidgeULNML(Learner, BaseEstimator):

    def __init__(self, lam_min: float=1e-8, lam_max: float=1e8,
                 n_iter: int=1000, fit_intercept=True, eps: float=1e-5, verbose=False):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.eps = eps
        self.verbose = verbose

        self.intercept_X_ = 0
        self.intercept_y_ = 0
        self.beta_ = None
        self.sigma2_ = None
        self.lam_ = None

    def fit(self, X: ndarray, y: ndarray) -> 'RidgeULNML':

        if self.fit_intercept:
            self.intercept_X_ = np.mean(X, axis=0)
            X = X - self.intercept_X_
            self.intercept_y_ = np.mean(y, axis=0)
            y = y - self.intercept_y_

        else:
            X = np.concatenate([X, np.ones([len(y), 1])], axis=1)

        n = len(y)
        C = X.T @ X / n
        b = X.T @ y / n

        if self.lam_ is None:
            self.lam_ = np.ones(X.shape[1]) * 1  # self.lam_min

        self.call_before_fit(X, y)

        for i in range(self.n_iter):
            self.beta_ = self.fit_beta(C, b, self.lam_)
            self.sigma2_ = self.fit_sigma2(X, y, self.beta_, self.lam_)
            self.lam_ = self.fit_lam(
                n, C, self.beta_, self.sigma2_, self.lam_, lam_bound=(self.lam_min, self.lam_max)
            )

            grad = self.gradient(n, C, b, X, y, self.beta_, self.sigma2_, self.lam_,
                                 lam_bound=(self.lam_min, self.lam_max))

            if self.verbose:
                obj = self.objective(C, X, y, self.beta_, self.sigma2_, self.lam_)
                g_sum = [np.max(np.abs(g)) for g in grad]
                print("iters: {}, obj: {}, grad: {}".format(i, obj, g_sum))

            if np.max([np.max(abs(g)) for g in grad]) < self.eps:
                break

        else:
            warnings.warn("{}: did not converge after {} iterations".format(self.__class__.__name__, self.n_iter),
                          ConvergenceWarning)
        return self

    def call_before_fit(self, X, y):
        pass

    @property
    def result(self) -> ndarray:
        return self.beta_

    def set_lam(self, lam: ndarray):
        self.lam_ = lam

    def predict(self, X):
        if self.fit_intercept:
            return (X - self.intercept_X_) @ self.beta_ + self.intercept_y_
        else:
            X = np.concatenate([X, np.ones([len(X), 1])], axis=1)
            return X @ self.beta_

    @staticmethod
    def objective(C: ndarray, X: ndarray, y: ndarray, beta: ndarray, sigma2: float, lam: ndarray):
        n = len(y)
        fg = 0.5 / sigma2 * (np.mean((y - X @ beta) ** 2) + np.sum(lam * beta ** 2)) + 0.5 * np.log(2 * np.pi * sigma2)
        logZ = 0.5 * (np.log(linalg.det(C + np.diag(lam)) - np.sum(np.log(lam)))) / n
        return fg + logZ

    @staticmethod
    def gradient(n: int, C: ndarray, b: ndarray, X: ndarray, y: ndarray,
                 beta: ndarray, sigma2: float, lam: ndarray, lam_bound: Tuple[float, float]):
        dbeta = 1 / sigma2 * (C @ beta - b + lam * beta)
        dsigma2 = -0.5 / sigma2 ** 2 * (np.mean((y - X @ beta) ** 2) + np.sum(lam * beta ** 2)) + 0.5 / sigma2
        dlam = 0.5 * beta ** 2 / sigma2 + 0.5 * np.diag(linalg.inv(C + np.diag(lam))) / n - 0.5 / lam / n
        at_boundary = ((lam == lam_bound[0]) * (dlam > 0)) + ((lam == lam_bound[1]) * (dlam < 0))
        return dbeta, dsigma2, dlam * ~at_boundary

    @staticmethod
    def fit_beta(C: ndarray, b: ndarray, lam: ndarray) -> ndarray:
        beta: ndarray = linalg.solve(C + np.diag(lam), b)
        return beta

    @staticmethod
    def fit_sigma2(X: ndarray, y: ndarray, beta: ndarray, lam: ndarray) -> float:
        return np.mean((y - X @ beta) ** 2) + np.sum(lam * beta ** 2)

    @staticmethod
    def fit_lam(n: int, C: ndarray, beta: ndarray, sigma2: float, lam0: ndarray, lam_bound: Tuple[float, float]
                ) -> ndarray:
        lam_min, lam_max = lam_bound
        b2hat = beta ** 2 / sigma2
        diag_H_inv = np.diag(linalg.inv(C + np.diag(lam0)))
        lam1 = 1 / np.clip(n * b2hat + diag_H_inv, 1 / lam_max, 1 / lam_min)
        return lam1
