import numpy as np
from numpy import ndarray
from typing import List, Callable
from functools import partial, wraps
from itertools import chain

from toolz import functoolz as fz

from scipy import stats

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge
from sklearn.model_selection import ShuffleSplit, GridSearchCV, RandomizedSearchCV

from ..common.lasso import lasso_hetero, soft_threshold
from ..common.object import Numeric, Learner, RNG


class FlexibleRidge(BaseEstimator, Learner):

    def __init__(self, lam: ndarray=1e-2, fit_intercept=True):
        self.lam = lam
        self.fit_intercept = fit_intercept

        self.intercept_X_ = 0
        self.intercept_y_ = 0

    def fit(self, X: ndarray, y: ndarray) -> 'FlexibleRidge':

        if self.fit_intercept:
            self.intercept_X_ = np.mean(X, axis=0)
            X = X - self.intercept_X_
            self.intercept_y_ = np.mean(y, axis=0)
            y = y - self.intercept_y_

        n = len(y)
        C = X.T @ X / n
        b = X.T @ y / n

        self.beta_ = self.fit_beta(C, b, self.lam)
        self.sigma2_ = self.fit_sigma2(X, y, self.beta_, self.lam)
        return self

    @property
    def result(self) -> ndarray:
        return self.beta_

    def set_lam(self, lam: ndarray):
        self.lam = lam

    def predict(self, X):
        if self.fit_intercept:
            return (X - self.intercept_X_) @ self.beta_ + self.intercept_y_
        else:
            return X @ self.beta_

    @staticmethod
    def fit_beta(C: ndarray, b: ndarray, lam: ndarray) -> ndarray:
        beta: ndarray = np.linalg.solve(C + np.diag(lam), b)
        return beta

    @staticmethod
    def fit_sigma2(X: ndarray, y: ndarray, beta: ndarray, lam: ndarray) -> float:
        return np.mean((y - X @ beta) ** 2)


class RidgeCVProb(RidgeCV):  # XXX: duplicated

    @wraps(RidgeCV.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        RidgeCV.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)


class LassoCVProb(LassoCV):  # XXX: duplicated

    @wraps(LassoCV.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        LassoCV.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)


class RidgeProb(Ridge):

    def __init__(self, lam: float=1.0, fit_intercept=True):
        self.lam = lam
        super().__init__(alpha=lam, fit_intercept=fit_intercept)

    def fit(self, X, y):
        self.alpha = self.lam
        Ridge.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)


class LassoProb(Lasso):

    def __init__(self, lam: float=1.0, fit_intercept=True):
        self.lam = lam
        super().__init__(alpha=lam, fit_intercept=fit_intercept)

    def fit(self, X, y):
        self.alpha = self.lam
        Lasso.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)


class FlexibleRidgeRandomCV(BaseEstimator):

    def __init__(
            self, lam_min, lam_max,
            scoring=None, n_iter=100,
            cv=5, random_state=None,
            fit_intercept=True):
        if random_state is None:
            random_state = RNG(42)

        self.lam_min = lam_min
        self.lam_max = lam_max
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def fit(self, X: ndarray, y: ndarray) -> 'FlexibleRidgeRandomCV':

        n, m = X.shape
        self.clf_ = RandomizedSearchCV(
                estimator=FlexibleRidge(fit_intercept=self.fit_intercept),
                param_distributions={
                    "lam": LogUniformDist(low=self.lam_min, high=self.lam_max, shape=(m, m))
                    },
                scoring=self.scoring,
                n_iter=self.n_iter,
                cv=self.cv,
                random_state=self.random_state
                )
        self.clf_.fit(X, y)
        self.sigma2_ = self.clf_.best_estimator_.sigma2_
        # import pdb
        # pdb.set_trace()
        return self

    def predict(self, X):
        return self.clf_.predict(X)


class LogUniformDist:

    def __init__(self, low, high, shape, rng=None):
        if rng is None:
            rng = RNG(42)
        self.low = low
        self.high = high
        self.shape = shape
        self.rng = rng

    def rvs(self, random_state=None) -> ndarray:
        if random_state is None:
            random_state = self.rng
        logx = random_state.uniform(
                low=np.log(self.low), high=np.log(self.high), size=self.shape)
        return np.exp(logx)


class GridCV(BaseEstimator):

    def __init__(
            self, estimator: "has sigma2_ attribute", lam_min, lam_max,
            scoring=None, n_iter=100,
            cv=5):
        self.estimator = estimator
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv

    def fit(self, X: ndarray, y: ndarray) -> 'GridCV':

        grid = np.exp(
                np.linspace(np.log(self.lam_min), np.log(self.lam_max), self.n_iter)
                )
        self.clf_ = GridSearchCV(
                estimator=self.estimator,
                param_grid={"lam": grid},
                scoring=self.scoring,
                cv=self.cv,
                )
        self.clf_.fit(X, y)
        self.sigma2_ = self.clf_.best_estimator_.sigma2_
        # import pdb
        # pdb.set_trace()
        return self

    def predict(self, X):
        return self.clf_.predict(X)
