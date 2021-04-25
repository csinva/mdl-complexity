import numpy as np
from numpy import ndarray
from typing import Any, Tuple, Callable
import warnings
from abc import ABCMeta, abstractmethod

from sklearn.exceptions import ConvergenceWarning
from glassobind import glasso

from ..common.object import Learner
from ..common.helper import EquipCallbackOnSelf
from ..common.model_interface import GraphicalLasso


class GraphicalLassoXIC(GraphicalLasso, EquipCallbackOnSelf, metaclass=ABCMeta):

    def __init__(self, rho_min: float=None, rho_max: float=None,
                 tol: float=1e-8, eps: float=1e-8, n_iter: int=100,
                 verbose: bool=False, callback: Callable=None):
        EquipCallbackOnSelf.__init__(self, callback)
        self.tol = tol
        self.eps = eps
        self.n_iter = n_iter
        self.verbose = verbose
        if rho_min is None:
            rho_min = 1e-15
        self.rho_min = rho_min
        if rho_max is None:
            rho_max = 1
        self.rho_max = rho_max
        self._criteria = []
        self.theta = None
        self.sigma = None
        self.rho = None
        self._criterion = None

    def clear(self):
        self._criteria = []

    def fit(self, n: int, corr: ndarray) -> 'GraphicalLassoXIC':
        rho_min = max(1e-15, self.rho_min)
        rho_max = self.rho_max
        num_grid = 1 + 4 * int(np.log2(rho_max) - np.log2(rho_min))
        rhos = np.exp(np.linspace(start=np.log(rho_min), stop=np.log(rho_max), num=num_grid))

        self._criteria = []
        theta = sigma = np.eye(corr.shape[1])
        self._criterion = np.inf
        for rho in (list(rhos))[::-1]:  # in the ascending order of rho for computational efficiency
            rho = float(rho)
            sigma, theta = self.fit_glasso(corr, rho, sigma, theta)
            crt = self.compute_criterion(n, corr, theta, sigma)
            if crt <= self._criterion:
                self.rho = rho
                self.theta = theta.copy()
                self.sigma = sigma.copy()
                self._criterion = crt
            self._criteria.append((crt, rho))
            self.hook_callback_on_self()
        return self

    def fit_glasso(self, emp_corr: ndarray, rho: float, sigma: ndarray, theta: ndarray):
        m = emp_corr.shape[0]
        gamma = rho * np.ones_like(emp_corr)
        res = glasso(emp_corr, gamma, sigma, theta, self.tol, self.n_iter, self.verbose, self.eps)
        if not res.converged:
            warnings.warn('fit_glasso: did not converge after %i iteration:'
                          ' dual gap: %.3e' % (self.n_iter, res.gap),
                          ConvergenceWarning)
        return res.sigma, res.theta

    @abstractmethod
    def compute_criterion(self, n: int, corr: ndarray, theta: ndarray, sigma: ndarray):
        pass

    def loglikelihood(self, n: int, corr: ndarray, theta: ndarray):
        m = theta.shape[1]
        eigvals = np.linalg.eigvalsh(theta)
        logdet_theta = np.sum(np.log(eigvals))
        if np.all(eigvals > 0):
            return n / 2 * (-np.sum(corr * theta) + logdet_theta - m * np.log(2 * np.pi))
        else:
            return -np.inf

    @property
    def result(self) -> ndarray:
        return self.theta


class GraphicalLassoEBIC(GraphicalLassoXIC):

    def __init__(self, gamma: float, rho_min: float = None, rho_max: float = None, tol: float = 1e-8, eps: float = 1e-8,
                 n_iter: int = 100, verbose: bool = False, callback: Callable = None):
        GraphicalLassoXIC.__init__(self, rho_min, rho_max, tol, eps, n_iter, verbose, callback)
        self.gamma = gamma

    def compute_criterion(self, n: int, corr: ndarray, theta: ndarray, sigma: ndarray):
        ll = self.loglikelihood(n, corr, theta)
        m = theta.shape[1]
        edge_size = (np.sum(theta != 0) - m) // 2
        return -2 * ll + edge_size * np.log(n) + 4 * edge_size * self.gamma * np.log(m)


class GraphicalLassoAIC(GraphicalLassoXIC):

    def compute_criterion(self, n: int, corr: ndarray, theta: ndarray, sigma: ndarray):
        ll = self.loglikelihood(n, corr, theta)
        m = theta.shape[1]
        edge_size = (np.sum(theta != 0) - m) // 2
        return -2 * ll + 2 * edge_size
