import warnings
from copy import deepcopy
from typing import Union, Optional, Callable, Tuple
import numpy as np
from numpy import ndarray
import scipy.linalg as sl

from numba import jit, float64, int64, bool_
import numba as nb
from glassobind import glasso
from sklearn.exceptions import ConvergenceWarning

from ..common.object import Numeric, RNG, Func
from ..common.model_interface import GraphicalLasso



class GraphicalURSC(GraphicalLasso):

    def __init__(self, delta: float=-1, hierarchical: bool=False, adaptive: bool=True,
                 lam_init: float=0, lam_min: float=1e-8, lam_max: float=1e8,
                 tol: float=1e-5, tol_inner: float=1e-5, n_iter: int=1000, n_iter_inner: int=10,
                 verbose: int=0, eps: float=1e-15):
        self.theta: ndarray = None
        self.sigma: ndarray = None
        self.lam: ndarray = None
        self.lam_init = lam_init
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.scale = 1
        self.tol = tol
        self.tol_inner = tol_inner
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.verbose = verbose
        self.eps = eps
        self.delta = delta
        self.hierarchical = hierarchical
        self.adaptive = adaptive
        self.code_length = None

    def fit(self, n: int, emp_cov: ndarray) -> 'GraphicalURSC':
        """
        Fit Graphica Lasso
        :param data: data (n_obs by n_features)
        :param weight: regularization weight (n_features by n_features)
        :return: self
        """
        if self.theta is None:
            self.theta = np.eye(emp_cov.shape[0])
        if self.sigma is None:
            self.sigma = np.eye(emp_cov.shape[0])
        if self.lam is None:
            if self.hierarchical:
                self.lam = np.ones(emp_cov.shape) * self.lam_init / n
            else:
                self.lam = np.ones(emp_cov.shape) * self.lam_init
            np.clip(self.lam, self.lam_min, self.lam_max, out=self.lam)

        C, self.scale = self.split_scale(emp_cov)
        # self.__eig_max = np.linalg.eigvalsh(C)[-1]
        self.__eig_max = np.max(np.sum(abs(C), axis=-1))
        # code_length = np.inf
        for i in range(self.n_iter):
            self._fit_theta_sigma(C, self.lam)

            crt_theta, crt_lam = self.stopping_criterion(n, C)
            self.code_length = self._code_length_per_n(n, C)

            if self.verbose > 0:
                print("iter: {},\t code length: {},\t crt: theta){}, lam){}"
                      .format(i, self.code_length, crt_theta, crt_lam))
            if crt_lam < self.tol:
                break

            self._fit_lam(n, self.theta)
        else:
            warnings.warn("fit: did not converge after {} iterations".format(self.n_iter), ConvergenceWarning)

        return self

    def split_scale(self, emp_cov: ndarray) -> Tuple[ndarray, ndarray]:
        d = np.sqrt(np.diag(emp_cov))
        C = (emp_cov / d).T / d
        C_sym = 0.5 * (C + C.T)
        return C_sym, d

    def _fit_theta_sigma(self, C: ndarray, lam: ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.theta, self.sigma = glasso_L2(
                C, self.theta, self.sigma, lam, self.n_iter_inner, self.verbose > 1, self.tol_inner
            )
        return self

    def _fit_lam(self, n: int, theta: ndarray):
        """
        Minimize 0.5 * n * lam * theta ** 2 + plog (1 / lam) / 2 + plog (plog (1 / lam)) + plog (plog (plog (1 / lam)))
        0.5 * lam * theta ** 2 + (0.5 + 1 / (1 + plog (1 / lam0)) * (1 + 2 / (1 + plog (plog (1 / lam0))))) * log (1 + 1 / lam)
        :param h:
        :param theta:
        :return:
        """
        if not self.adaptive:
            m = theta.shape[0] ** 2
        else:
            m = self.__eig_max ** 2 / 2
        if self.hierarchical:
            t = m * n * theta ** 2 / (1 + 2 / (1 + plog(m / self.lam)) * (1 + 2 / (1 + plog(plog(m / self.lam)))))
        else:
            t = m * n * theta ** 2 / (1 + 2 * (1 + self.delta) / (1 + plog(m / self.lam)))
        self.lam = 0.5 * m * (np.sqrt(1 + 4 / t) - 1)
        np.clip(self.lam, self.lam_min, self.lam_max, out=self.lam)

    def stopping_criterion(self, n, C) -> Tuple[float, float]:
        if not self.adaptive:
            m = self.theta.shape[0] ** 2
        else:
            m = self.__eig_max ** 2
        grad_theta = 0.5 * (C - self.sigma) + self.lam * self.theta
        if self.hierarchical:
            rlam = m / self.lam
            grad_lam = (
                0.5 * self.theta ** 2 +
                0.5 * (
                    1 +
                    2 / (1 + plog(rlam)) * (1 + 2 / (1 + plog(plog(rlam))))
                ) * (1 / (m + self.lam) - 1 / self.lam) / n
            )
        else:
            grad_lam = (
                    0.5 * self.theta ** 2 +
                    0.5 * (
                            1 +
                            2 * (1 + self.delta) / (1 + plog(m / self.lam))
                    ) * (1 / (m + self.lam) - 1 / self.lam) / n
            )
        boundary_lam = ((self.lam == self.lam_max) * (grad_lam < 0)) + ((self.lam == self.lam_min) * (grad_lam > 0))
        return np.max(abs(grad_theta)), np.max(abs(grad_lam) * ~boundary_lam)

    def _code_length_per_n(self, n: int, C: ndarray) -> float:
        if not self.adaptive:
            m = self.theta.shape[0] ** 2
        else:
            m = self.__eig_max ** 2
        fg: float = objective_glasso_L2(C, self.theta, self.lam)
        logZ: float = 0.5 * np.sum(plog(m / self.lam)) / n
        # norm: float = (1 + self.delta) * np.sum(plog(plog(1 / self.lam))) / n
        if self.hierarchical:
            norm: float = (
                np.sum(plog(plog(m / self.lam)))
                + 2 * np.sum(plog(plog(plog(m / self.lam))))
            ) / n
        else:
            norm: float = (
                (1 + self.delta) * np.sum(plog(plog(m / self.lam)))
            ) / n
        return fg + logZ + norm

    @property
    def result(self) -> ndarray:
        theta = (self.theta / self.scale).T / self.scale
        return theta


def plog(x):
    """
    Soft positive logarithm
    :param x:
    :return:
    """
    return np.log(1 + x)


@jit(nb.types.UniTuple(nb.float64[:, :], 2)(
    float64[:, :], float64[:, :], float64[:, :], float64[:, :], int64, bool_, float64
))
def glasso_L2(C: ndarray, theta: ndarray, sigma: ndarray, lam: ndarray, n_iter: int, verbose: bool, tol: float
              ) -> Tuple[ndarray, ndarray]:
    """
    Minimize 0.5 * (np.sum(theta * C) - np.log(np.det(theta))) + 0.5 * lam * theta ** 2

    :param C:
    :param theta:
    :param lam:
    :param n_iter:
    :param verbose:
    :param tol:
    :return:
    """

    C = sym(C)
    lam = sym(lam)
    crt = np.inf
    for i in range(n_iter):
        # compute gradient
        theta = sym(theta)
        sigma = sym(sigma)
        gradient = 0.5 * (C - sigma) + lam * theta

        # modify direction with diagonal hessian
        diag_hess = diag_hessian(sigma, lam)
        d_theta = sym(gradient / diag_hess)

        # check convergence
        crt = np.max(abs(d_theta))
        if crt < tol:
            break

        # initial guess of step size
        num = np.sum(gradient ** 2 / diag_hess)
        denom1 = 0.5 * np.sum((sigma @ d_theta @ sigma) * d_theta)
        denom2 = np.sum(lam * d_theta ** 2)
        t0 = num / (denom1 + denom2)
        if num == 0:
            t0 = 0

        # back tracking
        t = 1
        backs = 0
        theta0 = theta
        f0 = objective_glasso_L2(C, theta0, lam)
        f = f0 + 1
        while True:
            theta = theta0 - (t * t0) * d_theta
            f = objective_glasso_L2(C, theta, lam)
            if f <= f0:
                break
            t /= 2
            backs += 1
        sigma = np.linalg.inv(theta)

        if verbose:
            print("glasso_L2: iter: {},\t crt: {} at {} \trate: {} \tback track: {}\tnd: {}"
                  .format(i, crt, f, t0, backs, (num, denom1, denom2)))

    else:
        warnings.warn("glasso_L2: iteration limit reached: {}, residue: {}".format(n_iter, crt),
                      ConvergenceWarning)

    return theta, sigma


@jit(float64[:, :](float64[:, :], float64[:, :]))
def diag_hessian(sigma: ndarray, lam: ndarray) -> ndarray:
    m = sigma.shape[0]
    diagonal_sigma = np.diag(sigma)
    hessian = 0.5 * (np.outer(diagonal_sigma, diagonal_sigma) + sigma ** 2) + lam
    hessian.flat[::m + 1] = 0.5 * diagonal_sigma ** 2 + np.diag(lam)
    return hessian


@jit(float64[:, :](float64[:, :]))
def sym(a: ndarray) -> ndarray:
    return 0.5 * (a + a.T)


@jit(float64(float64[:, :], float64[:, :], float64[:, :]))
def objective_glasso_L2(C: ndarray, theta: ndarray, lam: ndarray) -> float:
    m = C.shape[1]
    eigs, _ = np.linalg.eig(sym(theta))
    if np.all(eigs >= 0):
        f = 0.5 * (np.sum(C * theta) - np.sum(np.log(eigs)) + m * np.log(2 * np.pi))
    else:
        f = np.inf
    g = 0.5 * np.sum(lam * theta ** 2)
    return f + g
