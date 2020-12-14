from itertools import count

import numpy as np
from numpy.core.multiarray import ndarray
import warnings
from sklearn.exceptions import ConvergenceWarning
from typing import Union, Tuple


def soft_threshold(x, lam):
    return np.clip(abs(x) - lam, 0, np.inf) * np.sign(x)


def lasso_obj(X: ndarray, y: ndarray, w: ndarray, lam: ndarray, sigma2: float=None) -> float:
    n = y.size
    r = y - X @ w
    if sigma2 is None:
        sigma2 = np.mean(r ** 2)
    f: float = 0.5 / sigma2 * np.sum(r ** 2) + np.sum(lam * abs(w)) + n / 2 * np.log(sigma2)
    return f


def lasso_hetero_gs(X: ndarray, y: ndarray, lam: ndarray, w_init: ndarray, tol: float,
                    verbose: bool=False) -> ndarray:
    n, m = X.shape
    scaler = np.sqrt(np.sum(X ** 2, axis=0))
    X = X / scaler
    w = w_init.copy()

    if m == 0:
        return w

    def subgradient(w, r):
        return -r @ X + lam * np.sign(w)

    r = 0
    for t in count():
        if t % 100 == 0:
            r = y - X @ w

        sg = subgradient(w, r)
        eff_lam = lam * (w == 0)
        abs_g = abs(soft_threshold(sg, eff_lam))
        i = np.argmax(abs_g)

        if verbose:
            print("lasso_hetero_gs: step: {}\tgrad: {}".format(t, abs_g[i]))
            if t % 100 == 99:
                import pdb
                pdb.set_trace()
        if abs_g[i] / np.fabs(w) < tol:
            break

        w_i_new = soft_threshold(w[i] + X[:, i] @ r, lam[i])
        r += (w[i] - w_i_new) * X[:, i]
        w[i] = w_i_new
    return w * scaler


def lasso_hetero(X: ndarray, y: ndarray, lam: ndarray, w_init: ndarray, tol: float, return_obj: bool=False,
                 copy_X: bool=True, max_iter: int=100, verbose: bool=False) -> Union[ndarray, Tuple[ndarray, float]]:
    n, m = X.shape
    scale = np.sqrt(np.sum(X ** 2, axis=0))
    if copy_X:
        X = X / scale
    else:
        X /= scale

    lam = lam / scale

    w = w_init.copy()

    if m == 0:
        return w

    r = y - X @ w
    gap = np.inf
    for t in range(max_iter):
        w_max = 0
        d_w_max = 0

        for i in range(m):
            w_i_new = soft_threshold(w[i] + X[:, i] @ r, lam[i])
            d_w = w[i] - w_i_new
            r += d_w * X[:, i]
            w[i] = w_i_new

            w_max = max(w_max, np.fabs(w_i_new))
            d_w_max = max(d_w_max, np.fabs(d_w))

        gap = d_w_max / (w_max + 1e-16)

        if verbose:
            print("lasso_hetero_gs: step: {}\tgrad: {}".format(t, gap))

        if gap < tol:
            break
    else:
        warnings.warn("not converged after {} iterations; gap: {}".format(max_iter, gap), ConvergenceWarning)

    ww: ndarray = w / scale
    if return_obj:
        obj: float = 0.5 * np.sum(r) + np.sum(lam * np.fabs(ww))
        return ww, obj
    else:
        return ww
