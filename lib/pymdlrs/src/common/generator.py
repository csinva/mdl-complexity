""" Synthetic data generator """


from abc import ABCMeta, abstractmethod
from typing import Iterable, Any, List, Union, Tuple, Dict, Optional

import numpy as np
from numpy import ndarray
from .helper import EquipRNG


RNG = np.random.RandomState


class Generator(EquipRNG, metaclass=ABCMeta):

    @abstractmethod
    def generate(self) -> Any:
        pass


class Gaussian(Generator):

    def __init__(self, n: int, dim: int, loc: ndarray = None, scale2: ndarray = None,
                 rng: RNG = None):
        """
        Generator of (multivariate) Gaussian distribution
        :param n: length of data
        :param dim: number of variables
        :param loc: location parameter (shape == `dim`)
        :param scale2: squared scale parameter (shape == `(dim, dim)`)
        """
        EquipRNG.__init__(self, rng)
        if loc is None:
            loc = np.zeros(shape=dim)
        if scale2 is None:
            scale2 = np.eye(dim)
        self.n = n
        self.dim = dim
        self.loc = loc
        self.scale2 = scale2

    def generate(self):
        return self.rng.multivariate_normal(self.loc, self.scale2, self.n)


class GaussianCorrelation(Gaussian):

    def __init__(self, n: int, dim: int, loc: ndarray = None, scale2: ndarray = None,
                 normalize: bool=True, rng: RNG = None):
        super().__init__(n, dim, loc, scale2, rng)
        self.normalize = normalize

    def generate(self):
        self._X = Gaussian.generate(self)
        ret = {"n": self.n, "correlation": self._correlation(self._X)}
        return ret

    def _correlation(self, X):
        S = np.cov(X, rowvar=False)
        if not self.normalize:
            return S

        d = 1 / np.sqrt(np.diag(S)).reshape(-1, 1)
        C = S * d * d.T
        return C

