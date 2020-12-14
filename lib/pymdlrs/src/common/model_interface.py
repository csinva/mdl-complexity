from .object import Learner
from numpy import ndarray
from abc import abstractmethod, ABCMeta


class GraphicalLasso(Learner, metaclass=ABCMeta):

    @property
    @abstractmethod
    def result(self) -> ndarray:
        pass

    @abstractmethod
    def fit(self, n: int, emp_cov: ndarray) -> 'GraphicalLasso':
        pass
