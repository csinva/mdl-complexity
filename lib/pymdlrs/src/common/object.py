from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Any

import numpy as np


# type alias
Numeric = Union[float, np.ndarray]
RNG = np.random.RandomState


def Func(*args):
    def Return(ret=None):
        return Callable[list(args), ret]
    return Return


class Learner(metaclass=ABCMeta):

    @property
    @abstractmethod
    def result(self) -> Any:
        pass
