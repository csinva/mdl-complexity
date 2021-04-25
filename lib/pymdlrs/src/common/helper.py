from typing import Union

import numpy as np
from contextlib import contextmanager


class EquipRNG:

    def __init__(self, rng: Union[np.random.RandomState, int]=None):
        """
        :param rng: PRNG
        """
        if rng is None:
            rng = np.random.RandomState(123)
        elif isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.rng = rng

    def seed_rng(self, val: int):
        """ Seed PRNG """
        self.rng.seed(val)


class EquipCallbackOnSelf:

    def __init__(self, callback=None):
        self.callback = callback

    def hook_callback_on_self(self):
        if self.callback is not None:
            self.callback(self)
