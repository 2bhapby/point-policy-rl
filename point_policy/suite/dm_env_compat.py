from enum import IntEnum
from typing import Any, NamedTuple

import numpy as np


class StepType(IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class Array:
    def __init__(self, shape, dtype, name="array"):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.name = name


class BoundedArray(Array):
    def __init__(self, shape, dtype, minimum, maximum, name="bounded_array"):
        super().__init__(shape=shape, dtype=dtype, name=name)
        self.minimum = minimum
        self.maximum = maximum


class _Specs:
    Array = Array
    BoundedArray = BoundedArray


class Environment:
    pass


class _DMEnvModule:
    Environment = Environment


dm_env = _DMEnvModule()
specs = _Specs()

