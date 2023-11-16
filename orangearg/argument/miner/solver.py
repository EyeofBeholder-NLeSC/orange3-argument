"""The solver module.
"""


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Collector:
    """Class to collect intermediate results of approximation steps."""

    def __init__(self, data: np.ndarray):
        self._data = np.array([data])

    def collect(self, new_data=np.ndarray):
        self._data = np.vstack([self._data, new_data])

    def output(self):
        return self._data.T

    def plot(self):
        pass


class Adaptor:
    """Class to transfer the output of the existing argument framework into the input data format of solver."""

    def __init__(
        self,
        arguments: pd.DataFrame,
        attacks: pd.DataFrame = None,
        supports: pd.DataFrame = None,
    ):
        self.arguments = arguments
        self.attacks = attacks
        self.supports = supports

    @property
    def arguments(self):
        return self._arguments

    @arguments.setter
    def arguments(self, value):
        self.validate(value, ["coherence"])
        self._arguments = value

    @property
    def attacks(self):
        return self._attacks

    @attacks.setter
    def attacks(self, value):
        if value is not None:
            self.validate(value, ["source", "target"])
        self._attacks = value

    @property
    def supports(self):
        return self._supports

    @supports.setter
    def supports(self, value):
        if value is not None:
            self.validate(value, ["source", "target"])
        self._supports = value

    @staticmethod
    def validate(data: pd.DataFrame, columns: list):
        if not set(columns).issubset(data.columns):
            raise ValueError(f"One or more columns in {columns} missing.")

    def compute_weights(self) -> np.ndarray:
        return self._arguments["coherence"].to_numpy(dtype=float)

    def compute_parent_vectors(self) -> np.ndarray:
        num_argu = len(self._arguments)
        result = np.zeros((num_argu, num_argu))

        def add_attack(r):
            result[r["target"], r["source"]] = -1

        def add_support(r):
            result[r["target"], r["source"]] = 1

        if self._attacks is not None:
            self._attacks.apply(lambda r: add_attack(r), axis=1)
        if self._supports is not None:
            self._supports.apply(lambda r: add_support(r), axis=1)

        return result


class Solver(ABC):
    """Solver class to learn strength of arguments from their attacking/supporting graph."""

    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def influence(self):
        pass

    def compute_delta(self):
        pass

    @abstractmethod
    def approximate(self):
        pass
