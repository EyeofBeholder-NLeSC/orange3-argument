"""Module of helper classes."""

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Collector:
    """Class to collect intermediate results of approximation steps."""

    def __init__(self, data: np.ndarray):
        self._data = np.array([data])

    @property
    def data(self):
        """The data collection.

        Returns:
            _type_: _description_
        """
        return self._data.T

    def collect(self, new_data: np.ndarray):
        """Collect new data and add to the existing data queue.

        Args:
            new_data (np.ndarray): _description_.
        """
        self._data = np.vstack([self._data, new_data])

    def plot(self):
        """Plot the data and return the figure object.

        Returns:
            _type_: _description_
        """
        data = self._data.T
        num_argu, num_steps = data.shape
        fig, ax = plt.subplots()
        plt.ylim(0, 1.1)

        x = range(num_steps)
        for i in range(num_argu):
            y = data[i]
            ax.plot(x, y, label=i)
        ax.set_xlabel("Step")
        ax.set_ylabel("Strength")
        ax.legend()

        return fig


class Adaptor:
    """_summary_"""

    def __init__(
        self,
        arguments: pd.DataFrame,
        weight_col: str,
        attacks: pd.DataFrame = None,
        supports: pd.DataFrame = None,
    ):
        self.weight_col = weight_col
        self.arguments = arguments
        self.attacks = attacks
        self.supports = supports

    @property
    def arguments(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._arguments

    @arguments.setter
    def arguments(self, value):
        self.validate(value, [self.weight_col])
        self._arguments = value

    @property
    def attacks(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._attacks

    @attacks.setter
    def attacks(self, value):
        if value is not None:
            self.validate(value, ["source", "target"])
        self._attacks = value

    @property
    def supports(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._supports

    @supports.setter
    def supports(self, value):
        if value is not None:
            self.validate(value, ["source", "target"])
        self._supports = value

    @staticmethod
    def validate(data: pd.DataFrame, columns: list):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            columns (list): _description_

        Raises:
            ValueError: _description_
        """
        if not set(columns).issubset(data.columns):
            raise ValueError(f"One or more columns in {columns} missing.")

    def get_weights(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        weights = self._arguments["coherence"].to_numpy(dtype=float)
        return deepcopy(weights)

    def get_parent_vectors(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        num_argu = len(self._arguments)
        result = np.zeros((num_argu, num_argu))

        def add_attack(r):
            result[r["target"], r["source"]] = -1

        def add_support(r):
            result[r["target"], r["source"]] = 1

        if self._attacks is not None:
            self._attacks.apply(add_attack, axis=1)
        if self._supports is not None:
            self._supports.apply(add_support, axis=1)

        return result
