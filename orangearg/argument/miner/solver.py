"""The solver module.
"""
from copy import deepcopy
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
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

        x = range(num_steps)
        for i in range(num_argu):
            y = data[i]
            ax.plot(x, y, label=i)
        ax.set_xlabel("Step")
        ax.set_ylabel("Strength")
        ax.legend()

        return fig


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

    def __init__(self, step_size: float, max_iter: int, data_adaptor: Adaptor):
        self._weights = data_adaptor.compute_weights()
        self._parent_vectors = data_adaptor.compute_parent_vectors()
        self._strength_vector = deepcopy(self.weights)
        self.step_size = step_size
        self.max_iter = max_iter

    @property
    def parent_vectors(self):
        return self._parent_vectors

    @property
    def strength_vector(self):
        return self._strength_vector

    @property
    def weights(self):
        return self._weights

    @abstractmethod
    def aggregate(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        """Aggregation function.

        Args:
            parent_vector (np.ndarray): parameter vector of an argument.
            strength_vector (np.ndarray): strength vector of all arguments.
        """

    @abstractmethod
    def influence(
        self, aggreg_strength: float, weight: float, p: int = 2, k: float = 1
    ):
        """Influence function.

        Args:
            aggreg_strength (float): _description_
            weight (float): _description_
            p (int, optional): Conservativeness parameter, available for Linear and p-Max kernel. Defaults to 2.
            k (float, optional): Power parameter, only available for p-Max kernel. Defaults to 1.
        """

    def compute_delta(self):
        """Compute increment of strength vector in each step.

        Returns:
            _type_: _description_
        """
        new_strengths = []
        for i in range(self._parent_vectors.shape[0]):
            aggreg_strength = self.aggregate(
                parent_vector=self._parent_vectors[i],
                strength_vector=self._strength_vector,
            )
            new_strength = self.influence(
                aggreg_strength=aggreg_strength, weight=self._weights[i]
            )
            new_strengths.append(new_strength)
        return np.array(new_strengths) - self._strength_vector

    @abstractmethod
    def approximate(
        self,
        parent_vectors: np.ndarray,
        strength_vector: np.ndarray,
        weights: np.ndarray,
        step_size: float,
        max_iter: int,
        data_collector: Collector = None,
    ) -> tuple[int, np.ndarray]:
        """Approximation function.

        Args:
            parent_vectors (np.ndarray): All the parent vectors.
            strength_vector (np.ndarray): The strength vector of all arguments.
            weights (np.ndarray): The weight vector of all arguments.
            step_size (float): Step size of the approximation process.
            max_iter (int): Maximum number of steps.
            data_collector (Collector): Data collector that keeps strength vectors in all steps.

        Returns:
            tuple[int, np.ndarray]: Index of the final step and the convergence result.
        """


def aggreg_sum(parent_vector: np.ndarray, strength_vector: np.ndarray) -> np.ndarray:
    """Sum aggregation function.

    Args:
        parent_vector (np.ndarray): _description_
        strength_vector (np.ndarray): _description_

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if parent_vector.size != strength_vector.size:
        raise ValueError(
            f"Size of input vectors doesn't match: {parent_vector.size}, {strength_vector.size}."
        )
    return parent_vector @ strength_vector


def infl_pmax(aggreg_strength: float, weight: float, p: int, k: float) -> float:
    """PMax influence function.

    Args:
        aggreg_strength (float): _description_
        weight (float): _description_
        p (int): _description_
        k (float): _description_

    Returns:
        float: _description_
    """

    def h(x):
        return max(0, x) ** p / (1 + max(0, x) ** p)

    return weight * (1 - h(-aggreg_strength / k) + h(aggreg_strength / k))


# TODO: think about how to integrate the compute_delta function.
def appr_rk4(
    parent_vectors: np.ndarray,
    strength_vector: np.ndarray,
    weights: np.ndarray,
    step_size: float,
    max_iter: int,
    data_collector: Collector = None,
) -> tuple[int, np.ndarray]:
    pass
