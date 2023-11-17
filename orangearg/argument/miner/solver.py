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

    def compute_weights(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        weights = self._arguments["coherence"].to_numpy(dtype=float)
        return deepcopy(weights)

    def compute_parent_vectors(self) -> np.ndarray:
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


class Solver(ABC):
    """Solver class to learn strength of arguments from their attacking/supporting graph."""

    approximator_options = ["RK4"]
    initial_strength_options = ["weight", "uniform"]

    def __init__(
        self,
        step_size: float,
        max_iter: int,
        epsilon: float,
        data_adaptor: Adaptor,
        init_method: str = "weight",
    ):
        self._weights = data_adaptor.compute_weights()
        self._parent_vectors = data_adaptor.compute_parent_vectors()
        self._strength_vector = None
        self.step_size = step_size
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.init_strength(init_method=init_method)

    def init_strength(self, init_method: str = "weight"):
        """Initial strength vector."""
        if init_method == "weight":
            self._strength_vector = deepcopy(self._weights)
        elif init_method == "uniform":
            self._strength_vector = 0.5 * np.ones(len(self._weights))
        else:
            raise ValueError(
                f"Method of initializing strength should be one of {self.initial_strength_options}, but {init_method} is given."
            )

    @property
    def parent_vectors(self):
        """Parent vectors of arguments.

        Returns:
            _type_: _description_
        """
        return self._parent_vectors

    @property
    def strength_vector(self):
        """Strength vector of arguments.

        Returns:
            _type_: _description_
        """
        return self._strength_vector

    @property
    def weights(self):
        """Weights of arguments.

        Returns:
            _type_: _description_
        """
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

    def compute_delta(self, strength_vector: np.ndarray):
        """Compute increment of strength vector in each step.

        Args:
            strength_vector (np.ndarray): the strength vector at which the function will compute increment.

        Returns:
            _type_: _description_
        """
        new_strengths = []
        for i in range(self._parent_vectors.shape[0]):
            aggreg_strength = self.aggregate(
                parent_vector=self._parent_vectors[i],
                strength_vector=strength_vector,
            )
            new_strength = self.influence(
                aggreg_strength=aggreg_strength, weight=self._weights[i]
            )
            new_strengths.append(new_strength)
        return np.array(new_strengths) - self._strength_vector

    def solve(
        self,
        approximator: str,
        collect_data: bool = False,
    ) -> tuple[int, Collector | None]:
        """_summary_

        Args:
            approximator (str): _description_
            collect_data (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            int: _description_
        """
        data_collector = Collector(data=self._strength_vector) if collect_data else None

        if approximator == "RK4":
            appr_func = self._appr_rk4
        else:
            raise ValueError(
                f"Approximator should be one of {self.approximator_options}, but {approximator} is given."
            )

        for step in range(int(self.max_iter)):
            derivative = appr_func()
            self._strength_vector += derivative

            # pylint: disable=W0106
            data_collector and data_collector.collect(self._strength_vector)
            if abs(derivative).max() < self.epsilon:
                break

        return step, data_collector

    def _aggreg_sum(
        self, parent_vector: np.ndarray, strength_vector: np.ndarray
    ) -> np.ndarray:
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

    def _infl_pmax(
        self, aggreg_strength: float, weight: float, p: int, k: float
    ) -> float:
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

    def _appr_rk4(self) -> np.ndarray:
        """Runge-Kutta order-4 approximator.

        Returns:
            np.ndarray: _description_
        """
        k1 = self.compute_delta(strength_vector=self._strength_vector)
        k2 = self.compute_delta(
            strength_vector=self._strength_vector + 0.5 * self.step_size * k1
        )
        k3 = self.compute_delta(
            strength_vector=self._strength_vector + 0.5 * self.step_size * k2
        )
        k4 = self.compute_delta(
            strength_vector=self.strength_vector + self.step_size * k3
        )
        return self.step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class QuadraticEnergySolver(Solver):
    """Quadratic Energy Solver.

    Args:
        Solver (_type_): _description_
    """

    def aggregate(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        return self._aggreg_sum(
            parent_vector=parent_vector, strength_vector=strength_vector
        )

    def influence(
        self, aggreg_strength: float, weight: float, p: int = 2, k: float = 1
    ):
        return self._infl_pmax(aggreg_strength=aggreg_strength, weight=weight, p=p, k=k)
