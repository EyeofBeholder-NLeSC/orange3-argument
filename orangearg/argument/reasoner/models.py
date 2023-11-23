"""Module of reasoning models."""

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from orangearg.argument.reasoner.utilities import Adaptor
from orangearg.argument.reasoner.aggregation_funcs import summate
from orangearg.argument.reasoner.influence_funcs import pmax, euler


class Model(ABC):
    """_summary_

    Args:
        ABC (_type_): _description_

    Raises:
        ValueError: _description_
    """

    init_options = ["weight", "uniform"]

    def __init__(self, data_adaptor: Adaptor, init_method: str = "weight"):
        self._weights = data_adaptor.get_weights()
        self._parent_vectors = data_adaptor.get_parent_vectors()
        self._strength_vector = None

        self.init_strength(init_method=init_method)

    @property
    def weights(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._weights

    @property
    def parent_vectors(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._parent_vectors

    @property
    def strength_vector(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._strength_vector

    def init_strength(self, init_method: str = "weight"):
        """_summary_

        Args:
            init_method (str, optional): _description_. Defaults to "weight".

        Raises:
            ValueError: _description_
        """
        if init_method == "weight":
            self._strength_vector = deepcopy(self._weights)
        elif init_method == "uniform":
            self._strength_vector = 0.5 * np.ones(len(self._weights))
        else:
            # pylint: disable=C0301
            raise ValueError(
                f"Init method of strength should be one of {self.init_options}, but {init_method} is given."
            )

    @abstractmethod
    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        """_summary_

        Args:
            parent_vector (np.ndarray): _description_
            strength_vector (np.ndarray): _description_
        """

    @abstractmethod
    def influence(self, agg_strength: float, weight: float):
        """_summary_

        Args:
            agg_strength (float): _description_
            weight (float): _description_
        """

    def compute_delta(self, strength_vector: np.ndarray):
        """Compute derivate of strength vector at given point.

        Args:
            strength_vector (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        new_strengths = []
        for i in range(self._parent_vectors.shape[0]):
            agg_strength = self.aggregation(
                parent_vector=self._parent_vectors[i], strength_vector=strength_vector
            )
            new_strength = self.influence(
                agg_strength=agg_strength, weight=self._weights[i]
            )
            new_strengths.append(new_strength)
        return np.array(new_strengths) - self._strength_vector

    def update(self, delta: np.ndarray):
        """_summary_

        Args:
            delta (np.ndarray): _description_
        """
        self._strength_vector += delta


class QuadraticEnergyModel(Model):
    """_summary_

    Args:
        Model (_type_): _description_
    """

    def __init__(
        self,
        data_adaptor: Adaptor,
        init_method: str = "weight",
        p: int = 2,
        k: float = 1,
    ):
        super().__init__(data_adaptor=data_adaptor, init_method=init_method)
        self.p = p
        self.k = k

    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        return summate(parent_vector=parent_vector, strength_vector=strength_vector)

    def influence(self, agg_strength: float, weight: float):
        return pmax(agg_strength=agg_strength, weight=weight, p=self.p, k=self.k)


class ContinuousEulerModel(Model):
    """_summary_

    Args:
        Model (_type_): _description_
    """

    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        return summate(parent_vector=parent_vector, strength_vector=strength_vector)

    def influence(self, agg_strength: float, weight: float):
        return euler(agg_strength=agg_strength, weight=weight)
