"""Module of reasoning models."""

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from orangearg.argument.reasoner.utilities import Adaptor
from orangearg.argument.reasoner.aggregation_funcs import summate, product
from orangearg.argument.reasoner.influence_funcs import pmax, euler, linear


class Model(ABC):
    """Base class of all reasoning models."""

    init_options = ["weight", "uniform"]

    def __init__(self, data_adaptor: Adaptor, init_method: str = "weight"):
        self._weights = data_adaptor.get_weights()
        self._parent_vectors = data_adaptor.get_parent_vectors()
        self._strength_vector = None

        self.init_strength(init_method=init_method)

    @property
    def weights(self):
        """Weight vector of arguments."""
        return self._weights

    @property
    def parent_vectors(self):
        """Parent vectors of arguments."""
        return self._parent_vectors

    @property
    def strength_vector(self):
        """Strength vector of arguments."""
        return self._strength_vector

    def init_strength(self, init_method: str = "weight"):
        """Initialize strength vector of the model.

        Currently, two methods are provided:
        - weight: using argument weights as the initital strengths
        - uniform: giving all arguments the same value of initital strength, default to be 1.

        Args:
            init_method (str, optional): Method of initialization. Defaults to "weight".

        Raises:
            ValueError: when the given method name is unknown.
        """
        if init_method == "weight":
            self._strength_vector = deepcopy(self._weights)
        elif init_method == "uniform":
            self._strength_vector = np.ones(len(self._weights))
        else:
            # pylint: disable=line-too-long
            raise ValueError(
                f"Init method of strength should be one of {self.init_options}, but {init_method} is given."
            )

    @abstractmethod
    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        """Aggregation function.

        Args:
            parent_vector (np.ndarray): Parent vector.
            strength_vector (np.ndarray): Strength vector.
        """

    @abstractmethod
    def influence(self, agg_strength: float, weight: float):
        """Influence function.

        Args:
            agg_strength (float): Aggregated strength.
            weight (float): Argument weight.
        """

    def compute_delta(self, strength_vector: np.ndarray):
        """Compute derivate of strength vector at given point.

        Args:
            strength_vector (np.ndarray): Strength vector.

        Returns:
            _type_: Derivate of strength vector.
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
        """Update strength vector by a given derivate.

        Args:
            delta (np.ndarray): derivate vector.
        """
        self._strength_vector += delta


class QuadraticEnergyModel(Model):
    """Quadratic Energy model."""

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
    """Countinuous Euler model."""

    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        return summate(parent_vector=parent_vector, strength_vector=strength_vector)

    def influence(self, agg_strength: float, weight: float):
        return euler(agg_strength=agg_strength, weight=weight)


class CountinuousDFQuADModel(Model):
    """Countinuous DF-QuAD model."""

    def __init__(
        self, data_adaptor: Adaptor, init_method: str = "weight", k: float = 1
    ):
        super().__init__(data_adaptor=data_adaptor, init_method=init_method)
        self.k = k

    def aggregation(self, parent_vector: np.ndarray, strength_vector: np.ndarray):
        return product(parent_vector=parent_vector, strength_vector=strength_vector)

    def influence(self, agg_strength: float, weight: float):
        return linear(agg_strength=agg_strength, weight=weight, k=self.k)
