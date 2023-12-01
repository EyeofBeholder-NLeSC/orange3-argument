"""Module of solver."""

from typing import Type, TypeVar, Union
import numpy as np
from orangearg.argument.reasoner.models import Model
from orangearg.argument.reasoner.utilities import Collector

MODEL = TypeVar("MODEL", bound=Model)


# pylint: disable=too-few-public-methods
class Solver:
    """The Solver model."""

    solver_options = ["RK4"]

    def __init__(
        self, model: Type[MODEL], step_size: float, max_steps: int, epsilon: float
    ):
        self.model = model
        self.step_size = step_size
        self.max_steps = max_steps
        self.epsilon = epsilon

    def solve(
        self, solver: str, collect_data: bool = False
    ) -> tuple[int, Union[Collector, None]]:
        """Solve the reasoner graph by a selected algorithm.

        Currently, only one algorithm is provided:
        - RK4: forth-order version of the Eunge-Kutta algorithm, details can be found here: https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

        Args:
            solver (str): Name of the solver algorithm.
            collect_data (bool, optional): Indicate if want to collect the strength vector of each step. Defaults to False.

        Raises:
            ValueError: When the given solver name is unknown.

        Returns:
            tuple[int, Union[Collector, None]]:
                - The index of the final step of solving
                - The data collector object that collects all the intermediate strength vectors if `collect_data` is `True`. Otherwise, None is returned.
        """
        if solver == "RK4":
            app_func = self._rk4
        else:
            raise ValueError(
                f"Approximator should be one of {self.solver_options}, but {solver} is given."
            )

        data_collector = (
            Collector(data=self.model.strength_vector) if collect_data else None
        )

        for step in range(int(self.max_steps)):
            delta = app_func()
            self.model.update(delta=delta)

            # pylint: disable=expression-not-assigned
            data_collector and data_collector.collect(self.model.strength_vector)
            if abs(delta).max() < self.epsilon:
                break

        return step, data_collector

    def _rk4(self) -> np.ndarray:
        """The forth-order Runge-Kutta algorithm.

        Returns:
            np.ndarray: The derivate at the current strength vector.
        """
        strength_vector = self.model.strength_vector
        k1 = self.model.compute_delta(strength_vector=strength_vector)
        k2 = self.model.compute_delta(
            strength_vector=strength_vector + 0.5 * self.step_size * k1
        )
        k3 = self.model.compute_delta(
            strength_vector=strength_vector + 0.5 * self.step_size * k2
        )
        k4 = self.model.compute_delta(
            strength_vector=strength_vector + self.step_size * k3
        )
        return self.step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
