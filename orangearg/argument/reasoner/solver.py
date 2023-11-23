"""Module of solver."""

from typing import Type, TypeVar
import numpy as np
from orangearg.argument.reasoner.models import Model
from orangearg.argument.reasoner.utilities import Collector

MODEL = TypeVar("MODEL", bound=Model)


class Solver:
    """_summary_

    Raises:
        ValueError: _description_
    """

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
    ) -> tuple[int, Collector | None]:
        """_summary_

        Args:
            approximator (str): _description_
            collect_data (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            tuple[int, Collector | None]: _description_
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

            # pylint: disable=W0106
            data_collector and data_collector.collect(self.model.strength_vector)
            if abs(delta).max() < self.epsilon:
                break

        return step, data_collector

    def _rk4(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
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
