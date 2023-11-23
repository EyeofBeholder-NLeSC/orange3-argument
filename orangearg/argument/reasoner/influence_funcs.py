"""Module of influence functions."""

from math import exp


def pmax(agg_strength: float, weight: float, p: int, k: float) -> float:
    """P-max influence function.

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

    return weight * (1 - h(-agg_strength / k) + h(agg_strength / k))


def euler(agg_strength: float, weight: float) -> float:
    """Euler influence function.

    Args:
        aggreg_strength (float): _description_
        weight (float): _description_

    Returns:
        float: _description_
    """
    return 1 - (1 - weight**2) / (1 + weight * exp(agg_strength))
