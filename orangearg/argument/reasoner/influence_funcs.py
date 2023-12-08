"""Module of influence functions."""

from math import exp


def pmax(agg_strength: float, weight: float, p: int, k: float) -> float:
    """P-max influence function.

    Args:
        agg_strength (float): Aggregated strength.
        weight (float): Argument weight.
        p (int): Order of PMax fucntion.
        k (float): Conservativeness parameter.

    Returns:
        float: Influenced strength.
    """
    if not isinstance(p, int):
        raise ValueError(f"P should be integer: {p}.")
    if p < 0:
        raise ValueError(f"P should be non-negative: {p}.")
    if k < 0:
        raise ValueError(f"K should be positive: {k}.")
    if not -k <= agg_strength <= k:
        raise ValueError(
            f"Aggregated strength should be in range of [-{k}, {k}]: {agg_strength}."
        )

    def h(x):
        return max(0, x) ** p / (1 + max(0, x) ** p)

    return weight * (1 - h(-agg_strength / k) + h(agg_strength / k))


def euler(agg_strength: float, weight: float) -> float:
    """Euler influence function.

    Args:
        agg_strength (float): Aggregated strength.
        weight (float): Argument weight.

    Returns:
        float: Influenced strength.
    """
    return 1 - (1 - weight**2) / (1 + weight * exp(agg_strength))


def linear(agg_strength: float, weight: float, k: float) -> float:
    """_summary_

    Args:
        agg_strength (float): Aggregated strength.
        weight (float): Argument weight.
        k (float): Conservativeness parameter.


    Returns:
        float: Influenced strength.
    """
    if k < 0:
        raise ValueError(f"K should be positive: {k}.")
    if not -k <= agg_strength <= k:
        raise ValueError(
            f"Aggregated strength should be in range of [-{k}, {k}]: {agg_strength}."
        )

    return (
        weight
        - weight * max(0, -agg_strength) / k
        + (1 - weight) * max(0, agg_strength) / k
    )
