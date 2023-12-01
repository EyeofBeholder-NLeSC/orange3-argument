"""Module of aggregation kernels."""

import numpy as np


def summate(parent_vector: np.ndarray, strength_vector: np.ndarray) -> float:
    """Sum aggregation function.

    Args:
        parent_vector (np.ndarray): Parent vector.
        strength_vector (np.ndarray): Strength vector.

    Returns:
        float: Aggregate strength.
    """
    return parent_vector @ strength_vector


def product(parent_vector: np.ndarray, strength_vector: np.ndarray) -> np.ndarray:
    """Product aggregation function.

    Args:
        parent_vector (np.ndarray): Parent vector.
        strength_vector (np.ndarray): Strength vector.

    Returns:
        float: Aggregate strength.
    """
    base = 0
    attack_part = 1
    support_part = 1
    for i, v in enumerate(parent_vector):
        update = 1 - strength_vector[i]
        if v == -1:
            attack_part *= update
        elif v == 1:
            support_part *= update

    return base + attack_part - support_part
