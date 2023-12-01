"""Module of aggregation kernels."""

import numpy as np


def summate(parent_vector: np.ndarray, strength_vector: np.ndarray) -> np.ndarray:
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


def product(parent_vector: np.ndarray, strength_vector: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        parent_vector (np.ndarray): _description_
        strength_vector (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
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
