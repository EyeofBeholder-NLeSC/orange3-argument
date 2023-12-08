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
    try:
        return parent_vector @ strength_vector
    except ValueError as e:
        # pylint: disable=line-too-long
        raise ValueError(
            f"Length of parent and strength vector doesn't match: {parent_vector.size, strength_vector.size}."
        ) from e


def product(parent_vector: np.ndarray, strength_vector: np.ndarray) -> np.ndarray:
    """Product aggregation function.

    Args:
        parent_vector (np.ndarray): Parent vector.
        strength_vector (np.ndarray): Strength vector.

    Returns:
        float: Aggregate strength.
    """
    if parent_vector.size != strength_vector.size:
        # pylint: disable=line-too-long
        raise ValueError(
            f"Length of parent and strength vector doesn't match: {parent_vector.size, strength_vector.size}."
        )

    complete_strength = 1 - strength_vector
    attack_part = complete_strength[parent_vector == -1]
    attack_part = np.prod(attack_part) if attack_part.size > 0 else 0
    support_part = complete_strength[parent_vector == 1]
    support_part = np.prod(support_part) if support_part.size > 0 else 0

    return attack_part - support_part
