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
