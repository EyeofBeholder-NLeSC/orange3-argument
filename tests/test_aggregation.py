"""Test of aggregation_funcs."""
from contextlib import nullcontext as does_not_raise
import pytest
from pytest import approx
import numpy as np

from orangearg.argument.reasoner.aggregation_funcs import summate, product


@pytest.mark.parametrize(
    "parent_vector, strength_vector, expected_result, exception_context",
    [
        (
            np.array([-1, 0, 1]),
            np.array([0.3, 0.2, 0.7]),
            0.4,
            does_not_raise(enter_result=None),
        ),  # attack + support
        (
            np.array([0, 0, 0]),
            np.array([0.3, 0.2, 0.7]),
            0,
            does_not_raise(enter_result=None),
        ),  # no link
        (
            np.array([-1, 0, -1]),
            np.array([0.3, 0.2, 0.7]),
            -1.0,
            does_not_raise(enter_result=None),
        ),  # only attack
        (
            np.array([-1, 0, 1]),
            np.array([0.3, 0.2, 0.7, 0.4]),
            None,
            pytest.raises(ValueError),
        ),  # length not match
    ],
)
def test_summate(parent_vector, strength_vector, expected_result, exception_context):
    with exception_context:
        result = summate(parent_vector=parent_vector, strength_vector=strength_vector)
        assert expected_result == approx(result, 0.01)


@pytest.mark.parametrize(
    "parent_vector, strength_vector, expected_result, exception_context",
    [
        (
            np.array([-1, 0, -1, 1, 1]),
            np.array([0.3, 0.2, 0.7, 0.4, 0.6]),
            -0.03,
            does_not_raise(enter_result=None),
        ),  # attack + support
        (
            np.array([0, 0, 0, 0, 0]),
            np.array([0.3, 0.2, 0.7, 0.4, 0.6]),
            0,
            does_not_raise(enter_result=None),
        ),  # no link
        (
            np.array([-1, 0, -1, 0, 0]),
            np.array([0.3, 0.2, 0.7, 0.4, 0.6]),
            0.21,
            does_not_raise(enter_result=None),
        ),  # only attack
        (
            np.array([-1, 0, -1, 1, 1]),
            np.array([0.3, 0.2, 0.7, 0.4]),
            None,
            pytest.raises(ValueError),
        ),  # length not match
    ],
)
def test_product(parent_vector, strength_vector, expected_result, exception_context):
    with exception_context:
        result = product(parent_vector=parent_vector, strength_vector=strength_vector)
        assert expected_result == approx(result, 0.01)
