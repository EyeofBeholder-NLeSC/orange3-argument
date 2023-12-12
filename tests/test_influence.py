"""Test of influence_funcs."""
from contextlib import nullcontext as does_not_raise
import pytest
from pytest import approx

from orangearg.argument.reasoner.influence_funcs import pmax, euler, linear


@pytest.mark.parametrize(
    "agg_strength, weight, p, k, expected_result, exception_context",
    [
        (0.5, 0.8, 2, 1, 0.96, does_not_raise(enter_result=None)),  # positive agg
        (0, 0.8, 2, 1, 0.8, does_not_raise(enter_result=None)),  # 0 agg
        (-0.5, 0.8, 2, 1, 0.64, does_not_raise(enter_result=None)),  # negative agg
        (0.5, 0.8, 1.5, 1, None, pytest.raises(ValueError)),  # float p
        (0.5, 0.8, -1, 1, None, pytest.raises(ValueError)),  # negative p
        (0.5, 0.8, 2, -1, None, pytest.raises(ValueError)),  # negative k
        (0.5, 0.8, 2, 0.4, None, pytest.raises(ValueError)),  # k < agg
    ],
)
def test_pmax(agg_strength, weight, p, k, expected_result, exception_context):
    with exception_context:
        result = pmax(agg_strength=agg_strength, weight=weight, p=p, k=k)
        assert expected_result == approx(result, 0.01)


@pytest.mark.parametrize(
    "agg_strength, weight, expected_result, exception_context",
    [
        (0.5, 0.8, 0.84, does_not_raise(enter_result=None)),  # positive agg
        (0, 0.8, 0.8, does_not_raise(enter_result=None)),  # 0 agg
        (-0.5, 0.8, 0.76, does_not_raise(enter_result=None)),  # negative agg
    ],
)
def test_euler(agg_strength, weight, expected_result, exception_context):
    with exception_context:
        result = euler(agg_strength=agg_strength, weight=weight)
        assert expected_result == approx(result, 0.01)


@pytest.mark.parametrize(
    "agg_strength, weight, k, expected_result, exception_context",
    [
        (0.5, 0.8, 1, 0.9, does_not_raise(enter_result=None)),  # positive agg
        (0, 0.8, 1, 0.8, does_not_raise(enter_result=None)),  # 0 agg
        (-0.5, 0.8, 1, 0.4, does_not_raise(enter_result=None)),  # negative agg
        (0.5, 0.8, -1, None, pytest.raises(ValueError)),  # negative k
        (0.5, 0.8, 0.4, None, pytest.raises(ValueError)),  # k < agg
    ],
)
def test_linear(agg_strength, weight, k, expected_result, exception_context):
    with exception_context:
        result = linear(agg_strength=agg_strength, weight=weight, k=k)
        assert expected_result == approx(result, 0.01)
