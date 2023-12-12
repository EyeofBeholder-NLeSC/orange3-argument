"""Test of model.py in reasoner module."""
from contextlib import nullcontext as does_not_raise
import pytest
import numpy as np

from orangearg.argument.reasoner.models import (
    Model,
    QuadraticEnergyModel,
    ContinuousEulerModel,
    CountinuousDFQuADModel,
)
import orangearg.argument.reasoner.aggregation_funcs as agg_funcs
import orangearg.argument.reasoner.influence_funcs as inf_funcs

weights = np.array([0.5, 0.8, 0.3])
parent_vectors = np.array([[0, 0, 1], [-1, 0, 1], [0, 0, 0]])


class Adapter:
    def get_weights(self):
        return weights

    def get_parent_vectors(self):
        return parent_vectors


@pytest.fixture(scope="function")
def model_instance(mocker):
    Model.__abstractmethods__ = set()
    # pylint: disable=abstract-class-instantiated
    model = Model(data_adaptor=Adapter())
    return model


class TestModel:
    def test_weights(self, model_instance):
        assert np.array_equal(model_instance.weights, weights)

    def test_parent_vectors(self, model_instance):
        assert np.array_equal(model_instance.parent_vectors, parent_vectors)

    @pytest.mark.parametrize(
        "init_method, expected_strengths, exception_context",
        [
            ("weight", weights, does_not_raise(enter_result=None)),
            ("uniform", np.ones(len(weights)), does_not_raise(enter_result=None)),
            ("foo", None, pytest.raises(ValueError)),
        ],
    )
    def test_init_strength(
        self, init_method, expected_strengths, exception_context, model_instance
    ):
        with exception_context:
            model_instance.init_strength(init_method=init_method)
            assert np.array_equal(model_instance.strength_vector, expected_strengths)

    def test_compute_delta(self, model_instance, mocker):
        mocker.patch.object(model_instance, "aggregation", return_value=0)
        mocker.patch.object(model_instance, "influence", return_value=0.5)
        delta = model_instance.compute_delta(model_instance.strength_vector)
        assert np.allclose(delta, np.array([0.0, -0.3, 0.2]))

    def test_update(self, model_instance, mocker):
        delta = np.array([0.1, 0.1, 0.1])
        model_instance.update(delta=delta)
        assert np.array_equal(model_instance.strength_vector, np.array([0.6, 0.9, 0.4]))


def test_qe_model():
    model = QuadraticEnergyModel(data_adaptor=Adapter())
    pv = parent_vectors[0]
    sv = model.strength_vector
    s = 0.5
    w = 0.8
    assert model.aggregation(pv, sv) == agg_funcs.summate(pv, sv)
    assert model.influence(s, w) == inf_funcs.pmax(s, w, model.p, model.k)


def test_ce_model():
    model = ContinuousEulerModel(data_adaptor=Adapter())
    pv = parent_vectors[0]
    sv = model.strength_vector
    s = 0.5
    w = 0.8
    assert model.aggregation(pv, sv) == agg_funcs.summate(pv, sv)
    assert model.influence(s, w) == inf_funcs.euler(s, w)


def test_cd_model():
    model = CountinuousDFQuADModel(data_adaptor=Adapter())
    pv = parent_vectors[0]
    sv = model.strength_vector
    s = 0.5
    w = 0.8
    assert model.aggregation(pv, sv) == agg_funcs.product(pv, sv)
    assert model.influence(s, w) == inf_funcs.linear(s, w, model.k)
