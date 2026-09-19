"""Regression tests for the operator-budget accuracy benchmark schema v2."""

import math

import mpmath as mp
import pytest
import torch

from experiments.general_taylor_operator_benchmark import FUNCTIONS, _gauss_rule
from experiments.operator_budget_accuracy_benchmark import (
    _mp_discrete_finite_part,
    _mp_from_binary_float,
    _record_fixed,
    _record_weighted,
    _torch_analytic_moment,
)
from neural_network_solvers.general_taylor_finite_part import (
    taylor_finite_part_operator,
)


@pytest.fixture(autouse=True)
def high_precision_reference():
    with mp.workdps(90):
        yield


def test_analytic_moment_enables_input_derivatives() -> None:
    point = torch.tensor([[0.23]], dtype=torch.float64)
    value = _torch_analytic_moment(torch.exp, point, 2)
    expected = torch.exp(point) * (
        1.0 / (-1.0 - point) - 1.0 / (1.0 - point)
        + torch.log((1.0 - point) / (point + 1.0))
    )
    torch.testing.assert_close(value.detach(), expected, rtol=1e-14, atol=1e-14)
    assert not point.requires_grad


def test_float32_reference_uses_the_stored_target_coordinate() -> None:
    point = torch.tensor([[-0.95]], dtype=torch.float32)
    stored = float(point.item())
    reference_coordinate = _mp_from_binary_float(stored)

    assert reference_coordinate == mp.mpf(stored)
    assert reference_coordinate != mp.mpf("-0.95")


def test_weighted_record_full_error_decomposition_closes() -> None:
    spec = FUNCTIONS[0]
    dtype = torch.float64
    point = torch.tensor([[0.23]], dtype=dtype)
    nodes, weights = _gauss_rule(12, dtype)
    t_mp = _mp_from_binary_float(float(point.item()))
    (
        discrete_reference,
        analytic_reference,
        quotient_reference,
        quotient_values_reference,
    ) = _mp_discrete_finite_part(spec, t_mp, 2, nodes, weights)
    record = _record_weighted(
        spec=spec,
        order=2,
        point=point,
        dtype_name="float64",
        nodes=nodes,
        weights=weights,
        residual_budget=1.0e-4,
        max_p=2,
        discrete_reference=discrete_reference,
        analytic_reference=analytic_reference,
        quotient_reference=quotient_reference,
        quotient_values_reference=quotient_values_reference,
        continuous_reference=discrete_reference,
    )

    reconstructed = (
        float(record["signed_moment_error"])
        + float(record["signed_quotient_weighted_sum_error"])
        + float(record["signed_accumulation_rounding_error"])
    )
    assert math.isclose(
        float(record["signed_full_discrete_error"]),
        reconstructed,
        rel_tol=0.0,
        abs_tol=2.0e-15,
    )
    assert record["representation_error_scope"] == (
        "full_discrete_operator_including_moment_and_sum"
    )
    assert float(record["weighted_quotient_absolute_error"]) >= 0.0
    assert math.isclose(
        float(record["signed_accumulation_rounding_error"]),
        float(record["signed_product_sum_rounding_error"])
        + float(record["signed_final_addition_rounding_error"]),
        rel_tol=0.0,
        abs_tol=2.0e-30,
    )


def test_fixed_record_uses_the_same_binary_float_reference_coordinate() -> None:
    spec = FUNCTIONS[0]
    dtype = torch.float32
    point = torch.tensor([[0.1]], dtype=dtype)
    nodes, weights = _gauss_rule(12, dtype)
    t_mp = _mp_from_binary_float(float(point.item()))
    discrete_reference, _, _, _ = _mp_discrete_finite_part(
        spec, t_mp, 2, nodes, weights
    )
    record = _record_fixed(
        spec=spec,
        order=2,
        point=point,
        dtype_name="float32",
        nodes=nodes,
        weights=weights,
        continuation_order=2,
        discrete_reference=discrete_reference,
        continuous_reference=discrete_reference,
    )
    actual = taylor_finite_part_operator(
        spec.torch_function,
        point,
        nodes,
        weights,
        order=2,
        continuation_order=2,
        threshold_mode="balanced",
    )
    expected_error = abs(
        _mp_from_binary_float(float(actual.item())) - discrete_reference
    )

    assert float(record["t"]) == float(point.item())
    assert math.isclose(
        float(record["representation_error"]),
        float(expected_error),
        rel_tol=0.0,
        abs_tol=0.0,
    )
