"""Focused tests for the parameterized operator-budget benchmark."""

from dataclasses import replace
from unittest.mock import patch

import mpmath as mp
import torch

from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule
from experiments.parameterized_operator_budget_benchmark import (
    PRESETS,
    ExactExpSlice,
    ParameterizedConfig,
    ParameterizedTanhNetwork,
    _calibrate_ad_error,
    _exact_rhs,
    run_one,
)
from neural_network_solvers.general_taylor_finite_part import (
    taylor_finite_part_operator,
)


def test_manufactured_rhs_has_zero_canonical_residual() -> None:
    dtype = torch.float64
    config = ParameterizedConfig(order=4)
    points = torch.tensor([[-0.7], [0.0], [0.6]], dtype=dtype)
    nodes, weights = gauss_legendre_rule(96, dtype)
    xi = 1.125
    rhs = _exact_rhs(points, xi, nodes, weights, config)
    exact = ExactExpSlice(xi)
    finite_part = taylor_finite_part_operator(
        exact,
        points,
        nodes,
        weights,
        order=config.order,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
    )
    residual = (
        exact(points)
        + config.coupling
        * (1.0 - points.square()).pow(config.order - 1)
        * finite_part
        - rhs
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1.0e-13, rtol=0.0)


def test_smoke_run_is_finite_and_uses_disjoint_parameters() -> None:
    config = replace(
        ParameterizedConfig(),
        representation="operator_budget",
        **PRESETS["smoke"],
    )
    result = run_one(config)
    assert result["status"] == "ok"
    assert torch.isfinite(torch.tensor(result["heldout_solution_rmse"]))
    assert torch.isfinite(torch.tensor(result["heldout_residual_rmse"]))
    assert set(config.training_parameters).isdisjoint(config.validation_parameters)
    diagnostics = result["selection_diagnostics"]
    assert diagnostics is not None
    assert 0.0 <= diagnostics["budget_satisfaction_fraction"] <= 1.0
    assert result["ad_calibration_reference_policy"].startswith("v2:")


def test_calibration_subtracts_before_rounding_the_mp_reference() -> None:
    model = ParameterizedTanhNetwork(width=2, hidden_layers=1, parameter_interval=(0.5, 1.5)).double()
    with mp.workdps(90):
        reference = [mp.mpf("1.0000000000000006")]
    observed = [torch.tensor([[1.0]], dtype=torch.float64)]
    module = "experiments.parameterized_operator_budget_benchmark"
    with (
        patch(module + ".high_precision_tanh_partial_derivatives", return_value=reference),
        patch(module + ".value_and_derivatives", return_value=observed),
    ):
        result = _calibrate_ad_error(
            model, (0.5, 1.0, 1.5), highest_order=0,
            safety_factor=1.0, dtype=torch.float64, device=torch.device("cpu"),
        )
    assert abs(result[0] - 6e-16) < 1e-28
