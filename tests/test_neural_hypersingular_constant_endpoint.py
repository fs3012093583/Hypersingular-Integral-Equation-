"""Tests for the endpoint-value hypersingular representation."""

import math

import torch

from neural_network_solvers.neural_hypersingular_constant_endpoint import (
    DTYPE,
    TrainingConfig,
    endpoint_hypersingular_operator,
    gauss_legendre_rule,
    train_constant_solution,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import hypersingular_operator


def test_endpoint_formula_matches_taylor_formula_for_smooth_functions() -> None:
    nodes, weights = gauss_legendre_rule(48)
    t = nodes.clone()
    functions = (
        lambda x: torch.ones_like(x),
        lambda x: 1.2 - 0.37 * x,
        lambda x: x.square(),
        lambda x: 0.4 + 0.2 * x - 0.3 * x.square() + 0.1 * x.pow(3),
    )
    for function in functions:
        endpoint_value = endpoint_hypersingular_operator(function, t, nodes, weights)
        taylor_value = hypersingular_operator(function, t, nodes, weights)
        torch.testing.assert_close(endpoint_value, taylor_value, rtol=2.0e-12, atol=2.0e-12)


def test_quadratic_uses_endpoint_values_and_derivative_diagonal_limit() -> None:
    nodes, weights = gauss_legendre_rule(32)
    t = nodes.clone()
    actual = endpoint_hypersingular_operator(lambda x: x.square(), t, nodes, weights)
    expected = (
        -1.0 / (1.0 - t)
        - 1.0 / (1.0 + t)
        + 2.0 * t * torch.log((1.0 - t) / (1.0 + t))
        + 4.0
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_short_training_is_finite_reproducible_and_reports_full_interval() -> None:
    config = TrainingConfig(
        seed=17,
        epochs=8,
        learning_rate=2.0e-3,
        lbfgs_steps=0,
        hidden_width=8,
        hidden_layers=1,
        num_collocation=10,
        num_quadrature=20,
        report_every=100,
    )
    _, first = train_constant_solution(config, verbose=False)
    _, second = train_constant_solution(config, verbose=False)
    assert first["solution_evaluation_interval"] == [-1.0, 1.0]
    assert first["residual_evaluation_interval"] == [-0.95, 0.95]
    for key in (
        "final_normalized_loss",
        "rms_solution_error",
        "interior_rms_solution_error",
        "rms_equation_residual",
        "abs_solution_error_t_minus_1",
        "abs_solution_error_t_plus_1",
    ):
        assert math.isfinite(first[key])
    assert first["loss_history"] == second["loss_history"]
    assert first["rms_solution_error"] == second["rms_solution_error"]
