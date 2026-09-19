"""Tests for the parameterized endpoint-value PINN."""

import math

import torch

from neural_network_solvers.neural_hypersingular_parameterized_endpoint import (
    DTYPE,
    TrainingConfig,
    equation_operator_value,
    equation_rhs,
    exact_solution,
    gauss_legendre_rule,
    train_single_gamma,
)


def test_exact_affine_solution_satisfies_endpoint_equation() -> None:
    nodes, weights = gauss_legendre_rule(48)
    t = nodes.clone()
    for gamma in (0.0, 0.1, 0.5, 1.0, 5.0, 10.0):
        value = equation_operator_value(
            lambda points, gamma=gamma: exact_solution(points, gamma), t, nodes, weights
        )
        torch.testing.assert_close(value, equation_rhs(t, gamma), rtol=2.0e-12, atol=2.0e-12)


def test_short_training_is_finite_reproducible_and_reports_endpoints() -> None:
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
    _, first = train_single_gamma(0.5, config, verbose=False)
    _, second = train_single_gamma(0.5, config, verbose=False)
    assert first["solution_evaluation_interval"] == [-1.0, 1.0]
    assert first["residual_evaluation_interval"] == [-0.95, 0.95]
    for key in (
        "final_normalized_loss",
        "rms_solution_error",
        "interior_rms_solution_error",
        "rms_equation_residual",
        "max_endpoint_abs_solution_error",
    ):
        assert math.isfinite(first[key])
    assert first["loss_history"] == second["loss_history"]
    assert first["rms_solution_error"] == second["rms_solution_error"]
