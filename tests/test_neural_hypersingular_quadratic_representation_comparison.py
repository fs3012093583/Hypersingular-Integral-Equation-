"""Tests for the non-affine endpoint/Taylor representation comparison."""

import math

import torch

from neural_network_solvers.neural_hypersingular_quadratic_representation_comparison import (
    DTYPE,
    TrainingConfig,
    equation_operator_value,
    equation_rhs,
    exact_hypersingular_operator,
    exact_solution,
    gauss_legendre_rule,
    train_one,
)


def test_quadratic_exact_solution_has_the_same_value_under_both_representations() -> None:
    nodes, weights = gauss_legendre_rule(48)
    t = nodes.clone()
    for eta in (0.0, 0.1, 0.5, 1.0):
        exact = lambda points, eta=eta: exact_solution(points, 1.0, eta)
        endpoint = equation_operator_value("endpoint", exact, t, nodes, weights)
        taylor = equation_operator_value("taylor", exact, t, nodes, weights)
        rhs = equation_rhs(t, 1.0, eta)
        torch.testing.assert_close(endpoint, rhs, rtol=2.0e-12, atol=2.0e-12)
        torch.testing.assert_close(taylor, rhs, rtol=2.0e-12, atol=2.0e-12)
        torch.testing.assert_close(endpoint, taylor, rtol=2.0e-12, atol=2.0e-12)


def test_analytic_quadratic_operator_matches_endpoint_boundary_formula() -> None:
    t = torch.tensor([[-0.8], [-0.25], [0.0], [0.4], [0.85]], dtype=DTYPE)
    gamma = 1.0
    eta = 0.7
    endpoint_expression = (
        -(1.0 + gamma + eta) / (1.0 - t)
        -(1.0 - gamma + eta) / (1.0 + t)
        +(gamma + 2.0 * eta * t) * torch.log((1.0 - t) / (1.0 + t))
        +4.0 * eta
    )
    torch.testing.assert_close(
        exact_hypersingular_operator(t, gamma, eta),
        endpoint_expression,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cubic_exact_solution_has_the_same_value_under_both_representations() -> None:
    nodes, weights = gauss_legendre_rule(64)
    t = nodes.clone()
    gamma = 1.0
    eta = 1.0
    for xi in (0.0, 0.1, 0.5, 1.0):
        exact = lambda points, xi=xi: exact_solution(points, gamma, eta, xi)
        endpoint = equation_operator_value("endpoint", exact, t, nodes, weights)
        taylor = equation_operator_value("taylor", exact, t, nodes, weights)
        rhs = equation_rhs(t, gamma, eta, xi)
        torch.testing.assert_close(endpoint, rhs, rtol=3.0e-12, atol=3.0e-12)
        torch.testing.assert_close(taylor, rhs, rtol=3.0e-12, atol=3.0e-12)
        torch.testing.assert_close(endpoint, taylor, rtol=3.0e-12, atol=3.0e-12)


def test_analytic_cubic_operator_matches_endpoint_boundary_formula() -> None:
    t = torch.tensor([[-0.8], [-0.25], [0.0], [0.4], [0.85]], dtype=DTYPE)
    gamma = 1.0
    eta = 0.7
    xi = 0.3
    endpoint_expression = (
        -(1.0 + gamma + eta + xi) / (1.0 - t)
        -(1.0 - gamma + eta - xi) / (1.0 + t)
        +(gamma + 2.0 * eta * t + 3.0 * xi * t.square())
        * torch.log((1.0 - t) / (1.0 + t))
        +4.0 * eta
        +6.0 * xi * t
    )
    torch.testing.assert_close(
        exact_hypersingular_operator(t, gamma, eta, xi),
        endpoint_expression,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_short_training_is_finite_and_reproducible_for_both_representations() -> None:
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
    for representation in ("endpoint", "taylor"):
        _, first = train_one(1.0, 0.5, representation, config, verbose=False)
        _, second = train_one(1.0, 0.5, representation, config, verbose=False)
        for key in (
            "final_normalized_loss",
            "rms_solution_error",
            "rms_own_equation_residual",
            "rms_alternate_equation_residual",
            "rms_representation_operator_difference",
        ):
            assert math.isfinite(first[key])
        assert first["loss_history"] == second["loss_history"]
        assert first["rms_solution_error"] == second["rms_solution_error"]
