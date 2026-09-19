"""Tests for the manufactured parameterized hypersingular PINN."""

import math

import torch

from neural_network_solvers.neural_hypersingular_parameterized_subtraction import (
    DTYPE,
    TrainingConfig,
    equation_operator_value,
    equation_rhs,
    exact_hypersingular_operator,
    exact_solution,
    gauss_legendre_rule,
    normalized_equation_residual,
    train_single_gamma,
)


def test_rhs_is_fixed_and_matches_analytic_affine_substitution() -> None:
    t = torch.tensor([[-0.8], [-0.35], [0.0], [0.25], [0.7]], dtype=DTYPE)
    for gamma in (0.0, 0.1, 1.0, 10.0):
        actual = equation_rhs(t, gamma)
        affine = 1.0 + gamma * t
        analytic_h = (
            -affine * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
            + gamma * torch.log((1.0 - t) / (1.0 + t))
        )
        expected = t.pow(4) * affine + (2.0 - t) * (1.0 + t) * analytic_h
        torch.testing.assert_close(actual, expected, rtol=1.0e-13, atol=1.0e-13)
        torch.testing.assert_close(exact_hypersingular_operator(t, gamma), analytic_h, rtol=1.0e-13, atol=1.0e-13)

    # This call requires no trained model and therefore establishes that the
    # manufactured RHS is a pure fixed function of (t, gamma).
    torch.testing.assert_close(equation_rhs(t, 0.5), equation_rhs(t.clone(), 0.5), rtol=0.0, atol=0.0)


def test_exact_affine_solution_has_near_zero_residual_for_multiple_gammas() -> None:
    nodes, weights = gauss_legendre_rule(48)
    # Reusing nodes also exercises the diagonal Taylor-limit path in the shared operator.
    t = nodes.clone()
    for gamma in (0.0, 0.1, 0.5, 1.0, 5.0, 10.0):
        operator_value = equation_operator_value(
            lambda points, gamma=gamma: exact_solution(points, gamma),
            t,
            nodes,
            weights,
        )
        raw_residual = operator_value - equation_rhs(t, gamma)
        normalized = normalized_equation_residual(operator_value, t, gamma)
        torch.testing.assert_close(raw_residual, torch.zeros_like(raw_residual), rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(normalized, torch.zeros_like(normalized), rtol=1.0e-12, atol=1.0e-12)


def test_short_single_gamma_training_is_finite_and_reproducible() -> None:
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
    assert len(first["loss_history"]) == config.epochs
    assert math.isfinite(first["final_normalized_loss"])
    assert math.isfinite(first["rms_solution_error"])
    assert math.isfinite(first["rms_equation_residual"])
    assert first["loss_history"] == second["loss_history"]
    assert first["final_normalized_loss"] == second["final_normalized_loss"]
