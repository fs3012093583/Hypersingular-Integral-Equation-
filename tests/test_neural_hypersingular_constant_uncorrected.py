"""Focused tests for the constant-equation uncorrected PINN control."""

import math

import torch

from neural_network_solvers.neural_hypersingular_constant_subtraction import hypersingular_operator
from neural_network_solvers.neural_hypersingular_constant_uncorrected import (
    DTYPE,
    TrainingConfig,
    equation_rhs,
    gauss_legendre_rule,
    train_constant_solution,
    uncorrected_hypersingular_operator,
)


def test_constant_solution_satisfies_the_uncorrected_equation() -> None:
    nodes, weights = gauss_legendre_rule(32)
    t = torch.tensor([[-0.8], [-0.3], [0.0], [0.45], [0.85]], dtype=DTYPE)
    actual = uncorrected_hypersingular_operator(lambda points: torch.ones_like(points), t, nodes, weights)
    torch.testing.assert_close(actual, equation_rhs(t), rtol=1.0e-13, atol=1.0e-13)


def test_quadratic_uses_u_second_on_the_diagonal_without_a_boundary_correction() -> None:
    nodes, weights = gauss_legendre_rule(32)
    # Reusing nodes forces every diagonal entry through the u''(t) substitution.
    t = nodes.clone()
    quadratic = lambda points: points.square()
    actual = uncorrected_hypersingular_operator(quadratic, t, nodes, weights)
    expected = (
        -t.square() * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
        + 2.0 * t * torch.log((1.0 - t) / (1.0 + t))
        + 4.0
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    corrected = hypersingular_operator(quadratic, t, nodes, weights)
    torch.testing.assert_close(actual - corrected, torch.full_like(t, 2.0), rtol=1.0e-12, atol=1.0e-12)


def test_short_training_is_finite_reproducible_and_reports_both_residuals() -> None:
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
    assert len(first["loss_history"]) == config.epochs
    assert first["validation_num_quadrature"] == 128
    for key in (
        "final_normalized_loss",
        "rms_solution_error",
        "rms_uncorrected_equation_residual",
        "max_abs_uncorrected_equation_residual",
        "rms_corrected_equation_residual",
        "max_abs_corrected_equation_residual",
    ):
        assert math.isfinite(first[key])
    assert first["loss_history"] == second["loss_history"]
    assert first["final_normalized_loss"] == second["final_normalized_loss"]
    assert first["rms_uncorrected_equation_residual"] == second["rms_uncorrected_equation_residual"]
    assert first["rms_corrected_equation_residual"] == second["rms_corrected_equation_residual"]
