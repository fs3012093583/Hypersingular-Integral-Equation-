"""Tests for the isolated uncorrected-blackboard hypersingular comparison."""

import math

import torch

from neural_network_solvers.neural_hypersingular_constant_subtraction import hypersingular_operator
from neural_network_solvers.neural_hypersingular_uncorrected_comparison import (
    DTYPE,
    TrainingConfig,
    gauss_legendre_rule,
    train_single_gamma,
    uncorrected_hypersingular_operator,
)


def test_affine_functions_have_identical_uncorrected_and_corrected_operators() -> None:
    nodes, weights = gauss_legendre_rule(32)
    # Reusing nodes exercises the diagonal u''(t)=0 substitution as well.
    t = nodes.clone()
    affine = lambda points: 1.25 - 0.37 * points
    uncorrected = uncorrected_hypersingular_operator(affine, t, nodes, weights)
    corrected = hypersingular_operator(affine, t, nodes, weights)
    torch.testing.assert_close(uncorrected, corrected, rtol=1.0e-12, atol=1.0e-12)


def test_quadratic_exposes_the_missing_endpoint_correction_difference() -> None:
    nodes, weights = gauss_legendre_rule(32)
    # The uncorrected quotient is exactly 2, hence its integral is 4.  The
    # corrected Taylor remainder is exactly 1, hence its integral is 2.
    t = nodes.clone()
    quadratic = lambda points: points.square()
    uncorrected = uncorrected_hypersingular_operator(quadratic, t, nodes, weights)
    corrected = hypersingular_operator(quadratic, t, nodes, weights)
    torch.testing.assert_close(uncorrected - corrected, torch.full_like(t, 2.0), rtol=1.0e-12, atol=1.0e-12)


def test_short_training_is_finite_and_reproducible_with_both_validation_residuals() -> None:
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
    assert first["validation_num_quadrature"] == 192
    for key in (
        "final_normalized_loss",
        "rms_solution_error",
        "rms_uncorrected_equation_residual",
        "max_abs_uncorrected_equation_residual",
        "rms_corrected_physical_residual",
        "max_abs_corrected_physical_residual",
    ):
        assert math.isfinite(first[key])
    assert first["loss_history"] == second["loss_history"]
    assert first["final_normalized_loss"] == second["final_normalized_loss"]
    assert first["rms_uncorrected_equation_residual"] == second["rms_uncorrected_equation_residual"]
    assert first["rms_corrected_physical_residual"] == second["rms_corrected_physical_residual"]
