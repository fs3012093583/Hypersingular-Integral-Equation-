"""Tests for first- and third-order singular integral representations."""

import math

import torch

from neural_network_solvers.neural_hypersingular_constant_subtraction import gauss_legendre_rule
from neural_network_solvers.neural_singular_order_representation_comparison import (
    DTYPE,
    TrainingConfig,
    endpoint_integration_by_parts_operator,
    exact_singular_operator,
    exact_solution,
    taylor_subtraction_operator,
    train_one,
)


def test_order_one_taylor_formula_matches_cubic_analytic_value() -> None:
    nodes, weights = gauss_legendre_rule(64)
    t = torch.tensor([[-0.82], [-0.3], [0.0], [0.37], [0.86]], dtype=DTYPE)
    computed = taylor_subtraction_operator(exact_solution, t, nodes, weights, order=1)
    torch.testing.assert_close(computed, exact_singular_operator(t, 1), rtol=2.0e-13, atol=2.0e-13)


def test_order_one_log_endpoint_formula_converges_to_cubic_analytic_value() -> None:
    nodes, weights = gauss_legendre_rule(512)
    t = torch.tensor([[-0.82], [-0.3], [0.0], [0.37], [0.86]], dtype=DTYPE)
    computed = endpoint_integration_by_parts_operator(exact_solution, t, nodes, weights, order=1)
    torch.testing.assert_close(computed, exact_singular_operator(t, 1), rtol=2.0e-5, atol=2.0e-5)


def test_order_three_both_formulas_match_cubic_analytic_value() -> None:
    nodes, weights = gauss_legendre_rule(64)
    t = torch.tensor([[-0.82], [-0.3], [0.0], [0.37], [0.86]], dtype=DTYPE)
    endpoint = endpoint_integration_by_parts_operator(exact_solution, t, nodes, weights, order=3)
    taylor = taylor_subtraction_operator(exact_solution, t, nodes, weights, order=3)
    analytic = exact_singular_operator(t, 3)
    torch.testing.assert_close(endpoint, analytic, rtol=3.0e-12, atol=3.0e-12)
    torch.testing.assert_close(taylor, analytic, rtol=3.0e-12, atol=3.0e-12)
    torch.testing.assert_close(endpoint, taylor, rtol=3.0e-12, atol=3.0e-12)


def test_short_training_is_finite_and_reproducible() -> None:
    config = TrainingConfig(
        seed=17,
        epochs=5,
        learning_rate=2.0e-3,
        lbfgs_steps=0,
        hidden_width=8,
        hidden_layers=1,
        num_collocation=8,
        num_quadrature=16,
        report_every=100,
    )
    for order in (1, 3):
        for representation in ("endpoint", "taylor"):
            _, first = train_one(order, representation, config, verbose=False)
            _, second = train_one(order, representation, config, verbose=False)
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
