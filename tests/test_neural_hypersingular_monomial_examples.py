"""Tests for the photographed linear and quadratic hypersingular examples."""

import math

import torch

from neural_network_solvers.neural_hypersingular_constant_endpoint import (
    endpoint_hypersingular_operator,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    hypersingular_operator as taylor_hypersingular_operator,
)
from neural_network_solvers.neural_hypersingular_monomial_examples import (
    TrainingConfig,
    equation_rhs,
    exact_solution,
    train_one,
)


def test_specialized_rhs_matches_the_general_photographed_formulas() -> None:
    t = torch.tensor([[-0.8], [-0.25], [0.0], [0.4], [0.85]], dtype=DTYPE)
    a = -1.0
    b = 1.0
    logarithm = torch.log(torch.abs((b - t) / (a - t)))
    reciprocal_term = 1.0 / (a - t) - 1.0 / (b - t)
    torch.testing.assert_close(
        equation_rhs(t, "linear"), logarithm + t * reciprocal_term,
        rtol=1.0e-14, atol=1.0e-14,
    )
    torch.testing.assert_close(
        equation_rhs(t, "quadratic"),
        (b - a) + 2.0 * t * logarithm + t.square() * reciprocal_term,
        rtol=1.0e-14, atol=1.0e-14,
    )


def test_both_finite_part_representations_reproduce_each_rhs() -> None:
    nodes, weights = gauss_legendre_rule(64)
    t = torch.tensor([[-0.8], [-0.25], [0.0], [0.4], [0.85]], dtype=DTYPE)
    for case in ("linear", "quadratic"):
        exact = lambda points, case=case: exact_solution(points, case)
        endpoint = endpoint_hypersingular_operator(exact, t, nodes, weights)
        taylor = taylor_hypersingular_operator(exact, t, nodes, weights)
        rhs = equation_rhs(t, case)
        torch.testing.assert_close(endpoint, rhs, rtol=2.0e-13, atol=2.0e-13)
        torch.testing.assert_close(taylor, rhs, rtol=2.0e-13, atol=2.0e-13)
        torch.testing.assert_close(endpoint, taylor, rtol=2.0e-13, atol=2.0e-13)


def test_short_training_is_finite_and_reproducible() -> None:
    config = TrainingConfig(
        seed=17,
        epochs=6,
        learning_rate=2.0e-3,
        lbfgs_steps=0,
        hidden_width=8,
        hidden_layers=1,
        num_collocation=10,
        num_quadrature=20,
        report_every=100,
    )
    for case in ("linear", "quadratic"):
        for representation in ("endpoint", "taylor"):
            _, first = train_one(case, representation, config, verbose=False)
            _, second = train_one(case, representation, config, verbose=False)
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
