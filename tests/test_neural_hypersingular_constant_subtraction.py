import math

import torch
from torch import nn

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TrainingConfig,
    equation_rhs,
    gauss_legendre_rule,
    hypersingular_operator,
    train_constant_solution,
)


class AffineFunction(nn.Module):
    def __init__(self, intercept: float, slope: float) -> None:
        super().__init__()
        self.intercept = intercept
        self.slope = slope

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Keep the result connected to t even for the constant test case.
        return self.intercept + self.slope * t


def test_constant_solution_operator_identity() -> None:
    nodes, weights = gauss_legendre_rule(32)
    t = torch.tensor([[-0.8], [-0.3], [0.0], [0.45], [0.85]], dtype=DTYPE)
    actual = hypersingular_operator(lambda points: torch.ones_like(points), t, nodes, weights)
    expected = equation_rhs(t)
    torch.testing.assert_close(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_affine_solution_operator_identity() -> None:
    gamma = -0.37
    nodes, weights = gauss_legendre_rule(32)
    t = torch.tensor([[-0.75], [-0.2], [0.1], [0.55], [0.9]], dtype=DTYPE)
    actual = hypersingular_operator(AffineFunction(1.0, gamma), t, nodes, weights)
    expected = (
        -(1.0 + gamma * t) * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
        + gamma * torch.log((1.0 - t) / (1.0 + t))
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_quadratic_identity_including_regular_integral_and_diagonal_limit() -> None:
    nodes, weights = gauss_legendre_rule(32)
    # Reusing the integration nodes as collocation points exercises the
    # u''(t)/2 diagonal replacement rather than just the off-diagonal quotient.
    t = nodes.clone()
    actual = hypersingular_operator(lambda points: points.square(), t, nodes, weights)
    expected = (
        -t.square() * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
        + 2.0 * t * torch.log((1.0 - t) / (1.0 + t))
        + 2.0
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_short_training_smoke_is_finite_and_reproducible() -> None:
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
    assert math.isfinite(first["final_normalized_loss"])
    assert math.isfinite(first["rms_solution_error"])
    assert first["loss_history"] == second["loss_history"]
