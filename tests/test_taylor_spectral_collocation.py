"""Focused checks for the matched Chebyshev Taylor baseline."""

import numpy as np
import pytest
import torch

from conventional_solvers.taylor_spectral_collocation import (
    SpectralConfig,
    exact_solution,
    gauss_legendre_rule,
    manufactured_rhs,
    run,
    solve_linear_chebyshev,
)
from neural_network_solvers.general_taylor_finite_part import taylor_finite_part_operator


def test_manufactured_rhs_is_the_same_discrete_taylor_equation() -> None:
    points = torch.tensor([[-0.73], [-0.21], [0.13], [0.67]], dtype=torch.float64)
    nodes, weights = gauss_legendre_rule(128)
    coupling = 0.05
    rhs = manufactured_rhs(
        "exp", points, order=2, coupling=coupling,
        quadrature_nodes=nodes, quadrature_weights=weights, continuation_order=2,
    )
    finite_part = taylor_finite_part_operator(
        torch.exp, points, nodes, weights, order=2, continuation_order=2,
    )
    expected = torch.exp(points) + coupling * (1.0 - points.square()) * finite_part
    torch.testing.assert_close(rhs, expected, rtol=2.0e-13, atol=2.0e-13)


def test_linear_m2_exp_chebyshev_baseline_reaches_reasonable_error() -> None:
    result = run(
        SpectralConfig(
            order=2, family="exp", degree=12, collocation_points=16,
            quadrature_points=96, validation_points=81,
            validation_quadrature_points=160,
        )
    )
    assert result["status"] == "ok"
    assert float(result["solution_rmse_closed_interval"]) < 1.0e-8
    assert float(result["validation_residual_rmse"]) < 1.0e-8
    assert result["residual_interval"] == [-0.98, 0.98]


def test_spectral_interface_refuses_unmatched_nonlinear_case() -> None:
    with pytest.raises(NotImplementedError, match="mu=0"):
        solve_linear_chebyshev(SpectralConfig(nonlinear_coefficient=0.1))
