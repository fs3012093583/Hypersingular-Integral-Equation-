"""Chebyshev collocation baseline for Taylor-regularized finite-part equations.

The baseline is deliberately built from the same Taylor-subtraction operator
used by the neural experiments.  It solves the *linear* manufactured problem

    u(t) + lambda (1 - t**2)**(m - 1) I_m[u](t) = f(t),  -1 < t < 1,

where ``I_m`` is the Cauchy principal value for ``m=1`` and the Hadamard
finite part for ``m >= 2``.  Nonlinear ``mu*u**3`` cases are accepted by the
manufacturing helpers, but are intentionally not solved here: this module is
a conventional linear least-squares reference, not an undocumented nonlinear
iteration method.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor

from neural_network_solvers.general_taylor_finite_part import taylor_finite_part_operator


SolutionFamily = Literal["exp", "sin", "rational"]
ThresholdMode = Literal["balanced", "fixed", "diagonal"]
SUPPORTED_FAMILIES: tuple[SolutionFamily, ...] = ("exp", "sin", "rational")


def exact_solution(name: SolutionFamily, t: Tensor) -> Tensor:
    """Return one of the smooth manufactured solutions, preserving autograd."""
    if name == "exp":
        return torch.exp(t)
    if name == "sin":
        return torch.sin(torch.pi * t)
    if name == "rational":
        return 1.0 / (1.0 + 0.5 * t)
    raise ValueError(f"unsupported solution family: {name}")


def exact_solution_numpy(name: SolutionFamily, t: np.ndarray) -> np.ndarray:
    if name == "exp":
        return np.exp(t)
    if name == "sin":
        return np.sin(np.pi * t)
    if name == "rational":
        return 1.0 / (1.0 + 0.5 * t)
    raise ValueError(f"unsupported solution family: {name}")


def gauss_legendre_rule(num_points: int, dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor]:
    """A float-selectable Gauss--Legendre rule on ``[-1, 1]``."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    return (
        torch.as_tensor(nodes, dtype=dtype).reshape(-1, 1),
        torch.as_tensor(weights, dtype=dtype).reshape(-1, 1),
    )


def chebyshev_gauss_nodes(count: int, limit: float = 1.0) -> np.ndarray:
    """Interior Chebyshev nodes; ``limit < 1`` is suitable for residuals."""
    if count < 1 or not 0.0 < limit <= 1.0:
        raise ValueError("count must be positive and limit must lie in (0, 1]")
    k = np.arange(count, dtype=float)
    return limit * np.cos(np.pi * (k + 0.5) / count)


def chebyshev_polynomial(degree: int, points: Tensor) -> Tensor:
    """Evaluate ``T_degree(points)`` with an autograd-safe recurrence."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree == 0:
        return torch.ones_like(points)
    if degree == 1:
        return points
    previous = torch.ones_like(points)
    current = points
    for _ in range(2, degree + 1):
        previous, current = current, 2.0 * points * current - previous
    return current


def chebyshev_series(coefficients: Tensor, points: Tensor) -> Tensor:
    """Evaluate a Chebyshev series without converting away derivatives."""
    values = torch.zeros_like(points)
    for degree, coefficient in enumerate(coefficients.reshape(-1)):
        values = values + coefficient * chebyshev_polynomial(degree, points)
    return values


def manufactured_rhs(
    family: SolutionFamily,
    points: Tensor,
    *,
    order: int,
    coupling: float,
    nonlinear_coefficient: float = 0.0,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    continuation_order: int = 2,
    threshold_mode: ThresholdMode = "balanced",
    near_diagonal_threshold: float | None = None,
    threshold_multiplier: float = 1.0,
) -> Tensor:
    """Manufacture the RHS with precisely the discretized Taylor operator.

    This shared definition keeps neural and conventional methods on the same
    equation.  Points must be strictly interior because the finite-part
    operator has genuine endpoint singular factors.
    """
    u = lambda x: exact_solution(family, x)
    u_value = u(points)
    finite_part = taylor_finite_part_operator(
        u,
        points,
        quadrature_nodes,
        quadrature_weights,
        order=order,
        continuation_order=continuation_order,
        threshold_mode=threshold_mode,
        near_diagonal_threshold=near_diagonal_threshold,
        threshold_multiplier=threshold_multiplier,
    )
    balance = (1.0 - points.square()).pow(order - 1)
    return u_value + coupling * balance * finite_part + nonlinear_coefficient * u_value.pow(3)


@dataclass(frozen=True)
class SpectralConfig:
    order: int = 2
    family: SolutionFamily = "exp"
    coupling: float = 0.05
    nonlinear_coefficient: float = 0.0
    degree: int = 20
    collocation_points: int = 28
    quadrature_points: int = 160
    rhs_quadrature_points: int = 256
    validation_points: int = 161
    validation_quadrature_points: int = 320
    validation_limit: float = 0.98
    continuation_order: int = 2
    threshold_mode: ThresholdMode = "balanced"
    near_diagonal_threshold: float | None = None
    threshold_multiplier: float = 1.0
    rhs_continuation_order: int = 3
    rhs_threshold_multiplier: float = 1.0
    dtype: str = "float64"


@dataclass
class SpectralSolution:
    coefficients: np.ndarray
    config: SpectralConfig
    fit_residual_norm: float
    training_seconds: float

    def predict(self, points: np.ndarray | Tensor) -> np.ndarray:
        array = points.detach().cpu().numpy() if isinstance(points, Tensor) else np.asarray(points)
        return np.polynomial.chebyshev.chebval(array, self.coefficients)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float32 or float64")


def _basis_function(degree: int) -> Callable[[Tensor], Tensor]:
    return lambda x: chebyshev_polynomial(degree, x)


def solve_linear_chebyshev(config: SpectralConfig) -> SpectralSolution:
    """Solve the linear manufactured problem by Chebyshev least squares.

    ``nonlinear_coefficient != 0`` deliberately raises ``NotImplementedError``
    so benchmark output cannot accidentally describe a nonlinear spectral
    result as a matched conventional baseline.
    """
    if config.order < 1:
        raise ValueError("order must be positive")
    if config.degree < 0 or config.collocation_points < config.degree + 1:
        raise ValueError("collocation_points must be at least degree + 1")
    if config.nonlinear_coefficient != 0.0:
        raise NotImplementedError("the Chebyshev baseline supports linear mu=0 cases only")
    dtype = _torch_dtype(config.dtype)
    started = time.perf_counter()
    collocation_np = chebyshev_gauss_nodes(config.collocation_points, limit=config.validation_limit)
    collocation = torch.as_tensor(collocation_np, dtype=dtype).reshape(-1, 1)
    nodes, weights = gauss_legendre_rule(config.quadrature_points, dtype=dtype)
    rhs_nodes, rhs_weights = gauss_legendre_rule(config.rhs_quadrature_points, dtype=dtype)
    rhs = manufactured_rhs(
        config.family, collocation, order=config.order, coupling=config.coupling,
        nonlinear_coefficient=0.0, quadrature_nodes=rhs_nodes, quadrature_weights=rhs_weights,
        # The manufactured equation is always defined by one stable canonical
        # operator.  The tested method may be direct/fixed/balanced, but it is
        # not allowed to change the right-hand side and thereby solve a
        # different discrete problem.
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
        threshold_multiplier=config.rhs_threshold_multiplier,
    )
    matrix_columns: list[np.ndarray] = []
    balance = (1.0 - collocation.square()).pow(config.order - 1)
    for degree in range(config.degree + 1):
        basis_value = chebyshev_polynomial(degree, collocation)
        basis_operator = taylor_finite_part_operator(
            _basis_function(degree), collocation, nodes, weights, order=config.order,
            continuation_order=config.continuation_order, threshold_mode=config.threshold_mode,
            near_diagonal_threshold=config.near_diagonal_threshold,
            threshold_multiplier=config.threshold_multiplier,
        )
        column = basis_value + config.coupling * balance * basis_operator
        matrix_columns.append(column.detach().cpu().numpy().reshape(-1))
    matrix = np.column_stack(matrix_columns)
    target = rhs.detach().cpu().numpy().reshape(-1)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    fit = float(np.sqrt(np.mean((matrix @ coefficients - target) ** 2)))
    return SpectralSolution(coefficients, config, fit, time.perf_counter() - started)


def evaluate_solution(solution: SpectralSolution) -> dict[str, object]:
    """Compute closed-interval errors and a separate strictly-interior residual."""
    config = solution.config
    dtype = _torch_dtype(config.dtype)
    evaluation = np.linspace(-1.0, 1.0, config.validation_points)
    prediction = solution.predict(evaluation)
    error = prediction - exact_solution_numpy(config.family, evaluation)
    # The equation is never evaluated at endpoints: the finite-part operator's
    # endpoint factors are singular even though solution error is well-defined.
    residual_np = chebyshev_gauss_nodes(config.validation_points, config.validation_limit)
    residual_points = torch.as_tensor(residual_np, dtype=dtype).reshape(-1, 1)
    nodes, weights = gauss_legendre_rule(config.validation_quadrature_points, dtype=dtype)
    coefficients = torch.as_tensor(solution.coefficients, dtype=dtype)
    approximation = lambda x: chebyshev_series(coefficients, x)
    rhs = manufactured_rhs(
        config.family, residual_points, order=config.order, coupling=config.coupling,
        nonlinear_coefficient=0.0, quadrature_nodes=nodes, quadrature_weights=weights,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
        threshold_multiplier=config.rhs_threshold_multiplier,
    )
    value = approximation(residual_points)
    finite_part = taylor_finite_part_operator(
        approximation, residual_points, nodes, weights, order=config.order,
        continuation_order=config.continuation_order, threshold_mode=config.threshold_mode,
        near_diagonal_threshold=config.near_diagonal_threshold,
        threshold_multiplier=config.threshold_multiplier,
    )
    residual = value + config.coupling * (1.0 - residual_points.square()).pow(config.order - 1) * finite_part - rhs
    return {
        "solution_rmse_closed_interval": float(np.sqrt(np.mean(error**2))),
        "solution_max_abs_error_closed_interval": float(np.max(np.abs(error))),
        "validation_residual_rmse": float(torch.sqrt(torch.mean(residual.detach().square()))),
        "residual_interval": [-config.validation_limit, config.validation_limit],
        "validation_point_count": config.validation_points,
        "validation_quadrature_points": config.validation_quadrature_points,
    }


def run(config: SpectralConfig) -> dict[str, object]:
    started = time.perf_counter()
    solution = solve_linear_chebyshev(config)
    metrics = evaluate_solution(solution)
    return {
        "method": "chebyshev_taylor_collocation",
        "status": "ok",
        "config": asdict(config),
        "parameter_count": int(solution.coefficients.size),
        "training_seconds": solution.training_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
        "linear_fit_residual_rmse": solution.fit_residual_norm,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--family", choices=SUPPORTED_FAMILIES, default="exp")
    parser.add_argument("--lambda-value", dest="coupling", type=float, default=0.05)
    parser.add_argument("--mu", dest="nonlinear_coefficient", type=float, default=0.0)
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--collocation-points", type=int, default=28)
    parser.add_argument("--quadrature-points", type=int, default=160)
    parser.add_argument("--rhs-quadrature-points", type=int, default=256)
    parser.add_argument("--continuation-order", type=int, default=2)
    parser.add_argument(
        "--threshold-mode",
        choices=("balanced", "fixed", "diagonal"),
        default="balanced",
    )
    parser.add_argument("--near-diagonal-threshold", type=float)
    parser.add_argument("--threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--rhs-continuation-order", type=int, default=3)
    parser.add_argument("--rhs-threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SpectralConfig(**{key: value for key, value in vars(args).items() if key != "output"})
    payload = run(config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
