"""Pilot for reducing high-order finite-part equations to one Cauchy transform.

For

    I_m[u](t) = FP int_-1^1 u(tau)/(tau-t)^m d tau,

the distributional/finite-part identity

    I_m[u] = (d/dt)**(m-1) I_1[u] / (m-1)!

suggests an auxiliary field ``v = I_1[u]``.  This pilot jointly trains a
solution network ``u`` and a Cauchy-transform network ``v`` using only a
first-order singular integral in the coupling constraint.  It tests whether
the reduction is numerically trainable for orders two through four.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from torch import Tensor, nn

from neural_network_solvers.general_taylor_finite_part import (
    taylor_finite_part_operator,
    value_and_derivatives,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    set_reproducible_seed,
)


def exact_cauchy_exp(t: mp.mpf) -> mp.mpf:
    return mp.exp(t) * (mp.ei(1 - t) - mp.ei(-1 - t))


def exact_finite_part_exp(t: float, order: int) -> float:
    t_mp = mp.mpf(str(t))
    derivative = mp.diff(exact_cauchy_exp, t_mp, order - 1)
    return float(derivative / mp.factorial(order - 1))


def exact_cauchy_exp_array(points: Tensor) -> Tensor:
    values = [float(exact_cauchy_exp(mp.mpf(str(float(point))))) for point in points.reshape(-1)]
    return torch.tensor(values, dtype=points.dtype, device=points.device).reshape(-1, 1)


def exact_rhs(points: Tensor, order: int, coupling: float) -> Tensor:
    singular = torch.tensor(
        [exact_finite_part_exp(float(point), order) for point in points.reshape(-1)],
        dtype=points.dtype,
        device=points.device,
    ).reshape(-1, 1)
    endpoint_balance = (1.0 - points.square()).pow(order - 1)
    return torch.exp(points) + coupling * endpoint_balance * singular


class SmoothNetwork(nn.Module):
    def __init__(self, width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        return self.network(points)


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20260918
    adam_steps: int = 1200
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 120
    width: int = 32
    hidden_layers: int = 3
    collocation_points: int = 64
    training_quadrature: int = 64
    validation_quadrature: int = 192
    coupling: float = 0.05
    cauchy_constraint_weight: float = 1.0


def _loss(
    solution_network: nn.Module,
    transform_network: nn.Module,
    collocation: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    order: int,
    config: PilotConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    points = collocation.detach().clone().requires_grad_(True)
    solution = solution_network(points)
    transform_derivatives = value_and_derivatives(
        transform_network,
        points,
        order - 1,
    )
    singular = transform_derivatives[order - 1] / math.factorial(order - 1)
    endpoint_balance = (1.0 - points.square()).pow(order - 1)
    equation_residual = solution + config.coupling * endpoint_balance * singular - rhs

    cauchy_from_solution = taylor_finite_part_operator(
        solution_network,
        points,
        nodes,
        weights,
        order=1,
        continuation_order=1,
        threshold_mode="balanced",
    )
    transform_residual = transform_derivatives[0] - cauchy_from_solution
    equation_loss = torch.mean(equation_residual.square())
    transform_loss = torch.mean(transform_residual.square())
    total = equation_loss + config.cauchy_constraint_weight * transform_loss
    return total, equation_loss, transform_loss


def train_one(order: int, config: PilotConfig, *, verbose: bool = True) -> dict[str, object]:
    if order < 2:
        raise ValueError("the auxiliary reduction pilot targets order >= 2")
    set_reproducible_seed(config.seed + order)
    solution_network = SmoothNetwork(config.width, config.hidden_layers).to(dtype=DTYPE)
    transform_network = SmoothNetwork(config.width, config.hidden_layers).to(dtype=DTYPE)
    collocation_np = np.polynomial.legendre.leggauss(config.collocation_points)[0]
    collocation = torch.tensor(collocation_np, dtype=DTYPE).reshape(-1, 1)
    rhs = exact_rhs(collocation, order, config.coupling)
    nodes, weights = gauss_legendre_rule(config.training_quadrature)
    parameters = list(solution_network.parameters()) + list(transform_network.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, equation_loss, transform_loss = _loss(
            solution_network,
            transform_network,
            collocation,
            rhs,
            nodes,
            weights,
            order,
            config,
        )
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            record = {
                "stage": "adam",
                "step": step + 1,
                "total_loss": float(total.detach()),
                "equation_loss": float(equation_loss.detach()),
                "transform_loss": float(transform_loss.detach()),
            }
            history.append(record)
            if verbose:
                print(f"order={order} {record}", flush=True)

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total, _, _ = _loss(
                solution_network,
                transform_network,
                collocation,
                rhs,
                nodes,
                weights,
                order,
                config,
            )
            total.backward()
            return total

        lbfgs.step(closure)

    elapsed = time.perf_counter() - started
    final_total, final_equation_loss, final_transform_loss = _loss(
        solution_network,
        transform_network,
        collocation,
        rhs,
        nodes,
        weights,
        order,
        config,
    )

    evaluation = torch.linspace(-0.97, 0.97, 241, dtype=DTYPE).reshape(-1, 1)
    validation_nodes, validation_weights = gauss_legendre_rule(config.validation_quadrature)
    with torch.no_grad():
        prediction = solution_network(evaluation)
        solution_error = prediction - torch.exp(evaluation)
        transform_prediction = transform_network(evaluation)
        transform_error = transform_prediction - exact_cauchy_exp_array(evaluation)

    residual_points = evaluation.detach().clone().requires_grad_(True)
    residual_solution = solution_network(residual_points)
    transform_derivatives = value_and_derivatives(
        transform_network,
        residual_points,
        order - 1,
    )
    residual_rhs = exact_rhs(residual_points.detach(), order, config.coupling)
    endpoint_balance = (1.0 - residual_points.square()).pow(order - 1)
    equation_residual = (
        residual_solution
        + config.coupling
        * endpoint_balance
        * transform_derivatives[order - 1]
        / math.factorial(order - 1)
        - residual_rhs
    )
    cauchy_from_solution = taylor_finite_part_operator(
        solution_network,
        residual_points,
        validation_nodes,
        validation_weights,
        order=1,
        continuation_order=1,
        threshold_mode="balanced",
    )
    transform_constraint = transform_derivatives[0] - cauchy_from_solution

    metrics: dict[str, object] = {
        "order": order,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "final_total_loss": float(final_total.detach()),
        "final_equation_loss": float(final_equation_loss.detach()),
        "final_transform_loss": float(final_transform_loss.detach()),
        "solution_rms_error": float(torch.sqrt(torch.mean(solution_error.square()))),
        "solution_max_abs_error": float(torch.max(torch.abs(solution_error))),
        "transform_rms_error": float(torch.sqrt(torch.mean(transform_error.square()))),
        "validation_equation_rms": float(
            torch.sqrt(torch.mean(equation_residual.detach().square()))
        ),
        "validation_transform_constraint_rms": float(
            torch.sqrt(torch.mean(transform_constraint.detach().square()))
        ),
        "history": history,
    }
    if verbose:
        print(f"order={order} final={metrics}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", action="append", type=int)
    parser.add_argument("--adam-steps", type=int, default=PilotConfig.adam_steps)
    parser.add_argument("--lbfgs-steps", type=int, default=PilotConfig.lbfgs_steps)
    parser.add_argument("--width", type=int, default=PilotConfig.width)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/auxiliary_cauchy_pinn_pilot"),
    )
    args = parser.parse_args()
    orders = tuple(args.order) if args.order else (2, 3, 4)
    if any(order < 2 for order in orders):
        raise ValueError("all requested orders must be at least two")
    config = PilotConfig(
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        width=args.width,
    )
    mp.mp.dps = 60
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [train_one(order, config) for order in orders]
    payload = {
        "experiment": "auxiliary_cauchy_transform_order_reduction",
        "identity": "I_m[u] = d_t^(m-1) I_1[u] / (m-1)!",
        "exact_solution": "exp(t)",
        "orders": list(orders),
        "config": asdict(config),
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
