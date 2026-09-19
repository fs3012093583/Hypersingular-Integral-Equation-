"""Endpoint-factored first-derivative Cauchy chain for finite-part equations.

The auxiliary Cauchy transform is split as

    C[u](t) = u(t) log((1-t)/(1+t)) + h(t),

where ``h`` is regular.  Instead of differentiating one approximation of
``C[u]`` many times, the network outputs ``q_j ~= d^j C[u]/dt^j`` and enforces
only first-derivative links ``q_j = d q_{j-1}/dt``.  The high-order finite-part
operator is ``q_{m-1}/(m-1)!``.
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

from experiments.auxiliary_cauchy_pinn_pilot import (
    exact_cauchy_exp_array,
    exact_rhs,
)
from neural_network_solvers.general_taylor_finite_part import (
    regularized_taylor_quotient,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    set_reproducible_seed,
)


def _pointwise_derivative(values: Tensor, points: Tensor) -> Tensor:
    derivative = torch.autograd.grad(
        values,
        points,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]
    if derivative is None:
        raise RuntimeError("auxiliary derivative is unavailable")
    return derivative


class SmoothNetwork(nn.Module):
    def __init__(self, output_dim: int, width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        return self.network(points)


@dataclass(frozen=True)
class ChainConfig:
    seed: int = 20260918
    adam_steps: int = 1800
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 180
    width: int = 32
    hidden_layers: int = 3
    collocation_points: int = 64
    training_quadrature: int = 64
    validation_quadrature: int = 192
    collocation_limit: float = 0.97
    coupling: float = 0.05
    regular_constraint_weight: float = 1.0
    chain_constraint_weight: float = 1.0


def _regular_cauchy_part(
    solution_network: nn.Module,
    points: Tensor,
    nodes: Tensor,
    weights: Tensor,
) -> Tensor:
    quotient, _ = regularized_taylor_quotient(
        solution_network,
        points,
        nodes,
        order=1,
        continuation_order=1,
        threshold_mode="balanced",
    )
    return torch.sum(quotient * weights.reshape(1, -1), dim=1, keepdim=True)


def _loss(
    solution_network: nn.Module,
    auxiliary_network: nn.Module,
    collocation: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    order: int,
    config: ChainConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    points = collocation.detach().clone().requires_grad_(True)
    solution = solution_network(points)
    auxiliary = auxiliary_network(points)
    regular_part = auxiliary[:, 0:1]
    logarithm = torch.log((1.0 - points) / (1.0 + points))
    q_previous = solution * logarithm + regular_part

    regular_target = _regular_cauchy_part(solution_network, points, nodes, weights)
    regular_residual = regular_part - regular_target

    chain_losses: list[Tensor] = []
    endpoint_scale = 1.0 - points.square()
    for derivative_order in range(1, order):
        predicted_derivative = auxiliary[:, derivative_order : derivative_order + 1]
        actual_derivative = _pointwise_derivative(q_previous, points)
        # q_j has endpoint growth of order approximately (1-t^2)^(-j).
        # Scaling the link residual prevents the endpoint samples from
        # dominating every other part of the coupled optimization problem.
        scaled_residual = endpoint_scale.pow(derivative_order) * (
            predicted_derivative - actual_derivative
        )
        chain_losses.append(torch.mean(scaled_residual.square()))
        q_previous = predicted_derivative

    singular = q_previous / math.factorial(order - 1)
    equation_residual = (
        solution
        + config.coupling * endpoint_scale.pow(order - 1) * singular
        - rhs
    )
    equation_loss = torch.mean(equation_residual.square())
    regular_loss = torch.mean(regular_residual.square())
    chain_loss = torch.stack(chain_losses).mean()
    total = (
        equation_loss
        + config.regular_constraint_weight * regular_loss
        + config.chain_constraint_weight * chain_loss
    )
    return total, equation_loss, regular_loss, chain_loss


def train_one(order: int, config: ChainConfig, *, verbose: bool = True) -> dict[str, object]:
    if order < 2:
        raise ValueError("order must be at least two")
    set_reproducible_seed(config.seed + order)
    solution_network = SmoothNetwork(1, config.width, config.hidden_layers).to(dtype=DTYPE)
    auxiliary_network = SmoothNetwork(order, config.width, config.hidden_layers).to(dtype=DTYPE)
    collocation_np = np.polynomial.legendre.leggauss(config.collocation_points)[0]
    collocation = torch.tensor(
        config.collocation_limit * collocation_np,
        dtype=DTYPE,
    ).reshape(-1, 1)
    rhs = exact_rhs(collocation, order, config.coupling)
    nodes, weights = gauss_legendre_rule(config.training_quadrature)
    parameters = list(solution_network.parameters()) + list(auxiliary_network.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, equation_loss, regular_loss, chain_loss = _loss(
            solution_network,
            auxiliary_network,
            collocation,
            rhs,
            nodes,
            weights,
            order,
            config,
        )
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 300 == 0:
            record = {
                "stage": "adam",
                "step": step + 1,
                "total_loss": float(total.detach()),
                "equation_loss": float(equation_loss.detach()),
                "regular_loss": float(regular_loss.detach()),
                "chain_loss": float(chain_loss.detach()),
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
            total, _, _, _ = _loss(
                solution_network,
                auxiliary_network,
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
    final_total, final_equation, final_regular, final_chain = _loss(
        solution_network,
        auxiliary_network,
        collocation,
        rhs,
        nodes,
        weights,
        order,
        config,
    )

    evaluation = torch.linspace(
        -config.collocation_limit,
        config.collocation_limit,
        241,
        dtype=DTYPE,
    ).reshape(-1, 1)
    with torch.no_grad():
        solution_prediction = solution_network(evaluation)
        solution_error = solution_prediction - torch.exp(evaluation)
        auxiliary_prediction = auxiliary_network(evaluation)
        logarithm = torch.log((1.0 - evaluation) / (1.0 + evaluation))
        cauchy_prediction = solution_prediction * logarithm + auxiliary_prediction[:, 0:1]
        cauchy_error = cauchy_prediction - exact_cauchy_exp_array(evaluation)

    metrics: dict[str, object] = {
        "order": order,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "final_total_loss": float(final_total.detach()),
        "final_equation_loss": float(final_equation.detach()),
        "final_regular_loss": float(final_regular.detach()),
        "final_chain_loss": float(final_chain.detach()),
        "solution_rms_error": float(torch.sqrt(torch.mean(solution_error.square()))),
        "solution_max_abs_error": float(torch.max(torch.abs(solution_error))),
        "cauchy_transform_rms_error": float(torch.sqrt(torch.mean(cauchy_error.square()))),
        "history": history,
    }
    if verbose:
        print(f"order={order} final={metrics}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", action="append", type=int)
    parser.add_argument("--adam-steps", type=int, default=ChainConfig.adam_steps)
    parser.add_argument("--lbfgs-steps", type=int, default=ChainConfig.lbfgs_steps)
    parser.add_argument("--width", type=int, default=ChainConfig.width)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/endpoint_factored_cauchy_chain_pilot"),
    )
    args = parser.parse_args()
    orders = tuple(args.order) if args.order else (2, 3, 4)
    config = ChainConfig(
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        width=args.width,
    )
    mp.mp.dps = 60
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [train_one(order, config) for order in orders]
    payload = {
        "experiment": "endpoint_factored_first_derivative_cauchy_chain",
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
