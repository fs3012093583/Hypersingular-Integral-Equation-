"""Matched PINN pilot using the general cancellation-aware Taylor operator."""

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

from experiments.auxiliary_cauchy_pinn_pilot import exact_rhs
from neural_network_solvers.general_taylor_finite_part import taylor_finite_part_operator
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    set_reproducible_seed,
)


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
class TaylorPilotConfig:
    seed: int = 20260918
    adam_steps: int = 1200
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 120
    width: int = 32
    hidden_layers: int = 3
    collocation_points: int = 64
    training_quadrature: int = 64
    validation_quadrature: int = 192
    collocation_limit: float = 0.97
    coupling: float = 0.05
    continuation_order: int = 2


def _equation_residual(
    network: nn.Module,
    points: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    order: int,
    config: TaylorPilotConfig,
) -> Tensor:
    solution = network(points)
    finite_part = taylor_finite_part_operator(
        network,
        points,
        nodes,
        weights,
        order=order,
        continuation_order=config.continuation_order,
        threshold_mode="balanced",
    )
    endpoint_balance = (1.0 - points.square()).pow(order - 1)
    return solution + config.coupling * endpoint_balance * finite_part - rhs


def train_one(order: int, config: TaylorPilotConfig, *, verbose: bool = True) -> dict[str, object]:
    set_reproducible_seed(config.seed + order)
    network = SmoothNetwork(config.width, config.hidden_layers).to(dtype=DTYPE)
    collocation_np = np.polynomial.legendre.leggauss(config.collocation_points)[0]
    collocation = torch.tensor(
        config.collocation_limit * collocation_np,
        dtype=DTYPE,
    ).reshape(-1, 1)
    rhs = exact_rhs(collocation, order, config.coupling)
    nodes, weights = gauss_legendre_rule(config.training_quadrature)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        residual = _equation_residual(
            network,
            collocation,
            rhs,
            nodes,
            weights,
            order,
            config,
        )
        loss = torch.mean(residual.square())
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            record = {"stage": "adam", "step": step + 1, "loss": float(loss.detach())}
            history.append(record)
            if verbose:
                print(f"order={order} {record}", flush=True)

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            network.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            residual = _equation_residual(
                network,
                collocation,
                rhs,
                nodes,
                weights,
                order,
                config,
            )
            loss = torch.mean(residual.square())
            loss.backward()
            return loss

        lbfgs.step(closure)

    elapsed = time.perf_counter() - started
    final_residual = _equation_residual(
        network,
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
        prediction = network(evaluation)
        solution_error = prediction - torch.exp(evaluation)
    validation_nodes, validation_weights = gauss_legendre_rule(config.validation_quadrature)
    validation_rhs = exact_rhs(evaluation, order, config.coupling)
    validation_residual = _equation_residual(
        network,
        evaluation,
        validation_rhs,
        validation_nodes,
        validation_weights,
        order,
        config,
    )
    metrics: dict[str, object] = {
        "order": order,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
        "final_training_loss": float(torch.mean(final_residual.detach().square())),
        "solution_rms_error": float(torch.sqrt(torch.mean(solution_error.square()))),
        "solution_max_abs_error": float(torch.max(torch.abs(solution_error))),
        "validation_equation_rms": float(
            torch.sqrt(torch.mean(validation_residual.detach().square()))
        ),
        "history": history,
    }
    if verbose:
        print(f"order={order} final={metrics}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", action="append", type=int)
    parser.add_argument("--adam-steps", type=int, default=TaylorPilotConfig.adam_steps)
    parser.add_argument("--lbfgs-steps", type=int, default=TaylorPilotConfig.lbfgs_steps)
    parser.add_argument("--width", type=int, default=TaylorPilotConfig.width)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/general_taylor_pinn_pilot"),
    )
    args = parser.parse_args()
    orders = tuple(args.order) if args.order else (2, 3, 4)
    config = TaylorPilotConfig(
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        width=args.width,
    )
    mp.mp.dps = 60
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [train_one(order, config) for order in orders]
    payload = {
        "experiment": "general_cancellation_aware_taylor_pinn",
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
