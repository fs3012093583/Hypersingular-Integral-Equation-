"""Parameterized PINN using the endpoint-value hypersingular formula.

For every fixed ``gamma`` this module solves

    t**4 u(t) + (2-t)(1+t) H[u](t) = f_gamma(t),

with manufactured solution ``u_gamma(t)=1+gamma*t``.  The operator ``H`` is
the independently implemented endpoint-value/derivative-difference form from
``neural_hypersingular_constant_endpoint``.  The architecture, initialization,
collocation rule, quadrature and optimizers match the earlier Taylor-form
parameter sweep so that the representation is the intended changed factor.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

from neural_network_solvers.neural_hypersingular_constant_endpoint import endpoint_hypersingular_operator
from neural_network_solvers.neural_hypersingular_parameterized_subtraction import (
    DEFAULT_GAMMAS,
    DTYPE,
    ParameterizedTanhNetwork,
    TrainingConfig,
    _as_column,
    coefficient_a,
    coefficient_b,
    equation_rhs,
    exact_solution,
    gauss_legendre_rule,
    normalized_equation_residual,
    set_reproducible_seed,
)


def equation_operator_value(
    u: Callable[[Tensor], Tensor],
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the parameterized equation with the endpoint representation."""
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    if u_value.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * endpoint_hypersingular_operator(
        u, t_col, quadrature_nodes, quadrature_weights, near_zero_threshold
    )


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs < 0 or config.lbfgs_steps < 0 or config.epochs + config.lbfgs_steps < 1:
        raise ValueError("at least one Adam or L-BFGS step is required")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.num_collocation < 2 or config.num_quadrature < 2:
        raise ValueError("num_collocation and num_quadrature must both be at least 2")


def train_single_gamma(
    gamma: float, config: TrainingConfig, verbose: bool = True
) -> tuple[ParameterizedTanhNetwork, dict[str, object]]:
    """Train one fixed-gamma model and report full-domain solution errors."""
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite")
    _validate_config(config)
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)
    gamma_tensor = torch.tensor(float(gamma), dtype=DTYPE, device=device)
    model = ParameterizedTanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    quadrature_nodes, quadrature_weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def bound_model(points: Tensor) -> Tensor:
        return model(points, gamma_tensor)

    def objective() -> Tensor:
        operator_value = equation_operator_value(
            bound_model,
            collocation_points,
            quadrature_nodes,
            quadrature_weights,
            config.near_zero_threshold,
        )
        normalized = normalized_equation_residual(operator_value, collocation_points, gamma_tensor)
        return torch.mean(normalized.square())

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (epoch == 1 or epoch % config.report_every == 0 or epoch == config.epochs):
            print(f"gamma={gamma:g} epoch={epoch:5d} endpoint_loss={losses[-1]:.6e}")

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=config.lbfgs_steps,
            history_size=50,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss = objective()
            if not torch.isfinite(closure_loss):
                raise FloatingPointError("non-finite L-BFGS closure loss")
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        post_step_loss = objective()
        if not torch.isfinite(post_step_loss):
            raise FloatingPointError("non-finite loss after L-BFGS")
        losses.append(float(post_step_loss.detach().cpu()))
        if verbose:
            print(f"gamma={gamma:g} lbfgs_max_iter={config.lbfgs_steps:4d} endpoint_loss={losses[-1]:.6e}")

    model.eval()
    solution_points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    residual_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        raw_operator_values = equation_operator_value(
            bound_model,
            residual_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        solution_error = bound_model(solution_points) - exact_solution(solution_points, gamma_tensor)
        interior_error = bound_model(residual_points) - exact_solution(residual_points, gamma_tensor)
        raw_residual = raw_operator_values - equation_rhs(residual_points, gamma_tensor)
        balanced_residual = 0.5 * (1.0 - residual_points.square()) * raw_residual
        endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
        endpoint_errors = (bound_model(endpoint_points) - exact_solution(endpoint_points, gamma_tensor)).abs().reshape(-1)
        metrics: dict[str, object] = {
            "gamma": float(gamma),
            "config": asdict(config),
            "operator_representation": "endpoint_values_with_derivative_difference",
            "initial_normalized_loss": losses[0],
            "final_normalized_loss": losses[-1],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "solution_evaluation_interval": [-1.0, 1.0],
            "residual_evaluation_interval": [-0.95, 0.95],
            "num_solution_evaluation_points": int(solution_points.numel()),
            "num_residual_evaluation_points": int(residual_points.numel()),
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "rms_solution_error": float(torch.sqrt(torch.mean(solution_error.square())).cpu()),
            "interior_max_abs_solution_error": float(interior_error.abs().max().cpu()),
            "interior_rms_solution_error": float(torch.sqrt(torch.mean(interior_error.square())).cpu()),
            "abs_solution_error_t_minus_1": float(endpoint_errors[0].cpu()),
            "abs_solution_error_t_plus_1": float(endpoint_errors[1].cpu()),
            "max_endpoint_abs_solution_error": float(endpoint_errors.max().cpu()),
            "max_abs_equation_residual": float(raw_residual.abs().max().cpu()),
            "rms_equation_residual": float(torch.sqrt(torch.mean(raw_residual.square())).cpu()),
            "max_abs_balanced_residual": float(balanced_residual.abs().max().cpu()),
            "rms_balanced_residual": float(torch.sqrt(torch.mean(balanced_residual.square())).cpu()),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_sweep_plot(
    results: Sequence[tuple[float, ParameterizedTanhNetwork, dict[str, object]]],
    config: TrainingConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must contain at least one gamma solve")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False, constrained_layout=True
    )
    for axis, (gamma, model, metrics) in zip(axes.flat, results):
        device = torch.device(config.device)
        grid = torch.linspace(-1.0, 1.0, 401, dtype=DTYPE, device=device).reshape(-1, 1)
        gamma_tensor = torch.tensor(gamma, dtype=DTYPE, device=device)
        with torch.no_grad():
            values = model(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
            exact = exact_solution(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
        grid_np = grid.detach().cpu().numpy().reshape(-1)
        axis.plot(grid_np, exact, "k--", linewidth=1.25, label="exact")
        axis.plot(grid_np, values, linewidth=1.8, label="endpoint-form PINN")
        axis.set(
            title=f"gamma={gamma:g}; full RMS error={metrics['rms_solution_error']:.2e}",
            xlabel="t",
            ylabel="u(t)",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    for axis in axes.flat[len(results):]:
        axis.remove()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, action="append")
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--lbfgs-steps", type=int, default=TrainingConfig.lbfgs_steps)
    parser.add_argument("--hidden-width", type=int, default=TrainingConfig.hidden_width)
    parser.add_argument("--hidden-layers", type=int, default=TrainingConfig.hidden_layers)
    parser.add_argument("--num-collocation", type=int, default=TrainingConfig.num_collocation)
    parser.add_argument("--num-quadrature", type=int, default=TrainingConfig.num_quadrature)
    parser.add_argument("--near-zero-threshold", type=float, default=TrainingConfig.near_zero_threshold)
    parser.add_argument("--report-every", type=int, default=TrainingConfig.report_every)
    parser.add_argument("--device", default=TrainingConfig.device)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--png-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_argument_parser().parse_args(argv)
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_steps,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        num_collocation=args.num_collocation,
        num_quadrature=args.num_quadrature,
        near_zero_threshold=args.near_zero_threshold,
        report_every=args.report_every,
        device=args.device,
    )
    gammas = tuple(args.gamma) if args.gamma is not None else DEFAULT_GAMMAS
    results: list[tuple[float, ParameterizedTanhNetwork, dict[str, object]]] = []
    for gamma in gammas:
        model, metrics = train_single_gamma(gamma, config)
        results.append((gamma, model, metrics))
    summary: dict[str, object] = {
        "config": asdict(config),
        "operator_representation": "endpoint_values_with_derivative_difference",
        "gammas": list(gammas),
        "runs": [item[2] for item in results],
    }
    if args.json_out is not None:
        save_metrics_json(summary, args.json_out)
        print(f"wrote sweep JSON: {args.json_out}")
    if args.png_out is not None:
        save_sweep_plot(results, config, args.png_out)
        print(f"wrote sweep PNG: {args.png_out}")
    for _, _, metrics in results:
        print(
            f"gamma={metrics['gamma']:g} final_loss={metrics['final_normalized_loss']:.6e} "
            f"full_rms_solution_error={metrics['rms_solution_error']:.6e} "
            f"interior_rms_equation_residual={metrics['rms_equation_residual']:.6e}"
        )
    return summary


if __name__ == "__main__":
    main()
