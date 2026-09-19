"""PINN experiment using the endpoint-value hypersingular representation.

On ``(-1, 1)`` the finite-part operator is evaluated as

    H[u](t) = -u(1)/(1-t) - u(-1)/(1+t)
              + u'(t) log((1-t)/(1+t))
              + int_-1^1 (u'(tau)-u'(t))/(tau-t) d tau.

This is the endpoint-value formula obtained by integration by parts.  It is
mathematically equivalent to the complete Taylor-subtraction formula for
smooth functions, but is implemented independently here so that the two
representations can be checked and trained separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TanhNetwork,
    TrainingConfig,
    equation_rhs,
    exact_solution,
    gauss_legendre_rule,
    normalized_equation_residual,
    set_reproducible_seed,
)


def _as_column(value: Tensor | Sequence[float], device: torch.device | str | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def _first_derivative(values: Tensor, inputs: Tensor) -> Tensor:
    if not values.requires_grad:
        return torch.zeros_like(inputs)
    derivative = torch.autograd.grad(
        values,
        inputs,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return derivative if derivative is not None else torch.zeros_like(inputs)


def endpoint_hypersingular_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the endpoint-value/derivative-difference finite-part formula."""
    if near_zero_threshold <= 0.0:
        raise ValueError("near_zero_threshold must be positive")
    device = quadrature_nodes.device
    t_col = _as_column(t, device=device).detach().clone().requires_grad_(True)
    tau_col = _as_column(quadrature_nodes, device=device).detach().clone().requires_grad_(True)
    weights = _as_column(quadrature_weights, device=device)
    if tau_col.shape != weights.shape:
        raise ValueError("quadrature_nodes and quadrature_weights must have the same length")

    u_t = _as_column(u(t_col), device=device)
    if u_t.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    u_prime_t = _first_derivative(u_t, t_col)
    u_second_t = _first_derivative(u_prime_t, t_col) if u_prime_t.requires_grad else torch.zeros_like(t_col)

    u_tau = _as_column(u(tau_col), device=device)
    if u_tau.shape != tau_col.shape:
        raise ValueError("u(tau) must return one scalar value per quadrature point")
    u_prime_tau = _first_derivative(u_tau, tau_col).reshape(1, -1)

    endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
    endpoint_values = _as_column(u(endpoint_points), device=device)
    if endpoint_values.shape != endpoint_points.shape:
        raise ValueError("u must return one scalar value at each endpoint")
    u_minus_one = endpoint_values[0:1]
    u_plus_one = endpoint_values[1:2]

    tau = tau_col.reshape(1, -1)
    delta = tau - t_col
    near_zero = delta.abs() <= near_zero_threshold
    safe_delta = torch.where(near_zero, torch.ones_like(delta), delta)
    quotient = (u_prime_tau - u_prime_t) / safe_delta
    integrand = torch.where(near_zero, u_second_t, quotient)
    derivative_difference_integral = torch.sum(integrand * weights.reshape(1, -1), dim=1, keepdim=True)

    endpoint_term = -u_plus_one / (1.0 - t_col) - u_minus_one / (1.0 + t_col)
    logarithmic_term = u_prime_t * torch.log((1.0 - t_col) / (1.0 + t_col))
    return endpoint_term + logarithmic_term + derivative_difference_integral


def _solution_metrics(model: nn.Module, points: Tensor) -> dict[str, float]:
    with torch.no_grad():
        error = model(points) - exact_solution(points)
    return {
        "max_abs_solution_error": float(error.abs().max().cpu()),
        "rms_solution_error": float(torch.sqrt(torch.mean(error.square())).cpu()),
    }


def train_constant_solution(config: TrainingConfig, verbose: bool = True) -> tuple[TanhNetwork, dict[str, object]]:
    """Train the constant test equation with the endpoint-value operator."""
    if config.epochs < 0 or config.lbfgs_steps < 0 or config.epochs + config.lbfgs_steps < 1:
        raise ValueError("at least one Adam or L-BFGS step is required")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)

    model = TanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    quadrature_nodes, quadrature_weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def objective() -> Tensor:
        operator_value = endpoint_hypersingular_operator(
            model,
            collocation_points,
            quadrature_nodes,
            quadrature_weights,
            config.near_zero_threshold,
        )
        return torch.mean(normalized_equation_residual(operator_value, collocation_points).square())

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
            print(f"epoch={epoch:5d} endpoint_loss={losses[-1]:.6e}")

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=1, history_size=50, line_search_fn="strong_wolfe"
        )
        for step in range(1, config.lbfgs_steps + 1):
            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                closure_loss = objective()
                closure_loss.backward()
                return closure_loss

            lbfgs.step(closure)
            post_step_loss = objective()
            if not torch.isfinite(post_step_loss):
                raise FloatingPointError(f"non-finite L-BFGS loss at step {step}")
            losses.append(float(post_step_loss.detach().cpu()))
            if verbose and (step == 1 or step % config.report_every == 0 or step == config.lbfgs_steps):
                print(f"lbfgs_step={step:4d} endpoint_loss={losses[-1]:.6e}")

    model.eval()
    solution_points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    residual_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        operator_values = endpoint_hypersingular_operator(
            model, residual_points, validation_nodes, validation_weights, config.near_zero_threshold
        )
    with torch.no_grad():
        residual = operator_values - equation_rhs(residual_points)
        interior_error = model(residual_points) - exact_solution(residual_points)
        endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
        endpoint_errors = (model(endpoint_points) - exact_solution(endpoint_points)).abs().reshape(-1)
        metrics: dict[str, object] = {
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
            **_solution_metrics(model, solution_points),
            "interior_max_abs_solution_error": float(interior_error.abs().max().cpu()),
            "interior_rms_solution_error": float(torch.sqrt(torch.mean(interior_error.square())).cpu()),
            "abs_solution_error_t_minus_1": float(endpoint_errors[0].cpu()),
            "abs_solution_error_t_plus_1": float(endpoint_errors[1].cpu()),
            "max_endpoint_abs_solution_error": float(endpoint_errors.max().cpu()),
            "max_abs_equation_residual": float(residual.abs().max().cpu()),
            "rms_equation_residual": float(torch.sqrt(torch.mean(residual.square())).cpu()),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_solution_plot(model: nn.Module, config: TrainingConfig, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    grid = torch.linspace(-1.0, 1.0, 401, dtype=DTYPE, device=device).reshape(-1, 1)
    with torch.no_grad():
        values = model(grid).detach().cpu().numpy().reshape(-1)
    grid_np = grid.detach().cpu().numpy().reshape(-1)
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(grid_np, np.ones_like(grid_np), "k--", linewidth=1.5, label="exact u(t) = 1")
    axis.plot(grid_np, values, linewidth=2.0, label="endpoint-form PINN")
    axis.set(xlabel="t", ylabel="u(t)", title="Endpoint-value hypersingular PINN")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    model, metrics = train_constant_solution(config)
    if args.json_out is not None:
        save_metrics_json(metrics, args.json_out)
        print(f"wrote metrics JSON: {args.json_out}")
    if args.png_out is not None:
        save_solution_plot(model, config, args.png_out)
        print(f"wrote solution PNG: {args.png_out}")
    print(
        f"final metrics: full_rms_solution_error={metrics['rms_solution_error']:.6e}, "
        f"interior_rms_equation_residual={metrics['rms_equation_residual']:.6e}"
    )
    return metrics


if __name__ == "__main__":
    main()
