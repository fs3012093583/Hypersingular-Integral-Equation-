"""Isolated PINN comparison for the uncorrected blackboard operator.

For each fixed ``gamma`` this module solves the manufactured equation

    t**4 u(t) + (2-t)(1+t) H_unc[u](t) = f_gamma(t),

where ``u_gamma(t) = 1 + gamma*t`` and, deliberately without any endpoint
correction beyond the displayed blackboard formula,

    H_unc[u](t) = -u(t) (1/(1-t) + 1/(1+t))
                  + u'(t) log((1-t)/(1+t))
                  + int_-1^1 (u'(tau)-u'(t))/(tau-t) d tau.

The diagonal value of the final integrand is its Taylor limit ``u''(t)``.
Training is self-contained and never reads a corrected sweep.  At reporting
time, however, the same trained network is also evaluated with the complete
Taylor-subtracted finite-part operator so that a low uncorrected loss is not
mistaken for fidelity to the original corrected equation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    hypersingular_operator as corrected_hypersingular_operator,
)


DTYPE = torch.float64
DEFAULT_GAMMAS = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0)
DEFAULT_CORRECTED_SWEEP_JSON = Path("results/hypersingular_parameterized_subtraction/sweep_metrics.json")


def set_reproducible_seed(seed: int) -> None:
    """Set every RNG used by this CPU-float64 experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gauss_legendre_rule(num_points: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Return an independent float64 Gauss--Legendre rule on ``[-1, 1]``."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    return (
        torch.as_tensor(nodes, dtype=DTYPE, device=device).reshape(-1, 1),
        torch.as_tensor(weights, dtype=DTYPE, device=device).reshape(-1, 1),
    )


def _as_column(value: Tensor | Sequence[float] | float, device: torch.device | str | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def _first_derivative(values: Tensor, inputs: Tensor) -> Tensor:
    """Differentiate scalar samples, accepting literal constant callables."""
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


def coefficient_a(t: Tensor) -> Tensor:
    """Coefficient ``a(t) = t**4``."""
    return t.pow(4)


def coefficient_b(t: Tensor) -> Tensor:
    """Coefficient ``b(t) = (2-t)(1+t)``."""
    return (2.0 - t) * (1.0 + t)


def exact_solution(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Shared manufactured affine solution ``u_gamma(t) = 1 + gamma*t``."""
    gamma_col = _as_column(gamma, device=t.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t)
    elif gamma_col.shape != t.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    return 1.0 + gamma_col * t


def exact_uncorrected_hypersingular_operator(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Analytic ``H_unc[1 + gamma*t]``; the integral vanishes for affine ``u``."""
    gamma_col = _as_column(gamma, device=t.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t)
    elif gamma_col.shape != t.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    affine = exact_solution(t, gamma_col)
    return (
        -affine * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
        + gamma_col * torch.log((1.0 - t) / (1.0 + t))
    )


def equation_rhs(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Fixed manufactured RHS, independent of a neural-network prediction."""
    return coefficient_a(t) * exact_solution(t, gamma) + coefficient_b(t) * exact_uncorrected_hypersingular_operator(t, gamma)


def uncorrected_hypersingular_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate exactly the uncorrected derivative-quotient blackboard formula.

    No term is added to make this integral agree with the complete
    Taylor-subtracted finite-part operator.  Coincident quadrature points use
    ``u''(t)`` as required by the derivative quotient's Taylor limit.
    """
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

    tau = tau_col.reshape(1, -1)
    delta = tau - t_col
    near_zero = delta.abs() <= near_zero_threshold
    safe_delta = torch.where(near_zero, torch.ones_like(delta), delta)
    quotient = (u_prime_tau - u_prime_t) / safe_delta
    integrand = torch.where(near_zero, u_second_t, quotient)
    derivative_quotient_integral = torch.sum(integrand * weights.reshape(1, -1), dim=1, keepdim=True)

    endpoint_term = -u_t * (1.0 / (1.0 - t_col) + 1.0 / (1.0 + t_col))
    logarithmic_term = u_prime_t * torch.log((1.0 - t_col) / (1.0 + t_col))
    return endpoint_term + logarithmic_term + derivative_quotient_integral


def equation_operator_value(
    u: Callable[[Tensor], Tensor],
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the uncorrected manufactured equation's left-hand side."""
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    if u_value.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * uncorrected_hypersingular_operator(
        u, t_col, quadrature_nodes, quadrature_weights, near_zero_threshold
    )


def corrected_equation_operator_value(
    u: Callable[[Tensor], Tensor],
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the original complete finite-part equation for reporting only."""
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * corrected_hypersingular_operator(
        u, t_col, quadrature_nodes, quadrature_weights, near_zero_threshold
    )


def normalized_equation_residual(operator_value: Tensor, t: Tensor, gamma: Tensor | float) -> Tensor:
    """Boundary-balanced training residual for the uncorrected equation."""
    t_col = _as_column(t, device=operator_value.device)
    gamma_col = _as_column(gamma, device=operator_value.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t_col)
    elif gamma_col.shape != t_col.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    return 0.5 * (1.0 - t_col.square()) * (operator_value - equation_rhs(t_col, gamma_col)) / (1.0 + gamma_col.abs())


class ParameterizedTanhNetwork(nn.Module):
    """The matching 32x3 tanh ``(t, gamma)`` network with bounded gamma input."""

    def __init__(self, hidden_width: int = 32, hidden_layers: int = 3) -> None:
        super().__init__()
        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError("hidden_width and hidden_layers must both be positive")
        layers: list[nn.Module] = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t: Tensor, gamma: Tensor | float) -> Tensor:
        t_col = _as_column(t, device=t.device)
        gamma_col = _as_column(gamma, device=t_col.device)
        if gamma_col.numel() == 1:
            gamma_col = gamma_col.expand_as(t_col)
        elif gamma_col.shape != t_col.shape:
            raise ValueError("gamma must be scalar or have the same shape as t")
        gamma_feature = gamma_col / (1.0 + gamma_col.abs())
        return self.network(torch.cat((t_col, gamma_feature), dim=1))


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260916
    epochs: int = 3000
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 400
    hidden_width: int = 32
    hidden_layers: int = 3
    num_collocation: int = 96
    num_quadrature: int = 96
    near_zero_threshold: float = 1.0e-9
    report_every: int = 50
    device: str = "cpu"


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs < 0 or config.lbfgs_steps < 0 or config.epochs + config.lbfgs_steps < 1:
        raise ValueError("at least one Adam or L-BFGS step is required")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.num_collocation < 2 or config.num_quadrature < 2:
        raise ValueError("num_collocation and num_quadrature must both be at least 2")


def _raw_residual_metrics(residual: Tensor, prefix: str) -> dict[str, float]:
    return {
        f"max_abs_{prefix}": float(residual.abs().max().cpu()),
        f"rms_{prefix}": float(torch.sqrt(torch.mean(residual.square())).cpu()),
    }


def train_single_gamma(gamma: float, config: TrainingConfig, verbose: bool = True) -> tuple[ParameterizedTanhNetwork, dict[str, object]]:
    """Train only against ``H_unc`` and report both 192-point raw residuals."""
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
            bound_model, collocation_points, quadrature_nodes, quadrature_weights, config.near_zero_threshold
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
            print(f"gamma={gamma:g} epoch={epoch:5d} uncorrected_loss={losses[-1]:.6e}")

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.8, max_iter=config.lbfgs_steps, history_size=50,
            tolerance_grad=1.0e-12, tolerance_change=1.0e-14, line_search_fn="strong_wolfe",
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
            print(f"gamma={gamma:g} lbfgs_max_iter={config.lbfgs_steps:4d} uncorrected_loss={losses[-1]:.6e}")

    model.eval()
    evaluation_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    # This fixed rule is intentionally independent of the 96-point training
    # rule and is identical for both reported physical residuals.
    validation_num_quadrature = 192
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        uncorrected_operator_values = equation_operator_value(
            bound_model, evaluation_points, validation_nodes, validation_weights, config.near_zero_threshold
        )
        corrected_operator_values = corrected_equation_operator_value(
            bound_model, evaluation_points, validation_nodes, validation_weights, config.near_zero_threshold
        )
    with torch.no_grad():
        prediction = bound_model(evaluation_points)
        solution_error = prediction - exact_solution(evaluation_points, gamma_tensor)
        uncorrected_raw_residual = uncorrected_operator_values - equation_rhs(evaluation_points, gamma_tensor)
        corrected_raw_residual = corrected_operator_values - equation_rhs(evaluation_points, gamma_tensor)
        balanced_residual = 0.5 * (1.0 - evaluation_points.square()) * uncorrected_raw_residual
        metrics: dict[str, object] = {
            "gamma": float(gamma),
            "config": asdict(config),
            "initial_normalized_loss": losses[0],
            "final_normalized_loss": losses[-1],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "rms_solution_error": float(torch.sqrt(torch.mean(solution_error.square())).cpu()),
            **_raw_residual_metrics(uncorrected_raw_residual, "uncorrected_equation_residual"),
            **_raw_residual_metrics(corrected_raw_residual, "corrected_physical_residual"),
            **_raw_residual_metrics(balanced_residual, "balanced_uncorrected_residual"),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    """Write JSON metrics, creating only the specified parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_corrected_sweep_metrics(path: str | Path | None) -> dict[float, dict[str, object]]:
    """Best-effort loader for an existing corrected sweep; never used to train."""
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        runs = payload["runs"]
        if not isinstance(runs, list):
            return {}
        return {float(run["gamma"]): run for run in runs if isinstance(run, dict) and "gamma" in run}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def build_comparison_table(runs: Sequence[dict[str, object]], corrected_runs: dict[float, dict[str, object]]) -> list[dict[str, object]]:
    """Join per-gamma uncorrected metrics with any available corrected metrics."""
    table: list[dict[str, object]] = []
    corrected_columns = ("final_normalized_loss", "rms_solution_error", "rms_equation_residual", "max_abs_equation_residual")
    for run in runs:
        gamma = float(run["gamma"])
        row: dict[str, object] = {
            "gamma": gamma,
            "uncorrected_final_normalized_loss": run["final_normalized_loss"],
            "uncorrected_rms_solution_error": run["rms_solution_error"],
            "uncorrected_rms_equation_residual": run["rms_uncorrected_equation_residual"],
            "uncorrected_max_abs_equation_residual": run["max_abs_uncorrected_equation_residual"],
            "same_network_rms_corrected_physical_residual": run["rms_corrected_physical_residual"],
            "same_network_max_abs_corrected_physical_residual": run["max_abs_corrected_physical_residual"],
        }
        corrected_run = corrected_runs.get(gamma)
        if corrected_run is not None:
            for column in corrected_columns:
                if column in corrected_run:
                    row[f"corrected_sweep_{column}"] = corrected_run[column]
        table.append(row)
    return table


def save_sweep_plot(
    results: Sequence[tuple[float, ParameterizedTanhNetwork, dict[str, object]]], config: TrainingConfig, path: str | Path
) -> None:
    """Write a PNG of each uncorrected-trained prediction against the affine target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must contain at least one gamma solve")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.0 * rows), squeeze=False, constrained_layout=True)
    for axis, (gamma, model, metrics) in zip(axes.flat, results):
        device = torch.device(config.device)
        grid = torch.linspace(-0.98, 0.98, 401, dtype=DTYPE, device=device).reshape(-1, 1)
        gamma_tensor = torch.tensor(gamma, dtype=DTYPE, device=device)
        with torch.no_grad():
            values = model(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
            exact = exact_solution(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
        grid_np = grid.detach().cpu().numpy().reshape(-1)
        axis.plot(grid_np, exact, "k--", linewidth=1.25, label="affine exact")
        axis.plot(grid_np, values, linewidth=1.8, label="uncorrected PINN")
        axis.set(
            title=(f"gamma={gamma:g}; unc RMS={metrics['rms_uncorrected_equation_residual']:.2e}\n"
                   f"corr RMS={metrics['rms_corrected_physical_residual']:.2e}"),
            xlabel="t", ylabel="u(t)",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    for axis in axes.flat[len(results):]:
        axis.remove()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, action="append", help="gamma to solve; repeat to form a sweep")
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
    parser.add_argument("--device", default=TrainingConfig.device, help="PyTorch device; CPU is the reproducible default")
    parser.add_argument("--json-out", type=Path, help="optional sweep-summary JSON output path")
    parser.add_argument("--png-out", type=Path, help="optional multi-panel sweep PNG output path")
    parser.add_argument(
        "--corrected-sweep-json", type=Path, default=DEFAULT_CORRECTED_SWEEP_JSON,
        help="optional corrected sweep metrics to join into JSON; missing files are ignored",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_argument_parser().parse_args(argv)
    config = TrainingConfig(
        seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate, lbfgs_steps=args.lbfgs_steps,
        hidden_width=args.hidden_width, hidden_layers=args.hidden_layers, num_collocation=args.num_collocation,
        num_quadrature=args.num_quadrature, near_zero_threshold=args.near_zero_threshold,
        report_every=args.report_every, device=args.device,
    )
    gammas = tuple(args.gamma) if args.gamma is not None else DEFAULT_GAMMAS
    results: list[tuple[float, ParameterizedTanhNetwork, dict[str, object]]] = []
    for gamma in gammas:
        model, metrics = train_single_gamma(gamma, config)
        results.append((gamma, model, metrics))
    corrected_runs = load_corrected_sweep_metrics(args.corrected_sweep_json)
    summary: dict[str, object] = {
        "config": asdict(config),
        "gammas": list(gammas),
        "runs": [item[2] for item in results],
        "corrected_sweep_metrics_source": str(args.corrected_sweep_json),
        "corrected_sweep_metrics_loaded": bool(corrected_runs),
        "comparison_table": build_comparison_table([item[2] for item in results], corrected_runs),
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
            f"uncorrected_rms={metrics['rms_uncorrected_equation_residual']:.6e} "
            f"corrected_rms={metrics['rms_corrected_physical_residual']:.6e}"
        )
    return summary


if __name__ == "__main__":
    main()
