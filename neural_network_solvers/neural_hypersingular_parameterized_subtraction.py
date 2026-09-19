"""Parameterized PINN for a Taylor-subtracted hypersingular equation.

For a fixed real parameter ``gamma`` this module solves on ``(-1, 1)``

    t**4 u(t) + (2-t)(1+t) H[u](t) = f_gamma(t),

where ``H`` is the finite-part hypersingular operator implemented in
``neural_hypersingular_constant_subtraction``.  The manufactured solution is
``u_gamma(t) = 1 + gamma*t``.  Crucially, the right hand side is constructed
analytically from that solution before training; it never depends on a neural
network prediction.

The network has ``(t, gamma)`` as its two inputs so the parameter is explicit
in its interface.  ``train_single_gamma`` deliberately holds gamma fixed for
one reproducible solve, while the command line can sweep several independent
single-gamma solves.  Training uses a boundary-balanced residual.  Reported
residuals are recomputed with an independent, denser Gauss--Legendre rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    hypersingular_operator,
    set_reproducible_seed,
)


def _as_column(value: Tensor | Sequence[float] | float, device: torch.device | str | None = None) -> Tensor:
    """Return ``value`` as a float64 column tensor, optionally on ``device``."""
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def coefficient_a(t: Tensor) -> Tensor:
    """Coefficient ``a(t) = t**4``."""
    return t.pow(4)


def coefficient_b(t: Tensor) -> Tensor:
    """Coefficient ``b(t) = (2-t)(1+t)``."""
    return (2.0 - t) * (1.0 + t)


def exact_solution(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Manufactured affine solution ``u_gamma(t) = 1 + gamma*t``."""
    gamma_col = _as_column(gamma, device=t.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t)
    elif gamma_col.shape != t.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    return 1.0 + gamma_col * t


def exact_hypersingular_operator(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Analytic value of ``H[1 + gamma*t]`` under the shared convention."""
    gamma_col = _as_column(gamma, device=t.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t)
    elif gamma_col.shape != t.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    solution = 1.0 + gamma_col * t
    endpoint = -solution * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
    logarithmic = gamma_col * torch.log((1.0 - t) / (1.0 + t))
    return endpoint + logarithmic


def equation_rhs(t: Tensor, gamma: Tensor | float) -> Tensor:
    """Fixed manufactured RHS ``f_gamma(t)`` with no model dependence."""
    return coefficient_a(t) * exact_solution(t, gamma) + coefficient_b(t) * exact_hypersingular_operator(t, gamma)


def equation_operator_value(u: Callable[[Tensor], Tensor], t: Tensor, quadrature_nodes: Tensor, quadrature_weights: Tensor,
                            near_zero_threshold: float = 1.0e-9) -> Tensor:
    """Evaluate ``a(t)u(t) + b(t)H[u](t)`` for a scalar-input callable."""
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    if u_value.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * hypersingular_operator(
        u, t_col, quadrature_nodes, quadrature_weights, near_zero_threshold
    )


def normalized_equation_residual(operator_value: Tensor, t: Tensor, gamma: Tensor | float) -> Tensor:
    """Boundary-balanced residual used for optimization.

    The raw equation has the remaining ``1/(1-t)`` endpoint growth inherited
    from ``H`` at ``t=1``.  Multiplying the raw residual by
    ``(1-t**2)/2`` balances this known growth while preserving the equation's
    zero set on the open interval.  Dividing by ``1 + abs(gamma)`` keeps the
    loss scale comparable over the requested sweep without changing its zero
    set.  ``gamma`` is included deliberately to make the objective's fixed
    RHS explicit in its public interface.
    """
    t_col = _as_column(t, device=operator_value.device)
    gamma_col = _as_column(gamma, device=operator_value.device)
    if gamma_col.numel() == 1:
        gamma_col = gamma_col.expand_as(t_col)
    elif gamma_col.shape != t_col.shape:
        raise ValueError("gamma must be scalar or have the same shape as t")
    return 0.5 * (1.0 - t_col.square()) * (operator_value - equation_rhs(t_col, gamma_col)) / (1.0 + gamma_col.abs())


class ParameterizedTanhNetwork(nn.Module):
    """Smooth two-input network for ``(t, gamma) -> u_gamma(t)``.

    The raw parameter is mapped to ``gamma / (1 + abs(gamma))`` before the
    first tanh layer.  This keeps large requested values such as ``gamma=10``
    from saturating the network solely because of their input scale; the
    physical equation and its right hand side still use the unscaled gamma.
    """

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


def train_single_gamma(gamma: float, config: TrainingConfig, verbose: bool = True) -> tuple[ParameterizedTanhNetwork, dict[str, object]]:
    """Train one fixed-gamma PINN and return model plus JSON-serializable metrics."""
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite")
    _validate_config(config)
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)
    gamma_tensor = torch.tensor(float(gamma), dtype=DTYPE, device=device)
    model = ParameterizedTanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Deterministic endpoint-aware collocation, without sampling singular endpoints.
    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    quadrature_nodes, quadrature_weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def bound_model(t: Tensor) -> Tensor:
        return model(t, gamma_tensor)

    def objective() -> Tensor:
        raw_operator_value = equation_operator_value(
            bound_model,
            collocation_points,
            quadrature_nodes,
            quadrature_weights,
            config.near_zero_threshold,
        )
        normalized = normalized_equation_residual(raw_operator_value, collocation_points, gamma_tensor)
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
            print(f"gamma={gamma:g} epoch={epoch:5d} loss={losses[-1]:.6e}")

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

        # One L-BFGS call performs the configured internal quasi-Newton
        # iterations.  Recreating it with max_iter=1 would discard its line
        # search/history and is materially less effective for large gamma.
        lbfgs.step(closure)
        post_step_loss = objective()
        if not torch.isfinite(post_step_loss):
            raise FloatingPointError("non-finite loss after L-BFGS")
        losses.append(float(post_step_loss.detach().cpu()))
        if verbose:
            print(f"gamma={gamma:g} lbfgs_max_iter={config.lbfgs_steps:4d} loss={losses[-1]:.6e}")

    model.eval()
    evaluation_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    # Separate, denser quadrature rule for final raw-residual reporting.
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        raw_operator_values = equation_operator_value(
            bound_model,
            evaluation_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        prediction = bound_model(evaluation_points)
        solution_error = prediction - exact_solution(evaluation_points, gamma_tensor)
        raw_residual = raw_operator_values - equation_rhs(evaluation_points, gamma_tensor)
        balanced_residual = 0.5 * (1.0 - evaluation_points.square()) * raw_residual
        metrics: dict[str, object] = {
            "gamma": float(gamma),
            "config": asdict(config),
            "initial_normalized_loss": losses[0],
            "final_normalized_loss": losses[-1],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "rms_solution_error": float(torch.sqrt(torch.mean(solution_error.square())).cpu()),
            "max_abs_equation_residual": float(raw_residual.abs().max().cpu()),
            "rms_equation_residual": float(torch.sqrt(torch.mean(raw_residual.square())).cpu()),
            "max_abs_balanced_residual": float(balanced_residual.abs().max().cpu()),
            "rms_balanced_residual": float(torch.sqrt(torch.mean(balanced_residual.square())).cpu()),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    """Write JSON metrics, creating the requested parent directory when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_sweep_plot(results: Sequence[tuple[float, ParameterizedTanhNetwork, dict[str, object]]], config: TrainingConfig,
                    path: str | Path) -> None:
    """Write a multi-panel PNG comparing each sweep model against its exact solution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must contain at least one gamma solve")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_columns = min(3, len(results))
    num_rows = math.ceil(len(results) / num_columns)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(5.0 * num_columns, 3.6 * num_rows), squeeze=False,
                                constrained_layout=True)
    for axis, (gamma, model, metrics) in zip(axes.flat, results):
        device = torch.device(config.device)
        grid = torch.linspace(-0.98, 0.98, 401, dtype=DTYPE, device=device).reshape(-1, 1)
        gamma_tensor = torch.tensor(gamma, dtype=DTYPE, device=device)
        with torch.no_grad():
            values = model(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
            exact = exact_solution(grid, gamma_tensor).detach().cpu().numpy().reshape(-1)
        grid_np = grid.detach().cpu().numpy().reshape(-1)
        axis.plot(grid_np, exact, "k--", linewidth=1.25, label="exact")
        axis.plot(grid_np, values, linewidth=1.8, label="tanh PINN")
        axis.set(title=f"gamma={gamma:g}; RMS error={metrics['rms_solution_error']:.2e}", xlabel="t", ylabel="u(t)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    for axis in axes.flat[len(results):]:
        axis.remove()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


DEFAULT_GAMMAS = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0)


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
        # Use the same predeclared initialization for every gamma.  This keeps
        # the parameter sweep comparable instead of confounding gamma with six
        # unrelated random initializations.
        model, metrics = train_single_gamma(gamma, config)
        results.append((gamma, model, metrics))
    summary: dict[str, object] = {"config": asdict(config), "gammas": list(gammas), "runs": [item[2] for item in results]}
    if args.json_out is not None:
        save_metrics_json(summary, args.json_out)
        print(f"wrote sweep JSON: {args.json_out}")
    if args.png_out is not None:
        save_sweep_plot(results, config, args.png_out)
        print(f"wrote sweep PNG: {args.png_out}")
    for _, _, metrics in results:
        print(
            f"gamma={metrics['gamma']:g} final_loss={metrics['final_normalized_loss']:.6e} "
            f"rms_solution_error={metrics['rms_solution_error']:.6e} "
            f"rms_equation_residual={metrics['rms_equation_residual']:.6e}"
        )
    return summary


if __name__ == "__main__":
    main()
