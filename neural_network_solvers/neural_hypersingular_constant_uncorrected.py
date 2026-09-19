"""Uncorrected PINN control for the constant hypersingular test equation.

This module solves, on ``(-1, 1)``,

    H[u](t) = 2 / ((t - 1) (t + 1)),

whose exact solution is ``u(t) = 1``.  It is intentionally a control against
the Taylor-subtracted baseline: training uses *only* the uncorrected operator

    H_unc[u](t) = -u(t) [1/(1-t) + 1/(1+t)]
                  + u'(t) log((1-t)/(1+t))
                  + int_-1^1 (u'(tau)-u'(t))/(tau-t) d tau.

At a coincident quadrature/collocation node the integral's value is replaced
by its diagonal Taylor limit ``u''(t)``.  In particular, no endpoint or
boundary correction is added to this displayed formula.  The random seed,
network, optimizers, collocation rule, training rule, validation rule, and
evaluation interval exactly match ``neural_hypersingular_constant_subtraction``.
The final metrics report both this uncorrected equation residual and the
residual obtained by putting the same trained network into the corrected
Taylor-subtracted operator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TanhNetwork,
    equation_rhs,
    exact_solution,
    gauss_legendre_rule,
    hypersingular_operator as corrected_hypersingular_operator,
    normalized_equation_residual,
    set_reproducible_seed,
)


def _as_column(t: Tensor | Sequence[float], device: torch.device | str | None = None) -> Tensor:
    """Return ``t`` as a float64 column tensor, optionally on ``device``."""
    if isinstance(t, Tensor):
        result = t.to(dtype=DTYPE, device=device if device is not None else t.device)
    else:
        result = torch.as_tensor(t, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def _first_derivative(values: Tensor, inputs: Tensor) -> Tensor:
    """Differentiate scalar samples, including literal constant callables."""
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


def uncorrected_hypersingular_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the prescribed uncorrected derivative-quotient operator.

    The final integral is evaluated exactly as
    ``(u'(tau) - u'(t)) / (tau - t)``.  Coincident nodes use ``u''(t)``, its
    diagonal Taylor limit.  No term is added to reconcile this formula with
    the corrected Taylor-subtracted finite-part operator.
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


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration deliberately identical to the corrected constant baseline."""

    seed: int = 20260916
    epochs: int = 3000
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 300
    hidden_width: int = 32
    hidden_layers: int = 3
    num_collocation: int = 96
    num_quadrature: int = 96
    near_zero_threshold: float = 1.0e-9
    report_every: int = 50
    device: str = "cpu"


def _residual_metrics(residual: Tensor, prefix: str) -> dict[str, float]:
    """Return raw equation-residual metrics under an unambiguous prefix."""
    return {
        f"max_abs_{prefix}": float(residual.abs().max().cpu()),
        f"rms_{prefix}": float(torch.sqrt(torch.mean(residual.square())).cpu()),
    }


def train_constant_solution(config: TrainingConfig, verbose: bool = True) -> tuple[TanhNetwork, dict[str, object]]:
    """Train against ``H_unc`` and report both uncorrected and corrected residuals."""
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
        operator_value = uncorrected_hypersingular_operator(
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
            print(f"epoch={epoch:5d} uncorrected_loss={losses[-1]:.6e}")

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=1,
            history_size=50,
            line_search_fn="strong_wolfe",
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
                print(f"lbfgs_step={step:4d} uncorrected_loss={losses[-1]:.6e}")

    model.eval()
    evaluation_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        uncorrected_operator_values = uncorrected_hypersingular_operator(
            model,
            evaluation_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
        corrected_operator_values = corrected_hypersingular_operator(
            model,
            evaluation_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        prediction = model(evaluation_points)
        solution_error = prediction - exact_solution(evaluation_points)
        uncorrected_residual = uncorrected_operator_values - equation_rhs(evaluation_points)
        corrected_residual = corrected_operator_values - equation_rhs(evaluation_points)
        metrics: dict[str, object] = {
            "config": asdict(config),
            "final_normalized_loss": losses[-1],
            "initial_normalized_loss": losses[0],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "rms_solution_error": float(torch.sqrt(torch.mean(solution_error.square())).cpu()),
            **_residual_metrics(uncorrected_residual, "uncorrected_equation_residual"),
            **_residual_metrics(corrected_residual, "corrected_equation_residual"),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    """Write metrics to a requested JSON path, creating only its parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_solution_plot(model: nn.Module, config: TrainingConfig, path: str | Path) -> None:
    """Save a prediction plot labelled as the uncorrected-operator control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    grid = torch.linspace(-0.98, 0.98, 401, dtype=DTYPE, device=device).reshape(-1, 1)
    with torch.no_grad():
        values = model(grid).detach().cpu().numpy().reshape(-1)
    grid_np = grid.detach().cpu().numpy().reshape(-1)
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(grid_np, torch.ones_like(grid).cpu().numpy().reshape(-1), "k--", linewidth=1.5, label="exact u(t) = 1")
    axis.plot(grid_np, values, linewidth=2.0, label="uncorrected PINN")
    axis.set(xlabel="t", ylabel="u(t)", title="Uncorrected hypersingular PINN control")
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
    parser.add_argument("--device", default=TrainingConfig.device, help="PyTorch device; CPU is the reproducible default")
    parser.add_argument("--json-out", type=Path, help="optional metrics JSON output path")
    parser.add_argument("--png-out", type=Path, help="optional solution PNG output path")
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
        "final metrics: "
        f"rms_solution_error={metrics['rms_solution_error']:.6e}, "
        f"rms_uncorrected_equation_residual={metrics['rms_uncorrected_equation_residual']:.6e}, "
        f"rms_corrected_equation_residual={metrics['rms_corrected_equation_residual']:.6e}"
    )
    return metrics


if __name__ == "__main__":
    main()
