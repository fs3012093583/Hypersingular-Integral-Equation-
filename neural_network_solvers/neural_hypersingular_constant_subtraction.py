"""Reproducible PINN experiment for a finite-part hypersingular equation.

This module solves, on ``(-1, 1)``,

    Fp int_-1^1 u(tau) / (tau - t)^2 d tau = 2 / ((t - 1) (t + 1)).

The exact regular solution is ``u(t) = 1``.  The operator is evaluated after
subtracting the left Taylor polynomial of ``u`` at each collocation point:

    H[u](t) = -u(t) [1/(1-t) + 1/(1+t)]
              + u'(t) log((1-t)/(1+t))
              + int_-1^1 (u(tau)-u(t)-u'(t)(tau-t))/(tau-t)^2 d tau.

The last integral is regular.  A Gauss--Legendre rule is used for it, and its
diagonal/near-diagonal values are replaced by the Taylor limit ``u''(t)/2``.
The code deliberately uses CPU float64 by default so that a fixed seed gives
an independently reproducible small experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn


DTYPE = torch.float64


def set_reproducible_seed(seed: int) -> None:
    """Set all RNGs used by this experiment to a deterministic initial state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gauss_legendre_rule(num_points: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Return nodes and weights for integration on ``[-1, 1]`` in float64."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    return (
        torch.as_tensor(nodes, dtype=DTYPE, device=device).reshape(-1, 1),
        torch.as_tensor(weights, dtype=DTYPE, device=device).reshape(-1, 1),
    )


def equation_rhs(t: Tensor) -> Tensor:
    """Right hand side ``2 / ((t - 1) * (t + 1))`` of the target equation."""
    return 2.0 / ((t - 1.0) * (t + 1.0))


def exact_solution(t: Tensor) -> Tensor:
    """The exact constant solution, written to retain an autograd path in ``t``."""
    return t * 0.0 + 1.0


def normalized_equation_residual(operator_value: Tensor, t: Tensor) -> Tensor:
    """Return the boundary-balanced residual used as the PINN objective.

    The equation itself is always evaluated with :func:`hypersingular_operator`.
    Multiplication by ``(1-t**2)/2`` only balances the known endpoint growth of
    the right hand side during optimization; it has exactly the same zero set
    on the open interval.
    """
    return 0.5 * (1.0 - t.square()) * (operator_value - equation_rhs(t))


def _as_column(t: Tensor | Sequence[float], device: torch.device | str | None = None) -> Tensor:
    if isinstance(t, Tensor):
        result = t.to(dtype=DTYPE, device=device if device is not None else t.device)
    else:
        result = torch.as_tensor(t, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def hypersingular_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    """Evaluate the Taylor-subtracted finite-part hypersingular operator.

    ``u`` can be a PyTorch module or any tensor-to-tensor callable.  The
    returned tensor has shape ``(len(t), 1)`` and remains differentiable with
    respect to parameters used by ``u``.  At coincident (or very close)
    quadrature/collocation nodes, the regular integrand is set to ``u''(t)/2``.
    """
    if near_zero_threshold <= 0.0:
        raise ValueError("near_zero_threshold must be positive")
    device = quadrature_nodes.device
    t_col = _as_column(t, device=device).detach().clone().requires_grad_(True)
    tau = _as_column(quadrature_nodes, device=device).reshape(1, -1)
    weights = _as_column(quadrature_weights, device=device).reshape(1, -1)
    if tau.shape != weights.shape:
        raise ValueError("quadrature_nodes and quadrature_weights must have the same length")

    u_t = _as_column(u(t_col), device=device)
    if u_t.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    if u_t.requires_grad:
        first_derivative = torch.autograd.grad(
            u_t,
            t_col,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        u_prime = first_derivative if first_derivative is not None else torch.zeros_like(t_col)
    else:
        # Accept literal constant callables such as ``lambda x: ones_like(x)``
        # as well as differentiable neural modules.
        u_prime = torch.zeros_like(t_col)
    # For an affine callable PyTorch correctly returns a constant first
    # derivative without a grad_fn.  Its second derivative is then exactly
    # zero, while a tanh network follows the differentiable branch below.
    if u_prime.requires_grad:
        u_second = torch.autograd.grad(
            u_prime,
            t_col,
            grad_outputs=torch.ones_like(u_prime),
            create_graph=True,
            retain_graph=True,
        )[0]
    else:
        u_second = torch.zeros_like(u_prime)

    # Evaluating u(tau) separately preserves gradients to neural-network
    # parameters while avoiding an unnecessary derivative with respect to tau.
    u_tau = _as_column(u(tau.reshape(-1, 1)), device=device).reshape(1, -1)
    delta = tau - t_col
    near_zero = delta.abs() <= near_zero_threshold
    safe_delta = torch.where(near_zero, torch.ones_like(delta), delta)
    numerator = u_tau - u_t - u_prime * delta
    regular_value = numerator / safe_delta.square()
    integrand = torch.where(near_zero, 0.5 * u_second, regular_value)
    regular_integral = torch.sum(integrand * weights, dim=1, keepdim=True)

    endpoint_term = -u_t * (1.0 / (1.0 - t_col) + 1.0 / (1.0 + t_col))
    logarithmic_term = u_prime * torch.log((1.0 - t_col) / (1.0 + t_col))
    return endpoint_term + logarithmic_term + regular_integral


class TanhNetwork(nn.Module):
    """A deliberately small smooth network; tanh makes the second derivative available."""

    def __init__(self, hidden_width: int = 32, hidden_layers: int = 2) -> None:
        super().__init__()
        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError("hidden_width and hidden_layers must both be positive")
        layers: list[nn.Module] = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t: Tensor) -> Tensor:
        return self.network(t)


@dataclass(frozen=True)
class TrainingConfig:
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


def train_constant_solution(config: TrainingConfig, verbose: bool = True) -> tuple[TanhNetwork, dict[str, object]]:
    """Train a tanh PINN and return its model and JSON-serializable metrics."""
    if config.epochs < 0 or config.lbfgs_steps < 0 or config.epochs + config.lbfgs_steps < 1:
        raise ValueError("at least one Adam or L-BFGS step is required")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)

    model = TanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Cosine clustering resolves the endpoint terms while stopping short of
    # the singular endpoints themselves.  This is deterministic and gives a
    # different node set from the Gauss--Legendre integration rule.
    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    quadrature_nodes, quadrature_weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def objective() -> Tensor:
        operator_value = hypersingular_operator(
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
            print(f"epoch={epoch:5d} loss={losses[-1]:.6e}")

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
                print(f"lbfgs_step={step:4d} loss={losses[-1]:.6e}")

    model.eval()
    evaluation_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    # Recompute the reported equation residual with a denser rule that was not
    # used by the optimizer.  This distinguishes solving the regularized
    # equation from merely fitting one fixed quadrature discretization.
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        operator_values = hypersingular_operator(
            model,
            evaluation_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        prediction = model(evaluation_points)
        solution_error = prediction - exact_solution(evaluation_points)
        residual = operator_values - equation_rhs(evaluation_points)
        metrics: dict[str, object] = {
            "config": asdict(config),
            "final_normalized_loss": losses[-1],
            "initial_normalized_loss": losses[0],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "rms_solution_error": float(torch.sqrt(torch.mean(solution_error.square())).cpu()),
            "max_abs_equation_residual": float(residual.abs().max().cpu()),
            "rms_equation_residual": float(torch.sqrt(torch.mean(residual.square())).cpu()),
            "loss_history": losses,
        }
    return model, metrics


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    """Write metrics to a requested JSON path, creating only its parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_solution_plot(model: nn.Module, config: TrainingConfig, path: str | Path) -> None:
    """Save a PNG comparing the trained prediction against the known constant solution."""
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
    axis.plot(grid_np, np.ones_like(grid_np), "k--", linewidth=1.5, label="exact u(t) = 1")
    axis.plot(grid_np, values, linewidth=2.0, label="tanh PINN")
    axis.set(xlabel="t", ylabel="u(t)", title="Taylor-subtracted hypersingular PINN")
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
        f"rms_equation_residual={metrics['rms_equation_residual']:.6e}"
    )
    return metrics


if __name__ == "__main__":
    main()
