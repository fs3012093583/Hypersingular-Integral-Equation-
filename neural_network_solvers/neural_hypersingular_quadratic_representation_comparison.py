"""Compare two correct hypersingular representations on polynomial targets.

The manufactured solution is

    u(t) = 1 + gamma*t + eta*t**2 + xi*t**3,

and the equation is

    t**4 u(t) + (2-t)(1+t) H[u](t) = f_{gamma,eta}(t).

Two independent PINNs are trained for every ``eta``:

1. the endpoint-value/derivative-difference representation;
2. the complete Taylor-subtraction representation.

Both represent the same finite-part operator.  Setting ``xi=0`` gives the
quadratic comparison; a nonzero ``xi`` adds the cubic comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor

from neural_network_solvers.neural_hypersingular_constant_endpoint import endpoint_hypersingular_operator
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TanhNetwork,
    gauss_legendre_rule,
    hypersingular_operator as taylor_hypersingular_operator,
    set_reproducible_seed,
)


Representation = Literal["endpoint", "taylor"]
DEFAULT_ETAS = (0.0, 0.1, 0.5, 1.0)
DEFAULT_XIS = (0.0, 0.1, 0.5, 1.0)


def _as_column(value: Tensor | Sequence[float] | float, device: torch.device | str | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def coefficient_a(t: Tensor) -> Tensor:
    return t.pow(4)


def coefficient_b(t: Tensor) -> Tensor:
    return (2.0 - t) * (1.0 + t)


def exact_solution(
    t: Tensor,
    gamma: float | Tensor,
    eta: float | Tensor,
    xi: float | Tensor = 0.0,
) -> Tensor:
    gamma_tensor = torch.as_tensor(gamma, dtype=DTYPE, device=t.device)
    eta_tensor = torch.as_tensor(eta, dtype=DTYPE, device=t.device)
    xi_tensor = torch.as_tensor(xi, dtype=DTYPE, device=t.device)
    return 1.0 + gamma_tensor * t + eta_tensor * t.square() + xi_tensor * t.pow(3)


def exact_hypersingular_operator(
    t: Tensor,
    gamma: float | Tensor,
    eta: float | Tensor,
    xi: float | Tensor = 0.0,
) -> Tensor:
    """Analytic finite-part value for the manufactured cubic polynomial.

    This is written in the complete Taylor-subtraction form.  Its regular
    The regular Taylor remainder is ``eta + xi*(tau + 2*t)``, so its
    integral on ``[-1,1]`` is exactly ``2*eta + 4*xi*t``.
    """
    gamma_tensor = torch.as_tensor(gamma, dtype=DTYPE, device=t.device)
    eta_tensor = torch.as_tensor(eta, dtype=DTYPE, device=t.device)
    xi_tensor = torch.as_tensor(xi, dtype=DTYPE, device=t.device)
    value = exact_solution(t, gamma_tensor, eta_tensor, xi_tensor)
    derivative = gamma_tensor + 2.0 * eta_tensor * t + 3.0 * xi_tensor * t.square()
    return (
        -value * (1.0 / (1.0 - t) + 1.0 / (1.0 + t))
        + derivative * torch.log((1.0 - t) / (1.0 + t))
        + 2.0 * eta_tensor
        + 4.0 * xi_tensor * t
    )


def equation_rhs(
    t: Tensor,
    gamma: float | Tensor,
    eta: float | Tensor,
    xi: float | Tensor = 0.0,
) -> Tensor:
    """Fixed manufactured right-hand side, independent of the neural model."""
    exact = exact_solution(t, gamma, eta, xi)
    return coefficient_a(t) * exact + coefficient_b(t) * exact_hypersingular_operator(
        t, gamma, eta, xi
    )


def hypersingular_value(
    representation: Representation,
    u: Callable[[Tensor], Tensor],
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float,
) -> Tensor:
    if representation == "endpoint":
        return endpoint_hypersingular_operator(
            u, t, quadrature_nodes, quadrature_weights, near_zero_threshold
        )
    if representation == "taylor":
        return taylor_hypersingular_operator(
            u, t, quadrature_nodes, quadrature_weights, near_zero_threshold
        )
    raise ValueError(f"unknown representation: {representation}")


def equation_operator_value(
    representation: Representation,
    u: Callable[[Tensor], Tensor],
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float = 1.0e-9,
) -> Tensor:
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    if u_value.shape != t_col.shape:
        raise ValueError("u(t) must return one scalar value per input point")
    hypersingular = hypersingular_value(
        representation, u, t_col, quadrature_nodes, quadrature_weights, near_zero_threshold
    )
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * hypersingular


def normalized_equation_residual(
    operator_value: Tensor,
    t: Tensor,
    gamma: float,
    eta: float,
    xi: float = 0.0,
) -> Tensor:
    scale = 1.0 + abs(gamma) + abs(eta) + abs(xi)
    return 0.5 * (1.0 - t.square()) * (
        operator_value - equation_rhs(t, gamma, eta, xi)
    ) / scale


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
    report_every: int = 500
    device: str = "cpu"


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs < 0 or config.lbfgs_steps < 0 or config.epochs + config.lbfgs_steps < 1:
        raise ValueError("at least one Adam or L-BFGS step is required")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.num_collocation < 2 or config.num_quadrature < 2:
        raise ValueError("num_collocation and num_quadrature must both be at least 2")


def _rms(value: Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.square())).cpu())


def train_one(
    gamma: float,
    eta: float,
    representation: Representation,
    config: TrainingConfig,
    verbose: bool = True,
    xi: float = 0.0,
) -> tuple[TanhNetwork, dict[str, object]]:
    """Train one representation for one fixed ``(gamma, eta, xi)`` tuple."""
    if not math.isfinite(gamma) or not math.isfinite(eta) or not math.isfinite(xi):
        raise ValueError("gamma, eta, and xi must be finite")
    _validate_config(config)
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)
    model = TanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    quadrature_nodes, quadrature_weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def objective() -> Tensor:
        value = equation_operator_value(
            representation,
            model,
            collocation_points,
            quadrature_nodes,
            quadrature_weights,
            config.near_zero_threshold,
        )
        normalized = normalized_equation_residual(
            value, collocation_points, gamma, eta, xi
        )
        return torch.mean(normalized.square())

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite {representation} loss at eta={eta:g}, xi={xi:g}, epoch={epoch}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (epoch == 1 or epoch % config.report_every == 0 or epoch == config.epochs):
            print(
                f"eta={eta:g} xi={xi:g} representation={representation} "
                f"epoch={epoch:5d} loss={losses[-1]:.6e}"
            )

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
                raise FloatingPointError(
                    f"non-finite {representation} L-BFGS loss at eta={eta:g}, xi={xi:g}"
                )
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        post_step_loss = objective()
        if not torch.isfinite(post_step_loss):
            raise FloatingPointError(
                f"non-finite {representation} loss after L-BFGS at eta={eta:g}, xi={xi:g}"
            )
        losses.append(float(post_step_loss.detach().cpu()))
        if verbose:
            print(
                f"eta={eta:g} xi={xi:g} representation={representation} "
                f"lbfgs_max_iter={config.lbfgs_steps:4d} loss={losses[-1]:.6e}"
            )

    model.eval()
    solution_points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    residual_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)

    with torch.enable_grad():
        endpoint_values = equation_operator_value(
            "endpoint", model, residual_points, validation_nodes, validation_weights,
            config.near_zero_threshold,
        )
        taylor_values = equation_operator_value(
            "taylor", model, residual_points, validation_nodes, validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        target_solution = exact_solution(solution_points, gamma, eta, xi)
        solution_error = model(solution_points) - target_solution
        interior_error = model(residual_points) - exact_solution(
            residual_points, gamma, eta, xi
        )
        rhs = equation_rhs(residual_points, gamma, eta, xi)
        endpoint_residual = endpoint_values - rhs
        taylor_residual = taylor_values - rhs
        representation_difference = endpoint_values - taylor_values
        endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
        endpoint_errors = (
            model(endpoint_points) - exact_solution(endpoint_points, gamma, eta, xi)
        ).abs().reshape(-1)
        own_residual = endpoint_residual if representation == "endpoint" else taylor_residual
        alternate_residual = taylor_residual if representation == "endpoint" else endpoint_residual
        metrics: dict[str, object] = {
            "gamma": gamma,
            "eta": eta,
            "xi": xi,
            "representation": representation,
            "config": asdict(config),
            "initial_normalized_loss": losses[0],
            "final_normalized_loss": losses[-1],
            "minimum_normalized_loss": min(losses),
            "validation_num_quadrature": validation_num_quadrature,
            "solution_evaluation_interval": [-1.0, 1.0],
            "residual_evaluation_interval": [-0.95, 0.95],
            "rms_solution_error": _rms(solution_error),
            "max_abs_solution_error": float(solution_error.abs().max().cpu()),
            "interior_rms_solution_error": _rms(interior_error),
            "abs_solution_error_t_minus_1": float(endpoint_errors[0].cpu()),
            "abs_solution_error_t_plus_1": float(endpoint_errors[1].cpu()),
            "rms_own_equation_residual": _rms(own_residual),
            "rms_alternate_equation_residual": _rms(alternate_residual),
            "rms_endpoint_equation_residual": _rms(endpoint_residual),
            "rms_taylor_equation_residual": _rms(taylor_residual),
            "max_abs_representation_operator_difference": float(
                representation_difference.abs().max().cpu()
            ),
            "rms_representation_operator_difference": _rms(representation_difference),
            "loss_history": losses,
        }
    return model, metrics


def build_comparison(
    eta: float,
    xi: float,
    endpoint_model: TanhNetwork,
    endpoint_metrics: dict[str, object],
    taylor_model: TanhNetwork,
    taylor_metrics: dict[str, object],
    gamma: float,
    device: torch.device,
) -> dict[str, object]:
    points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    with torch.no_grad():
        prediction_difference = endpoint_model(points) - taylor_model(points)
    return {
        "gamma": gamma,
        "eta": eta,
        "xi": xi,
        "endpoint_final_normalized_loss": endpoint_metrics["final_normalized_loss"],
        "taylor_final_normalized_loss": taylor_metrics["final_normalized_loss"],
        "endpoint_rms_solution_error": endpoint_metrics["rms_solution_error"],
        "taylor_rms_solution_error": taylor_metrics["rms_solution_error"],
        "rms_prediction_difference": _rms(prediction_difference),
        "max_abs_prediction_difference": float(prediction_difference.abs().max().cpu()),
        "endpoint_model_rms_cross_operator_difference": endpoint_metrics[
            "rms_representation_operator_difference"
        ],
        "taylor_model_rms_cross_operator_difference": taylor_metrics[
            "rms_representation_operator_difference"
        ],
    }


def save_metrics_json(payload: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_comparison_plot(
    results: Sequence[
        tuple[float, float, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
    ],
    gamma: float,
    config: TrainingConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(7.2 * columns, 4.2 * rows), squeeze=False, constrained_layout=True
    )
    for axis, (eta, xi, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics) in zip(
        axes.flat, results
    ):
        device = torch.device(config.device)
        grid = torch.linspace(-1.0, 1.0, 401, dtype=DTYPE, device=device).reshape(-1, 1)
        target = exact_solution(grid, gamma, eta, xi)
        with torch.no_grad():
            endpoint_error = (endpoint_model(grid) - target).cpu().numpy().reshape(-1)
            taylor_error = (taylor_model(grid) - target).cpu().numpy().reshape(-1)
        grid_np = grid.cpu().numpy().reshape(-1)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.plot(grid_np, endpoint_error, linewidth=1.8, label="endpoint formula")
        axis.plot(grid_np, taylor_error, linewidth=1.8, label="Taylor formula")
        axis.set(
            title=(
                f"gamma={gamma:g}, eta={eta:g}, xi={xi:g}; "
                f"RMS={endpoint_metrics['rms_solution_error']:.2e} / "
                f"{taylor_metrics['rms_solution_error']:.2e}"
            ),
            xlabel="t",
            ylabel="prediction error",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    for axis in axes.flat[len(results):]:
        axis.remove()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, action="append")
    parser.add_argument("--xi", type=float, action="append")
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
    if args.xi is None:
        etas = tuple(args.eta) if args.eta is not None else DEFAULT_ETAS
        pairs = tuple((eta, 0.0) for eta in etas)
    else:
        etas = tuple(args.eta) if args.eta is not None else (1.0,)
        if len(etas) != 1:
            raise ValueError("a cubic xi sweep requires exactly one fixed eta")
        pairs = tuple((etas[0], xi) for xi in args.xi)
    if not pairs:
        raise ValueError("at least one (eta, xi) pair is required")

    results: list[
        tuple[float, float, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
    ] = []
    comparisons: list[dict[str, object]] = []
    for eta, xi in pairs:
        endpoint_model, endpoint_metrics = train_one(
            args.gamma, eta, "endpoint", config, xi=xi
        )
        taylor_model, taylor_metrics = train_one(
            args.gamma, eta, "taylor", config, xi=xi
        )
        results.append(
            (eta, xi, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics)
        )
        comparisons.append(
            build_comparison(
                eta,
                xi,
                endpoint_model,
                endpoint_metrics,
                taylor_model,
                taylor_metrics,
                args.gamma,
                torch.device(config.device),
            )
        )

    payload: dict[str, object] = {
        "experiment": "polynomial_exact_solution_representation_comparison",
        "gamma": args.gamma,
        "pairs": [{"eta": eta, "xi": xi} for eta, xi in pairs],
        "exact_solution": "1 + gamma*t + eta*t^2 + xi*t^3",
        "config": asdict(config),
        "runs": [
            item
            for _, _, _, endpoint_metrics, _, taylor_metrics in results
            for item in (endpoint_metrics, taylor_metrics)
        ],
        "comparisons": comparisons,
    }
    if args.json_out is not None:
        save_metrics_json(payload, args.json_out)
        print(f"wrote comparison JSON: {args.json_out}")
    if args.png_out is not None:
        save_comparison_plot(results, args.gamma, config, args.png_out)
        print(f"wrote comparison PNG: {args.png_out}")
    for comparison in comparisons:
        print(
            f"eta={comparison['eta']:g} "
            f"xi={comparison['xi']:g} "
            f"endpoint_rms={comparison['endpoint_rms_solution_error']:.6e} "
            f"taylor_rms={comparison['taylor_rms_solution_error']:.6e} "
            f"prediction_difference={comparison['rms_prediction_difference']:.6e}"
        )
    return payload


if __name__ == "__main__":
    main()
