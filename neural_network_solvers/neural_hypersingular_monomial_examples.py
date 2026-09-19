"""Neural solutions of two handwritten monomial hypersingular examples.

On ``(-1, 1)`` solve

    FP int_-1^1 u(x)/(x-t)^2 dx = f(t)

for the manufactured exact solutions ``u(t)=t`` and ``u(t)=t**2``.  Every
case is trained independently with both the endpoint-value representation and
the complete Taylor-subtraction representation of the same finite-part
operator.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor

from neural_network_solvers.neural_hypersingular_constant_endpoint import (
    endpoint_hypersingular_operator,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TanhNetwork,
    gauss_legendre_rule,
    hypersingular_operator as taylor_hypersingular_operator,
    set_reproducible_seed,
)


CaseName = Literal["linear", "quadratic"]
Representation = Literal["endpoint", "taylor"]
DEFAULT_CASES: tuple[CaseName, ...] = ("linear", "quadratic")


def _as_column(
    value: Tensor | Sequence[float] | float,
    device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def exact_solution(t: Tensor, case: CaseName) -> Tensor:
    if case == "linear":
        return t
    if case == "quadratic":
        return t.square()
    raise ValueError(f"unknown case: {case}")


def equation_rhs(t: Tensor, case: CaseName) -> Tensor:
    """Specialize the photographed formulas to ``a=-1`` and ``b=1``."""
    singular_coefficient = 1.0 / (1.0 - t) + 1.0 / (1.0 + t)
    logarithm = torch.log((1.0 - t) / (1.0 + t))
    if case == "linear":
        return -t * singular_coefficient + logarithm
    if case == "quadratic":
        return -t.square() * singular_coefficient + 2.0 * t * logarithm + 2.0
    raise ValueError(f"unknown case: {case}")


def operator_value(
    representation: Representation,
    model: TanhNetwork,
    t: Tensor,
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    near_zero_threshold: float,
) -> Tensor:
    if representation == "endpoint":
        return endpoint_hypersingular_operator(
            model,
            t,
            quadrature_nodes,
            quadrature_weights,
            near_zero_threshold,
        )
    if representation == "taylor":
        return taylor_hypersingular_operator(
            model,
            t,
            quadrature_nodes,
            quadrature_weights,
            near_zero_threshold,
        )
    raise ValueError(f"unknown representation: {representation}")


def normalized_residual(operator: Tensor, t: Tensor, case: CaseName) -> Tensor:
    """Balance the first-order endpoint growth without changing interior zeros."""
    return 0.5 * (1.0 - t.square()) * (operator - equation_rhs(t, case))


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
    case: CaseName,
    representation: Representation,
    config: TrainingConfig,
    verbose: bool = True,
) -> tuple[TanhNetwork, dict[str, object]]:
    _validate_config(config)
    device = torch.device(config.device)
    set_reproducible_seed(config.seed)
    model = TanhNetwork(config.hidden_width, config.hidden_layers).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    angles = torch.linspace(0.0, torch.pi, config.num_collocation, dtype=DTYPE, device=device)
    collocation_points = (0.97 * torch.cos(angles)).reshape(-1, 1)
    nodes, weights = gauss_legendre_rule(config.num_quadrature, device)
    losses: list[float] = []

    def objective() -> Tensor:
        value = operator_value(
            representation,
            model,
            collocation_points,
            nodes,
            weights,
            config.near_zero_threshold,
        )
        return torch.mean(normalized_residual(value, collocation_points, case).square())

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite {case} {representation} loss at epoch {epoch}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (epoch == 1 or epoch % config.report_every == 0 or epoch == config.epochs):
            print(
                f"case={case} representation={representation} "
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
                raise FloatingPointError(f"non-finite {case} {representation} L-BFGS loss")
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        post_step_loss = objective()
        if not torch.isfinite(post_step_loss):
            raise FloatingPointError(
                f"non-finite {case} {representation} loss after L-BFGS"
            )
        losses.append(float(post_step_loss.detach().cpu()))
        if verbose:
            print(
                f"case={case} representation={representation} "
                f"lbfgs_max_iter={config.lbfgs_steps:4d} loss={losses[-1]:.6e}"
            )

    model.eval()
    solution_points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    residual_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        endpoint_values = operator_value(
            "endpoint",
            model,
            residual_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
        taylor_values = operator_value(
            "taylor",
            model,
            residual_points,
            validation_nodes,
            validation_weights,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        solution_error = model(solution_points) - exact_solution(solution_points, case)
        interior_error = model(residual_points) - exact_solution(residual_points, case)
        rhs = equation_rhs(residual_points, case)
        endpoint_residual = endpoint_values - rhs
        taylor_residual = taylor_values - rhs
        difference = endpoint_values - taylor_values
        endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
        endpoint_errors = (
            model(endpoint_points) - exact_solution(endpoint_points, case)
        ).abs().reshape(-1)
        own_residual = endpoint_residual if representation == "endpoint" else taylor_residual
        alternate_residual = taylor_residual if representation == "endpoint" else endpoint_residual
        metrics: dict[str, object] = {
            "case": case,
            "exact_solution": "t" if case == "linear" else "t^2",
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
            "rms_representation_operator_difference": _rms(difference),
            "max_abs_representation_operator_difference": float(difference.abs().max().cpu()),
            "loss_history": losses,
        }
    return model, metrics


def build_comparison(
    case: CaseName,
    endpoint_model: TanhNetwork,
    endpoint_metrics: dict[str, object],
    taylor_model: TanhNetwork,
    taylor_metrics: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    with torch.no_grad():
        prediction_difference = endpoint_model(points) - taylor_model(points)
    return {
        "case": case,
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
        tuple[CaseName, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
    ],
    config: TrainingConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, len(results), figsize=(7.0 * len(results), 4.3), squeeze=False, constrained_layout=True
    )
    device = torch.device(config.device)
    grid = torch.linspace(-1.0, 1.0, 401, dtype=DTYPE, device=device).reshape(-1, 1)
    for axis, (case, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics) in zip(
        axes.flat, results
    ):
        target = exact_solution(grid, case)
        with torch.no_grad():
            endpoint_error = (endpoint_model(grid) - target).cpu().numpy().reshape(-1)
            taylor_error = (taylor_model(grid) - target).cpu().numpy().reshape(-1)
        grid_np = grid.cpu().numpy().reshape(-1)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.plot(grid_np, endpoint_error, linewidth=1.8, label="endpoint formula")
        axis.plot(grid_np, taylor_error, linewidth=1.8, label="Taylor formula")
        exact_label = "u(t)=t" if case == "linear" else "u(t)=t^2"
        axis.set(
            title=(
                f"{exact_label}; RMS={endpoint_metrics['rms_solution_error']:.2e} / "
                f"{taylor_metrics['rms_solution_error']:.2e}"
            ),
            xlabel="t",
            ylabel="prediction error",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=DEFAULT_CASES, action="append")
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
    cases: tuple[CaseName, ...] = tuple(args.case) if args.case is not None else DEFAULT_CASES
    if len(set(cases)) != len(cases):
        raise ValueError("each requested case must appear only once")
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
    results: list[
        tuple[CaseName, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
    ] = []
    comparisons: list[dict[str, object]] = []
    for case in cases:
        endpoint_model, endpoint_metrics = train_one(case, "endpoint", config)
        taylor_model, taylor_metrics = train_one(case, "taylor", config)
        results.append((case, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics))
        comparisons.append(
            build_comparison(
                case,
                endpoint_model,
                endpoint_metrics,
                taylor_model,
                taylor_metrics,
                torch.device(config.device),
            )
        )

    payload: dict[str, object] = {
        "experiment": "handwritten_linear_quadratic_hypersingular_examples",
        "interval": [-1.0, 1.0],
        "equation": "FP integral u(x)/(x-t)^2 dx = f(t)",
        "cases": list(cases),
        "config": asdict(config),
        "runs": [
            item
            for _, _, endpoint_metrics, _, taylor_metrics in results
            for item in (endpoint_metrics, taylor_metrics)
        ],
        "comparisons": comparisons,
    }
    if args.json_out is not None:
        save_metrics_json(payload, args.json_out)
        print(f"wrote comparison JSON: {args.json_out}")
    if args.png_out is not None:
        save_comparison_plot(results, config, args.png_out)
        print(f"wrote comparison PNG: {args.png_out}")
    for comparison in comparisons:
        print(
            f"case={comparison['case']} "
            f"endpoint_rms={comparison['endpoint_rms_solution_error']:.6e} "
            f"taylor_rms={comparison['taylor_rms_solution_error']:.6e} "
            f"prediction_difference={comparison['rms_prediction_difference']:.6e}"
        )
    return payload


if __name__ == "__main__":
    main()
