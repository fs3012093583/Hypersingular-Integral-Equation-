"""Compare two finite-part representations for first- and third-order kernels.

For ``m=1`` the operator is the Cauchy principal value

    I_1[u](t) = PV int_-1^1 u(tau)/(tau-t) d tau.

For ``m=3`` it is the Hadamard finite part

    I_3[u](t) = FP int_-1^1 u(tau)/(tau-t)^3 d tau.

Each operator is evaluated in two analytically equivalent ways: a complete
Taylor-subtraction formula and a formula obtained by integration by parts.
The manufactured equation uses ``u(t)=1+t+t^2+t^3`` and otherwise retains the
coefficient functions used by the preceding second-order experiments.
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

from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    TanhNetwork,
    gauss_legendre_rule,
    set_reproducible_seed,
)


Representation = Literal["endpoint", "taylor"]
SupportedOrder = Literal[1, 3]
DEFAULT_ORDERS = (1, 3)


def _as_column(
    value: Tensor | Sequence[float] | float,
    device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(dtype=DTYPE, device=device if device is not None else value.device)
    else:
        result = torch.as_tensor(value, dtype=DTYPE, device=device)
    return result.reshape(-1, 1)


def _derivative(values: Tensor, inputs: Tensor) -> Tensor:
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


def _value_and_derivatives(
    u: Callable[[Tensor], Tensor],
    points: Tensor,
    highest_order: int,
) -> list[Tensor]:
    values = _as_column(u(points), device=points.device)
    if values.shape != points.shape:
        raise ValueError("u must return one scalar value per input point")
    result = [values]
    for _ in range(highest_order):
        result.append(_derivative(result[-1], points))
    return result


def coefficient_a(t: Tensor) -> Tensor:
    return t.pow(4)


def coefficient_b(t: Tensor) -> Tensor:
    return (2.0 - t) * (1.0 + t)


def exact_solution(t: Tensor) -> Tensor:
    return 1.0 + t + t.square() + t.pow(3)


def exact_singular_operator(t: Tensor, order: int) -> Tensor:
    """Analytic ``I_1`` or ``I_3`` for ``u(t)=1+t+t^2+t^3``."""
    value = exact_solution(t)
    first = 1.0 + 2.0 * t + 3.0 * t.square()
    logarithm = torch.log((1.0 - t) / (1.0 + t))
    if order == 1:
        return value * logarithm + 8.0 / 3.0 + 2.0 * t + 2.0 * t.square()
    if order == 3:
        second = 2.0 + 6.0 * t
        j3 = 0.5 * (1.0 / (1.0 + t).square() - 1.0 / (1.0 - t).square())
        j2 = -(1.0 / (1.0 + t) + 1.0 / (1.0 - t))
        return value * j3 + first * j2 + 0.5 * second * logarithm + 2.0
    raise ValueError("only orders 1 and 3 are supported")


def equation_rhs(t: Tensor, order: int) -> Tensor:
    return coefficient_a(t) * exact_solution(t) + coefficient_b(t) * exact_singular_operator(
        t, order
    )


def taylor_subtraction_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    order: int,
    near_zero_threshold: float = 1.0e-3,
) -> Tensor:
    """Evaluate the complete Taylor-subtraction representation for order 1 or 3."""
    if order not in (1, 3):
        raise ValueError("only orders 1 and 3 are supported")
    if near_zero_threshold <= 0.0:
        raise ValueError("near_zero_threshold must be positive")
    device = quadrature_nodes.device
    t_col = _as_column(t, device=device).detach().clone().requires_grad_(True)
    tau = _as_column(quadrature_nodes, device=device).reshape(1, -1)
    weights = _as_column(quadrature_weights, device=device).reshape(1, -1)
    if tau.shape != weights.shape:
        raise ValueError("quadrature_nodes and quadrature_weights must have the same length")

    highest_t_derivative = 4 if order == 3 else 1
    derivatives = _value_and_derivatives(u, t_col, highest_t_derivative)
    u_t = derivatives[0]
    u_tau = _as_column(u(tau.reshape(-1, 1)), device=device).reshape(1, -1)
    delta = tau - t_col
    near_zero = delta.abs() <= near_zero_threshold
    safe_delta = torch.where(near_zero, torch.ones_like(delta), delta)
    logarithm = torch.log((1.0 - t_col) / (1.0 + t_col))

    if order == 1:
        quotient = (u_tau - u_t) / safe_delta
        integrand = torch.where(near_zero, derivatives[1], quotient)
        regular_integral = torch.sum(integrand * weights, dim=1, keepdim=True)
        return u_t * logarithm + regular_integral

    first = derivatives[1]
    second = derivatives[2]
    numerator = u_tau - u_t - first * delta - 0.5 * second * delta.square()
    quotient = numerator / safe_delta.pow(3)
    # The direct cubic divided difference suffers severe cancellation close to
    # the diagonal.  Keeping the next Taylor term makes the replacement
    # fourth-order accurate in the numerator while retaining differentiability.
    diagonal_expansion = derivatives[3] / 6.0 + derivatives[4] * delta / 24.0
    integrand = torch.where(near_zero, diagonal_expansion, quotient)
    regular_integral = torch.sum(integrand * weights, dim=1, keepdim=True)
    j3 = 0.5 * (1.0 / (1.0 + t_col).square() - 1.0 / (1.0 - t_col).square())
    j2 = -(1.0 / (1.0 + t_col) + 1.0 / (1.0 - t_col))
    return u_t * j3 + first * j2 + 0.5 * second * logarithm + regular_integral


def endpoint_integration_by_parts_operator(
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    order: int,
    near_zero_threshold: float = 1.0e-3,
) -> Tensor:
    """Evaluate the integration-by-parts endpoint representation.

    At order one this is the logarithmic integration-by-parts formula.  At
    order three it is obtained by applying integration by parts twice and
    evaluating the remaining principal value after derivative subtraction.
    """
    if order not in (1, 3):
        raise ValueError("only orders 1 and 3 are supported")
    if near_zero_threshold <= 0.0:
        raise ValueError("near_zero_threshold must be positive")
    device = quadrature_nodes.device
    t_col = _as_column(t, device=device).detach().clone().requires_grad_(True)
    tau_col = _as_column(quadrature_nodes, device=device).detach().clone().requires_grad_(True)
    weights = _as_column(quadrature_weights, device=device).reshape(1, -1)
    if tau_col.numel() != weights.numel():
        raise ValueError("quadrature_nodes and quadrature_weights must have the same length")

    t_derivatives = _value_and_derivatives(u, t_col, 4 if order == 3 else 1)
    tau_derivatives = _value_and_derivatives(u, tau_col, 2 if order == 3 else 1)
    endpoint_points = torch.tensor(
        [[-1.0], [1.0]], dtype=DTYPE, device=device, requires_grad=True
    )
    endpoint_derivatives = _value_and_derivatives(u, endpoint_points, 1 if order == 3 else 0)
    u_minus = endpoint_derivatives[0][0:1]
    u_plus = endpoint_derivatives[0][1:2]
    tau = tau_col.reshape(1, -1)
    delta = tau - t_col
    near_zero = delta.abs() <= near_zero_threshold
    safe_delta = torch.where(near_zero, torch.ones_like(delta), delta)

    if order == 1:
        log_abs_delta = torch.log(safe_delta.abs())
        difference = tau_derivatives[1].reshape(1, -1) - t_derivatives[1]
        integrand = torch.where(near_zero, torch.zeros_like(delta), difference * log_abs_delta)
        regular_integral = torch.sum(integrand * weights, dim=1, keepdim=True)
        distance_right = 1.0 - t_col
        distance_left = 1.0 + t_col
        log_moment = (
            distance_right * torch.log(distance_right)
            + distance_left * torch.log(distance_left)
            - 2.0
        )
        boundary = u_plus * torch.log(distance_right) - u_minus * torch.log(distance_left)
        return boundary - t_derivatives[1] * log_moment - regular_integral

    u_prime_minus = endpoint_derivatives[1][0:1]
    u_prime_plus = endpoint_derivatives[1][1:2]
    second_difference = tau_derivatives[2].reshape(1, -1) - t_derivatives[2]
    quotient = second_difference / safe_delta
    diagonal_expansion = t_derivatives[3] + 0.5 * t_derivatives[4] * delta
    integrand = torch.where(near_zero, diagonal_expansion, quotient)
    regular_integral = torch.sum(integrand * weights, dim=1, keepdim=True)
    logarithm = torch.log((1.0 - t_col) / (1.0 + t_col))
    boundary = (
        u_minus / (2.0 * (1.0 + t_col).square())
        - u_plus / (2.0 * (1.0 - t_col).square())
        - u_prime_minus / (2.0 * (1.0 + t_col))
        - u_prime_plus / (2.0 * (1.0 - t_col))
    )
    return boundary + 0.5 * t_derivatives[2] * logarithm + 0.5 * regular_integral


def singular_operator(
    representation: Representation,
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    order: int,
    near_zero_threshold: float = 1.0e-3,
) -> Tensor:
    if representation == "endpoint":
        return endpoint_integration_by_parts_operator(
            u, t, quadrature_nodes, quadrature_weights, order, near_zero_threshold
        )
    if representation == "taylor":
        return taylor_subtraction_operator(
            u, t, quadrature_nodes, quadrature_weights, order, near_zero_threshold
        )
    raise ValueError(f"unknown representation: {representation}")


def equation_operator_value(
    representation: Representation,
    u: Callable[[Tensor], Tensor],
    t: Tensor | Sequence[float],
    quadrature_nodes: Tensor,
    quadrature_weights: Tensor,
    order: int,
    near_zero_threshold: float = 1.0e-3,
) -> Tensor:
    t_col = _as_column(t, device=quadrature_nodes.device)
    u_value = _as_column(u(t_col), device=t_col.device)
    return coefficient_a(t_col) * u_value + coefficient_b(t_col) * singular_operator(
        representation,
        u,
        t_col,
        quadrature_nodes,
        quadrature_weights,
        order,
        near_zero_threshold,
    )


def normalized_equation_residual(operator_value: Tensor, t: Tensor, order: int) -> Tensor:
    endpoint_balance = (0.5 * (1.0 - t.square())).pow(order - 1)
    return endpoint_balance * (operator_value - equation_rhs(t, order)) / 4.0


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
    near_zero_threshold: float = 1.0e-3
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
    order: int,
    representation: Representation,
    config: TrainingConfig,
    verbose: bool = True,
) -> tuple[TanhNetwork, dict[str, object]]:
    if order not in (1, 3):
        raise ValueError("only orders 1 and 3 are supported")
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
        operator_value = equation_operator_value(
            representation,
            model,
            collocation_points,
            nodes,
            weights,
            order,
            config.near_zero_threshold,
        )
        return torch.mean(
            normalized_equation_residual(operator_value, collocation_points, order).square()
        )

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite order-{order} {representation} loss at epoch {epoch}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (epoch == 1 or epoch % config.report_every == 0 or epoch == config.epochs):
            print(
                f"order={order} representation={representation} "
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
                    f"non-finite order-{order} {representation} L-BFGS loss"
                )
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        post_step_loss = objective()
        if not torch.isfinite(post_step_loss):
            raise FloatingPointError(
                f"non-finite order-{order} {representation} loss after L-BFGS"
            )
        losses.append(float(post_step_loss.detach().cpu()))
        if verbose:
            print(
                f"order={order} representation={representation} "
                f"lbfgs_max_iter={config.lbfgs_steps:4d} loss={losses[-1]:.6e}"
            )

    model.eval()
    solution_points = torch.linspace(-1.0, 1.0, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    residual_points = torch.linspace(-0.95, 0.95, 201, dtype=DTYPE, device=device).reshape(-1, 1)
    validation_num_quadrature = max(2 * config.num_quadrature, 128)
    validation_nodes, validation_weights = gauss_legendre_rule(validation_num_quadrature, device)
    with torch.enable_grad():
        endpoint_values = equation_operator_value(
            "endpoint",
            model,
            residual_points,
            validation_nodes,
            validation_weights,
            order,
            config.near_zero_threshold,
        )
        taylor_values = equation_operator_value(
            "taylor",
            model,
            residual_points,
            validation_nodes,
            validation_weights,
            order,
            config.near_zero_threshold,
        )
    with torch.no_grad():
        target_solution = exact_solution(solution_points)
        solution_error = model(solution_points) - target_solution
        interior_error = model(residual_points) - exact_solution(residual_points)
        rhs = equation_rhs(residual_points, order)
        endpoint_residual = endpoint_values - rhs
        taylor_residual = taylor_values - rhs
        representation_difference = endpoint_values - taylor_values
        endpoint_points = torch.tensor([[-1.0], [1.0]], dtype=DTYPE, device=device)
        endpoint_errors = (model(endpoint_points) - exact_solution(endpoint_points)).abs().reshape(-1)
        own_residual = endpoint_residual if representation == "endpoint" else taylor_residual
        alternate_residual = taylor_residual if representation == "endpoint" else endpoint_residual
        metrics: dict[str, object] = {
            "order": order,
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
    order: int,
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
        "order": order,
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
        tuple[int, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
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
    target = exact_solution(grid)
    for axis, (order, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics) in zip(
        axes.flat, results
    ):
        with torch.no_grad():
            endpoint_error = (endpoint_model(grid) - target).cpu().numpy().reshape(-1)
            taylor_error = (taylor_model(grid) - target).cpu().numpy().reshape(-1)
        grid_np = grid.cpu().numpy().reshape(-1)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.plot(grid_np, endpoint_error, linewidth=1.8, label="integration by parts")
        axis.plot(grid_np, taylor_error, linewidth=1.8, label="Taylor subtraction")
        axis.set(
            title=(
                f"kernel order m={order}; RMS="
                f"{endpoint_metrics['rms_solution_error']:.2e} / "
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
    parser.add_argument("--order", type=int, choices=(1, 3), action="append")
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
    orders = tuple(args.order) if args.order is not None else DEFAULT_ORDERS
    if len(set(orders)) != len(orders):
        raise ValueError("each requested order must appear only once")
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
        tuple[int, TanhNetwork, dict[str, object], TanhNetwork, dict[str, object]]
    ] = []
    comparisons: list[dict[str, object]] = []
    for order in orders:
        endpoint_model, endpoint_metrics = train_one(order, "endpoint", config)
        taylor_model, taylor_metrics = train_one(order, "taylor", config)
        results.append(
            (order, endpoint_model, endpoint_metrics, taylor_model, taylor_metrics)
        )
        comparisons.append(
            build_comparison(
                order,
                endpoint_model,
                endpoint_metrics,
                taylor_model,
                taylor_metrics,
                torch.device(config.device),
            )
        )

    payload: dict[str, object] = {
        "experiment": "singular_kernel_order_representation_comparison",
        "orders": list(orders),
        "exact_solution": "1+t+t^2+t^3",
        "equation": "t^4*u(t)+(2-t)*(1+t)*I_m[u](t)=f_m(t)",
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
            f"order={comparison['order']} "
            f"endpoint_rms={comparison['endpoint_rms_solution_error']:.6e} "
            f"taylor_rms={comparison['taylor_rms_solution_error']:.6e} "
            f"prediction_difference={comparison['rms_prediction_difference']:.6e}"
        )
    return payload


if __name__ == "__main__":
    main()
