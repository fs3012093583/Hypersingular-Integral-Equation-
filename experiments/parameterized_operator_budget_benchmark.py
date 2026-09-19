"""Parameterized neural-collocation validation for the operator-budget method.

A single network represents ``u(t, xi)`` on a parameter interval and is
trained on

    u(t, xi) + lambda (1-t**2)**(m-1) I_m[u(., xi)](t) = f(t, xi),

with the manufactured family ``u(t, xi)=exp(xi*t)``.  The training parameters
and held-out interpolation parameters are disjoint.  Fixed local continuation
degree ``p=2`` and the row-wise operator-budget selector share initial weights,
collocation points, quadrature rules, right-hand sides, and optimizers.

This experiment tests compatibility with one parameterized solution map.  It
is not evidence of high-dimensional generalization or superiority over a
specialized spectral solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import torch
from torch import Tensor, nn

from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule
from experiments.general_taylor_neural_benchmark import (
    _interior_chebyshev_points,
    set_seed,
)
from neural_network_solvers.general_taylor_finite_part import (
    OperatorBudgetTaylorDiagnostics,
    lazy_taylor_finite_part_operator,
    operator_budget_taylor_finite_part_operator,
    scaled_derivative_error_model,
    taylor_finite_part_operator,
    value_and_derivatives,
)
from neural_network_solvers.high_precision_tanh_reference import (
    high_precision_tanh_partial_derivatives,
)


Representation = Literal["fixed_p2", "lazy_fixed_p2", "operator_budget"]


@dataclass(frozen=True)
class ParameterizedConfig:
    representation: Representation = "operator_budget"
    order: int = 4
    coupling: float = 0.05
    seed: int = 20260918
    steps: int = 300
    lbfgs_steps: int = 30
    learning_rate: float = 2.0e-3
    width: int = 24
    hidden_layers: int = 3
    collocation_points: int = 32
    training_quadrature_points: int = 48
    rhs_quadrature_points: int = 128
    validation_points: int = 121
    validation_quadrature_points: int = 160
    collocation_limit: float = 0.95
    validation_limit: float = 0.98
    training_parameters: tuple[float, ...] = (0.5, 1.0, 1.5)
    validation_parameters: tuple[float, ...] = (0.625, 0.875, 1.125, 1.375)
    max_continuation_order: int = 3
    target_residual_error: float = 1.0e-4
    derivative_envelope_safety: float = 4.0
    derivative_error_safety: float = 8.0
    envelope_refresh_interval: int = 100
    envelope_calibration_points: int = 33
    rhs_continuation_order: int = 3
    timing_repeats: int = 3
    dtype: str = "float64"
    device: str = "cpu"


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "steps": 2,
        "lbfgs_steps": 1,
        "width": 8,
        "hidden_layers": 1,
        "collocation_points": 6,
        "training_quadrature_points": 12,
        "rhs_quadrature_points": 32,
        "validation_points": 21,
        "validation_quadrature_points": 32,
        "envelope_refresh_interval": 1,
        "envelope_calibration_points": 7,
        "timing_repeats": 1,
    },
    "paper": {},
}


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _device(name: str) -> torch.device:
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ParameterizedTanhNetwork(nn.Module):
    """Two-input tanh MLP with the parameter interval mapped to [-1, 1]."""

    def __init__(
        self,
        width: int,
        hidden_layers: int,
        parameter_interval: tuple[float, float],
    ) -> None:
        super().__init__()
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive")
        left, right = parameter_interval
        if not left < right:
            raise ValueError("parameter_interval must be increasing")
        self.parameter_interval = parameter_interval
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def parameter_feature(self, xi: Tensor) -> Tensor:
        left, right = self.parameter_interval
        return 2.0 * (xi - left) / (right - left) - 1.0

    def forward(self, t: Tensor, xi: Tensor | float) -> Tensor:
        t_col = t.reshape(-1, 1)
        xi_col = torch.as_tensor(xi, dtype=t.dtype, device=t.device).reshape(-1, 1)
        if xi_col.numel() == 1:
            xi_col = xi_col.expand_as(t_col)
        elif xi_col.shape != t_col.shape:
            raise ValueError("xi must be scalar or have one value per t")
        return self.layers(torch.cat((t_col, self.parameter_feature(xi_col)), dim=1))


class ParameterSlice(nn.Module):
    def __init__(self, model: ParameterizedTanhNetwork, xi: float) -> None:
        super().__init__()
        self.model = model
        self.xi = float(xi)

    def forward(self, t: Tensor) -> Tensor:
        return self.model(t, self.xi)


class ExactExpSlice(nn.Module):
    def __init__(self, xi: float) -> None:
        super().__init__()
        self.xi = float(xi)

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(self.xi * t)


class ParameterizedDerivativeEnvelope:
    """Monotone empirical derivative envelope over t and training parameters."""

    def __init__(
        self,
        model: ParameterizedTanhNetwork,
        parameters: Sequence[float],
        *,
        highest_order: int,
        calibration_points: int,
        safety_factor: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.parameters = tuple(float(value) for value in parameters)
        self.highest_order = highest_order
        self.safety_factor = safety_factor
        self.points = torch.linspace(
            -1.0, 1.0, calibration_points, dtype=dtype, device=device
        ).reshape(-1, 1)
        self.bounds: dict[int, float] = {}
        self.records: list[dict[str, object]] = []
        self.total_seconds = 0.0

    def refresh(self, stage: str, step: int) -> None:
        started = time.perf_counter()
        observed = {order: 0.0 for order in range(self.highest_order + 1)}
        for xi in self.parameters:
            points = self.points.detach().clone().requires_grad_(True)
            derivatives = value_and_derivatives(
                ParameterSlice(self.model, xi), points, self.highest_order
            )
            for order, derivative in enumerate(derivatives):
                observed[order] = max(
                    observed[order], float(torch.max(torch.abs(derivative.detach())))
                )
        violations = [
            order
            for order, magnitude in observed.items()
            if order in self.bounds and magnitude > self.bounds[order]
        ]
        for order, magnitude in observed.items():
            candidate = self.safety_factor * max(magnitude, 1.0e-15)
            self.bounds[order] = max(self.bounds.get(order, 0.0), candidate)
        elapsed = time.perf_counter() - started
        self.total_seconds += elapsed
        self.records.append(
            {
                "stage": stage,
                "step": step,
                "observed": {str(key): value for key, value in observed.items()},
                "bounds": {str(key): value for key, value in self.bounds.items()},
                "previous_bound_violations": violations,
                "seconds": elapsed,
            }
        )

    def __call__(self, t: Tensor, tau: Tensor, derivative_order: int) -> Tensor:
        return torch.full_like(t + tau, self.bounds[derivative_order])


def _calibrate_ad_error(
    model: ParameterizedTanhNetwork,
    parameters: Sequence[float],
    *,
    highest_order: int,
    safety_factor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[int, float]:
    """Build an offline scaled AD-error envelope for this two-input MLP."""
    epsilon = torch.finfo(dtype).eps
    scaled = {order: 2.0 * epsilon for order in range(highest_order + 1)}
    calibration_t = (-0.8, 0.0, 0.8)
    parameter_subset = (min(parameters), statistics.median(parameters), max(parameters))
    for xi in parameter_subset:
        feature = float(
            model.parameter_feature(
                torch.tensor([[xi]], dtype=dtype, device=device)
            ).item()
        )
        for t_value in calibration_t:
            point = torch.tensor(
                [[t_value]], dtype=dtype, device=device, requires_grad=True
            )
            observed = value_and_derivatives(
                ParameterSlice(model, xi), point, highest_order
            )
            reference = high_precision_tanh_partial_derivatives(
                model.layers,
                [float(point.item()), feature],
                highest_order,
                derivative_input_index=0,
                decimal_digits=90,
            )
            with mp.workdps(90):
                for order in range(highest_order + 1):
                    observed_mp = mp.mpf(float(observed[order].detach().item()))
                    error = abs(observed_mp - reference[order])
                    scale = max(abs(reference[order]), mp.mpf(1))
                    scaled[order] = max(scaled[order], float(error / scale))
    return {order: safety_factor * value for order, value in scaled.items()}


def _exact_rhs(
    points: Tensor,
    xi: float,
    nodes: Tensor,
    weights: Tensor,
    config: ParameterizedConfig,
) -> Tensor:
    exact = ExactExpSlice(xi)
    finite_part = taylor_finite_part_operator(
        exact,
        points,
        nodes,
        weights,
        order=config.order,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
    )
    value = exact(points)
    return (
        value
        + config.coupling
        * (1.0 - points.square()).pow(config.order - 1)
        * finite_part
    ).detach()


def _finite_part(
    model_slice: ParameterSlice,
    points: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: ParameterizedConfig,
    *,
    derivative_error_model=None,
    derivative_bound=None,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, OperatorBudgetTaylorDiagnostics]:
    if config.representation in ("fixed_p2", "lazy_fixed_p2"):
        if return_diagnostics:
            raise ValueError("fixed p=2 does not return selector diagnostics")
        fixed_operator = (
            lazy_taylor_finite_part_operator
            if config.representation == "lazy_fixed_p2"
            else taylor_finite_part_operator
        )
        return fixed_operator(
            model_slice,
            points,
            nodes,
            weights,
            order=config.order,
            continuation_order=2,
            threshold_mode="balanced",
        )
    budget = config.target_residual_error / abs(config.coupling)
    return operator_budget_taylor_finite_part_operator(
        model_slice,
        points,
        nodes,
        weights,
        order=config.order,
        target_operator_error=budget,
        max_continuation_order=config.max_continuation_order,
        trust_region_multiplier=None,
        derivative_error_model=derivative_error_model,
        derivative_bound=derivative_bound,
        return_diagnostics=return_diagnostics,
    )


def _parameter_residual(
    model: ParameterizedTanhNetwork,
    points: Tensor,
    rhs_by_parameter: dict[float, Tensor],
    nodes: Tensor,
    weights: Tensor,
    config: ParameterizedConfig,
    *,
    derivative_error_model=None,
    derivative_bound=None,
) -> Tensor:
    residuals: list[Tensor] = []
    for xi in config.training_parameters:
        model_slice = ParameterSlice(model, xi)
        value = model_slice(points)
        finite_part = _finite_part(
            model_slice,
            points,
            nodes,
            weights,
            config,
            derivative_error_model=derivative_error_model,
            derivative_bound=derivative_bound,
        )
        assert isinstance(finite_part, Tensor)
        residuals.append(
            value
            + config.coupling
            * (1.0 - points.square()).pow(config.order - 1)
            * finite_part
            - rhs_by_parameter[xi]
        )
    return torch.cat(residuals, dim=0)


def _saved_tensor_bytes(loss_action) -> int:
    saved = 0

    def pack(tensor: Tensor) -> Tensor:
        nonlocal saved
        saved += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_action()
        loss.backward()
    return saved


def run_one(config: ParameterizedConfig, *, verbose: bool = False) -> dict[str, object]:
    if config.order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    if config.representation not in ("fixed_p2", "lazy_fixed_p2", "operator_budget"):
        raise ValueError("unsupported representation")
    if set(config.training_parameters) & set(config.validation_parameters):
        raise ValueError("training and validation parameters must be disjoint")
    all_parameters = config.training_parameters + config.validation_parameters
    parameter_interval = (min(all_parameters), max(all_parameters))
    dtype = _dtype(config.dtype)
    device = _device(config.device)
    set_seed(config.seed)
    _synchronize(device)
    total_started = time.perf_counter()
    model = ParameterizedTanhNetwork(
        config.width, config.hidden_layers, parameter_interval
    ).to(dtype=dtype, device=device)
    initial_fingerprint = _fingerprint(model)
    points = _interior_chebyshev_points(
        config.collocation_points, config.collocation_limit, dtype
    ).to(device=device)
    train_nodes, train_weights = gauss_legendre_rule(
        config.training_quadrature_points, dtype
    )
    rhs_nodes, rhs_weights = gauss_legendre_rule(config.rhs_quadrature_points, dtype)
    train_nodes, train_weights = train_nodes.to(device), train_weights.to(device)
    rhs_nodes, rhs_weights = rhs_nodes.to(device), rhs_weights.to(device)
    rhs_by_parameter = {
        xi: _exact_rhs(points, xi, rhs_nodes, rhs_weights, config)
        for xi in config.training_parameters
    }

    derivative_error_model = None
    derivative_error_envelope = None
    online_envelope = None
    calibration_seconds = 0.0
    _synchronize(device)
    method_setup_started = time.perf_counter()
    if config.representation == "operator_budget":
        calibration_started = time.perf_counter()
        highest_order = config.order + config.max_continuation_order + 1
        derivative_error_envelope = _calibrate_ad_error(
            model,
            config.training_parameters,
            highest_order=highest_order,
            safety_factor=config.derivative_error_safety,
            dtype=dtype,
            device=device,
        )
        derivative_error_model = scaled_derivative_error_model(
            derivative_error_envelope
        )
        calibration_seconds = time.perf_counter() - calibration_started
        online_envelope = ParameterizedDerivativeEnvelope(
            model,
            config.training_parameters,
            highest_order=highest_order,
            calibration_points=config.envelope_calibration_points,
            safety_factor=config.derivative_envelope_safety,
            dtype=dtype,
            device=device,
        )
        online_envelope.refresh("initial", 0)

    _synchronize(device)
    method_setup_seconds = time.perf_counter() - method_setup_started
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    _synchronize(device)
    training_started = time.perf_counter()
    for step in range(config.steps):
        if (
            online_envelope is not None
            and step > 0
            and step % config.envelope_refresh_interval == 0
        ):
            online_envelope.refresh("adam", step)
        optimizer.zero_grad(set_to_none=True)
        residual = _parameter_residual(
            model,
            points,
            rhs_by_parameter,
            train_nodes,
            train_weights,
            config,
            derivative_error_model=derivative_error_model,
            derivative_bound=online_envelope,
        )
        loss = torch.mean(residual.square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Adam loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == config.steps or (step + 1) % 100 == 0:
            history.append({"stage": "adam", "step": step + 1, "loss": float(loss.detach())})
            if verbose:
                print(
                    f"{config.representation} seed={config.seed} "
                    f"Adam {step + 1}/{config.steps} loss={float(loss.detach()):.4e}",
                    flush=True,
                )

    if online_envelope is not None:
        online_envelope.refresh("pre_lbfgs", config.steps)
    closure_calls = 0
    if config.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer_lbfgs.zero_grad(set_to_none=True)
            residual = _parameter_residual(
                model,
                points,
                rhs_by_parameter,
                train_nodes,
                train_weights,
                config,
                derivative_error_model=derivative_error_model,
                derivative_bound=online_envelope,
            )
            current_loss = torch.mean(residual.square())
            if not torch.isfinite(current_loss):
                raise FloatingPointError("non-finite L-BFGS loss")
            current_loss.backward()
            return current_loss

        optimizer_lbfgs.step(closure)
    if online_envelope is not None:
        online_envelope.refresh("final", config.steps)
    final_loss = torch.mean(
        _parameter_residual(
            model,
            points,
            rhs_by_parameter,
            train_nodes,
            train_weights,
            config,
            derivative_error_model=derivative_error_model,
            derivative_bound=online_envelope,
        ).square()
    )
    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    solve_seconds = time.perf_counter() - total_started

    closed = torch.linspace(
        -1.0, 1.0, config.validation_points, dtype=dtype, device=device
    ).reshape(-1, 1)
    interior = _interior_chebyshev_points(
        config.validation_points, config.validation_limit, dtype
    ).to(device=device)
    validation_nodes, validation_weights = gauss_legendre_rule(
        config.validation_quadrature_points, dtype
    )
    validation_nodes = validation_nodes.to(device)
    validation_weights = validation_weights.to(device)

    solution_by_parameter: dict[str, float] = {}
    residual_by_parameter: dict[str, float] = {}
    solution_errors: list[Tensor] = []
    residuals: list[Tensor] = []
    for xi in config.validation_parameters:
        model_slice = ParameterSlice(model, xi)
        with torch.no_grad():
            error = model_slice(closed) - ExactExpSlice(xi)(closed)
        validation_rhs = _exact_rhs(
            interior, xi, validation_nodes, validation_weights, config
        )
        canonical_finite_part = taylor_finite_part_operator(
            model_slice,
            interior,
            validation_nodes,
            validation_weights,
            order=config.order,
            continuation_order=config.rhs_continuation_order,
            threshold_mode="balanced",
        )
        canonical_residual = (
            model_slice(interior)
            + config.coupling
            * (1.0 - interior.square()).pow(config.order - 1)
            * canonical_finite_part
            - validation_rhs
        )
        solution_errors.append(error.detach())
        residuals.append(canonical_residual.detach())
        solution_by_parameter[str(xi)] = float(torch.sqrt(torch.mean(error.square())))
        residual_by_parameter[str(xi)] = float(
            torch.sqrt(torch.mean(canonical_residual.detach().square()))
        )

    timings: list[float] = []
    payloads: list[int] = []
    for repeat in range(config.timing_repeats + 1):
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        payload = _saved_tensor_bytes(
            lambda: torch.mean(
                _parameter_residual(
                    model,
                    points,
                    rhs_by_parameter,
                    train_nodes,
                    train_weights,
                    config,
                    derivative_error_model=derivative_error_model,
                    derivative_bound=online_envelope,
                ).square()
            )
        )
        _synchronize(device)
        if repeat > 0:
            timings.append(time.perf_counter() - started)
            payloads.append(payload)

    selection_summary = None
    if config.representation == "operator_budget":
        counts: Counter[int] = Counter()
        highest_orders: list[int] = []
        budget_ratios: list[float] = []
        for xi in config.training_parameters:
            _, diagnostics = _finite_part(
                ParameterSlice(model, xi),
                points,
                train_nodes,
                train_weights,
                config,
                derivative_error_model=derivative_error_model,
                derivative_bound=online_envelope,
                return_diagnostics=True,
            )
            counts.update(int(value) for value in diagnostics.selected_order.reshape(-1).tolist())
            highest_orders.append(diagnostics.highest_derivative_order)
            budget_ratios.extend(
                (
                    diagnostics.selected_operator_error_estimate
                    / diagnostics.target_operator_error
                )
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )
        total = sum(counts.values())
        selection_summary = {
            "selected_counts": {str(key): value for key, value in sorted(counts.items())},
            "direct_fraction": counts[-1] / total,
            "mean_highest_derivative_order": statistics.mean(highest_orders),
            "maximum_operator_budget_ratio": max(budget_ratios),
            "budget_satisfaction_fraction": statistics.mean(
                float(value <= 1.0) for value in budget_ratios
            ),
        }

    all_solution_errors = torch.cat(solution_errors, dim=0)
    all_residuals = torch.cat(residuals, dim=0)
    return {
        "experiment": "parameterized_operator_budget_benchmark",
        "status": "ok",
        "method": config.representation,
        "config": asdict(config),
        "initial_model_fingerprint": initial_fingerprint,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_training_loss": float(final_loss.detach()),
        "loss_history": history,
        "lbfgs_closure_calls": closure_calls,
        "heldout_solution_rmse": float(
            torch.sqrt(torch.mean(all_solution_errors.square()))
        ),
        "heldout_residual_rmse": float(torch.sqrt(torch.mean(all_residuals.square()))),
        "heldout_solution_rmse_by_parameter": solution_by_parameter,
        "heldout_residual_rmse_by_parameter": residual_by_parameter,
        "solution_interval": [-1.0, 1.0],
        "residual_interval": [-config.validation_limit, config.validation_limit],
        "training_seconds_including_envelope_refresh": training_seconds,
        "method_setup_seconds": method_setup_seconds,
        "method_setup_and_training_seconds": method_setup_seconds + training_seconds,
        "solve_seconds_before_validation": solve_seconds,
        "offline_ad_calibration_seconds": calibration_seconds,
        "post_training_cost": {
            "median_residual_forward_backward_seconds": statistics.median(timings),
            "median_saved_tensor_payload_bytes": int(statistics.median(payloads)),
        },
        "selection_diagnostics": selection_summary,
        "derivative_error_scaled_envelope": derivative_error_envelope,
        "ad_calibration_reference_policy": (
            "v2: stored binary inputs/parameters; 90-digit subtraction and reference scaling"
            if derivative_error_envelope is not None else None
        ),
        "online_derivative_envelope_refreshes": (
            online_envelope.records if online_envelope is not None else None
        ),
        "online_derivative_envelope_refresh_seconds": (
            online_envelope.total_seconds if online_envelope is not None else 0.0
        ),
    }


def summarize(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ("fixed_p2", "operator_budget"):
        group = [record for record in records if record["method"] == method]
        if not group:
            continue
        def values(key: str) -> list[float]:
            return [float(record[key]) for record in group]
        solution = values("heldout_solution_rmse")
        residual = values("heldout_residual_rmse")
        times = values("training_seconds_including_envelope_refresh")
        payloads = [
            float(record["post_training_cost"]["median_saved_tensor_payload_bytes"])
            for record in group
        ]
        rows.append(
            {
                "method": method,
                "seeds": [int(record["config"]["seed"]) for record in group],
                "heldout_solution_rmse_mean": statistics.mean(solution),
                "heldout_solution_rmse_sample_std": statistics.stdev(solution) if len(solution) > 1 else 0.0,
                "heldout_residual_rmse_mean": statistics.mean(residual),
                "heldout_residual_rmse_sample_std": statistics.stdev(residual) if len(residual) > 1 else 0.0,
                "training_seconds_mean": statistics.mean(times),
                "saved_tensor_payload_bytes_mean": statistics.mean(payloads),
            }
        )
    return rows


def plot_summary(summary: Sequence[dict[str, object]], output_path: Path) -> None:
    labels = ["fixed p=2", "operator budget"]
    indexed = {str(row["method"]): row for row in summary}
    methods = ["fixed_p2", "operator_budget"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    solution = [float(indexed[method]["heldout_solution_rmse_mean"]) for method in methods]
    solution_std = [float(indexed[method]["heldout_solution_rmse_sample_std"]) for method in methods]
    times = [float(indexed[method]["training_seconds_mean"]) for method in methods]
    payload = [float(indexed[method]["saved_tensor_payload_bytes_mean"]) / 2**20 for method in methods]
    colors = ["#4472C4", "#ED7D31"]
    axes[0].bar(labels, solution, yerr=solution_std, color=colors, capsize=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("held-out solution RMSE")
    axes[1].bar(labels, times, color=colors)
    axes[1].set_ylabel("training time (s)")
    axes[2].bar(labels, payload, color=colors)
    axes[2].set_ylabel("saved tensor payload (MiB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="paper")
    parser.add_argument("--representation", choices=("fixed_p2", "operator_budget"), action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--order", type=int, default=ParameterizedConfig.order)
    parser.add_argument("--output-dir", type=Path, default=Path("results/parameterized_operator_budget"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    base = ParameterizedConfig(order=args.order)
    if PRESETS[args.preset]:
        base = replace(base, **PRESETS[args.preset])
    methods = tuple(args.representation or ("fixed_p2", "operator_budget"))
    seeds = tuple(args.seed or (20260918, 20260919, 20260920))
    records: list[dict[str, object]] = []
    for seed in seeds:
        for method in methods:
            records.append(
                run_one(replace(base, representation=method, seed=seed), verbose=args.verbose)
            )
    summary = summarize(records)
    payload = {"records": records, "summary": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_summary(summary, args.output_dir / "parameterized_comparison.pdf")
    print(json.dumps({"summary": summary}, indent=2))
    return payload


if __name__ == "__main__":
    main()
