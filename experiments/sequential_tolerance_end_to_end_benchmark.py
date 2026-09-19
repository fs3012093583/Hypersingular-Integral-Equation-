"""End-to-end comparison of fixed and error-controlled Taylor representations.

The experiment trains the same tanh network on the manufactured equation

    u(t) + lambda (1-t**2)**(m-1) I_m[u](t) + mu*u(t)**3 = f(t)

with three representations of the regularized finite-part quotient:

* fixed Taylor continuation degree p=2;
* fixed Taylor continuation degree p=3;
* diagonal-limit replacement (p=0);
* pointwise sequential tolerance selection with p <= 3;
* quadrature-weighted operator-budget selection with p <= 3.

All methods share the initial weights, collocation points, quadrature rules,
right-hand side, optimizer, and canonical validation operator.  The sequential
method refreshes an empirical derivative-magnitude envelope during training;
the refresh cost is included in the reported training time.  That envelope is
an observed network-family control, not a rigorous interval certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from conventional_solvers.taylor_spectral_collocation import (
    SUPPORTED_FAMILIES,
    exact_solution,
    gauss_legendre_rule,
    manufactured_rhs,
)
from experiments.general_taylor_neural_benchmark import (
    TanhNetwork,
    _interior_chebyshev_points,
    set_seed,
)
from neural_network_solvers.general_taylor_finite_part import (
    OperatorBudgetTaylorDiagnostics,
    SequentialTaylorDiagnostics,
    lazy_taylor_finite_part_operator,
    operator_budget_taylor_finite_part_operator,
    scaled_derivative_error_model,
    sequential_tolerance_taylor_finite_part_operator,
    taylor_finite_part_operator,
    value_and_derivatives,
)


Representation = Literal[
    "fixed_p2",
    "lazy_fixed_p2",
    "fixed_p3",
    "limit_p0",
    "sequential",
    "operator_budget",
]


@dataclass(frozen=True)
class EndToEndConfig:
    representation: Representation = "sequential"
    order: int = 2
    family: str = "exp"
    coupling: float = 0.05
    nonlinear_coefficient: float = 0.0
    seed: int = 20260918
    steps: int = 600
    lbfgs_steps: int = 60
    learning_rate: float = 2.0e-3
    width: int = 32
    hidden_layers: int = 3
    collocation_points: int = 48
    training_quadrature_points: int = 64
    rhs_quadrature_points: int = 192
    validation_points: int = 161
    validation_quadrature_points: int = 192
    collocation_limit: float = 0.95
    validation_limit: float = 0.98
    max_continuation_order: int = 3
    target_error: float = 1.0e-3
    target_residual_error: float = 1.0e-4
    derivative_envelope_safety: float = 4.0
    envelope_refresh_interval: int = 100
    envelope_calibration_points: int = 65
    selector_batch_size: int = 48
    derivative_metrics_path: str = "results/neural_high_order_derivatives/metrics.json"
    rhs_continuation_order: int = 3
    dtype: str = "float64"
    device: str = "cpu"
    timing_repeats: int = 3


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "steps": 3,
        "lbfgs_steps": 1,
        "width": 8,
        "hidden_layers": 1,
        "collocation_points": 6,
        "training_quadrature_points": 12,
        "rhs_quadrature_points": 32,
        "validation_points": 25,
        "validation_quadrature_points": 32,
        "envelope_refresh_interval": 2,
        "envelope_calibration_points": 9,
        "timing_repeats": 1,
    },
    "paper": {
        "steps": 600,
        "lbfgs_steps": 60,
        "width": 32,
        "hidden_layers": 3,
        "collocation_points": 48,
        "training_quadrature_points": 64,
        "rhs_quadrature_points": 192,
        "validation_points": 161,
        "validation_quadrature_points": 192,
        "envelope_refresh_interval": 100,
        "envelope_calibration_points": 65,
        "timing_repeats": 3,
    },
}


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_derivative_error_model(config: EndToEndConfig):
    path = Path(config.derivative_metrics_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["eta_scaled_envelope"][config.dtype]
    envelope = {int(order): float(value) for order, value in raw.items()}
    required_order = config.order + config.max_continuation_order
    if required_order not in envelope:
        raise ValueError(
            f"derivative error envelope stops before required order {required_order}"
        )
    return scaled_derivative_error_model(envelope), envelope


class OnlineDerivativeEnvelope:
    """Monotone empirical derivative envelope refreshed during training."""

    def __init__(
        self,
        model: nn.Module,
        *,
        highest_order: int,
        dtype: torch.dtype,
        calibration_points: int,
        safety_factor: float,
        device: torch.device | str = "cpu",
    ) -> None:
        if highest_order < 1:
            raise ValueError("highest_order must be positive")
        if calibration_points < 3:
            raise ValueError("calibration_points must be at least three")
        if safety_factor <= 1.0:
            raise ValueError("safety_factor must exceed one")
        self.model = model
        self.highest_order = highest_order
        self.dtype = dtype
        self.device = torch.device(device)
        self.safety_factor = safety_factor
        self.points = torch.linspace(
            -1.0,
            1.0,
            calibration_points,
            dtype=dtype,
            device=self.device,
        ).reshape(-1, 1)
        self.bounds: dict[int, float] = {}
        self.refresh_records: list[dict[str, object]] = []
        self.total_refresh_seconds = 0.0

    def refresh(self, stage: str, step: int) -> None:
        started = time.perf_counter()
        points = self.points.detach().clone().requires_grad_(True)
        derivatives = value_and_derivatives(
            self.model, points, self.highest_order
        )
        observed = {
            order: float(torch.max(torch.abs(value.detach())))
            for order, value in enumerate(derivatives)
        }
        previous_violations = [
            order
            for order, magnitude in observed.items()
            if order in self.bounds and magnitude > self.bounds[order]
        ]
        for order, magnitude in observed.items():
            candidate = self.safety_factor * max(magnitude, 1.0e-15)
            self.bounds[order] = max(self.bounds.get(order, 0.0), candidate)
        elapsed = time.perf_counter() - started
        self.total_refresh_seconds += elapsed
        self.refresh_records.append(
            {
                "stage": stage,
                "step": step,
                "observed_max_abs_derivative": {
                    str(order): magnitude for order, magnitude in observed.items()
                },
                "active_bound": {
                    str(order): self.bounds[order] for order in sorted(self.bounds)
                },
                "violations_of_previous_bound": previous_violations,
                "refresh_seconds": elapsed,
            }
        )

    def __call__(self, t: Tensor, tau: Tensor, derivative_order: int) -> Tensor:
        if derivative_order not in self.bounds:
            raise KeyError(f"missing online derivative bound for order {derivative_order}")
        return torch.full_like(t + tau, self.bounds[derivative_order])


def _fixed_order(representation: Representation) -> int:
    if representation in ("fixed_p2", "lazy_fixed_p2"):
        return 2
    if representation == "fixed_p3":
        return 3
    raise ValueError(f"{representation} is not a fixed-order representation")


def _finite_part(
    model: nn.Module,
    points: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: EndToEndConfig,
    *,
    derivative_error_model=None,
    derivative_bound=None,
    return_diagnostics: bool = False,
) -> Tensor | tuple[
    Tensor, SequentialTaylorDiagnostics | OperatorBudgetTaylorDiagnostics
]:
    if config.representation in ("limit_p0", "sequential", "operator_budget"):
        batch_size = config.selector_batch_size
        if batch_size <= 0:
            raise ValueError("selector_batch_size must be positive")
        chunks = list(torch.split(points, batch_size, dim=0))
        values: list[Tensor] = []
        diagnostics_list: list[
            SequentialTaylorDiagnostics | OperatorBudgetTaylorDiagnostics
        ] = []
        for chunk in chunks:
            if config.representation == "operator_budget":
                if abs(config.coupling) <= 0.0:
                    raise ValueError(
                        "operator-budget selection requires nonzero coupling"
                    )
                # The manufactured equation weight satisfies
                # (1-t**2)**(m-1) <= 1.  Using its global maximum keeps the
                # finite-part operator itself accurate near the endpoints and
                # simultaneously guarantees the requested residual budget.
                operator_budget = torch.full_like(
                    chunk,
                    config.target_residual_error / abs(config.coupling),
                )
                result = operator_budget_taylor_finite_part_operator(
                    model,
                    chunk,
                    nodes,
                    weights,
                    order=config.order,
                    target_operator_error=operator_budget,
                    max_continuation_order=config.max_continuation_order,
                    trust_region_multiplier=None,
                    derivative_error_model=derivative_error_model,
                    derivative_bound=derivative_bound,
                    return_diagnostics=return_diagnostics,
                )
            else:
                maximum_order = (
                    0
                    if config.representation == "limit_p0"
                    else config.max_continuation_order
                )
                result = sequential_tolerance_taylor_finite_part_operator(
                    model,
                    chunk,
                    nodes,
                    weights,
                    order=config.order,
                    target_error=config.target_error,
                    max_continuation_order=maximum_order,
                    trust_region_multiplier=None,
                    derivative_error_model=derivative_error_model,
                    derivative_bound=derivative_bound,
                    return_diagnostics=return_diagnostics,
                )
            if return_diagnostics:
                value, diagnostics = result
                values.append(value)
                diagnostics_list.append(diagnostics)
            else:
                assert isinstance(result, Tensor)
                values.append(result)
        combined = torch.cat(values, dim=0)
        if not return_diagnostics:
            return combined
        common = {
            "direct_error_estimate": torch.cat(
                [item.direct_error_estimate for item in diagnostics_list], dim=0
            ),
            "local_error_estimates": torch.cat(
                [item.local_error_estimates for item in diagnostics_list], dim=0
            ),
            "selected_error_estimate": torch.cat(
                [item.selected_error_estimate for item in diagnostics_list], dim=0
            ),
            "selected_order": torch.cat(
                [item.selected_order for item in diagnostics_list], dim=0
            ),
            "highest_derivative_order": max(
                item.highest_derivative_order for item in diagnostics_list
            ),
            "generated_derivative_count": sum(
                item.generated_derivative_count for item in diagnostics_list
            ),
            "derivative_tensor_bytes": sum(
                item.derivative_tensor_bytes for item in diagnostics_list
            ),
            "active_row_counts": tuple(
                sum(
                    item.active_row_counts[stage]
                    if stage < len(item.active_row_counts)
                    else 0
                    for item in diagnostics_list
                )
                for stage in range(
                    max(
                        (
                            len(item.active_row_counts)
                            for item in diagnostics_list
                        ),
                        default=0,
                    )
                )
            ),
        }
        if config.representation == "operator_budget":
            weighted_diagnostics = [
                item
                for item in diagnostics_list
                if isinstance(item, OperatorBudgetTaylorDiagnostics)
            ]
            if len(weighted_diagnostics) != len(diagnostics_list):
                raise TypeError("operator-budget diagnostics were not returned")
            diagnostics = OperatorBudgetTaylorDiagnostics(
                **common,
                direct_operator_error_estimate=torch.cat(
                    [
                        item.direct_operator_error_estimate
                        for item in weighted_diagnostics
                    ],
                    dim=0,
                ),
                selected_operator_error_estimate=torch.cat(
                    [
                        item.selected_operator_error_estimate
                        for item in weighted_diagnostics
                    ],
                    dim=0,
                ),
                target_operator_error=torch.cat(
                    [item.target_operator_error for item in weighted_diagnostics],
                    dim=0,
                ),
            )
            return combined, diagnostics
        diagnostics = SequentialTaylorDiagnostics(
            **common,
        )
        return combined, diagnostics
    if return_diagnostics:
        raise ValueError("fixed representations do not return sequential diagnostics")
    fixed_operator = (
        lazy_taylor_finite_part_operator
        if config.representation == "lazy_fixed_p2"
        else taylor_finite_part_operator
    )
    return fixed_operator(
        model,
        points,
        nodes,
        weights,
        order=config.order,
        continuation_order=_fixed_order(config.representation),
        threshold_mode="balanced",
    )


def _residual(
    model: nn.Module,
    points: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: EndToEndConfig,
    *,
    derivative_error_model=None,
    derivative_bound=None,
) -> Tensor:
    value = model(points)
    finite_part = _finite_part(
        model,
        points,
        nodes,
        weights,
        config,
        derivative_error_model=derivative_error_model,
        derivative_bound=derivative_bound,
    )
    assert isinstance(finite_part, Tensor)
    return (
        value
        + config.coupling
        * (1.0 - points.square()).pow(config.order - 1)
        * finite_part
        + config.nonlinear_coefficient * value.pow(3)
        - rhs
    )


def _canonical_residual(
    model: nn.Module,
    points: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: EndToEndConfig,
) -> Tensor:
    value = model(points)
    finite_part = taylor_finite_part_operator(
        model,
        points,
        nodes,
        weights,
        order=config.order,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
    )
    return (
        value
        + config.coupling
        * (1.0 - points.square()).pow(config.order - 1)
        * finite_part
        + config.nonlinear_coefficient * value.pow(3)
        - rhs
    )


def _saved_tensor_payload_bytes(loss_action: Callable[[], Tensor]) -> int:
    saved_bytes = 0

    def pack(tensor: Tensor) -> Tensor:
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_action()
        loss.backward()
    return saved_bytes


def _post_training_cost(
    model: nn.Module,
    points: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: EndToEndConfig,
    *,
    derivative_error_model=None,
    derivative_bound=None,
) -> dict[str, float | int]:
    device = next(model.parameters()).device
    timings: list[float] = []
    payloads: list[int] = []
    for repeat in range(config.timing_repeats + 1):
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        payload = _saved_tensor_payload_bytes(
            lambda: torch.mean(
                _residual(
                    model,
                    points,
                    rhs,
                    nodes,
                    weights,
                    config,
                    derivative_error_model=derivative_error_model,
                    derivative_bound=derivative_bound,
                ).square()
            )
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if repeat > 0:
            timings.append(elapsed)
            payloads.append(payload)
    return {
        "median_residual_forward_backward_seconds": statistics.median(timings),
        "median_saved_tensor_payload_bytes": int(statistics.median(payloads)),
    }


def _selection_diagnostics(
    model: nn.Module,
    points: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: EndToEndConfig,
    *,
    derivative_error_model,
    derivative_bound,
) -> dict[str, object] | None:
    if config.representation not in ("limit_p0", "sequential", "operator_budget"):
        return None
    _, diagnostics = _finite_part(
        model,
        points,
        nodes,
        weights,
        config,
        derivative_error_model=derivative_error_model,
        derivative_bound=derivative_bound,
        return_diagnostics=True,
    )
    counts = Counter(
        int(item) for item in diagnostics.selected_order.reshape(-1).tolist()
    )
    total = sum(counts.values())
    local_total = total - counts[-1]
    summary: dict[str, object] = {
        "selected_counts": {str(key): value for key, value in sorted(counts.items())},
        "direct_fraction": counts[-1] / total if total else None,
        "mean_selected_local_order": (
            sum(order * count for order, count in counts.items() if order >= 0)
            / local_total
            if local_total
            else None
        ),
        "highest_derivative_order": diagnostics.highest_derivative_order,
        "maximum_selected_error_estimate": float(
            torch.max(diagnostics.selected_error_estimate)
        ),
        "active_row_counts_by_local_order": list(diagnostics.active_row_counts),
    }
    if isinstance(diagnostics, OperatorBudgetTaylorDiagnostics):
        achieved = (
            diagnostics.selected_operator_error_estimate
            <= diagnostics.target_operator_error
        )
        summary.update(
            {
                "maximum_selected_operator_error_estimate": float(
                    torch.max(diagnostics.selected_operator_error_estimate)
                ),
                "maximum_target_operator_error": float(
                    torch.max(diagnostics.target_operator_error)
                ),
                "operator_budget_satisfaction_fraction": float(
                    torch.mean(achieved.to(torch.float64))
                ),
                "maximum_operator_budget_ratio": float(
                    torch.max(
                        diagnostics.selected_operator_error_estimate
                        / diagnostics.target_operator_error
                    )
                ),
            }
        )
    return summary


def run_one(config: EndToEndConfig, *, verbose: bool = False) -> dict[str, object]:
    if config.order not in (2, 4, 6):
        raise ValueError("end-to-end comparison is restricted to orders 2, 4, and 6")
    if config.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported family: {config.family}")
    if config.representation not in (
        "fixed_p2",
        "lazy_fixed_p2",
        "fixed_p3",
        "limit_p0",
        "sequential",
        "operator_budget",
    ):
        raise ValueError(f"unsupported representation: {config.representation}")
    if config.steps < 1 or config.lbfgs_steps < 0:
        raise ValueError("steps must be positive and lbfgs_steps non-negative")
    if config.envelope_refresh_interval < 1:
        raise ValueError("envelope_refresh_interval must be positive")
    if config.target_residual_error <= 0.0:
        raise ValueError("target_residual_error must be positive")

    dtype = _dtype(config.dtype)
    device = _device(config.device)
    set_seed(config.seed)
    _synchronize(device)
    total_started = time.perf_counter()
    model = TanhNetwork(config.width, config.hidden_layers).to(
        dtype=dtype, device=device
    )
    initial_fingerprint = _model_fingerprint(model)
    points = _interior_chebyshev_points(
        config.collocation_points, config.collocation_limit, dtype
    ).to(device=device)
    train_nodes, train_weights = gauss_legendre_rule(
        config.training_quadrature_points, dtype
    )
    train_nodes = train_nodes.to(device=device)
    train_weights = train_weights.to(device=device)
    rhs_nodes, rhs_weights = gauss_legendre_rule(
        config.rhs_quadrature_points, dtype
    )
    rhs_nodes = rhs_nodes.to(device=device)
    rhs_weights = rhs_weights.to(device=device)
    rhs = manufactured_rhs(
        config.family,
        points,
        order=config.order,
        coupling=config.coupling,
        nonlinear_coefficient=config.nonlinear_coefficient,
        quadrature_nodes=rhs_nodes,
        quadrature_weights=rhs_weights,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
    ).detach()

    derivative_error_model = None
    derivative_error_envelope = None
    online_envelope = None
    _synchronize(device)
    method_setup_started = time.perf_counter()
    if config.representation in ("limit_p0", "sequential", "operator_budget"):
        derivative_error_model, derivative_error_envelope = (
            _load_derivative_error_model(config)
        )
        highest_local_order = (
            0
            if config.representation == "limit_p0"
            else config.max_continuation_order
        )
        online_envelope = OnlineDerivativeEnvelope(
            model,
            highest_order=config.order + highest_local_order + 1,
            dtype=dtype,
            calibration_points=config.envelope_calibration_points,
            safety_factor=config.derivative_envelope_safety,
            device=device,
        )
        online_envelope.refresh("initial", 0)

    _synchronize(device)
    method_setup_seconds = time.perf_counter() - method_setup_started
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    _synchronize(device)
    training_started = time.perf_counter()
    model.train()
    for step in range(config.steps):
        if (
            online_envelope is not None
            and step > 0
            and step % config.envelope_refresh_interval == 0
        ):
            online_envelope.refresh("adam", step)
        optimizer.zero_grad(set_to_none=True)
        residual = _residual(
            model,
            points,
            rhs,
            train_nodes,
            train_weights,
            config,
            derivative_error_model=derivative_error_model,
            derivative_bound=online_envelope,
        )
        loss = torch.mean(residual.square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at Adam step {step + 1}")
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == config.steps or (step + 1) % 100 == 0:
            history.append(
                {"stage": "adam", "step": step + 1, "loss": float(loss.detach())}
            )
            if verbose:
                print(
                    f"{config.representation} m={config.order} seed={config.seed} "
                    f"Adam {step + 1}/{config.steps} loss={float(loss.detach()):.4e}",
                    flush=True,
                )

    if online_envelope is not None:
        online_envelope.refresh("pre_lbfgs", config.steps)
    if config.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11 if dtype == torch.float64 else 1.0e-7,
            tolerance_change=1.0e-13 if dtype == torch.float64 else 1.0e-9,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer_lbfgs.zero_grad(set_to_none=True)
            current_residual = _residual(
                model,
                points,
                rhs,
                train_nodes,
                train_weights,
                config,
                derivative_error_model=derivative_error_model,
                derivative_bound=online_envelope,
            )
            current_loss = torch.mean(current_residual.square())
            if not torch.isfinite(current_loss):
                raise FloatingPointError("non-finite loss during L-BFGS")
            current_loss.backward()
            return current_loss

        optimizer_lbfgs.step(closure)
        final_train_loss = torch.mean(
            _residual(
                model,
                points,
                rhs,
                train_nodes,
                train_weights,
                config,
                derivative_error_model=derivative_error_model,
                derivative_bound=online_envelope,
            ).square()
        )
        history.append(
            {
                "stage": "lbfgs",
                "step": closure_calls,
                "loss": float(final_train_loss.detach()),
            }
        )
    else:
        closure_calls = 0
        final_train_loss = torch.mean(
            _residual(
                model,
                points,
                rhs,
                train_nodes,
                train_weights,
                config,
                derivative_error_model=derivative_error_model,
                derivative_bound=online_envelope,
            ).square()
        )
    if online_envelope is not None:
        online_envelope.refresh("final", config.steps)
    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    solve_seconds = time.perf_counter() - total_started

    closed = torch.linspace(
        -1.0,
        1.0,
        config.validation_points,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    with torch.no_grad():
        solution_error = model(closed) - exact_solution(config.family, closed)
    validation = _interior_chebyshev_points(
        config.validation_points, config.validation_limit, dtype
    ).to(device=device)
    validation_nodes, validation_weights = gauss_legendre_rule(
        config.validation_quadrature_points, dtype
    )
    validation_nodes = validation_nodes.to(device=device)
    validation_weights = validation_weights.to(device=device)
    validation_rhs = manufactured_rhs(
        config.family,
        validation,
        order=config.order,
        coupling=config.coupling,
        nonlinear_coefficient=config.nonlinear_coefficient,
        quadrature_nodes=validation_nodes,
        quadrature_weights=validation_weights,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
    )
    canonical_residual = _canonical_residual(
        model,
        validation,
        validation_rhs,
        validation_nodes,
        validation_weights,
        config,
    )
    cost = _post_training_cost(
        model,
        points,
        rhs,
        train_nodes,
        train_weights,
        config,
        derivative_error_model=derivative_error_model,
        derivative_bound=online_envelope,
    )
    diagnostics = _selection_diagnostics(
        model,
        points,
        train_nodes,
        train_weights,
        config,
        derivative_error_model=derivative_error_model,
        derivative_bound=online_envelope,
    )
    _synchronize(device)
    return {
        "experiment": "sequential_tolerance_end_to_end",
        "method": config.representation,
        "status": "ok",
        "config": asdict(config),
        "initial_model_fingerprint": initial_fingerprint,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_training_loss": float(final_train_loss.detach()),
        "loss_history": history,
        "lbfgs_closure_calls": closure_calls,
        "solution_rmse_closed_interval": float(
            torch.sqrt(torch.mean(solution_error.square()))
        ),
        "solution_max_abs_error_closed_interval": float(
            torch.max(torch.abs(solution_error))
        ),
        "validation_residual_rmse": float(
            torch.sqrt(torch.mean(canonical_residual.detach().square()))
        ),
        "residual_interval": [-config.validation_limit, config.validation_limit],
        "training_seconds_including_envelope_refresh": training_seconds,
        "method_setup_seconds": method_setup_seconds,
        "method_setup_and_training_seconds": method_setup_seconds + training_seconds,
        "solve_seconds_before_validation": solve_seconds,
        "ad_calibration_source": (
            "precomputed empirical envelope; its original generation cost is not included"
            if online_envelope is not None else None
        ),
        "end_to_end_seconds": time.perf_counter() - total_started,
        "post_training_cost": cost,
        "selection_diagnostics": diagnostics,
        "derivative_error_scaled_envelope": derivative_error_envelope,
        "online_derivative_envelope_refreshes": (
            online_envelope.refresh_records if online_envelope is not None else None
        ),
        "online_derivative_envelope_refresh_seconds": (
            online_envelope.total_refresh_seconds
            if online_envelope is not None
            else 0.0
        ),
    }


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, object]]] = {}
    for record in records:
        config = record["config"]
        assert isinstance(config, dict)
        key = (
            str(record["method"]),
            int(config["order"]),
            str(config["family"]),
            str(config["dtype"]),
        )
        groups.setdefault(key, []).append(record)
    rows: list[dict[str, object]] = []
    for (method, order, family, dtype), group in sorted(groups.items()):
        succeeded = [record for record in group if record["status"] == "ok"]
        row: dict[str, object] = {
            "method": method,
            "order": order,
            "family": family,
            "dtype": dtype,
            "runs": len(group),
            "successes": len(succeeded),
            "failure_rate": 1.0 - len(succeeded) / len(group),
        }
        for metric in (
            "solution_rmse_closed_interval",
            "validation_residual_rmse",
            "training_seconds_including_envelope_refresh",
            "end_to_end_seconds",
        ):
            values = [float(record[metric]) for record in succeeded]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values)) if values else None
        cost_values = [
            int(record["post_training_cost"]["median_saved_tensor_payload_bytes"])
            for record in succeeded
        ]
        row["median_saved_tensor_payload_bytes_mean"] = (
            float(np.mean(cost_values)) if cost_values else None
        )
        rows.append(row)
    return rows


def _plot(summary: list[dict[str, object]], path: Path) -> None:
    successful = [row for row in summary if row["successes"]]
    if not successful:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    specifications = (
        ("solution_rmse_closed_interval_mean", "closed-interval solution RMSE", True),
        (
            "training_seconds_including_envelope_refresh_mean",
            "complete training time (s)",
            True,
        ),
        (
            "median_saved_tensor_payload_bytes_mean",
            "saved-tensor payload (bytes)",
            True,
        ),
    )
    styles = {
        "fixed_p2": ("o", "--"),
        "fixed_p3": ("s", ":"),
        "limit_p0": ("D", "-."),
        "sequential": ("^", "-"),
        "operator_budget": ("v", "-"),
    }
    for axis, (metric, label, logarithmic) in zip(axes, specifications):
        for method in (
            "fixed_p2",
            "fixed_p3",
            "limit_p0",
            "sequential",
            "operator_budget",
        ):
            selected = sorted(
                (row for row in successful if row["method"] == method),
                key=lambda row: int(row["order"]),
            )
            if not selected:
                continue
            marker, linestyle = styles[method]
            axis.plot(
                [row["order"] for row in selected],
                [row[metric] for row in selected],
                marker=marker,
                linestyle=linestyle,
                label=method,
            )
        axis.set_xlabel("kernel order m")
        axis.set_ylabel(label)
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    template: EndToEndConfig,
    *,
    representations: list[Representation],
    orders: list[int],
    families: list[str],
    seeds: list[int],
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    run_index = 0
    for family in families:
        for order in orders:
            for seed in seeds:
                for representation in representations:
                    run_index += 1
                    config = replace(
                        template,
                        representation=representation,
                        order=order,
                        family=family,
                        seed=seed,
                    )
                    try:
                        record = run_one(config, verbose=verbose)
                    except Exception as error:
                        record = {
                            "experiment": "sequential_tolerance_end_to_end",
                            "method": representation,
                            "status": "failed",
                            "config": asdict(config),
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    records.append(record)
                    (output_dir / f"run_{run_index:03d}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    summary = summarize(records)
    payload = {
        "experiment": "sequential_tolerance_end_to_end",
        "comparison_contract": {
            "matched": [
                "initial weights",
                "collocation points",
                "quadrature rules",
                "manufactured right-hand side",
                "Adam and L-BFGS settings",
                "canonical validation operator",
            ],
            "sequential_training_time_includes_online_envelope_refresh": True,
            "memory_metric": "autograd saved-tensor payload for one full residual forward-backward pass",
        },
        "records": records,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(summary, output_dir / "end_to_end_comparison.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument(
        "--representation",
        dest="representations",
        choices=(
            "fixed_p2",
            "fixed_p3",
            "limit_p0",
            "sequential",
            "operator_budget",
        ),
        action="append",
    )
    parser.add_argument("--order", dest="orders", type=int, action="append")
    parser.add_argument(
        "--family", dest="families", choices=SUPPORTED_FAMILIES, action="append"
    )
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:0",
    )
    parser.add_argument("--target-error", type=float)
    parser.add_argument("--target-residual-error", type=float)
    parser.add_argument("--max-p", dest="max_continuation_order", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lbfgs-steps", type=int)
    parser.add_argument("--envelope-refresh-interval", type=int)
    parser.add_argument("--envelope-safety", type=float, default=4.0)
    parser.add_argument("--selector-batch-size", type=int, default=48)
    parser.add_argument(
        "--derivative-metrics-path",
        type=Path,
        default=Path("results/neural_high_order_derivatives/metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sequential_tolerance_end_to_end"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    values = asdict(EndToEndConfig())
    for field in (
        "steps",
        "lbfgs_steps",
        "width",
        "hidden_layers",
        "collocation_points",
        "training_quadrature_points",
        "rhs_quadrature_points",
        "validation_points",
        "validation_quadrature_points",
        "envelope_refresh_interval",
        "envelope_calibration_points",
        "timing_repeats",
    ):
        command_line_value = getattr(args, field, None)
        values[field] = (
            command_line_value
            if command_line_value is not None
            else preset[field]
        )
    values.update(
        {
            "dtype": args.dtype,
            "device": args.device,
            "target_error": (
                args.target_error
                if args.target_error is not None
                else 1.0e-3
            ),
            "target_residual_error": (
                args.target_residual_error
                if args.target_residual_error is not None
                else EndToEndConfig.target_residual_error
            ),
            "max_continuation_order": (
                args.max_continuation_order
                if args.max_continuation_order is not None
                else EndToEndConfig.max_continuation_order
            ),
            "derivative_envelope_safety": args.envelope_safety,
            "selector_batch_size": args.selector_batch_size,
            "derivative_metrics_path": str(args.derivative_metrics_path),
        }
    )
    template = EndToEndConfig(**values)
    payload = run_benchmark(
        template,
        representations=args.representations
        or [
            "fixed_p2",
            "fixed_p3",
            "limit_p0",
            "sequential",
            "operator_budget",
        ],
        orders=args.orders or [2, 4, 6],
        families=args.families or ["exp"],
        seeds=args.seeds or [template.seed],
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
