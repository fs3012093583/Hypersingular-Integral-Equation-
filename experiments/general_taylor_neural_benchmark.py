"""Matched neural and Chebyshev benchmarks for general Taylor finite parts.

The manufactured equation is

    u(t) + lambda (1-t**2)**(m-1) I_m[u](t) + mu*u(t)**3 = f(t),

on ``(-1, 1)``.  ``f`` is always manufactured with the same high-precision
Taylor-subtraction discretization that is used for training and validation.
Solution errors are reported on the closed interval ``[-1, 1]``; equation
residuals are reported only on the recorded strict interior interval.

The conventional Chebyshev baseline is intentionally linear-only (``mu=0``).
For nonlinear cases this script writes an explicit ``skipped`` record rather
than presenting an unmatched nonlinear solver as a spectral baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

# ``python experiments/general_taylor_neural_benchmark.py`` places only the
# experiments directory on sys.path; make the documented direct CLI invocation
# behave the same as ``python -m experiments.general_taylor_neural_benchmark``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conventional_solvers.taylor_spectral_collocation import (
    SUPPORTED_FAMILIES,
    SpectralConfig,
    exact_solution,
    run as run_spectral,
)
from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule as typed_gauss_legendre_rule
from conventional_solvers.taylor_spectral_collocation import manufactured_rhs
from neural_network_solvers.general_taylor_finite_part import taylor_finite_part_operator


Method = Literal["neural", "spectral", "both"]
ThresholdMode = Literal["balanced", "fixed", "diagonal"]


@dataclass(frozen=True)
class NeuralConfig:
    order: int = 2
    family: str = "exp"
    coupling: float = 0.05
    nonlinear_coefficient: float = 0.0
    seed: int = 20260918
    steps: int = 300
    lbfgs_steps: int = 0
    learning_rate: float = 2.0e-3
    width: int = 24
    hidden_layers: int = 2
    collocation_points: int = 24
    training_quadrature_points: int = 64
    rhs_quadrature_points: int = 256
    validation_points: int = 161
    validation_quadrature_points: int = 256
    collocation_limit: float = 0.95
    validation_limit: float = 0.98
    continuation_order: int = 2
    threshold_mode: ThresholdMode = "balanced"
    near_diagonal_threshold: float | None = None
    threshold_multiplier: float = 1.0
    rhs_continuation_order: int = 3
    rhs_threshold_multiplier: float = 1.0
    dtype: str = "float64"


PRESETS: dict[str, dict[str, int | float]] = {
    # A reproducible seconds-scale sanity check, suitable for CI and laptops.
    "smoke": {
        "steps": 3, "lbfgs_steps": 0,
        "width": 8, "hidden_layers": 1, "collocation_points": 8,
        "training_quadrature_points": 16, "rhs_quadrature_points": 48,
        "validation_points": 33, "validation_quadrature_points": 64,
        "spectral_degree": 8, "spectral_collocation_points": 12,
        "spectral_quadrature_points": 48,
    },
    # Kept finite on CPU while being materially larger than smoke.  For a full
    # paper sweep specify several --family/--order/--seed values explicitly.
    "paper": {
        "steps": 600, "lbfgs_steps": 60,
        "width": 32, "hidden_layers": 3, "collocation_points": 48,
        "training_quadrature_points": 64, "rhs_quadrature_points": 192,
        "validation_points": 161, "validation_quadrature_points": 192,
        "spectral_degree": 20, "spectral_collocation_points": 28,
        "spectral_quadrature_points": 160,
    },
}


class TanhNetwork(nn.Module):
    def __init__(self, width: int, hidden_layers: int) -> None:
        super().__init__()
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive")
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        return self.layers(points)


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _interior_chebyshev_points(count: int, limit: float, dtype: torch.dtype) -> Tensor:
    k = torch.arange(count, dtype=dtype)
    return (limit * torch.cos(torch.pi * (k + 0.5) / count)).reshape(-1, 1)


def _equation_residual(
    model: nn.Module,
    points: Tensor,
    rhs: Tensor,
    nodes: Tensor,
    weights: Tensor,
    config: NeuralConfig,
) -> Tensor:
    value = model(points)
    finite_part = taylor_finite_part_operator(
        model, points, nodes, weights, order=config.order,
        continuation_order=config.continuation_order, threshold_mode=config.threshold_mode,
        near_diagonal_threshold=config.near_diagonal_threshold,
        threshold_multiplier=config.threshold_multiplier,
    )
    return (
        value
        + config.coupling * (1.0 - points.square()).pow(config.order - 1) * finite_part
        + config.nonlinear_coefficient * value.pow(3)
        - rhs
    )


def run_neural(config: NeuralConfig, *, verbose: bool = False) -> dict[str, object]:
    """Train one neural solve and return metrics without writing any files."""
    if config.order not in (2, 4, 6):
        raise ValueError("benchmark orders are restricted to 2, 4, and 6")
    if config.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported family: {config.family}")
    if config.steps < 1 or config.collocation_points < 2:
        raise ValueError("steps must be positive and collocation_points must be at least 2")
    if config.threshold_mode == "fixed" and (config.near_diagonal_threshold is None or config.near_diagonal_threshold <= 0):
        raise ValueError("fixed threshold mode requires --near-diagonal-threshold > 0")
    dtype = _dtype(config.dtype)
    set_seed(config.seed)
    started = time.perf_counter()
    model = TanhNetwork(config.width, config.hidden_layers).to(dtype=dtype)
    points = _interior_chebyshev_points(config.collocation_points, config.collocation_limit, dtype)
    train_nodes, train_weights = typed_gauss_legendre_rule(config.training_quadrature_points, dtype)
    rhs_nodes, rhs_weights = typed_gauss_legendre_rule(config.rhs_quadrature_points, dtype)
    rhs = manufactured_rhs(
        config.family, points, order=config.order, coupling=config.coupling,
        nonlinear_coefficient=config.nonlinear_coefficient, quadrature_nodes=rhs_nodes,
        quadrature_weights=rhs_weights,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
        threshold_multiplier=config.rhs_threshold_multiplier,
    ).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[float] = []
    training_started = time.perf_counter()
    model.train()
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        residual = _equation_residual(model, points, rhs, train_nodes, train_weights, config)
        loss = torch.mean(residual.square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if verbose and (step == 0 or step + 1 == config.steps or (step + 1) % 100 == 0):
            print(f"neural m={config.order} family={config.family} step={step + 1} loss={history[-1]:.4e}")
    if config.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11 if dtype == torch.float64 else 1.0e-7,
            tolerance_change=1.0e-13 if dtype == torch.float64 else 1.0e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            current_residual = _equation_residual(
                model, points, rhs, train_nodes, train_weights, config
            )
            current_loss = torch.mean(current_residual.square())
            if not torch.isfinite(current_loss):
                raise FloatingPointError("non-finite loss during L-BFGS")
            current_loss.backward()
            return current_loss

        optimizer_lbfgs.step(closure)
        with torch.enable_grad():
            post_lbfgs = torch.mean(
                _equation_residual(
                    model, points, rhs, train_nodes, train_weights, config
                ).square()
            )
        history.append(float(post_lbfgs.detach()))
    training_seconds = time.perf_counter() - training_started

    # Closed-interval error is valid for u, while residuals intentionally use
    # only strict interior points and state their interval in the JSON record.
    closed = torch.linspace(-1.0, 1.0, config.validation_points, dtype=dtype).reshape(-1, 1)
    with torch.no_grad():
        solution_error = model(closed) - exact_solution(config.family, closed)
    validation = _interior_chebyshev_points(config.validation_points, config.validation_limit, dtype)
    validation_nodes, validation_weights = typed_gauss_legendre_rule(config.validation_quadrature_points, dtype)
    validation_rhs = manufactured_rhs(
        config.family, validation, order=config.order, coupling=config.coupling,
        nonlinear_coefficient=config.nonlinear_coefficient, quadrature_nodes=validation_nodes,
        quadrature_weights=validation_weights,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
        threshold_multiplier=config.rhs_threshold_multiplier,
    )
    method_validation_residual = _equation_residual(
        model, validation, validation_rhs, validation_nodes, validation_weights, config
    )
    canonical_config = replace(
        config,
        continuation_order=config.rhs_continuation_order,
        threshold_mode="balanced",
        near_diagonal_threshold=None,
        threshold_multiplier=config.rhs_threshold_multiplier,
    )
    canonical_validation_residual = _equation_residual(
        model,
        validation,
        validation_rhs,
        validation_nodes,
        validation_weights,
        canonical_config,
    )
    return {
        "method": "neural_taylor_pinn",
        "status": "ok",
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_training_loss": history[-1],
        "loss_history": history,
        "solution_rmse_closed_interval": float(torch.sqrt(torch.mean(solution_error.square()))),
        "solution_max_abs_error_closed_interval": float(torch.max(torch.abs(solution_error))),
        "validation_residual_rmse": float(
            torch.sqrt(torch.mean(canonical_validation_residual.detach().square()))
        ),
        "method_representation_validation_residual_rmse": float(
            torch.sqrt(torch.mean(method_validation_residual.detach().square()))
        ),
        "residual_interval": [-config.validation_limit, config.validation_limit],
        "training_seconds": training_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
    }


def _failure_record(method: str, config: dict[str, object], error: Exception, *, skipped: bool = False) -> dict[str, object]:
    return {
        "method": method, "status": "skipped" if skipped else "failed", "config": config,
        "reason": f"{type(error).__name__}: {error}",
    }


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Mean/std over successful seeds, with failure rate kept explicit."""
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for record in records:
        config = record["config"]
        assert isinstance(config, dict)
        key = (str(record["method"]), int(config["order"]), str(config["family"]))
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, object]] = []
    for (method, order, family), group in sorted(grouped.items()):
        succeeded = [row for row in group if row["status"] == "ok"]
        row: dict[str, object] = {
            "method": method, "order": order, "family": family, "runs": len(group),
            "successes": len(succeeded), "failure_rate": 1.0 - len(succeeded) / len(group),
        }
        for metric in ("solution_rmse_closed_interval", "solution_max_abs_error_closed_interval", "validation_residual_rmse", "training_seconds", "end_to_end_seconds"):
            values = [float(item[metric]) for item in succeeded if metric in item]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values)) if values else None
        rows.append(row)
    return rows


def plot_summary(summary: list[dict[str, object]], path: Path) -> None:
    successful = [row for row in summary if row["solution_rmse_closed_interval_mean"] is not None]
    if not successful:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for metric, axis, title in (
        ("solution_rmse_closed_interval_mean", axes[0], "Closed-interval solution RMSE"),
        ("validation_residual_rmse_mean", axes[1], "Interior validation residual RMSE"),
    ):
        for method in sorted({str(row["method"]) for row in successful}):
            rows = [row for row in successful if row["method"] == method]
            for family in sorted({str(row["family"]) for row in rows}):
                data = sorted((row for row in rows if row["family"] == family), key=lambda row: int(row["order"]))
                axis.semilogy([row["order"] for row in data], [row[metric] for row in data], "o-", label=f"{method}, {family}")
        axis.set_xlabel("finite-part order m")
        axis.set_ylabel("RMSE")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(
    neural_template: NeuralConfig,
    *,
    orders: list[int], families: list[str], seeds: list[int], method: Method,
    spectral_degree: int, spectral_collocation_points: int, spectral_quadrature_points: int,
    output_dir: Path | None = None, verbose: bool = False,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for family in families:
        for order in orders:
            for seed in seeds:
                neural_config = replace(neural_template, family=family, order=order, seed=seed)
                if method in ("neural", "both"):
                    try:
                        record = run_neural(neural_config, verbose=verbose)
                    except Exception as error:  # retain failed seeds for the failure rate
                        record = _failure_record("neural_taylor_pinn", asdict(neural_config), error)
                    records.append(record)
                    if output_dir:
                        _write_run(output_dir, len(records), record)
            # The spectral solve is deterministic and has no random seed.  Run
            # it once per (family, order), rather than duplicating the same
            # record for every neural seed and understating its variance.
            if method in ("spectral", "both"):
                spectral_config = SpectralConfig(
                    order=order, family=family, coupling=neural_template.coupling,
                    nonlinear_coefficient=neural_template.nonlinear_coefficient,
                    degree=spectral_degree, collocation_points=spectral_collocation_points,
                    quadrature_points=spectral_quadrature_points,
                    rhs_quadrature_points=neural_template.rhs_quadrature_points,
                    validation_points=neural_template.validation_points,
                    validation_quadrature_points=neural_template.validation_quadrature_points,
                    validation_limit=neural_template.validation_limit,
                    continuation_order=neural_template.continuation_order,
                    threshold_mode=neural_template.threshold_mode,
                    near_diagonal_threshold=neural_template.near_diagonal_threshold,
                    threshold_multiplier=neural_template.threshold_multiplier,
                    rhs_continuation_order=neural_template.rhs_continuation_order,
                    rhs_threshold_multiplier=neural_template.rhs_threshold_multiplier,
                    dtype=neural_template.dtype,
                )
                if neural_template.nonlinear_coefficient != 0.0:
                    record = _failure_record("chebyshev_taylor_collocation", asdict(spectral_config), NotImplementedError("linear baseline supports mu=0 only"), skipped=True)
                else:
                    try:
                        record = run_spectral(spectral_config)
                    except Exception as error:
                        record = _failure_record("chebyshev_taylor_collocation", asdict(spectral_config), error)
                records.append(record)
                if output_dir:
                    _write_run(output_dir, len(records), record)
    summary = summarize(records)
    payload = {"equation": "u + lambda*(1-t^2)^(m-1)*I_m[u] + mu*u^3 = f", "records": records, "summary": summary}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_summary(summary, output_dir / "method_order_comparison.png")
    return payload


def _write_run(output_dir: Path, index: int, record: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"run_{index:03d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--method", choices=("neural", "spectral", "both"), default="both")
    parser.add_argument("--order", dest="orders", type=int, action="append")
    parser.add_argument("--family", dest="families", choices=SUPPORTED_FAMILIES, action="append")
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--lambda-value", dest="coupling", type=float, default=0.05)
    parser.add_argument("--mu", dest="nonlinear_coefficient", type=float, default=0.0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lbfgs-steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--collocation-points", type=int)
    parser.add_argument("--training-quadrature-points", type=int)
    parser.add_argument("--rhs-quadrature-points", type=int)
    parser.add_argument("--validation-points", type=int)
    parser.add_argument("--validation-quadrature-points", type=int)
    parser.add_argument("--p", "--continuation-order", dest="continuation_order", type=int, default=2)
    parser.add_argument("--threshold-mode", choices=("balanced", "fixed", "diagonal"), default="balanced")
    parser.add_argument("--near-diagonal-threshold", type=float)
    parser.add_argument("--threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--rhs-continuation-order", type=int, default=3)
    parser.add_argument("--rhs-threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--spectral-degree", type=int)
    parser.add_argument("--spectral-collocation-points", type=int)
    parser.add_argument("--spectral-quadrature-points", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs/general_taylor_neural_benchmark"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    values = asdict(NeuralConfig())
    for field in (
        "steps", "lbfgs_steps", "width", "hidden_layers", "collocation_points",
        "training_quadrature_points", "rhs_quadrature_points",
        "validation_points", "validation_quadrature_points",
    ):
        values[field] = getattr(args, field) if getattr(args, field) is not None else preset[field]
    values.update({
        "coupling": args.coupling,
        "nonlinear_coefficient": args.nonlinear_coefficient,
        "continuation_order": args.continuation_order,
        "threshold_mode": args.threshold_mode,
        "near_diagonal_threshold": args.near_diagonal_threshold,
        "threshold_multiplier": args.threshold_multiplier,
        "rhs_continuation_order": args.rhs_continuation_order,
        "rhs_threshold_multiplier": args.rhs_threshold_multiplier,
        "dtype": args.dtype,
    })
    config = NeuralConfig(**values)
    payload = run_benchmark(
        config, orders=args.orders or [2, 4, 6], families=args.families or ["exp"], seeds=args.seeds or [config.seed], method=args.method,
        spectral_degree=args.spectral_degree or int(preset["spectral_degree"]),
        spectral_collocation_points=args.spectral_collocation_points or int(preset["spectral_collocation_points"]),
        spectral_quadrature_points=args.spectral_quadrature_points or int(preset["spectral_quadrature_points"]),
        output_dir=args.output_dir, verbose=args.verbose,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
