"""Benchmark genuine sequential derivative generation and early stopping.

Two complementary parts are reported:

1. finite-part operator accuracy on the established manufactured functions;
2. neural-network forward/backward cost using the measured high-order AD-error
   envelope from ``neural_high_order_derivative_benchmark.py``.

The memory statistic is the payload of tensors saved by autograd for backward.
It is a reproducible graph-memory proxy, not process RSS or GPU peak memory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch
from torch import Tensor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.general_taylor_adaptive_order_benchmark import (
    FIXED_METHODS,
    _derivative_bound,
    _mp_absolute_error,
    load_reference_payload,
)
from experiments.general_taylor_neural_benchmark import TanhNetwork, set_seed
from experiments.general_taylor_operator_benchmark import FUNCTIONS, INTERVAL, ORDERS, POINTS, _gauss_rule
from neural_network_solvers.general_taylor_finite_part import (
    regularized_taylor_quotient,
    scaled_derivative_error_model,
    sequential_tolerance_taylor_finite_part_operator,
    sequential_tolerance_taylor_quotient,
)


# Local quotient budgets are chosen one order tighter than the intended
# residual scale used in the collocation experiments.  Their effect is also
# reported explicitly, rather than hidden as a tuning constant.
OPERATOR_TARGETS = {"float32": 1.0e-4, "float64": 1.0e-10}
NEURAL_TARGETS = {"float32": 1.0e-3, "float64": 1.0e-6}


def _operator_experiment(
    reference_path: Path,
    *,
    quadrature_points: int,
    max_continuation_order: int,
) -> dict[str, object]:
    references, fixed_records, reference_precision = load_reference_payload(
        reference_path, quadrature_points
    )
    records: list[dict[str, object]] = []
    for row in fixed_records:
        if row["method"] == "balanced_p3":
            records.append(dict(row))
    stopping: list[dict[str, object]] = []
    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        nodes, weights = _gauss_rule(quadrature_points, dtype)
        for order in ORDERS:
            for spec in FUNCTIONS:
                for point in POINTS:
                    point_tensor = torch.tensor([[point]], dtype=dtype)
                    t_stored = float(point_tensor.item())
                    value, diagnostics = sequential_tolerance_taylor_finite_part_operator(
                        spec.torch_function,
                        point_tensor,
                        nodes,
                        weights,
                        order=order,
                        target_error=OPERATOR_TARGETS[dtype_name],
                        interval=INTERVAL,
                        max_continuation_order=max_continuation_order,
                        trust_region_multiplier=None,
                        derivative_bound=_derivative_bound(spec.name),
                        return_diagnostics=True,
                    )
                    scalar = float(value.item())
                    reference_decimal = references[(spec.name, order, dtype_name, t_stored)]
                    records.append(
                        {
                            "dtype": dtype_name,
                            "order": order,
                            "function": spec.name,
                            "t": point,
                            "t_stored": t_stored,
                            "method": "sequential_tolerance",
                            "value": scalar,
                            "reference": float(mp.mpf(reference_decimal)),
                            "reference_decimal": reference_decimal,
                            "absolute_error": _mp_absolute_error(
                                scalar, reference_decimal, reference_precision
                            ),
                            "nonfinite": not math.isfinite(scalar),
                        }
                    )
                    counts = Counter(int(value) for value in diagnostics.selected_order.reshape(-1).tolist())
                    stopping.append(
                        {
                            "dtype": dtype_name,
                            "order": order,
                            "function": spec.name,
                            "t": point,
                            "highest_derivative_order": diagnostics.highest_derivative_order,
                            "selected_counts": {str(key): value for key, value in sorted(counts.items())},
                        }
                    )
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dtype"]), int(row["order"]), str(row["method"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dtype_name, order, method), group in sorted(grouped.items()):
        errors = np.asarray([float(row["absolute_error"]) for row in group], dtype=float)
        summary.append(
            {
                "dtype": dtype_name,
                "order": order,
                "method": method,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "maximum_error": float(np.max(errors)),
                "sample_count": len(group),
            }
        )
    return {
        "records": records, "stopping": stopping, "summary": summary,
        "reference_scope": "schema-v2 dtype-specific stored coordinates; MP subtraction",
        "reference_decimal_precision": reference_precision,
    }


def _saved_tensor_payload_bytes(action: Callable[[], tuple[Tensor, object]]) -> tuple[Tensor, object, int]:
    saved_bytes = 0

    def pack(tensor: Tensor) -> Tensor:
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        value, diagnostics = action()
        value.square().mean().backward()
    return value, diagnostics, saved_bytes


def _time_neural_method(
    model: TanhNetwork,
    t_value: float,
    nodes: Tensor,
    *,
    order: int,
    method: str,
    target_error: float,
    derivative_error_model,
    derivative_bound,
    repeats: int,
) -> dict[str, object]:
    timings: list[float] = []
    saved_payloads: list[int] = []
    highest_orders: list[int] = []
    selected_counts: Counter[int] = Counter()
    last_value: Tensor | None = None
    for repeat in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        t_tensor = torch.tensor([[t_value]], dtype=nodes.dtype)
        started = time.perf_counter()
        if method == "fixed_p3":
            value, _, saved_bytes = _saved_tensor_payload_bytes(
                lambda: (
                    regularized_taylor_quotient(
                        model,
                        t_tensor,
                        nodes,
                        order=order,
                        continuation_order=3,
                        threshold_mode="balanced",
                    )[0],
                    None,
                )
            )
            highest_order = order + 3
        else:
            value, diagnostics, saved_bytes = _saved_tensor_payload_bytes(
                lambda: sequential_tolerance_taylor_quotient(
                    model,
                    t_tensor,
                    nodes,
                    order=order,
                    target_error=target_error,
                    max_continuation_order=3,
                    derivative_error_model=derivative_error_model,
                    derivative_bound=derivative_bound,
                )
            )
            highest_order = diagnostics.highest_derivative_order
            if repeat > 0:
                selected_counts.update(
                    int(item) for item in diagnostics.selected_order.reshape(-1).tolist()
                )
        elapsed = time.perf_counter() - started
        last_value = value.detach()
        if repeat > 0:
            timings.append(elapsed)
            saved_payloads.append(saved_bytes)
            highest_orders.append(highest_order)
    assert last_value is not None
    return {
        "median_forward_backward_seconds": statistics.median(timings),
        "median_saved_tensor_payload_bytes": int(statistics.median(saved_payloads)),
        "mean_highest_derivative_order": float(statistics.mean(highest_orders)),
        "selected_counts": {str(key): value for key, value in sorted(selected_counts.items())},
        "value": last_value.cpu().numpy().tolist(),
    }


def _neural_experiment(
    derivative_metrics_path: Path,
    *,
    quadrature_points: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    derivative_metrics = json.loads(derivative_metrics_path.read_text(encoding="utf-8"))
    eta = derivative_metrics["eta_scaled_envelope"]
    magnitude_payload = derivative_metrics.get("derivative_magnitude_envelope")
    points = (-0.8, -0.4, 0.0, 0.4, 0.8)
    records: list[dict[str, object]] = []
    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        set_seed(seed)
        model = TanhNetwork(width=32, hidden_layers=3).to(dtype=dtype)
        envelope = {int(order): float(value) for order, value in eta[dtype_name].items()}
        error_model = scaled_derivative_error_model(envelope)
        if magnitude_payload is None:
            magnitude_envelope = {
                int(row["order"]): float(row["maximum_reference_magnitude"])
                for row in derivative_metrics["summary"]
                if row["dtype"] == dtype_name
            }
        else:
            magnitude_envelope = {
                int(order): float(value)
                for order, value in magnitude_payload[dtype_name].items()
            }

        def empirical_derivative_bound(
            t: Tensor, tau: Tensor, derivative_order: int
        ) -> Tensor:
            # Two is an explicit held-out safety factor.  The bound is an
            # empirical network-family envelope, not a rigorous analytic one.
            return torch.full_like(
                t + tau,
                2.0 * magnitude_envelope[derivative_order],
            )

        nodes, _ = _gauss_rule(quadrature_points, dtype)
        for order in (2, 4, 6):
            for point in points:
                fixed = _time_neural_method(
                    model,
                    point,
                    nodes,
                    order=order,
                    method="fixed_p3",
                    target_error=NEURAL_TARGETS[dtype_name],
                    derivative_error_model=error_model,
                    derivative_bound=empirical_derivative_bound,
                    repeats=repeats,
                )
                sequential = _time_neural_method(
                    model,
                    point,
                    nodes,
                    order=order,
                    method="sequential_tolerance",
                    target_error=NEURAL_TARGETS[dtype_name],
                    derivative_error_model=error_model,
                    derivative_bound=empirical_derivative_bound,
                    repeats=repeats,
                )
                fixed_value = np.asarray(fixed.pop("value"), dtype=float)
                sequential_value = np.asarray(sequential.pop("value"), dtype=float)
                disagreement = float(np.sqrt(np.mean((fixed_value - sequential_value) ** 2)))
                for method, result in (("fixed_p3", fixed), ("sequential_tolerance", sequential)):
                    records.append(
                        {
                            "dtype": dtype_name,
                            "order": order,
                            "t": point,
                            "method": method,
                            "target_error": NEURAL_TARGETS[dtype_name],
                            "rmse_difference_from_fixed_p3": 0.0 if method == "fixed_p3" else disagreement,
                            **result,
                        }
                    )
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dtype"]), int(row["order"]), str(row["method"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dtype_name, order, method), group in sorted(grouped.items()):
        counts: Counter[int] = Counter()
        for row in group:
            counts.update(
                {int(key): int(value) for key, value in row["selected_counts"].items()}
            )
        total = sum(counts.values())
        local_total = total - counts[-1]
        summary.append(
            {
                "dtype": dtype_name,
                "order": order,
                "method": method,
                "median_forward_backward_seconds": float(np.median([row["median_forward_backward_seconds"] for row in group])),
                "median_saved_tensor_payload_bytes": int(np.median([row["median_saved_tensor_payload_bytes"] for row in group])),
                "mean_highest_derivative_order": float(np.mean([row["mean_highest_derivative_order"] for row in group])),
                "rmse_difference_from_fixed_p3": float(np.sqrt(np.mean([row["rmse_difference_from_fixed_p3"] ** 2 for row in group]))),
                "direct_fraction": counts[-1] / total if total else None,
                "mean_selected_local_order": (
                    sum(key * value for key, value in counts.items() if key >= 0)
                    / local_total
                    if local_total
                    else None
                ),
            }
        )
    return {"records": records, "summary": summary}


def _plot(payload: dict[str, object], path: Path) -> None:
    rows = payload["neural"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    for axis, metric, ylabel in (
        (axes[0], "median_forward_backward_seconds", "forward+backward time (s)"),
        (axes[1], "median_saved_tensor_payload_bytes", "saved-tensor payload (bytes)"),
        (axes[2], "mean_highest_derivative_order", "highest generated derivative"),
    ):
        for dtype, color in (("float32", "#d95f8d"), ("float64", "#009e73")):
            for method, marker, linestyle in (("fixed_p3", "s", "--"), ("sequential_tolerance", "o", "-")):
                selected = sorted(
                    (row for row in rows if row["dtype"] == dtype and row["method"] == method),
                    key=lambda row: row["order"],
                )
                axis.plot(
                    [row["order"] for row in selected],
                    [row[metric] for row in selected],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    label=f"{dtype}, {method}",
                )
        axis.set_xlabel("kernel order m")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-metrics", type=Path, default=Path("results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json"))
    parser.add_argument("--derivative-metrics", type=Path, default=Path("results/neural_high_order_derivatives/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sequential_tolerance"))
    parser.add_argument("--operator-quadrature-points", type=int, default=256)
    parser.add_argument("--neural-quadrature-points", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--operator-only", action="store_true", help="Recheck accuracy without rerunning neural timing probes.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "operator_targets": OPERATOR_TARGETS,
        "neural_targets": NEURAL_TARGETS,
        "operator": _operator_experiment(
            args.reference_metrics,
            quadrature_points=args.operator_quadrature_points,
            max_continuation_order=3,
        ),
        "neural": None if args.operator_only else _neural_experiment(
            args.derivative_metrics,
            quadrature_points=args.neural_quadrature_points,
            repeats=args.repeats,
            seed=args.seed,
        ),
        "memory_metric": "sum of tensor payload bytes saved by autograd hooks for backward",
        "neural_remainder_policy": "2x empirical maximum derivative magnitude from the independent 100-digit benchmark",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.operator_only:
        _plot(payload, args.output_dir / "sequential_cost.png")
    print(json.dumps({"operator": payload["operator"]["summary"], "neural": None if args.operator_only else payload["neural"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
