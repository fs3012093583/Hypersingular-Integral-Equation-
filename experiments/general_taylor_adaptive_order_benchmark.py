"""Benchmark local error-driven switching and continuation-order selection.

The experiment compares fixed-order balanced continuations with two joint
selectors over the candidate set ``{direct, p=0, ..., p=p_max}``:

* ``adaptive_accuracy`` minimizes the local error indicator;
* ``adaptive_cost_aware`` chooses the lowest-cost candidate within a factor
  ``rho`` of the minimum predicted error.

High-precision reference values are read from the established full operator
benchmark.  The output also records selection frequencies, making the choice
of ``p`` auditable rather than treating it as a hidden hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.general_taylor_operator_benchmark import (
    FUNCTIONS,
    INTERVAL,
    POINTS,
    _gauss_rule,
)
from neural_network_solvers.general_taylor_finite_part import (
    adaptive_taylor_finite_part_operator,
)


ORDERS = tuple(range(1, 7))
DTYPES = (("float32", torch.float32), ("float64", torch.float64))
FIXED_METHODS = tuple(f"balanced_p{order}" for order in range(4))
ADAPTIVE_METHODS = (
    "adaptive_accuracy",
    "adaptive_cost_aware",
    "bounded_remainder_cost_aware",
)


def _derivative_bound(name: str):
    """Return exact segment bounds for the three manufactured functions."""
    if name == "exp(t)":
        return lambda t, tau, order: torch.exp(torch.maximum(t, tau))
    if name == "sin(pi*t)":
        return lambda t, tau, order: torch.full_like(
            t + tau, math.pi**order
        )
    if name == "1/(1+0.5*t)":
        return lambda t, tau, order: (
            0.5**order
            * math.factorial(order)
            / (1.0 + 0.5 * torch.minimum(t, tau)).pow(order + 1)
        )
    raise ValueError(f"no derivative bound registered for {name}")


def load_reference_payload(path: Path, quadrature_points: int) -> tuple[
    dict[tuple[str, int, str, float], str],
    list[dict[str, object]],
    int,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("adaptive benchmark requires schema_version=2 references")
    metadata = payload.get("reference")
    if not isinstance(metadata, dict):
        raise ValueError("adaptive benchmark reference metadata is missing")
    try:
        precision = int(metadata["decimal_precision"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("adaptive benchmark reference precision is missing") from error
    if precision < 40:
        raise ValueError("adaptive benchmark reference precision is too low")
    references: dict[tuple[str, int, str, float], str] = {}
    fixed_records: list[dict[str, object]] = []
    source = payload.get("operator_records")
    if not isinstance(source, list):
        raise ValueError("adaptive benchmark reference records are missing")
    for row in source:
        if not isinstance(row, dict):
            raise ValueError("adaptive benchmark reference record must be an object")
        try:
            key = (
                str(row["function"]),
                int(row["order"]),
                str(row["dtype"]),
                float(row["t_stored"]),
            )
            reference_decimal = str(row["reference_decimal"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "adaptive benchmark requires schema-v2 t_stored/reference_decimal"
            ) from error
        previous = references.setdefault(key, reference_decimal)
        if previous != reference_decimal:
            raise ValueError(f"inconsistent reference_decimal for {key}")
        if (
            int(row["quadrature_size"]) == quadrature_points
            and str(row["strategy"]) in FIXED_METHODS
        ):
            fixed_records.append(
                {
                    "dtype": str(row["dtype"]),
                    "order": int(row["order"]),
                    "function": str(row["function"]),
                    "t": float(row["t"]),
                    "t_stored": float(row["t_stored"]),
                    "method": str(row["strategy"]),
                    "value": row["value"],
                    "reference": float(row["reference"]),
                    "reference_decimal": reference_decimal,
                    "absolute_error": row["absolute_error"],
                    "nonfinite": bool(row["nonfinite"]),
                }
            )
    for dtype_name, _ in DTYPES:
        for spec in FUNCTIONS:
            for order in ORDERS:
                count = sum(
                    1
                    for function, item_order, item_dtype, _ in references
                    if (function, item_order, item_dtype)
                    == (spec.name, order, dtype_name)
                )
                if count != len(POINTS):
                    raise ValueError(
                        "adaptive benchmark is missing dtype-specific stored-coordinate "
                        f"cases for {spec.name}, m={order}, {dtype_name}"
                    )
    return references, fixed_records, precision


def _mp_absolute_error(value: float, reference_decimal: str, precision: int) -> float:
    with mp.workdps(precision):
        return float(abs(mp.mpf(float(value)) - mp.mpf(reference_decimal)))


def run(
    reference_path: Path,
    *,
    quadrature_points: int,
    max_continuation_order: int,
    near_optimal_factor: float,
) -> dict[str, object]:
    references, fixed_records, reference_precision = load_reference_payload(
        reference_path, quadrature_points
    )
    records = list(fixed_records)
    selection_records: list[dict[str, object]] = []
    for dtype_name, dtype in DTYPES:
        nodes, weights = _gauss_rule(quadrature_points, dtype)
        for order in ORDERS:
            for spec in FUNCTIONS:
                for point in POINTS:
                    point_tensor = torch.tensor([[point]], dtype=dtype)
                    t_stored = float(point_tensor.item())
                    reference_decimal = references[
                        (spec.name, order, dtype_name, t_stored)
                    ]
                    for method, selection_mode, use_bound in (
                        ("adaptive_accuracy", "accuracy", False),
                        ("adaptive_cost_aware", "cost_aware", False),
                        ("bounded_remainder_cost_aware", "cost_aware", True),
                    ):
                        result = adaptive_taylor_finite_part_operator(
                            spec.torch_function,
                            point_tensor,
                            nodes,
                            weights,
                            order=order,
                            interval=INTERVAL,
                            max_continuation_order=max_continuation_order,
                            selection_mode=selection_mode,
                            near_optimal_factor=near_optimal_factor,
                            trust_region_multiplier=None if use_bound else 1.0,
                            derivative_bound=_derivative_bound(spec.name) if use_bound else None,
                            return_diagnostics=True,
                        )
                        value_tensor, diagnostics = result
                        value = float(value_tensor.item())
                        finite = math.isfinite(value)
                        records.append(
                            {
                                "dtype": dtype_name,
                                "order": order,
                                "function": spec.name,
                                "t": point,
                                "t_stored": t_stored,
                                "method": method,
                                "value": value if finite else None,
                                "reference": float(mp.mpf(reference_decimal)),
                                "reference_decimal": reference_decimal,
                                "absolute_error": (
                                    _mp_absolute_error(
                                        value, reference_decimal, reference_precision
                                    )
                                    if finite
                                    else None
                                ),
                                "nonfinite": not finite,
                            }
                        )
                        counts = Counter(
                            int(selected)
                            for selected in diagnostics.selected_order.reshape(-1).tolist()
                        )
                        selection_records.append(
                            {
                                "dtype": dtype_name,
                                "order": order,
                                "function": spec.name,
                                "t": point,
                                "t_stored": t_stored,
                                "method": method,
                                "counts": {
                                    "direct": counts.get(-1, 0),
                                    **{
                                        f"p{local_order}": counts.get(local_order, 0)
                                        for local_order in range(max_continuation_order + 1)
                                    },
                                },
                            }
                        )

    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dtype"]), int(row["order"]), str(row["method"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dtype_name, order, method), subset in sorted(grouped.items()):
        errors = np.asarray(
            [float(row["absolute_error"]) for row in subset if row["absolute_error"] is not None],
            dtype=float,
        )
        summary.append(
            {
                "dtype": dtype_name,
                "order": order,
                "method": method,
                "sample_count": len(subset),
                "nonfinite_count": sum(bool(row["nonfinite"]) for row in subset),
                "absolute_error_rmse": float(np.sqrt(np.mean(np.square(errors))))
                if len(errors)
                else None,
                "absolute_error_max": float(np.max(errors)) if len(errors) else None,
            }
        )

    selection_summary: list[dict[str, object]] = []
    selection_groups: dict[tuple[str, int, str], Counter[int]] = defaultdict(Counter)
    for row in selection_records:
        key = (str(row["dtype"]), int(row["order"]), str(row["method"]))
        counts = row["counts"]
        assert isinstance(counts, dict)
        for label, count in counts.items():
            selected_order = -1 if label == "direct" else int(label[1:])
            selection_groups[key][selected_order] += int(count)
    for (dtype_name, order, method), counts in sorted(selection_groups.items()):
        total = sum(counts.values())
        selection_summary.append(
            {
                "dtype": dtype_name,
                "order": order,
                "method": method,
                "total_pairs": total,
                "fractions": {
                    "direct": counts[-1] / total,
                    **{
                        f"p{local_order}": counts[local_order] / total
                        for local_order in range(max_continuation_order + 1)
                    },
                },
            }
        )

    return {
        "experiment": "local_error_adaptive_order_benchmark",
        "reference_file": str(reference_path),
        "reference_scope": "schema-v2 dtype-specific stored coordinates; MP subtraction",
        "reference_decimal_precision": reference_precision,
        "interval": list(INTERVAL),
        "quadrature_points": quadrature_points,
        "orders": list(ORDERS),
        "max_continuation_order": max_continuation_order,
        "near_optimal_factor": near_optimal_factor,
        "records": records,
        "selection_records": selection_records,
        "summary": summary,
        "selection_summary": selection_summary,
    }


def plot(payload: dict[str, object], output_base: Path) -> None:
    summary = payload["summary"]
    assert isinstance(summary, list)
    methods = (*FIXED_METHODS, *ADAPTIVE_METHODS)
    labels = {
        "balanced_p0": "fixed p=0",
        "balanced_p1": "fixed p=1",
        "balanced_p2": "fixed p=2",
        "balanced_p3": "fixed p=3",
        "adaptive_accuracy": "adaptive, accuracy",
        "adaptive_cost_aware": "adaptive, cost-aware",
        "bounded_remainder_cost_aware": "bounded remainder",
    }
    colors = {
        "balanced_p0": "#999999",
        "balanced_p1": "#777777",
        "balanced_p2": "#555555",
        "balanced_p3": "#333333",
        "adaptive_accuracy": "#0072B2",
        "adaptive_cost_aware": "#D55E00",
        "bounded_remainder_cost_aware": "#009E73",
    }
    markers = {
        "balanced_p0": "o",
        "balanced_p1": "s",
        "balanced_p2": "^",
        "balanced_p3": "v",
        "adaptive_accuracy": "D",
        "adaptive_cost_aware": "P",
        "bounded_remainder_cost_aware": "X",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for axis, dtype_name in zip(axes, ("float32", "float64")):
        for method in methods:
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["dtype"] == dtype_name and row["method"] == method
                ),
                key=lambda row: int(row["order"]),
            )
            axis.semilogy(
                [int(row["order"]) for row in rows],
                [max(float(row["absolute_error_rmse"]), 1.0e-18) for row in rows],
                color=colors[method],
                marker=markers[method],
                label=labels[method],
            )
        axis.set_title(dtype_name)
        axis.set_xlabel("kernel order m")
        axis.set_xticks(ORDERS)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("operator absolute-error RMSE")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{suffix}"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        default=Path("results/general_taylor_operator_benchmark_full/metrics.json"),
    )
    parser.add_argument("--quadrature-points", type=int, default=256)
    parser.add_argument("--p-max", type=int, default=3)
    parser.add_argument("--near-optimal-factor", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/general_taylor_adaptive_order"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run(
        args.reference_metrics,
        quadrature_points=args.quadrature_points,
        max_continuation_order=args.p_max,
        near_optimal_factor=args.near_optimal_factor,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot(payload, args.output_dir / "adaptive_order_comparison")


if __name__ == "__main__":
    main()
