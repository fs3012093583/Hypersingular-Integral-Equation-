"""Reproducible accuracy benchmark for the general Taylor finite-part operator.

The benchmark keeps the problem deliberately small and deterministic.  It
compares direct subtraction, balanced continuations of degrees zero through
three, and several fixed neighbourhoods on three smooth functions.  Quotient
and finite-part reference values are independently evaluated with ``mpmath``
at configurable precision.

Examples
--------
Quick smoke-sized run::

    python experiments/general_taylor_operator_benchmark.py --preset light

Publication-sized grid (more quadrature resolutions and diagonal offsets)::

    python experiments/general_taylor_operator_benchmark.py --preset full
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

# Support both ``python -m experiments...`` and the direct invocation shown in
# the module docstring.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network_solvers.general_taylor_finite_part import (
    regularized_taylor_quotient,
    taylor_finite_part_operator,
)


INTERVAL = (-1.0, 1.0)
POINTS = (-0.95, -0.5, 0.0, 0.5, 0.95)
ORDERS = tuple(range(1, 7))
DTYPES = (("float32", torch.float32), ("float64", torch.float64))


class FunctionSpec(NamedTuple):
    name: str
    torch_function: Callable[[torch.Tensor], torch.Tensor]
    mp_function: Callable[[mp.mpf], mp.mpf]
    mp_derivative: Callable[[mp.mpf, int], mp.mpf]


FUNCTIONS = (
    FunctionSpec("exp(t)", torch.exp, mp.exp, lambda x, order: mp.exp(x)),
    FunctionSpec(
        "sin(pi*t)",
        lambda x: torch.sin(math.pi * x),
        lambda x: mp.sin(mp.pi * x),
        lambda x, order: mp.pi**order * mp.sin(mp.pi * x + order * mp.pi / 2),
    ),
    FunctionSpec(
        "1/(1+0.5*t)",
        lambda x: 1.0 / (1.0 + 0.5 * x),
        lambda x: 1 / (1 + mp.mpf("0.5") * x),
        lambda x, order: (-mp.mpf("0.5")) ** order
        * mp.factorial(order)
        / (1 + mp.mpf("0.5") * x) ** (order + 1),
    ),
)


def _strategies() -> tuple[dict[str, object], ...]:
    """Return the fixed, intentionally comparable regularization choices."""
    rows: list[dict[str, object]] = [
        {"name": "direct", "mode": "diagonal", "continuation_order": 0, "fixed": None}
    ]
    rows.extend(
        {
            "name": f"balanced_p{continuation_order}",
            "mode": "balanced",
            "continuation_order": continuation_order,
            "fixed": None,
        }
        for continuation_order in range(4)
    )
    rows.extend(
        {
            "name": f"fixed_{threshold:.0e}_p1",
            "mode": "fixed",
            "continuation_order": 1,
            "fixed": threshold,
        }
        for threshold in (5.0e-2, 5.0e-3, 5.0e-4)
    )
    return tuple(rows)


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _errors(value: float, reference: mp.mpf) -> tuple[float | None, float | None]:
    if not _finite(value):
        return None, None
    absolute = abs(mp.mpf(float(value)) - reference)
    # A finite-part value may legitimately vanish by symmetry.  Normalize by
    # one near zero so the displayed comparison remains interpretable instead
    # of turning roundoff-level absolute errors into artificial 1e+300 ratios.
    return float(absolute), float(absolute / max(abs(reference), mp.mpf("1")))


def mp_regularized_quotient(
    function: Callable[[mp.mpf], mp.mpf],
    derivative: Callable[[mp.mpf, int], mp.mpf],
    t: mp.mpf | float,
    tau: mp.mpf | float,
    order: int,
) -> mp.mpf:
    """High-precision Taylor quotient, including its removable diagonal limit."""
    # ``mp.mpf`` accepts an mp value unchanged and lifts a Python float from
    # its binary representation.  Do not round through ``str(float)``: this
    # benchmark intentionally measures the coordinates actually stored by
    # each torch dtype.
    t_mp, tau_mp = mp.mpf(t), mp.mpf(tau)
    delta = tau_mp - t_mp
    if delta == 0:
        return derivative(t_mp, order) / mp.factorial(order)
    numerator = function(tau_mp)
    for derivative_order in range(order):
        numerator -= (
            derivative(t_mp, derivative_order)
            * delta**derivative_order
            / mp.factorial(derivative_order)
        )
    return numerator / delta**order


def mp_finite_part(
    function: Callable[[mp.mpf], mp.mpf],
    derivative: Callable[[mp.mpf, int], mp.mpf],
    t: mp.mpf | float,
    order: int,
) -> mp.mpf:
    """High-precision finite-part identity on the benchmark interval."""
    left, right = (mp.mpf(value) for value in INTERVAL)
    t_mp = mp.mpf(t)
    continuation_terms = 40
    derivatives = [
        derivative(t_mp, derivative_order)
        for derivative_order in range(order + continuation_terms + 1)
    ]
    analytic = mp.mpf("0")
    for derivative_order in range(order - 1):
        power = order - derivative_order - 1
        moment = ((left - t_mp) ** (-power) - (right - t_mp) ** (-power)) / power
        analytic += derivatives[derivative_order] * moment / mp.factorial(derivative_order)
    analytic += (
        derivatives[order - 1]
        * mp.log((right - t_mp) / (t_mp - left))
        / mp.factorial(order - 1)
    )
    # Keep Taylor data outside the adaptive integrand.  Recomputing it at every
    # quadrature sample is needlessly expensive and makes a lightweight preset
    # unexpectedly slow.
    def regular(tau: mp.mpf) -> mp.mpf:
        delta = tau - t_mp
        if abs(delta) <= mp.mpf("1e-8"):
            return mp.fsum(
                derivatives[order + local_order]
                * delta**local_order
                / mp.factorial(order + local_order)
                for local_order in range(continuation_terms + 1)
            )
        numerator = function(tau)
        for derivative_order in range(order):
            numerator -= derivatives[derivative_order] * delta**derivative_order / mp.factorial(
                derivative_order
            )
        return numerator / delta**order

    return analytic + mp.quad(
        regular,
        [left, t_mp, right],
    )


def _gauss_rule(num_points: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    return (
        torch.as_tensor(nodes, dtype=dtype).reshape(-1, 1),
        torch.as_tensor(weights, dtype=dtype).reshape(-1, 1),
    )


def _as_serializable_float(value: float) -> float | None:
    return value if _finite(value) else None


def run_benchmark(
    *,
    quotient_deltas: tuple[float, ...],
    quadrature_sizes: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all requested comparisons and return raw quotient/operator rows."""
    quotient_rows: list[dict[str, object]] = []
    operator_rows: list[dict[str, object]] = []
    rule_cache = {
        (quadrature_size, dtype_name): _gauss_rule(quadrature_size, dtype)
        for quadrature_size in quadrature_sizes
        for dtype_name, dtype in DTYPES
    }
    interval_length = INTERVAL[1] - INTERVAL[0]

    for spec in FUNCTIONS:
        for order in ORDERS:
            for t_value in POINTS:
                for dtype_name, dtype in DTYPES:
                    t_tensor = torch.tensor([[t_value]], dtype=dtype)
                    t_stored = float(t_tensor.item())
                    t_mp = mp.mpf(t_stored)
                    for requested_delta in quotient_deltas:
                        tau_tensor = torch.tensor([[t_value + requested_delta]], dtype=dtype)
                        tau_stored = float(tau_tensor.item())
                        tau_mp = mp.mpf(tau_stored)
                        actual_delta = float((tau_tensor - t_tensor).item())
                        if actual_delta == 0.0:
                            # This is an observable float32 spacing limitation, not an
                            # operator failure; report it explicitly rather than silently
                            # changing the requested local problem.
                            quotient_rows.append(
                                {
                                    "function": spec.name,
                                    "order": order,
                                    "t": t_value,
                                    "t_stored": t_stored,
                                    "tau_stored": tau_stored,
                                    "dtype": dtype_name,
                                    "requested_delta": requested_delta,
                                    "actual_delta": actual_delta,
                                    "strategy": "all_skipped_coincident",
                                    "skipped": True,
                                }
                            )
                            continue
                        reference = mp_regularized_quotient(
                            spec.mp_function,
                            spec.mp_derivative,
                            t_mp,
                            tau_mp,
                            order,
                        )
                        for strategy in _strategies():
                            quotient, threshold = regularized_taylor_quotient(
                                spec.torch_function,
                                t_tensor,
                                tau_tensor,
                                order=order,
                                continuation_order=int(strategy["continuation_order"]),
                                threshold_mode=str(strategy["mode"]),
                                near_diagonal_threshold=strategy["fixed"],
                                interval_length=interval_length,
                            )
                            value = float(quotient.item())
                            absolute, scaled = _errors(value, reference)
                            quotient_rows.append(
                                {
                                    "function": spec.name,
                                    "order": order,
                                    "t": t_value,
                                    "t_stored": t_stored,
                                    "tau_stored": tau_stored,
                                    "dtype": dtype_name,
                                    "requested_delta": requested_delta,
                                    "actual_delta": actual_delta,
                                    "strategy": strategy["name"],
                                    "threshold": threshold,
                                    "value": _as_serializable_float(value),
                                    "reference": float(reference),
                                    "reference_decimal": str(reference),
                                    "absolute_error": absolute,
                                    # This is deliberately a symmetry-safe scaled
                                    # error, not a strict relative error: the exact
                                    # quotient can vanish at isolated samples.
                                    "scaled_error": scaled,
                                    "nonfinite": not _finite(value),
                                    "skipped": False,
                                }
                            )

                    reference = mp_finite_part(
                        spec.mp_function, spec.mp_derivative, t_mp, order
                    )
                    for quadrature_size in quadrature_sizes:
                        nodes, weights = rule_cache[(quadrature_size, dtype_name)]
                        for strategy in _strategies():
                            value_tensor = taylor_finite_part_operator(
                                spec.torch_function,
                                t_tensor,
                                nodes,
                                weights,
                                order=order,
                                interval=INTERVAL,
                                continuation_order=int(strategy["continuation_order"]),
                                threshold_mode=str(strategy["mode"]),
                                near_diagonal_threshold=strategy["fixed"],
                            )
                            value = float(value_tensor.item())
                            absolute, scaled = _errors(value, reference)
                            operator_rows.append(
                                {
                                    "function": spec.name,
                                    "order": order,
                                    "t": t_value,
                                    "t_stored": t_stored,
                                    "dtype": dtype_name,
                                    "quadrature_size": quadrature_size,
                                    "strategy": strategy["name"],
                                    "value": _as_serializable_float(value),
                                    "reference": float(reference),
                                    "reference_decimal": str(reference),
                                    "absolute_error": absolute,
                                    "scaled_error": scaled,
                                    "nonfinite": not _finite(value),
                                }
                            )
    return quotient_rows, operator_rows


def _summarize(rows: list[dict[str, object]], *, kind: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        key = (row["dtype"], row["order"], row["strategy"])
        grouped[key].append(row)
    summary: list[dict[str, object]] = []
    for (dtype, order, strategy), subset in sorted(grouped.items()):
        finite_errors = [float(row["scaled_error"]) for row in subset if not row["nonfinite"]]
        summary.append(
            {
                "kind": kind,
                "dtype": dtype,
                "order": order,
                "strategy": strategy,
                "sample_count": len(subset),
                "nonfinite_count": sum(bool(row["nonfinite"]) for row in subset),
                "nonfinite_proportion": sum(bool(row["nonfinite"]) for row in subset) / len(subset),
                "median_scaled_error": float(np.median(finite_errors)) if finite_errors else None,
                "max_scaled_error": float(np.max(finite_errors)) if finite_errors else None,
            }
        )
    return summary


def _plot_metric(
    rows: list[dict[str, object]], output_path: Path, *, title: str, operator: bool
) -> None:
    colors = {
        "direct": "#b23a48",
        "balanced_p0": "#d8821f",
        "balanced_p1": "#3579b8",
        "balanced_p2": "#20815e",
        "balanced_p3": "#6a4c93",
        "fixed_5e-02_p1": "#888888",
        "fixed_5e-03_p1": "#555555",
        "fixed_5e-04_p1": "#222222",
    }
    strategies = tuple(strategy["name"] for strategy in _strategies())
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 12.8), sharex=True, sharey=True)
    for order_index, order in enumerate(ORDERS):
        for row_index, (dtype_name, _) in enumerate(DTYPES):
            # Each dtype receives a two-by-three order block, keeping the
            # figure readable in a conventional single-column-wide layout.
            axis = axes[2 * row_index + order_index // 3, order_index % 3]
            for strategy in strategies:
                subset = [
                    row
                    for row in rows
                    if row.get("dtype") == dtype_name
                    and row.get("order") == order
                    and row.get("strategy") == strategy
                    and not row.get("nonfinite")
                    and row.get("scaled_error") is not None
                ]
                if operator and subset:
                    largest = max(int(row["quadrature_size"]) for row in subset)
                    subset = [row for row in subset if row["quadrature_size"] == largest]
                if not subset:
                    continue
                median_error = float(np.median([float(row["scaled_error"]) for row in subset]))
                axis.scatter(
                    strategies.index(strategy),
                    max(median_error, 1.0e-30),
                    s=28,
                    color=colors[strategy],
                    label=strategy,
                    zorder=3,
                )
            axis.set_yscale("log")
            axis.set_title(f"m={order}, {dtype_name}")
            axis.grid(True, which="both", alpha=0.24)
            axis.set_xticks(range(len(strategies)))
            axis.set_xticklabels([name.replace("balanced_", "b_") for name in strategies], rotation=55, ha="right", fontsize=7)
    for axis in axes[:, 0]:
        axis.set_ylabel("median scaled error")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("light", "full"), default="light")
    parser.add_argument("--output-dir", type=Path, default=Path("results/general_taylor_operator_benchmark"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--mp-dps", type=int, default=90)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.mp_dps < 40:
        raise ValueError("--mp-dps must be at least 40 for a useful reference")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mp.mp.dps = args.mp_dps
    if args.preset == "light":
        quotient_deltas = (1.0e-2, 1.0e-4)
        quadrature_sizes = (96,)
    else:
        quotient_deltas = (1.0e-1, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9)
        quadrature_sizes = (32, 64, 128, 256, 512, 1024, 2048)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quotient_rows, operator_rows = run_benchmark(
        quotient_deltas=quotient_deltas,
        quadrature_sizes=quadrature_sizes,
    )
    quotient_summary = _summarize(quotient_rows, kind="quotient")
    operator_summary = _summarize(operator_rows, kind="finite_part_operator")
    payload = {
        "schema_version": 2,
        "experiment": "general_taylor_operator_benchmark",
        "seed": args.seed,
        "preset": args.preset,
        "reference": {
            "backend": "mpmath",
            "decimal_precision": args.mp_dps,
            "coordinate_scope": "per-dtype stored torch coordinates",
            "error_subtraction": "mpmath before JSON float serialization",
            "reference_decimal_field": "reference_decimal",
        },
        "interval": list(INTERVAL),
        "points": list(POINTS),
        "orders": list(ORDERS),
        "quotient_deltas": list(quotient_deltas),
        "quadrature_sizes": list(quadrature_sizes),
        "quotient_records": quotient_rows,
        "operator_records": operator_rows,
        "summary": {"quotient": quotient_summary, "finite_part_operator": operator_summary},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.no_plots:
        _plot_metric(
            quotient_rows,
            args.output_dir / "quotient_scaled_error.png",
            title="Taylor quotient error against high-precision references",
            operator=False,
        )
        _plot_metric(
            operator_rows,
            args.output_dir / "finite_part_operator_scaled_error.png",
            title="Finite-part operator error against high-precision references",
            operator=True,
        )
    print(
        json.dumps(
            {
                "metrics_file": str(args.output_dir / "metrics.json"),
                "quotient_records": len(quotient_rows),
                "operator_records": len(operator_rows),
                "quotient_nonfinite": sum(bool(row.get("nonfinite")) for row in quotient_rows),
                "operator_nonfinite": sum(bool(row.get("nonfinite")) for row in operator_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
