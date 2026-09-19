"""Pilot study for cancellation-aware arbitrary-order Taylor regularization.

The experiment has two parts:

1. Compare direct Taylor divided differences with balanced local
   continuations against an 80-digit reference for ``u(t)=exp(t)``.
2. Compare the resulting finite-part operators for orders one through four as
   the Gauss--Legendre quadrature resolution increases.

The study is intentionally optimizer-free: it isolates the numerical mechanism
that would otherwise be confounded with neural-network training variability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

from neural_network_solvers.general_taylor_finite_part import (
    balanced_threshold,
    regularized_taylor_quotient,
    taylor_finite_part_operator,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    gauss_legendre_rule,
)


ORDERS = tuple(range(1, 7))


def _mp_regularized_exp_quotient(t: float, delta: float, order: int) -> mp.mpf:
    t_mp = mp.mpf(str(t))
    delta_mp = mp.mpf(str(delta))
    # 1F1(1; m+1; delta) / m! equals
    # sum_{j>=0} delta**j/(m+j)! and remains stable even when an adaptive
    # high-precision quadrature samples extremely close to the diagonal.
    tail = mp.hyper([1], [order + 1], delta_mp) / mp.factorial(order)
    return mp.exp(t_mp) * tail


def _mp_exp_finite_part(t: float, order: int) -> mp.mpf:
    t_mp = mp.mpf(str(t))
    value = mp.exp(t_mp)
    analytic = mp.mpf("0")
    for derivative_order in range(order - 1):
        power = order - derivative_order - 1
        moment = ((-1 - t_mp) ** (-power) - (1 - t_mp) ** (-power)) / power
        analytic += value * moment / mp.factorial(derivative_order)
    analytic += (
        value
        * mp.log((1 - t_mp) / (1 + t_mp))
        / mp.factorial(order - 1)
    )

    def regular(tau: mp.mpf) -> mp.mpf:
        delta = tau - t_mp
        return value * mp.hyper([1], [order + 1], delta) / mp.factorial(order)

    return analytic + mp.quad(regular, [-1, t_mp, 1])


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def run_quotient_study() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    t_value = 0.23
    dtype_specs = (
        ("float64", torch.float64, np.logspace(-1, -14, 14)),
        ("float32", torch.float32, np.logspace(-1, -7, 13)),
    )
    for dtype_name, dtype, requested_deltas in dtype_specs:
        for order in ORDERS:
            for requested_delta in requested_deltas:
                t = torch.tensor([[t_value]], dtype=dtype)
                tau = torch.tensor([[t_value + float(requested_delta)]], dtype=dtype)
                actual_delta = float((tau - t).item())
                if actual_delta == 0.0:
                    continue
                reference = float(_mp_regularized_exp_quotient(t_value, actual_delta, order))
                strategies: dict[str, tuple[str, int, float | None]] = {
                    "direct": ("diagonal", 0, None),
                    "balanced_p0": ("balanced", 0, None),
                    "balanced_p1": ("balanced", 1, None),
                    "balanced_p2": ("balanced", 2, None),
                    "fixed_1e-3_p1": ("fixed", 1, 1.0e-3),
                }
                for strategy, (mode, continuation_order, fixed_threshold) in strategies.items():
                    quotient, threshold = regularized_taylor_quotient(
                        torch.exp,
                        t,
                        tau,
                        order=order,
                        continuation_order=continuation_order,
                        threshold_mode=mode,
                        near_diagonal_threshold=fixed_threshold,
                    )
                    value = float(quotient.item())
                    records.append(
                        {
                            "dtype": dtype_name,
                            "order": order,
                            "requested_delta": float(requested_delta),
                            "actual_delta": actual_delta,
                            "strategy": strategy,
                            "threshold": threshold,
                            "value": value,
                            "reference": reference,
                            "absolute_error": abs(value - reference),
                            "relative_error": _relative_error(value, reference),
                        }
                    )
    return records


def run_operator_study() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    t_value = 0.0
    quadrature_sizes = (32, 64, 128, 256, 512, 1024, 2048)
    references = {order: float(_mp_exp_finite_part(t_value, order)) for order in ORDERS}
    for order in ORDERS:
        for quadrature_size in quadrature_sizes:
            nodes, weights = gauss_legendre_rule(quadrature_size)
            t = torch.tensor([[t_value]], dtype=torch.float64)
            strategies: dict[str, tuple[str, int, float | None]] = {
                "direct": ("diagonal", 0, None),
                "balanced_p0": ("balanced", 0, None),
                "balanced_p1": ("balanced", 1, None),
                "balanced_p2": ("balanced", 2, None),
                "fixed_1e-3_p1": ("fixed", 1, 1.0e-3),
            }
            for strategy, (mode, continuation_order, fixed_threshold) in strategies.items():
                value_tensor = taylor_finite_part_operator(
                    torch.exp,
                    t,
                    nodes,
                    weights,
                    order=order,
                    continuation_order=continuation_order,
                    threshold_mode=mode,
                    near_diagonal_threshold=fixed_threshold,
                )
                value = float(value_tensor.item())
                reference = references[order]
                records.append(
                    {
                        "order": order,
                        "quadrature_size": quadrature_size,
                        "strategy": strategy,
                        "threshold": (
                            0.0
                            if mode == "diagonal"
                            else fixed_threshold
                            if mode == "fixed"
                            else balanced_threshold(
                                order,
                                continuation_order,
                                dtype=torch.float64,
                            )
                        ),
                        "value": value,
                        "reference": reference,
                        "absolute_error": abs(value - reference),
                        "relative_error": _relative_error(value, reference),
                    }
                )
    return records


def summarize(
    quotient_records: list[dict[str, object]],
    operator_records: list[dict[str, object]],
) -> dict[str, object]:
    quotient_summary: list[dict[str, object]] = []
    for dtype in ("float64", "float32"):
        for order in ORDERS:
            subset = [
                row
                for row in quotient_records
                if row["dtype"] == dtype and row["order"] == order
            ]
            direct = [row for row in subset if row["strategy"] == "direct"]
            balanced = [row for row in subset if row["strategy"] == "balanced_p2"]
            quotient_summary.append(
                {
                    "dtype": dtype,
                    "order": order,
                    "balanced_p2_threshold": balanced[0]["threshold"],
                    "direct_max_relative_error": max(row["relative_error"] for row in direct),
                    "balanced_p2_max_relative_error": max(
                        row["relative_error"] for row in balanced
                    ),
                    "direct_median_relative_error": float(
                        np.median([row["relative_error"] for row in direct])
                    ),
                    "balanced_p2_median_relative_error": float(
                        np.median([row["relative_error"] for row in balanced])
                    ),
                }
            )

    operator_summary: list[dict[str, object]] = []
    for order in ORDERS:
        subset = [row for row in operator_records if row["order"] == order]
        finest = max(row["quadrature_size"] for row in subset)
        finest_rows = [row for row in subset if row["quadrature_size"] == finest]
        operator_summary.append(
            {
                "order": order,
                "quadrature_size": finest,
                "errors": {
                    row["strategy"]: row["absolute_error"] for row in finest_rows
                },
            }
        )
    return {
        "quotient_summary": quotient_summary,
        "operator_summary_at_finest_quadrature": operator_summary,
    }


def plot_results(
    quotient_records: list[dict[str, object]],
    operator_records: list[dict[str, object]],
    output_dir: Path,
) -> None:
    strategies = ("direct", "balanced_p0", "balanced_p1", "balanced_p2")
    colors = {
        "direct": "#b33c2f",
        "balanced_p0": "#dd8a1f",
        "balanced_p1": "#2f78b7",
        "balanced_p2": "#27835c",
    }
    labels = {
        "direct": "direct quotient",
        "balanced_p0": "balanced, p=0",
        "balanced_p1": "balanced, p=1",
        "balanced_p2": "balanced, p=2",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True, sharey=True)
    for axis, order in zip(axes.flat, ORDERS):
        for strategy in strategies:
            rows = [
                row
                for row in quotient_records
                if row["dtype"] == "float64"
                and row["order"] == order
                and row["strategy"] == strategy
            ]
            rows.sort(key=lambda row: row["actual_delta"], reverse=True)
            axis.loglog(
                [row["actual_delta"] for row in rows],
                [max(row["relative_error"], 1.0e-18) for row in rows],
                marker="o",
                markersize=3.5,
                linewidth=1.35,
                color=colors[strategy],
                label=labels[strategy],
            )
        axis.set_title(f"kernel order m={order}")
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes[1, :]:
        axis.set_xlabel(r"$|\tau-t|$")
    axes[0, 0].set_ylabel("relative quotient error")
    axes[1, 0].set_ylabel("relative quotient error")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Cancellation in Taylor-regularized divided differences", y=0.995)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "quotient_cancellation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True, sharey=True)
    for axis, order in zip(axes.flat, ORDERS):
        for strategy in strategies:
            rows = [
                row
                for row in operator_records
                if row["order"] == order and row["strategy"] == strategy
            ]
            rows.sort(key=lambda row: row["quadrature_size"])
            axis.loglog(
                [row["quadrature_size"] for row in rows],
                [max(row["absolute_error"], 1.0e-18) for row in rows],
                marker="o",
                markersize=3.5,
                linewidth=1.35,
                color=colors[strategy],
                label=labels[strategy],
            )
        axis.set_title(f"kernel order m={order}")
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes[1, :]:
        axis.set_xlabel("Gauss points")
    axes[0, 0].set_ylabel("absolute operator error")
    axes[1, 0].set_ylabel("absolute operator error")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Finite-part operator error for u(t)=exp(t)", y=0.995)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "operator_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/taylor_cancellation_study"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mp.mp.dps = 80

    quotient_records = run_quotient_study()
    operator_records = run_operator_study()
    summary = summarize(quotient_records, operator_records)
    payload = {
        "experiment": "cancellation_aware_general_taylor_finite_part",
        "reference_precision_digits": mp.mp.dps,
        "function": "exp(t)",
        "interval": [-1.0, 1.0],
        "quotient_records": quotient_records,
        "operator_records": operator_records,
        "summary": summary,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_results(quotient_records, operator_records, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
