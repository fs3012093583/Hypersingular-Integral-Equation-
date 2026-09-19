"""Build manuscript figures and a compact metric summary from raw experiments."""

from __future__ import annotations

import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"
GENERATED_DIR = ROOT / "paper" / "generated"
OPERATOR_METRICS = ROOT / "results" / "general_taylor_operator_benchmark_full" / "metrics.json"
MULTIPLIER_METRICS = ROOT / "results" / "general_taylor_multiplier_sweep" / "metrics.json"
NEURAL_DIRS = (
    ROOT / "results" / "general_taylor_neural_benchmark_main",
    ROOT / "results" / "general_taylor_neural_benchmark_seed20260919",
    ROOT / "results" / "general_taylor_neural_benchmark_seed20260920",
)
SPECTRAL_METRICS = ROOT / "results" / "general_taylor_spectral_baseline" / "summary.json"
NONLINEAR_METRICS = ROOT / "results" / "general_taylor_neural_nonlinear_m4" / "summary.json"
FAMILY_METRICS = ROOT / "results" / "general_taylor_neural_m6_families" / "summary.json"
DIRECT_NEURAL_DIRS = (
    ROOT / "results" / "general_taylor_neural_m6_direct",
    ROOT / "results" / "general_taylor_neural_m6_direct_seeds",
)
DERIVATIVE_METRICS = ROOT / "results" / "neural_high_order_derivatives" / "metrics.json"
SEQUENTIAL_METRICS = ROOT / "results" / "sequential_tolerance" / "metrics.json"
OPERATOR_BUDGET_METRICS = (
    ROOT / "results" / "operator_budget_accuracy_global_cap" / "metrics.json"
)
OPERATOR_BUDGET_MULTI_PATHS = (
    ROOT
    / "results"
    / "operator_budget_end_to_end_m2_seeds20260918_20"
    / "summary.json",
    ROOT
    / "results"
    / "operator_budget_end_to_end_globalcap_tol1e4_seed20260918"
    / "summary.json",
    ROOT
    / "results"
    / "operator_budget_end_to_end_globalcap_tol1e4_seeds20260919_20"
    / "summary.json",
)
END_TO_END_RECORD_PATHS = {
    2: {
        "fixed_p2": ROOT / "results" / "sequential_tolerance_end_to_end_m2_seed20260918" / "run_001.json",
        "fixed_p3": ROOT / "results" / "sequential_tolerance_end_to_end_m2_seed20260918" / "run_002.json",
        "sequential": ROOT / "results" / "sequential_tolerance_end_to_end_progressive_m2_tol1e3_seed20260918" / "run_001.json",
        "limit_p0": ROOT / "results" / "operator_budget_end_to_end_m2_seeds20260918_20" / "run_001.json",
        "operator_budget": ROOT / "results" / "operator_budget_end_to_end_m2_seeds20260918_20" / "run_002.json",
    },
    4: {
        "fixed_p2": ROOT / "results" / "sequential_tolerance_end_to_end_m4_seed20260918" / "run_001.json",
        "fixed_p3": ROOT / "results" / "sequential_tolerance_end_to_end_m4_seed20260918" / "run_002.json",
        "sequential": ROOT / "results" / "sequential_tolerance_end_to_end_progressive_m4_tol1e3_seed20260918" / "run_001.json",
        "limit_p0": ROOT / "results" / "limit_p0_end_to_end_seed20260918" / "run_001.json",
        "operator_budget": ROOT / "results" / "operator_budget_end_to_end_globalcap_tol1e4_seed20260918" / "run_001.json",
    },
    6: {
        "fixed_p2": ROOT / "results" / "sequential_tolerance_end_to_end_fixed_p2_m6_seed20260918" / "run_001.json",
        "fixed_p3": ROOT / "results" / "sequential_tolerance_end_to_end_fixed_p3_m6_seed20260918" / "run_001.json",
        "sequential": ROOT / "results" / "sequential_tolerance_end_to_end_progressive_m6_tol1e3_seed20260918" / "run_001.json",
        "limit_p0": ROOT / "results" / "limit_p0_end_to_end_seed20260918" / "run_002.json",
        "operator_budget": ROOT / "results" / "operator_budget_end_to_end_globalcap_tol1e4_seed20260918" / "run_002.json",
    },
}


COLORS = {
    "direct": "#D55E00",
    "balanced": "#0072B2",
    "float32": "#CC79A7",
    "float64": "#009E73",
    "spectral": "#E69F00",
    "neural": "#0072B2",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"{name}.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    return float(np.mean(array)), float(np.std(array, ddof=1 if array.size > 1 else 0))


def build_operator_figure(payload: dict[str, object]) -> dict[str, object]:
    quotient = payload["quotient_records"]
    operator = payload["operator_records"]
    assert isinstance(quotient, list) and isinstance(operator, list)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))

    # Panel (a): the m=6 cancellation curve.  Aggregate across all three
    # functions and five target locations to avoid selecting a favourable case.
    for strategy, label, color, marker, linestyle in (
        ("direct", "direct", COLORS["direct"], "o", "--"),
        ("balanced_p0", "balanced, p=0", "#E69F00", "s", ":"),
        ("balanced_p1", "balanced, p=1", "#56B4E9", "D", "-."),
        ("balanced_p2", "balanced, p=2", COLORS["balanced"], "^", "-"),
        ("balanced_p3", "balanced, p=3", "#009E73", "v", "-"),
    ):
        rows = [
            row
            for row in quotient
            if row.get("dtype") == "float64"
            and row.get("order") == 6
            and row.get("strategy") == strategy
            and row.get("scaled_error") is not None
        ]
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            grouped[float(row["requested_delta"])].append(float(row["scaled_error"]))
        x = sorted(grouped)
        y = [float(np.median(grouped[value])) for value in x]
        axes[0].loglog(x, np.maximum(y, 1.0e-18), marker=marker, linestyle=linestyle, color=color, label=label)
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"near-diagonal distance $|\tau-t|$")
    axes[0].set_ylabel("median scaled quotient error")
    axes[0].set_title("(a) Sixth-order local quotient, float64")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    # Panel (b): operator RMSE across functions and locations at the finest
    # shared quadrature resolution.
    for dtype, color, marker in (("float32", COLORS["float32"], "s"), ("float64", COLORS["float64"], "o")):
        for strategy, label_suffix, linestyle, fillstyle in (
            ("direct", "direct", "--", "none"),
            ("balanced_p2", "balanced, p=2", "-", "full"),
        ):
            orders, errors = [], []
            for order in range(1, 7):
                rows = [
                    row
                    for row in operator
                    if row["dtype"] == dtype
                    and row["order"] == order
                    and row["strategy"] == strategy
                    and row["quadrature_size"] == 256
                ]
                orders.append(order)
                errors.append(rmse([float(row["absolute_error"]) for row in rows]))
            axes[1].semilogy(
                orders,
                errors,
                marker=marker,
                linestyle=linestyle,
                fillstyle=fillstyle,
                color=color,
                label=f"{dtype}, {label_suffix}",
            )
    axes[1].set_xlabel("kernel order m")
    axes[1].set_ylabel("operator absolute-error RMSE")
    axes[1].set_title("(b) 256-point finite-part operator")
    axes[1].set_xticks(range(1, 7))
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "operator_cancellation_comparison")
    plt.close(fig)

    summary: dict[str, object] = {"operator_rmse_n256": {}}
    for dtype in ("float32", "float64"):
        dtype_rows: dict[str, object] = {}
        for order in range(1, 7):
            order_rows: dict[str, float] = {}
            for strategy in ("direct", "balanced_p0", "balanced_p1", "balanced_p2", "balanced_p3"):
                rows = [
                    row
                    for row in operator
                    if row["dtype"] == dtype
                    and row["order"] == order
                    and row["strategy"] == strategy
                    and row["quadrature_size"] == 256
                ]
                order_rows[strategy] = rmse([float(row["absolute_error"]) for row in rows])
            dtype_rows[str(order)] = order_rows
        summary["operator_rmse_n256"][dtype] = dtype_rows
    summary["quotient_nonfinite_count"] = sum(bool(row.get("nonfinite")) for row in quotient)
    summary["operator_nonfinite_count"] = sum(bool(row.get("nonfinite")) for row in operator)
    return summary


def build_quadrature_convergence_figure(payload: dict[str, object]) -> None:
    operator = payload["operator_records"]
    assert isinstance(operator, list)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=True)
    styles = (
        ("direct", "direct", COLORS["direct"], "o", "--"),
        ("fixed_5e-03_p1", r"fixed $\delta=5\times10^{-3}$, $p=1$", "#7A7A7A", "s", ":"),
        ("balanced_p2", "balanced, p=2", COLORS["balanced"], "^", "-"),
    )
    for axis, order in zip(axes, (2, 4, 6)):
        for strategy, label, color, marker, linestyle in styles:
            subset = [
                row
                for row in operator
                if row["dtype"] == "float64"
                and row["order"] == order
                and row["strategy"] == strategy
            ]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                grouped[int(row["quadrature_size"])].append(float(row["absolute_error"]))
            sizes = sorted(grouped)
            errors = [rmse(grouped[size]) for size in sizes]
            axis.loglog(sizes, errors, marker=marker, linestyle=linestyle, color=color, label=label)
        axis.set_title(f"m={order}")
        axis.set_xlabel(r"quadrature points $N_q$")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("operator absolute-error RMSE")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, "operator_quadrature_convergence")
    plt.close(fig)


def collect_neural_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for directory in NEURAL_DIRS:
        for path in sorted(directory.glob("run_*.json")):
            record = load_json(path)
            if record.get("method") == "neural_taylor_pinn":
                records.append(record)
    return records


def summarize_neural(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in (2, 4, 6):
        subset = [record for record in records if int(record["config"]["order"]) == order]
        succeeded = [record for record in subset if record.get("status") == "ok"]
        if not succeeded:
            raise RuntimeError(f"no successful neural records for order {order}")
        expected_keys = (
            "family", "coupling", "nonlinear_coefficient", "steps", "lbfgs_steps",
            "width", "hidden_layers", "collocation_points", "training_quadrature_points",
            "rhs_quadrature_points", "validation_points", "validation_quadrature_points",
            "continuation_order", "threshold_mode", "threshold_multiplier", "dtype",
        )
        signatures = {
            tuple(record["config"].get(key) for key in expected_keys)
            for record in succeeded
        }
        if len(signatures) != 1:
            raise RuntimeError(f"inconsistent neural configurations for order {order}")
        row: dict[str, object] = {
            "order": order,
            "runs": len(subset),
            "successes": len(succeeded),
            "failure_rate": 1.0 - len(succeeded) / len(subset),
        }
        for metric in (
            "solution_rmse_closed_interval",
            "solution_max_abs_error_closed_interval",
            "validation_residual_rmse",
            "training_seconds",
            "end_to_end_seconds",
        ):
            values = np.asarray([float(record[metric]) for record in succeeded], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(
                np.std(values, ddof=1 if values.size > 1 else 0)
            )
        rows.append(row)
    return rows


def collect_direct_neural_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for directory in DIRECT_NEURAL_DIRS:
        for path in sorted(directory.glob("run_*.json")):
            record = load_json(path)
            if record.get("method") == "neural_taylor_pinn":
                records.append(record)
    return records


def summarize_direct_neural(records: list[dict[str, object]]) -> dict[str, object] | None:
    if not records:
        return None
    succeeded = [record for record in records if record.get("status") == "ok"]
    row: dict[str, object] = {
        "runs": len(records),
        "successes": len(succeeded),
        "failure_rate": 1.0 - len(succeeded) / len(records),
    }
    for metric in (
        "solution_rmse_closed_interval",
        "solution_max_abs_error_closed_interval",
        "validation_residual_rmse",
        "training_seconds",
        "end_to_end_seconds",
    ):
        values = np.asarray([float(record[metric]) for record in succeeded], dtype=float)
        row[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
        row[f"{metric}_std"] = (
            float(np.std(values, ddof=1 if values.size > 1 else 0))
            if values.size
            else None
        )
    return row


def build_neural_figure(neural_summary: list[dict[str, object]], spectral_payload: dict[str, object]) -> None:
    spectral = [
        row
        for row in spectral_payload["summary"]
        if row["family"] == "exp" and row["method"] == "chebyshev_taylor_collocation"
    ]
    spectral_by_order = {int(row["order"]): row for row in spectral}
    orders = [int(row["order"]) for row in neural_summary]
    neural_rmse = [float(row["solution_rmse_closed_interval_mean"]) for row in neural_summary]
    neural_std = [float(row["solution_rmse_closed_interval_std"]) for row in neural_summary]
    neural_time = [float(row["training_seconds_mean"]) for row in neural_summary]
    neural_time_std = [float(row["training_seconds_std"]) for row in neural_summary]
    spectral_rmse = [float(spectral_by_order[order]["solution_rmse_closed_interval_mean"]) for order in orders]
    spectral_time = [float(spectral_by_order[order]["training_seconds_mean"]) for order in orders]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    axes[0].errorbar(
        orders,
        neural_rmse,
        yerr=neural_std,
        marker="o",
        capsize=3,
        color=COLORS["neural"],
        label="neural collocation (3 seeds)",
    )
    axes[0].semilogy(
        orders,
        spectral_rmse,
        marker="s",
        linestyle="--",
        color=COLORS["spectral"],
        label="Chebyshev collocation",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(orders)
    axes[0].set_xlabel("kernel order m")
    axes[0].set_ylabel("closed-interval solution RMSE")
    axes[0].set_title("(a) Manufactured solution $u(t)=e^t$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        orders,
        neural_time,
        yerr=neural_time_std,
        marker="o",
        capsize=3,
        color=COLORS["neural"],
        label="neural training",
    )
    axes[1].semilogy(
        orders,
        spectral_time,
        marker="s",
        linestyle="--",
        color=COLORS["spectral"],
        label="spectral solve",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(orders)
    axes[1].set_xlabel("kernel order m")
    axes[1].set_ylabel("wall time (s)")
    axes[1].set_title("(b) CPU cost")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "neural_and_spectral_comparison")
    plt.close(fig)


def build_derivative_and_stopping_figure(
    derivative_payload: dict[str, object],
    sequential_payload: dict[str, object],
) -> dict[str, object]:
    derivative_summary = derivative_payload["summary"]
    stopping_summary = sequential_payload["neural"]["summary"]
    assert isinstance(derivative_summary, list) and isinstance(stopping_summary, list)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))

    for dtype, color, marker in (
        ("float32", COLORS["float32"], "s"),
        ("float64", COLORS["float64"], "o"),
    ):
        selected = sorted(
            (row for row in derivative_summary if row["dtype"] == dtype),
            key=lambda row: int(row["order"]),
        )
        axes[0].semilogy(
            [int(row["order"]) for row in selected],
            [max(float(row["maximum_scaled_error"]), 1.0e-18) for row in selected],
            marker=marker,
            color=color,
            label=dtype,
        )
    axes[0].set_xlabel("derivative order k")
    axes[0].set_ylabel("maximum scaled AD error")
    axes[0].set_title("(a) AD error envelope")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    for axis, metric, title, ylabel in (
        (axes[1], "median_forward_backward_seconds", "(b) Runtime ratio", "sequential / fixed p=3"),
        (axes[2], "median_saved_tensor_payload_bytes", "(c) Saved-tensor ratio", "sequential / fixed p=3"),
    ):
        for dtype, color, marker in (
            ("float32", COLORS["float32"], "s"),
            ("float64", COLORS["float64"], "o"),
        ):
            fixed = {
                int(row["order"]): float(row[metric])
                for row in stopping_summary
                if row["dtype"] == dtype and row["method"] == "fixed_p3"
            }
            sequential = {
                int(row["order"]): float(row[metric])
                for row in stopping_summary
                if row["dtype"] == dtype and row["method"] == "sequential_tolerance"
            }
            orders = sorted(fixed)
            axis.plot(
                orders,
                [sequential[order] / fixed[order] for order in orders],
                marker=marker,
                color=color,
                label=dtype,
            )
        axis.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
        axis.set_xticks((2, 4, 6))
        axis.set_xlabel("kernel order m")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "derivative_error_and_early_stopping")
    plt.close(fig)
    return {
        "derivative_error": derivative_payload,
        "sequential_tolerance": sequential_payload,
    }


def build_end_to_end_training_figure() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for order, methods in END_TO_END_RECORD_PATHS.items():
        loaded = {method: load_json(path) for method, path in methods.items()}
        fingerprints = {
            str(record["initial_model_fingerprint"]) for record in loaded.values()
        }
        if len(fingerprints) != 1:
            raise RuntimeError(f"unmatched initial weights in m={order} comparison")
        for method, record in loaded.items():
            config = record["config"]
            row = {
                "order": order,
                "method": method,
                "seed": int(config["seed"]),
                "target_error": config.get("target_error") if method == "sequential" else None,
                "target_residual_error": (
                    config.get("target_residual_error")
                    if method == "operator_budget"
                    else None
                ),
                "max_continuation_order": (
                    config.get("max_continuation_order")
                    if method in ("sequential", "operator_budget")
                    else (0 if method == "limit_p0" else int(method[-1]))
                ),
                "solution_rmse_closed_interval": float(record["solution_rmse_closed_interval"]),
                "validation_residual_rmse": float(record["validation_residual_rmse"]),
                "training_seconds": float(record["training_seconds_including_envelope_refresh"]),
                "saved_tensor_payload_bytes": int(
                    record["post_training_cost"]["median_saved_tensor_payload_bytes"]
                ),
                "selection_diagnostics": record.get("selection_diagnostics"),
            }
            records.append(row)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    styles = {
        "fixed_p2": ("fixed p=2", "o", "--", "#E69F00"),
        "fixed_p3": ("fixed p=3", "s", ":", "#D55E00"),
        "limit_p0": ("limit only, p=0", "D", "-.", "#999999"),
        "sequential": ("pairwise budget, p<=3", "^", "-", "#56B4E9"),
        "operator_budget": ("operator budget, p<=3", "v", "-", "#0072B2"),
    }
    specifications = (
        ("solution_rmse_closed_interval", "solution RMSE", "(a) Closed-interval error"),
        ("training_seconds", "complete training time (s)", "(b) Adam + L-BFGS"),
        ("saved_tensor_payload_bytes", "saved-tensor payload (bytes)", "(c) One residual backward"),
    )
    for axis, (metric, ylabel, title) in zip(axes, specifications):
        for method, (label, marker, linestyle, color) in styles.items():
            selected = sorted(
                (row for row in records if row["method"] == method),
                key=lambda row: int(row["order"]),
            )
            axis.plot(
                [row["order"] for row in selected],
                [row[metric] for row in selected],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xticks((2, 4, 6))
        axis.set_xlabel("kernel order m")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "end_to_end_training_comparison")
    plt.close(fig)
    return records


def collect_operator_budget_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in OPERATOR_BUDGET_MULTI_PATHS:
        payload = load_json(path)
        for record in payload["records"]:
            if record.get("method") == "operator_budget" and record.get("status") == "ok":
                records.append(record)
    return records


def build_operator_budget_figure(
    operator_payload: dict[str, object],
    fixed_records: list[dict[str, object]],
    adaptive_records: list[dict[str, object]],
) -> dict[str, object]:
    summary = operator_payload["summary"]
    assert isinstance(summary, list)
    selected_strategies = (
        ("fixed_p0", "fixed p=0", "D", "-.", "#999999"),
        ("fixed_p2", "fixed p=2", "o", "--", "#E69F00"),
        ("fixed_p3", "fixed p=3", "s", ":", "#D55E00"),
        (
            "operator_budget_1e-04_pmax3",
            r"operator budget, $\mathrm{tol}_R=10^{-4}$",
            "v",
            "-",
            "#0072B2",
        ),
    )
    orders = (2, 4, 6)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    for strategy, label, marker, linestyle, color in selected_strategies:
        rows = {
            int(row["order"]): row
            for row in summary
            if row["dtype"] == "float64" and row["strategy"] == strategy
        }
        axes[0, 0].semilogy(
            orders,
            [float(rows[order]["representation_error_rmse"]) for order in orders],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            orders,
            [float(rows[order]["mean_highest_derivative_order"]) for order in orders],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    axes[0, 0].set_ylabel("representation-error RMSE")
    axes[0, 0].set_title("(a) Float64, 256-point operator")
    axes[0, 1].set_ylabel("mean highest derivative order")
    axes[0, 1].set_title("(b) Derivative demand")
    for axis in axes[0]:
        axis.set_xticks(orders)
        axis.set_xlabel("kernel order m")
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)

    fixed_by_order = {
        order: [
            record
            for record in fixed_records
            if int(record["config"]["order"]) == order
            and record["config"]["family"] == "exp"
            and float(record["config"]["nonlinear_coefficient"]) == 0.0
        ]
        for order in orders
    }
    adaptive_by_order = {
        order: [
            record
            for record in adaptive_records
            if int(record["config"]["order"]) == order
        ]
        for order in orders
    }
    multiseed_summary: list[dict[str, object]] = []
    for method, grouped, label, marker, color in (
        ("fixed_p2", fixed_by_order, "fixed p=2", "o", "#E69F00"),
        (
            "operator_budget",
            adaptive_by_order,
            "operator budget",
            "v",
            "#0072B2",
        ),
    ):
        for axis, metric, title, ylabel in (
            (
                axes[1, 0],
                "solution_rmse_closed_interval",
                "(c) Three-seed solution error",
                "closed-interval solution RMSE",
            ),
            (
                axes[1, 1],
                "validation_residual_rmse",
                "(d) Three-seed residual",
                "interior residual RMSE",
            ),
        ):
            means, standard_deviations = [], []
            for order in orders:
                values = [float(record[metric]) for record in grouped[order]]
                if len(values) != 3:
                    raise RuntimeError(
                        f"expected three {method} records for m={order}, got {len(values)}"
                    )
                mean, standard_deviation = sample_mean_std(values)
                means.append(mean)
                standard_deviations.append(standard_deviation)
                if metric == "solution_rmse_closed_interval":
                    multiseed_summary.append(
                        {
                            "method": method,
                            "order": order,
                            "runs": len(values),
                            "solution_rmse_mean": mean,
                            "solution_rmse_sample_std": standard_deviation,
                            "residual_rmse_mean": sample_mean_std(
                                [
                                    float(record["validation_residual_rmse"])
                                    for record in grouped[order]
                                ]
                            )[0],
                            "residual_rmse_sample_std": sample_mean_std(
                                [
                                    float(record["validation_residual_rmse"])
                                    for record in grouped[order]
                                ]
                            )[1],
                        }
                    )
            axis.errorbar(
                orders,
                means,
                yerr=standard_deviations,
                marker=marker,
                capsize=3,
                color=color,
                label=label,
            )
            axis.set_yscale("log")
            axis.set_xticks(orders)
            axis.set_xlabel("kernel order m")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "operator_budget_accuracy_and_robustness")
    plt.close(fig)
    return {
        "operator_budget_operator_summary": summary,
        "operator_budget_multiseed_summary": multiseed_summary,
    }


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    operator_payload = load_json(OPERATOR_METRICS)
    compact = build_operator_figure(operator_payload)
    build_quadrature_convergence_figure(operator_payload)
    neural_records = collect_neural_records()
    neural_summary = summarize_neural(neural_records)
    spectral_payload = load_json(SPECTRAL_METRICS)
    build_neural_figure(neural_summary, spectral_payload)
    if DERIVATIVE_METRICS.exists() and SEQUENTIAL_METRICS.exists():
        compact.update(
            build_derivative_and_stopping_figure(
                load_json(DERIVATIVE_METRICS),
                load_json(SEQUENTIAL_METRICS),
            )
        )
    if all(
        path.exists()
        for methods in END_TO_END_RECORD_PATHS.values()
        for path in methods.values()
    ):
        compact["end_to_end_training"] = build_end_to_end_training_figure()
    if OPERATOR_BUDGET_METRICS.exists() and all(
        path.exists() for path in OPERATOR_BUDGET_MULTI_PATHS
    ):
        compact.update(
            build_operator_budget_figure(
                load_json(OPERATOR_BUDGET_METRICS),
                neural_records,
                collect_operator_budget_records(),
            )
        )
    compact["neural_main"] = neural_summary
    compact["neural_direct_m6"] = summarize_direct_neural(collect_direct_neural_records())
    compact["spectral"] = spectral_payload["summary"]
    compact["multiplier_sweep"] = load_json(MULTIPLIER_METRICS)["summary"]
    if NONLINEAR_METRICS.exists():
        compact["nonlinear_m4"] = load_json(NONLINEAR_METRICS)["summary"]
    if FAMILY_METRICS.exists():
        compact["m6_families"] = load_json(FAMILY_METRICS)["summary"]
    compact["runtime"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (GENERATED_DIR / "paper_metrics.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figures": sorted(path.name for path in FIGURE_DIR.glob("*")), "neural_main": neural_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
