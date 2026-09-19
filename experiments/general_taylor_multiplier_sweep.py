"""Sensitivity study for the dimensionless switching multiplier ``c``.

The high-precision references are read from the complete operator benchmark so
that this experiment measures only the effect of changing

    delta_* = c L eps**(1/(m+p+1)).

No reference value is recomputed in floating point by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

from experiments.general_taylor_operator_benchmark import (
    FUNCTIONS,
    INTERVAL,
    POINTS,
    _gauss_rule,
)
from neural_network_solvers.general_taylor_finite_part import (
    balanced_threshold,
    taylor_finite_part_operator,
)


ORDERS = (2, 4, 6)
MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
DTYPES = (("float32", torch.float32), ("float64", torch.float64))


def load_references(path: Path) -> tuple[dict[tuple[str, int, str, float], str], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("reference benchmark must use schema_version=2")
    reference_metadata = payload.get("reference")
    if not isinstance(reference_metadata, dict):
        raise ValueError("reference benchmark lacks reference metadata")
    try:
        precision = int(reference_metadata["decimal_precision"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reference benchmark lacks decimal precision") from error
    if precision < 40:
        raise ValueError("reference benchmark precision is too low")
    source = payload.get("operator_records")
    if not isinstance(source, list):
        raise ValueError("reference benchmark lacks operator records")
    references: dict[tuple[str, int, str, float], str] = {}
    for row in source:
        if not isinstance(row, dict):
            raise ValueError("reference operator record must be an object")
        try:
            key = (
                str(row["function"]),
                int(row["order"]),
                str(row["dtype"]),
                float(row["t_stored"]),
            )
            decimal = str(row["reference_decimal"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "reference benchmark lacks schema-v2 t_stored/reference_decimal fields"
            ) from error
        previous = references.setdefault(key, decimal)
        if previous != decimal:
            raise ValueError(f"inconsistent reference_decimal for {key}")
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
                        "reference benchmark is missing dtype-specific stored-coordinate "
                        f"cases for {spec.name}, m={order}, {dtype_name}"
                    )
    return references, precision


def _mp_errors(value: float, reference_decimal: str, precision: int) -> tuple[float, float]:
    with mp.workdps(precision):
        reference = mp.mpf(reference_decimal)
        absolute = abs(mp.mpf(float(value)) - reference)
        return float(absolute), float(absolute / max(abs(reference), mp.mpf("1")))


def run(reference_path: Path, quadrature_points: int) -> dict[str, object]:
    references, reference_precision = load_references(reference_path)
    records: list[dict[str, object]] = []
    for dtype_name, dtype in DTYPES:
        nodes, weights = _gauss_rule(quadrature_points, dtype)
        for order in ORDERS:
            for multiplier in MULTIPLIERS:
                threshold = balanced_threshold(
                    order,
                    2,
                    dtype=dtype,
                    interval_length=INTERVAL[1] - INTERVAL[0],
                    threshold_multiplier=multiplier,
                )
                for spec in FUNCTIONS:
                    for point in POINTS:
                        point_tensor = torch.tensor([[point]], dtype=dtype)
                        t_stored = float(point_tensor.item())
                        value = float(
                            taylor_finite_part_operator(
                                spec.torch_function,
                                point_tensor,
                                nodes,
                                weights,
                                order=order,
                                interval=INTERVAL,
                                continuation_order=2,
                                threshold_mode="balanced",
                                threshold_multiplier=multiplier,
                            ).item()
                        )
                        reference_decimal = references[
                            (spec.name, order, dtype_name, t_stored)
                        ]
                        finite = math.isfinite(value)
                        absolute = (
                            _mp_errors(value, reference_decimal, reference_precision)[0]
                            if finite
                            else None
                        )
                        records.append(
                            {
                                "dtype": dtype_name,
                                "order": order,
                                "continuation_order": 2,
                                "multiplier": multiplier,
                                "threshold": threshold,
                                "function": spec.name,
                                "t": point,
                                "t_stored": t_stored,
                                "value": value if finite else None,
                                "reference": float(mp.mpf(reference_decimal)),
                                "reference_decimal": reference_decimal,
                                "absolute_error": absolute,
                                "nonfinite": not finite,
                            }
                        )
    summary: list[dict[str, object]] = []
    for dtype_name, _ in DTYPES:
        for order in ORDERS:
            for multiplier in MULTIPLIERS:
                subset = [
                    row
                    for row in records
                    if row["dtype"] == dtype_name
                    and row["order"] == order
                    and row["multiplier"] == multiplier
                ]
                errors = [
                    float(row["absolute_error"])
                    for row in subset
                    if row["absolute_error"] is not None
                ]
                summary.append(
                    {
                        "dtype": dtype_name,
                        "order": order,
                        "continuation_order": 2,
                        "multiplier": multiplier,
                        "threshold": subset[0]["threshold"],
                        "sample_count": len(subset),
                        "nonfinite_count": sum(bool(row["nonfinite"]) for row in subset),
                        "absolute_error_rmse": float(
                            math.sqrt(sum(value * value for value in errors) / len(errors))
                        ),
                        "absolute_error_max": float(max(errors)),
                    }
                )
    return {
        "experiment": "balanced_threshold_multiplier_sweep",
        "reference_file": str(reference_path),
        "reference_scope": "schema-v2 dtype-specific stored coordinates; MP subtraction",
        "reference_decimal_precision": reference_precision,
        "interval": list(INTERVAL),
        "quadrature_points": quadrature_points,
        "orders": list(ORDERS),
        "continuation_order": 2,
        "multipliers": list(MULTIPLIERS),
        "records": records,
        "summary": summary,
    }


def plot(payload: dict[str, object], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        }
    )
    summary = payload["summary"]
    assert isinstance(summary, list)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=True)
    colors = {2: "#0072B2", 4: "#E69F00", 6: "#CC79A7"}
    markers = {2: "o", 4: "s", 6: "^"}
    for axis, dtype_name in zip(axes, ("float32", "float64")):
        for order in ORDERS:
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["dtype"] == dtype_name and row["order"] == order
                ),
                key=lambda row: float(row["multiplier"]),
            )
            axis.loglog(
                [float(row["multiplier"]) for row in rows],
                [float(row["absolute_error_rmse"]) for row in rows],
                marker=markers[order],
                color=colors[order],
                label=f"m={order}",
            )
        axis.axvline(1.0, color="0.45", linestyle="--", linewidth=1.0)
        axis.set_title(dtype_name)
        axis.set_xlabel("threshold multiplier c")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("operator absolute-error RMSE")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{suffix}"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/general_taylor_multiplier_sweep"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args.reference_metrics, args.quadrature_points)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(payload, args.output_dir / "threshold_multiplier_sensitivity")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
