"""Measure high-order tanh-network AD error against 100-digit Taylor jets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.general_taylor_neural_benchmark import TanhNetwork, set_seed
from neural_network_solvers.general_taylor_finite_part import value_and_derivatives
from neural_network_solvers.high_precision_tanh_reference import high_precision_tanh_derivatives


def _copy_model(model: TanhNetwork, dtype: torch.dtype) -> TanhNetwork:
    copied = TanhNetwork(
        width=model.layers[0].out_features,
        hidden_layers=sum(isinstance(layer, torch.nn.Tanh) for layer in model.layers),
    ).to(dtype=dtype)
    copied.load_state_dict({name: value.to(dtype=dtype) for name, value in model.state_dict().items()})
    return copied


def reference_errors(
    observed: float, reference: mp.mpf, decimal_digits: int
) -> tuple[float, float, str]:
    """Subtract before binary64 serialization, including sub-ulp errors."""
    with mp.workdps(decimal_digits):
        error = abs(mp.mpf(observed) - reference)
        return float(error), float(error / max(abs(reference), mp.mpf(1))), str(reference)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dtype"]), int(row["order"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dtype, order), group in sorted(groups.items()):
        absolute = np.asarray([float(row["absolute_error"]) for row in group])
        scaled = np.asarray([float(row["scaled_error"]) for row in group])
        magnitude = np.asarray([float(row["reference_magnitude"]) for row in group])
        summary.append(
            {
                "dtype": dtype,
                "order": order,
                "sample_count": len(group),
                "median_absolute_error": float(np.median(absolute)),
                "rmse_absolute_error": float(np.sqrt(np.mean(absolute ** 2))),
                "maximum_absolute_error": float(np.max(absolute)),
                "median_scaled_error": float(np.median(scaled)),
                "rmse_scaled_error": float(np.sqrt(np.mean(scaled ** 2))),
                "maximum_scaled_error": float(np.max(scaled)),
                "median_reference_magnitude": float(np.median(magnitude)),
                "maximum_reference_magnitude": float(np.max(magnitude)),
            }
        )
    return summary


def run(
    *,
    seeds: list[int],
    points: list[float],
    highest_order: int,
    width: int,
    hidden_layers: int,
    decimal_digits: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        master = TanhNetwork(width=width, hidden_layers=hidden_layers).to(dtype=torch.float64)
        for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
            model = _copy_model(master, dtype)
            for point in points:
                input_point = torch.tensor([[point]], dtype=dtype, requires_grad=True)
                observed = value_and_derivatives(model, input_point, highest_order)
                reference = high_precision_tanh_derivatives(
                    model,
                    float(input_point.detach().cpu().item()),
                    highest_order,
                    decimal_digits=decimal_digits,
                )
                for order in range(highest_order + 1):
                    observed_value = float(observed[order].detach().cpu().item())
                    reference_value = float(reference[order])
                    absolute_error, scaled_error, reference_decimal = reference_errors(
                        observed_value, reference[order], decimal_digits
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "dtype": dtype_name,
                            "point": point,
                            "quantized_point": float(input_point.detach().cpu().item()),
                            "order": order,
                            "observed": observed_value,
                            "reference": reference_value,
                            "reference_decimal": reference_decimal,
                            "reference_magnitude": abs(reference_value),
                            "absolute_error": absolute_error,
                            "scaled_error": scaled_error,
                        }
                    )
    summary = _summarize(rows)
    eta_envelope = {
        dtype: {
            str(item["order"]): float(item["maximum_scaled_error"])
            for item in summary
            if item["dtype"] == dtype
        }
        for dtype in ("float32", "float64")
    }
    derivative_magnitude_envelope = {
        dtype: {
            str(item["order"]): float(item["maximum_reference_magnitude"])
            for item in summary
            if item["dtype"] == dtype
        }
        for dtype in ("float32", "float64")
    }
    return {
        "schema_version": 2,
        "configuration": {
            "seeds": seeds,
            "points": points,
            "highest_order": highest_order,
            "width": width,
            "hidden_layers": hidden_layers,
            "decimal_digits": decimal_digits,
            "scaled_error_definition": "abs(observed-reference)/max(abs(reference),1)",
            "reference_parameter_policy": "each dtype uses its own stored quantized parameters",
            "reference_coordinate_policy": "each dtype uses its actual stored input coordinate",
            "reference_error_policy": "mpmath subtraction and scaling before float serialization",
        },
        "eta_scaled_envelope": eta_envelope,
        "derivative_magnitude_envelope": derivative_magnitude_envelope,
        "summary": summary,
        "rows": rows,
    }


def _plot(summary: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = {"float32": "#d95f8d", "float64": "#009e73"}
    for dtype in ("float32", "float64"):
        selected = [row for row in summary if row["dtype"] == dtype]
        orders = np.asarray([int(row["order"]) for row in selected])
        median = np.maximum([float(row["median_scaled_error"]) for row in selected], 1.0e-20)
        maximum = np.maximum([float(row["maximum_scaled_error"]) for row in selected], 1.0e-20)
        axes[0].plot(orders, median, marker="o", color=colors[dtype], label=f"{dtype}, median")
        axes[0].plot(orders, maximum, marker="s", linestyle="--", color=colors[dtype], label=f"{dtype}, max")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("derivative order k")
    axes[0].set_ylabel("scaled AD error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    selected = [row for row in summary if row["dtype"] == "float64"]
    orders = np.asarray([int(row["order"]) for row in selected])
    median = np.maximum([float(row["median_reference_magnitude"]) for row in selected], 1.0e-20)
    maximum = np.maximum([float(row["maximum_reference_magnitude"]) for row in selected], 1.0e-20)
    axes[1].plot(orders, median, marker="o", color="#0072b2", label="median")
    axes[1].plot(orders, maximum, marker="s", linestyle="--", color="#e69f00", label="max")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("derivative order k")
    axes[1].set_ylabel("reference derivative magnitude")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/neural_high_order_derivatives"))
    parser.add_argument("--highest-order", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--decimal-digits", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260918, 20260919, 20260920])
    parser.add_argument("--points", type=float, nargs="+", default=[-0.9, -0.5, 0.0, 0.5, 0.9])
    args = parser.parse_args()
    if args.highest_order < 1:
        raise ValueError("highest-order must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run(
        seeds=args.seeds,
        points=args.points,
        highest_order=args.highest_order,
        width=args.width,
        hidden_layers=args.hidden_layers,
        decimal_digits=args.decimal_digits,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(results["rows"])
    _plot(results["summary"], args.output_dir / "derivative_error.png")
    print(json.dumps({"eta_scaled_envelope": results["eta_scaled_envelope"], "derivative_magnitude_envelope": results["derivative_magnitude_envelope"], "summary": results["summary"]}, indent=2))


if __name__ == "__main__":
    main()
