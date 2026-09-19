"""Three-way ablation of the direct, local, and balanced Taylor quotients.

This experiment separates the two numerical representations used by the
hybrid finite-part operator:

1. ``direct_only`` evaluates the regularized quotient by direct subtraction
   away from exact coincidence;
2. ``taylor_only_p2`` uses the degree-two local continuation over the whole
   integration interval;
3. ``balanced_hybrid_p2`` switches between them with the proposed
   order/precision-dependent threshold.

The high-precision references and the direct/hybrid records are read from the
full operator benchmark so this ablation uses exactly the same functions,
target points, floating-point formats, and quadrature rules as the main paper.
"""

from __future__ import annotations

import argparse
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

from experiments.general_taylor_operator_benchmark import FUNCTIONS, INTERVAL, _gauss_rule
from neural_network_solvers.general_taylor_finite_part import (
    regularized_taylor_quotient,
    taylor_finite_part_operator,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "general_taylor_operator_benchmark_full" / "metrics.json"
DEFAULT_OUTPUT = ROOT / "results" / "general_taylor_component_ablation"
METHODS = ("direct_only", "taylor_only_p2", "balanced_hybrid_p2")
COLORS = {
    "direct_only": "#D55E00",
    "taylor_only_p2": "#009E73",
    "balanced_hybrid_p2": "#0072B2",
}
LABELS = {
    "direct_only": "direct only, Eq. (2.3)",
    "taylor_only_p2": "local Taylor only, Eq. (2.5)",
    "balanced_hybrid_p2": "balanced hybrid, Eq. (2.8)",
}


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def _function_map() -> dict[str, object]:
    return {spec.name: spec.torch_function for spec in FUNCTIONS}


def _reference_precision(payload: dict[str, object]) -> int:
    if payload.get("schema_version") != 2:
        raise ValueError("component ablation requires schema_version=2 references")
    metadata = payload.get("reference")
    if not isinstance(metadata, dict):
        raise ValueError("component ablation reference metadata is missing")
    try:
        precision = int(metadata["decimal_precision"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("component ablation reference precision is missing") from error
    if precision < 40:
        raise ValueError("component ablation reference precision is too low")
    return precision


def _reference_decimal(row: dict[str, object]) -> str:
    try:
        value = row["reference_decimal"]
    except KeyError as error:
        raise ValueError("component ablation requires reference_decimal records") from error
    if not isinstance(value, str) or not value:
        raise ValueError("component ablation reference_decimal must be non-empty")
    return value


def _errors(
    value: float, reference_decimal: str, precision: int
) -> tuple[float | None, float | None]:
    if not math.isfinite(value):
        return None, None
    with mp.workdps(precision):
        reference = mp.mpf(reference_decimal)
        absolute = abs(mp.mpf(float(value)) - reference)
        return float(absolute), float(absolute / max(abs(reference), mp.mpf("1")))


def _copy_record(row: dict[str, object], method: str) -> dict[str, object]:
    copied = dict(row)
    copied["method"] = method
    copied.pop("strategy", None)
    return copied


def build_quotient_records(payload: dict[str, object]) -> list[dict[str, object]]:
    precision = _reference_precision(payload)
    functions = _function_map()
    source = payload["quotient_records"]
    assert isinstance(source, list)
    for row in source:
        if row.get("strategy") in ("direct", "balanced_p2"):
            for field in ("t_stored", "tau_stored"):
                if field not in row:
                    raise ValueError(
                        f"component ablation requires schema-v2 {field} quotient records"
                    )
            _reference_decimal(row)
    records: list[dict[str, object]] = []
    direct_rows = [
        row
        for row in source
        if row.get("strategy") == "direct" and not row.get("skipped", False)
    ]
    hybrid_index = {
        (
            row["function"],
            row["order"],
            row["t_stored"],
            row["tau_stored"],
            row["dtype"],
            row["requested_delta"],
        ): row
        for row in source
        if row.get("strategy") == "balanced_p2" and not row.get("skipped", False)
    }
    for direct in direct_rows:
        key = (
            direct["function"],
            direct["order"],
            direct["t_stored"],
            direct["tau_stored"],
            direct["dtype"],
            direct["requested_delta"],
        )
        records.append(_copy_record(direct, "direct_only"))
        records.append(_copy_record(hybrid_index[key], "balanced_hybrid_p2"))

        dtype = _dtype(str(direct["dtype"]))
        t = torch.tensor([[float(direct["t_stored"])]], dtype=dtype)
        tau = torch.tensor([[float(direct["tau_stored"])]], dtype=dtype)
        actual_delta = float((tau - t).item())
        quotient, _ = regularized_taylor_quotient(
            functions[str(direct["function"])],
            t,
            tau,
            order=int(direct["order"]),
            continuation_order=2,
            threshold_mode="fixed",
            # Every source/target pair on [-1,1] satisfies |tau-t| <= 2.
            near_diagonal_threshold=INTERVAL[1] - INTERVAL[0],
            interval_length=INTERVAL[1] - INTERVAL[0],
        )
        value = float(quotient.item())
        reference_decimal = _reference_decimal(direct)
        absolute, scaled = _errors(value, reference_decimal, precision)
        records.append(
            {
                "function": direct["function"],
                "order": direct["order"],
                "t": direct["t"],
                "t_stored": float(t.item()),
                "tau_stored": float(tau.item()),
                "dtype": direct["dtype"],
                "requested_delta": direct["requested_delta"],
                "actual_delta": actual_delta,
                "method": "taylor_only_p2",
                "value": value if math.isfinite(value) else None,
                "reference": float(mp.mpf(reference_decimal)),
                "reference_decimal": reference_decimal,
                "absolute_error": absolute,
                "scaled_error": scaled,
                "nonfinite": not math.isfinite(value),
                "skipped": False,
            }
        )
    return records


def build_operator_records(
    payload: dict[str, object], *, quadrature_size: int
) -> list[dict[str, object]]:
    precision = _reference_precision(payload)
    functions = _function_map()
    source = payload["operator_records"]
    assert isinstance(source, list)
    for row in source:
        if row.get("strategy") in ("direct", "balanced_p2"):
            if "t_stored" not in row:
                raise ValueError(
                    "component ablation requires schema-v2 t_stored operator records"
                )
            _reference_decimal(row)
    records: list[dict[str, object]] = []
    direct_rows = [
        row
        for row in source
        if row.get("strategy") == "direct" and int(row["quadrature_size"]) == quadrature_size
    ]
    hybrid_index = {
        (row["function"], row["order"], row["t_stored"], row["dtype"]): row
        for row in source
        if row.get("strategy") == "balanced_p2"
        and int(row["quadrature_size"]) == quadrature_size
    }
    rule_cache = {
        name: _gauss_rule(quadrature_size, _dtype(name)) for name in ("float32", "float64")
    }
    for direct in direct_rows:
        key = (
            direct["function"], direct["order"], direct["t_stored"], direct["dtype"]
        )
        records.append(_copy_record(direct, "direct_only"))
        records.append(_copy_record(hybrid_index[key], "balanced_hybrid_p2"))

        dtype_name = str(direct["dtype"])
        dtype = _dtype(dtype_name)
        nodes, weights = rule_cache[dtype_name]
        t = torch.tensor([[float(direct["t_stored"])]], dtype=dtype)
        value_tensor = taylor_finite_part_operator(
            functions[str(direct["function"])],
            t,
            nodes,
            weights,
            order=int(direct["order"]),
            interval=INTERVAL,
            continuation_order=2,
            threshold_mode="fixed",
            near_diagonal_threshold=INTERVAL[1] - INTERVAL[0],
        )
        value = float(value_tensor.item())
        reference_decimal = _reference_decimal(direct)
        absolute, scaled = _errors(value, reference_decimal, precision)
        records.append(
            {
                "function": direct["function"],
                "order": direct["order"],
                "t": direct["t"],
                "t_stored": float(t.item()),
                "dtype": dtype_name,
                "quadrature_size": quadrature_size,
                "method": "taylor_only_p2",
                "value": value if math.isfinite(value) else None,
                "reference": float(mp.mpf(reference_decimal)),
                "reference_decimal": reference_decimal,
                "absolute_error": absolute,
                "scaled_error": scaled,
                "nonfinite": not math.isfinite(value),
            }
        )
    return records


def summarize_operator(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    nonfinite: dict[tuple[str, int, str], int] = defaultdict(int)
    counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in records:
        key = (str(row["dtype"]), int(row["order"]), str(row["method"]))
        counts[key] += 1
        if row.get("absolute_error") is None:
            nonfinite[key] += 1
        else:
            grouped[key].append(float(row["absolute_error"]))
    summary: list[dict[str, object]] = []
    for key in sorted(counts):
        dtype, order, method = key
        values = np.asarray(grouped[key], dtype=float)
        summary.append(
            {
                "dtype": dtype,
                "order": order,
                "method": method,
                "sample_count": counts[key],
                "nonfinite_count": nonfinite[key],
                "absolute_error_rmse": float(np.sqrt(np.mean(values**2))) if values.size else None,
                "absolute_error_median": float(np.median(values)) if values.size else None,
                "absolute_error_max": float(np.max(values)) if values.size else None,
            }
        )
    return summary


def plot_ablation(
    quotient: list[dict[str, object]],
    operator_summary: list[dict[str, object]],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    styles = {
        "direct_only": ("o", "--"),
        "taylor_only_p2": ("s", ":"),
        "balanced_hybrid_p2": ("^", "-"),
    }
    for method in METHODS:
        subset = [
            row
            for row in quotient
            if row["dtype"] == "float64"
            and int(row["order"]) == 6
            and row["method"] == method
            and row.get("scaled_error") is not None
        ]
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in subset:
            grouped[float(row["requested_delta"])].append(float(row["scaled_error"]))
        x = sorted(grouped)
        y = [float(np.median(grouped[value])) for value in x]
        marker, linestyle = styles[method]
        axes[0].loglog(
            x,
            np.maximum(y, 1.0e-18),
            marker=marker,
            linestyle=linestyle,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"distance $|\tau-t|$")
    axes[0].set_ylabel("median scaled quotient error")
    axes[0].set_title("(a) Sixth-order quotient, float64")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    for method in METHODS:
        subset = [
            row
            for row in operator_summary
            if row["dtype"] == "float64" and row["method"] == method
        ]
        subset.sort(key=lambda row: int(row["order"]))
        marker, linestyle = styles[method]
        axes[1].semilogy(
            [int(row["order"]) for row in subset],
            [float(row["absolute_error_rmse"]) for row in subset],
            marker=marker,
            linestyle=linestyle,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[1].set_xlabel("kernel order $m$")
    axes[1].set_ylabel("operator absolute-error RMSE")
    axes[1].set_title("(b) 256-point finite-part operator")
    axes[1].set_xticks(range(1, 7))
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"direct_local_hybrid_ablation.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quadrature-size", type=int, default=256)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    quotient = build_quotient_records(payload)
    operator = build_operator_records(payload, quadrature_size=args.quadrature_size)
    summary = summarize_operator(operator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "general_taylor_component_ablation",
        "source_benchmark": str(args.source),
        "interpretation": {
            "direct_only": "Eq. (2.3) away from exact coincidence",
            "taylor_only_p2": "Eq. (2.5), p=2, over the complete interval",
            "balanced_hybrid_p2": "Eq. (2.8), combining Eqs. (2.3), (2.5), and (2.7)",
        },
        "quadrature_size": args.quadrature_size,
        "quotient_records": quotient,
        "operator_records": operator,
        "operator_summary": summary,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_ablation(quotient, summary, args.output_dir)
    print(
        json.dumps(
            {
                "metrics": str(args.output_dir / "metrics.json"),
                "quotient_records": len(quotient),
                "operator_records": len(operator),
                "operator_summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
