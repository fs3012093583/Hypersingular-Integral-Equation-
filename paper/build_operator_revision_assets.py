"""Build v2 operator tables and figures from schema-v2 reference-chain runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_TABLES = ROOT / "paper" / "tables_operator_v2"
DEFAULT_FIGURES = ROOT / "paper" / "figures"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 16, "axes.labelsize": 22,
        "axes.titlesize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20,
        "legend.fontsize": 14, "axes.linewidth": 1.2, "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required metrics: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"metrics root must be an object: {path}")
    return payload


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing numeric {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}")
    return result


def require_schema2(payload: dict[str, Any], name: str) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError(f"{name} must be schema_version=2")
    reference = payload.get("reference")
    if not isinstance(reference, dict) or reference.get("coordinate_scope") != "per-dtype stored torch coordinates":
        raise ValueError(f"{name} lacks the dtype-stored reference scope")


def records(payload: dict[str, Any], field: str, name: str) -> list[dict[str, Any]]:
    result = payload.get(field)
    if not isinstance(result, list) or not result:
        raise ValueError(f"{name} lacks nonempty {field}")
    rows = [row for row in result if isinstance(row, dict)]
    if len(rows) != len(result):
        raise ValueError(f"{name} has malformed {field}")
    for row in rows:
        if not row.get("nonfinite") and "absolute_error" in row and row["absolute_error"] is not None:
            finite(row["absolute_error"], f"{name}.absolute_error")
    return rows


def rmse(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("cannot aggregate an empty row set")
    if any(row.get("nonfinite") or row.get("absolute_error") is None for row in rows):
        raise ValueError("requested numerical aggregate contains non-finite records")
    return float(np.sqrt(np.mean([finite(row["absolute_error"], "absolute_error") ** 2 for row in rows])))


def rmse_or_nonfinite(rows: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    bad = sum(bool(row.get("nonfinite")) or row.get("absolute_error") is None for row in rows)
    return (None, bad, len(rows)) if bad else (rmse(rows), 0, len(rows))


def select(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]


def _fifteen_unique(rows: list[dict[str, Any]], metric: str, context: str) -> list[float]:
    if len(rows) != 15:
        raise ValueError(f"{context} requires 15 function-position records, got {len(rows)}")
    identities = {(row.get("function"), row.get("t_stored")) for row in rows}
    if len(identities) != 15:
        raise ValueError(f"{context} has duplicate function-position records")
    if any(row.get("nonfinite") or row.get(metric) is None for row in rows):
        raise ValueError(f"{context} contains non-finite values")
    return [finite(row[metric], metric) for row in rows]


def quotient_median_curves(component_quotient: list[dict[str, Any]], base_quotient: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    """m=6 float64 quotient curves: median scaled error over 15 samples."""
    sources = ((component_quotient, "method", "direct_only", "direct"), (component_quotient, "method", "taylor_only_p2", "global local p=2"), (component_quotient, "method", "balanced_hybrid_p2", "hybrid p=2"), (base_quotient, "strategy", "balanced_p0", "hybrid p=0"), (base_quotient, "strategy", "balanced_p1", "hybrid p=1"), (base_quotient, "strategy", "balanced_p3", "hybrid p=3"))
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for source, selector, method, label in sources:
        rows = [row for row in source if row.get("dtype") == "float64" and row.get("order") == 6 and row.get(selector) == method and not row.get("skipped")]
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: grouped[float(row["requested_delta"])].append(row)
        if len(grouped) < 2: raise ValueError(f"{label} lacks local-distance coverage")
        distances = sorted(grouped)
        curves[label] = (distances, [float(np.median(_fifteen_unique(grouped[distance], "scaled_error", f"{label}, |h|={distance}"))) for distance in distances])
    return curves


def component_rmse_curves(component_operator: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[float]]]:
    """Float64 component-operator RMSE curves, never median error curves."""
    result: dict[str, tuple[list[int], list[float]]] = {}
    for method, label in (("direct_only", "direct"), ("taylor_only_p2", "global local p=2"), ("balanced_hybrid_p2", "hybrid p=2")):
        values = []
        for order in range(1, 7):
            rows = [row for row in component_operator if row.get("dtype") == "float64" and row.get("order") == order and row.get("method") == method]
            errors = _fifteen_unique(rows, "absolute_error", f"{label}, m={order}")
            values.append(float(np.sqrt(np.mean(np.square(errors)))))
        result[label] = (list(range(1, 7)), values)
    return result


def tex_number(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def table_operator(base: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    lines = ["\\begin{table}[H]", "\\centering", "\\caption{256 点求积下算子绝对误差 RMSE\\\\Operator absolute-error RMSE with 256-point quadrature}", "\\label{tab:operator}", "\\small", "\\begin{tabular}{ccccc}", "\\toprule", "\\multirow{2}{*}{$m$} & \\multicolumn{2}{c}{float32} & \\multicolumn{2}{c}{float64}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "& 直接差商 & 平衡延拓 $p=2$ & 直接差商 & 平衡延拓 $p=2$\\\\", "\\midrule"]
    summary: dict[str, Any] = {}
    for order in range(1, 7):
        values: list[float | None] = []
        nonfinite: list[tuple[int, int]] = []
        for dtype, strategy in (("float32", "direct"), ("float32", "balanced_p2"), ("float64", "direct"), ("float64", "balanced_p2")):
            rows = select(base, dtype=dtype, order=order, strategy=strategy, quadrature_size=256)
            value, bad, total = rmse_or_nonfinite(rows); values.append(value); nonfinite.append((bad, total))
        summary[str(order)] = dict(zip(("direct_f32", "hybrid_f32", "direct_f64", "hybrid_f64"), values))
        cells = [tex_number(value) if value is not None else f"\\text{{nonfinite}} ({bad}/{total})" for value, (bad, total) in zip(values, nonfinite)]
        lines.append(f"{order} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), summary


def table_component(summary_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    lines = ["\\begin{table}[H]", "\\centering", "\\caption{组成分支与混合公式的双精度算子 RMSE\\\\Float64 operator RMSE of component branches and hybrid}", "\\label{tab:component-ablation}", "\\small", "\\begin{tabular}{cccc}", "\\toprule", "$m$ & 全程直接差商 & 全程局部延拓 $p=2$ & 平衡混合\\\\", "\\midrule"]
    out: dict[str, Any] = {}
    for order in range(1, 7):
        values = []
        for method in ("direct_only", "taylor_only_p2", "balanced_hybrid_p2"):
            matches = [row for row in summary_rows if row.get("dtype") == "float64" and row.get("order") == order and row.get("method") == method]
            if len(matches) != 1:
                raise ValueError(f"component summary incomplete for m={order}, {method}")
            values.append(finite(matches[0]["absolute_error_rmse"], "component RMSE"))
        out[str(order)] = dict(zip(("direct", "local", "hybrid"), values))
        lines.append(f"{order} & " + " & ".join(tex_number(value) for value in values) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), out


def table_p(base: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    lines = ["\\begin{table}[H]", "\\centering", "\\caption{不同延拓阶数的双精度算子误差（$N_q=256$）\\\\Float64 operator error for continuation orders ($N_q=256$)}", "\\label{tab:p-ablation}", "\\small", "\\begin{tabular}{ccccc}", "\\toprule", "$m$ & $p=0$ & $p=1$ & $p=2$ & $p=3$\\\\", "\\midrule"]
    out: dict[str, Any] = {}
    for order in range(3, 7):
        values = [rmse(select(base, dtype="float64", order=order, strategy=f"balanced_p{p}", quadrature_size=256)) for p in range(4)]
        out[str(order)] = values
        lines.append(f"{order} & " + " & ".join(tex_number(value) for value in values) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), out


def table_adaptive(summary_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    lines = ["\\begin{table}[H]", "\\centering", "\\caption{固定延拓阶数与局部自适应阶数的双精度算子误差\\\\Float64 operator error of fixed and locally adaptive continuation orders}", "\\label{tab:adaptive-p}", "\\small", "\\begin{tabular}{cccc}", "\\toprule", "$m$ & 事后最佳固定 $p$ & 实用局部选择（$\\rho=2$） & 解析导数界选择\\\\", "\\midrule"]
    methods = ("adaptive_cost_aware", "bounded_remainder_cost_aware")
    out: dict[str, Any] = {}
    for order in range(1, 7):
        fixed = [(int(str(row["method"])[-1]), finite(row["absolute_error_rmse"], "adaptive fixed RMSE")) for row in summary_rows if row.get("dtype") == "float64" and row.get("order") == order and str(row.get("method", "")).startswith("balanced_p")]
        adaptive = []
        for method in methods:
            matches = [row for row in summary_rows if row.get("dtype") == "float64" and row.get("order") == order and row.get("method") == method]
            if len(matches) != 1:
                raise ValueError(f"adaptive summary incomplete for m={order}, {method}")
            adaptive.append(finite(matches[0]["absolute_error_rmse"], "adaptive RMSE"))
        if len(fixed) != 4:
            raise ValueError(f"adaptive fixed-p summary incomplete for m={order}")
        best_p, best = min(fixed, key=lambda item: item[1])
        out[str(order)] = {"best_fixed_p": best_p, "best_fixed": best, "practical": adaptive[0], "bounded": adaptive[1]}
        lines.append(f"{order} & $p={best_p}$: {tex_number(best)} & {tex_number(adaptive[0])} & {tex_number(adaptive[1])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), out


def table_c(summary_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    lines = ["\\begin{table}[H]", "\\centering", "\\caption{阈值系数 $c$ 的灵敏度：双精度算子 RMSE\\\\Sensitivity to multiplier $c$: float64 operator RMSE}", "\\label{tab:c-sweep}", "\\small", "\\begin{tabular}{cccccc}", "\\toprule", "$m$ & $c=.25$ & $c=.5$ & $c=1$ & $c=2$ & $c=4$\\\\", "\\midrule"]
    out: dict[str, Any] = {}
    for order in (2, 4, 6):
        values = []
        for multiplier in multipliers:
            matches = [row for row in summary_rows if row.get("dtype") == "float64" and row.get("order") == order and float(row.get("multiplier")) == multiplier]
            if len(matches) != 1:
                raise ValueError(f"multiplier summary incomplete for m={order}, c={multiplier}")
            values.append(finite(matches[0]["absolute_error_rmse"], "multiplier RMSE"))
        out[str(order)] = dict(zip(map(str, multipliers), values))
        lines.append(f"{order} & " + " & ".join(tex_number(value) for value in values) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), out


def save(fig: plt.Figure, path: Path) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)


def figures(base: list[dict[str, Any]], base_quotient: list[dict[str, Any]], component: list[dict[str, Any]], component_quotient: list[dict[str, Any]], output: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
    styles = {"direct": ("#D55E00", "o", "--"), "global local p=2": ("#009E73", "s", ":"), "hybrid p=2": ("#0072B2", "^", "-"), "hybrid p=0": ("#E69F00", "D", "-."), "hybrid p=1": ("#56B4E9", "P", (0, (5, 2))), "hybrid p=3": ("#6A4C93", "v", (0, (1, 1)))}
    for label, (distance, error) in quotient_median_curves(component_quotient, base_quotient).items():
        color, marker, line = styles[label]
        axes[0].loglog(distance, np.maximum(error, 1.0e-30), marker=marker, linestyle=line, color=color, label=label)
    for label, (orders, error) in component_rmse_curves(component).items():
        color, marker, line = styles[label]
        axes[1].semilogy(orders, error, marker=marker, linestyle=line, color=color, label=label)
    axes[0].invert_xaxis(); axes[0].set_xlabel(r"near-diagonal distance $|h|$", fontweight="bold"); axes[0].set_ylabel("median scaled quotient error", fontweight="bold"); axes[0].set_title("(a) Sixth-order local quotient, float64", fontweight="bold")
    axes[1].set_xlabel("kernel order m", fontweight="bold"); axes[1].set_ylabel("operator absolute-error RMSE", fontweight="bold"); axes[1].set_title("(b) 256-point finite-part operator, float64", fontweight="bold"); axes[1].set_xticks(range(1, 7))
    for axis in axes: axis.grid(True, which="both", alpha=.25); axis.legend(frameon=False)
    fig.tight_layout(); save(fig, output / "direct_local_hybrid_ablation_v2"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    for axis, order in zip(axes, (2, 4, 6)):
        for strategy, label, color, marker, line in (("direct", "direct", "#D55E00", "o", "--"), ("fixed_5e-03_p1", "fixed p=1", "#777777", "s", ":"), ("balanced_p2", "hybrid p=2", "#0072B2", "^", "-")):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in select(base, dtype="float64", order=order, strategy=strategy): grouped[int(row["quadrature_size"])].append(finite(row["absolute_error"], "quadrature error"))
            sizes = sorted(grouped)
            axis.loglog(sizes, [rmse([{"absolute_error": value} for value in grouped[size]]) for size in sizes], marker=marker, linestyle=line, color=color, label=label)
        axis.set_title(f"m={order}", fontweight="bold"); axis.set_xlabel(r"quadrature points $N_q$", fontweight="bold"); axis.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("operator absolute-error RMSE", fontweight="bold"); axes[-1].legend(frameon=False); fig.tight_layout(); save(fig, output / "operator_quadrature_convergence_v2"); plt.close(fig)


def build(base_path: Path, multiplier_path: Path, component_path: Path, adaptive_path: Path, tables: Path, figures_dir: Path) -> dict[str, Any]:
    base_payload, multiplier_payload, component_payload, adaptive_payload = map(load, (base_path, multiplier_path, component_path, adaptive_path))
    require_schema2(base_payload, "operator benchmark")
    for name, payload in (("multiplier", multiplier_payload), ("adaptive", adaptive_payload)):
        if not str(payload.get("reference_scope", "")).startswith("schema-v2"):
            raise ValueError(f"{name} output is not derived from schema-v2 references")
    # Component ablation records carry the required reference_decimal values,
    # while its historical output schema names only the source benchmark.
    if "dtype_ref_v2" not in str(component_payload.get("source_benchmark", "")):
        raise ValueError("component output is not derived from the schema-v2 source benchmark")
    base = records(base_payload, "operator_records", "operator benchmark")
    quotient = records(base_payload, "quotient_records", "operator benchmark")
    component = records(component_payload, "operator_records", "component")
    multiplier_summary = records(multiplier_payload, "summary", "multiplier")
    component_summary = records(component_payload, "operator_summary", "component")
    adaptive_summary = records(adaptive_payload, "summary", "adaptive")
    tables.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"operator": table_operator(base), "component-ablation": table_component(component_summary), "p-ablation": table_p(base), "adaptive-p": table_adaptive(adaptive_summary), "c-sweep": table_c(multiplier_summary)}
    filenames = {"operator": "operator.tex", "component-ablation": "component_ablation.tex", "p-ablation": "p_ablation.tex", "adaptive-p": "adaptive_p.tex", "c-sweep": "c_sweep.tex"}
    summary = {key: data for key, (_, data) in outputs.items()}
    summary["nonfinite_counts"] = {
        "operator_records_total": sum(bool(row.get("nonfinite")) for row in base),
        "quotient_records_total": sum(bool(row.get("nonfinite")) for row in quotient),
        "operator_records_by_dtype_strategy": {
            f"{dtype}/{strategy}": sum(bool(row.get("nonfinite")) for row in base if row.get("dtype") == dtype and row.get("strategy") == strategy)
            for dtype in ("float32", "float64") for strategy in ("direct", "balanced_p0", "balanced_p1", "balanced_p2", "balanced_p3")
        },
        "quotient_records_by_dtype_strategy": {
            f"{dtype}/{strategy}": sum(bool(row.get("nonfinite")) for row in quotient if row.get("dtype") == dtype and row.get("strategy") == strategy)
            for dtype in ("float32", "float64") for strategy in ("direct", "balanced_p0", "balanced_p1", "balanced_p2", "balanced_p3")
        },
    }
    for key, (text, _) in outputs.items(): (tables / filenames[key]).write_text(text, encoding="utf-8")
    component_quotient = records(component_payload, "quotient_records", "component")
    figures(base, quotient, component, component_quotient, figures_dir)
    (tables / "operator_revision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_RESULTS / "general_taylor_operator_benchmark_full_dtype_ref_v2" / "metrics.json")
    parser.add_argument("--multiplier", type=Path, default=DEFAULT_RESULTS / "general_taylor_multiplier_sweep_dtype_ref_v2" / "metrics.json")
    parser.add_argument("--component", type=Path, default=DEFAULT_RESULTS / "general_taylor_component_ablation_dtype_ref_v2" / "metrics.json")
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_RESULTS / "general_taylor_adaptive_order_dtype_ref_v2" / "metrics.json")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args(); build(args.operator, args.multiplier, args.component, args.adaptive, args.tables_dir, args.figures_dir)


if __name__ == "__main__": main()
