"""Build reproducible LaTeX quality-audit tables from verified JSON metrics.

This module intentionally uses only the Python standard library.  It is a
reporting boundary: it validates the paired experimental controls before
formatting any table and never imports benchmark, neural-network, or plotting
code.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = ("fixed_p2", "lazy_fixed_p2", "operator_budget")
METHOD_LABELS = {
    "fixed_p2": "全行固定 $p=2$",
    "lazy_fixed_p2": "按需固定 $p=2$",
    "operator_budget": "算子预算",
}
QUALITY_METRICS = (
    "solution_rmse",
    "method_setup_and_training_seconds",
    "saved_tensor_payload_bytes",
)
OPERATOR_STRATEGY = "operator_budget_1e-04_pmax3"


class ValidationError(ValueError):
    """Raised when a table input cannot support the claimed comparison."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"missing metrics file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"metrics root must be an object: {path}")
    return payload


def _finite_number(record: dict[str, Any], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"record lacks numeric {field!r}") from error
    if not math.isfinite(value):
        raise ValidationError(f"record has non-finite {field!r}")
    return value


def _quality_records(payload: dict[str, Any], task: str) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValidationError("quality metrics require a records list")
    selected = [record for record in records if isinstance(record, dict) and record.get("task") == task]
    if not selected:
        raise ValidationError(f"quality metrics contain no task={task!r} records")
    return selected


def _config_without(config: dict[str, Any], *excluded: str) -> dict[str, Any]:
    """Return the comparison projection of a run configuration."""
    excluded_fields = set(excluded)
    return {key: value for key, value in config.items() if key not in excluded_fields}


def _config_differences(reference: Any, candidate: Any, *, path: str = "config") -> list[str]:
    """Describe every recursive configuration difference, including absence.

    Configurations are JSON-shaped, but comparing mappings and sequences
    recursively makes an error name the actionable option (for example,
    ``config.calibration.path``), instead of presenting two opaque dicts.
    """
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences: list[str] = []
        keys = sorted(set(reference) | set(candidate), key=str)
        for key in keys:
            child_path = f"{path}.{key}"
            if key not in reference:
                differences.append(f"{child_path} (missing from reference)")
            elif key not in candidate:
                differences.append(f"{child_path} (missing from candidate)")
            else:
                differences.extend(
                    _config_differences(reference[key], candidate[key], path=child_path)
                )
        return differences
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        differences = []
        common_length = min(len(reference), len(candidate))
        for index in range(common_length):
            differences.extend(
                _config_differences(reference[index], candidate[index], path=f"{path}[{index}]")
            )
        for index in range(common_length, len(reference)):
            differences.append(f"{path}[{index}] (missing from candidate)")
        for index in range(common_length, len(candidate)):
            differences.append(f"{path}[{index}] (missing from reference)")
        return differences
    if type(reference) is not type(candidate) or reference != candidate:
        return [path]
    return []


def validate_and_aggregate_quality(
    payload: dict[str, Any], *, task: str, expected_orders: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Validate three-seed paired controls and aggregate the requested groups."""
    records = _quality_records(payload, task)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {
        order: {method: [] for method in METHODS} for order in expected_orders
    }
    for record in records:
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValidationError("quality record lacks config object")
        try:
            order = int(config["order"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("quality record lacks config.order") from error
        if order not in grouped:
            continue
        method = record.get("method")
        if method not in METHODS:
            raise ValidationError(f"unexpected method {method!r} for m={order}")
        if config.get("representation") not in (None, method):
            raise ValidationError(
                f"method/config representation mismatch for m={order}: {method!r}"
            )
        if record.get("status") != "ok":
            raise ValidationError(f"m={order}, method={method} has non-ok status")
        fingerprint = record.get("initial_model_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValidationError(f"m={order}, method={method} lacks fingerprint")
        try:
            int(config["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(f"m={order}, method={method} lacks config.seed") from error
        for metric in QUALITY_METRICS:
            _finite_number(record, metric)
        grouped[order][method].append(record)

    rows: list[dict[str, Any]] = []
    for order in expected_orders:
        by_method = grouped[order]
        seed_sets: dict[str, set[int]] = {}
        fingerprints_by_seed: dict[int, set[str]] = defaultdict(set)
        for method, group in by_method.items():
            if len(group) != 3:
                raise ValidationError(
                    f"incomplete comparison: m={order}, method={method} has "
                    f"{len(group)} records; expected three"
                )
            seeds = [int(record["config"]["seed"]) for record in group]
            if len(set(seeds)) != 3:
                raise ValidationError(
                    f"incomplete comparison: m={order}, method={method} needs "
                    "three distinct seeds"
                )
            seed_sets[method] = set(seeds)
            for record in group:
                fingerprints_by_seed[int(record["config"]["seed"])].add(
                    record["initial_model_fingerprint"]
                )
        reference_seeds = seed_sets[METHODS[0]]
        if any(seed_sets[method] != reference_seeds for method in METHODS[1:]):
            raise ValidationError(f"m={order} methods do not share the same seed set")
        for seed, fingerprints in fingerprints_by_seed.items():
            if len(fingerprints) != 1:
                raise ValidationError(
                    f"m={order}, seed={seed} has inconsistent initial fingerprints"
                )

        by_method_and_seed = {
            method: {int(record["config"]["seed"]): record for record in group}
            for method, group in by_method.items()
        }
        for seed in sorted(reference_seeds):
            reference_config = _config_without(
                by_method_and_seed[METHODS[0]][seed]["config"], "representation"
            )
            for method in METHODS[1:]:
                candidate_config = _config_without(
                    by_method_and_seed[method][seed]["config"], "representation"
                )
                differences = _config_differences(reference_config, candidate_config)
                if differences:
                    raise ValidationError(
                        f"m={order}, seed={seed} paired method configs disagree after "
                        "removing representation: "
                        + ", ".join(differences)
                    )

        canonical_seed = min(reference_seeds)
        cross_seed_reference = _config_without(
            by_method_and_seed[METHODS[0]][canonical_seed]["config"],
            "representation",
            "seed",
        )
        for seed in sorted(reference_seeds - {canonical_seed}):
            candidate_config = _config_without(
                by_method_and_seed[METHODS[0]][seed]["config"],
                "representation",
                "seed",
            )
            differences = _config_differences(cross_seed_reference, candidate_config)
            if differences:
                raise ValidationError(
                    f"m={order}, seed={seed} config differs from seed={canonical_seed} "
                    "after removing representation and seed: "
                    + ", ".join(differences)
                )
        for method in METHODS:
            group = by_method[method]
            row: dict[str, Any] = {
                "task": task,
                "order": order,
                "method": method,
                "samples": len(group),
                "seeds": tuple(sorted(seed_sets[method])),
            }
            for metric in QUALITY_METRICS:
                values = [_finite_number(record, metric) for record in group]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_sample_std"] = statistics.stdev(values)
            rows.append(row)
    return rows


def _format_mean_std(mean: float, std: float, *, scientific: bool = False) -> str:
    if scientific:
        scale = max(abs(mean), abs(std))
        exponent = math.floor(math.log10(scale)) if scale else 0
        factor = 10.0**exponent
        return f"$({mean / factor:.3f}\\pm{std / factor:.3f})\\times10^{{{exponent}}}$"
    return f"${mean:.2f}\\pm{std:.2f}$"


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    return f"${value / 10.0**exponent:.2f}\\times10^{{{exponent}}}$"


def _format_mib(byte_count: float) -> str:
    return f"{byte_count / (1024.0**2):.2f}"


def render_quality_table(rows: Iterable[dict[str, Any]], *, task: str) -> str:
    """Render the task-specific paired quality table."""
    rows = list(rows)
    show_order = task == "single"
    caption = (
        "三随机种子的解误差与计算成本（均值$\\pm$样本标准差，载荷为均值）\\\\"
        "Paired solution errors and costs over three seeds (mean$\\pm$sample standard deviation)."
        if show_order
        else "未见参数上的三随机种子结果（均值$\\pm$样本标准差，载荷为均值）\\\\"
        "Three-seed results at held-out parameters (mean$\\pm$sample standard deviation)."
    )
    label = "tab:operator-budget-seeds" if show_order else "tab:parameterized"
    columns = "clrrr" if show_order else "lrrr"
    header = (
        "$m$ & 方法 & 闭区间解 RMSE & 准备+训练 (s) & 保存张量 (MiB)"
        if show_order
        else "方法 & 闭区间解 RMSE & 准备+训练 (s) & 保存张量 (MiB)"
    )
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = []
        if show_order:
            cells.append(str(row["order"]))
        cells.extend(
            [
                METHOD_LABELS[row["method"]],
                _format_mean_std(
                    row["solution_rmse_mean"],
                    row["solution_rmse_sample_std"],
                    scientific=True,
                ),
                _format_mean_std(
                    row["method_setup_and_training_seconds_mean"],
                    row["method_setup_and_training_seconds_sample_std"],
                ),
                _format_mib(row["saved_tensor_payload_bytes_mean"]),
            ]
        )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _summary_number(record: dict[str, Any], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"operator summary lacks numeric {field!r}") from error
    if not math.isfinite(value):
        raise ValidationError(f"operator summary has non-finite {field!r}")
    return value


def _format_verified_fraction(value: float, samples: int, field: str) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"operator summary {field!r} must lie in [0, 1]")
    passed = round(value * samples)
    if not math.isclose(value, passed / samples, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValidationError(
            f"operator summary {field!r} is not a verified whole-sample fraction"
        )
    return f"{passed}/{samples}"


def validate_operator_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Select and validate the six v2 weighted-budget summary rows."""
    if payload.get("schema_version") != 2:
        raise ValidationError("operator metrics must use schema_version=2")
    summary = payload.get("summary")
    if not isinstance(summary, list):
        raise ValidationError("operator metrics require a summary list")
    expected = {(dtype, order) for dtype in ("float32", "float64") for order in (2, 4, 6)}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for item in summary:
        if not isinstance(item, dict) or item.get("strategy") != OPERATOR_STRATEGY:
            continue
        try:
            key = (str(item["dtype"]), int(item["order"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("operator summary lacks dtype/order") from error
        if key not in expected:
            continue
        if key in selected:
            raise ValidationError(f"duplicate operator summary row for {key}")
        if item.get("samples") != 15:
            raise ValidationError(f"operator summary {key} must contain 15 samples")
        for field in (
            "weighted_quotient_absolute_error_rmse",
            "moment_error_rmse",
            "representation_error_rmse",
            "quotient_indicator_coverage",
            "actual_quotient_budget_satisfaction",
            "actual_full_discrete_budget_satisfaction",
        ):
            _summary_number(item, field)
        selected[key] = item
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise ValidationError(f"incomplete operator summary rows: {missing}")
    return [selected[(dtype, order)] for dtype in ("float32", "float64") for order in (2, 4, 6)]


def render_operator_table(rows: Iterable[dict[str, Any]]) -> str:
    """Render the verified v2 operator-budget audit table."""
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{差商预算与完整离散误差的独立核验（每行 15 个样本）\\\\Independent audit of quotient budgets and full discrete errors (15 samples per row).}",
        "\\label{tab:budget-audit}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "精度 & $m$ & $E_Q$ RMSE & $E_M$ RMSE & $E_G$ RMSE & 指标覆盖 & $E_Q$ 达标 & $E_G$ 达标 \\\\",
        "\\midrule",
    ]
    for row in rows:
        samples = int(row["samples"])
        cells = [
            str(row["dtype"]),
            str(row["order"]),
            _format_scientific(_summary_number(row, "weighted_quotient_absolute_error_rmse")),
            _format_scientific(_summary_number(row, "moment_error_rmse")),
            _format_scientific(_summary_number(row, "representation_error_rmse")),
            _format_verified_fraction(
                _summary_number(row, "quotient_indicator_coverage"),
                samples,
                "quotient_indicator_coverage",
            ),
            _format_verified_fraction(
                _summary_number(row, "actual_quotient_budget_satisfaction"),
                samples,
                "actual_quotient_budget_satisfaction",
            ),
            _format_verified_fraction(
                _summary_number(row, "actual_full_discrete_budget_satisfaction"),
                samples,
                "actual_full_discrete_budget_satisfaction",
            ),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def lazy_baseline_comparisons(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return compact console summaries of lazy-vs-full fixed-p baselines."""
    by_group = {(row["task"], row["order"], row["method"]): row for row in rows}
    messages: list[str] = []
    for task, order in sorted({(row["task"], row["order"]) for row in rows}):
        fixed = by_group[(task, order, "fixed_p2")]
        lazy = by_group[(task, order, "lazy_fixed_p2")]
        time_reduction = 1.0 - (
            lazy["method_setup_and_training_seconds_mean"]
            / fixed["method_setup_and_training_seconds_mean"]
        )
        rmse_ratio = lazy["solution_rmse_mean"] / fixed["solution_rmse_mean"]
        messages.append(
            f"{task} m={order}: lazy-fixed-p2 time reduction vs full fixed-p2 "
            f"{time_reduction:.2%}; RMSE ratio {rmse_ratio:.4g}"
        )
        budget = by_group[(task, order, "operator_budget")]
        messages.append(
            f"{task} m={order}: operator-budget time reduction vs lazy-fixed-p2 "
            f"{1.0 - budget['method_setup_and_training_seconds_mean'] / lazy['method_setup_and_training_seconds_mean']:.2%}; "
            f"RMSE ratio {budget['solution_rmse_mean'] / lazy['solution_rmse_mean']:.4g}"
        )
    return messages


def build_tables(
    *, single_path: Path, parameterized_path: Path, operator_path: Path, output_dir: Path
) -> list[str]:
    """Validate inputs, write three tables, and return lazy-baseline messages."""
    single_rows = validate_and_aggregate_quality(
        _load_json(single_path), task="single", expected_orders=(4, 6)
    )
    parameterized_rows = validate_and_aggregate_quality(
        _load_json(parameterized_path), task="parameterized", expected_orders=(4,)
    )
    operator_rows = validate_operator_summary(_load_json(operator_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "single.tex").write_text(
        render_quality_table(single_rows, task="single"), encoding="utf-8"
    )
    (output_dir / "parameterized.tex").write_text(
        render_quality_table(parameterized_rows, task="parameterized"), encoding="utf-8"
    )
    (output_dir / "operator.tex").write_text(
        render_operator_table(operator_rows), encoding="utf-8"
    )
    return lazy_baseline_comparisons([*single_rows, *parameterized_rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True, help="single-task metrics.json")
    parser.add_argument(
        "--parameterized", type=Path, required=True, help="parameterized-task metrics.json"
    )
    parser.add_argument("--operator", type=Path, required=True, help="schema-v2 operator metrics.json")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables_quality"))
    args = parser.parse_args()
    for message in build_tables(
        single_path=args.single,
        parameterized_path=args.parameterized,
        operator_path=args.operator,
        output_dir=args.output_dir,
    ):
        print(message)


if __name__ == "__main__":
    main()
