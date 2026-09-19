"""Deterministic validation of quadrature-weighted Taylor error budgets.

The benchmark separates three errors that are otherwise mixed during neural
training: representation error against a high-precision evaluation of the same
Gauss rule, quadrature error against the continuous finite-part integral, and
their combined operator error.  This makes it possible to test whether the
weighted indicator controls the quantity it claims to control before running a
non-convex neural solve.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

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
    OperatorBudgetTaylorDiagnostics,
    operator_budget_taylor_finite_part_operator,
    operator_budget_taylor_quotient,
    taylor_finite_part_operator,
    value_and_derivatives,
)


ORDERS = (2, 4, 6)
DTYPES = (("float32", torch.float32), ("float64", torch.float64))
COUPLING = 0.05


def _exact_derivative_bound(function_name: str):
    if function_name == "exp(t)":
        return lambda t, tau, order: torch.exp(torch.maximum(t, tau))
    if function_name == "sin(pi*t)":
        return lambda t, tau, order: torch.full_like(t + tau, math.pi**order)
    if function_name == "1/(1+0.5*t)":
        def rational_bound(
            t: torch.Tensor, tau: torch.Tensor, order: int
        ) -> torch.Tensor:
            minimum = torch.minimum(t, tau)
            return (
                math.factorial(order)
                * 0.5**order
                / (1.0 + 0.5 * minimum).pow(order + 1)
            )

        return rational_bound
    raise KeyError(function_name)


def _mp_from_binary_float(value: float) -> mp.mpf:
    """Lift a stored IEEE float to mpmath without decimal-string rounding.

    ``mp.mpf(float)`` receives the binary float directly, unlike
    ``mp.mpf(str(float))`` which first rounds it to a shortest decimal string.
    This is important here because float32 nodes and target points define the
    actual discrete problem being measured.
    """
    return mp.mpf(float(value))


def _mp_regularized_quotient(spec, t_mp: mp.mpf, tau_mp: mp.mpf, order: int) -> mp.mpf:
    """Reference quotient at already-exact benchmark coordinates."""
    delta = tau_mp - t_mp
    if delta == 0:
        return spec.mp_derivative(t_mp, order) / mp.factorial(order)
    numerator = spec.mp_function(tau_mp)
    for derivative_order in range(order):
        numerator -= (
            spec.mp_derivative(t_mp, derivative_order)
            * delta**derivative_order
            / mp.factorial(derivative_order)
        )
    return numerator / delta**order


def _mp_analytic_moment(spec, t_mp: mp.mpf, order: int) -> mp.mpf:
    left, right = (_mp_from_binary_float(value) for value in INTERVAL)
    analytic = mp.mpf("0")
    for derivative_order in range(order - 1):
        power = order - derivative_order - 1
        moment = ((left - t_mp) ** (-power) - (right - t_mp) ** (-power)) / power
        analytic += (
            spec.mp_derivative(t_mp, derivative_order)
            * moment
            / mp.factorial(derivative_order)
        )
    analytic += (
        spec.mp_derivative(t_mp, order - 1)
        * mp.log((right - t_mp) / (t_mp - left))
        / mp.factorial(order - 1)
    )
    return analytic


def _mp_discrete_finite_part(
    spec,
    t_mp: mp.mpf,
    order: int,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[mp.mpf, mp.mpf, mp.mpf, list[mp.mpf]]:
    """Reference the exact float-dtype Gauss rule, retaining mp values."""
    node_values = [
        _mp_from_binary_float(float(node.item())) for node in nodes.reshape(-1)
    ]
    weight_values = [
        _mp_from_binary_float(float(weight.item())) for weight in weights.reshape(-1)
    ]
    quotient_values = [
        _mp_regularized_quotient(spec, t_mp, node, order) for node in node_values
    ]
    analytic = _mp_analytic_moment(spec, t_mp, order)
    regular = mp.fsum(
        weight * quotient for weight, quotient in zip(weight_values, quotient_values)
    )
    return analytic + regular, analytic, regular, quotient_values


def _mp_continuous_finite_part(spec, t_mp: mp.mpf, order: int) -> mp.mpf:
    """Continuous reference at the same stored target coordinate."""
    left, right = (_mp_from_binary_float(value) for value in INTERVAL)
    continuation_terms = 40
    derivatives = [
        spec.mp_derivative(t_mp, derivative_order)
        for derivative_order in range(order + continuation_terms + 1)
    ]
    analytic = _mp_analytic_moment(spec, t_mp, order)

    def regular(tau_mp: mp.mpf) -> mp.mpf:
        delta = tau_mp - t_mp
        if abs(delta) <= mp.mpf("1e-8"):
            return mp.fsum(
                derivatives[order + local_order]
                * delta**local_order
                / mp.factorial(order + local_order)
                for local_order in range(continuation_terms + 1)
            )
        numerator = spec.mp_function(tau_mp)
        for derivative_order in range(order):
            numerator -= (
                derivatives[derivative_order]
                * delta**derivative_order
                / mp.factorial(derivative_order)
            )
        return numerator / delta**order

    return analytic + mp.quad(regular, [left, t_mp, right])


def _torch_analytic_moment(
    function, point: torch.Tensor, order: int
) -> torch.Tensor:
    """The analytic-moment portion of the core finite-part operator formula."""
    point = point.detach().clone().requires_grad_(True)
    derivatives = value_and_derivatives(function, point, order - 1)
    analytic = torch.zeros_like(point)
    left, right = INTERVAL
    for derivative_order in range(order - 1):
        power = order - derivative_order - 1
        moment = (
            (left - point).pow(-power) - (right - point).pow(-power)
        ) / power
        analytic = analytic + derivatives[derivative_order] * moment / math.factorial(
            derivative_order
        )
    return analytic + (
        derivatives[order - 1]
        * torch.log((right - point) / (point - left))
        / math.factorial(order - 1)
    )


def _record_weighted(
    *,
    spec,
    order: int,
    point: torch.Tensor,
    dtype_name: str,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    residual_budget: float,
    max_p: int,
    discrete_reference: mp.mpf,
    analytic_reference: mp.mpf,
    quotient_reference: mp.mpf,
    quotient_values_reference: list[mp.mpf],
    continuous_reference: mp.mpf,
) -> dict[str, object]:
    # The equation weight is bounded by one.  A global operator cap therefore
    # controls both the raw finite-part value and its residual contribution.
    target_operator_error = torch.full_like(point, residual_budget / COUPLING)
    value, diagnostics = operator_budget_taylor_finite_part_operator(
        spec.torch_function,
        point,
        nodes,
        weights,
        order=order,
        target_operator_error=target_operator_error,
        max_continuation_order=max_p,
        trust_region_multiplier=None,
        derivative_bound=_exact_derivative_bound(spec.name),
        return_diagnostics=True,
    )
    assert isinstance(diagnostics, OperatorBudgetTaylorDiagnostics)
    quotient, quotient_diagnostics = operator_budget_taylor_quotient(
        spec.torch_function,
        point,
        nodes,
        weights,
        order=order,
        target_operator_error=target_operator_error,
        max_continuation_order=max_p,
        trust_region_multiplier=None,
        derivative_bound=_exact_derivative_bound(spec.name),
    )
    assert isinstance(quotient_diagnostics, OperatorBudgetTaylorDiagnostics)
    analytic = _torch_analytic_moment(spec.torch_function, point, order)
    weighted_quotient = torch.sum(
        quotient * weights.reshape(1, -1), dim=1, keepdim=True
    )
    reconstructed = analytic + weighted_quotient
    if not torch.equal(value, reconstructed):
        raise AssertionError(
            "operator-budget decomposition did not reconstruct the core operator"
        )

    numerical_mp = _mp_from_binary_float(float(value.item()))
    analytic_mp = _mp_from_binary_float(float(analytic.item()))
    quotient_sum_mp = _mp_from_binary_float(float(weighted_quotient.item()))
    quotient_values = [
        _mp_from_binary_float(float(item.item())) for item in quotient.reshape(-1)
    ]
    weight_values = [
        _mp_from_binary_float(float(item.item())) for item in weights.reshape(-1)
    ]
    quotient_absolute_error = mp.fsum(
        abs(weight) * abs(actual - reference)
        for weight, actual, reference in zip(
            weight_values, quotient_values, quotient_values_reference
        )
    )
    exact_sum_of_computed_quotients = mp.fsum(
        weight * actual for weight, actual in zip(weight_values, quotient_values)
    )
    quotient_weighted_sum_error = (
        exact_sum_of_computed_quotients - quotient_reference
    )
    product_sum_rounding_error = (
        quotient_sum_mp - exact_sum_of_computed_quotients
    )
    moment_error = analytic_mp - analytic_reference
    full_discrete_error_signed = numerical_mp - discrete_reference
    final_addition_rounding_error = numerical_mp - analytic_mp - quotient_sum_mp
    accumulation_rounding_error = (
        product_sum_rounding_error + final_addition_rounding_error
    )
    representation_error = abs(full_discrete_error_signed)
    total_error = abs(numerical_mp - continuous_reference)
    estimate = float(diagnostics.selected_operator_error_estimate.item())
    counts = Counter(int(item) for item in diagnostics.selected_order.reshape(-1))
    target = float(target_operator_error.item())
    estimated_budget_met = estimate <= target
    actual_quotient_budget_met = float(quotient_absolute_error) <= target
    actual_full_discrete_budget_met = float(representation_error) <= target
    return {
        "function": spec.name,
        "order": order,
        "t": float(point.item()),
        "dtype": dtype_name,
        "strategy": f"operator_budget_{residual_budget:.0e}_pmax{max_p}",
        "residual_budget": residual_budget,
        "target_operator_error": target,
        "value": float(numerical_mp),
        "discrete_reference": float(discrete_reference),
        "continuous_reference": float(continuous_reference),
        "representation_error": float(representation_error),
        "representation_error_scope": "full_discrete_operator_including_moment_and_sum",
        "quadrature_error": float(abs(discrete_reference - continuous_reference)),
        "total_error": float(total_error),
        "worst_case_residual_representation_error": (
            COUPLING * float(representation_error)
        ),
        "estimated_operator_error": estimate,
        # Legacy fields retain their old full-discrete scope.  They are not
        # used as the quotient-indicator coverage conclusion below.
        "estimate_covers_representation_error": float(representation_error) <= estimate,
        "operator_budget_met": estimated_budget_met,
        "signed_quotient_weighted_sum_error": float(quotient_weighted_sum_error),
        "quotient_weighted_sum_error": float(abs(quotient_weighted_sum_error)),
        "weighted_quotient_absolute_error": float(quotient_absolute_error),
        "signed_moment_error": float(moment_error),
        "moment_error": float(abs(moment_error)),
        "signed_full_discrete_error": float(full_discrete_error_signed),
        "signed_product_sum_rounding_error": float(product_sum_rounding_error),
        "product_sum_rounding_error": float(abs(product_sum_rounding_error)),
        "signed_final_addition_rounding_error": float(final_addition_rounding_error),
        "final_addition_rounding_error": float(abs(final_addition_rounding_error)),
        "signed_accumulation_rounding_error": float(accumulation_rounding_error),
        "accumulation_rounding_error": float(abs(accumulation_rounding_error)),
        "estimate_covers_quotient_absolute_error": (
            float(quotient_absolute_error) <= estimate
        ),
        "estimated_budget_met": estimated_budget_met,
        "actual_quotient_budget_met": actual_quotient_budget_met,
        "actual_full_discrete_budget_met": actual_full_discrete_budget_met,
        "false_accept_quotient": estimated_budget_met and not actual_quotient_budget_met,
        "false_accept_full_discrete": (
            estimated_budget_met and not actual_full_discrete_budget_met
        ),
        "derivative_error_model": "none_uncalibrated_analytic_function",
        "selected_counts": {str(key): value for key, value in sorted(counts.items())},
        "highest_derivative_order": diagnostics.highest_derivative_order,
    }


def _record_fixed(
    *,
    spec,
    order: int,
    point: torch.Tensor,
    dtype_name: str,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    continuation_order: int,
    discrete_reference: mp.mpf,
    continuous_reference: mp.mpf,
) -> dict[str, object]:
    value = taylor_finite_part_operator(
        spec.torch_function,
        point,
        nodes,
        weights,
        order=order,
        continuation_order=continuation_order,
        threshold_mode="balanced",
    )
    numerical_mp = _mp_from_binary_float(float(value.item()))
    representation_error = abs(numerical_mp - discrete_reference)
    return {
        "function": spec.name,
        "order": order,
        "t": float(point.item()),
        "dtype": dtype_name,
        "strategy": f"fixed_p{continuation_order}",
        "residual_budget": None,
        "target_operator_error": None,
        "value": float(numerical_mp),
        "discrete_reference": float(discrete_reference),
        "continuous_reference": float(continuous_reference),
        "representation_error": float(representation_error),
        "representation_error_scope": "full_discrete_operator_including_moment_and_sum",
        "quadrature_error": float(abs(discrete_reference - continuous_reference)),
        "total_error": float(abs(numerical_mp - continuous_reference)),
        "estimated_operator_error": None,
        "estimate_covers_representation_error": None,
        "operator_budget_met": None,
        "signed_quotient_weighted_sum_error": None,
        "quotient_weighted_sum_error": None,
        "weighted_quotient_absolute_error": None,
        "signed_moment_error": None,
        "moment_error": None,
        "signed_full_discrete_error": float(numerical_mp - discrete_reference),
        "signed_product_sum_rounding_error": None,
        "product_sum_rounding_error": None,
        "signed_final_addition_rounding_error": None,
        "final_addition_rounding_error": None,
        "signed_accumulation_rounding_error": None,
        "accumulation_rounding_error": None,
        "estimate_covers_quotient_absolute_error": None,
        "estimated_budget_met": None,
        "actual_quotient_budget_met": None,
        "actual_full_discrete_budget_met": None,
        "false_accept_quotient": None,
        "false_accept_full_discrete": None,
        "derivative_error_model": None,
        "selected_counts": None,
        "highest_derivative_order": order + continuation_order,
    }


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(record["dtype"], record["order"], record["strategy"])].append(record)
    summary: list[dict[str, object]] = []
    for (dtype, order, strategy), group in sorted(groups.items()):
        representation = np.asarray(
            [float(item["representation_error"]) for item in group]
        )
        total = np.asarray([float(item["total_error"]) for item in group])
        coverage = [
            bool(item["estimate_covers_representation_error"])
            for item in group
            if item["estimate_covers_representation_error"] is not None
        ]
        estimated_budget = [
            bool(item["estimated_budget_met"])
            for item in group
            if item["estimated_budget_met"] is not None
        ]
        quotient_coverage = [
            bool(item["estimate_covers_quotient_absolute_error"])
            for item in group
            if item["estimate_covers_quotient_absolute_error"] is not None
        ]
        quotient_budget = [
            bool(item["actual_quotient_budget_met"])
            for item in group
            if item["actual_quotient_budget_met"] is not None
        ]
        full_discrete_budget = [
            bool(item["actual_full_discrete_budget_met"])
            for item in group
            if item["actual_full_discrete_budget_met"] is not None
        ]
        false_accept_quotient = [
            bool(item["false_accept_quotient"])
            for item in group
            if item["false_accept_quotient"] is not None
        ]
        false_accept_full_discrete = [
            bool(item["false_accept_full_discrete"])
            for item in group
            if item["false_accept_full_discrete"] is not None
        ]
        summary.append(
            {
                "dtype": dtype,
                "order": order,
                "strategy": strategy,
                "samples": len(group),
                "representation_error_rmse": float(
                    np.sqrt(np.mean(representation**2))
                ),
                "representation_error_max": float(np.max(representation)),
                "total_error_rmse": float(np.sqrt(np.mean(total**2))),
                "total_error_max": float(np.max(total)),
                # Kept for schema compatibility only: this compares a
                # quotient indicator to a full discrete operator error, so it
                # is not the relevant quotient-coverage statement.
                "indicator_coverage": float(np.mean(coverage)) if coverage else None,
                "indicator_coverage_scope": "legacy_full_discrete_comparison_not_quotient_coverage",
                "budget_satisfaction": (
                    float(np.mean(estimated_budget)) if estimated_budget else None
                ),
                "quotient_indicator_coverage": (
                    float(np.mean(quotient_coverage)) if quotient_coverage else None
                ),
                "actual_quotient_budget_satisfaction": (
                    float(np.mean(quotient_budget)) if quotient_budget else None
                ),
                "actual_full_discrete_budget_satisfaction": (
                    float(np.mean(full_discrete_budget))
                    if full_discrete_budget
                    else None
                ),
                "false_accept_quotient_fraction": (
                    float(np.mean(false_accept_quotient))
                    if false_accept_quotient
                    else None
                ),
                "false_accept_full_discrete_fraction": (
                    float(np.mean(false_accept_full_discrete))
                    if false_accept_full_discrete
                    else None
                ),
                "mean_highest_derivative_order": float(
                    np.mean([item["highest_derivative_order"] for item in group])
                ),
            }
        )
        for key in (
            "weighted_quotient_absolute_error", "quotient_weighted_sum_error",
            "moment_error", "accumulation_rounding_error", "quadrature_error",
        ):
            values = [float(item[key]) for item in group if item.get(key) is not None]
            summary[-1][key + "_rmse"] = (
                float(np.sqrt(np.mean(np.square(values)))) if values else None
            )
    return summary


def run_benchmark(
    *, quadrature_points: int, residual_budgets: tuple[float, ...]
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for spec in FUNCTIONS:
        for order in ORDERS:
            for t_value in POINTS:
                for dtype_name, dtype in DTYPES:
                    nodes, weights = _gauss_rule(quadrature_points, dtype)
                    # The tensor is the experimental target coordinate; the
                    # reference must use its stored dtype value, not the
                    # original Python request in POINTS.
                    point = torch.tensor([[t_value]], dtype=dtype)
                    t_mp = _mp_from_binary_float(float(point.item()))
                    continuous = _mp_continuous_finite_part(spec, t_mp, order)
                    (
                        discrete,
                        analytic_reference,
                        quotient_reference,
                        quotient_values_reference,
                    ) = _mp_discrete_finite_part(
                        spec, t_mp, order, nodes, weights
                    )
                    for continuation_order in (0, 2, 3):
                        records.append(
                            _record_fixed(
                                spec=spec,
                                order=order,
                                point=point,
                                dtype_name=dtype_name,
                                nodes=nodes,
                                weights=weights,
                                continuation_order=continuation_order,
                                discrete_reference=discrete,
                                continuous_reference=continuous,
                            )
                        )
                    for residual_budget in residual_budgets:
                        records.append(
                            _record_weighted(
                                spec=spec,
                                order=order,
                                point=point,
                                dtype_name=dtype_name,
                                nodes=nodes,
                                weights=weights,
                                residual_budget=residual_budget,
                                max_p=3,
                                discrete_reference=discrete,
                                analytic_reference=analytic_reference,
                                quotient_reference=quotient_reference,
                                quotient_values_reference=quotient_values_reference,
                                continuous_reference=continuous,
                            )
                        )
    return {
        "schema_version": 2,
        "experiment": "operator_budget_accuracy_benchmark",
        "interval": list(INTERVAL),
        "orders": list(ORDERS),
        "points": list(POINTS),
        "functions": [spec.name for spec in FUNCTIONS],
        "quadrature_points": quadrature_points,
        "coupling": COUPLING,
        "residual_budgets": list(residual_budgets),
        "reference": {
            "backend": "mpmath",
            "decimal_precision": mp.mp.dps,
            "target_coordinates": "actual values stored in each torch dtype",
            "discrete_reference": (
                "same stored floating Gauss nodes and weights, lifted directly "
                "from binary floats to mpmath"
            ),
            "error_subtraction": "mpmath values retained until each reported difference",
        },
        "indicator_scope": (
            "selected_operator_error_estimate is compared for coverage only with "
            "sum_j abs(w_j)*abs(Qhat_j-Qref_j); full-discrete errors additionally "
            "contain analytic-moment and floating accumulation terms"
        ),
        "decomposition": (
            "full discrete signed error = analytic moment error + exact weighted "
            "sum of quotient errors + floating product/sum rounding + final "
            "addition rounding; all differences are formed in mpmath"
        ),
        "derivative_error_model": (
            "none: this analytic-function benchmark supplies exact derivative "
            "bounds but no calibrated neural AD derivative-error model"
        ),
        "records": records,
        "summary": _summarize(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrature-points", type=int, default=256)
    parser.add_argument("--mp-dps", type=int, default=90)
    parser.add_argument(
        "--residual-budget", type=float, action="append", dest="budgets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/operator_budget_accuracy_v2"),
    )
    args = parser.parse_args()
    if args.quadrature_points < 16:
        raise ValueError("quadrature_points must be at least 16")
    if args.mp_dps < 40:
        raise ValueError("mp_dps must be at least 40")
    budgets = tuple(args.budgets or (1.0e-4, 3.0e-5, 1.0e-5))
    if any(value <= 0.0 for value in budgets):
        raise ValueError("residual budgets must be positive")
    mp.mp.dps = args.mp_dps
    payload = run_benchmark(
        quadrature_points=args.quadrature_points,
        residual_budgets=budgets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
