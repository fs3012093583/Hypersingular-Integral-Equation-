"""Tests for the pure-standard-library quality-table generator."""

import unittest

from paper.build_quality_revision_tables import (
    ValidationError,
    render_operator_table,
    render_quality_table,
    validate_and_aggregate_quality,
    validate_operator_summary,
)


METHODS = ("fixed_p2", "lazy_fixed_p2", "operator_budget")


def _record(
    *, task: str, order: int, seed: int, method: str, fingerprint: str = "paired"
) -> dict:
    method_offset = METHODS.index(method)
    return {
        "task": task,
        "method": method,
        "status": "ok",
        "config": {"order": order, "seed": seed, "representation": method},
        "initial_model_fingerprint": fingerprint,
        "solution_rmse": float(seed - 10 + method_offset),
        "method_setup_and_training_seconds": float(10 * (seed - 10) + method_offset),
        "saved_tensor_payload_bytes": float((method_offset + 1) * 1024**2),
    }


def _quality_payload(task: str, orders: tuple[int, ...]) -> dict:
    return {
        "records": [
            _record(task=task, order=order, seed=seed, method=method)
            for order in orders
            for seed in (11, 12, 13)
            for method in METHODS
        ]
    }


def _operator_payload() -> dict:
    summary = []
    for dtype in ("float32", "float64"):
        for order in (2, 4, 6):
            summary.append(
                {
                    "dtype": dtype,
                    "order": order,
                    "strategy": "operator_budget_1e-04_pmax3",
                    "samples": 15,
                    "weighted_quotient_absolute_error_rmse": 1.0e-5 * order,
                    "moment_error_rmse": 2.0e-6 * order,
                    "representation_error_rmse": 3.0e-5 * order,
                    "quotient_indicator_coverage": 1.0,
                    "actual_quotient_budget_satisfaction": 14 / 15,
                    "actual_full_discrete_budget_satisfaction": 13 / 15,
                }
            )
    return {"schema_version": 2, "summary": summary}


class QualityRevisionTableTests(unittest.TestCase):
    def test_aggregates_and_formats_single_table(self) -> None:
        rows = validate_and_aggregate_quality(
            _quality_payload("single", (4, 6)), task="single", expected_orders=(4, 6)
        )
        fixed_m4 = next(
            row for row in rows if row["order"] == 4 and row["method"] == "fixed_p2"
        )
        self.assertEqual(fixed_m4["samples"], 3)
        self.assertEqual(fixed_m4["solution_rmse_mean"], 2.0)
        self.assertEqual(fixed_m4["solution_rmse_sample_std"], 1.0)
        self.assertEqual(fixed_m4["method_setup_and_training_seconds_mean"], 20.0)
        table = render_quality_table(rows, task="single")
        self.assertIn("\\label{tab:operator-budget-seeds}", table)
        self.assertIn("按需固定 $p=2$", table)
        self.assertIn("$(2.000\\pm1.000)\\times10^{0}$", table)
        self.assertIn("1.00", table)

    def test_rejects_missing_seed_control(self) -> None:
        payload = _quality_payload("parameterized", (4,))
        payload["records"] = [
            record
            for record in payload["records"]
            if not (record["method"] == "lazy_fixed_p2" and record["config"]["seed"] == 13)
        ]
        with self.assertRaisesRegex(ValidationError, "expected three"):
            validate_and_aggregate_quality(
                payload, task="parameterized", expected_orders=(4,)
            )

    def test_rejects_inconsistent_paired_fingerprint(self) -> None:
        payload = _quality_payload("parameterized", (4,))
        for record in payload["records"]:
            if record["method"] == "operator_budget" and record["config"]["seed"] == 12:
                record["initial_model_fingerprint"] = "different"
        with self.assertRaisesRegex(ValidationError, "inconsistent initial fingerprints"):
            validate_and_aggregate_quality(
                payload, task="parameterized", expected_orders=(4,)
            )

    def test_rejects_paired_config_difference_and_names_nested_field(self) -> None:
        payload = _quality_payload("parameterized", (4,))
        for record in payload["records"]:
            record["config"]["calibration"] = {"path": "shared-calibration.json"}
            if record["method"] == "operator_budget" and record["config"]["seed"] == 12:
                record["config"]["calibration"] = {"path": "other-calibration.json"}
        with self.assertRaisesRegex(ValidationError, r"config\.calibration\.path"):
            validate_and_aggregate_quality(
                payload, task="parameterized", expected_orders=(4,)
            )

    def test_rejects_cross_seed_config_difference_and_names_field(self) -> None:
        payload = _quality_payload("parameterized", (4,))
        for record in payload["records"]:
            record["config"]["quadrature"] = {"nodes": 64}
            if record["config"]["seed"] == 13:
                record["config"]["quadrature"] = {"nodes": 128}
        with self.assertRaisesRegex(ValidationError, r"config\.quadrature\.nodes"):
            validate_and_aggregate_quality(
                payload, task="parameterized", expected_orders=(4,)
            )

    def test_validates_and_formats_verified_operator_summary(self) -> None:
        rows = validate_operator_summary(_operator_payload())
        table = render_operator_table(rows)
        self.assertEqual(len(rows), 6)
        self.assertIn("\\label{tab:budget-audit}", table)
        self.assertIn("15/15", table)
        self.assertIn("14/15", table)
        self.assertIn("13/15", table)


if __name__ == "__main__":
    unittest.main()
