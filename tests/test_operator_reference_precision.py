"""Reference-coordinate and mpmath-precision regressions for operator reports."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mpmath as mp
import torch

from experiments.general_taylor_multiplier_sweep import load_references
from experiments.general_taylor_operator_benchmark import (
    FUNCTIONS,
    ORDERS,
    POINTS,
    _errors,
    mp_regularized_quotient,
)


class OperatorReferencePrecisionTests(unittest.TestCase):
    def test_error_subtraction_keeps_the_mpmath_reference_until_after_subtraction(self) -> None:
        with mp.workdps(80):
            reference = mp.mpf("1.0000000000000000000000000000000000000001")
            absolute, scaled = _errors(1.0, reference)
        self.assertGreater(absolute, 0.0)
        self.assertEqual(absolute, scaled)

    def test_regularized_quotient_preserves_mpf_coordinates(self) -> None:
        with mp.workdps(80):
            t = mp.mpf("0.1000000000000000000000000000000000000001")
            tau = mp.mpf("0.1000000000000000000000000000000000000002")
            quotient = mp_regularized_quotient(
                mp.exp,
                lambda point, derivative_order: mp.exp(point),
                t,
                tau,
                1,
            )
            expected = (mp.exp(tau) - mp.exp(t)) / (tau - t)
        self.assertLess(abs(quotient - expected), mp.mpf("1e-70"))

    def test_downstream_reference_loader_rejects_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"operator_records": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version=2"):
                load_references(path)

    def test_downstream_reference_loader_keeps_dtype_specific_stored_keys(self) -> None:
        rows = []
        for dtype in ("float32", "float64"):
            decimal = "1.0000000000000000000000000000000000000001" if dtype == "float32" else "2.0000000000000000000000000000000000000002"
            for spec in FUNCTIONS:
                for order in ORDERS:
                    for point in POINTS:
                        rows.append(
                            {
                                "function": spec.name,
                                "order": order,
                                "dtype": dtype,
                                "t_stored": point,
                                "reference_decimal": decimal,
                            }
                        )
        payload = {
            "schema_version": 2,
            "reference": {"decimal_precision": 80},
            "operator_records": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            references, precision = load_references(path)
        key_prefix = (FUNCTIONS[0].name, ORDERS[0])
        self.assertEqual(precision, 80)
        self.assertNotEqual(
            references[(*key_prefix, "float32", POINTS[0])],
            references[(*key_prefix, "float64", POINTS[0])],
        )

    def test_sequential_accuracy_uses_dtype_key_and_mp_subtraction(self) -> None:
        from experiments import sequential_tolerance_benchmark as benchmark

        spec = FUNCTIONS[0]
        references = {
            (spec.name, 2, dtype_name, float(torch.tensor(0.1, dtype=dtype).item())):
                "1.0000000000000000000000000000000000000001"
            for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64))
        }
        diagnostics = SimpleNamespace(
            selected_order=torch.tensor([[-1]]), highest_derivative_order=1
        )
        with (
            patch.object(benchmark, "FUNCTIONS", (spec,)),
            patch.object(benchmark, "ORDERS", (2,)),
            patch.object(benchmark, "POINTS", (0.1,)),
            patch.object(benchmark, "load_reference_payload", return_value=(references, [], 80)),
            patch.object(benchmark, "sequential_tolerance_taylor_finite_part_operator",
                         return_value=(torch.tensor([[1.0]]), diagnostics)),
        ):
            result = benchmark._operator_experiment(
                Path("unused.json"), quadrature_points=2, max_continuation_order=0
            )
        self.assertEqual(len(result["records"]), 2)
        for row in result["records"]:
            self.assertGreater(row["absolute_error"], 0.0)
            self.assertLess(row["absolute_error"], 1e-30)
            self.assertIn("t_stored", row)


if __name__ == "__main__":
    unittest.main()
