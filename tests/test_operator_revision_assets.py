"""Focused validation tests for revision-asset aggregation."""

import unittest

from paper.build_operator_revision_assets import (
    component_rmse_curves,
    quotient_median_curves,
    rmse_or_nonfinite,
    table_operator,
)


class OperatorRevisionAssetTests(unittest.TestCase):
    def test_nonfinite_rows_are_explicit_not_silently_dropped(self) -> None:
        value, bad, total = rmse_or_nonfinite(
            [{"absolute_error": 1.0, "nonfinite": False}, {"absolute_error": None, "nonfinite": True}]
        )
        self.assertIsNone(value)
        self.assertEqual((bad, total), (1, 2))

    def test_operator_table_marks_nonfinite_cell_and_keeps_finite_cells(self) -> None:
        rows = []
        for order in range(1, 7):
            for dtype in ("float32", "float64"):
                for strategy in ("direct", "balanced_p2"):
                    rows.append({
                        "dtype": dtype, "order": order, "strategy": strategy,
                        "quadrature_size": 256, "nonfinite": dtype == "float32" and order == 6 and strategy == "direct",
                        "absolute_error": None if dtype == "float32" and order == 6 and strategy == "direct" else 10.0 ** (-order),
                    })
        text, summary = table_operator(rows)
        self.assertIn("\\text{nonfinite} (1/1)", text)
        self.assertIn("$1.00\\times10^{-6}$", text)
        self.assertIsNone(summary["6"]["direct_f32"])

    def test_component_curve_uses_rmse_not_median(self) -> None:
        rows = []
        for order in range(1, 7):
            for method in ("direct_only", "taylor_only_p2", "balanced_hybrid_p2"):
                for position in range(15):
                    rows.append({
                        "dtype": "float64", "order": order, "method": method,
                        "function": f"f{position % 3}", "t_stored": position,
                        "absolute_error": 10.0 if position == 14 else 0.0, "nonfinite": False,
                    })
        curves = component_rmse_curves(rows)
        self.assertAlmostEqual(curves["direct"][1][0], 10.0 / (15.0 ** 0.5))
        self.assertEqual(curves["direct"][1][0], curves["global local p=2"][1][0])

    def test_quotient_curve_requires_fifteen_unique_samples(self) -> None:
        component = []
        base = []
        for method in ("direct_only", "taylor_only_p2", "balanced_hybrid_p2"):
            for position in range(15):
                component.append({"dtype": "float64", "order": 6, "method": method, "function": f"f{position % 3}", "t_stored": position, "requested_delta": 1e-3, "scaled_error": 1.0, "nonfinite": False})
        for strategy in ("balanced_p0", "balanced_p1", "balanced_p3"):
            for position in range(15):
                base.append({"dtype": "float64", "order": 6, "strategy": strategy, "function": f"f{position % 3}", "t_stored": position, "requested_delta": 1e-3, "scaled_error": 1.0, "nonfinite": False})
        with self.assertRaisesRegex(ValueError, "local-distance coverage"):
            quotient_median_curves(component, base)

    def test_component_curve_rejects_duplicate_function_position_pairs(self) -> None:
        rows = []
        for order in range(1, 7):
            for method in ("direct_only", "taylor_only_p2", "balanced_hybrid_p2"):
                for _ in range(15):
                    rows.append({"dtype": "float64", "order": order, "method": method, "function": "same", "t_stored": 0.0, "absolute_error": 1.0, "nonfinite": False})
        with self.assertRaisesRegex(ValueError, "duplicate function-position"):
            component_rmse_curves(rows)


if __name__ == "__main__":
    unittest.main()
