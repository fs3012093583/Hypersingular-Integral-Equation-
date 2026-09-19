from dataclasses import replace

import pytest
import torch

from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule
from experiments.parameterized_operator_budget_benchmark import (
    PRESETS as PARAMETER_PRESETS,
    ParameterizedConfig,
    run_one as run_parameterized,
)
from experiments.sequential_tolerance_end_to_end_benchmark import (
    PRESETS as SINGLE_PRESETS,
    EndToEndConfig,
    run_one as run_single,
)
from neural_network_solvers.general_taylor_finite_part import taylor_finite_part_operator


@pytest.mark.parametrize("parameterized", [False, True])
def test_lazy_baseline_tiny_training_and_cost_scope(parameterized):
    base = ParameterizedConfig() if parameterized else EndToEndConfig()
    preset = PARAMETER_PRESETS["smoke"] if parameterized else SINGLE_PRESETS["smoke"]
    run = run_parameterized if parameterized else run_single
    base = replace(base, **preset, order=4)
    eager = run(replace(base, representation="fixed_p2"))
    lazy = run(replace(base, representation="lazy_fixed_p2"))
    assert eager["initial_model_fingerprint"] == lazy["initial_model_fingerprint"]
    key = "heldout_solution_rmse" if parameterized else "solution_rmse_closed_interval"
    assert lazy[key] == pytest.approx(eager[key], rel=1e-9, abs=1e-11)
    for record in (eager, lazy):
        assert record["method_setup_and_training_seconds"] == pytest.approx(
            record["method_setup_seconds"] + record["training_seconds_including_envelope_refresh"]
        )
        assert record["solve_seconds_before_validation"] >= record["method_setup_and_training_seconds"]


def test_zero_quotient_error_does_not_remove_quadrature_error():
    # At t=0, I_4[tau**8] is the ordinary integral of tau**4.  The
    # quotient is exact as a polynomial; the two-point rule is not exact.
    nodes, weights = gauss_legendre_rule(2, torch.float64)
    value = taylor_finite_part_operator(
        lambda x: x**8, torch.tensor([[0.0]], dtype=torch.float64), nodes, weights,
        order=4, continuation_order=2, threshold_mode="diagonal",
    )
    assert value.item() == pytest.approx(2 / 9, abs=1e-14)
    assert abs(value.item() - 2 / 5) == pytest.approx(8 / 45, abs=1e-14)
