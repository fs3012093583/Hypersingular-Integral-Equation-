"""Tests for the order-independent Taylor finite-part operator."""

import math

import pytest
import torch

import neural_network_solvers.general_taylor_finite_part as general_taylor
from neural_network_solvers.general_taylor_finite_part import (
    adaptive_regularized_taylor_quotient,
    adaptive_taylor_finite_part_operator,
    balanced_threshold,
    lazy_regularized_taylor_quotient,
    lazy_taylor_finite_part_operator,
    operator_budget_taylor_finite_part_operator,
    operator_budget_taylor_quotient,
    regularized_taylor_quotient,
    scaled_derivative_error_model,
    sequential_tolerance_taylor_quotient,
    taylor_finite_part_operator,
)
from neural_network_solvers.neural_hypersingular_constant_subtraction import (
    DTYPE,
    gauss_legendre_rule,
    hypersingular_operator,
)
from neural_network_solvers.neural_singular_order_representation_comparison import (
    exact_singular_operator,
    exact_solution,
    taylor_subtraction_operator,
)


def test_general_operator_matches_existing_orders_one_two_three() -> None:
    nodes, weights = gauss_legendre_rule(96)
    points = torch.tensor([[-0.73], [-0.2], [0.0], [0.41], [0.79]], dtype=DTYPE)

    for order in (1, 3):
        general = taylor_finite_part_operator(
            exact_solution,
            points,
            nodes,
            weights,
            order=order,
            threshold_mode="fixed",
            near_diagonal_threshold=1.0e-3,
        )
        existing = taylor_subtraction_operator(
            exact_solution,
            points,
            nodes,
            weights,
            order=order,
            near_zero_threshold=1.0e-3,
        )
        torch.testing.assert_close(general, existing, rtol=2.0e-12, atol=2.0e-12)

    general_second = taylor_finite_part_operator(
        exact_solution,
        points,
        nodes,
        weights,
        order=2,
        threshold_mode="fixed",
        near_diagonal_threshold=1.0e-9,
    )
    existing_second = hypersingular_operator(
        exact_solution,
        points,
        nodes,
        weights,
        near_zero_threshold=1.0e-9,
    )
    torch.testing.assert_close(general_second, existing_second, rtol=2.0e-12, atol=2.0e-12)


def test_order_four_monomial_identity() -> None:
    nodes, weights = gauss_legendre_rule(96)
    points = torch.tensor([[-0.7], [-0.1], [0.28], [0.76]], dtype=DTYPE)

    def fourth_power(x: torch.Tensor) -> torch.Tensor:
        return x.pow(4)

    computed = taylor_finite_part_operator(
        fourth_power,
        points,
        nodes,
        weights,
        order=4,
        continuation_order=1,
        threshold_mode="balanced",
    )
    logarithm = torch.log((1.0 - points) / (1.0 + points))
    j4 = (
        (-1.0 - points).pow(-3) - (1.0 - points).pow(-3)
    ) / 3.0
    j3 = 0.5 * (
        (-1.0 - points).pow(-2) - (1.0 - points).pow(-2)
    )
    j2 = (-1.0 - points).pow(-1) - (1.0 - points).pow(-1)
    expected = (
        points.pow(4) * j4
        + 4.0 * points.pow(3) * j3
        + 6.0 * points.square() * j2
        + 4.0 * points * logarithm
        + 2.0
    )
    torch.testing.assert_close(computed, expected, rtol=2.0e-11, atol=2.0e-11)


def test_order_six_monomial_has_unit_regular_remainder() -> None:
    nodes, weights = gauss_legendre_rule(128)
    points = torch.tensor([[-0.61], [-0.13], [0.22], [0.68]], dtype=DTYPE)

    def sixth_power(x: torch.Tensor) -> torch.Tensor:
        return x.pow(6)

    computed = taylor_finite_part_operator(
        sixth_power,
        points,
        nodes,
        weights,
        order=6,
        continuation_order=2,
        threshold_mode="balanced",
    )
    # For u(tau)=tau^6 the sixth-order Taylor remainder is exactly
    # (tau-t)^6, so the regular integral contributes the interval length 2.
    expected = torch.zeros_like(points)
    factorial = math.factorial
    derivatives = [
        factorial(6) / factorial(6 - k) * points.pow(6 - k)
        for k in range(6)
    ]
    for derivative_order in range(5):
        power = 6 - derivative_order - 1
        moment = (
            (-1.0 - points).pow(-power) - (1.0 - points).pow(-power)
        ) / power
        expected = expected + derivatives[derivative_order] * moment / factorial(
            derivative_order
        )
    expected = expected + derivatives[5] * torch.log(
        (1.0 - points) / (1.0 + points)
    ) / factorial(5)
    expected = expected + 2.0
    torch.testing.assert_close(computed, expected, rtol=2.0e-9, atol=2.0e-9)


def test_balanced_continuation_prevents_fourth_order_cancellation() -> None:
    t = torch.tensor([[0.23]], dtype=torch.float64)
    tau = t.detach() + 1.0e-5
    reference = torch.exp(t) * (
        1.0 / 24.0 + (tau - t) / 120.0
    )
    direct, _ = regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        continuation_order=1,
        threshold_mode="diagonal",
    )
    balanced, threshold = regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        continuation_order=1,
        threshold_mode="balanced",
    )
    assert threshold > 1.0e-5
    assert torch.abs(balanced - reference) < 1.0e-12
    assert torch.abs(direct - reference) > 1.0e-4


def test_balanced_threshold_grows_with_singularity_order() -> None:
    thresholds = [
        balanced_threshold(order, 1, dtype=torch.float64)
        for order in (1, 2, 3, 4)
    ]
    assert thresholds == sorted(thresholds)


def test_balanced_threshold_uses_physical_interval_and_multiplier() -> None:
    base = balanced_threshold(4, 2, dtype=torch.float64, interval_length=2.0)
    scaled = balanced_threshold(
        4,
        2,
        dtype=torch.float64,
        interval_length=5.0,
        threshold_multiplier=1.5,
    )
    assert scaled == pytest.approx(base * 2.5 * 1.5)

    t = torch.tensor([[1.25]], dtype=torch.float64)
    tau = torch.tensor([[1.251]], dtype=torch.float64)
    _, threshold = regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        continuation_order=2,
        threshold_mode="balanced",
        interval_length=5.0,
        threshold_multiplier=1.5,
    )
    assert threshold == pytest.approx(scaled)


def test_operator_passes_actual_interval_length_to_balanced_quotient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float] = {}
    original = general_taylor.regularized_taylor_quotient

    def recording_quotient(*args: object, **kwargs: object) -> tuple[torch.Tensor, float]:
        observed["interval_length"] = float(kwargs["interval_length"])
        observed["threshold_multiplier"] = float(kwargs["threshold_multiplier"])
        return original(*args, **kwargs)

    monkeypatch.setattr(general_taylor, "regularized_taylor_quotient", recording_quotient)
    nodes = torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.float64)
    weights = torch.full_like(nodes, 1.0)
    general_taylor.taylor_finite_part_operator(
        torch.exp,
        torch.tensor([[1.25]], dtype=torch.float64),
        nodes,
        weights,
        order=2,
        interval=(0.0, 4.0),
        threshold_mode="balanced",
        threshold_multiplier=1.25,
    )
    assert observed == {"interval_length": 4.0, "threshold_multiplier": 1.25}


def test_threshold_multiplier_scales_fixed_threshold_and_is_validated() -> None:
    t = torch.tensor([[0.2]], dtype=torch.float64)
    tau = torch.tensor([[0.201]], dtype=torch.float64)
    _, threshold = regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=2,
        threshold_mode="fixed",
        near_diagonal_threshold=1.0e-3,
        threshold_multiplier=2.5,
    )
    assert threshold == pytest.approx(2.5e-3)
    with pytest.raises(ValueError, match="threshold_multiplier"):
        balanced_threshold(2, 1, dtype=torch.float64, threshold_multiplier=0.0)
    with pytest.raises(ValueError, match="threshold_multiplier"):
        regularized_taylor_quotient(
            torch.exp,
            t,
            tau,
            order=2,
            threshold_multiplier=-1.0,
        )


def test_constant_and_low_degree_polynomials_have_zero_exhausted_derivatives() -> None:
    t = torch.tensor([[-0.35], [0.4]], dtype=torch.float64)
    tau = torch.tensor([[-0.3], [0.45]], dtype=torch.float64)

    def constant(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * 3.0

    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return 2.0 - 3.0 * x + 0.5 * x.square()

    for function in (constant, quadratic):
        quotient, _ = regularized_taylor_quotient(
            function,
            t,
            tau,
            order=4,
            continuation_order=3,
            threshold_mode="fixed",
            near_diagonal_threshold=2.0,
        )
        torch.testing.assert_close(quotient, torch.zeros_like(quotient), atol=1.0e-14, rtol=0.0)


def test_adaptive_selector_uses_local_formula_near_diagonal_and_direct_far_away() -> None:
    t = torch.tensor([[0.23]], dtype=torch.float64)
    tau = torch.tensor([[0.23001], [0.73]], dtype=torch.float64)
    quotient, diagnostics = adaptive_regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        max_continuation_order=3,
        selection_mode="cost_aware",
        near_optimal_factor=2.0,
    )
    delta = tau.reshape(1, -1) - t
    reference_near = torch.exp(t) * sum(
        delta[:, :1].pow(local_order) / math.factorial(4 + local_order)
        for local_order in range(8)
    )
    assert diagnostics.selected_order[0, 0] >= 0
    assert diagnostics.selected_order[0, 1] == -1
    torch.testing.assert_close(
        quotient[:, :1], reference_near, rtol=2.0e-12, atol=2.0e-12
    )


def test_adaptive_selector_prefers_lowest_exact_order_for_exhausted_polynomial() -> None:
    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return 1.0 + x + x.square()

    t = torch.tensor([[0.1]], dtype=torch.float64)
    tau = torch.tensor([[0.100001]], dtype=torch.float64)
    quotient, diagnostics = adaptive_regularized_taylor_quotient(
        quadratic,
        t,
        tau,
        order=2,
        max_continuation_order=3,
        selection_mode="cost_aware",
    )
    assert diagnostics.selected_order.item() == 0
    torch.testing.assert_close(quotient, torch.ones_like(quotient), atol=1.0e-14, rtol=0.0)


def test_adaptive_operator_preserves_parameter_gradients() -> None:
    nodes, weights = gauss_legendre_rule(64)
    amplitude = torch.nn.Parameter(torch.tensor(1.25, dtype=DTYPE))

    def parameterized_exp(x: torch.Tensor) -> torch.Tensor:
        return amplitude * torch.exp(x)

    value, diagnostics = adaptive_taylor_finite_part_operator(
        parameterized_exp,
        torch.tensor([[0.2]], dtype=DTYPE),
        nodes,
        weights,
        order=4,
        max_continuation_order=3,
        return_diagnostics=True,
    )
    value.sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)
    assert diagnostics.selected_order.shape == (1, nodes.numel())


def test_adaptive_selector_validates_policy_controls() -> None:
    t = torch.tensor([[0.2]], dtype=torch.float64)
    tau = torch.tensor([[0.21]], dtype=torch.float64)
    with pytest.raises(ValueError, match="near_optimal_factor"):
        adaptive_regularized_taylor_quotient(
            torch.exp,
            t,
            tau,
            order=2,
            near_optimal_factor=0.9,
        )
    with pytest.raises(ValueError, match="max_continuation_order"):
        adaptive_regularized_taylor_quotient(
            torch.exp,
            t,
            tau,
            order=2,
            max_continuation_order=-1,
        )
    with pytest.raises(ValueError, match="target_error"):
        adaptive_regularized_taylor_quotient(
            torch.exp,
            t,
            tau,
            order=2,
            selection_mode="tolerance",
        )


def test_tolerance_selector_chooses_lowest_order_meeting_error_budget() -> None:
    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return 1.0 + x + x.square()

    quotient, diagnostics = adaptive_regularized_taylor_quotient(
        quadratic,
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.20001]], dtype=torch.float64),
        order=2,
        max_continuation_order=3,
        selection_mode="tolerance",
        target_error=1.0e-12,
    )
    assert diagnostics.selected_order.item() == 0
    torch.testing.assert_close(quotient, torch.ones_like(quotient), atol=1.0e-14, rtol=0.0)


def test_segment_derivative_bound_can_replace_global_trust_region() -> None:
    def exp_derivative_bound(
        t: torch.Tensor, tau: torch.Tensor, derivative_order: int
    ) -> torch.Tensor:
        del derivative_order
        return torch.exp(torch.maximum(t, tau))

    t = torch.tensor([[0.1]], dtype=torch.float64)
    tau = torch.tensor([[0.10001], [0.9]], dtype=torch.float64)
    quotient, diagnostics = adaptive_regularized_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        max_continuation_order=3,
        derivative_bound=exp_derivative_bound,
        trust_region_multiplier=None,
    )
    assert diagnostics.selected_order[0, 0] >= 0
    assert diagnostics.selected_order[0, 1] == -1
    assert torch.all(torch.isfinite(quotient))


def test_sequential_selector_stops_before_maximum_order() -> None:
    t = torch.tensor([[0.23]], dtype=torch.float64)
    tau = torch.tensor([[0.23001]], dtype=torch.float64)
    quotient, diagnostics = sequential_tolerance_taylor_quotient(
        torch.exp,
        t,
        tau,
        order=4,
        target_error=1.0e-6,
        max_continuation_order=3,
    )
    assert diagnostics.selected_order.item() == 0
    assert diagnostics.highest_derivative_order == 5
    assert diagnostics.highest_derivative_order < 4 + 3
    reference = torch.exp(t) * sum(
        (tau - t).pow(local_order) / math.factorial(4 + local_order)
        for local_order in range(6)
    )
    assert torch.max(torch.abs(quotient - reference)) < 1.0e-6


def test_sequential_selector_does_not_generate_omitted_derivative_when_bounded() -> None:
    def exp_bound(t: torch.Tensor, tau: torch.Tensor, order: int) -> torch.Tensor:
        del order
        return torch.exp(torch.maximum(t, tau))

    _, diagnostics = sequential_tolerance_taylor_quotient(
        torch.exp,
        torch.tensor([[0.23]], dtype=torch.float64),
        torch.tensor([[0.23001]], dtype=torch.float64),
        order=4,
        target_error=1.0e-6,
        max_continuation_order=3,
        derivative_bound=exp_bound,
    )
    assert diagnostics.selected_order.item() == 0
    assert diagnostics.highest_derivative_order == 4


def test_sequential_selector_accepts_far_direct_branch_without_extra_derivatives() -> None:
    quotient, diagnostics = sequential_tolerance_taylor_quotient(
        torch.exp,
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.8]], dtype=torch.float64),
        order=4,
        target_error=1.0e-10,
        max_continuation_order=3,
    )
    assert diagnostics.selected_order.item() == -1
    assert diagnostics.highest_derivative_order == 3
    assert torch.isfinite(quotient).all()


def test_sequential_selector_preserves_parameter_gradient() -> None:
    amplitude = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

    def parameterized_exp(x: torch.Tensor) -> torch.Tensor:
        return amplitude * torch.exp(x)

    quotient, _ = sequential_tolerance_taylor_quotient(
        parameterized_exp,
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.20001], [0.7]], dtype=torch.float64),
        order=4,
        target_error=1.0e-9,
    )
    quotient.sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)


def test_sequential_selector_compacts_rows_before_high_order_ad() -> None:
    call_sizes: list[int] = []

    def recorded_exp(points: torch.Tensor) -> torch.Tensor:
        call_sizes.append(points.numel())
        return torch.exp(points)

    quotient, diagnostics = sequential_tolerance_taylor_quotient(
        recorded_exp,
        torch.tensor([[0.1], [-0.8], [0.8]], dtype=torch.float64),
        torch.tensor([[0.10001]], dtype=torch.float64),
        order=4,
        target_error=1.0e-6,
        max_continuation_order=2,
    )
    assert torch.isfinite(quotient).all()
    assert diagnostics.selected_order[0].item() >= 0
    assert diagnostics.selected_order[1].item() == -1
    assert diagnostics.selected_order[2].item() == -1
    # The first two evaluations cover all targets and the source node.  Every
    # later function evaluation belongs to a compact unresolved target set.
    assert call_sizes[:2] == [3, 1]
    assert all(size < 3 for size in call_sizes[2:])


def test_operator_budget_can_ignore_a_tiny_weighted_near_pair() -> None:
    t = torch.tensor([[0.23]], dtype=torch.float64)
    tau = torch.tensor([[0.23001], [0.73]], dtype=torch.float64)
    weights = torch.tensor([[1.0e-14], [1.0]], dtype=torch.float64)
    quotient, diagnostics = operator_budget_taylor_quotient(
        torch.exp,
        t,
        tau,
        weights,
        order=4,
        target_operator_error=1.0e-6,
        max_continuation_order=3,
    )
    assert torch.all(diagnostics.selected_order == -1)
    assert diagnostics.active_row_counts == ()
    assert diagnostics.highest_derivative_order == 3
    assert diagnostics.selected_operator_error_estimate.item() <= 1.0e-6
    assert torch.all(torch.isfinite(quotient))


def test_operator_budget_uses_limit_before_higher_taylor_terms() -> None:
    def exp_bound(
        t: torch.Tensor, tau: torch.Tensor, derivative_order: int
    ) -> torch.Tensor:
        del derivative_order
        return torch.exp(torch.maximum(t, tau))

    t = torch.tensor([[0.23]], dtype=torch.float64)
    tau = torch.tensor([[0.23001], [0.73]], dtype=torch.float64)
    weights = torch.ones_like(tau)
    quotient, diagnostics = operator_budget_taylor_quotient(
        torch.exp,
        t,
        tau,
        weights,
        order=4,
        target_operator_error=1.0e-6,
        max_continuation_order=3,
        derivative_bound=exp_bound,
        trust_region_multiplier=None,
    )
    assert diagnostics.selected_order[0, 0].item() == 0
    assert diagnostics.selected_order[0, 1].item() == -1
    assert diagnostics.highest_derivative_order == 4
    assert diagnostics.active_row_counts == (1,)
    assert diagnostics.selected_operator_error_estimate.item() <= 1.0e-6
    assert diagnostics.estimated_budget_met_mask.item()
    assert diagnostics.unresolved_row_count == 0
    torch.testing.assert_close(
        quotient[0, 0], torch.exp(t[0, 0]) / math.factorial(4)
    )


def test_operator_budget_exposes_and_optionally_rejects_unresolved_rows() -> None:
    t = torch.tensor([[0.23]], dtype=torch.float64)
    nodes = torch.tensor([[0.23001], [0.73]], dtype=torch.float64)
    weights = torch.ones_like(nodes)
    kwargs = dict(
        order=4,
        target_operator_error=1.0e-30,
        max_continuation_order=0,
        derivative_bound=lambda x, y, k: torch.exp(torch.maximum(x, y)),
        trust_region_multiplier=None,
    )
    quotient, diagnostics = operator_budget_taylor_quotient(
        torch.exp, t, nodes, weights, **kwargs
    )
    assert torch.all(torch.isfinite(quotient))
    assert diagnostics.unresolved_row_count == 1
    assert not diagnostics.estimated_budget_met_mask.item()
    for evaluator in (
        operator_budget_taylor_quotient,
        operator_budget_taylor_finite_part_operator,
    ):
        with pytest.raises(RuntimeError, match="estimated quotient budget"):
            evaluator(
                torch.exp, t, nodes, weights,
                require_estimated_budget=True, **kwargs,
            )


def test_operator_budget_accepts_one_budget_per_target_row() -> None:
    t = torch.tensor([[0.23], [-0.4]], dtype=torch.float64)
    tau = torch.tensor([[0.23001], [0.73]], dtype=torch.float64)
    weights = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    budgets = torch.tensor([[1.0e-6], [1.0e-3]], dtype=torch.float64)
    _, diagnostics = operator_budget_taylor_quotient(
        torch.exp,
        t,
        tau,
        weights,
        order=4,
        target_operator_error=budgets,
        max_continuation_order=2,
    )
    torch.testing.assert_close(diagnostics.target_operator_error, budgets)
    assert diagnostics.selected_order.shape == (2, 2)
    with pytest.raises(ValueError, match="one value per target row"):
        operator_budget_taylor_quotient(
            torch.exp,
            t,
            tau,
            weights,
            order=4,
            target_operator_error=torch.ones(3, dtype=torch.float64),
        )


def test_operator_budget_finite_part_operator_preserves_parameter_gradient() -> None:
    nodes, weights = gauss_legendre_rule(48)
    amplitude = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

    def parameterized_exp(x: torch.Tensor) -> torch.Tensor:
        return amplitude * torch.exp(x)

    value, diagnostics = operator_budget_taylor_finite_part_operator(
        parameterized_exp,
        torch.tensor([[0.2]], dtype=torch.float64),
        nodes,
        weights,
        order=4,
        target_operator_error=2.0e-3,
        max_continuation_order=3,
        return_diagnostics=True,
    )
    value.sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)
    assert diagnostics.selected_operator_error_estimate.shape == (1, 1)
    assert diagnostics.selected_order.shape == (1, nodes.numel())


def test_scaled_derivative_error_model_uses_observed_derivative_scale() -> None:
    model = scaled_derivative_error_model({0: 1.0e-7, 1: 2.0e-7}, safety_factor=3.0)
    observed = torch.tensor([[0.2], [5.0]], dtype=torch.float64)
    expected = torch.tensor([[6.0e-7], [3.0e-6]], dtype=torch.float64)
    torch.testing.assert_close(model(1, observed), expected)


@pytest.mark.parametrize(
    "order, continuation_order",
    [(2, 0), (2, 2), (4, 0), (4, 2), (6, 0), (6, 2)],
)
def test_lazy_fixed_quotient_matches_fixed_reference_at_diagonal(
    order: int, continuation_order: int
) -> None:
    t = torch.tensor([[-0.4], [0.15], [0.7]], dtype=torch.float64)
    tau = torch.tensor([[-0.4], [0.15], [0.7], [-0.8]], dtype=torch.float64)
    kwargs = dict(
        order=order,
        continuation_order=continuation_order,
        threshold_mode="diagonal",
    )
    reference, reference_threshold = regularized_taylor_quotient(torch.exp, t, tau, **kwargs)
    actual, threshold, diagnostics = lazy_regularized_taylor_quotient(
        torch.exp, t, tau, return_diagnostics=True, **kwargs
    )
    assert threshold == reference_threshold == 0.0
    assert diagnostics.active_row_count == t.numel()
    assert diagnostics.highest_derivative_order == order + continuation_order
    torch.testing.assert_close(actual, reference, rtol=2.0e-13, atol=2.0e-13)


def test_lazy_fixed_quotient_compacts_mixed_near_and_far_rows() -> None:
    t = torch.tensor([[0.1], [-0.8], [0.8]], dtype=torch.float64)
    tau = torch.tensor([[0.10001], [-0.3], [0.6]], dtype=torch.float64)
    kwargs = dict(
        order=4,
        continuation_order=2,
        threshold_mode="fixed",
        near_diagonal_threshold=1.0e-3,
    )
    reference, _ = regularized_taylor_quotient(torch.exp, t, tau, **kwargs)
    actual, _, diagnostics = lazy_regularized_taylor_quotient(
        torch.exp, t, tau, return_diagnostics=True, **kwargs
    )
    assert diagnostics.active_row_count == 1
    assert diagnostics.highest_derivative_order == 6
    torch.testing.assert_close(actual, reference, rtol=2.0e-13, atol=2.0e-13)


@pytest.mark.parametrize("threshold_mode", ["balanced", "fixed", "diagonal"])
def test_lazy_fixed_quotient_preserves_each_geometric_threshold_policy(
    threshold_mode: str,
) -> None:
    t = torch.tensor([[0.23], [-0.42]], dtype=torch.float64)
    tau = torch.tensor([[0.23], [0.23001], [-0.42], [0.7]], dtype=torch.float64)
    kwargs: dict[str, object] = dict(
        order=4,
        continuation_order=2,
        threshold_mode=threshold_mode,
    )
    if threshold_mode == "fixed":
        kwargs["near_diagonal_threshold"] = 1.0e-4
    reference, reference_threshold = regularized_taylor_quotient(
        torch.exp, t, tau, **kwargs
    )
    actual, threshold, _ = lazy_regularized_taylor_quotient(
        torch.exp, t, tau, return_diagnostics=True, **kwargs
    )
    assert threshold == reference_threshold
    torch.testing.assert_close(actual, reference, rtol=2.0e-13, atol=2.0e-13)


def test_lazy_fixed_quotient_without_near_rows_never_requests_high_ad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_orders: list[int] = []
    original = general_taylor.value_and_derivatives

    def recording_derivatives(
        function: object, points: torch.Tensor, highest_order: int
    ) -> list[torch.Tensor]:
        requested_orders.append(highest_order)
        return original(function, points, highest_order)  # type: ignore[arg-type]

    monkeypatch.setattr(general_taylor, "value_and_derivatives", recording_derivatives)
    quotient, _, diagnostics = lazy_regularized_taylor_quotient(
        torch.exp,
        torch.tensor([[-0.6], [0.6]], dtype=torch.float64),
        torch.tensor([[-0.1], [0.1]], dtype=torch.float64),
        order=4,
        continuation_order=2,
        threshold_mode="fixed",
        near_diagonal_threshold=1.0e-4,
        return_diagnostics=True,
    )
    assert requested_orders == [3]
    assert diagnostics.active_row_count == 0
    assert diagnostics.highest_derivative_order == 3
    assert torch.isfinite(quotient).all()


def test_lazy_fixed_operator_matches_reference_on_arbitrary_interval_and_gradients() -> None:
    left, right = 2.0, 5.0
    nodes = torch.tensor([[2.1], [2.75], [3.5], [4.25], [4.9]], dtype=torch.float64)
    weights = torch.full_like(nodes, (right - left) / nodes.numel())
    points = torch.tensor([[2.75], [3.2], [4.25]], dtype=torch.float64)
    amplitude_reference = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
    amplitude_lazy = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))

    def reference_function(x: torch.Tensor) -> torch.Tensor:
        return amplitude_reference * torch.exp(x / right)

    def lazy_function(x: torch.Tensor) -> torch.Tensor:
        return amplitude_lazy * torch.exp(x / right)

    kwargs = dict(
        order=4,
        interval=(left, right),
        continuation_order=2,
        threshold_mode="fixed",
        near_diagonal_threshold=1.0e-8,
    )
    reference = taylor_finite_part_operator(
        reference_function, points, nodes, weights, **kwargs
    )
    actual, diagnostics = lazy_taylor_finite_part_operator(
        lazy_function, points, nodes, weights, return_diagnostics=True, **kwargs
    )
    reference.sum().backward()
    actual.sum().backward()
    assert diagnostics.active_row_count == 2
    assert diagnostics.highest_derivative_order == 6
    torch.testing.assert_close(actual, reference, rtol=2.0e-13, atol=2.0e-13)
    assert amplitude_reference.grad is not None
    assert amplitude_lazy.grad is not None
    torch.testing.assert_close(
        amplitude_lazy.grad, amplitude_reference.grad, rtol=2.0e-12, atol=2.0e-12
    )
