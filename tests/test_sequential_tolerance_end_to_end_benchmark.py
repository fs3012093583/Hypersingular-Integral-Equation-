"""Smoke tests for the end-to-end sequential stopping comparison."""

from dataclasses import replace

import torch

from experiments.general_taylor_neural_benchmark import TanhNetwork, set_seed
from experiments.sequential_tolerance_end_to_end_benchmark import (
    EndToEndConfig,
    OnlineDerivativeEnvelope,
    _finite_part,
    _load_derivative_error_model,
    run_one,
)
from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule


def test_online_derivative_envelope_is_monotone() -> None:
    set_seed(7)
    model = TanhNetwork(width=4, hidden_layers=1).to(dtype=torch.float64)
    envelope = OnlineDerivativeEnvelope(
        model,
        highest_order=4,
        dtype=torch.float64,
        calibration_points=7,
        safety_factor=2.0,
    )
    envelope.refresh("initial", 0)
    first = dict(envelope.bounds)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(1.1)
    envelope.refresh("updated", 1)
    assert all(envelope.bounds[order] >= value for order, value in first.items())
    bound = envelope(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[0.2]], dtype=torch.float64),
        4,
    )
    assert bound.shape == (1, 1)
    assert torch.isfinite(bound).all()


def test_requested_unavailable_cuda_is_reported() -> None:
    if torch.cuda.is_available():
        return
    base = EndToEndConfig(
        device="cuda",
        steps=1,
        lbfgs_steps=0,
        width=4,
        hidden_layers=1,
        collocation_points=4,
        training_quadrature_points=8,
        rhs_quadrature_points=12,
        validation_points=9,
        validation_quadrature_points=12,
    )
    try:
        run_one(base)
    except RuntimeError as error:
        assert "CUDA was requested" in str(error)
    else:
        raise AssertionError("requesting unavailable CUDA should fail explicitly")


def test_fixed_and_sequential_complete_tiny_training() -> None:
    base = EndToEndConfig(
        order=2,
        family="exp",
        steps=2,
        lbfgs_steps=0,
        width=6,
        hidden_layers=1,
        collocation_points=5,
        training_quadrature_points=10,
        rhs_quadrature_points=24,
        validation_points=17,
        validation_quadrature_points=24,
        envelope_refresh_interval=1,
        envelope_calibration_points=7,
        timing_repeats=1,
    )
    fixed = run_one(replace(base, representation="fixed_p3"))
    sequential = run_one(replace(base, representation="sequential"))
    assert fixed["initial_model_fingerprint"] == sequential["initial_model_fingerprint"]
    for record in (fixed, sequential):
        assert record["status"] == "ok"
        assert record["solution_rmse_closed_interval"] >= 0.0
        assert record["training_seconds_including_envelope_refresh"] > 0.0
        assert record["post_training_cost"]["median_saved_tensor_payload_bytes"] > 0
    assert sequential["selection_diagnostics"] is not None


def test_limit_and_operator_budget_complete_tiny_training() -> None:
    base = EndToEndConfig(
        order=4,
        family="exp",
        steps=2,
        lbfgs_steps=0,
        width=6,
        hidden_layers=1,
        collocation_points=5,
        training_quadrature_points=10,
        rhs_quadrature_points=24,
        validation_points=17,
        validation_quadrature_points=24,
        envelope_refresh_interval=1,
        envelope_calibration_points=7,
        timing_repeats=1,
    )
    limit = run_one(replace(base, representation="limit_p0"))
    weighted = run_one(replace(base, representation="operator_budget"))
    assert limit["initial_model_fingerprint"] == weighted[
        "initial_model_fingerprint"
    ]
    for record in (limit, weighted):
        assert record["status"] == "ok"
        assert record["selection_diagnostics"] is not None
        assert record["solution_rmse_closed_interval"] >= 0.0
    assert "operator_budget_satisfaction_fraction" in weighted[
        "selection_diagnostics"
    ]


def test_chunked_and_full_batch_sequential_values_and_gradients_match() -> None:
    set_seed(11)
    model = TanhNetwork(width=5, hidden_layers=1).to(dtype=torch.float64)
    points = torch.tensor([[-0.7], [-0.2], [0.3], [0.8]], dtype=torch.float64)
    nodes, weights = gauss_legendre_rule(12, torch.float64)
    base = EndToEndConfig(
        representation="sequential",
        order=2,
        width=5,
        hidden_layers=1,
        collocation_points=4,
        training_quadrature_points=12,
        selector_batch_size=4,
    )
    error_model, _ = _load_derivative_error_model(base)
    envelope = OnlineDerivativeEnvelope(
        model,
        highest_order=6,
        dtype=torch.float64,
        calibration_points=9,
        safety_factor=4.0,
    )
    envelope.refresh("initial", 0)
    full = _finite_part(
        model,
        points,
        nodes,
        weights,
        base,
        derivative_error_model=error_model,
        derivative_bound=envelope,
    )
    assert isinstance(full, torch.Tensor)
    full.square().sum().backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    chunked = _finite_part(
        model,
        points,
        nodes,
        weights,
        replace(base, selector_batch_size=1),
        derivative_error_model=error_model,
        derivative_bound=envelope,
    )
    assert isinstance(chunked, torch.Tensor)
    chunked.square().sum().backward()
    torch.testing.assert_close(chunked, full, rtol=1.0e-12, atol=1.0e-12)
    for chunked_gradient, full_gradient in zip(
        (parameter.grad for parameter in model.parameters()), full_gradients
    ):
        torch.testing.assert_close(
            chunked_gradient, full_gradient, rtol=1.0e-10, atol=1.0e-10
        )


def test_chunked_and_full_batch_operator_budget_values_and_gradients_match() -> None:
    set_seed(13)
    model = TanhNetwork(width=5, hidden_layers=1).to(dtype=torch.float64)
    points = torch.tensor([[-0.7], [-0.2], [0.3], [0.8]], dtype=torch.float64)
    nodes, weights = gauss_legendre_rule(12, torch.float64)
    base = EndToEndConfig(
        representation="operator_budget",
        order=4,
        width=5,
        hidden_layers=1,
        collocation_points=4,
        training_quadrature_points=12,
        selector_batch_size=4,
    )
    error_model, _ = _load_derivative_error_model(base)
    envelope = OnlineDerivativeEnvelope(
        model,
        highest_order=8,
        dtype=torch.float64,
        calibration_points=9,
        safety_factor=4.0,
    )
    envelope.refresh("initial", 0)
    full = _finite_part(
        model,
        points,
        nodes,
        weights,
        base,
        derivative_error_model=error_model,
        derivative_bound=envelope,
    )
    assert isinstance(full, torch.Tensor)
    full.square().sum().backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    chunked = _finite_part(
        model,
        points,
        nodes,
        weights,
        replace(base, selector_batch_size=1),
        derivative_error_model=error_model,
        derivative_bound=envelope,
    )
    assert isinstance(chunked, torch.Tensor)
    chunked.square().sum().backward()
    torch.testing.assert_close(chunked, full, rtol=1.0e-12, atol=1.0e-12)
    for chunked_gradient, full_gradient in zip(
        (parameter.grad for parameter in model.parameters()), full_gradients
    ):
        torch.testing.assert_close(
            chunked_gradient, full_gradient, rtol=1.0e-10, atol=1.0e-10
        )
