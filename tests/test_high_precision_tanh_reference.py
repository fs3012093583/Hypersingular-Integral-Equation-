"""Tests for the independent high-precision tanh-network derivative path."""

import mpmath as mp
import torch
from torch import nn

from neural_network_solvers.general_taylor_finite_part import value_and_derivatives
from neural_network_solvers.high_precision_tanh_reference import (
    high_precision_tanh_derivatives,
    high_precision_tanh_partial_derivatives,
    tanh_jet,
)


def test_tanh_jet_matches_mpmath_differentiation() -> None:
    with mp.workdps(80):
        point = mp.mpf("0.37")
        jet = [point, mp.mpf("1"), mp.mpf("0"), mp.mpf("0"), mp.mpf("0")]
        coefficients = tanh_jet(jet)
        for order, coefficient in enumerate(coefficients):
            expected = mp.diff(mp.tanh, point, order) / mp.factorial(order)
            assert mp.almosteq(
                coefficient,
                expected,
                rel_eps=mp.mpf("1e-60"),
                abs_eps=mp.mpf("1e-60"),
            )


def test_high_precision_network_reference_matches_float64_autograd() -> None:
    torch.manual_seed(17)
    model = nn.Sequential(
        nn.Linear(1, 5), nn.Tanh(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 1)
    ).to(dtype=torch.float64)
    point = 0.21
    points = torch.tensor([[point]], dtype=torch.float64, requires_grad=True)
    observed = value_and_derivatives(model, points, 6)
    reference = high_precision_tanh_derivatives(model, point, 6, decimal_digits=90)
    for order in range(7):
        torch.testing.assert_close(
            observed[order].detach(),
            torch.tensor([[float(reference[order])]], dtype=torch.float64),
            rtol=2.0e-11,
            atol=2.0e-12,
        )


def test_reference_uses_stored_float32_parameters() -> None:
    layer = nn.Linear(1, 1).to(dtype=torch.float32)
    with torch.no_grad():
        layer.weight.fill_(0.1)
        layer.bias.fill_(0.2)
    model = nn.Sequential(layer, nn.Tanh())
    point = -0.4
    reference = high_precision_tanh_derivatives(model, point, 1)
    stored_weight = mp.mpf(float(layer.weight.item()))
    stored_bias = mp.mpf(float(layer.bias.item()))
    argument = stored_weight * mp.mpf(point) + stored_bias
    assert mp.almosteq(reference[0], mp.tanh(argument))
    assert mp.almosteq(reference[1], stored_weight / mp.cosh(argument) ** 2)


def test_high_precision_partial_derivative_for_two_input_network() -> None:
    torch.manual_seed(19)
    model = nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 1)
    ).to(dtype=torch.float64)
    t = torch.tensor([[0.23]], dtype=torch.float64, requires_grad=True)
    xi = torch.tensor([[0.71]], dtype=torch.float64)
    observed = value_and_derivatives(
        lambda current_t: model(torch.cat((current_t, xi), dim=1)),
        t,
        4,
    )
    reference = high_precision_tanh_partial_derivatives(
        model,
        [float(t.item()), float(xi.item())],
        4,
        derivative_input_index=0,
        decimal_digits=90,
    )
    for order in range(5):
        torch.testing.assert_close(
            observed[order].detach(),
            torch.tensor([[float(reference[order])]], dtype=torch.float64),
            rtol=2.0e-11,
            atol=2.0e-12,
        )


def test_derivative_benchmark_preserves_sub_ulp_reference_error() -> None:
    from experiments.neural_high_order_derivative_benchmark import reference_errors

    with mp.workdps(80):
        reference = mp.mpf("1.0000000000000000000000000000000000000001")
    # The caller's default MP precision is deliberately lower than the
    # reference precision; the helper must restore the calculation context.
    absolute, scaled, decimal = reference_errors(1.0, reference, 80)
    assert 0 < absolute < 1e-30
    assert 0 < scaled <= absolute
    assert decimal != "1.0"
