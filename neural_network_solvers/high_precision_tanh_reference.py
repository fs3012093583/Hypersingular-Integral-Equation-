"""High-precision Taylor-jet reference for scalar-input tanh networks.

The reference deliberately evaluates the *quantized parameters* stored by a
PyTorch model.  Callers comparing dtypes should likewise pass the stored input
coordinate (for example ``float(input_tensor.item())``).  The resulting
float32 comparison measures floating-point evaluation and automatic-
differentiation error, rather than also charging the method for one-time
parameter or coordinate conversion.

Taylor coefficients are propagated with ``mpmath``.  If

    z(s) = sum_n z_n s**n,  y(s) = tanh(z(s)) = sum_n y_n s**n,

then ``y' = z' (1-y**2)`` gives a triangular recurrence for ``y_{n+1}``.
The returned derivatives are ``n! y_n``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mpmath as mp
import torch
from torch import nn


Jet = list[mp.mpf]


def _mp_parameter(value: torch.Tensor) -> mp.mpf:
    """Convert one stored tensor scalar without introducing decimal rounding."""
    return mp.mpf(float(value.detach().cpu().item()))


def tanh_jet(input_jet: Sequence[mp.mpf]) -> Jet:
    """Return Taylor coefficients of ``tanh(input_jet)``."""
    if not input_jet:
        raise ValueError("input_jet must contain at least its constant coefficient")
    highest_order = len(input_jet) - 1
    output = [mp.mpf("0") for _ in range(highest_order + 1)]
    output[0] = mp.tanh(input_jet[0])
    for degree in range(highest_order):
        rhs = mp.mpf("0")
        for input_degree in range(degree + 1):
            residual_degree = degree - input_degree
            square_coefficient = mp.fsum(
                output[j] * output[residual_degree - j]
                for j in range(residual_degree + 1)
            )
            one_minus_square = (
                mp.mpf("1") - square_coefficient
                if residual_degree == 0
                else -square_coefficient
            )
            rhs += (
                (input_degree + 1)
                * input_jet[input_degree + 1]
                * one_minus_square
            )
        output[degree + 1] = rhs / (degree + 1)
    return output


def _linear_jets(layer: nn.Linear, inputs: Sequence[Jet]) -> list[Jet]:
    if len(inputs) != layer.in_features:
        raise ValueError("jet width does not match linear layer input width")
    highest_order = len(inputs[0]) - 1
    if any(len(jet) != highest_order + 1 for jet in inputs):
        raise ValueError("all jets must have the same order")
    outputs: list[Jet] = []
    for output_index in range(layer.out_features):
        coefficients: Jet = []
        for degree in range(highest_order + 1):
            value = mp.fsum(
                _mp_parameter(layer.weight[output_index, input_index])
                * inputs[input_index][degree]
                for input_index in range(layer.in_features)
            )
            if degree == 0 and layer.bias is not None:
                value += _mp_parameter(layer.bias[output_index])
            coefficients.append(value)
        outputs.append(coefficients)
    return outputs


def high_precision_tanh_partial_derivatives(
    model: nn.Module,
    point: float | Sequence[float],
    highest_order: int,
    *,
    derivative_input_index: int = 0,
    decimal_digits: int = 100,
) -> list[mp.mpf]:
    """Evaluate derivatives along one input coordinate of a tanh MLP.

    The supported model is a sequential composition of ``nn.Linear`` and
    ``nn.Tanh`` modules.  Nested ``nn.Sequential`` containers are flattened by
    iterating over their leaf modules.  ``point`` may contain one value per
    network input.  Only ``derivative_input_index`` receives a unit first-order
    Taylor coefficient; all other input coordinates remain fixed.
    """
    if highest_order < 0:
        raise ValueError("highest_order must be non-negative")
    if decimal_digits < 30:
        raise ValueError("decimal_digits must be at least 30")

    leaves = [module for module in model.modules() if len(list(module.children())) == 0]
    unsupported = [module.__class__.__name__ for module in leaves if not isinstance(module, (nn.Linear, nn.Tanh))]
    if unsupported:
        raise TypeError(f"unsupported leaf modules: {unsupported}")
    linear_layers = [module for module in leaves if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("model must contain at least one linear layer")
    input_width = linear_layers[0].in_features
    if isinstance(point, Sequence):
        coordinates = [float(value) for value in point]
    else:
        coordinates = [float(point)]
    if len(coordinates) != input_width:
        raise ValueError(
            f"point has {len(coordinates)} coordinates but model expects {input_width}"
        )
    if not 0 <= derivative_input_index < input_width:
        raise ValueError("derivative_input_index is outside the model input range")

    with mp.workdps(decimal_digits):
        jets: list[Jet] = []
        for input_index, coordinate in enumerate(coordinates):
            input_jet = [mp.mpf(coordinate)]
            if highest_order >= 1:
                input_jet.append(
                    mp.mpf("1")
                    if input_index == derivative_input_index
                    else mp.mpf("0")
                )
            input_jet.extend(
                mp.mpf("0") for _ in range(max(0, highest_order - 1))
            )
            jets.append(input_jet[: highest_order + 1])
        for module in leaves:
            if isinstance(module, nn.Linear):
                jets = _linear_jets(module, jets)
            else:
                jets = [tanh_jet(jet) for jet in jets]
        if len(jets) != 1:
            raise ValueError("model output must be scalar")
        coefficients = jets[0]
        return [mp.mpf(math.factorial(order)) * coefficients[order] for order in range(highest_order + 1)]


def high_precision_tanh_derivatives(
    model: nn.Module,
    point: float,
    highest_order: int,
    *,
    decimal_digits: int = 100,
) -> list[mp.mpf]:
    """Backward-compatible scalar-input wrapper."""
    return high_precision_tanh_partial_derivatives(
        model,
        point,
        highest_order,
        derivative_input_index=0,
        decimal_digits=decimal_digits,
    )
