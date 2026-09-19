"""Analytic checks for the auxiliary Cauchy-transform reduction."""

import math

import mpmath as mp

from experiments.auxiliary_cauchy_pinn_pilot import (
    exact_cauchy_exp,
    exact_finite_part_exp,
)


def _finite_part_exp_by_taylor_integral(t: float, order: int) -> float:
    t_mp = mp.mpf(str(t))
    value = mp.exp(t_mp)
    analytic = mp.mpf("0")
    for derivative_order in range(order - 1):
        power = order - derivative_order - 1
        moment = ((-1 - t_mp) ** (-power) - (1 - t_mp) ** (-power)) / power
        analytic += value * moment / mp.factorial(derivative_order)
    analytic += value * mp.log((1 - t_mp) / (1 + t_mp)) / mp.factorial(order - 1)

    def regular(tau: mp.mpf) -> mp.mpf:
        delta = tau - t_mp
        return value * mp.hyper([1], [order + 1], delta) / mp.factorial(order)

    return float(analytic + mp.quad(regular, [-1, t_mp, 1]))


def test_cauchy_transform_derivative_identity_orders_one_to_four() -> None:
    mp.mp.dps = 70
    for t in (-0.73, -0.2, 0.0, 0.41, 0.79):
        for order in (1, 2, 3, 4):
            recurrence_value = exact_finite_part_exp(t, order)
            taylor_value = _finite_part_exp_by_taylor_integral(t, order)
            assert math.isclose(recurrence_value, taylor_value, rel_tol=2.0e-13, abs_tol=2.0e-13)


def test_exact_cauchy_formula_is_real_and_finite_inside_interval() -> None:
    mp.mp.dps = 50
    for t in (-0.9, -0.1, 0.0, 0.55, 0.9):
        value = exact_cauchy_exp(mp.mpf(str(t)))
        assert mp.isfinite(value)
        assert mp.im(value) == 0
