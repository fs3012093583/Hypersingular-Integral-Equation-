# Milestone summaries

## 2026-09-16 - Constant hypersingular PINN works

- Added a float64 tanh PINN using the complete Taylor-subtraction finite-part operator.
- Validated the operator on constant, affine, and quadratic functions; the quadratic case exercises the nonzero regular remainder and the `u''/2` diagonal limit.
- A full deterministic run recovered `u(t)=1` with RMS error `2.3792e-5` and max error `1.2120e-4` on `[-0.95, 0.95]`.
- The reported equation residual uses an independent 192-point Gauss rule, twice the training quadrature resolution.
- Reread `neural_network_solvers/neural_hypersingular_constant_subtraction.py`, its test file, and the result JSON before extending the experiment.
