# Current work

- Goal achieved: the Taylor-subtraction PINN solved the blackboard constant-solution hypersingular equation on `[-0.95, 0.95]`.
- Added isolated implementation and tests; preserved the pre-existing user changes in `Neraul_Singularity_Removal_by_Subtraction.py` and `neural_solver_ln2_approx.py`.
- Verification: 4 tests pass, including constant, affine, quadratic/diagonal-limit, and deterministic training checks.
- Full run: seed 20260916, float64 CPU, 96 training quadrature points, independent 192-point residual validation.
- Result: max solution error `1.2120e-4`, RMS solution error `2.3792e-5`, independent RMS equation residual `7.0999e-4`.
- Outputs: `results/hypersingular_constant_subtraction/metrics.json` and ignored PNG `solution.png` in the same directory.
- Next: if requested, extend the same operator to the paper's parameterized solution `u(t)=1+gamma*t` and its `a(t), b(t)` equation.
