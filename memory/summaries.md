# Milestone summaries

## 2026-09-16 - Constant hypersingular PINN works

- Added a float64 tanh PINN using the complete Taylor-subtraction finite-part operator.
- Validated the operator on constant, affine, and quadratic functions; the quadratic case exercises the nonzero regular remainder and the `u''/2` diagonal limit.
- A full deterministic run recovered `u(t)=1` with RMS error `2.3792e-5` and max error `1.2120e-4` on `[-0.95, 0.95]`.
- The reported equation residual uses an independent 192-point Gauss rule, twice the training quadrature resolution.
- Reread `neural_network_solvers/neural_hypersingular_constant_subtraction.py`, its test file, and the result JSON before extending the experiment.

## 2026-09-17 - Lower-right parameterized equation sweep works

- Reused the exact same Taylor-subtraction finite-part operator for `a(t)=t^4`, `b(t)=(2-t)(1+t)`, and manufactured solutions `u_gamma(t)=1+gamma*t`.
- Verified that `f_gamma(t)` is fixed analytically and never depends on the neural prediction.
- Six independent fixed-gamma PINNs recovered the paper cases from `gamma=0` through `gamma=10`; all relative RMS solution errors are at most `2.11e-3` on `[-0.95,0.95]`.
- The parameter feature is bounded before entering tanh to avoid artificial saturation at `gamma=10`; the equation itself is unchanged.
- The `gamma=5` run is optimization-sensitive and is the current worst case; a 1000-iteration L-BFGS diagnostic did not materially improve it.
- Reread the parameterized solver, its test file, and `results/hypersingular_parameterized_subtraction/sweep_metrics.json` before further tuning.

## 2026-09-17 - Uncorrected derivative-remainder comparison

- Implemented the literal blackboard derivative-remainder operator without adding the correction needed to recover the Taylor-value finite-part formula.
- Verified exact agreement with the corrected operator for affine functions and the diagnostic identity `H_unc[t^2]-H_corr[t^2]=2`.
- Repeated the same six fixed-gamma runs with identical initialization and optimization settings; ten combined tests pass.
- The uncorrected-trained networks' own RMS equation residuals range from `2.57e-4` to `1.26e-2`, but their residuals under the corrected physical operator are 1.53x to 3.83x larger.
- Their RMS solution errors are worse than the corrected training runs for all six gamma values; the largest degradation is 15.98x at `gamma=10`.
- Because the manufactured target is affine, it exactly satisfies both operators. A non-affine target is the appropriate next experiment for separating operator-model bias from optimizer behavior.

## 2026-09-17 - First equation uncorrected run and full report

- Added an isolated uncorrected solver for the right-upper constant equation and retained the exact corrected-baseline configuration.
- The uncorrected run obtained RMS solution error `1.0011e-4`, 4.21x the corrected baseline; its own RMS residual is `6.3997e-4`, while the same network's corrected-operator RMS residual is `7.2352e-4`.
- A second full run reproduced the complete loss history and all core metrics exactly.
- The project now has 13 passing tests across both equations and both operator forms.
- Consolidated formula derivation, protocols, four experiment groups, residual audits, limitations, reproducibility commands, and next steps into Markdown, LaTeX, and a five-page visually verified PDF.
- Reread `reports/hypersingular_neural_network_experiments_report.md`, the four result JSON files, and the final PDF before extending the study.

## 2026-09-17 - Endpoint-value formula reruns completed

- Implemented the teacher's endpoint-value representation with `u(-1)` and `u(1)` in the boundary terms and the derivative-difference regular integral.
- Verified numerical equivalence to the complete Taylor-subtraction formula on constant, affine, quadratic, and cubic functions; the full test suite now has 23 passing tests.
- Reran the constant equation and all six parameterized equations under their prior matched configurations without overwriting older results.
- Solution error is now reported on the full interval `[-1,1]`, including both endpoints. Equation residual remains on `[-0.95,0.95]` because the raw equation is singular at the endpoints.
- Constant full-domain RMS solution error is `4.1080e-5`; parameterized full-domain RMS errors for gamma `[0,0.1,0.5,1,5,10]` are `[1.0162e-4,1.1184e-4,3.0496e-4,1.7655e-4,7.6768e-3,1.8394e-3]`.
- Exact deterministic reruns reproduced all loss histories and core metrics. Reread the two endpoint solver modules and their result JSON files before report integration or further tuning.

## 2026-09-17 - Non-affine endpoint versus Taylor comparison

- Added a manufactured quadratic solution `u(t)=1+t+eta*t^2` with eta `[0,0.1,0.5,1]` and trained matched PINNs using the endpoint-value and complete Taylor-subtraction representations.
- Both operators reproduce the same fixed analytic RHS. On the same trained network their independently validated RMS difference is only about `1e-9`, so they are numerically equivalent at the operator level.
- At seed 20260916, results agree closely through eta `0.5`; eta `1` produces different L-BFGS branches. Extra seeds reverse or narrow the ordering, showing optimization sensitivity rather than a stable formula advantage.
- Across three eta-1 seeds, mean full-domain solution RMS is `2.7052e-3` for endpoint and `2.7415e-3` for Taylor.
- The formal sweep reproduced exactly, 26 tests pass, and a clean Chinese experiment note was added. Reread the comparison solver, metrics, and `reports/quadratic_endpoint_taylor_comparison.md` before report integration.

## 2026-09-17 - Cubic endpoint versus Taylor comparison

- Extended the manufactured solution to `u(t)=1+t+t^2+xi*t^3` with xi `[0,0.1,0.5,1]` while retaining matched endpoint-form and complete-Taylor training.
- The Taylor regular integral is `2+4*xi*t`; the endpoint derivative-difference integral is `4+6*xi*t`, and the differing boundary terms make the complete operators equal.
- Same-network cross-operator RMS differences remain between `2.5e-10` and `1.5e-9` under independent 192-point validation.
- The apparent winner changes with xi: the two are nearly equal at xi `0.1`, endpoint is better at xi `0.5`, and Taylor is slightly better at xi `1`. This again indicates optimization sensitivity rather than formula inequality.
- The cubic sweep reproduced exactly, 28 tests pass, and the clean comparison report now contains both quadratic and cubic sections.

## 2026-09-17 - First- and third-order singular kernels

- Generalized the complete Taylor-subtraction formula to arbitrary positive integer order and the repeated-integration-by-parts endpoint formula to orders at least two; the order-one endpoint alternative uses a logarithmic integration by parts.
- Implemented and analytically checked the Cauchy principal-value kernel `(tau-t)^-1` and the third-order Hadamard finite-part kernel `(tau-t)^-3` using the shared cubic target `u(t)=1+t+t^2+t^3`.
- Order one reaches full-domain solution RMS `4.8507e-4` for integration by parts and `4.4207e-4` for Taylor subtraction. Its same-network cross difference remains about `1.91e-4` because ordinary Gauss quadrature resolves the retained logarithmic singularity slowly.
- Order three reaches full-domain solution RMS `9.0657e-3` and `1.6924e-2`; same-network operator differences are only `1.35e-7` to `2.73e-7`. A fourth-derivative local expansion stabilizes the cubic divided difference near the diagonal.
- The formal run reproduced exactly, 32 tests pass, and the clean Chinese report contains the general derivation, explicit formulas, results, and interpretation.

## 2026-09-17 - Handwritten linear and quadratic direct equations

- Verified the photographed general `[a,b]` finite-part values for `u(x)=x` and `u(x)=x^2`, then specialized them to the established interval `[-1,1]`.
- Trained the direct equation `FP int u(x)/(x-t)^2 dx=f(t)` without the coefficient functions used in the earlier parameterized equation.
- Full-domain solution RMS for endpoint/Taylor is `1.3951e-5/1.5076e-5` for the linear solution and `2.1594e-5/2.3764e-5` for the quadratic solution.
- Same-network operator differences are `2.0e-10` to `4.6e-10`, confirming the endpoint and complete Taylor formulas remain numerically equivalent when the quadratic regular remainder is nonzero.
- The full run reproduced exactly, 35 tests pass, and the clean report now contains the derivation, full-domain metrics, endpoint errors, and interpretation.

## 2026-09-17 - Complete experiment report

- Consolidated all second-order, endpoint-correction, nonlinear manufactured-solution, first-/third-order kernel, and handwritten monomial experiments into one nine-page Chinese report.
- The report distinguishes correct-form equivalence from the uncorrected-form diagnostic and explicitly separates early `[-0.95,0.95]` solution statistics from later full-domain `[-1,1]` statistics.
- It includes three publication-style summary/error figures, omits internal code and result paths from the body, and is intended for direct supervisor or leadership review.
- XeLaTeX compilation completed successfully, all nine pages passed visual inspection, the PDF text layer passed the forbidden-internal-name check, and all 35 tests remain passing.

## 2026-09-18 - Cancellation-controlled arbitrary-order Taylor operator

- Reframed the prospective paper around a real numerical defect: the Taylor-regularized quotient still suffers catastrophic floating-point cancellation near the diagonal, increasingly severely as the kernel order grows.
- Implemented a general finite-part operator for every positive integer order on a finite interval. Its local continuation of order `p` is activated at the balanced scale `L*eps^(1/(m+p+1))`, obtained by balancing roundoff `eps/delta^m` with truncation `delta^(p+1)`.
- With `u(t)=exp(t)` and 2,048-point Gauss validation, direct order-six evaluation has absolute error `4.5220e-1`, whereas the balanced `p=2` version has error `2.1075e-11`. Orders four and five improve by about `5.8e4` and `8.1e6`.
- A matched neural pilot for orders two, three, and four gives solution RMS errors `8.10e-4`, `7.11e-5`, and `7.60e-5`. Two auxiliary-Cauchy alternatives are mathematically valid but one to three orders of magnitude less accurate and are not recommended as the main contribution.
- The defensible contribution is not a new finite-part identity or the first neural hypersingular solver. It is an order-aware cancellation-control rule, its error balance, and its integration into a general-order neural collocation method.
- The full suite now has 42 passing tests. A Chinese feasibility note and two visually checked stability figures were produced. The saved GPU server closed all SSH connections, so this pass used the local CPU.

## 2026-09-19 - Quality audit and strong-baseline milestone

- The current paper is no longer at the feasibility stage. It contains weighted operator budgets and complete neural training, but its strongest claims need a sharper evidence boundary.
- Corrected the conditional solution-error argument to include quadrature, analytic-moment and floating accumulation errors in addition to quotient-replacement error. A zero quotient error cannot imply zero continuous-operator error.
- Added a mask-first fixed-p baseline. Full single-function three-seed runs completed for m4 and m6; budget setup+training is 5.490 and 70.612 s versus lazy fixed-p2 17.134 and 162.726 s. RMSE means are respectively 4.788e-4 / 4.765e-4 and 1.680e-3 / 1.873e-3 (budget / lazy). Three seeds do not establish non-inferiority; cold calibration generation is excluded from these single-function times.
- Found that older reference benchmarks mixed nominal decimal target positions with dtype-stored positions and rounded MP references before subtraction. Some low-error/high-order results must be regenerated. Operator-budget v2 now separates weighted quotient, moment and sum errors; it has not yet completed testing.
- The manuscript is being shortened by removing duplicate figures and repeated statistics rather than appending another report section. A revised PDF has not yet been produced. Training-trajectory coverage, unresolved-budget handling and optimized-AD comparisons remain open.

## 2026-09-19 - Reference audit completed and concise work draft

- Supersedes the unfinished reference/PDF status above: all stored-coordinate and MP-subtraction operator references, branch/p/c/adaptive comparisons and derivative-error statistics were regenerated; 113 tests pass.
- In float32 m6, quotient-indicator coverage and actual quotient-budget satisfaction are both 15/15, whereas complete discrete-budget satisfaction is 9/15. Analytic-moment RMSE is 4.2485e-2, much larger than weighted quotient RMSE 1.8286e-4. The earlier attribution of full-error failure to neural AD envelopes was unsupported and removed.
- Strong original-calibration paired baselines are complete: single m4/m6 and parameterized m4 show reduced cost at comparable RMSE scales, but three seeds do not prove non-inferiority. All historical data remain intact.
- MP subtraction also changed high-order AD calibration envelopes. Fresh complete paired repeats now run serially in exec session 49669; this is a real live process, not a monitoring placeholder. It executes single m4/m6 first, parameterized m4 second. No concurrent numerical work or dependency edits until termination.
- A 15-page work PDF at `output/pdf/quality_revision_20260919/main.pdf` replaces no older artifact. All pages were visually checked; no internal result paths or unresolved references. Duplicate plots/tables were removed rather than shrinking fonts. The draft explicitly labels its retained old-calibration training numbers pending v2 repeat.
- Remaining scientific priorities are training-trajectory and gradient reliability, a stronger fixed-p/optimized-AD cost frontier, and an independently converged nonmanufactured application. Conditional bounds and finite calibration coverage are not certifications; the publication-quality goal is not complete.

## 2026-09-19 - Calibration-v2 repeat completed; accuracy trade-off disclosed

- Supersedes the live-job status above: exec 49669 exited 0 after 18 single-function and 9 parameterized runs. All protocol, archive and live dependency hashes match. New source/calibration batches are kept separately from the original results.
- Against lazy fixed p2, budget m4 changes mean solution RMSE from 4.76493e-4 to 3.92465e-4 and time from 17.6733 to 5.6848 s. M6 changes RMSE from 1.87252e-3 to 2.11504e-3 and time from 156.0459 to 66.9979 s. The 13.0% mean m6 error increase invalidates any general no-accuracy-loss claim. N=3 is descriptive, not a non-inferiority test.
- Parameterized m4 gives 24.4335 versus 6.4580 s including calibration, with held-out RMSE 1.99686e-2 versus 1.99622e-2. Terminal all-direct selections support only a mild interpolation case.
- All nine budget runs show required remainder-derivative AD magnitudes above the previous B at the first 100-step refresh. This observable stale-envelope issue is distinct from actual quotient-budget failure, which still needs checkpoint MP auditing.
- Fixed methodological discrepancies: training uses monotone online B (65 or 33 calibration points, safety 4), not the earlier static MP envelope; budget training disables the fixed trust radius; discrete adaptive choices are detached during backprop. These actual settings are now stated in the paper.
- Final revised PDF is `output/pdf/quality_revision_calibration_v2_20260919/main.pdf`, 15 pages, all visually checked. Main tables use v2; old branch ablation explicitly historical. Original PDFs preserved. 115 tests pass; table builder rejects incompatible paired configurations.
- Next: implement the independent checkpoint/indicator/gradient audit in `reports/training_reliability_audit_plan.md`, then fair fixed-p and optimized-AD accuracy-cost frontiers. No live processes or ongoing benchmark freezes; publication-quality goal remains active.
