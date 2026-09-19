"""Matched fixed-p implementation audit, without regenerating manuscript figures.

All runs are sequential on one CPU thread.  Methods are rotated across seeds
to reduce systematic run-order bias.  Existing experiment outputs are never
overwritten: the output directory must be new.  Setup plus training includes
initial derivative-envelope work; the single-function experiment still uses
the explicitly identified precomputed AD-error calibration artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import torch

from conventional_solvers.taylor_spectral_collocation import gauss_legendre_rule
from experiments.general_taylor_neural_benchmark import _interior_chebyshev_points
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
from neural_network_solvers.general_taylor_finite_part import balanced_threshold


METHODS = ("fixed_p2", "lazy_fixed_p2", "operator_budget")


def geometry(config) -> dict[str, float | int]:
    points = _interior_chebyshev_points(
        config.collocation_points, config.collocation_limit, torch.float64
    )
    nodes, _ = gauss_legendre_rule(config.training_quadrature_points, torch.float64)
    delta = nodes.reshape(1, -1) - points
    threshold = balanced_threshold( config.order, 2, dtype=torch.float64, interval_length=2.0)
    mask = delta.abs() <= threshold
    return {
        "fixed_p2_threshold": threshold,
        "near_pairs": int(mask.sum()),
        "total_pairs": mask.numel(),
        "near_rows": int(mask.any(dim=1).sum()),
        "total_rows": config.collocation_points,
        "minimum_distance": float(delta.abs().min()),
    }


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["task"], record["config"]["order"], record["method"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (task, order, method), group in sorted(groups.items()):
        row = {"task": task, "order": order, "method": method, "n": len(group)}
        for key in (
            "solution_rmse", "residual_rmse", "training_seconds_including_envelope_refresh",
            "method_setup_seconds", "method_setup_and_training_seconds",
            "solve_seconds_before_validation", "saved_tensor_payload_bytes",
        ):
            values = [float(record[key]) for record in group]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("single", "parameterized"), required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260918, 20260919, 20260920])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--derivative-metrics", type=Path,
                        default=Path("results/neural_high_order_derivatives_ref_v2/metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = (
        "neural_network_solvers/general_taylor_finite_part.py",
        "experiments/sequential_tolerance_end_to_end_benchmark.py",
        "experiments/parameterized_operator_budget_benchmark.py",
        "experiments/lazy_fixed_quality_benchmark.py",
        str(args.derivative_metrics),
        "neural_network_solvers/high_precision_tanh_reference.py",
        "experiments/general_taylor_neural_benchmark.py",
        "conventional_solvers/taylor_spectral_collocation.py",
    )
    metadata = {
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "device": "cpu", "threads": torch.get_num_threads(), "deterministic_algorithms": True,
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        "time_scope": "method setup plus training; solve includes model/RHS construction, excludes validation and timing probes",
        "calibration_scope": "parameterized: per-run high-precision calibration included; single: existing calibration artifact reused, original generation excluded",
        "memory_scope": "summed autograd saved-tensor payload, not allocator peak memory",
        "paired_controls": "initial weights, equation, nodes, RHS, Adam/L-BFGS, canonical validation",
        "run_order": "cyclic rotation across seeds; no simultaneous benchmark processes",
        "source_snapshot": "source_snapshot.tar",
    }
    with tarfile.open(args.output_dir / "source_snapshot.tar", "w") as archive:
        for source in sources:
            archive.add(source, arcname=source)
    (args.output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run = run_single if args.task == "single" else run_parameterized
    base = EndToEndConfig() if args.task == "single" else ParameterizedConfig()
    if args.task == "single":
        base = replace(base, derivative_metrics_path=str(args.derivative_metrics))
    smoke = SINGLE_PRESETS["smoke"] if args.task == "single" else PARAMETER_PRESETS["smoke"]
    # Exclude one-off library and optimizer startup from every paired run.
    for method in METHODS:
        run(replace(base, **smoke, representation=method, order=4))
    records = []
    for order in args.orders:
        for seed_index, seed in enumerate(args.seeds):
            fingerprints = set()
            methods = METHODS[seed_index % 3:] + METHODS[:seed_index % 3]
            for method in methods:
                config = replace(base, representation=method, order=order, seed=seed)
                if args.smoke:
                    config = replace(config, **smoke)
                print(f"START {args.task} m={order} seed={seed} method={method}", flush=True)
                record = run(config, verbose=True)
                record["task"] = args.task
                record["fixed_p2_geometry"] = geometry(config)
                record["solution_rmse"] = record[
                    "solution_rmse_closed_interval" if args.task == "single" else "heldout_solution_rmse"
                ]
                record["residual_rmse"] = record[
                    "validation_residual_rmse" if args.task == "single" else "heldout_residual_rmse"
                ]
                record["saved_tensor_payload_bytes"] = record["post_training_cost"]["median_saved_tensor_payload_bytes"]
                fingerprints.add(record["initial_model_fingerprint"])
                if len(fingerprints) != 1:
                    raise RuntimeError("paired initial weights differ")
                records.append(record)
                path = args.output_dir / f"{args.task}_m{order}_seed{seed}_{method}.json"
                path.write_text(json.dumps(record, indent=2) + "\n")
                payload = {"protocol": metadata, "records": records, "summary": summarize(records)}
                (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({key: record[key] for key in (
                    "method", "solution_rmse", "method_setup_and_training_seconds", "saved_tensor_payload_bytes"
                )}), flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
