"""Build compact figures for the consolidated hypersingular experiment report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def grouped_bars(axis, labels, first, second, first_label, second_label, title) -> None:
    x = np.arange(len(labels))
    width = 0.36
    axis.bar(x - width / 2, first, width, label=first_label, color="#3B6FB6")
    axis.bar(x + width / 2, second, width, label=second_label, color="#E6863B")
    axis.set_xticks(x, labels)
    axis.set_yscale("log")
    axis.set_ylabel("solution RMSE")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=8)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    corrected = load("hypersingular_parameterized_subtraction/sweep_metrics.json")
    uncorrected = load("hypersingular_uncorrected_comparison/sweep_metrics.json")
    quadratic = load("hypersingular_quadratic_representation_comparison/metrics.json")
    cubic = load("hypersingular_cubic_representation_comparison/metrics.json")
    orders = load("singular_order_representation_comparison/metrics.json")
    monomials = load("hypersingular_monomial_examples/metrics.json")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    gamma = [str(run["gamma"]).rstrip("0").rstrip(".") for run in corrected["runs"]]
    grouped_bars(
        axes[0, 0],
        gamma,
        [run["rms_solution_error"] for run in corrected["runs"]],
        [run["rms_solution_error"] for run in uncorrected["runs"]],
        "complete Taylor",
        "uncorrected derivative form",
        "(a) Affine parameter sweep",
    )
    axes[0, 0].set_xlabel(r"parameter $\gamma$")

    polynomial_labels: list[str] = []
    polynomial_endpoint: list[float] = []
    polynomial_taylor: list[float] = []
    for item in quadratic["comparisons"]:
        polynomial_labels.append(rf"$\eta={item['eta']:g}$")
        polynomial_endpoint.append(item["endpoint_rms_solution_error"])
        polynomial_taylor.append(item["taylor_rms_solution_error"])
    for item in cubic["comparisons"][1:]:
        polynomial_labels.append(rf"$\xi={item['xi']:g}$")
        polynomial_endpoint.append(item["endpoint_rms_solution_error"])
        polynomial_taylor.append(item["taylor_rms_solution_error"])
    grouped_bars(
        axes[0, 1],
        polynomial_labels,
        polynomial_endpoint,
        polynomial_taylor,
        "endpoint formula",
        "Taylor formula",
        "(b) Quadratic and cubic targets",
    )
    axes[0, 1].tick_params(axis="x", labelrotation=25)

    grouped_bars(
        axes[1, 0],
        [rf"$m={item['order']}$" for item in orders["comparisons"]],
        [item["endpoint_rms_solution_error"] for item in orders["comparisons"]],
        [item["taylor_rms_solution_error"] for item in orders["comparisons"]],
        "integration by parts",
        "Taylor subtraction",
        "(c) Kernel-order comparison",
    )

    grouped_bars(
        axes[1, 1],
        [r"$u=t$", r"$u=t^2$"],
        [item["endpoint_rms_solution_error"] for item in monomials["comparisons"]],
        [item["taylor_rms_solution_error"] for item in monomials["comparisons"]],
        "endpoint formula",
        "Taylor formula",
        "(d) Direct monomial equations",
    )

    figure.suptitle("Solution-error overview for all experiment groups", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES / "complete_experiment_summary.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "complete_experiment_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
