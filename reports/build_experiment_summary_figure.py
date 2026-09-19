"""Build the consolidated comparison figure for the hypersingular PINN report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures"


def load_json(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def main() -> None:
    constant_corrected = load_json("results/hypersingular_constant_subtraction/metrics.json")
    constant_uncorrected = load_json("results/hypersingular_constant_uncorrected/metrics.json")
    parameterized_corrected = load_json("results/hypersingular_parameterized_subtraction/sweep_metrics.json")
    parameterized_uncorrected = load_json("results/hypersingular_uncorrected_comparison/sweep_metrics.json")

    corrected_runs = {float(run["gamma"]): run for run in parameterized_corrected["runs"]}
    uncorrected_runs = {float(run["gamma"]): run for run in parameterized_uncorrected["runs"]}
    gammas = [float(value) for value in parameterized_corrected["gammas"]]
    indices = np.arange(len(gammas))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.labelweight": "bold",
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 15,
            "axes.linewidth": 1.3,
            "lines.linewidth": 2.4,
            "lines.markersize": 9,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    )

    colors = {
        "corrected": "#0072B2",
        "uncorrected": "#D55E00",
        "same_network": "#CC79A7",
        "neutral": "#4D4D4D",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.8), constrained_layout=True)

    ax = axes[0, 0]
    values = [constant_corrected["rms_solution_error"], constant_uncorrected["rms_solution_error"]]
    bars = ax.bar(
        [0, 1], values, width=0.62,
        color=[colors["corrected"], colors["uncorrected"]],
        edgecolor="black", linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Corrected", "Uncorrected"], fontweight="bold")
    ax.set_ylabel("Solution RMSE")
    ax.set_title("(a) Equation 1: solution error")
    ax.grid(axis="y", which="both", alpha=0.25)

    ax = axes[0, 1]
    labels = ["Corrected", "Uncorrected", "Uncorrected result\n(corrected formula)"]
    values = [
        constant_corrected["rms_equation_residual"],
        constant_uncorrected["rms_uncorrected_equation_residual"],
        constant_uncorrected["rms_corrected_equation_residual"],
    ]
    bars = ax.bar(
        np.arange(3), values, width=0.62,
        color=[colors["corrected"], colors["uncorrected"], colors["same_network"]],
        edgecolor="black", linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3), labels, fontweight="bold", fontsize=15)
    ax.set_ylabel("Equation residual RMS")
    ax.set_title("(b) Equation 1: equation residual")
    ax.grid(axis="y", which="both", alpha=0.25)

    ax = axes[1, 0]
    corrected_solution = [corrected_runs[gamma]["rms_solution_error"] for gamma in gammas]
    uncorrected_solution = [uncorrected_runs[gamma]["rms_solution_error"] for gamma in gammas]
    ax.semilogy(indices, corrected_solution, "o-", color=colors["corrected"], label="Corrected formula")
    ax.semilogy(indices, uncorrected_solution, "s--", color=colors["uncorrected"], label="Uncorrected formula")
    ax.set_xticks(indices, [f"{gamma:g}" for gamma in gammas], fontweight="bold")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Solution RMSE")
    ax.set_title("(c) Equation 2: solution error")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    corrected_residual = [corrected_runs[gamma]["rms_equation_residual"] for gamma in gammas]
    uncorrected_residual = [uncorrected_runs[gamma]["rms_uncorrected_equation_residual"] for gamma in gammas]
    same_network_corrected = [uncorrected_runs[gamma]["rms_corrected_physical_residual"] for gamma in gammas]
    ax.semilogy(indices, corrected_residual, "o-", color=colors["corrected"], label="Corrected formula")
    ax.semilogy(indices, uncorrected_residual, "s--", color=colors["uncorrected"], label="Uncorrected formula")
    ax.semilogy(
        indices, same_network_corrected, "D-.", color=colors["same_network"],
        label="Uncorrected solution in corrected formula",
    )
    ax.set_xticks(indices, [f"{gamma:g}" for gamma in gammas], fontweight="bold")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Equation residual RMS")
    ax.set_title("(d) Equation 2: equation residual")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="upper left")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"corrected_vs_uncorrected_summary_v3.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
