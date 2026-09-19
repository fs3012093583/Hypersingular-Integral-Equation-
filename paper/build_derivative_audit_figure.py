"""Plot the corrected MP-subtraction derivative audit without old cost probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("derivative audit requires schema_version=2")
    if payload["configuration"].get("reference_error_policy") != (
        "mpmath subtraction and scaling before float serialization"
    ):
        raise ValueError("missing high-precision error-subtraction policy")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "axes.labelsize": 22,
        "axes.labelweight": "bold", "xtick.labelsize": 20,
        "ytick.labelsize": 20, "legend.fontsize": 16,
        "lines.linewidth": 2, "lines.markersize": 6,
    })
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    for dtype, color, marker in (("float32", "#CC79A7", "s"), ("float64", "#0072B2", "o")):
        rows = sorted(
            (row for row in payload["summary"] if row["dtype"] == dtype),
            key=lambda row: row["order"],
        )
        if [row["order"] for row in rows] != list(range(11)):
            raise ValueError(f"incomplete derivative orders for {dtype}")
        if any(row["sample_count"] != 15 for row in rows):
            raise ValueError("expected three seeds times five targets")
        for statistic, label, line, fill in (
            ("maximum_scaled_error", "max", "-", "full"),
            ("median_scaled_error", "median", "--", "none"),
        ):
            values = [float(row[statistic]) for row in rows]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("log plot requires finite positive audit errors")
            axis.semilogy(range(11), values, color=color, marker=marker,
                          linestyle=line, fillstyle=fill, label=f"{dtype}, {label}")
    axis.set_xlabel("derivative order k")
    axis.set_ylabel("scaled AD error")
    axis.set_xticks(range(0, 11, 2))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="center left")
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white", transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/neural_high_order_derivatives_ref_v2/metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/derivative_error_audit_v2")
    args = parser.parse_args()
    build(args.source, args.output)
