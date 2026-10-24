#!/usr/bin/env python3
"""Sensitivity overview: mean annualized electricity disruption index per scenario.

Each bar splits the index into the shortfall already present in the undamaged
network (grey) and the storm-attributable part (blue); labels give the change
relative to the baseline.  Reads ``sensitivity_summary.json`` written by
``hybrid_sensitivity.py summarize`` (or its v2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ANNUAL = 2703 / 10_000
ORDER = [("baseline", "Baseline"), ("alpha_0.50", "α = 0.50"), ("demand_cap80", "Demand 1,001.76 MW (80% cap)"),
         ("generation_0.90", "Generation × 0.90"), ("generation_1.10", "Generation × 1.10"),
         ("ratings_1.25", "Ratings × 1.25"), ("demand_1093", "Demand 1,093 MW"), ("alpha_0.00", "α = 0"),
         ("ratings_0.75", "Ratings × 0.75"), ("ratings_0.50", "Ratings × 0.50")]
INK, STORM, DEFICIT, MUTED = "#1d2329", "#2a78d6", "#b7bec6", "#5f6b76"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True, help="sensitivity_summary.json")
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")
    rows = {x["scenario"]: x for x in json.loads(args.summary.read_text())["scenarios"]}

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    n = len(ORDER)
    totals = [rows[k]["annual_mean_facility_failure"] for k, _ in ORDER]
    x_max = max(totals) * 1.28
    for i, (key, _) in enumerate(ORDER):
        row = rows[key]
        total = row["annual_mean_facility_failure"]
        deficit = ANNUAL * row["intact_mean_facility_failure"]
        y = n - 1 - i
        ax.barh(y, deficit, color=DEFICIT, height=0.62, edgecolor="white", linewidth=0.8)
        ax.barh(y, max(total - deficit, 0), left=deficit, color=STORM, height=0.62, edgecolor="white", linewidth=0.8)
        change = "" if key == "baseline" else f"  ({row['relative_change_vs_baseline']:+.1%})"
        ax.text(total + 0.01 * x_max, y, f"{total:.4f}{change}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in ORDER][::-1], fontsize=9)
    ax.axhline(3.5, color="#d6dce1", lw=1)
    ax.axvline(rows["baseline"]["annual_mean_facility_failure"], color=INK, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("Mean annualized electricity disruption index (yr$^{-1}$)", fontsize=9.5)
    ax.set_xlim(0, x_max)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend(handles=[Patch(color=STORM, label="Storm-attributable"),
                       Patch(color=DEFICIT, label="Shortfall in the undamaged network")],
              loc="upper right", fontsize=8.5, frameon=False)
    ax.text(0.99 * x_max, 2.55, "Below the line, the undamaged\nnetwork cannot serve all demand:\none 66 kV Matola "
            "line (125 MW)\ncaps service for the whole grid", ha="right", va="center", fontsize=7.6, color=MUTED,
            linespacing=1.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
