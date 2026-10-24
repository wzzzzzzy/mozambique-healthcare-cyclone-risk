#!/usr/bin/env python3
"""Figure S1 (direct versus indirect failures by cyclone) with the regional damage function.

Identical to the manuscript Figure S1 workflow (``prepare_failure_figure_inputs.py``
and the Figure 4 notebook) except for the direct term: the saved building
damage is replaced by the Eberenz et al. (2021) South-Indian (TDR) damage
ratio from ``cyclone_direct_damage.py``.  The hospital/event cells evaluated
are those of the original direct matrix, so only the damage function changes.

Two direct measures are drawn:
* ``nonfunctional``: facilities with damage ratio > 0.2 (Espejo et al., 2025);
* ``ratio``: sum of damage ratios (expected count, as in the original S1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import prepare_failure_figure_inputs as pfi
from capacity_ablation_figure import load_hospital_data, load_power_summary
from synthetic_tracks import expected_track_columns

NONFUNCTIONAL = 0.2
Y_LIMIT = 350
LABELS = {
    "nonfunctional": "Direct failures (ratio > 0.2)",
    "ratio": "Total direct (wind) damage ratio",
}
PANELS = ("(a) Electricity", "(b) Water", "(c) Communication")


def plot(summaries: dict[str, pd.DataFrame], x_label: str, output: Path) -> float:
    # Sherlock has no Arial, so seaborn falls back to DejaVu Sans there; pin it everywhere.
    sns.set_theme(style="white", font="DejaVu Sans")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharex=True, sharey=True)
    x_max = max(float(frame["wind"].max()) for frame in summaries.values())
    x_limit = float(np.ceil(x_max / 20.0) * 20.0)
    for axis, (title, frame) in zip(axes, summaries.items()):
        sns.scatterplot(data=frame, x="wind", y="indirect", hue="cyclone_intensity", palette="viridis",
                        s=55, edgecolor="black", linewidth=0.5, legend=axis is axes[2], ax=axis)
        axis.plot([0, x_limit], [0, x_limit], "--", color="black", linewidth=1)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_xlabel(x_label)
        axis.set_xlim(0, x_limit)
        axis.set_ylim(0, Y_LIMIT)
        axis.grid(False)
    axes[0].set_ylabel("Total indirect failures")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    if axes[2].get_legend() is not None:
        axes[2].legend(title="Cyclone intensity", loc="center right")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return x_limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-summary", type=Path)
    parser.add_argument("--water-root", type=Path)
    parser.add_argument("--communication-root", type=Path)
    parser.add_argument("--direct-root", type=Path, help="original synthetic_new_hospital layers")
    parser.add_argument("--regional-ratio", type=Path, help="damage_ratio_si_tdr.csv.gz")
    parser.add_argument("--hospital-data", type=Path)
    parser.add_argument("--storm-root", type=Path)
    parser.add_argument("--replot-from", type=Path, help="redraw from saved cyclone_summary_*.csv only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if args.replot_from:
        args.output_dir.mkdir(parents=True)
        for variant, label in LABELS.items():
            summaries = {title: pd.read_csv(args.replot_from / f"cyclone_summary_{title.split(') ')[1].lower()}"
                                                                f"_{variant}.csv") for title in PANELS}
            plot(summaries, label, args.output_dir / f"figure_S1_direct_indirect_si_tdr_{variant}")
        return 0

    events = list(expected_track_columns())
    hospitals = load_hospital_data(args.hospital_data)
    original = pfi.load_direct_matrix(args.direct_root, hospitals)
    regional = pd.read_csv(args.regional_ratio, dtype={"node_id": str}).set_index("node_id")[events]
    regional = regional.reindex(index=original.index, columns=original.columns)
    if (original.notna() & regional.isna()).to_numpy().any():
        raise ValueError("regional damage ratio is missing for an originally evaluated cell")
    regional = regional.where(original.notna())
    direct = {"nonfunctional": (regional > NONFUNCTIONAL).astype(float).where(regional.notna()),
              "ratio": regional}

    public = load_power_summary(args.public_summary, require_complete=True, allow_unavailable=True)
    indirect = {
        "(a) Electricity": public.set_index(public["node_id"].astype(str))[events].where(lambda v: v >= 0.0),
        "(b) Water": pfi.load_downstream_matrix(args.water_root, "water"),
        "(c) Communication": pfi.load_downstream_matrix(args.communication_root, "comm"),
    }
    intensities = pfi.load_intensities(args.storm_root)

    args.output_dir.mkdir(parents=True)
    report = {"function": "Eberenz et al. 2021 SI (TDR): V_thresh 25.7 m/s, V_half 52.4 m/s, 1-min = 10-min / 0.93",
              "nonfunctional_threshold": NONFUNCTIONAL,
              "evaluated_cells": "cells evaluated in the original direct matrix",
              "variants": {}}
    for variant, matrix in direct.items():
        summaries = {title: pfi.cyclone_summary(values, matrix, intensities) for title, values in indirect.items()}
        for title, frame in summaries.items():
            system = title.split(") ")[1].lower()
            frame.to_csv(args.output_dir / f"cyclone_summary_{system}_{variant}.csv", index=False)
        x_limit = plot(summaries, LABELS[variant], args.output_dir / f"figure_S1_direct_indirect_si_tdr_{variant}")
        report["variants"][variant] = {
            "x_limit": x_limit,
            **{title: {"max_direct": float(frame["wind"].max()),
                       "cyclones_direct_gt_indirect": int((frame["wind"] > frame["indirect"]).sum()),
                       "total_direct": float(frame["wind"].sum()), "total_indirect": float(frame["indirect"].sum())}
               for title, frame in summaries.items()},
        }
    (args.output_dir / "figure_S1_si_tdr_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
