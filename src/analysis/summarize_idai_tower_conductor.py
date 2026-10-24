#!/usr/bin/env python3
"""Summarize the Cyclone Idai tower/conductor fragility test across thresholds and L0.

Reads ``threshold_<T>/{probabilities,results}`` written by
``run_idai_tower_conductor.sbatch`` and the per-branch path lengths from
``power_segment_lengths.py`` (``branches.csv``).  Damaged length is the
expected length of power branches removed by wind, sum_b p_b L_b, split into
OSM transmission lines (line, cable, plant-substation) and gridfinder/other
lines, the quantity of Figure S3.
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

TRANSMISSION = ("line", "cable", "plant-substation")
REPORTED_TRANSMISSION_KM = 1345.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    branches = pd.read_csv(args.branches)
    power = branches[branches["hospital_connection"] == 0][["branch_id", "edge_type", "path_length_km"]]
    transmission = power["edge_type"].isin(TRANSMISSION).to_numpy()
    rows = []
    for folder in sorted(args.result_root.glob("threshold_*"), key=lambda p: int(p.name.split("_")[1])):
        threshold = int(folder.name.split("_")[1])
        for summary in sorted((folder / "results").glob("*_summary.json")):
            tag = summary.name[: -len("_summary.json")]
            p = pd.read_csv(folder / "probabilities" / f"{tag}.csv").rename(columns={"branch_uid": "branch_id"})
            m = power.merge(p, on="branch_id", how="left", validate="one_to_one")
            cut = m["failure_probability"].to_numpy() * m["path_length_km"].to_numpy()
            hospitals = pd.read_csv(folder / "results" / f"{tag}_hospitals.csv")
            modeled = hospitals.loc[hospitals["modeled"] == 1, "failure_probability"]
            result = json.loads(summary.read_text())["result"]
            rule = tag.rsplit("_l0_", 1)[0]
            rows.append({
                "threshold_ms": threshold, "variant": tag, "rule": rule,
                "l0_km": float(tag.rsplit("_l0_", 1)[1][:-2]) if "_l0_" in tag else np.nan,
                "transmission_cut_km": float(cut[transmission].sum()),
                "other_cut_km": float(cut[~transmission].sum()),
                "mean_facility_failure": result["mean_hospital_failure_probability"],
                "facilities_failure_gt_0.5": int((modeled > 0.5).sum()),
                "mean_failed_edges": result["mean_failed_edges"],
            })
    table = pd.DataFrame(rows).sort_values(["rule", "l0_km", "threshold_ms"])
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "idai_tower_conductor_summary.csv", index=False)

    styles = {"conductor": "-", "tower_conductor": "--"}
    colors = dict(zip((0.25, 0.5, 1.0, 2.0), ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for (rule, l0), group in table.groupby(["rule", "l0_km"], dropna=False):
        if rule.startswith("previous"):
            continue
        label = f"{'conductor' if rule == 'conductor' else 'tower + conductor'}, $L_0$ = {l0:g} km"
        for axis, column in zip(axes, ("transmission_cut_km", "mean_facility_failure")):
            axis.plot(group["threshold_ms"], group[column], styles[rule], color=colors[l0], lw=1.5, label=label)
    for rule, color, label in (("previous_line", "black", "previous (line only)"),
                               ("previous_combined", "grey", "previous (combined)")):
        group = table[table["rule"] == rule]
        for axis, column in zip(axes, ("transmission_cut_km", "mean_facility_failure")):
            axis.plot(group["threshold_ms"], group[column], ":", color=color, lw=1.8, marker="o", ms=3, label=label)
    axes[0].axhline(REPORTED_TRANSMISSION_KM, color="black", lw=0.8)
    axes[0].text(18.2, REPORTED_TRANSMISSION_KM, " reported (PDNA) 1,345 km", va="bottom", fontsize=8)
    axes[0].set_ylabel("Damaged transmission lines (km)")
    axes[1].set_ylabel("Mean facility electricity failure")
    for axis, title in zip(axes, ("(a) Transmission lines cut by wind", "(b) Healthcare-facility electricity failure")):
        axis.set_xlabel("Wind threshold (m s$^{-1}$)")
        axis.set_title(title, loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.3, wspace=0.25)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_dir / f"idai_tower_conductor{suffix}", dpi=250, bbox_inches="tight")
    pivot = table[table["threshold_ms"].isin([24, 25])].pivot_table(
        index=["rule", "l0_km"], columns="threshold_ms",
        values=["transmission_cut_km", "mean_facility_failure"], dropna=False)
    print(pivot.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
