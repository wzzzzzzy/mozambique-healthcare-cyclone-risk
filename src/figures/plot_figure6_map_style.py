#!/usr/bin/env python3
"""Redraw Figure 6 (all-grid ablation) in the map style of Figures 2 and 3.

Reads the per-facility table written by ``capacity_ablation_figure.py``
(``*_per_hospital.csv``) and plots the annualized electricity disruption index
(yr^-1) with the Figure 2/3 conventions: gainsboro land with a black outline,
BuGn on a fixed 0-0.05 range, borderless markers, silver "data missing"
points, "(a)"-style titles, vertical colorbars, and a legend below the maps.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from capacity_ablation_figure import load_boundary

ANNUAL_FREQUENCY = 2703 / 10_000
INDEX_MAX = 0.05
MARKERS = {"public": "o", "solar_only": "^"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-hospital", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    table = pd.read_csv(args.per_hospital)
    facilities = gpd.GeoDataFrame(
        table, geometry=gpd.points_from_xy(table["longitude"], table["latitude"]), crs="EPSG:4326"
    )
    facilities["current"] = facilities["baseline_mean"].where(facilities["baseline_mean"] >= 0) * ANNUAL_FREQUENCY
    facilities["all_grid"] = facilities["all_grid_mean"].where(facilities["all_grid_mean"] >= 0) * ANNUAL_FREQUENCY
    facilities["change"] = facilities["delta"] * ANNUAL_FREQUENCY
    mozambique = load_boundary(args.boundary)

    change_limit = math.ceil(np.nanmax(np.abs(facilities["change"])) * 100) / 100
    index_norm = Normalize(0, INDEX_MAX)
    change_norm = Normalize(-change_limit, change_limit)

    fig = plt.figure(figsize=(10.4, 4.4))
    width, height, bottom = 0.25, 0.80, 0.14
    axes = [fig.add_axes([left, bottom, width, height]) for left in (0.01, 0.28, 0.625)]
    panels = [
        (axes[0], "current", "BuGn", index_norm, "(a) Current system"),
        (axes[1], "all_grid", "BuGn", index_norm, "(b) All facilities on public grid"),
        (axes[2], "change", "RdBu_r", change_norm, "(c) Change, (b) − (a)"),
    ]
    for axis, column, cmap, norm, title in panels:
        mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.8)
        missing = facilities[facilities[column].isna()]
        if not missing.empty:
            missing.plot(ax=axis, color="silver", edgecolor="black", markersize=14, linewidth=0.25)
        for source, marker in MARKERS.items():
            valid = facilities[facilities[column].notna() & facilities["baseline_source"].eq(source)]
            valid.plot(ax=axis, column=column, cmap=cmap, norm=norm, marker=marker,
                       markersize=15 if marker == "o" else 18, linewidth=0)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set(xlim=(30, 41), ylim=(-27, -10))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_aspect("equal")

    index_bar = fig.colorbar(cm.ScalarMappable(norm=index_norm, cmap="BuGn"),
                             cax=fig.add_axes([0.545, 0.20, 0.012, 0.66]))
    index_bar.set_label("Annualized electricity disruption index (yr$^{-1}$)", fontsize=9)
    index_bar.set_ticks(np.arange(0, INDEX_MAX + 1e-9, 0.01),
                        labels=[f"{x:.2f}" for x in np.arange(0, INDEX_MAX + 1e-9, 0.01)])
    change_bar = fig.colorbar(cm.ScalarMappable(norm=change_norm, cmap="RdBu_r"),
                              cax=fig.add_axes([0.885, 0.20, 0.012, 0.66]))
    change_bar.set_label("Change in index (yr$^{-1}$)", fontsize=9)
    step = 0.02 if change_limit >= 0.04 else 0.01
    ticks = np.arange(-math.floor(change_limit / step) * step, change_limit + 1e-9, step)
    change_bar.set_ticks(ticks, labels=[f"{x:+.2f}".replace("-", "\u2212") if abs(x) > 1e-12 else "0"
                                        for x in ticks])

    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#6a9e8e", label="Currently on public grid"),
            Line2D([], [], marker="^", linestyle="none", color="#6a9e8e", label="Currently self-reliant (solar)"),
            Line2D([], [], marker="o", linestyle="none", color="silver", markeredgecolor="black",
                   label="Data missing"),
        ],
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.47, 0.0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    above = int(((facilities["current"] > INDEX_MAX) | (facilities["all_grid"] > INDEX_MAX)).sum())
    print(f"change colour limit ±{change_limit}; facilities above {INDEX_MAX} in (a) or (b): {above}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
