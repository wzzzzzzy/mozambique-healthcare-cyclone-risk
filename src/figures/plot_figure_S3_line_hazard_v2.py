#!/usr/bin/env python3
"""Figure S3 with the line-hazard v2 main case.

Damaged transmission-line length for Cyclone Idai against the wind threshold:
the adopted model (towers + length-normalized conductor, L0 = 0.25 km) and the
previous line-only model, both as the expected length of OSM transmission
branches removed by wind (``idai_tower_conductor_summary.csv``), with the
reported 1,345 km and the adopted 24 m/s threshold.  Style follows
``plot_idai_threshold_lengths.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from plot_idai_threshold_lengths import GRID, GUIDE, MODEL, ORIGINAL, REPORTED, crossing

ADOPTED_THRESHOLD = 24
REPORTED_KM = 1345.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True, help="idai_tower_conductor_summary.csv")
    parser.add_argument("--l0-km", type=float, default=0.25)
    parser.add_argument("--show-previous", action="store_true", help="also draw the previous line-only model")
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    table = pd.read_csv(args.summary)
    model = table[(table["rule"] == "tower_conductor") & np.isclose(table["l0_km"], args.l0_km)].sort_values("threshold_ms")
    previous = table[table["rule"] == "previous_line"].sort_values("threshold_ms")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#444444",
                         "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    ax.axvline(ADOPTED_THRESHOLD, color=GUIDE, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
    ax.axhline(REPORTED_KM, color=REPORTED, lw=1.6, ls=(0, (6, 3)), zorder=2, label=f"Reported ({REPORTED_KM:,.0f} km)")
    if args.show_previous:
        ax.plot(previous["threshold_ms"], previous["transmission_cut_km"], color=ORIGINAL, lw=1.6, marker="s",
                ms=3.6, mec="white", mew=0.6, zorder=3, label="Previous line model")
    t = model["threshold_ms"].to_numpy(float)
    y = model["transmission_cut_km"].to_numpy(float)
    ax.plot(t, y, color=MODEL, lw=2.0, marker="o", ms=4.2, mec="white", mew=0.7, zorder=4,
            label="Modeled" if not args.show_previous else f"Tower + conductor ($L_0$ = {args.l0_km:g} km)")
    x = crossing(t, y, REPORTED_KM)
    if x is not None:
        ax.plot([x], [REPORTED_KM], marker="o", ms=6.5, mfc="white", mec=MODEL, mew=1.5, zorder=5)
        ax.annotate(f"{x:.1f} m s$^{{-1}}$", (x, REPORTED_KM), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=MODEL)
    shown = [y] + ([previous["transmission_cut_km"].to_numpy(float)] if args.show_previous else [])
    top = max(max(v.max() for v in shown), REPORTED_KM)
    bottom = min(v.min() for v in shown)
    lo = max(0, np.floor(bottom * 0.85 / 200) * 200)
    hi = np.ceil(top * 1.08 / 200) * 200
    ax.set_ylim(lo, hi)
    ax.text(ADOPTED_THRESHOLD + 0.15, hi - 0.02 * (hi - lo), "adopted\nthreshold", fontsize=8, color="#6b747c",
            va="top", ha="left", linespacing=1.1)
    ax.set_xlim(17.5, 30.5)
    ax.set_xticks(range(18, 31, 2))
    ax.set_xlabel("Wind threshold $\\nu_1$ (m s$^{-1}$)", fontsize=11.5)
    ax.set_ylabel("Length of damaged\ntransmission lines (km)", fontsize=11.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, handlelength=2.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"crossing of reported length at {x:.2f} m/s" if x is not None else "no crossing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
