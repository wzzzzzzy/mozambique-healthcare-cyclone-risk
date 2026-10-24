#!/usr/bin/env python3
"""Plot Figure S3: destroyed transmission-line length against the Idai threshold.

Destroyed length follows the original connected-component definition: lines cut
by wind plus lines left in islands with no supply. The hybrid model's
length-weighted partial service loss is written to the data CSV but not drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

MODEL = "#111111"
ORIGINAL = "#8a96a3"
REPORTED = "#d62728"
GUIDE = "#9aa3ab"
GRID = "#e6e9ec"
ANALYSIS_THRESHOLD = 25


def load_hybrid(length_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(length_dir.glob("threshold_*.json")):
        result = json.loads(path.read_text())
        values = result["result"]
        rows.append({
            "threshold_ms": int(result["track_id"]),
            "cut_by_wind_km": values["mean_removed_failure_length_km"],
            "no_supply_km": values["mean_isolated_zero_service_length_km"],
            "partial_service_loss_km": values["mean_cascade_failure_length_km"],
            "mean_hospital_failure_probability": values["mean_hospital_failure_probability"],
        })
    frame = pd.DataFrame(rows).sort_values("threshold_ms").reset_index(drop=True)
    frame["destroyed_km"] = frame["cut_by_wind_km"] + frame["no_supply_km"]
    return frame


def crossing(thresholds: np.ndarray, values: np.ndarray, level: float) -> float | None:
    for i in range(len(values) - 1):
        a, b = values[i] - level, values[i + 1] - level
        if a == 0:
            return float(thresholds[i])
        if a * b < 0:
            return float(thresholds[i] + (thresholds[i + 1] - thresholds[i]) * a / (a - b))
    return None


def plot(hybrid: pd.DataFrame, reported: float, output: Path, original: pd.DataFrame | None = None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    ax.axvline(ANALYSIS_THRESHOLD, color=GUIDE, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
    ax.axhline(reported, color=REPORTED, lw=1.6, ls=(0, (6, 3)), zorder=2,
               label=f"Reported ({reported:,.0f} km)")
    if original is not None:
        ax.plot(original["threshold_ms"], original["destroyed_km"], color=ORIGINAL, lw=1.6,
                marker="s", ms=3.6, mec="white", mew=0.6, zorder=3, label="Original model")
    t = hybrid["threshold_ms"].to_numpy(float)
    y = hybrid["destroyed_km"].to_numpy(float)
    ax.plot(t, y, color=MODEL, lw=2.0, marker="o", ms=4.2, mec="white", mew=0.7, zorder=4,
            label="Hybrid model" if original is not None else "Modeled")

    x = crossing(t, y, reported)
    if x is not None:
        ax.plot([x], [reported], marker="o", ms=6.5, mfc="white", mec=MODEL, mew=1.5, zorder=5)
        ax.annotate(f"{x:.1f} m s$^{{-1}}$", (x, reported), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color=MODEL)

    top = max(y.max(), reported, original["destroyed_km"].max() if original is not None else 0)
    bottom = min(y.min(), original["destroyed_km"].min() if original is not None else np.inf)
    lo = max(0, np.floor(bottom * 0.85 / 200) * 200)
    hi = np.ceil(top * 1.08 / 200) * 200
    ax.set_ylim(lo, hi)
    ax.text(ANALYSIS_THRESHOLD + 0.15, lo + 0.03 * (hi - lo), "analysis\nthreshold",
            fontsize=8, color="#6b747c", va="bottom", ha="left", linespacing=1.1)
    ax.set_xlim(17.5, 30.5)
    ax.set_xticks(range(18, 31, 2))
    ax.set_xlabel("Wind threshold $\\nu_1$ (m s$^{-1}$)", fontsize=11.5)
    ax.set_ylabel("Length of damaged\ntransmission lines (km)", fontsize=11.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False, handlelength=2.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-dir", type=Path, required=True)
    parser.add_argument("--reported-km", type=float, default=1345.0)
    parser.add_argument("--original-csv", type=Path,
                        help="threshold_ms,destroyed_km; also writes a comparison figure")
    parser.add_argument("--output", type=Path, required=True,
                        help="output path without suffix; .png and .pdf are written")
    args = parser.parse_args()

    hybrid = load_hybrid(args.length_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(args.output.with_name(args.output.name + "_data.csv"), index=False)
    plot(hybrid, args.reported_km, args.output)
    if args.original_csv:
        original = pd.read_csv(args.original_csv)
        plot(hybrid, args.reported_km,
             args.output.with_name(args.output.name + "_vs_original"), original)
    print(hybrid.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
