#!/usr/bin/env python3
"""Supplementary figure: electricity supply, generator reporting and modeled
disruption by facility type.

All plotted values are recomputed from the facility-level records
(``facility_values.csv`` from ``facility_type_electricity_services.py``) and
checked against ``table_numbers.json`` before plotting.

A  Main electricity source (SARA Q410; Q408 = no -> no electricity reported;
   no Q408 response -> missing), 100% of surveyed facilities per type.
B  Generator named as main or alternative source (Q410/Q411), and generator
   reported functioning on the survey day (Q413), % of surveyed facilities.
C  Mean annualized electricity disruption index for all evaluated facilities
   (grid hybrid model or solar model) and for evaluated grid-supplied
   facilities.  Unavailable estimates are excluded.  The groups overlap.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from helvetica_style import use_helvetica

TYPES = ["Hospitals", "Health centers", "Health posts"]
SOURCES = [  # (category in facility_values.csv, label, colour, hatch)
    ("Public grid", "Public grid", "#0072B2", None),
    ("Solar", "Solar", "#E69F00", None),
    ("Generator, gas or other", "Generator, gas or other", "#009E73", None),
    ("No electricity reported", "No electricity reported", "#BBBBBB", None),
    ("Not recorded", "Missing response", "white", "////"),
]
GENERATOR, FUNCTIONING = "#CC79A7", "#7A2E5E"
GENERATOR_TEXT = "#A8508A"  # darker shade of GENERATOR for legible labels
ALL_EVALUATED, GRID = "#4D4D4D", "#0072B2"
MM = 1 / 25.4


def compute(values: pd.DataFrame) -> pd.DataFrame:
    for column in ("generator_reported", "generator_functioning"):
        values[column] = values[column].astype(str).str.lower().eq("true")
    rows = []
    for facility_type in TYPES:
        d = values[values["facility_type"] == facility_type]
        n = len(d)
        row = {"facility_type": facility_type, "facilities": n}
        for category, *_ in SOURCES:
            row[f"source_n:{category}"] = int((d["main_source"] == category).sum())
            row[f"source_pct:{category}"] = 100 * row[f"source_n:{category}"] / n
        row["generator_reported_n"] = int(d["generator_reported"].sum())
        row["generator_functioning_n"] = int(d["generator_functioning"].sum())
        row["generator_reported_pct"] = 100 * row["generator_reported_n"] / n
        row["generator_functioning_pct"] = 100 * row["generator_functioning_n"] / n
        evaluated = d[d["index"].notna()]
        grid = evaluated[evaluated["model"] == "grid"]
        row.update({"all_evaluated_n": len(evaluated), "all_evaluated_mean": evaluated["index"].mean(),
                    "grid_evaluated_n": len(grid), "grid_evaluated_mean": grid["index"].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def validate(data: pd.DataFrame, table: dict) -> list[str]:
    checks = []
    for _, r in data.iterrows():
        t = table["facility_type"][r["facility_type"]]
        total = sum(r[f"source_n:{c}"] for c, *_ in SOURCES)
        assert total == r["facilities"] == t["facilities"], r["facility_type"]
        for category, *_ in SOURCES:
            assert r[f"source_n:{category}"] == t["main_source"].get(category, 0), (r["facility_type"], category)
        assert r["generator_reported_n"] == t["generator_reported"]
        assert r["generator_functioning_n"] == t["generator_functioning"]
        assert r["all_evaluated_n"] == t["index_all"]["n"] and r["grid_evaluated_n"] == t["index_grid"]["n"]
        assert abs(r["all_evaluated_mean"] - t["index_all"]["mean"]) < 1e-12
        assert abs(r["grid_evaluated_mean"] - t["index_grid"]["mean"]) < 1e-12
        checks.append(f"{r['facility_type']}: sources sum to {total}; generator {r['generator_reported_n']}/"
                      f"{r['generator_functioning_n']}; all n={r['all_evaluated_n']} mean={r['all_evaluated_mean']:.4f}; "
                      f"grid n={r['grid_evaluated_n']} mean={r['grid_evaluated_mean']:.4f}")
    return checks


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)


def panel_label(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=8, fontweight="semibold", pad=6)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


def draw(data: pd.DataFrame, output: Path) -> None:
    use_helvetica(7)
    plt.rcParams.update({"axes.edgecolor": "#444444", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(180 * MM, 132 * MM))
    top = fig.add_gridspec(1, 1, left=0.115, right=0.97, top=0.93, bottom=0.64)
    bottom = fig.add_gridspec(1, 2, left=0.115, right=0.85, top=0.42, bottom=0.19, wspace=0.95)
    y = np.arange(len(TYPES))[::-1]  # hospitals on top
    type_labels = [f"{t}\n(n = {n:,})" for t, n in zip(TYPES, data["facilities"])]

    # A: main electricity source
    ax = fig.add_subplot(top[0, 0])
    left = np.zeros(len(TYPES))
    for category, label, colour, hatch in SOURCES:
        pct = data[f"source_pct:{category}"].to_numpy()
        ax.barh(y, pct, left=left, height=0.62, color=colour, hatch=hatch, edgecolor="#333333" if hatch else "white",
                linewidth=0.5 if hatch else 0.6, label=label)
        for yi, l, p in zip(y, left, pct):
            if p >= 7:
                ax.text(l + p / 2, yi, f"{p:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if colour in ("#0072B2", "#009E73") else "#1a1a1a")
        left += pct
    ax.set(xlim=(0, 100), yticks=y, yticklabels=type_labels)
    ax.set_xlabel("Share of surveyed facilities (%)", fontsize=7)
    style_axis(ax)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=len(SOURCES), frameon=False, fontsize=6.5,
              handlelength=1.4, handleheight=1.0, columnspacing=1.4, borderaxespad=0)
    panel_label(ax, "A", "Main electricity source")

    # B: generator reporting
    ax = fig.add_subplot(bottom[0, 0])
    for yi, (_, r) in zip(y, data.iterrows()):
        ax.plot([r["generator_functioning_pct"], r["generator_reported_pct"]], [yi, yi], color="#BBBBBB", lw=1.2,
                zorder=1)
        # Larger circle behind a smaller diamond keeps both visible when the shares are close.
        ax.scatter(r["generator_reported_pct"], yi, s=58, marker="o", color=GENERATOR, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.scatter(r["generator_functioning_pct"], yi, s=16, marker="D", color=FUNCTIONING, edgecolor="white",
                   linewidth=0.4, zorder=4)
        n = r["facilities"]
        ax.text(1.03, yi + 0.13, f"{r['generator_reported_n']}/{n:,} ({r['generator_reported_pct']:.0f}%)",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.3, color=GENERATOR_TEXT)
        ax.text(1.03, yi - 0.17, f"{r['generator_functioning_n']}/{n:,} ({r['generator_functioning_pct']:.0f}%)",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.3, color=FUNCTIONING)
    ax.set(xlim=(0, 100), ylim=(-0.6, len(TYPES) - 0.4), yticks=y, yticklabels=TYPES)
    ax.set_xlabel("Share of surveyed facilities (%)", fontsize=7)
    style_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=GENERATOR, markersize=6.5,
               label="Generator reported (main or alternative source)"),
        Line2D([], [], marker="D", ls="none", color=FUNCTIONING, markersize=3.6,
               label="Generator reported functioning (survey day)"),
    ], loc="upper left", bbox_to_anchor=(-0.02, -0.30), frameon=False, fontsize=6.3, handletextpad=0.4,
        borderaxespad=0)
    panel_label(ax, "B", "Generator reporting")

    # C: mean annualized electricity disruption index
    ax = fig.add_subplot(bottom[0, 1])
    for yi, (_, r) in zip(y, data.iterrows()):
        a, g = r["all_evaluated_mean"], r["grid_evaluated_mean"]
        ax.plot([min(a, g), max(a, g)], [yi, yi], color="#BBBBBB", lw=1.2, zorder=1)
        ax.scatter(a, yi, s=58, marker="o", facecolor="none", edgecolor=ALL_EVALUATED, linewidth=1.1, zorder=3)
        ax.scatter(g, yi, s=18, marker="s", color=GRID, zorder=4)
        ax.text(1.03, yi + 0.13, f"{a:.4f} (n = {r['all_evaluated_n']:,})", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.3, color=ALL_EVALUATED)
        ax.text(1.03, yi - 0.17, f"{g:.4f} (n = {r['grid_evaluated_n']:,})", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.3, color=GRID)
    ax.set(xlim=(0, 0.02), ylim=(-0.6, len(TYPES) - 0.4), yticks=y, yticklabels=TYPES)
    ax.set_xticks(np.arange(0, 0.0201, 0.005))
    ax.set_xlabel("Mean annualized electricity disruption index (yr$^{-1}$)", fontsize=7)
    style_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=ALL_EVALUATED,
               markeredgewidth=1.1, markersize=6.5, label="All evaluated facilities (grid or solar model)"),
        Line2D([], [], marker="s", ls="none", color=GRID, markersize=4, label="Evaluated grid-supplied facilities"),
    ], loc="upper left", bbox_to_anchor=(-0.02, -0.30), frameon=False, fontsize=6.3, handletextpad=0.4,
        borderaxespad=0)
    panel_label(ax, "C", "Modeled electricity disruption")

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(output.with_suffix(suffix), **kwargs)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facility-values", type=Path, required=True)
    parser.add_argument("--table-numbers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".pdf", ".png", ".csv"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    values = pd.read_csv(args.facility_values)
    if not values["facility_i"].is_unique:
        raise ValueError("facility IDs are not unique")
    data = compute(values)
    for line in validate(data, json.loads(args.table_numbers.read_text())):
        print("OK", line)
    draw(data, args.output)
    data.to_csv(args.output.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
