#!/usr/bin/env python3
"""Supplementary maps: modeled electricity disruption at facilities with selected
electricity-dependent clinical equipment.

Panels: (A) oxygen concentrators, (B) neonatal incubators, (C) vaccine
refrigerators powered by grid or generator electricity (SARA Q1109 = 1).  Each
panel shows the facilities that record the item as available and functional
and have an evaluated public-grid estimate (hybrid model; unavailable
estimates excluded).  Facility attributes come from ``facility_values.csv``
(``facility_type_electricity_services.py``); coordinates from the model's
``SARA_processed`` layer, joined on the SARA facility number (``facility_i``).

Fill: annualized electricity disruption index (BuGn, shared scale from 0, as in
the manuscript maps).  Shape: facility type.  Thick magenta outline: generator
named as main or alternative source (``generator_reported``, Q410/Q411); thin
grey outline otherwise.  Non-generator facilities are drawn first (low to high
index), then generator-reporting facilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.lines import Line2D

from helvetica_style import use_helvetica

PANELS = [  # (indicator column, title, expected evaluated n, expected n without generator)
    ("oxygen", "Oxygen concentrators", 71, 36),
    ("incubator", "Neonatal incubators", 54, 26),
    ("fridge_grid", "Vaccine refrigerators (grid/generator)", 588, 541),
]
SHAPES = {"Hospitals": "o", "Health centers": "^", "Health posts": "s"}
SHAPE_LABELS = {"Hospitals": "Hospital", "Health centers": "Health center", "Health posts": "Health post"}
SIZE = {"Hospitals": 15, "Health centers": 13, "Health posts": 11}  # roughly equal visual area
GENERATOR_EDGE, PLAIN_EDGE = "#D4007A", "#6E6E6E"
LAND, NEIGHBOUR, PROVINCE_LINE = "#E3E3E3", "#F2F2F2", "#FFFFFF"
VMAX = 0.05
EXTENT = (30.0, 41.2, -27.1, -10.2)
MM = 1 / 25.4


def load(args: argparse.Namespace) -> pd.DataFrame:
    values = pd.read_csv(args.facility_values)
    for column in ("generator_reported", "oxygen", "incubator", "fridge_grid"):
        values[column] = values[column].astype(str).str.lower().eq("true")
    sara = gpd.read_file(args.sara)
    if not values["facility_i"].is_unique or not sara["facility_i"].is_unique:
        raise ValueError("facility IDs are not unique")
    sara = sara[["facility_i", "geometry"]].copy()
    sara["lon"] = sara.geometry.x
    sara["lat"] = sara.geometry.y
    merged = values.merge(sara[["facility_i", "lon", "lat"]], on="facility_i", how="left", validate="1:1")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facility-values", type=Path, required=True)
    parser.add_argument("--sara", type=Path, required=True, help="SARA_processed (model facility layer)")
    parser.add_argument("--countries", type=Path, required=True, help="World_Countries layer")
    parser.add_argument("--provinces", type=Path, required=True, help="GAUL admin-1 layer (afr_g2014_2013_1)")
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".pdf", ".png", ".csv", ".json"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    f = load(args)
    countries = gpd.read_file(args.countries).to_crs(4326)
    mozambique = countries[countries["COUNTRY"] == "Mozambique"]
    provinces = gpd.read_file(args.provinces).to_crs(4326)
    provinces = provinces[provinces["ADM0_NAME"] == "Mozambique"]
    lake = provinces[provinces["ADM1_NAME"].str.lower().str.startswith("lago")]
    provinces = provinces.drop(lake.index)
    outline = mozambique.geometry.union_all()

    report, plotted = {"vmax": VMAX, "panels": {}}, []
    cohorts = []
    for column, title, expected_n, expected_no_gen in PANELS:
        d = f[f[column] & (f["model"] == "grid") & f["grid_index"].notna()].copy()
        no_gen = int((~d["generator_reported"]).sum())
        if (len(d), no_gen) != (expected_n, expected_no_gen):
            raise SystemExit(f"{column}: {len(d)} evaluated / {no_gen} without generator, "
                             f"expected {expected_n} / {expected_no_gen}")
        if not np.allclose(d["grid_index"], d["index"]):
            raise ValueError(f"{column}: grid index differs from facility index")
        missing = int(d[["lon", "lat"]].isna().any(axis=1).sum())
        inside = gpd.GeoSeries(gpd.points_from_xy(d["lon"], d["lat"]), crs=4326).buffer(0.05).intersects(outline)
        report["panels"][column] = {
            "evaluated": len(d), "without_reported_generator": no_gen,
            "with_reported_generator": int(d["generator_reported"].sum()),
            "by_type": d["facility_type"].value_counts().to_dict(),
            "missing_coordinates": missing, "outside_mozambique_0.05deg": int((~inside.to_numpy()).sum()),
            "index_min": float(d["grid_index"].min()), "index_max": float(d["grid_index"].max()),
            "clipped_above_vmax": int((d["grid_index"] > VMAX).sum())}
        if missing:
            raise SystemExit(f"{column}: {missing} facilities without coordinates")
        cohorts.append((column, title, d))
        plotted.append(d.assign(panel=column)[["panel", "facility_i", "name", "facility_type", "lon", "lat",
                                               "generator_reported", "grid_index"]])

    use_helvetica(7)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 112 * MM))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.2, wspace=0.03)
    norm = colors.Normalize(vmin=0.0, vmax=VMAX, clip=False)
    cmap = matplotlib.colormaps["BuGn"]
    for ax, letter, (column, title, d) in zip(axes, "ABC", cohorts):
        countries.plot(ax=ax, color=NEIGHBOUR, edgecolor="white", linewidth=0.4, zorder=0)
        provinces.plot(ax=ax, color=LAND, edgecolor=PROVINCE_LINE, linewidth=0.5, zorder=1)
        lake.plot(ax=ax, color="white", edgecolor="#BDBDBD", linewidth=0.3, zorder=1)
        mozambique.boundary.plot(ax=ax, color="#222222", linewidth=0.7, zorder=2)
        for generator in (False, True):
            layer = d[d["generator_reported"] == generator].sort_values("grid_index")
            for facility_type, marker in SHAPES.items():
                p = layer[layer["facility_type"] == facility_type]
                if p.empty:
                    continue
                ax.scatter(p["lon"], p["lat"], c=np.minimum(p["grid_index"], VMAX), cmap=cmap, norm=norm,
                           marker=marker, s=SIZE[facility_type] * (1.5 if generator else 1.0),
                           edgecolors=GENERATOR_EDGE if generator else PLAIN_EDGE,
                           linewidths=1.0 if generator else 0.3, zorder=4 if generator else 3)
        ax.set_xlim(EXTENT[:2])
        ax.set_ylim(EXTENT[2:])
        ax.set_aspect(1 / np.cos(np.radians(-18.5)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#9E9E9E")
            spine.set_linewidth(0.6)
        ax.set_facecolor("white")
        ax.set_title(title, loc="left", fontsize=7.5, pad=4, x=0.085)
        ax.text(0.015, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0})
        r = report["panels"][column]
        ax.text(0.97, 0.04, f"{r['evaluated']:,} evaluated\n{r['without_reported_generator']:,} without\nreported generator",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, linespacing=1.15)

    cax = fig.add_axes([0.07, 0.115, 0.36, 0.022])
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max")
    bar.set_ticks(np.arange(0, VMAX + 1e-9, 0.01))
    bar.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    clipped = sum(p["clipped_above_vmax"] for p in report["panels"].values())
    bar.set_label("Annualized electricity disruption index (yr$^{-1}$)", fontsize=6.8)
    cax.text(1.06, 0.5, f"> {VMAX:g}", transform=cax.transAxes, ha="left", va="center", fontsize=6.3)
    handles = [Line2D([], [], marker=SHAPES[t], ls="none", markerfacecolor=cmap(0.55), markeredgecolor=PLAIN_EDGE,
                      markeredgewidth=0.4, markersize=5, label=SHAPE_LABELS[t]) for t in SHAPES]
    handles += [
        Line2D([], [], marker="o", ls="none", markerfacecolor=cmap(0.55), markeredgecolor=GENERATOR_EDGE,
               markeredgewidth=1.1, markersize=6, label="Generator reported (main or alternative source)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=cmap(0.55), markeredgecolor=PLAIN_EDGE,
               markeredgewidth=0.4, markersize=5, label="No generator reported"),
    ]
    fig.legend(handles=handles[:3], loc="lower left", bbox_to_anchor=(0.5, 0.075), ncol=3, frameon=False,
               fontsize=6.6, handletextpad=0.3, columnspacing=1.0)
    fig.legend(handles=handles[3:], loc="lower left", bbox_to_anchor=(0.5, 0.025), ncol=1, frameon=False,
               fontsize=6.6, handletextpad=0.3, labelspacing=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=600)
    plt.close(fig)
    pd.concat(plotted).to_csv(args.output.with_suffix(".csv"), index=False)
    report["clipped_above_vmax_total_rows"] = clipped
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
