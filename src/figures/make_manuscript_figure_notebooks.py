#!/usr/bin/env python3
"""Build draft, one-figure-per-notebook manuscript plotting notebooks.

The source notebooks in ``moz_notebook`` are exploratory and contain many
unrelated cells.  This script keeps a small, deterministic set of draft
notebooks that can be refined as the manuscript figures change.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "moz_notebook"


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(PROJECT_ROOT))


COMMON_SETUP = r'''
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("INFRACASCADE_PROJECT_DIR", Path.cwd())).resolve()
if PROJECT_ROOT.name == "moz_notebook":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = Path(os.environ.get("MOZ_FIGURE_DATA_ROOT", PROJECT_ROOT / "figure_data"))
FIGURE_DIR = Path(os.environ.get(
    "MOZ_MANUSCRIPT_FIGURE_DIR",
    PROJECT_ROOT / "outputs" / "manuscript_figures",
))
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Project:", PROJECT_ROOT)
print("Data:", DATA_ROOT)
print("Figures:", FIGURE_DIR)
'''


def figure_2() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 2 — public-grid and self-reliant electricity risk

Draft extracted from `moz_paper_figure.ipynb`, primarily cell index 14.
It intentionally uses a real 2 × 2 layout rather than the legacy 2 × 4
canvas that was later cropped for the manuscript.

The loader below uses the current 2,703-track public-grid summary and the
independent-solar track outputs directly. It therefore does not require the
old hand-prepared GeoPackages. Paths can be overridden with the same
`INFRACASCADE_*` variables used by the Slurm workflow.

The annualization factor, labels, and final typography remain
manuscript-editing points.
''',
            "fig2-title",
        ),
        code(COMMON_SETUP, "fig2-setup"),
        code(
            r'''
from capacity_ablation_figure import (
    attach_hospital_geometry,
    load_boundary,
    load_hospital_data,
    load_power_summary,
    load_solar_summary,
    validate_standard_cohorts,
)
from synthetic_tracks import expected_track_columns

SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "/scratch/users/zywei"))
WIND_ROOT = Path(os.environ.get(
    "INFRACASCADE_WIND_ROOT", SCRATCH_ROOT / "wind_result"
))
HOSPITAL_PATH = Path(os.environ.get(
    "INFRACASCADE_HOSPITAL_DATA", PROJECT_ROOT / "SARA_processed"
))
PUBLIC_SUMMARY_PATH = Path(os.environ.get(
    "INFRACASCADE_PUBLIC_SUMMARY", WIND_ROOT / "df_final_public_25.csv"
))
SOLAR_ROOT = Path(os.environ.get(
    "INFRACASCADE_SOLAR_ROOT", WIND_ROOT / "synthetic_solar"
))
ALL_GRID_SUMMARY_PATH = Path(os.environ.get(
    "INFRACASCADE_ALL_GRID_SUMMARY", WIND_ROOT / "df_final_all_grid_25.csv"
))
WORLD_PATH = Path(os.environ.get(
    "INFRACASCADE_BOUNDARY_DATA", SCRATCH_ROOT / "World_Countries"
))

hospital = load_hospital_data(HOSPITAL_PATH)
public_summary = load_power_summary(
    PUBLIC_SUMMARY_PATH, require_complete=True, allow_unavailable=True
)
solar_summary = load_solar_summary(SOLAR_ROOT, hospital, zero_empty_events=True)
# This check also confirms that the public and solar cohorts form the same
# 1,380-hospital universe used by the all-grid ablation.
all_grid_summary = load_power_summary(ALL_GRID_SUMMARY_PATH, require_complete=True)
validate_standard_cohorts(public_summary, all_grid_summary, solar_summary)

gdf_elec = attach_hospital_geometry(public_summary, hospital)
solar = attach_hospital_geometry(solar_summary, hospital)
mozambique = load_boundary(WORLD_PATH)

def facility_key(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text

urban = pd.to_numeric(hospital["urban"], errors="coerce")
urban_ids = set(hospital.loc[urban.eq(1), "facility_i"].map(facility_key))
rural_ids = set(hospital.loc[urban.eq(2), "facility_i"].map(facility_key))
generator_ids = set(
    hospital.loc[
        hospital["has_genera"].astype(str).str.strip().str.lower().eq("yes"),
        "facility_i",
    ].map(facility_key)
)
gdf_elec["facility_key"] = gdf_elec["facility_i"].map(facility_key)
solar["facility_key"] = solar["facility_i"].map(facility_key)

gdf_elec = gdf_elec.copy()
solar = solar.copy()
track_columns = list(expected_track_columns())
ANNUALIZATION_FACTOR = len(track_columns) / 10_000
gdf_elec["mean"] = gdf_elec[track_columns].where(
    gdf_elec[track_columns] >= 0
).mean(axis=1).fillna(-1.0)
solar["mean"] = solar[track_columns].mean(axis=1)
gdf_elec["annual_probability"] = gdf_elec["mean"] * ANNUALIZATION_FACTOR
solar["annual_probability"] = solar["mean"] * ANNUALIZATION_FACTOR
''',
            "fig2-load",
        ),
        code(
            r'''
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

cmap = "BuGn"
norm = Normalize(0, 0.05)
fig, axes = plt.subplots(2, 2, figsize=(6.5, 8), sharex=True, sharey=True)

for axis in axes.flat:
    mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.8)
    axis.set(xlim=(30, 41), ylim=(-27, -10))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")

panels = [
    (axes[0, 0], gdf_elec, urban_ids, "(a)", "Urban"),
    (axes[0, 1], gdf_elec, rural_ids, "(b)", "Rural"),
    (axes[1, 0], solar, urban_ids, "(c)", ""),
    (axes[1, 1], solar, rural_ids, "(d)", ""),
]
for axis, layer, facility_ids, panel, title in panels:
    selected = layer[
        layer["facility_key"].isin(facility_ids) & layer["mean"].ge(0)
    ]
    selected.plot(
        ax=axis,
        column="annual_probability",
        cmap=cmap,
        norm=norm,
        markersize=15,
        linewidth=0,
    )
    with_generator = selected[selected["facility_key"].isin(generator_ids)]
    if not with_generator.empty:
        with_generator.plot(
            ax=axis,
            facecolor="none",
            edgecolor="deeppink",
            markersize=18,
            linewidth=0.5,
        )
    axis.text(0.02, 0.98, panel, transform=axis.transAxes, va="top")
    axis.set_title(title)

fig.text(0.015, 0.70, "Depend on public infrastructure", rotation=90, va="center")
fig.text(0.015, 0.29, "Self-reliant", rotation=90, va="center")

colorbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.035, pad=0.025
)
colorbar.set_label("Annualized electricity disruption index (yr$^{-1}$)")
colorbar.set_ticks(np.arange(0, 0.051, 0.01), labels=[f"{x:.2f}" for x in np.arange(0, 0.051, 0.01)])

fig.legend(
    handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                    markeredgecolor="deeppink", label="Has generator")],
    loc="lower center",
    frameon=False,
)
fig.subplots_adjust(left=0.10, right=0.88, bottom=0.08, top=0.94, wspace=0.03, hspace=0.03)

for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_02.{suffix}", dpi=300, bbox_inches="tight")
plt.show()
''',
            "fig2-render",
        ),
    ]


def figure_3() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 3 — cascading healthcare-service failures

Draft extracted from `moz_paper_figure.ipynb`, cell index 10. The four compact
GeoPackages are regenerated from the current non-ablation public-grid,
downstream water/communication, and direct-wind results by
`prepare_failure_figure_inputs.py`.

The `-1` sentinel is displayed as unavailable/uncertain. Confirm the final
annualization factor and the precise eligible-facility denominators before
publication.
''',
            "fig3-title",
        ),
        code(COMMON_SETUP, "fig3-setup"),
        code(
            r'''
from capacity_ablation_figure import load_boundary

WORLD_PATH = Path(os.environ.get(
    "INFRACASCADE_BOUNDARY_DATA", DATA_ROOT / "World_Countries"
))
ELECTRICITY_PATH = DATA_ROOT / "df_final_public_25.gpkg"
WATER_PATH = DATA_ROOT / "water_hospital_mean.gpkg"
COMMUNICATION_PATH = DATA_ROOT / "communication_hospital_mean.gpkg"
DIRECT_PATH = DATA_ROOT / "direct_hospital_mean.gpkg"

mozambique = load_boundary(WORLD_PATH)
layers = [
    (gpd.read_file(ELECTRICITY_PATH), "mean", "(a) Electricity"),
    (gpd.read_file(WATER_PATH), "water_mean", "(b) Water"),
    (gpd.read_file(COMMUNICATION_PATH), "comm_mean", "(c) Communication"),
    (gpd.read_file(DIRECT_PATH), "direct", "(d) Direct"),
]

ANNUALIZATION_FACTOR = 2703 / 10_000
''',
            "fig3-load",
        ),
        code(
            r'''
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

cmap = "BuGn"
norm = Normalize(0, 0.05)
fig, axes = plt.subplots(1, 4, figsize=(12, 4.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.02, right=0.88, bottom=0.12, top=0.95, wspace=0.02)

for axis, (layer, value_column, title) in zip(axes, layers):
    mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.8)
    layer = layer.copy()
    unavailable = layer[layer[value_column].eq(-1)]
    valid = layer[layer[value_column].ge(0)].copy()
    valid["annual_probability"] = valid[value_column] * ANNUALIZATION_FACTOR

    if not unavailable.empty:
        unavailable.plot(
            ax=axis, color="silver", edgecolor="black", markersize=14, linewidth=0.25
        )
    valid.plot(
        ax=axis,
        column="annual_probability",
        cmap=cmap,
        norm=norm,
        markersize=15,
        linewidth=0,
    )
    axis.set_title(title, loc="left", fontsize=10)
    axis.set(xlim=(30, 41), ylim=(-27, -10))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")

cbar_axis = fig.add_axes([0.90, 0.18, 0.015, 0.68])
colorbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_axis
)
colorbar.set_label("Annualized service disruption index (yr$^{-1}$)")
colorbar.set_ticks(np.arange(0, 0.051, 0.01), labels=[f"{x:.2f}" for x in np.arange(0, 0.051, 0.01)])
fig.legend(
    handles=[Line2D([], [], marker="o", linestyle="none", color="silver",
                    markeredgecolor="black", label="Data missing")],
    loc="lower center",
    frameon=False,
)
for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_03.{suffix}", dpi=300, bbox_inches="tight")
plt.show()
''',
            "fig3-render",
        ),
    ]


def figure_4() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 4 — direct versus indirect failures by cyclone

Draft extracted from `Another copy of direct_communication.ipynb`, cell index
44. It uses one CSV per infrastructure system. Each CSV needs `wind`,
`indirect`, and `cyclone_intensity` columns.
''',
            "fig4-title",
        ),
        code(COMMON_SETUP + "\nimport seaborn as sns", "fig4-setup"),
        code(
            r'''
SUMMARY_PATHS = {
    "(a) Electricity": DATA_ROOT / "cyclone_summary_elec.csv",
    "(b) Water": DATA_ROOT / "cyclone_summary_water.csv",
    "(c) Communication": DATA_ROOT / "cyclone_summary_comm.csv",
}
summaries = {name: pd.read_csv(path) for name, path in SUMMARY_PATHS.items()}

required = {"wind", "indirect", "cyclone_intensity"}
for name, frame in summaries.items():
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
''',
            "fig4-load",
        ),
        code(
            r'''
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharex=True, sharey=True)

x_limit = 40
y_limit = 350

for axis, (title, frame) in zip(axes, summaries.items()):
    sns.scatterplot(
        data=frame,
        x="wind",
        y="indirect",
        hue="cyclone_intensity",
        palette="viridis",
        s=55,
        edgecolor="black",
        linewidth=0.5,
        legend=axis is axes[2],
        ax=axis,
    )
    axis.plot([0, x_limit], [0, x_limit], "--", color="black", linewidth=1)
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_xlabel("Total direct (wind) failures")
    axis.set_xlim(0, x_limit)
    axis.set_ylim(0, y_limit)
    axis.grid(False)

axes[0].set_ylabel("Total indirect failures")
axes[1].set_ylabel("")
axes[2].set_ylabel("")
if axes[2].get_legend() is not None:
    axes[2].legend(title="Cyclone intensity", loc="center right")

fig.tight_layout()
for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_04.{suffix}", dpi=300, bbox_inches="tight")
plt.show()
''',
            "fig4-render",
        ),
    ]


def figure_5() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 5 — hybrid power-grid disruption

Draft consolidated from the newly added `cascade.ipynb` and updated for the
final hybrid model:

- cell 32 draws the Category 1/2 length bars and impact lines;
- cell 40 draws the coastal network and hospital-density map.

The province summary is generated by
`prepare_hybrid_cascade_admin_summary.py`. In the legacy-compatible main-figure
convention, initial failure is directly removed transmission-line length and
cascade failure is surviving transmission length in completely unpowered
components. This matches the previous `cascade.ipynb` post-processing while
using the new hybrid service model. Partially served component length is not
included in these two bars. Simplified-edge lengths are sums over their full
unsimplified paths.
''',
            "fig5-title",
        ),
        code(COMMON_SETUP + "\nimport pickle\nfrom shapely.geometry import shape", "fig5-setup"),
        code(
            r'''
ADMIN_SUMMARY_PATH = Path(os.environ.get(
    "INFRACASCADE_CASCADE_ADMIN_SUMMARY",
    DATA_ROOT / "cascade_admin_summary.gpkg",
))
POWER_GRAPH_PATH = Path(os.environ.get(
    "INFRACASCADE_POWER_GRAPH",
    PROJECT_ROOT / "network_save" / "simplified_hospital_MG.pickle",
))

admin = gpd.read_file(ADMIN_SUMMARY_PATH)
with POWER_GRAPH_PATH.open("rb") as handle:
    power_graph = pickle.load(handle)

NAME_COLUMN = "ADM1_NAME"
COASTAL_ORDER = [
    "Maputo", "Gaza", "Inhambane", "Sofala", "Zambezia", "Nampula", "Cabo Delgado"
]
required = {
    NAME_COLUMN,
    "num_hospital",
    "1_mean_initial_failure_length_km", "1_mean_cascade_failure_length_km",
    "1_cascade_initial_ratio", "1_mean_hospital_failure_probability",
    "2_mean_initial_failure_length_km", "2_mean_cascade_failure_length_km",
    "2_cascade_initial_ratio", "2_mean_hospital_failure_probability",
}
missing = required.difference(admin.columns)
if missing:
    raise ValueError(
        "Run prepare_cascade_admin_summary.py before executing this notebook. "
        f"Missing columns: {sorted(missing)}"
    )

coastal = admin[admin[NAME_COLUMN].isin(COASTAL_ORDER)].copy()
coastal[NAME_COLUMN] = pd.Categorical(
    coastal[NAME_COLUMN], categories=COASTAL_ORDER, ordered=True
)
coastal = coastal.sort_values(NAME_COLUMN)
''',
            "fig5-load",
        ),
        code(
            r'''
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter

fig = plt.figure(figsize=(8.333, 9.583), facecolor="white")
text_color = "#203040"
accent_color = "#14739B"

# Manuscript title and panel A reproduce the original composite layout.
fig.text(
    0.035, 0.963,
    "Category 1 and 2 cyclones can cause substantial electricity disruption",
    color=text_color, fontsize=11.5, va="top",
)
fig.add_artist(Line2D(
    [0.035, 0.965], [0.927, 0.927], transform=fig.transFigure,
    color="#B8C8D3", linewidth=0.6,
))
fig.text(0.035, 0.900, "A   Modeled disruption pathway", color=text_color, fontsize=8.5)

box_labels = (
    "Cyclone wind", "Grid damage", "Network disruption", "Facility electricity loss"
)
box_left = (0.035, 0.275, 0.515, 0.755)
box_width = 0.205
box_y = 0.795
box_height = 0.060
for index, (left, label) in enumerate(zip(box_left, box_labels)):
    fig.add_artist(Rectangle(
        (left, box_y), box_width, box_height,
        transform=fig.transFigure, facecolor="#EEF4F7",
        edgecolor=accent_color, linewidth=0.8,
    ))
    fig.text(
        left + box_width / 2, box_y + box_height / 2, label,
        color=text_color, fontsize=7.2, ha="center", va="center",
    )
    if index < len(box_left) - 1:
        fig.add_artist(FancyArrowPatch(
            (left + box_width, box_y + box_height / 2),
            (box_left[index + 1], box_y + box_height / 2),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=7, color=accent_color, linewidth=0.7,
        ))

fig.text(
    0.035, 0.758,
    "Direct physical damage and subsequent loss of network supply are distinct outcomes.",
    color="#5A6B79", fontsize=7.2,
)
fig.text(
    0.035, 0.713, "B   Regional Category 1 / Category 2 results",
    color=text_color, fontsize=8.5,
)

map_axis = fig.add_axes([0.035, 0.095, 0.440, 0.555])
bar_axes = [
    fig.add_axes([0.530, 0.135, 0.190, 0.455]),
    fig.add_axes([0.755, 0.135, 0.190, 0.455]),
]

# Panel A: coastal provinces, hospital density, and power-grid edges.
coastal.plot(
    ax=map_axis,
    column="num_hospital",
    cmap="Greys",
    norm=Normalize(50, 200),
    edgecolor="black",
    linewidth=0.65,
)
for _, row in coastal.iterrows():
    point = row.geometry.representative_point()
    map_axis.text(
        point.x, point.y, str(row[NAME_COLUMN]),
        fontsize=5.7, ha="center", va="center", color="black",
    )

for edge in power_graph.edges:
    source, target = edge[:2]
    source_data = power_graph.nodes[source]
    target_data = power_graph.nodes[target]
    if source_data.get("power") == "hospital" or target_data.get("power") == "hospital":
        continue
    source_point = shape(source_data["geometry"])
    target_point = shape(target_data["geometry"])
    source_point = source_point if source_point.geom_type == "Point" else source_point.centroid
    target_point = target_point if target_point.geom_type == "Point" else target_point.centroid
    map_axis.plot(
        [source_point.x, target_point.x], [source_point.y, target_point.y],
        color="gold", linewidth=0.9, zorder=3,
    )

map_axis.set_xlim(30, 41)
map_axis.set_ylim(-27, -10)
map_axis.set_xticks([])
map_axis.set_yticks([])
map_axis.set_aspect("equal")
for spine in map_axis.spines.values():
    spine.set_visible(False)

map_axis.legend(
    handles=[Line2D([], [], color="gold", linewidth=1.2, label="power line")],
    loc="upper left", bbox_to_anchor=(0.10, 0.96), frameon=False,
    fontsize=5.5, handlelength=1.6,
)
colorbar_axis = map_axis.inset_axes([0.705, 0.055, 0.045, 0.475])
colorbar = fig.colorbar(
    ScalarMappable(norm=Normalize(50, 200), cmap="Greys"), cax=colorbar_axis
)
colorbar.set_ticks(np.arange(60, 201, 20))
colorbar.ax.tick_params(labelsize=5.5, length=2)
colorbar.outline.set_linewidth(0.5)
colorbar.set_label(
    "# Hospitals depending on public electrical grid", fontsize=6.2, labelpad=3
)

# Panels B/C: use the exact legacy notebook axes and styling.
positions = np.arange(len(coastal))
for category, axis, panel in zip((1, 2), bar_axes, ("(b)", "(c)")):
    initial = coastal[
        f"{category}_mean_initial_failure_length_km"
    ].to_numpy(float)
    cascade = coastal[
        f"{category}_mean_cascade_failure_length_km"
    ].to_numpy(float)
    ratio = coastal[f"{category}_cascade_initial_ratio"].to_numpy(float) * 100
    hospital_impact = coastal[
        f"{category}_mean_hospital_failure_probability"
    ].to_numpy(float) * 100

    axis.barh(positions, initial, color="#332288", height=0.30)
    axis.barh(
        positions, cascade, left=initial, color="#88CCEE", height=0.30
    )
    axis.set_xlim(0.0, 1000.0)
    axis.set_xticks(np.arange(0.0, 1001.0, 200.0))
    axis.set_xlabel("Failed Power Line Length (km)", fontsize=5.7, labelpad=2)
    axis.set_yticks(positions)
    axis.set_yticklabels(coastal[NAME_COLUMN] if category == 1 else [])
    if category == 1:
        plt.setp(axis.get_yticklabels(), rotation=45, ha="right")
    axis.tick_params(labelsize=5.5, width=0.5, length=2)
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    axis.set_title(f"TC Category {category}", fontsize=7.2, pad=25)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)

    percentage_axis = axis.twiny()
    percentage_axis.plot(
        ratio, positions, color="#D55E00", marker="o",
        linewidth=0.9, markersize=2.8,
    )
    percentage_axis.plot(
        hospital_impact, positions, color="#999933", marker="s",
        linewidth=0.9, markersize=2.8,
    )
    percentage_axis.set_xlim(0.0, 25.0)
    percentage_axis.set_xticks(np.arange(0.0, 25.1, 5.0))
    percentage_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0f}%")
    )
    percentage_axis.set_xlabel("Percentage", fontsize=6.2, labelpad=1)
    percentage_axis.tick_params(labelsize=5.5, width=0.5, length=2)
    for spine in percentage_axis.spines.values():
        spine.set_linewidth(0.5)

legend = [
    Patch(color="#332288", label="Initial Failure"),
    Patch(color="#88CCEE", label="Cascading Failure"),
    Line2D(
        [], [], color="#D55E00", marker="o", linewidth=0.9, markersize=2.8,
        label="Failure Ratio (Cascading / Initial) (%)",
    ),
    Line2D(
        [], [], color="#999933", marker="s", linewidth=0.9, markersize=2.8,
        label="Potential Hospital Impact (%)",
    ),
]
fig.legend(
    handles=legend, loc="upper left", bbox_to_anchor=(0.525, 0.680),
    ncol=2, frameon=False, fontsize=5.5, handlelength=1.8,
    columnspacing=1.0,
)

fig.text(0.245, 0.082, "(a)", fontsize=8, ha="center")
fig.text(0.625, 0.082, "(b)", fontsize=8, ha="center")
fig.text(0.850, 0.082, "(c)", fontsize=8, ha="center")
fig.text(
    0.035, 0.035,
    "Original regional map, axes and legends are retained; quantitative series use hybrid-model results.",
    color="#5A6B79", fontsize=6.3,
)
fig.text(
    0.035, 0.017,
    "Upper percentage axis: failure ratio and potentially affected facilities; lower axis: power-line length.",
    color="#5A6B79", fontsize=6.3,
)

for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_05.{suffix}", dpi=300)
plt.show()
''',
            "fig5-render",
        ),
    ]


def figure_6() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 6 — universal public-grid dependence ablation

Draft wrapper around the maintained implementation in
`capacity_ablation_figure.py`. It follows
`notebooks/capacity_v2_ablation_figure.ipynb` and produces the figure plus its
CSV/JSON audit bundle. The defaults match Sherlock; override paths through
environment variables for local runs.
''',
            "fig6-title",
        ),
        code(
            r'''
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("INFRACASCADE_PROJECT_DIR", Path.cwd())).resolve()
if PROJECT_ROOT.name == "moz_notebook":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capacity_ablation_figure import generate_outputs

SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "/scratch/users/zywei"))
WIND_ROOT = Path(os.environ.get("INFRACASCADE_WIND_ROOT", SCRATCH_ROOT / "wind_result"))
FIGURE_DIR = Path(os.environ.get(
    "MOZ_MANUSCRIPT_FIGURE_DIR",
    PROJECT_ROOT / "outputs" / "manuscript_figures",
))
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
''',
            "fig6-setup",
        ),
        code(
            r'''
PUBLIC_SUMMARY = Path(os.environ.get(
    "INFRACASCADE_PUBLIC_SUMMARY", WIND_ROOT / "df_final_public_25.csv"
))
ALL_GRID_SUMMARY = Path(os.environ.get(
    "INFRACASCADE_ALL_GRID_SUMMARY", WIND_ROOT / "df_final_all_grid_25.csv"
))
SOLAR_ROOT = Path(os.environ.get(
    "INFRACASCADE_SOLAR_ROOT", WIND_ROOT / "synthetic_solar"
))
HOSPITAL_DATA = Path(os.environ.get(
    "INFRACASCADE_HOSPITAL_DATA", PROJECT_ROOT / "SARA_processed"
))
BOUNDARY_DATA = Path(os.environ.get(
    "INFRACASCADE_BOUNDARY_DATA", SCRATCH_ROOT / "World_Countries"
))
OUTPUT_PREFIX = Path(os.environ.get(
    "INFRACASCADE_FIGURE_PREFIX", FIGURE_DIR / "figure_06"
))

{
    "public": PUBLIC_SUMMARY,
    "all_grid": ALL_GRID_SUMMARY,
    "solar": SOLAR_ROOT,
    "hospitals": HOSPITAL_DATA,
    "boundary": BOUNDARY_DATA,
    "output": OUTPUT_PREFIX,
}
''',
            "fig6-paths",
        ),
        code(
            r'''
comparison, summary, figure, metadata = generate_outputs(
    public_summary=PUBLIC_SUMMARY,
    all_grid_summary=ALL_GRID_SUMMARY,
    solar_root=SOLAR_ROOT,
    hospital_data=HOSPITAL_DATA,
    boundary_path=BOUNDARY_DATA,
    output_prefix=OUTPUT_PREFIX,
)
display(summary)
print("Solar empty-event tracks set to zero:", metadata["solar_empty_tracks_set_to_zero"])
print("Output prefix:", OUTPUT_PREFIX)
figure
''',
            "fig6-render",
        ),
    ]


def figure_7() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 7 — historical versus synthetic cyclone catalogues

Draft based on `SynVSHistory.ipynb`, especially cells 166–170 and 180. The
legacy notebook generated the eight risk maps but did not contain the final
2 × 5 assembly or track-distribution plotting code. This draft adds those
pieces.

Expected `COMPARE_DIR` layers:

- `health_elec_historical_0` through `_3`
- `health_elec_0` through `_3`

Optional track layers should contain line geometries and, preferably, a
`category` column. If they are absent, the first column remains a labelled
placeholder so the risk panels can still be edited.
''',
            "fig7-title",
        ),
        code(COMMON_SETUP, "fig7-setup"),
        code(
            r'''
WORLD_PATH = Path(os.environ.get(
    "INFRACASCADE_BOUNDARY_DATA", DATA_ROOT / "World_Countries"
))
COMPARE_DIR = DATA_ROOT / "compare"
HISTORICAL_TRACK_PATH = DATA_ROOT / "historical_tracks.gpkg"
SYNTHETIC_TRACK_PATH = DATA_ROOT / "synthetic_tracks.gpkg"
METADATA_PATH = DATA_ROOT / "history_figure_inputs.metadata.json"

world = gpd.read_file(WORLD_PATH)
mozambique = world.loc[world["COUNTRY"].eq("Mozambique")]

historical_risk = [gpd.read_file(COMPARE_DIR / f"health_elec_historical_{i}.gpkg") for i in range(4)]
synthetic_risk = [gpd.read_file(COMPARE_DIR / f"health_elec_{i}.gpkg") for i in range(4)]
historical_tracks = gpd.read_file(HISTORICAL_TRACK_PATH) if HISTORICAL_TRACK_PATH.exists() else None
synthetic_tracks = gpd.read_file(SYNTHETIC_TRACK_PATH) if SYNTHETIC_TRACK_PATH.exists() else None

CLASS_TITLES = ["Tropical Storm", "Category 1", "Category 2", "Category 3"]
if METADATA_PATH.exists():
    import json
    metadata = json.loads(METADATA_PATH.read_text())
    EVENT_COUNTS = {
        "historical": metadata["historical_category_counts"],
        "synthetic": metadata["synthetic_category_counts"],
    }
else:
    EVENT_COUNTS = {
        "historical": [7, 8, 4, 2],
        "synthetic": [1904, 510, 115, 25],
    }
''',
            "fig7-load",
        ),
        code(
            r'''
from matplotlib import cm
from matplotlib.colors import Normalize

def base_map(axis):
    mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.8)
    axis.set(xlim=(30, 41), ylim=(-27, -10))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


def plot_tracks(axis, tracks, placeholder):
    base_map(axis)
    if tracks is None:
        axis.text(
            0.5, 0.5, placeholder, transform=axis.transAxes,
            ha="center", va="center", fontsize=8, color="dimgray"
        )
        return
    if "category" in tracks.columns:
        palette = {0: "#35d0ce", 1: "#70e0bc", 2: "#edd85c", 3: "#e69f3b"}
        for category, group in tracks.groupby("category"):
            group.plot(ax=axis, color=palette.get(category, "#777777"), linewidth=0.5)
    else:
        tracks.plot(ax=axis, color="#35d0ce", linewidth=0.5)


fig, axes = plt.subplots(2, 5, figsize=(11, 5.2), sharex=True, sharey=True)
plot_tracks(axes[0, 0], historical_tracks, "TODO: historical track layer")
plot_tracks(axes[1, 0], synthetic_tracks, "TODO: synthetic track layer")
axes[0, 0].set_title("Tracks Distribution", fontsize=9)

norm = Normalize(0, 0.05)
for row, (catalogue, risk_layers, catalogue_name) in enumerate([
    (historical_tracks, historical_risk, "historical"),
    (synthetic_tracks, synthetic_risk, "synthetic"),
]):
    for column, (title, risk) in enumerate(zip(CLASS_TITLES, risk_layers), start=1):
        axis = axes[row, column]
        base_map(axis)
        valid = risk[risk["mean"].ge(0)]
        valid.plot(
            ax=axis, column="mean", cmap="BuGn", norm=norm,
            markersize=15, linewidth=0,
        )
        if row == 0:
            axis.set_title(title, fontsize=9)
        count = EVENT_COUNTS[catalogue_name][column - 1]
        axis.text(
            0.95, 0.04, f"{count} events", transform=axis.transAxes,
            ha="right", va="bottom", fontsize=7,
        )

for index, axis in enumerate(axes.flat):
    axis.text(0.02, 0.98, f"({chr(ord('a') + index)})", transform=axis.transAxes,
              va="top", fontsize=8)

fig.text(0.015, 0.69, "Historical events", rotation=90, va="center")
fig.text(0.015, 0.29, "Synthetic events", rotation=90, va="center")
colorbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap="BuGn"), ax=axes, fraction=0.018, pad=0.015
)
colorbar.set_label("Annualized electricity disruption index (yr$^{-1}$)")
colorbar.set_ticks(np.arange(0, 0.051, 0.01), labels=[f"{x:.2f}" for x in np.arange(0, 0.051, 0.01)])
fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.04, wspace=0.03, hspace=0.03)

for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_07.{suffix}", dpi=300, bbox_inches="tight")
plt.show()
''',
            "fig7-render",
        ),
    ]


def figure_8() -> list[dict]:
    return [
        markdown(
            r'''
# Figure 8 — historical deviation from the synthetic ensemble

Draft extracted from `SynVSHistory.ipynb`, cell index 351. The legacy cell
rendered electricity, water, and communication together; the manuscript uses
only the electricity panel. This notebook therefore renders the electricity
layer directly.

The input layer must include `history`, `mean`, and `std`, where `mean` and
`std` summarize the matched synthetic intervals.
''',
            "fig8-title",
        ),
        code(COMMON_SETUP, "fig8-setup"),
        code(
            r'''
WORLD_PATH = Path(os.environ.get(
    "INFRACASCADE_BOUNDARY_DATA", DATA_ROOT / "World_Countries"
))
DEVIATION_PATH = DATA_ROOT / "over_under2" / "elec.gpkg"

world = gpd.read_file(WORLD_PATH)
mozambique = world.loc[world["COUNTRY"].eq("Mozambique")]
deviation = gpd.read_file(DEVIATION_PATH).copy()

required = {"history", "mean", "std"}
missing = required.difference(deviation.columns)
if missing:
    raise ValueError(f"Deviation layer is missing columns: {sorted(missing)}")

deviation["sigma"] = np.divide(
    deviation["history"] - deviation["mean"],
    deviation["std"],
    out=np.full(len(deviation), np.nan, dtype=float),
    where=deviation["std"].to_numpy() > 0,
)
unavailable = deviation[
    deviation["history"].lt(0) | deviation["mean"].lt(0) | deviation["std"].le(0)
]
''',
            "fig8-load",
        ),
        code(
            r'''
from matplotlib import cm
from matplotlib.colors import Normalize

fig, axis = plt.subplots(figsize=(5.2, 7.2))
mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.8)
if not unavailable.empty:
    unavailable.plot(
        ax=axis, color="silver", edgecolor="black", markersize=18, linewidth=0.25
    )
deviation.plot(
    ax=axis,
    column="sigma",
    cmap="bwr_r",
    norm=Normalize(-3, 3),
    markersize=18,
    linewidth=0,
)
axis.set(xlim=(30, 41), ylim=(-27, -10))
axis.set_xticks([])
axis.set_yticks([])
axis.set_aspect("equal")

for name, longitude, latitude in [
    ("Maputo", 32.57, -25.97),
    ("Inhambane", 35.38, -23.86),
]:
    axis.annotate(
        name, xy=(longitude, latitude), xytext=(longitude + 0.3, latitude + 0.15),
        arrowprops={"arrowstyle": "-", "linewidth": 0.7}, fontsize=8,
    )

colorbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(-3, 3), cmap="bwr_r"), ax=axis)
colorbar.set_label(
    "Deviation of historical value from synthetic mean\n($\\sigma$ units)"
)
colorbar.set_ticks([-3, 3])
fig.tight_layout()

for suffix in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure_08.{suffix}", dpi=300, bbox_inches="tight")
plt.show()
''',
            "fig8-render",
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "figure_02_electricity_sources.ipynb": figure_2(),
        "figure_03_cascading_service_failures.ipynb": figure_3(),
        "figure_04_direct_vs_indirect.ipynb": figure_4(),
        "figure_05_grid_cascade.ipynb": figure_5(),
        "figure_06_grid_dependence_ablation.ipynb": figure_6(),
        "figure_07_history_vs_synthetic.ipynb": figure_7(),
        "figure_08_historical_deviation.ipynb": figure_8(),
    }
    for filename, cells in notebooks.items():
        write_notebook(filename, cells)


if __name__ == "__main__":
    main()
