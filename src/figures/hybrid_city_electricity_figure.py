"""Rebuild Supplementary Figure S2 with the final hybrid electricity result.

The layout and cohort definitions follow cells 15--24 of
``moz_notebook/moz_paper_figure.ipynb``:

* urban hospitals are shown in 14 city windows;
* coastal and inland cities are separated;
* public-grid and independent-solar facilities occupy separate rows;
* conditional synthetic-cyclone probabilities are annualized by the synthetic
  catalogue rate (2,703 events / 10,000 years);
* the displayed color range is fixed at 0--5%; and
* hospitals recorded as having a generator receive a pink outline.

Unlike the exploratory notebook, unavailable public-grid simulations are not
drawn at zero.  They remain missing and are plotted in gray.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from shapely.geometry import Point

from capacity_ablation_figure import (
    load_hospital_data,
    load_power_summary,
    load_solar_summary,
)
from synthetic_tracks import expected_track_columns


SYNTHETIC_CATALOGUE_YEARS = 10_000.0
DEFAULT_COLOR_MAXIMUM = 0.05
DEFAULT_ZOOM_BUFFER = 0.10

CITY_COORDINATES = {
    "Maputo": (-25.966, 32.583),
    "Matola": (-25.967, 32.459),
    "Nampula": (-15.116, 39.266),
    "Beira": (-19.817, 34.850),
    "Chimoio": (-19.116, 33.483),
    "Quelimane": (-17.883, 36.883),
    "Tete": (-16.166, 33.583),
    "Lichinga": (-13.300, 35.266),
    "Pemba": (-12.983, 40.517),
    "Mocuba": (-16.800, 36.950),
    "Xai-Xai": (-25.050, 33.650),
    "Maxixe": (-23.882, 35.330),
    "Inhambane": (-23.876, 35.400),
    "Cuamba": (-14.800, 36.500),
}

COASTAL_CITIES = (
    "Maputo",
    "Matola",
    "Beira",
    "Quelimane",
    "Pemba",
    "Xai-Xai",
    "Maxixe",
    "Inhambane",
)
INLAND_CITIES = tuple(
    city for city in CITY_COORDINATES if city not in COASTAL_CITIES
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_equals(series: pd.Series, value: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").eq(value)


def _facility_key(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]
    return text


def _has_generator(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold().eq("yes")


def _annual_risk(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    tracks = list(expected_track_columns())
    values = frame.loc[:, tracks].apply(pd.to_numeric, errors="coerce")
    valid = values.ge(0.0)
    conditional_mean = values.where(valid).mean(axis=1)
    annual = conditional_mean * (len(tracks) / SYNTHETIC_CATALOGUE_YEARS)
    return annual, valid.sum(axis=1)


def _hospital_metadata(hospitals: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    wanted = (
        "node_id",
        "facility_i",
        "urban",
        "has_genera",
        "ele_any_so",
        "ele_main_s",
        "geometry",
    )
    missing = [column for column in wanted if column not in hospitals]
    if missing:
        raise ValueError(f"SARA data is missing required columns: {missing}")
    metadata = hospitals.loc[:, wanted].copy()
    metadata["node_id"] = metadata["node_id"].astype(str)
    return gpd.GeoDataFrame(metadata, geometry="geometry", crs=hospitals.crs)


def prepare_public_layer(
    public_summary: Path,
    hospitals: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return all recorded public-grid hospitals with hybrid risk attached."""

    summary = load_power_summary(
        public_summary,
        require_complete=True,
        allow_unavailable=True,
    )
    annual, valid_tracks = _annual_risk(summary)
    result = summary.loc[:, ["node_id", "facility_i"]].copy()
    result["annual_failure_probability"] = annual
    result["valid_track_count"] = valid_tracks

    metadata = _hospital_metadata(hospitals)
    public_mask = (
        _numeric_equals(metadata["ele_any_so"], 1)
        & _numeric_equals(metadata["ele_main_s"], 1)
    )
    public = metadata.loc[public_mask].merge(
        result,
        on="node_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    facility_mismatch = public["facility_i_result"].notna() & public[
        "facility_i"
    ].map(_facility_key).ne(public["facility_i_result"].map(_facility_key))
    if facility_mismatch.any():
        examples = public.loc[facility_mismatch, "node_id"].head().tolist()
        raise ValueError(
            "Hybrid public identities disagree with SARA for nodes: "
            f"{examples}"
        )

    public["result_status"] = np.select(
        [
            public["facility_i_result"].isna(),
            public["annual_failure_probability"].isna(),
        ],
        ["not_in_hybrid_aggregate", "hybrid_unavailable"],
        default="evaluated",
    )
    public["electricity_source"] = "public_grid_hybrid"
    public["has_generator"] = _has_generator(public["has_genera"])
    return gpd.GeoDataFrame(public, geometry="geometry", crs=hospitals.crs)


def prepare_solar_layer(
    solar_root: Path,
    hospitals: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return the independently modeled solar cohort and annual risk."""

    summary = load_solar_summary(solar_root, hospitals, zero_empty_events=True)
    annual, valid_tracks = _annual_risk(summary)
    result = summary.loc[:, ["node_id", "facility_i"]].copy()
    result["annual_failure_probability"] = annual
    result["valid_track_count"] = valid_tracks

    metadata = _hospital_metadata(hospitals)
    solar = metadata.merge(
        result,
        on="node_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    solar = solar.loc[solar["facility_i_result"].notna()].copy()
    facility_mismatch = solar["facility_i"].map(_facility_key).ne(
        solar["facility_i_result"].map(_facility_key)
    )
    if facility_mismatch.any():
        examples = solar.loc[facility_mismatch, "node_id"].head().tolist()
        raise ValueError(
            "Solar identities disagree with SARA for nodes: " + str(examples)
        )
    solar["result_status"] = "evaluated"
    solar["electricity_source"] = "independent_solar"
    solar["has_generator"] = _has_generator(solar["has_genera"])
    return gpd.GeoDataFrame(solar, geometry="geometry", crs=hospitals.crs)


def _load_admin3(path: Path) -> gpd.GeoDataFrame:
    admin3 = gpd.read_file(path)
    if admin3.crs is None:
        raise ValueError(f"Administrative boundaries have no CRS: {path}")
    return admin3.to_crs(epsg=4326)


def _city_axes(
    figure: plt.Figure,
    cell,
    cities: tuple[str, ...],
    columns: int,
) -> dict[str, plt.Axes]:
    grid = cell.subgridspec(
        2,
        columns,
        wspace=0.015,
        hspace=0.015,
    )
    return {
        city: figure.add_subplot(grid[index // columns, index % columns])
        for index, city in enumerate(cities)
    }


def _draw_city(
    axis: plt.Axes,
    city: str,
    layer: gpd.GeoDataFrame,
    admin3: gpd.GeoDataFrame,
    norm: colors.Normalize,
    *,
    marker_size: float,
    zoom_buffer: float,
) -> dict[str, int]:
    latitude, longitude = CITY_COORDINATES[city]
    city_window = Point(longitude, latitude).buffer(zoom_buffer)
    local_admin = admin3.loc[admin3.intersects(city_window)]
    if len(local_admin):
        local_admin.plot(
            ax=axis,
            facecolor="#d9d9d9",
            edgecolor="#202020",
            linewidth=0.45,
            zorder=0,
        )
    else:
        axis.set_facecolor("#d9d9d9")

    local = layer.loc[
        layer.geometry.x.between(longitude - zoom_buffer, longitude + zoom_buffer)
        & layer.geometry.y.between(latitude - zoom_buffer, latitude + zoom_buffer)
    ]
    evaluated = local.loc[local["result_status"].eq("evaluated")]
    unavailable = local.loc[~local["result_status"].eq("evaluated")]
    if len(evaluated):
        evaluated.plot(
            ax=axis,
            column="annual_failure_probability",
            cmap="BuGn",
            norm=norm,
            markersize=marker_size,
            edgecolor="none",
            zorder=3,
        )
    if len(unavailable):
        unavailable.plot(
            ax=axis,
            color="#bdbdbd",
            markersize=marker_size,
            edgecolor="#555555",
            linewidth=0.25,
            zorder=3,
        )
    generators = local.loc[local["has_generator"]]
    if len(generators):
        generators.plot(
            ax=axis,
            facecolor="none",
            edgecolor="deeppink",
            linewidth=0.55,
            markersize=marker_size,
            zorder=4,
        )

    axis.set_xlim(longitude - zoom_buffer, longitude + zoom_buffer)
    axis.set_ylim(latitude - zoom_buffer, latitude + zoom_buffer)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.6)
    axis.text(
        0.025,
        0.975,
        city,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.76,
            "edgecolor": "none",
            "pad": 0.8,
        },
        zorder=5,
    )
    return {
        "facilities": int(len(local)),
        "evaluated": int(len(evaluated)),
        "unavailable": int(len(unavailable)),
    }


def render_figure(
    public: gpd.GeoDataFrame,
    solar: gpd.GeoDataFrame,
    admin3: gpd.GeoDataFrame,
    *,
    color_maximum: float,
    zoom_buffer: float,
) -> tuple[plt.Figure, dict[str, dict[str, int]]]:
    """Render the four source panels in one reproducible composite."""

    public_urban = public.loc[_numeric_equals(public["urban"], 1)].copy()
    solar_urban = solar.loc[_numeric_equals(solar["urban"], 1)].copy()
    norm = colors.Normalize(vmin=0.0, vmax=color_maximum, clip=True)

    figure = plt.figure(figsize=(12.4, 6.2))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(4.0, 3.0),
        left=0.075,
        right=0.925,
        bottom=0.12,
        top=0.91,
        wspace=0.035,
        hspace=0.10,
    )
    axes = {
        (0, "coastal"): _city_axes(
            figure, outer[0, 0], COASTAL_CITIES, 4
        ),
        (0, "inland"): _city_axes(figure, outer[0, 1], INLAND_CITIES, 3),
        (1, "coastal"): _city_axes(
            figure, outer[1, 0], COASTAL_CITIES, 4
        ),
        (1, "inland"): _city_axes(figure, outer[1, 1], INLAND_CITIES, 3),
    }

    city_counts: dict[str, dict[str, int]] = {}
    for row, layer, size, source in (
        (0, public_urban, 25.0, "public_grid_hybrid"),
        (1, solar_urban, 34.0, "independent_solar"),
    ):
        for region, cities in (
            ("coastal", COASTAL_CITIES),
            ("inland", INLAND_CITIES),
        ):
            for city in cities:
                city_counts[f"{source}:{city}"] = _draw_city(
                    axes[(row, region)][city],
                    city,
                    layer,
                    admin3,
                    norm,
                    marker_size=size,
                    zoom_buffer=zoom_buffer,
                )

    figure.text(0.32, 0.958, "Coastal", ha="center", va="center", fontsize=13)
    figure.text(0.735, 0.958, "Inland", ha="center", va="center", fontsize=13)
    figure.text(
        0.023,
        0.695,
        "Depend on public infrastructure",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.text(
        0.023,
        0.325,
        "Self-reliant electricity source",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    for x, y, label in (
        (0.051, 0.918, "(a)"),
        (0.565, 0.918, "(b)"),
        (0.051, 0.468, "(c)"),
        (0.565, 0.468, "(d)"),
    ):
        figure.text(x, y, label, ha="left", va="bottom", fontsize=10)

    color_axis = figure.add_axes([0.945, 0.21, 0.014, 0.59])
    colorbar = figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap="BuGn"),
        cax=color_axis,
    )
    colorbar.set_ticks(np.linspace(0.0, color_maximum, 6))
    colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Annualized electricity disruption index (yr$^{-1}$)", fontsize=9)

    legend = (
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="deeppink",
            markeredgewidth=0.8,
            markersize=5.5,
            label="Has generator",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#bdbdbd",
            markeredgecolor="#555555",
            markeredgewidth=0.5,
            markersize=5.5,
            label="Hybrid result unavailable",
        ),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.025),
    )
    return figure, city_counts


def _cohort_summary(frame: gpd.GeoDataFrame, color_maximum: float) -> dict:
    risk = pd.to_numeric(frame["annual_failure_probability"], errors="coerce")
    statuses = frame["result_status"].value_counts(dropna=False).to_dict()
    return {
        "facilities": int(len(frame)),
        "urban_facilities": int(_numeric_equals(frame["urban"], 1).sum()),
        "rural_facilities": int(_numeric_equals(frame["urban"], 2).sum()),
        "status_counts": {str(key): int(value) for key, value in statuses.items()},
        "annual_failure_probability_min": (
            float(risk.min()) if risk.notna().any() else None
        ),
        "annual_failure_probability_mean": (
            float(risk.mean()) if risk.notna().any() else None
        ),
        "annual_failure_probability_max": (
            float(risk.max()) if risk.notna().any() else None
        ),
        "above_display_maximum": int(risk.gt(color_maximum).sum()),
        "has_generator": int(frame["has_generator"].sum()),
    }


def write_outputs(
    public: gpd.GeoDataFrame,
    solar: gpd.GeoDataFrame,
    admin3: gpd.GeoDataFrame,
    *,
    public_summary: Path,
    solar_root: Path,
    hospital_data: Path,
    admin3_path: Path,
    output_dir: Path,
    dpi: int,
    color_maximum: float,
    zoom_buffer: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, city_counts = render_figure(
        public,
        solar,
        admin3,
        color_maximum=color_maximum,
        zoom_buffer=zoom_buffer,
    )
    pdf_path = output_dir / "figure_S2_city_electricity_hybrid.pdf"
    png_path = output_dir / "figure_S2_city_electricity_hybrid.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    export_columns = (
        "node_id",
        "facility_i",
        "electricity_source",
        "urban",
        "has_generator",
        "result_status",
        "valid_track_count",
        "annual_failure_probability",
    )
    table_frames = []
    for layer in (public, solar):
        table = layer.loc[:, export_columns].copy()
        table["longitude"] = layer.geometry.x.to_numpy()
        table["latitude"] = layer.geometry.y.to_numpy()
        table_frames.append(table)
    data_path = output_dir / "figure_S2_city_electricity_hybrid_data.csv"
    pd.concat(table_frames, ignore_index=True).to_csv(data_path, index=False)

    metadata = {
        "figure": "Supplementary Figure S2: city electricity failure",
        "method": {
            "public_grid": "final hybrid network model",
            "self_reliant": "independent solar irradiation and damage model",
            "source_notebook": "moz_notebook/moz_paper_figure.ipynb",
            "source_cells": [15, 16, 20, 21, 22, 23, 24],
            "annualization": (
                "conditional synthetic-cyclone mean multiplied by the 2,703 "
                "catalogue events / 10,000 simulated years"
            ),
            "annualization_factor": (
                len(expected_track_columns()) / SYNTHETIC_CATALOGUE_YEARS
            ),
            "legacy_figure_factor_not_used": "21/48",
            "urban_code": 1,
            "unavailable_treatment": "gray; never assigned zero",
            "generator_symbol": "deeppink outline",
            "color_range": [0.0, color_maximum],
            "zoom_buffer_degrees": zoom_buffer,
        },
        "inputs": {
            "public_summary": str(public_summary),
            "public_summary_sha256": _sha256(public_summary),
            "solar_root": str(solar_root),
            "hospital_data": str(hospital_data),
            "admin3_path": str(admin3_path),
        },
        "cohorts": {
            "public_grid": _cohort_summary(public, color_maximum),
            "independent_solar": _cohort_summary(solar, color_maximum),
        },
        "city_panel_counts": city_counts,
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "data": str(data_path),
        },
    }
    metadata_path = output_dir / "figure_S2_city_electricity_hybrid_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "pdf": pdf_path,
        "png": png_path,
        "data": data_path,
        "metadata": metadata_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate city electricity Supplementary Figure S2."
    )
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--solar-root", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--admin3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--color-maximum", type=float, default=DEFAULT_COLOR_MAXIMUM)
    parser.add_argument("--zoom-buffer", type=float, default=DEFAULT_ZOOM_BUFFER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.color_maximum <= 0.0:
        raise ValueError("--color-maximum must be positive")
    if args.zoom_buffer <= 0.0:
        raise ValueError("--zoom-buffer must be positive")

    hospitals = load_hospital_data(args.hospital_data)
    public = prepare_public_layer(args.public_summary, hospitals)
    solar = prepare_solar_layer(args.solar_root, hospitals)
    admin3 = _load_admin3(args.admin3)
    paths = write_outputs(
        public,
        solar,
        admin3,
        public_summary=args.public_summary,
        solar_root=args.solar_root,
        hospital_data=args.hospital_data,
        admin3_path=args.admin3,
        output_dir=args.output_dir,
        dpi=args.dpi,
        color_maximum=args.color_maximum,
        zoom_buffer=args.zoom_buffer,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
