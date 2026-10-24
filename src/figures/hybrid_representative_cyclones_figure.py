"""Plot five representative synthetic cyclones with hybrid hospital failures.

The panel selection and wind footprints reproduce the first five tracks from
synthetic catalogue 0 used by the exploratory paper figure.  Hospital failure
probabilities come from the final hybrid public-grid aggregate.  The hospital
palette is ``BuGn``, matching manuscript Figures 2 and 3, while retaining the
0--100% scale required for single-event probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd

from capacity_ablation_figure import (
    load_boundary,
    load_hospital_data,
    load_power_summary,
)
from synthetic_tracks import TRACK_COUNTS


DEFAULT_TRACKS = ("0_0", "0_1", "0_2", "0_3", "0_4")
WIND_MINIMUM = 18.0
WIND_MAXIMUM = 70.0
FAILURE_MINIMUM = 0.0
FAILURE_MAXIMUM = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_track_key(value: str) -> tuple[int, int]:
    pieces = value.split("_")
    if len(pieces) != 2:
        raise ValueError(f"Invalid track key {value!r}; expected SET_TRACK")
    try:
        storm_set, track_id = (int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError(
            f"Invalid track key {value!r}; set and track must be integers"
        ) from error
    if storm_set < 0 or storm_set >= len(TRACK_COUNTS):
        raise ValueError(f"Storm set is outside 0--{len(TRACK_COUNTS) - 1}: {value}")
    if track_id < 0 or track_id >= TRACK_COUNTS[storm_set]:
        raise ValueError(
            f"Track is outside 0--{TRACK_COUNTS[storm_set] - 1}: {value}"
        )
    return storm_set, track_id


def _wind_path(wind_root: Path, storm_set: int, track_id: int) -> Path:
    folder = f"STORM_DATA_IBTRACS_SI_1000_YEARS_{storm_set}"
    candidates = (
        wind_root / f"{folder}_var_track_{track_id}",
        wind_root / folder / f"{folder}_var_track_{track_id}",
        wind_root / f"circle_track_{track_id}" if storm_set == 0 else None,
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"No wind footprint for {storm_set}_{track_id}; checked {checked}")


def _load_wind(path: Path) -> gpd.GeoDataFrame:
    wind = gpd.read_file(path)
    if wind.crs is None:
        raise ValueError(f"Wind footprint has no CRS: {path}")
    if "speed" not in wind:
        raise ValueError(f"Wind footprint is missing speed: {path}")
    wind = wind.to_crs(epsg=4326).copy()
    wind["speed"] = pd.to_numeric(wind["speed"], errors="coerce")
    if wind["speed"].isna().any():
        raise ValueError(f"Wind footprint contains nonnumeric speeds: {path}")
    return wind.loc[wind["speed"].gt(WIND_MINIMUM)].sort_values("speed")


def _wind_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: _sha256(path)}
    components = {}
    for component in sorted(path.iterdir()):
        if component.is_file():
            components[component.name] = _sha256(component)
    if not components:
        raise FileNotFoundError(f"Wind footprint contains no files: {path}")
    return components


def _global_track_id(storm_set: int, track_id: int) -> int:
    return int(sum(TRACK_COUNTS[:storm_set]) + track_id)


def _generate_draw_cascade_footprints(
    parsed_tracks: list[tuple[int, int]],
    *,
    storm_root: Path,
    er11_dir: Path,
    matlab_helper_dir: Path,
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[str]]:
    """Generate the compact ER11E04 ellipses used by draw_cascade.py."""

    from draw_cascade import (
        cyclone_centers,
        generate_footprints,
        load_track_sources,
        select_track,
    )

    sources: dict[int, tuple[object, object]] = {}
    center_rows: list[dict[str, float | int]] = []
    descriptions = []
    for storm_set, track_id in parsed_tracks:
        if storm_set not in sources:
            sources[storm_set] = load_track_sources(storm_root, storm_set)
        catalogue, all_tracks = sources[storm_set]
        track = select_track(catalogue, all_tracks, track_id)
        global_id = _global_track_id(storm_set, track_id)
        center_rows.extend(cyclone_centers(global_id, track))
        descriptions.append(
            f"draw_cascade ER11E04 ellipses for {storm_set}_{track_id}"
        )

    centers = pd.DataFrame(
        center_rows,
        columns=[
            "global_id",
            "longitude",
            "latitude",
            "maximum_wind_ms",
            "radius_m",
        ],
    )
    if centers.empty:
        footprints = pd.DataFrame(
            columns=[
                "global_id",
                "longitude",
                "latitude",
                "wind_speed_ms",
                "radius_m",
            ]
        )
    else:
        footprints = generate_footprints(
            centers,
            er11_dir=er11_dir,
            matlab_helper_dir=matlab_helper_dir,
        )
    layers = [
        footprints.loc[
            footprints["global_id"].eq(_global_track_id(storm_set, track_id))
        ].copy()
        for storm_set, track_id in parsed_tracks
    ]
    return layers, footprints, descriptions


def _attach_geometry(
    summary: pd.DataFrame,
    hospitals: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    geometry = hospitals.loc[:, ["node_id", "geometry"]].copy()
    geometry["node_id"] = geometry["node_id"].astype(str)
    attached = summary.merge(
        geometry,
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    if attached["geometry"].isna().any():
        examples = attached.loc[attached["geometry"].isna(), "node_id"].head()
        raise ValueError(
            "Hybrid-result hospitals are missing from SARA geometry: "
            + ", ".join(examples.astype(str))
        )
    return gpd.GeoDataFrame(attached, geometry="geometry", crs=hospitals.crs)


def build_figure(
    public_summary: Path,
    hospital_data: Path,
    boundary_data: Path,
    wind_root: Path | None,
    output_dir: Path,
    track_keys: tuple[str, ...],
    dpi: int,
    wind_style: str = "shapefile",
    storm_root: Path | None = None,
    er11_dir: Path | None = None,
    matlab_helper_dir: Path | None = None,
) -> dict[str, object]:
    if len(track_keys) != 5:
        raise ValueError("This figure requires exactly five track keys")

    parsed_tracks = [_parse_track_key(key) for key in track_keys]
    summary = load_power_summary(
        public_summary,
        require_complete=True,
        allow_unavailable=True,
    )
    hospitals = load_hospital_data(hospital_data).to_crs(epsg=4326)
    hospital_results = _attach_geometry(summary, hospitals)
    boundary = load_boundary(boundary_data).to_crs(epsg=4326)

    draw_cascade_footprints = None
    if wind_style == "shapefile":
        if wind_root is None:
            raise ValueError("wind_root is required for shapefile wind rendering")
        wind_paths: list[Path | str] = [
            _wind_path(wind_root, storm_set, track_id)
            for storm_set, track_id in parsed_tracks
        ]
        wind_layers = [_load_wind(Path(path)) for path in wind_paths]
    elif wind_style == "draw-cascade":
        required = {
            "storm_root": storm_root,
            "er11_dir": er11_dir,
            "matlab_helper_dir": matlab_helper_dir,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise ValueError(
                "draw-cascade wind rendering requires: " + ", ".join(missing)
            )
        wind_layers, draw_cascade_footprints, descriptions = (
            _generate_draw_cascade_footprints(
                parsed_tracks,
                storm_root=storm_root,
                er11_dir=er11_dir,
                matlab_helper_dir=matlab_helper_dir,
            )
        )
        wind_paths = descriptions
    else:
        raise ValueError(f"Unknown wind style: {wind_style}")

    wind_norm = colors.Normalize(WIND_MINIMUM, WIND_MAXIMUM)
    failure_norm = colors.Normalize(FAILURE_MINIMUM, FAILURE_MAXIMUM)
    wind_cmap = plt.get_cmap("Oranges")
    failure_cmap = plt.get_cmap("BuGn")

    figure = plt.figure(figsize=(16.0, 5.1))
    grid = figure.add_gridspec(
        1,
        7,
        width_ratios=(0.055, 1, 1, 1, 1, 1, 0.055),
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.985,
        wspace=0.055,
    )
    wind_color_axis = figure.add_subplot(grid[0, 0])
    axes = [figure.add_subplot(grid[0, index]) for index in range(1, 6)]
    failure_color_axis = figure.add_subplot(grid[0, 6])

    records = []
    panel_metadata = []
    for axis, track_key, wind_path, wind, parsed in zip(
        axes,
        track_keys,
        wind_paths,
        wind_layers,
        parsed_tracks,
    ):
        boundary.plot(
            ax=axis,
            color="#f5f5f5" if wind_style == "draw-cascade" else "#f1f1f1",
            edgecolor="none",
            zorder=0,
        )
        if wind_style == "draw-cascade":
            from draw_cascade import footprint_patches

            patches, speeds = footprint_patches(wind)
            if patches:
                wind_layer = PatchCollection(
                    patches,
                    cmap=wind_cmap,
                    norm=wind_norm,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.92,
                    zorder=1,
                )
                wind_layer.set_array(np.asarray(speeds, dtype=float))
                axis.add_collection(wind_layer)
        else:
            wind.plot(
                ax=axis,
                column="speed",
                cmap=wind_cmap,
                norm=wind_norm,
                linewidth=0,
                zorder=1,
            )
        boundary.boundary.plot(
            ax=axis,
            color="#252525",
            linewidth=0.9 if wind_style == "draw-cascade" else 0.65,
            zorder=2,
        )

        plotted = hospital_results.loc[
            :, ["node_id", "facility_i", track_key, "geometry"]
        ].copy()
        plotted = plotted.rename(columns={track_key: "failure_probability"})
        plotted["result_status"] = np.where(
            plotted["failure_probability"].ge(0.0),
            "evaluated",
            "unavailable",
        )
        unavailable = plotted.loc[plotted["result_status"].eq("unavailable")]
        evaluated = plotted.loc[plotted["result_status"].eq("evaluated")]

        if len(unavailable):
            unavailable.plot(
                ax=axis,
                color="silver",
                edgecolor="#202020",
                linewidth=0.25,
                markersize=11,
                zorder=3,
            )
        evaluated.plot(
            ax=axis,
            column="failure_probability",
            cmap=failure_cmap,
            norm=failure_norm,
            edgecolor="#202020",
            linewidth=0.12,
            markersize=11,
            zorder=4,
        )

        axis.set_xlim(30, 41)
        axis.set_ylim(-27, -10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#303030")
            spine.set_linewidth(0.55)

        exported = pd.DataFrame(
            {
                "track_key": track_key,
                "node_id": plotted["node_id"].astype(str),
                "facility_i": plotted["facility_i"],
                "longitude": plotted.geometry.x,
                "latitude": plotted.geometry.y,
                "failure_probability": plotted["failure_probability"],
                "result_status": plotted["result_status"],
            }
        )
        records.append(exported)
        panel_metadata.append(
            {
                "track_key": track_key,
                "storm_set": parsed[0],
                "track_id": parsed[1],
                "wind_footprint": str(wind_path),
                "wind_rendering": wind_style,
                "wind_shape_count_above_18_ms": int(len(wind)),
                "evaluated_hospitals": int(len(evaluated)),
                "unavailable_hospitals": int(len(unavailable)),
                "mean_failure_probability": float(
                    evaluated["failure_probability"].mean()
                ),
                "maximum_failure_probability": float(
                    evaluated["failure_probability"].max()
                ),
                "hospitals_with_positive_failure_probability": int(
                    evaluated["failure_probability"].gt(0.0).sum()
                ),
            }
        )

    wind_colorbar = figure.colorbar(
        cm.ScalarMappable(norm=wind_norm, cmap=wind_cmap),
        cax=wind_color_axis,
    )
    wind_colorbar.set_ticks([WIND_MINIMUM, WIND_MAXIMUM])
    wind_colorbar.set_label("Wind speed (m s$^{-1}$)", fontsize=10)
    wind_color_axis.yaxis.set_ticks_position("left")
    wind_color_axis.yaxis.set_label_position("left")

    failure_colorbar = figure.colorbar(
        cm.ScalarMappable(norm=failure_norm, cmap=failure_cmap),
        cax=failure_color_axis,
    )
    failure_colorbar.set_ticks(np.linspace(0.0, 1.0, 5))
    failure_colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    failure_colorbar.set_label("Event electricity disruption, $\\hat{s}_h(z)$", fontsize=10)

    for color_axis in (wind_color_axis, failure_color_axis):
        color_axis.tick_params(labelsize=8, length=2.5, width=0.5)
        for spine in color_axis.spines.values():
            spine.set_color("#303030")
            spine.set_linewidth(0.55)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "figure_representative_cyclones_hybrid"
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")
    data_path = output_dir / "figure_representative_cyclones_hybrid_data.csv"
    metadata_path = output_dir / "figure_representative_cyclones_hybrid_metadata.json"
    footprint_path = (
        output_dir / "figure_representative_cyclones_hybrid_wind_footprints.csv"
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    pd.concat(records, ignore_index=True).to_csv(data_path, index=False)
    if draw_cascade_footprints is not None:
        draw_cascade_footprints.to_csv(footprint_path, index=False)
    metadata = {
        "schema_version": 1,
        "description": (
            "Five synthetic-cyclone wind footprints and event-specific hybrid "
            "public-grid hospital failure probabilities"
        ),
        "track_keys": list(track_keys),
        "hospital_color_map": "BuGn",
        "hospital_color_range": [FAILURE_MINIMUM, FAILURE_MAXIMUM],
        "wind_color_map": "Oranges",
        "wind_color_range_ms": [WIND_MINIMUM, WIND_MAXIMUM],
        "wind_rendering": wind_style,
        "public_summary": str(public_summary),
        "public_summary_sha256": _sha256(public_summary),
        "hospital_data": str(hospital_data),
        "boundary_data": str(boundary_data),
        "wind_footprints": (
            {
                track_key: {
                    "path": str(path),
                    "component_sha256": _wind_hashes(Path(path)),
                }
                for track_key, path in zip(track_keys, wind_paths)
            }
            if wind_style == "shapefile"
            else {
                "method": "draw_cascade ER11E04 elliptical patches",
                "storm_root": str(storm_root),
                "er11_dir": str(er11_dir),
                "matlab_helper_dir": str(matlab_helper_dir),
                "compact_footprint_csv": str(footprint_path),
            }
        ),
        "panels": panel_metadata,
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "data_csv": str(data_path),
            "wind_footprint_csv": (
                str(footprint_path)
                if draw_cascade_footprints is not None
                else None
            ),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--wind-root", type=Path)
    parser.add_argument(
        "--wind-style",
        choices=("shapefile", "draw-cascade"),
        default="shapefile",
    )
    parser.add_argument("--storm-root", type=Path)
    parser.add_argument("--er11-dir", type=Path)
    parser.add_argument("--matlab-helper-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracks",
        default=",".join(DEFAULT_TRACKS),
        help="Comma-separated SET_TRACK keys; exactly five are required",
    )
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    track_keys = tuple(
        value.strip() for value in arguments.tracks.split(",") if value.strip()
    )
    metadata = build_figure(
        public_summary=arguments.public_summary,
        hospital_data=arguments.hospital_data,
        boundary_data=arguments.boundary,
        wind_root=arguments.wind_root,
        output_dir=arguments.output_dir,
        track_keys=track_keys,
        dpi=arguments.dpi,
        wind_style=arguments.wind_style,
        storm_root=arguments.storm_root,
        er11_dir=arguments.er11_dir,
        matlab_helper_dir=arguments.matlab_helper_dir,
    )
    print(json.dumps(metadata["panels"], indent=2))


if __name__ == "__main__":
    main()
