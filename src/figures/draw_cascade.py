#!/usr/bin/env python3
"""Draw cyclone wind footprints and hospital failure probabilities by track.

This is the batch-capable form of the original ``draw_cascade.py`` workflow.
For every synthetic event it overlays:

* ER11E04 wind-intensity footprints generated from the same Year/Month/TCID,
  MaxSpeed and Raius inputs used by ``draw_cyclone_single.py``; and
* the event-specific failure probability of every public-grid hospital from
  the current non-ablation ``df_final_public_25.csv`` result.

The renderer keeps one MATLAB engine alive for a chunk of tracks and writes
PNG files atomically, so interrupted Slurm arrays can safely be rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


TRACK_COUNTS = (290, 274, 263, 276, 273, 279, 251, 264, 242, 291)
TOTAL_TRACKS = sum(TRACK_COUNTS)
FOLDER_PREFIX = "STORM_DATA_IBTRACS_SI_1000_YEARS_"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def storm_folder(storm_set: int) -> str:
    if not 0 <= storm_set < len(TRACK_COUNTS):
        raise ValueError(f"storm set must be between 0 and 9: {storm_set}")
    return f"{FOLDER_PREFIX}{storm_set}"


def global_to_track(global_id: int) -> tuple[int, int]:
    if not 0 <= global_id < TOTAL_TRACKS:
        raise ValueError(f"global track ID must be in [0, {TOTAL_TRACKS}): {global_id}")
    offset = 0
    for storm_set, count in enumerate(TRACK_COUNTS):
        if global_id < offset + count:
            return storm_set, global_id - offset
        offset += count
    raise AssertionError("unreachable global track mapping")


def output_path(output_root: Path, storm_set: int, track_id: int) -> Path:
    return (
        output_root
        / f"dataset_{storm_set}"
        / f"track_{track_id:04d}.png"
    )


def valid_png(path: Path, minimum_bytes: int = 10_000) -> bool:
    try:
        if path.stat().st_size < minimum_bytes:
            return False
        with path.open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Ellipse

    return (
        gpd,
        plt,
        np,
        pd,
        colormaps,
        ScalarMappable,
        PatchCollection,
        Normalize,
        Ellipse,
    )


def load_mozambique_boundary(countries_path: Path):
    gpd, *_ = _plot_modules()
    countries = gpd.read_file(countries_path)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    else:
        countries = countries.to_crs("EPSG:4326")
    for column in ("COUNTRY", "ADMIN", "NAME", "COUNTRYAFF", "CNTRY_NAME", "name"):
        if column not in countries.columns:
            continue
        mask = countries[column].astype(str).str.casefold().eq("mozambique")
        if mask.any():
            return countries.loc[mask, ["geometry"]].copy()
    raise ValueError(
        f"could not identify Mozambique in {countries_path}; "
        f"columns={list(countries.columns)}"
    )


def load_hospital_results(hospital_path: Path, result_csv: Path):
    gpd, _, np, pd, *_ = _plot_modules()
    hospitals = gpd.read_file(hospital_path).to_crs("EPSG:4326")
    results = pd.read_csv(result_csv)
    required = {"node_id", "facility_i"}
    if not required.issubset(results.columns):
        raise ValueError(f"hospital result lacks {sorted(required)}: {result_csv}")
    expected_events = {
        f"{storm_set}_{track_id}"
        for storm_set, count in enumerate(TRACK_COUNTS)
        for track_id in range(count)
    }
    event_columns = set(results.columns) - required
    if event_columns != expected_events:
        missing = sorted(expected_events - event_columns)[:10]
        extra = sorted(event_columns - expected_events)[:10]
        raise ValueError(
            f"hospital result event columns do not match the 2,703-track catalogue; "
            f"missing={missing}, extra={extra}"
        )
    if len(results) != 720:
        raise ValueError(f"expected 720 public-grid hospitals, found {len(results)}")
    if results["facility_i"].duplicated().any() or results["node_id"].duplicated().any():
        raise ValueError("hospital result contains duplicate facility or node IDs")
    geometry = hospitals[["facility_i", "ele_main_s", "geometry"]]
    merged = results.merge(
        geometry,
        on="facility_i",
        how="left",
        validate="one_to_one",
    )
    if merged["geometry"].isna().any():
        raise ValueError("one or more result hospitals lack SARA geometry")
    if not np.allclose(merged["ele_main_s"].to_numpy(dtype=float), 1.0):
        raise ValueError("the public-grid result contains a non-public hospital")
    values = merged[sorted(expected_events)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("hospital failure probabilities must be finite and within [0, 1]")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def load_track_sources(storm_root: Path, storm_set: int):
    gpd, _, _, pd, *_ = _plot_modules()
    folder = storm_folder(storm_set)
    catalogue = gpd.read_file(storm_root / "Moz_cyclone_track" / folder)
    tracks = pd.read_csv(storm_root / "all_tracks" / f"{folder}.csv")
    required_catalogue = {"Year", "Month", "TCID"}
    required_tracks = required_catalogue | {
        "Timestep",
        "Lon",
        "Lat",
        "MaxSpeed",
        "Raius",
    }
    if not required_catalogue.issubset(catalogue.columns):
        raise ValueError("cyclone catalogue lacks Year, Month or TCID")
    if not required_tracks.issubset(tracks.columns):
        raise ValueError("all-tracks CSV lacks required cyclone fields")
    if len(catalogue) != TRACK_COUNTS[storm_set]:
        raise ValueError(
            f"set {storm_set} has {len(catalogue)} tracks; "
            f"expected {TRACK_COUNTS[storm_set]}"
        )
    return catalogue.reset_index(drop=True), tracks


def select_track(catalogue, all_tracks, track_id: int):
    _, _, np, _, *_ = _plot_modules()
    event = catalogue.iloc[track_id]
    mask = (
        np.isclose(all_tracks["Year"].to_numpy(dtype=float), float(event["Year"]))
        & np.isclose(all_tracks["Month"].to_numpy(dtype=float), float(event["Month"]))
        & np.isclose(all_tracks["TCID"].to_numpy(dtype=float), float(event["TCID"]))
    )
    track = all_tracks.loc[
        mask, ["Timestep", "Lon", "Lat", "MaxSpeed", "Raius"]
    ].copy()
    track = track.replace([np.inf, -np.inf], np.nan).dropna()
    track = track.sort_values("Timestep", kind="stable").reset_index(drop=True)
    if track.empty:
        raise ValueError(
            f"no track matches Year={event['Year']}, Month={event['Month']}, "
            f"TCID={event['TCID']}"
        )
    return track


def interpolate_track(track, subdivisions: int = 6):
    _, _, np, pd, *_ = _plot_modules()
    values = track[["Lon", "Lat", "MaxSpeed", "Raius"]].to_numpy(dtype=float)
    if len(values) == 1:
        return track[["Lon", "Lat", "MaxSpeed", "Raius"]].copy()
    pieces = [values[:1]]
    for first, second in zip(values[:-1], values[1:]):
        fractions = np.arange(1, subdivisions + 1, dtype=float) / subdivisions
        pieces.append(first + fractions[:, None] * (second - first))
    return pd.DataFrame(
        np.vstack(pieces), columns=["Lon", "Lat", "MaxSpeed", "Raius"]
    )


def cyclone_centers(
    global_id: int,
    track,
    minimum_wind: float = 18.0,
    wind_interval: float = 2.0,
) -> list[dict[str, float | int]]:
    """Return interpolated cyclone centers that require ER11 profiles."""

    rows: list[dict[str, float | int]] = []
    dense = interpolate_track(track)
    for row in dense.itertuples(index=False):
        lon = float(row.Lon)
        lat = float(row.Lat)
        maximum_wind = float(row.MaxSpeed)
        radius_m = float(row.Raius) * 1000.0
        if lon > 41.5 or lat > -9.5 or lat < -27.5:
            continue
        highest_level = math.floor(maximum_wind / wind_interval) * wind_interval
        if highest_level <= minimum_wind or radius_m <= 0.0:
            continue
        rows.append(
            {
                "global_id": global_id,
                "longitude": lon,
                "latitude": lat,
                "maximum_wind_ms": maximum_wind,
                "radius_m": radius_m,
            }
        )
    return rows


def matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def generate_footprints(
    centers,
    *,
    er11_dir: Path,
    matlab_helper_dir: Path,
):
    """Run one MATLAB batch process for all cyclone centers in a job chunk."""

    _, _, _, pd, *_ = _plot_modules()
    scratch = os.environ.get("L_SCRATCH")
    with tempfile.TemporaryDirectory(prefix="cascade-er11-", dir=scratch) as directory:
        work = Path(directory)
        input_csv = work / "cyclone_centers.csv"
        output_csv = work / "wind_footprints.csv"
        centers.to_csv(input_csv, index=False)
        expression = (
            f"addpath('{matlab_quote(matlab_helper_dir)}'); "
            f"generate_cyclone_footprints('{matlab_quote(input_csv)}',"
            f"'{matlab_quote(output_csv)}','{matlab_quote(er11_dir)}')"
        )
        subprocess.run(
            ["matlab", "-singleCompThread", "-batch", expression],
            check=True,
        )
        if not output_csv.is_file():
            raise RuntimeError("MATLAB did not write the ER11 footprint table")
        footprints = pd.read_csv(output_csv)
    required = {"global_id", "longitude", "latitude", "wind_speed_ms", "radius_m"}
    if not required.issubset(footprints.columns):
        raise ValueError(
            f"MATLAB footprint table lacks {sorted(required - set(footprints.columns))}"
        )
    return footprints


def footprint_patches(footprints) -> tuple[list[object], list[float]]:
    """Convert the compact MATLAB radius table to map ellipses."""

    *_, Ellipse = _plot_modules()
    patches: list[object] = []
    speeds: list[float] = []
    for row in footprints.itertuples(index=False):
        lon = float(row.longitude)
        lat = float(row.latitude)
        radius_km = max(0.0, float(row.radius_m) / 1000.0)
        km_per_degree_lon = 111.32 * max(0.2, math.cos(math.radians(lat)))
        patches.append(
            Ellipse(
                (lon, lat),
                width=2.0 * radius_km / km_per_degree_lon,
                height=2.0 * radius_km / 110.57,
            )
        )
        speeds.append(float(row.wind_speed_ms))

    order = sorted(range(len(speeds)), key=speeds.__getitem__)
    return [patches[index] for index in order], [speeds[index] for index in order]


def atomic_figure(figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def render_track(
    *,
    storm_set: int,
    track_id: int,
    track,
    footprints,
    hospitals,
    boundary,
    output_root: Path,
    dpi: int,
    overwrite: bool,
) -> Path:
    (
        _,
        plt,
        np,
        _,
        colormaps,
        ScalarMappable,
        PatchCollection,
        Normalize,
        _,
    ) = _plot_modules()
    destination = output_path(output_root, storm_set, track_id)
    if valid_png(destination) and not overwrite:
        return destination

    patches, speeds = footprint_patches(footprints)
    event_column = f"{storm_set}_{track_id}"
    failures = hospitals[event_column].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.5, 9.5))
    boundary.plot(ax=axis, color="#f5f5f5", edgecolor="none", zorder=0)
    wind_norm = Normalize(vmin=18.0, vmax=50.0, clip=True)
    wind_cmap = colormaps["Oranges"]
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
    boundary.boundary.plot(ax=axis, color="#252525", linewidth=0.9, zorder=3)

    hospital_layer = axis.scatter(
        hospitals.geometry.x,
        hospitals.geometry.y,
        c=failures,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=25,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="#78d5e3",
        zorder=4,
    )
    wind_mappable = ScalarMappable(norm=wind_norm, cmap=wind_cmap)
    wind_mappable.set_array([])
    wind_bar = figure.colorbar(
        wind_mappable,
        ax=axis,
        orientation="horizontal",
        fraction=0.035,
        pad=0.035,
    )
    wind_bar.set_label("Cyclone wind intensity (m s$^{-1}$)")
    failure_bar = figure.colorbar(
        hospital_layer,
        ax=axis,
        orientation="horizontal",
        fraction=0.035,
        pad=0.085,
    )
    failure_bar.set_label("Hospital public-grid failure probability")

    maximum_wind = float(track["MaxSpeed"].max())
    mean_failure = float(np.mean(failures))
    axis.set_xlim(30.0, 41.0)
    axis.set_ylim(-27.0, -10.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"Synthetic cyclone set {storm_set}, track {track_id}\n"
        f"Peak wind {maximum_wind:.1f} m s$^{{-1}}$; "
        f"mean hospital failure {100.0 * mean_failure:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    figure.text(
        0.5,
        0.012,
        "Wind field: ER11E04; points: 720 hospitals connected to the public grid.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    atomic_figure(figure, destination, dpi=dpi)
    plt.close(figure)
    if not valid_png(destination):
        raise RuntimeError(f"rendered figure failed validation: {destination}")
    return destination


def render_range(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.global_start <= args.global_stop <= TOTAL_TRACKS:
        raise ValueError(
            f"invalid global interval [{args.global_start}, {args.global_stop})"
        )
    requested = list(range(args.global_start, args.global_stop))
    incomplete = [
        global_id
        for global_id in requested
        if args.overwrite
        or not valid_png(
            output_path(args.output_root, *global_to_track(global_id))
        )
    ]
    if not incomplete:
        return {
            "global_start": args.global_start,
            "global_stop": args.global_stop,
            "rendered": 0,
            "skipped": len(requested),
            "output_root": str(args.output_root),
        }

    hospitals = load_hospital_results(args.hospitals, args.result_csv)
    boundary = load_mozambique_boundary(args.countries)
    sources: dict[int, tuple[object, object]] = {}
    tracks: dict[int, tuple[int, int, object]] = {}
    center_rows: list[dict[str, float | int]] = []
    for global_id in incomplete:
        storm_set, track_id = global_to_track(global_id)
        if storm_set not in sources:
            sources[storm_set] = load_track_sources(args.storm_root, storm_set)
        catalogue, all_tracks = sources[storm_set]
        track = select_track(catalogue, all_tracks, track_id)
        tracks[global_id] = storm_set, track_id, track
        center_rows.extend(cyclone_centers(global_id, track))

    _, _, _, pd, *_ = _plot_modules()
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
            columns=["global_id", "longitude", "latitude", "wind_speed_ms", "radius_m"]
        )
    else:
        footprints = generate_footprints(
            centers,
            er11_dir=args.er11_dir,
            matlab_helper_dir=args.matlab_helper_dir,
        )

    for position, global_id in enumerate(incomplete, start=1):
        storm_set, track_id, track = tracks[global_id]
        event_footprints = footprints.loc[footprints["global_id"] == global_id]
        render_track(
            storm_set=storm_set,
            track_id=track_id,
            track=track,
            footprints=event_footprints,
            hospitals=hospitals,
            boundary=boundary,
            output_root=args.output_root,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
        print(
            f"rendered {position}/{len(incomplete)}: "
            f"set {storm_set}, track {track_id}",
            flush=True,
        )
    return {
        "global_start": args.global_start,
        "global_stop": args.global_stop,
        "rendered": len(incomplete),
        "skipped": len(requested) - len(incomplete),
        "output_root": str(args.output_root),
    }


def validate_outputs(output_root: Path, report_path: Path) -> dict[str, object]:
    missing: list[str] = []
    invalid: list[str] = []
    total_bytes = 0
    for storm_set, count in enumerate(TRACK_COUNTS):
        for track_id in range(count):
            path = output_path(output_root, storm_set, track_id)
            if not path.exists():
                missing.append(str(path))
            elif not valid_png(path):
                invalid.append(str(path))
            else:
                total_bytes += path.stat().st_size
    report = {
        "schema_version": 1,
        "expected_figures": TOTAL_TRACKS,
        "valid_figures": TOTAL_TRACKS - len(missing) - len(invalid),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "figure_bytes": total_bytes,
        "missing": missing[:100],
        "invalid": invalid[:100],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(report_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    if missing or invalid:
        raise RuntimeError(
            f"cascade validation failed: {len(missing)} missing, "
            f"{len(invalid)} invalid"
        )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render a global track interval")
    render.add_argument("--global-start", type=int, required=True)
    render.add_argument("--global-stop", type=int, required=True)
    render.add_argument(
        "--storm-root",
        type=Path,
        default=Path("/scratch/users/zywei/cyclone_map/STORM"),
    )
    render.add_argument(
        "--result-csv",
        type=Path,
        default=Path("/scratch/users/zywei/wind_result/df_final_public_25.csv"),
    )
    render.add_argument("--hospitals", type=Path, default=Path("SARA_processed"))
    render.add_argument(
        "--countries", type=Path, default=Path("/scratch/users/zywei/World_Countries")
    )
    render.add_argument("--er11-dir", type=Path, default=Path("ER11E04"))
    render.add_argument("--matlab-helper-dir", type=Path, default=Path("matlab"))
    render.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/users/zywei/wind_result/cascade_capacity_v2_public"),
    )
    render.add_argument("--dpi", type=int, default=150)
    render.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="validate all 2,703 maps")
    validate.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/users/zywei/wind_result/cascade_capacity_v2_public"),
    )
    validate.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/scratch/users/zywei/wind_result/cascade_capacity_v2_public/"
            "validation_report.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "render":
        report = render_range(args)
    else:
        report = validate_outputs(args.output_root, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
