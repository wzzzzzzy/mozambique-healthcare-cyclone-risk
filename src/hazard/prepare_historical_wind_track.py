#!/usr/bin/env python3
"""Generate one historical-node wind layer with the Chavas MATLAB model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from utils import generate_shp_some_nodes


EARTH_RADIUS_M = 6_371_008.8
KNOT_TO_MS = 0.514444
NAUTICAL_MILE_TO_M = 1852.0


def load_country(path: Path):
    countries = gpd.read_file(path)
    if countries.crs is None:
        raise ValueError(f"country layer has no CRS: {path}")
    countries = countries.to_crs(epsg=4326)
    for column in ("COUNTRY", "ADMIN", "NAME_0", "ADM0_EN"):
        if column in countries:
            selected = countries.loc[
                countries[column].astype(str).str.casefold().eq("mozambique")
            ]
            if not selected.empty:
                return selected.geometry.unary_union
    raise ValueError(f"could not identify Mozambique in {path}")


def haversine_m(
    longitude: float,
    latitude: float,
    node_longitude_rad: np.ndarray,
    node_latitude_rad: np.ndarray,
) -> np.ndarray:
    longitude_rad = np.radians(longitude)
    latitude_rad = np.radians(latitude)
    delta_longitude = node_longitude_rad - longitude_rad
    delta_latitude = node_latitude_rad - latitude_rad
    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_rad)
        * np.cos(node_latitude_rad)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def generate(args: argparse.Namespace) -> dict[str, object]:
    # MATLAB is imported only inside the Sherlock worker so repository tests
    # and summary jobs do not require the proprietary Python engine.
    import matlab.engine
    import pandas as pd

    points = pd.read_csv(args.track_points)
    points = points.loc[points["track_id"].eq(args.track_id)].copy()
    if points.empty:
        raise ValueError(f"historical track {args.track_id} is absent")
    with args.network.open("rb") as handle:
        graph = pickle.load(handle)
    node_ids = list(graph.nodes)
    nodes = generate_shp_some_nodes(graph, node_ids).to_crs(epsg=4326)
    node_points = nodes.geometry.apply(
        lambda geometry: geometry if geometry.geom_type == "Point" else geometry.centroid
    )
    node_longitude_rad = np.radians(node_points.x.to_numpy(dtype=float))
    node_latitude_rad = np.radians(node_points.y.to_numpy(dtype=float))
    maximum_wind = np.zeros(len(nodes), dtype=float)
    country = load_country(args.country_layer)

    engine = matlab.engine.start_matlab()
    try:
        engine.cd(str(args.chavas_directory), nargout=0)
        centers_used = 0
        for row in points.itertuples(index=False):
            longitude = float(row.longitude)
            latitude = float(row.latitude)
            if not country.covers(Point(longitude, latitude)):
                continue
            vmax = float(row.wind_knots) * KNOT_TO_MS
            rmax = float(row.rmw_nm) * NAUTICAL_MILE_TO_M
            if not np.isfinite(vmax) or not np.isfinite(rmax) or vmax <= 0 or rmax <= 0:
                continue
            radial = engine.ER11E04_nondim_rmaxinput(
                vmax,
                rmax,
                5e-5,
                0,
                1.5e-3,
                2 / 1000,
                0,
                1.0,
                0,
                0.15,
                nargout=11,
            )
            radii = np.asarray(radial[0], dtype=float).reshape(-1)
            wind = np.asarray(radial[1], dtype=float).reshape(-1)
            distance = haversine_m(
                longitude,
                latitude,
                node_longitude_rad,
                node_latitude_rad,
            )
            current = np.interp(distance, radii, wind, left=wind[0], right=0.0)
            maximum_wind = np.maximum(maximum_wind, current)
            centers_used += 1
    finally:
        engine.quit()

    threshold_knots = args.threshold * 3600.0 / 1850.0
    wind_knots = maximum_wind * 3600.0 / 1850.0
    normalized = np.clip(
        (wind_knots - threshold_knots) / (110.0 - threshold_knots), 0.0, None
    )
    tower_failure = normalized**3 / (1.0 + normalized**3)
    clipped = np.clip(maximum_wind, args.threshold, args.threshold + 30.0)
    line_failure = np.exp(
        0.6931 * (clipped - args.threshold) / 30.0
    ) - 1.0
    line_failure[maximum_wind <= args.threshold] = 0.0

    output = nodes.copy()
    output["wind_intensity"] = maximum_wind
    output["tower_frac_prop_lost"] = tower_failure
    output["line_frac_prop_lost"] = line_failure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(args.output, driver="ESRI Shapefile")
    report = {
        "schema_version": 1,
        "track_id": args.track_id,
        "event_id": str(points["event_id"].iloc[0]),
        "centers": len(points),
        "centers_inside_mozambique": centers_used,
        "nodes": len(output),
        "threshold_ms": args.threshold,
        "maximum_node_wind_ms": float(maximum_wind.max(initial=0.0)),
        "nodes_with_positive_line_failure": int(np.count_nonzero(line_failure > 0)),
        "output": str(args.output),
    }
    report_path = args.output.parent / f"track_{args.track_id:02d}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-points", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--country-layer", type=Path, required=True)
    parser.add_argument("--chavas-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    report = generate(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
