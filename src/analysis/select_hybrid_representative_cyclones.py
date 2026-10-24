"""Select five spatially distinct, long-reach hybrid cyclone cases.

Candidate cases are the 290 tracks in synthetic catalogue 0, for which the
original wind-footprint shapefiles are available.  For each case, the script
measures hybrid hospital failures outside the >18 m/s footprint.  Candidates
are clustered into five geographic landing areas, and the case with the
largest cascade-extent score is retained in each cluster.

The cascade-extent score is the expected number of failed hospitals outside
the wind footprint multiplied by the probability-weighted 90th-percentile
distance of those hospitals from the footprint.  It therefore rewards both
failure magnitude and propagation distance without selecting a case because
of a single distant, low-probability hospital.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from capacity_ablation_figure import (
    load_boundary,
    load_hospital_data,
    load_power_summary,
)
from synthetic_tracks import TRACK_COUNTS


PROJECTED_CRS = "EPSG:32736"
WIND_THRESHOLD_MS = 18.0
OUTSIDE_TOLERANCE_KM = 0.1


def _wind_path(wind_root: Path, track_id: int) -> Path:
    path = wind_root / f"circle_track_{track_id}"
    if not path.exists():
        raise FileNotFoundError(f"Missing catalogue-0 wind footprint: {path}")
    return path


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(sorted_values[index])


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
        raise ValueError("Some hybrid-result hospitals have no SARA geometry")
    return gpd.GeoDataFrame(attached, geometry="geometry", crs=hospitals.crs)


def _track_metrics(
    track_id: int,
    wind_root: Path,
    hospital_results: gpd.GeoDataFrame,
    hospital_projected: gpd.GeoDataFrame,
    mozambique_projected,
) -> dict[str, object] | None:
    track_key = f"0_{track_id}"
    wind_path = _wind_path(wind_root, track_id)
    wind = gpd.read_file(wind_path)
    if wind.crs is None or "speed" not in wind:
        raise ValueError(f"Invalid wind footprint: {wind_path}")
    wind["speed"] = pd.to_numeric(wind["speed"], errors="coerce")
    if wind["speed"].isna().any():
        raise ValueError(f"Nonnumeric wind speed in {wind_path}")
    wind = wind.loc[wind["speed"].gt(WIND_THRESHOLD_MS)]
    if wind.empty:
        return None
    footprint = wind.to_crs(PROJECTED_CRS).geometry.unary_union
    land_footprint = footprint.intersection(mozambique_projected)
    if land_footprint.is_empty:
        return None

    probabilities = pd.to_numeric(
        hospital_results[track_key], errors="coerce"
    ).to_numpy(dtype=float)
    evaluated = probabilities >= 0.0
    positive = probabilities > 0.0
    if not np.any(positive):
        return {
            "track_key": track_key,
            "track_id": track_id,
            "wind_path": str(wind_path),
            "landing_easting_m": float(land_footprint.centroid.x),
            "landing_northing_m": float(land_footprint.centroid.y),
            "evaluated_hospitals": int(evaluated.sum()),
            "positive_failure_hospitals": 0,
            "expected_failed_hospitals": 0.0,
            "outside_failure_hospitals": 0,
            "outside_expected_failed_hospitals": 0.0,
            "outside_failure_share": 0.0,
            "outside_weighted_p90_distance_km": 0.0,
            "outside_maximum_distance_km": 0.0,
            "cascade_extent_score": 0.0,
        }

    positive_points = hospital_projected.geometry.loc[positive]
    distances_km = positive_points.distance(footprint).to_numpy(dtype=float) / 1000.0
    positive_probabilities = probabilities[positive]
    outside = distances_km > OUTSIDE_TOLERANCE_KM
    outside_probabilities = positive_probabilities[outside]
    outside_distances = distances_km[outside]
    outside_expected = float(outside_probabilities.sum())
    p90_distance = _weighted_quantile(
        outside_distances,
        outside_probabilities,
        0.90,
    )
    expected_failures = float(positive_probabilities.sum())
    return {
        "track_key": track_key,
        "track_id": track_id,
        "wind_path": str(wind_path),
        "landing_easting_m": float(land_footprint.centroid.x),
        "landing_northing_m": float(land_footprint.centroid.y),
        "evaluated_hospitals": int(evaluated.sum()),
        "positive_failure_hospitals": int(positive.sum()),
        "expected_failed_hospitals": expected_failures,
        "outside_failure_hospitals": int(outside.sum()),
        "outside_expected_failed_hospitals": outside_expected,
        "outside_failure_share": (
            outside_expected / expected_failures if expected_failures > 0.0 else 0.0
        ),
        "outside_weighted_p90_distance_km": p90_distance,
        "outside_maximum_distance_km": (
            float(outside_distances.max()) if len(outside_distances) else 0.0
        ),
        "cascade_extent_score": outside_expected * p90_distance,
    }


def _landing_coordinates(metrics: pd.DataFrame) -> pd.DataFrame:
    points = gpd.GeoSeries(
        gpd.points_from_xy(
            metrics["landing_easting_m"],
            metrics["landing_northing_m"],
        ),
        crs=PROJECTED_CRS,
    ).to_crs(epsg=4326)
    result = metrics.copy()
    result["landing_longitude"] = points.x.to_numpy()
    result["landing_latitude"] = points.y.to_numpy()
    return result


def select_cases(metrics: pd.DataFrame, clusters: int = 5) -> pd.DataFrame:
    eligible = metrics.loc[
        metrics["outside_expected_failed_hospitals"].gt(0.0)
        & metrics["outside_weighted_p90_distance_km"].gt(0.0)
    ].copy()
    if len(eligible) < clusters:
        raise ValueError(
            f"Only {len(eligible)} cases have failures outside the wind footprint; "
            f"cannot select {clusters}"
        )

    coordinates = eligible.loc[
        :, ["landing_easting_m", "landing_northing_m"]
    ].to_numpy(dtype=float)
    model = KMeans(n_clusters=clusters, random_state=1234, n_init=20)
    eligible["landing_cluster"] = model.fit_predict(coordinates)
    selected = (
        eligible.sort_values(
            [
                "landing_cluster",
                "cascade_extent_score",
                "outside_expected_failed_hospitals",
                "outside_weighted_p90_distance_km",
            ],
            ascending=[True, False, False, False],
        )
        .groupby("landing_cluster", as_index=False)
        .first()
    )
    selected = selected.sort_values(
        ["landing_latitude", "landing_longitude"],
        ascending=[False, True],
    ).reset_index(drop=True)
    selected["panel"] = [chr(ord("a") + index) for index in range(len(selected))]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--wind-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-wind-root", type=Path, required=True)
    arguments = parser.parse_args()

    summary = load_power_summary(
        arguments.public_summary,
        require_complete=True,
        allow_unavailable=True,
    )
    hospitals = load_hospital_data(arguments.hospital_data).to_crs(epsg=4326)
    hospital_results = _attach_geometry(summary, hospitals)
    hospital_projected = hospital_results.to_crs(PROJECTED_CRS)
    boundary = load_boundary(arguments.boundary).to_crs(PROJECTED_CRS)
    mozambique_projected = boundary.geometry.unary_union

    records = []
    for track_id in range(TRACK_COUNTS[0]):
        record = _track_metrics(
            track_id,
            arguments.wind_root,
            hospital_results,
            hospital_projected,
            mozambique_projected,
        )
        if record is not None:
            records.append(record)
    metrics = _landing_coordinates(pd.DataFrame.from_records(records))
    selected = select_cases(metrics, clusters=5)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.selected_wind_root.mkdir(parents=True, exist_ok=True)
    for row in selected.itertuples(index=False):
        source = _wind_path(arguments.wind_root, int(row.track_id))
        destination = arguments.selected_wind_root / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True)

    candidate_path = arguments.output_dir / "representative_case_candidates.csv"
    selection_path = arguments.output_dir / "representative_case_selection.csv"
    track_path = arguments.output_dir / "representative_track_keys.txt"
    metadata_path = arguments.output_dir / "representative_case_selection.json"
    metrics.sort_values("cascade_extent_score", ascending=False).to_csv(
        candidate_path, index=False
    )
    selected.to_csv(selection_path, index=False)
    track_keys = selected["track_key"].tolist()
    track_path.write_text(",".join(track_keys) + "\n")
    metadata = {
        "schema_version": 1,
        "candidate_catalogue": 0,
        "candidate_track_count": TRACK_COUNTS[0],
        "wind_threshold_ms": WIND_THRESHOLD_MS,
        "outside_tolerance_km": OUTSIDE_TOLERANCE_KM,
        "geographic_selection": (
            "K-means clustering of >18 m/s Mozambique-footprint centroids into "
            "five landing-area groups"
        ),
        "selection_within_group": (
            "maximum expected outside failures multiplied by their "
            "probability-weighted 90th-percentile distance from the footprint"
        ),
        "selected_track_keys": track_keys,
        "selected_cases": selected.to_dict(orient="records"),
        "outputs": {
            "candidate_csv": str(candidate_path),
            "selection_csv": str(selection_path),
            "track_keys": str(track_path),
            "selected_wind_root": str(arguments.selected_wind_root),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
