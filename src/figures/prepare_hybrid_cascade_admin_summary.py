#!/usr/bin/env python3
"""Aggregate hybrid cyclone disruption metrics by crossed province for Figure 5."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from capacity_ablation_figure import (
    attach_hospital_geometry,
    load_hospital_data,
    load_power_summary,
)
from prepare_cascade_admin_summary import load_admin_boundaries, wind_category
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS


AGGREGATE_METRICS = (
    "mean_initial_failure_length_km",
    "mean_cascade_failure_length_km",
    "mean_hospital_failure_probability",
)
CATEGORIES = (1, 2)
LENGTH_CONVENTIONS = ("legacy_compatible", "hybrid_partition")


def apply_length_convention(
    tracks: pd.DataFrame, convention: str
) -> pd.DataFrame:
    """Map hybrid outputs to the line-failure quantities shown in Figure 5."""

    tracks = tracks.copy()
    if convention == "legacy_compatible":
        required = {
            "mean_removed_failure_length_km",
            "mean_isolated_zero_service_length_km",
        }
        if not required.issubset(tracks):
            raise ValueError(
                "hybrid track summary is missing "
                f"{sorted(required - set(tracks))} for legacy-compatible lengths"
            )
        tracks["mean_initial_failure_length_km"] = tracks[
            "mean_removed_failure_length_km"
        ]
        tracks["mean_cascade_failure_length_km"] = tracks[
            "mean_isolated_zero_service_length_km"
        ]
    elif convention != "hybrid_partition":
        raise ValueError(
            f"length convention must be one of {LENGTH_CONVENTIONS}, "
            f"not {convention!r}"
        )

    initial = tracks["mean_initial_failure_length_km"].to_numpy(float)
    cascade = tracks["mean_cascade_failure_length_km"].to_numpy(float)
    tracks["cascade_initial_ratio"] = np.divide(
        cascade,
        initial,
        out=np.zeros_like(cascade),
        where=initial > 0.0,
    )
    return tracks


def prepare(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    admin = load_admin_boundaries(args.admin_boundaries)
    country = admin.geometry.unary_union
    tracks = pd.read_csv(args.hybrid_track_summary)
    required = {
        "storm_set",
        "track_id",
        "mean_initial_failure_length_km",
        "mean_cascade_failure_length_km",
        "mean_hospital_failure_probability",
    }
    if not required.issubset(tracks):
        raise ValueError(
            f"hybrid track summary is missing {sorted(required - set(tracks))}"
        )
    if tracks.duplicated(["storm_set", "track_id"]).any():
        raise ValueError("hybrid track summary contains duplicate track keys")
    tracks = apply_length_convention(tracks, args.length_convention)
    lookup = tracks.set_index(["storm_set", "track_id"])

    values = {
        province: {category: defaultdict(list) for category in CATEGORIES}
        for province in admin["ADM1_NAME"]
    }
    assigned = {category: 0 for category in CATEGORIES}
    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        events = gpd.read_file(args.storm_root / "Moz_cyclone_track" / folder)
        if len(events) != count:
            raise ValueError(f"{folder} has {len(events)} events; expected {count}")
        events = events.to_crs(epsg=4326).reset_index(drop=True)
        speed_column = "MAX_Speed" if "MAX_Speed" in events else "MaxSpeed"
        for track_id, event in events.iterrows():
            geometry = event.geometry.intersection(country)
            if geometry.is_empty:
                continue
            category = wind_category(float(event[speed_column]))
            if category not in CATEGORIES:
                continue
            crossed = admin.loc[admin.geometry.intersects(geometry), "ADM1_NAME"]
            if crossed.empty:
                continue
            row = lookup.loc[(storm_set, track_id)]
            assigned[category] += 1
            for province in crossed:
                for metric in AGGREGATE_METRICS:
                    values[province][category][metric].append(float(row[metric]))
                values[province][category]["cascade_initial_ratio"].append(
                    float(row["cascade_initial_ratio"])
                )

    summary = admin.copy()
    for category in CATEGORIES:
        for metric in AGGREGATE_METRICS:
            summary[f"{category}_{metric}"] = [
                float(np.mean(values[province][category][metric]))
                if values[province][category][metric]
                else 0.0
                for province in summary["ADM1_NAME"]
            ]
        if args.length_convention == "legacy_compatible":
            summary[f"{category}_cascade_initial_ratio"] = [
                float(np.mean(values[province][category]["cascade_initial_ratio"]))
                if values[province][category]["cascade_initial_ratio"]
                else 0.0
                for province in summary["ADM1_NAME"]
            ]
        else:
            initial = summary[
                f"{category}_mean_initial_failure_length_km"
            ].to_numpy()
            cascade = summary[
                f"{category}_mean_cascade_failure_length_km"
            ].to_numpy()
            summary[f"{category}_cascade_initial_ratio"] = np.divide(
                cascade,
                initial,
                out=np.zeros_like(cascade, dtype=float),
                where=initial > 0.0,
            )
        summary[f"{category}_event_count"] = [
            len(values[province][category][AGGREGATE_METRICS[0]])
            for province in summary["ADM1_NAME"]
        ]

    hospitals = load_hospital_data(args.hospital_data)
    public = load_power_summary(
        args.public_summary, require_complete=True, allow_unavailable=True
    )
    public = attach_hospital_geometry(public, hospitals)
    summary["num_hospital"] = [
        int(public.geometry.intersects(geometry).sum())
        for geometry in summary.geometry
    ]
    metadata = {
        "schema_version": 4,
        "model": "final hybrid connected-component/DC-service model",
        "length_convention": args.length_convention,
        "track_summary": str(args.hybrid_track_summary),
        "tracks": len(tracks),
        "assigned_events": assigned,
        "categories": list(CATEGORIES),
        "metric_note": (
            "Legacy-compatible convention: initial failure is directly removed "
            "transmission length; cascade failure is surviving transmission length "
            "in completely unpowered components. Partially served component length "
            "is excluded, matching the previous cascade.ipynb workflow. Each track "
            "ratio is cascade divided by initial (zero when initial is zero), and "
            "province values average those track ratios over crossing events."
            if args.length_convention == "legacy_compatible"
            else
            "Hybrid-partition convention: initial failure is directly removed "
            "transmission length plus surviving transmission length in completely "
            "unpowered components; cascade failure is the length-weighted unserved "
            "fraction of lines in partially served components. Each province ratio "
            "is its displayed mean cascade divided by displayed mean initial."
        ) + " All simplified-edge lengths are sums over unsimplified reference paths.",
    }
    return summary, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-boundaries", type=Path, required=True)
    parser.add_argument("--storm-root", type=Path, required=True)
    parser.add_argument("--hybrid-track-summary", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument(
        "--length-convention",
        choices=LENGTH_CONVENTIONS,
        default="legacy_compatible",
        help=(
            "Figure 5 line-failure definition. legacy_compatible reproduces the "
            "previous cascade.ipynb semantics using hybrid model outputs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, metadata = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_file(args.output, driver="GPKG")
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary[["ADM1_NAME", "num_hospital", "1_event_count", "2_event_count"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
