#!/usr/bin/env python3
"""Prepare the province-level inputs used by manuscript Figure 5.

This is a reproducible form of the aggregation in ``moz_notebook/cascade.ipynb``
cells 12, 15, and 19.  It preserves the notebook's definitions: a cyclone is
assigned to every province crossed by its Mozambique track, its category is
based on peak wind over the in-country track, and country-level cascade metrics
are averaged over the events assigned to each province.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np

from capacity_ablation_figure import (
    attach_hospital_geometry,
    load_hospital_data,
    load_power_summary,
)
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS


METRICS = (
    "threshold",
    "avg_hospital",
    "num_hopsital",
    "failed_hopsital",
    "fail_transmission",
    "initial_fail",
)
CATEGORIES = (1, 2)

def wind_category(maximum_wind: float) -> int | None:
    """Return the Saffir-Simpson-like bin used by the source notebook."""

    if 18 <= maximum_wind < 33:
        return 0
    if 33 <= maximum_wind < 43:
        return 1
    if 43 <= maximum_wind < 50:
        return 2
    if 50 <= maximum_wind < 58:
        return 3
    if 58 <= maximum_wind < 70:
        return 4
    if maximum_wind >= 70:
        return 5
    return None


def index_failure_records(root: Path, scenario: str) -> dict[tuple[int, int], Path]:
    expression = re.compile(
        rf"^25_STORM_DATA_IBTRACS_SI_1000_YEARS_(\d+)_+"
        rf"{re.escape(scenario)}_(\d+)\.pickle$"
    )
    records: dict[tuple[int, int], Path] = {}
    for path in root.glob(f"*{scenario}*.pickle"):
        match = expression.fullmatch(path.name)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in records:
            raise ValueError(f"Duplicate failure record for {key}: {path}")
        records[key] = path

    expected = {
        (storm_set, track_id)
        for storm_set, count in enumerate(TRACK_COUNTS)
        for track_id in range(count)
    }
    missing = sorted(expected.difference(records))
    extra = sorted(set(records).difference(expected))
    if missing or extra:
        raise ValueError(
            f"Failure catalogue mismatch for {scenario}: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return records


def load_admin_boundaries(path: Path) -> gpd.GeoDataFrame:
    admin = gpd.read_file(path)
    if admin.crs is None:
        raise ValueError(f"Administrative boundaries have no CRS: {path}")
    admin = admin.to_crs(epsg=4326)
    required = {"ADM0_NAME", "ADM1_NAME", "geometry"}
    missing = required.difference(admin.columns)
    if missing:
        raise ValueError(f"Administrative boundaries lack {sorted(missing)}")
    admin = admin.loc[
        admin["ADM0_NAME"].astype(str).str.casefold().eq("mozambique")
    ].copy()
    if admin.empty:
        raise ValueError("Administrative layer contains no Mozambique provinces")
    if admin["ADM1_NAME"].duplicated().any():
        raise ValueError("Mozambique ADM1 names are not unique")
    return admin


def load_event_catalogue(storm_root: Path, storm_set: int) -> gpd.GeoDataFrame:
    folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
    catalogue = gpd.read_file(storm_root / "Moz_cyclone_track" / folder)
    required_catalogue = {"Year", "Month", "TCID", "MAX_Speed", "geometry"}
    if not required_catalogue.issubset(catalogue.columns):
        raise ValueError(
            f"{folder} catalogue lacks {sorted(required_catalogue - set(catalogue.columns))}"
        )
    if len(catalogue) != TRACK_COUNTS[storm_set]:
        raise ValueError(
            f"{folder} contains {len(catalogue)} events; "
            f"expected {TRACK_COUNTS[storm_set]}"
        )
    if catalogue.crs is None:
        raise ValueError(f"{folder} catalogue has no CRS")
    return catalogue.to_crs(epsg=4326).reset_index(drop=True)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def prepare_summary(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict]:
    admin = load_admin_boundaries(args.admin_boundaries)
    country_geometry = admin.geometry.unary_union
    failure_records = index_failure_records(args.failure_root, args.scenario)

    # province -> category -> metric -> values
    values = {
        province: {
            category: defaultdict(list)
            for category in CATEGORIES
        }
        for province in admin["ADM1_NAME"]
    }
    assigned_events = {category: 0 for category in CATEGORIES}
    province_assignments = {category: 0 for category in CATEGORIES}

    for storm_set in range(len(TRACK_COUNTS)):
        catalogue = load_event_catalogue(args.storm_root, storm_set)
        for track_id, event in catalogue.iterrows():
            geometry = event.geometry.intersection(country_geometry)
            if geometry.is_empty:
                continue
            category = wind_category(float(event["MAX_Speed"]))
            if category not in CATEGORIES:
                continue

            with failure_records[(storm_set, track_id)].open("rb") as handle:
                record = pickle.load(handle)
            missing_metrics = set(METRICS).difference(record)
            if missing_metrics:
                raise ValueError(
                    f"{failure_records[(storm_set, track_id)]} lacks "
                    f"{sorted(missing_metrics)}"
                )
            initial = float(record["initial_fail"])
            ratio = float(record["fail_transmission"]) / initial - 1.0 if initial else 0.0
            record_values = {metric: float(record[metric]) for metric in METRICS}
            record_values["ratio"] = ratio

            crossed = admin.loc[admin.geometry.intersects(geometry), "ADM1_NAME"]
            if crossed.empty:
                continue
            assigned_events[category] += 1
            province_assignments[category] += len(crossed)
            for province in crossed:
                for metric, value in record_values.items():
                    values[province][category][metric].append(value)

    summary = admin.copy()
    for category in CATEGORIES:
        for metric in (*METRICS, "ratio"):
            column = f"{category}_{metric}"
            summary[column] = [
                mean_or_zero(values[province][category][metric])
                for province in summary["ADM1_NAME"]
            ]
        summary[f"{category}_failed_hopsital"] /= 720.0
        summary[f"{category}_event_count"] = [
            len(values[province][category]["threshold"])
            for province in summary["ADM1_NAME"]
        ]

    hospitals = load_hospital_data(args.hospital_data)
    public = load_power_summary(args.public_summary, require_complete=True)
    public = attach_hospital_geometry(public, hospitals)
    summary["num_hospital"] = [
        int(public.geometry.intersects(geometry).sum())
        for geometry in summary.geometry
    ]

    metadata = {
        "schema_version": 1,
        "scenario": args.scenario,
        "failure_records": len(failure_records),
        "province_count": len(summary),
        "assigned_events": assigned_events,
        "province_assignments": province_assignments,
        "category_definition_ms": {
            "1": "33 <= maximum in-country wind < 43",
            "2": "43 <= maximum in-country wind < 50",
        },
        "source_paths": {
            "admin_boundaries": str(args.admin_boundaries),
            "storm_root": str(args.storm_root),
            "failure_root": str(args.failure_root),
            "hospital_data": str(args.hospital_data),
            "public_summary": str(args.public_summary),
        },
    }
    return summary, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-boundaries", type=Path, required=True)
    parser.add_argument("--storm-root", type=Path, required=True)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="capacity_v2_public")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, metadata = prepare_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_file(args.output, driver="GPKG")
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary[["ADM1_NAME", "num_hospital", "1_event_count", "2_event_count"]].to_string(index=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
