#!/usr/bin/env python3
"""Prepare current-result inputs for manuscript Figures 3 and 4.

Direct wind damage comes from the ten ``synthetic_new_hospital`` layers.
Indirect electricity, water, and communication probabilities come from the
current non-ablation public-grid workflow.  Negative downstream sentinels are
treated as unavailable values, never as failures or zeros.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from capacity_ablation_figure import (
    attach_hospital_geometry,
    load_hospital_data,
    load_power_summary,
)
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS, expected_track_columns


def facility_key(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\.0+$", "", text)


def validate_probabilities(values: pd.DataFrame, label: str, allow_missing: bool) -> None:
    array = values.to_numpy(dtype=float)
    finite = np.isfinite(array)
    if not allow_missing and not finite.all():
        row, column = np.argwhere(~finite)[0]
        raise ValueError(
            f"{label} contains a missing value at row {row}, "
            f"track {values.columns[column]}"
        )
    observed = array[finite]
    if observed.size and (observed.min() < 0.0 or observed.max() > 1.0):
        raise ValueError(
            f"{label} probabilities fall outside [0, 1]: "
            f"[{observed.min()}, {observed.max()}]"
        )


def load_direct_matrix(root: Path, hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
    identity = hospitals[["node_id", "facility_i"]].copy()
    identity["facility_key"] = identity["facility_i"].map(facility_key)
    if identity["facility_key"].duplicated().any():
        raise ValueError("SARA facility_i values are not unique")

    pieces: list[pd.DataFrame] = []
    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        path = root / f"track_{folder}"
        direct = gpd.read_file(path)
        if "facility_i" not in direct:
            raise ValueError(f"Direct layer lacks facility_i: {path}")
        direct = direct.copy()
        stored_node_id = None
        if "node_id" in direct:
            stored_node_id = direct.pop("node_id").astype(str).str.strip()
        direct["facility_key"] = direct["facility_i"].map(facility_key)
        if direct["facility_key"].duplicated().any():
            raise ValueError(f"Direct layer contains duplicate facilities: {path}")
        direct = direct.merge(
            identity[["node_id", "facility_key"]],
            on="facility_key",
            how="left",
            validate="one_to_one",
        )
        if direct["node_id"].isna().any() or len(direct) != len(hospitals):
            raise ValueError(f"Direct layer does not match the SARA cohort: {path}")
        if stored_node_id is not None:
            stored_by_facility = pd.Series(
                stored_node_id.to_numpy(),
                index=direct["facility_key"],
            )
            expected_by_facility = direct.set_index("facility_key")["node_id"].astype(str)
            if not stored_by_facility.reindex(expected_by_facility.index).equals(
                expected_by_facility
            ):
                raise ValueError(f"Stored direct node IDs do not match SARA: {path}")

        source_columns = [f"{storm_set}_{track_id}_b" for track_id in range(count)]
        missing = [column for column in source_columns if column not in direct]
        if missing:
            raise ValueError(
                f"Direct layer is missing {len(missing)} tracks, including "
                f"{missing[:8]}: {path}"
            )
        values = direct[source_columns].apply(pd.to_numeric, errors="coerce")
        values.columns = [column[:-2] for column in source_columns]
        values.index = direct["node_id"].astype(str)
        # The legacy direct-wind archive contains genuinely unevaluated
        # hospital/event cells. Preserve them as missing; event comparisons
        # below use only hospitals evaluated by both models.
        validate_probabilities(values, str(path), allow_missing=True)
        pieces.append(values)

    matrix = pd.concat(pieces, axis=1)
    expected = list(expected_track_columns())
    if list(matrix.columns) != expected:
        raise ValueError("Direct track columns are not in the canonical 2,703-event order")
    return matrix


def load_downstream_matrix(root: Path, system: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    cohort: set[str] | None = None
    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        path = root / f"{system}_cyclone_all_{folder}_25.csv"
        frame = pd.read_csv(path, dtype={"id": "string"})
        if "id" not in frame or frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"Invalid downstream identity column: {path}")
        columns = [f"{storm_set}_{track_id}" for track_id in range(count)]
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} tracks, including {missing[:8]}"
            )
        current_cohort = set(frame["id"].astype(str))
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise ValueError(f"The {system} hospital cohort changes between storm sets")
        values = frame.set_index(frame["id"].astype(str))[columns]
        values = values.apply(pd.to_numeric, errors="coerce").where(lambda item: item >= 0)
        validate_probabilities(values, str(path), allow_missing=True)
        pieces.append(values)

    matrix = pd.concat(pieces, axis=1)
    if list(matrix.columns) != list(expected_track_columns()):
        raise ValueError(f"{system} columns are not in canonical event order")
    return matrix


def attach_risk(
    matrix: pd.DataFrame,
    hospitals: gpd.GeoDataFrame,
    value_column: str,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    risk = matrix.mean(axis=1, skipna=True).where(matrix.notna().any(axis=1))
    frame = pd.DataFrame({"node_id": matrix.index.astype(str), value_column: risk.to_numpy()})
    geometry = hospitals[["node_id", "facility_i", "geometry"]].copy()
    geometry["node_id"] = geometry["node_id"].astype(str)
    merged = frame.merge(geometry, on="node_id", how="left", validate="one_to_one")
    unmatched = merged.loc[merged["geometry"].isna(), "node_id"].astype(str).tolist()
    merged = merged.loc[merged["geometry"].notna()].copy()
    if merged.empty:
        raise ValueError(f"No {value_column} rows match SARA hospitals")
    merged[value_column] = merged[value_column].fillna(-1.0)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=hospitals.crs), unmatched


def load_intensities(storm_root: Path) -> pd.Series:
    values: dict[str, float] = {}
    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        catalogue = gpd.read_file(storm_root / "Moz_cyclone_track" / folder)
        if len(catalogue) != count:
            raise ValueError(f"Unexpected catalogue length for {folder}: {len(catalogue)}")
        column = next(
            (name for name in ("MAX_Speed", "MaxSpeed", "max_speed") if name in catalogue),
            None,
        )
        if column is None:
            raise ValueError(f"Cyclone catalogue lacks maximum wind: {folder}")
        for track_id, value in enumerate(catalogue[column]):
            values[f"{storm_set}_{track_id}"] = float(value)
    return pd.Series(values).reindex(expected_track_columns())


def cyclone_summary(
    indirect: pd.DataFrame,
    direct: pd.DataFrame,
    intensities: pd.Series,
) -> pd.DataFrame:
    matched_direct = direct.reindex(indirect.index)
    if not matched_direct.notna().to_numpy().any():
        raise ValueError("Direct and indirect hospital identities do not overlap")
    records = []
    for event in expected_track_columns():
        evaluated = indirect[event].notna() & matched_direct[event].notna()
        evaluated_count = int(evaluated.sum())
        records.append({
            "cyclone": event,
            "wind": (
                float(matched_direct.loc[evaluated, event].sum())
                if evaluated_count else np.nan
            ),
            "indirect": (
                float(indirect.loc[evaluated, event].sum())
                if evaluated_count else np.nan
            ),
            "cyclone_intensity": float(intensities[event]),
            "evaluated_hospitals": evaluated_count,
        })
    return pd.DataFrame.from_records(records)


def prepare(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hospitals = load_hospital_data(args.hospital_data)
    direct = load_direct_matrix(args.direct_root, hospitals)
    public = load_power_summary(
        args.public_summary, require_complete=True, allow_unavailable=True
    )
    public_matrix = public.set_index(public["node_id"].astype(str))[
        list(expected_track_columns())
    ].where(lambda values: values >= 0.0)
    water = load_downstream_matrix(args.water_root, "water")
    communication = load_downstream_matrix(args.communication_root, "comm")
    intensities = load_intensities(args.storm_root)

    electricity_layer = attach_hospital_geometry(public, hospitals)
    electricity_layer["mean"] = public_matrix.where(
        public_matrix >= 0.0
    ).mean(axis=1).fillna(-1.0).to_numpy()
    electricity_layer = electricity_layer[[
        "node_id", "facility_i", "mean", "geometry"
    ]].copy()
    water_layer, unmatched_water = attach_risk(water, hospitals, "water_mean")
    communication_layer, unmatched_communication = attach_risk(
        communication, hospitals, "comm_mean"
    )
    direct_layer, unmatched_direct = attach_risk(direct, hospitals, "direct")

    layers = {
        "df_final_public_25.gpkg": electricity_layer,
        "water_hospital_mean.gpkg": water_layer,
        "communication_hospital_mean.gpkg": communication_layer,
        "direct_hospital_mean.gpkg": direct_layer,
    }
    for filename, layer in layers.items():
        layer.to_file(args.output_dir / filename, driver="GPKG")

    summaries = {
        "cyclone_summary_elec.csv": cyclone_summary(public_matrix, direct, intensities),
        "cyclone_summary_water.csv": cyclone_summary(water, direct, intensities),
        "cyclone_summary_comm.csv": cyclone_summary(communication, direct, intensities),
    }
    for filename, frame in summaries.items():
        frame.to_csv(args.output_dir / filename, index=False)

    metadata = {
        "schema_version": 1,
        "track_count": len(expected_track_columns()),
        "cohort_sizes": {
            "electricity": len(public_matrix),
            "water": len(water),
            "communication": len(communication),
            "direct": len(direct),
        },
        "unavailable_cells": {
            "water": int(water.isna().sum().sum()),
            "communication": int(communication.isna().sum().sum()),
            "direct": int(direct.isna().sum().sum()),
        },
        "entirely_unavailable_direct_events": direct.columns[
            direct.notna().sum(axis=0).eq(0)
        ].tolist(),
        "non_hospital_ids_excluded": {
            "water": unmatched_water,
            "communication": unmatched_communication,
            "direct": unmatched_direct,
        },
        "negative_sentinel_policy": "Values below zero are unavailable and excluded.",
        "summary_definition": (
            "For each system/event, direct and indirect probabilities are summed "
            "over the same hospitals with evaluated values."
        ),
        "source_paths": {
            "public_summary": str(args.public_summary),
            "water_root": str(args.water_root),
            "communication_root": str(args.communication_root),
            "direct_root": str(args.direct_root),
            "hospital_data": str(args.hospital_data),
            "storm_root": str(args.storm_root),
        },
    }
    (args.output_dir / "failure_figure_inputs.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--water-root", type=Path, required=True)
    parser.add_argument("--communication-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--hospital-data", type=Path, required=True)
    parser.add_argument("--storm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    metadata = prepare(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
