#!/usr/bin/env python3
"""Validate and aggregate hybrid electricity/water/communication track outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd

from dependency_models import DependencyModels
from hospital_grid_ablation import hospital_node_id
from hybrid_multisystem_model import MODEL_VERSION, POWER_SCENARIO
from synthetic_tracks import (
    STORM_FOLDER_TEMPLATE,
    TRACK_COUNTS,
    expected_track_columns,
)


def atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_models(path: Path) -> DependencyModels:
    with path.open("rb") as handle:
        models = pickle.load(handle)
    if not isinstance(models, DependencyModels):
        raise TypeError(f"unexpected dependency model in {path}")
    return models


def track_path(
    root: Path,
    service: str,
    folder: str,
    track_id: int,
    threshold: int,
) -> Path:
    return root / service / "tracks" / (
        f"{service}_cyclone_{folder}_{track_id}_{threshold}.csv"
    )


def read_service_track(
    path: Path,
    ids: tuple[str, ...],
    samples: int,
) -> pd.Series:
    frame = pd.read_csv(path, dtype={"id": "string"})
    required = {
        "id",
        "mean",
        "sample_num",
        "evaluated_sample_num",
        "power_scenario",
        "model_version",
    }
    if not required.issubset(frame):
        raise ValueError(f"invalid downstream schema: {path}")
    if tuple(frame["id"].astype(str)) != ids:
        raise ValueError(f"hospital identity mismatch: {path}")
    if not pd.to_numeric(frame["sample_num"], errors="coerce").eq(samples).all():
        raise ValueError(f"sample count mismatch: {path}")
    if not frame["power_scenario"].eq(POWER_SCENARIO).all():
        raise ValueError(f"power scenario mismatch: {path}")
    if not frame["model_version"].eq(MODEL_VERSION).all():
        raise ValueError(f"model version mismatch: {path}")
    values = pd.to_numeric(frame["mean"], errors="coerce")
    if values.isna().any() or not values.between(-1.0, 1.0).all():
        raise ValueError(f"invalid probabilities: {path}")
    return values.reset_index(drop=True)


def summarize_synthetic(args: argparse.Namespace) -> dict[str, object]:
    models = load_models(args.dependency_model)
    ids = {
        "water": models.water.hospital_ids,
        "comm": models.communication.hospital_ids,
    }
    aggregate_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        for service in ("water", "comm"):
            columns: dict[str, pd.Series] = {}
            for track_id in range(count):
                path = track_path(
                    args.result_root,
                    service,
                    folder,
                    track_id,
                    args.threshold,
                )
                columns[f"{storm_set}_{track_id}"] = read_service_track(
                    path, ids[service], args.samples
                )
            frame = pd.concat(
                [pd.Series(ids[service], name="id"), pd.DataFrame(columns)], axis=1
            )
            values = frame[list(columns)].where(frame[list(columns)] >= 0.0)
            mean_name = "water_mean" if service == "water" else "comm_mean"
            frame[mean_name] = values.mean(axis=1).fillna(-1.0)
            output = args.result_root / service / (
                f"{service}_cyclone_all_{folder}_{args.threshold}.csv"
            )
            atomic_dataframe(frame, output)
            aggregate_records.append(
                {
                    "service": service,
                    "storm_set": storm_set,
                    "tracks": count,
                    "rows": len(frame),
                    "output": str(output),
                }
            )

        for track_id in range(count):
            path = args.result_root / "summaries" / (
                f"summary_{folder}_{track_id}_{args.threshold}.json"
            )
            with path.open() as handle:
                summary = json.load(handle)
            if summary.get("model_version") != MODEL_VERSION:
                raise ValueError(f"summary version mismatch: {path}")
            if int(summary.get("samples", -1)) != args.samples:
                raise ValueError(f"summary sample mismatch: {path}")
            summary_rows.append(
                {
                    "storm_set": storm_set,
                    "storm_folder": folder,
                    "track_id": track_id,
                    **summary["result"],
                }
            )

    expected = list(expected_track_columns())
    observed = [
        f"{row['storm_set']}_{row['track_id']}" for row in summary_rows
    ]
    if observed != expected:
        raise ValueError("synthetic summary rows are not in canonical track order")
    summary_output = args.result_root / "hybrid_multisystem_track_summary.csv"
    atomic_dataframe(pd.DataFrame(summary_rows), summary_output)
    manifest = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "power_scenario": POWER_SCENARIO,
        "tracks": len(summary_rows),
        "samples_per_track": args.samples,
        "water_hospitals": len(ids["water"]),
        "communication_hospitals": len(ids["comm"]),
        "aggregates": aggregate_records,
        "track_summary": str(summary_output),
    }
    manifest_path = args.result_root / "hybrid_multisystem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def public_hospital_identity(path: Path) -> pd.DataFrame:
    hospitals = gpd.read_file(path)
    valid = hospitals.geometry.notna() & ~hospitals.geometry.is_empty
    public = hospitals.loc[hospitals["ele_main_s"].eq(1) & valid].copy()
    public["node_id"] = [hospital_node_id(index) for index in public.index]
    identity = public[["node_id", "facility_i"]].reset_index(drop=True)
    if len(identity) != 720 or identity["node_id"].duplicated().any():
        raise ValueError("expected 720 unique public-grid hospitals")
    return identity


def summarize_historical(args: argparse.Namespace) -> dict[str, object]:
    events = pd.read_csv(args.event_manifest)
    required = {"track_id", "event_id", "sid", "category", "max_wind_ms"}
    if not required.issubset(events):
        raise ValueError(
            f"historical manifest is missing {sorted(required - set(events))}"
        )
    events = events.sort_values("track_id").reset_index(drop=True)
    if not np.array_equal(events["track_id"], np.arange(len(events))):
        raise ValueError("historical track_id values must be contiguous from zero")

    identity = public_hospital_identity(args.hospital_data)
    canonical_ids: list[str] | None = None
    modeled: pd.Series | None = None
    columns: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    folder = args.historical_folder
    for row in events.itertuples(index=False):
        track_id = int(row.track_id)
        path = track_path(
            args.result_root,
            "electricity",
            folder,
            track_id,
            args.threshold,
        )
        frame = pd.read_csv(path, dtype={"hospital_id": "string"})
        required_output = {
            "hospital_id",
            "modeled",
            "failure_probability",
            "sample_num",
            "power_scenario",
            "model_version",
        }
        if not required_output.issubset(frame):
            raise ValueError(f"invalid historical electricity schema: {path}")
        ids = frame["hospital_id"].astype(str).tolist()
        current_modeled = frame["modeled"].astype(int).eq(1).reset_index(drop=True)
        if canonical_ids is None:
            canonical_ids = ids
            modeled = current_modeled
        elif ids != canonical_ids or not current_modeled.equals(modeled):
            raise ValueError(f"historical hospital cohort differs: {path}")
        values = pd.to_numeric(frame["failure_probability"], errors="coerce")
        if values.isna().any():
            raise ValueError(f"invalid historical probabilities: {path}")
        columns[f"historical_{track_id}"] = values.reset_index(drop=True)

        summary_path = args.result_root / "summaries" / (
            f"summary_{folder}_{track_id}_{args.threshold}.json"
        )
        with summary_path.open() as handle:
            summary = json.load(handle)
        summary_rows.append(
            {
                "track_id": track_id,
                "event_id": row.event_id,
                "sid": row.sid,
                "category": int(row.category),
                "max_wind_ms": float(row.max_wind_ms),
                **summary["result"],
            }
        )

    if canonical_ids is None or modeled is None:
        raise ValueError("historical event manifest is empty")
    identity_by_node = identity.set_index("node_id")
    ordered = identity_by_node.loc[canonical_ids].reset_index()
    wide = pd.concat([ordered, pd.DataFrame(columns)], axis=1)
    output = args.result_root / "df_historical_hybrid_25.csv"
    summary_output = args.result_root / "hybrid_historical_track_summary.csv"
    atomic_dataframe(wide, output)
    atomic_dataframe(pd.DataFrame(summary_rows), summary_output)
    manifest = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "events": len(events),
        "modeled_hospitals": int(modeled.sum()),
        "unmodeled_hospitals": int((~modeled).sum()),
        "electricity_summary": str(output),
        "track_summary": str(summary_output),
        "event_manifest": str(args.event_manifest),
    }
    manifest_path = args.result_root / "hybrid_historical_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--result-root", type=Path, required=True)
    synthetic.add_argument("--dependency-model", type=Path, required=True)
    synthetic.add_argument("--samples", type=int, default=100)
    synthetic.add_argument("--threshold", type=int, default=25)

    historical = subparsers.add_parser("historical")
    historical.add_argument("--result-root", type=Path, required=True)
    historical.add_argument("--event-manifest", type=Path, required=True)
    historical.add_argument("--hospital-data", type=Path, required=True)
    historical.add_argument("--historical-folder", default="IBTRACS_HISTORICAL")
    historical.add_argument("--samples", type=int, default=100)
    historical.add_argument("--threshold", type=int, default=25)
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = (
        summarize_synthetic(args)
        if args.command == "synthetic"
        else summarize_historical(args)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
