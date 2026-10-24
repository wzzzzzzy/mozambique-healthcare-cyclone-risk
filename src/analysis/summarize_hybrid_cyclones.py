#!/usr/bin/env python3
"""Validate and aggregate all hybrid cyclone simulations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from hospital_grid_ablation import hospital_node_id
from synthetic_tracks import expected_track_columns, expected_track_keys


def result_paths(root: Path, storm_folder: str, track_id: int) -> dict[str, Path]:
    storm_set = int(storm_folder.rsplit("_", 1)[-1])
    directory = root / "results" / f"dataset_{storm_set}"
    prefix = directory / f"track_{track_id:04d}"
    return {
        "summary": Path(f"{prefix}_summary.json"),
        "samples": Path(f"{prefix}_samples.csv"),
        "hospitals": Path(f"{prefix}_hospitals.csv"),
    }


def validate_track(
    paths: dict[str, Path],
    storm_folder: str,
    track_id: int,
    samples: int,
    hospital_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {path}")
    with paths["summary"].open() as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != 1:
        raise ValueError(f"unexpected schema in {paths['summary']}")
    if summary.get("storm_folder") != storm_folder or int(summary.get("track_id", -1)) != track_id:
        raise ValueError(f"track identity mismatch in {paths['summary']}")
    if int(summary.get("samples", -1)) != samples:
        raise ValueError(f"sample count mismatch in {paths['summary']}")

    sample_frame = pd.read_csv(paths["samples"])
    if len(sample_frame) != samples or "sample_id" not in sample_frame:
        raise ValueError(f"invalid sample table: {paths['samples']}")
    hospitals = pd.read_csv(paths["hospitals"], dtype={"hospital_id": "string"})
    required = {"hospital_id", "modeled", "failure_probability"}
    if not required.issubset(hospitals):
        raise ValueError(f"invalid hospital table: {paths['hospitals']}")
    if len(hospitals) != hospital_count or hospitals["hospital_id"].duplicated().any():
        raise ValueError(
            f"hospital table is not a unique {hospital_count}-row cohort: "
            f"{paths['hospitals']}"
        )
    modeled = hospitals["modeled"].astype(int).eq(1)
    values = pd.to_numeric(hospitals["failure_probability"], errors="coerce")
    if values[modeled].isna().any() or ((values[modeled] < 0) | (values[modeled] > 1)).any():
        raise ValueError(f"modeled hospital probabilities are invalid: {paths['hospitals']}")
    if not values[~modeled].eq(-1).all():
        raise ValueError(f"unmodeled hospitals must be -1: {paths['hospitals']}")
    base = summary.get("base", {})
    if int(base.get("modeled_hospitals", -1)) != int(modeled.sum()):
        raise ValueError(f"modeled hospital count mismatch in {paths['summary']}")
    if int(base.get("unmodeled_hospitals", -1)) != int((~modeled).sum()):
        raise ValueError(f"unmodeled hospital count mismatch in {paths['summary']}")
    hospitals["failure_probability"] = values
    return summary, hospitals


def flatten_summary(summary: dict[str, object], storm_set: int) -> dict[str, object]:
    base = summary["base"]
    hazard = summary["hazard"]
    result = summary["result"]
    return {
        "schema_version": summary["schema_version"],
        "storm_set": storm_set,
        "storm_folder": summary["storm_folder"],
        "track_id": summary["track_id"],
        "samples": summary["samples"],
        "random_seed": summary["random_seed"],
        "modeled_hospitals": base["modeled_hospitals"],
        "unmodeled_hospitals": base["unmodeled_hospitals"],
        "load_mw": base["load_mw"],
        "generator_pmax_mw": base["generator_pmax_mw"],
        "edges_with_positive_probability": hazard["edges_with_positive_probability"],
        "expected_failed_edges": hazard["expected_failed_edges"],
        "maximum_failure_probability": hazard["maximum_failure_probability"],
        "mean_failed_edges": result["mean_failed_edges"],
        "probability_any_edge_failure": result["probability_any_edge_failure"],
        "mean_served_fraction": result["mean_served_fraction"],
        "mean_flow_coefficient": result["mean_flow_coefficient"],
        "mean_hospital_served_fraction": result["mean_hospital_served_fraction"],
        "mean_hospital_failure_probability": result["mean_hospital_failure_probability"],
        "median_hospital_failure_probability": result["median_hospital_failure_probability"],
        "p95_hospital_failure_probability": result["p95_hospital_failure_probability"],
        "probability_any_hospital_disruption": result["probability_any_hospital_disruption"],
        "samples_with_flow_constraint": result["samples_with_flow_constraint"],
    }


def atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def hospital_identity(path: Path, scenario: str) -> pd.DataFrame:
    hospitals = gpd.read_file(path)
    valid = hospitals.geometry.notna() & ~hospitals.geometry.is_empty
    public = hospitals["ele_main_s"].eq(1)
    solar = hospitals["ele_main_s"].eq(3) | hospitals["ele_sec_so"].eq(3)
    mask = public if scenario == "public" else (public | solar)
    selected = hospitals.loc[mask & valid].copy()
    selected["node_id"] = [hospital_node_id(index) for index in selected.index]
    identity = selected[["node_id", "facility_i"]].reset_index(drop=True)
    expected = 720 if scenario == "public" else 1380
    if len(identity) != expected or identity["node_id"].duplicated().any():
        raise ValueError(
            f"SARA {scenario} identity is not the expected {expected} hospitals"
        )
    return identity


def summarize(args: argparse.Namespace) -> dict[str, object]:
    expected = expected_track_keys()
    identity = hospital_identity(args.hospitals, args.hospital_scenario)
    canonical_ids: list[str] | None = None
    track_values: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    failures: list[str] = []
    modeled_mask: pd.Series | None = None

    for storm_folder, track_id in expected:
        storm_set = int(storm_folder.rsplit("_", 1)[-1])
        paths = result_paths(args.result_root, storm_folder, track_id)
        try:
            summary, hospitals = validate_track(
                paths, storm_folder, track_id, args.samples, len(identity)
            )
            ids = hospitals["hospital_id"].astype(str).tolist()
            if canonical_ids is None:
                canonical_ids = ids
                modeled_mask = hospitals["modeled"].astype(int).eq(1).reset_index(drop=True)
            elif ids != canonical_ids:
                raise ValueError("hospital row order differs from the first track")
            elif not hospitals["modeled"].astype(int).eq(1).reset_index(drop=True).equals(modeled_mask):
                raise ValueError("modeled hospital mask differs from the first track")
            column = f"{storm_set}_{track_id}"
            track_values[column] = hospitals["failure_probability"].reset_index(drop=True)
            summary_rows.append(flatten_summary(summary, storm_set))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"{storm_folder}:{track_id}: {exc}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} hybrid results are missing or invalid:\n"
            + "\n".join(failures[:100])
        )
    if canonical_ids is None or modeled_mask is None:
        raise RuntimeError("no hybrid results were found")
    if tuple(track_values) != expected_track_columns():
        raise ValueError("hybrid track columns are not in canonical order")

    identity_by_node = identity.set_index("node_id")
    missing_identity = sorted(set(canonical_ids) - set(identity_by_node.index))
    if missing_identity:
        raise ValueError(f"SARA identity is missing hospital nodes: {missing_identity[:5]}")
    ordered_identity = identity_by_node.loc[canonical_ids].reset_index()
    wide = pd.concat(
        [ordered_identity, pd.DataFrame(track_values)], axis=1
    )
    summary_frame = pd.DataFrame(summary_rows)
    atomic_dataframe(summary_frame, args.summary_output)
    atomic_dataframe(wide, args.hospital_output)

    report = {
        "schema_version": 1,
        "hospital_scenario": args.hospital_scenario,
        "tracks": len(summary_frame),
        "hospitals": len(wide),
        "modeled_hospitals": int(modeled_mask.sum()),
        "unmodeled_hospitals": int((~modeled_mask).sum()),
        "samples_per_track": args.samples,
        "track_summary": str(args.summary_output),
        "hospital_summary": str(args.hospital_output),
        "mean_track_hospital_failure_probability": float(
            summary_frame["mean_hospital_failure_probability"].mean()
        ),
        "mean_track_flow_coefficient": float(
            summary_frame["mean_flow_coefficient"].mean()
        ),
    }
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--hospitals", type=Path, default=Path("SARA_processed"))
    parser.add_argument(
        "--hospital-scenario",
        choices=("public", "all_grid"),
        default="public",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--hospital-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = summarize(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
