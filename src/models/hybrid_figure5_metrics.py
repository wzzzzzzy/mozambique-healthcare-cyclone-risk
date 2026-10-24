#!/usr/bin/env python3
"""Compute only the length-weighted hybrid metrics required by Figure 5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd

from build_hybrid_power_model import (
    NON_TRANSMISSION_EDGE_TYPES,
    atomic_npz,
    load_graph,
    real_edge_lengths,
)
from build_matpower_case import edge_kind
from hybrid_power_model import EPSILON, HybridPowerModel, load_probabilities
from prepare_cascade_admin_summary import load_admin_boundaries, wind_category
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS


def atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_length_cache(args: argparse.Namespace) -> None:
    model = HybridPowerModel(args.model)
    graph = load_graph(args.network)
    reference = load_graph(args.reference_network)
    edges = list(graph.edges(keys=True, data=True))
    if len(edges) != model.total_edge_count:
        raise ValueError(
            f"network has {len(edges)} edges; model expects {model.total_edge_count}"
        )
    length = real_edge_lengths(graph, reference)
    kinds = [edge_kind(data) for *_, data in edges]
    transmission = np.asarray(
        [kind not in NON_TRANSMISSION_EDGE_TYPES for kind in kinds], dtype=bool
    )
    atomic_npz(
        args.output,
        format_version=np.asarray([1], dtype=np.int16),
        total_edge_count=np.asarray([model.total_edge_count], dtype=np.int64),
        edge_uid=model.edge_uid,
        all_edge_length_km=length,
        all_edge_is_transmission=transmission,
    )
    print(
        f"Wrote {len(length)} path-summed edge lengths to {args.output}; "
        f"transmission_lines={np.count_nonzero(transmission)}"
    )


def load_length_cache(
    path: Path, model: HybridPowerModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if archive["format_version"].tolist() != [1]:
            raise ValueError(f"unsupported length cache: {path}")
        if archive["total_edge_count"].tolist() != [model.total_edge_count]:
            raise ValueError("length-cache edge count does not match model")
        if not np.array_equal(archive["edge_uid"], model.edge_uid):
            raise ValueError("length-cache retained edge IDs do not match model")
        length = archive["all_edge_length_km"].astype(float)
        transmission = archive["all_edge_is_transmission"].astype(bool)
    retained = model.edge_uid - 1
    return length, transmission, length[retained], transmission[retained]


def prepare_manifest(args: argparse.Namespace) -> None:
    admin = load_admin_boundaries(args.admin_boundaries)
    country = admin.geometry.unary_union
    rows: list[dict[str, object]] = []
    global_id = 0
    for storm_set, count in enumerate(TRACK_COUNTS):
        folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        events = gpd.read_file(args.storm_root / "Moz_cyclone_track" / folder)
        if len(events) != count:
            raise ValueError(f"{folder} has {len(events)} events; expected {count}")
        events = events.to_crs(epsg=4326).reset_index(drop=True)
        speed_column = "MAX_Speed" if "MAX_Speed" in events else "MaxSpeed"
        for track_id, event in events.iterrows():
            geometry = event.geometry.intersection(country)
            category = (
                wind_category(float(event[speed_column]))
                if not geometry.is_empty
                else None
            )
            if category in (1, 2):
                rows.append(
                    {
                        "global_id": global_id,
                        "storm_set": storm_set,
                        "storm_folder": folder,
                        "track_id": track_id,
                        "category": category,
                    }
                )
            global_id += 1
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} Category 1/2 tracks to {args.output}")


def simulate_track(args: argparse.Namespace) -> None:
    model = HybridPowerModel(args.model)
    probabilities = load_probabilities(args.probabilities, model.total_edge_count)
    all_length, all_transmission, retained_length, retained_transmission = (
        load_length_cache(args.length_cache, model)
    )
    positive = np.flatnonzero(probabilities > 0.0)
    retained_lookup = np.full(model.total_edge_count, -1, dtype=np.int64)
    retained_lookup[model.edge_uid - 1] = np.arange(len(model.edge_uid))
    rng = np.random.RandomState(args.seed)
    intact = model.intact_state()
    initial_values: list[float] = []
    removed_values: list[float] = []
    isolated_values: list[float] = []
    cascade_values: list[float] = []
    hospital_values: list[float] = []
    for _ in range(args.samples):
        failed_global = positive[
            rng.uniform(0.0, 1.0, len(positive)) < probabilities[positive]
        ]
        failed = np.zeros(len(model.edge_uid), dtype=bool)
        retained_failed = retained_lookup[failed_global]
        retained_failed = retained_failed[retained_failed >= 0]
        failed[retained_failed] = True
        state = model.solve_sample_state(failed) if np.any(failed) else intact
        removed = float(
            all_length[failed_global][all_transmission[failed_global]].sum()
        )
        surviving = (~failed) & retained_transmission
        indices = np.flatnonzero(surviving)
        labels = state.component_by_node[model.edge_u[indices]]
        if np.any(state.component_by_node[model.edge_v[indices]] != labels):
            raise RuntimeError("surviving edge endpoints have different components")
        service = state.service_by_component[labels]
        isolated_mask = service <= EPSILON
        isolated = float(retained_length[indices[isolated_mask]].sum())
        partial_mask = service > EPSILON
        unserved = 1.0 - service[partial_mask]
        cascade = float(
            np.sum(retained_length[indices[partial_mask]] * unserved)
        )
        removed_values.append(removed)
        isolated_values.append(isolated)
        initial_values.append(removed + isolated)
        cascade_values.append(cascade)
        hospital_values.append(1.0 - state.result.hospital_served_fraction)
    initial = float(np.mean(initial_values))
    cascade = float(np.mean(cascade_values))
    output = {
        "schema_version": 1,
        "length_definition": "sum of unsimplified reference-path segment lengths",
        "storm_set": args.storm_set,
        "storm_folder": args.storm_folder,
        "track_id": args.track_id,
        "samples": args.samples,
        "random_seed": args.seed,
        "result": {
            "mean_removed_failure_length_km": float(np.mean(removed_values)),
            "mean_isolated_zero_service_length_km": float(
                np.mean(isolated_values)
            ),
            "mean_initial_failure_length_km": initial,
            "mean_cascade_failure_length_km": cascade,
            "cascade_initial_ratio": cascade / initial if initial > 0.0 else 0.0,
            "mean_hospital_failure_probability": float(np.mean(hospital_values)),
        },
    }
    atomic_json(output, args.output)
    print(json.dumps(output, sort_keys=True))


def summarize(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    rows: list[dict[str, object]] = []
    for event in manifest.itertuples(index=False):
        path = args.result_root / "summaries" / (
            f"summary_{event.storm_folder}_{event.track_id}_25.json"
        )
        with path.open() as handle:
            summary = json.load(handle)
        if int(summary.get("samples", -1)) != args.samples:
            raise ValueError(f"sample count mismatch: {path}")
        rows.append(
            {
                "storm_set": int(event.storm_set),
                "storm_folder": event.storm_folder,
                "track_id": int(event.track_id),
                **summary["result"],
            }
        )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} track metrics to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--admin-boundaries", type=Path, required=True)
    manifest.add_argument("--storm-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(function=prepare_manifest)

    lengths = subparsers.add_parser("lengths")
    lengths.add_argument("--model", type=Path, required=True)
    lengths.add_argument("--network", type=Path, required=True)
    lengths.add_argument("--reference-network", type=Path, required=True)
    lengths.add_argument("--output", type=Path, required=True)
    lengths.set_defaults(function=prepare_length_cache)

    track = subparsers.add_parser("track")
    track.add_argument("--model", type=Path, required=True)
    track.add_argument("--length-cache", type=Path, required=True)
    track.add_argument("--probabilities", type=Path, required=True)
    track.add_argument("--storm-set", type=int, required=True)
    track.add_argument("--storm-folder", required=True)
    track.add_argument("--track-id", type=int, required=True)
    track.add_argument("--samples", type=int, default=100)
    track.add_argument("--seed", type=int, default=1234)
    track.add_argument("--output", type=Path, required=True)
    track.set_defaults(function=simulate_track)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--result-root", type=Path, required=True)
    summary.add_argument("--samples", type=int, default=100)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(function=summarize)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "samples", 1) < 1:
        raise ValueError("samples must be positive")
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
