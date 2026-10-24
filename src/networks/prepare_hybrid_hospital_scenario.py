#!/usr/bin/env python3
"""Create public-grid hospital topology inputs for a hybrid-model scenario."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np

from build_matpower_case import node_kind
from hospital_grid_ablation import (
    add_assigned_hospitals_to_reference_graph,
    attach_hospitals_to_nearest_grid,
    hospital_node_id,
)


EXPECTED_COUNTS = {"public": 720, "all_grid": 1380}


def load_graph(path: Path) -> nx.MultiGraph:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError(f"expected an undirected networkx.MultiGraph: {path}")
    return graph


def atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty assignment table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def active_grid_nodes(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"node_id", "active"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"active bus mapping lacks {sorted(required)}: {path}")
    nodes = {str(row["node_id"]) for row in rows if int(row["active"]) == 1}
    if not nodes:
        raise ValueError(f"active bus mapping contains no retained nodes: {path}")
    return nodes


def edge_identity(edge: tuple[object, object, object]) -> tuple[str, str, str]:
    return str(edge[0]), str(edge[1]), str(edge[2])


def prepare(args: argparse.Namespace) -> dict[str, object]:
    simplified = load_graph(args.network)
    reference = load_graph(args.reference_network)
    hospitals = gpd.read_file(args.hospitals)
    if hospitals.crs is None:
        raise ValueError(f"hospital data has no CRS: {args.hospitals}")
    if not hospitals.index.is_unique:
        raise ValueError("SARA hospital indices must be unique")
    hospitals = hospitals.to_crs(epsg=4326)

    valid = hospitals.geometry.notna() & ~hospitals.geometry.is_empty
    public = hospitals["ele_main_s"].eq(1)
    solar = hospitals["ele_main_s"].eq(3) | hospitals["ele_sec_so"].eq(3)
    selected = hospitals.loc[
        valid & (public if args.scenario == "public" else (public | solar))
    ].copy()
    selected["hospital_id"] = [hospital_node_id(index) for index in selected.index]
    if len(selected) != EXPECTED_COUNTS[args.scenario]:
        raise ValueError(
            f"{args.scenario} scenario has {len(selected)} hospitals; "
            f"expected {EXPECTED_COUNTS[args.scenario]}"
        )

    source_hospital_ids = {
        str(node)
        for node, data in simplified.nodes(data=True)
        if node_kind(data) == "hospital"
    }
    public_ids = {
        hospital_node_id(index)
        for index in hospitals.index[valid & public]
    }
    if source_hospital_ids != public_ids:
        raise ValueError(
            "source graph hospital identities do not match the spatially valid "
            f"public cohort: graph={len(source_hospital_ids)}, public={len(public_ids)}"
        )

    original_edge_order = [
        edge_identity(edge) for edge in simplified.edges(keys=True)
    ]
    candidate_nodes = (
        active_grid_nodes(args.active_bus_mapping)
        if args.active_bus_mapping is not None
        else None
    )
    if args.scenario == "all_grid" and candidate_nodes is None:
        raise ValueError(
            "all_grid preparation requires --active-bus-mapping so new "
            "connections use the retained generator-backed public grid"
        )

    records = list(zip(selected["hospital_id"], selected.geometry))
    scenario_graph, assignments = attach_hospitals_to_nearest_grid(
        simplified,
        records,
        copy_graph=True,
        candidate_grid_node_ids=candidate_nodes,
    )
    scenario_reference = add_assigned_hospitals_to_reference_graph(
        reference, records, assignments, copy_graph=True
    )

    shared_edge_order = [
        edge_identity((u, v, key))
        for u, v, key, data in scenario_graph.edges(keys=True, data=True)
        if not bool(data.get("ablation", False))
    ]
    if shared_edge_order != original_edge_order:
        raise ValueError(
            "adding ablation connections changed the relative order of shared edges"
        )

    assignment_by_id = {
        str(assignment.hospital_node_id): assignment for assignment in assignments
    }
    rows: list[dict[str, object]] = []
    for index, row in selected.iterrows():
        hospital_id = str(row["hospital_id"])
        assignment = assignment_by_id[hospital_id]
        rows.append(
            {
                "hospital_id": hospital_id,
                "facility_i": row["facility_i"],
                "grid_node_id": str(assignment.grid_node_id),
                "distance_km": f"{assignment.distance_km:.10f}",
                "connection_source": assignment.source,
                "public_grid_baseline": int(bool(public.loc[index])),
                "solar_baseline": int(bool(solar.loc[index])),
            }
        )

    added_distances = np.asarray(
        [
            assignment.distance_km
            for assignment in assignments
            if assignment.source == "ablation"
        ],
        dtype=float,
    )
    simplified_path = args.output_dir / "simplified_hospital_scenario.pickle"
    reference_path = args.output_dir / "hospital_scenario_reference.pickle"
    assignment_path = args.output_dir / "hospital_grid_assignments.csv"
    atomic_pickle(scenario_graph, simplified_path)
    atomic_pickle(scenario_reference, reference_path)
    write_csv(assignment_path, rows)

    report: dict[str, object] = {
        "schema_version": 1,
        "scenario": args.scenario,
        "counts": {
            "hospitals": len(selected),
            "existing_connections": sum(
                assignment.source == "existing" for assignment in assignments
            ),
            "ablation_connections": len(added_distances),
            "simplified_nodes": scenario_graph.number_of_nodes(),
            "simplified_edges": scenario_graph.number_of_edges(),
            "reference_nodes": scenario_reference.number_of_nodes(),
            "reference_edges": scenario_reference.number_of_edges(),
            "candidate_retained_grid_nodes": (
                len(candidate_nodes) if candidate_nodes is not None else None
            ),
        },
        "ablation_connection_distance_km": {
            "median": float(np.median(added_distances)) if len(added_distances) else 0.0,
            "p90": float(np.quantile(added_distances, 0.90)) if len(added_distances) else 0.0,
            "maximum": float(np.max(added_distances)) if len(added_distances) else 0.0,
        },
        "paired_sampling": {
            "shared_edge_order_preserved": True,
            "shared_random_seed": 1234,
            "ablation_random_seed": 1235,
        },
        "outputs": {
            "network": str(simplified_path),
            "reference_network": str(reference_path),
            "assignments": str(assignment_path),
        },
    }
    (args.output_dir / "hospital_scenario_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("network_save/simplified_hospital_MG.pickle"),
    )
    parser.add_argument(
        "--reference-network",
        type=Path,
        default=Path("network_save/MG_hospital.pickle"),
    )
    parser.add_argument("--hospitals", type=Path, default=Path("SARA_processed"))
    parser.add_argument("--active-bus-mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = prepare(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
