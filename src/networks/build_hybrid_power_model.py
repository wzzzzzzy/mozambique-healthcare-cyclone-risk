#!/usr/bin/env python3
"""Build the topology cache for the hybrid component/DC-flow model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from build_matpower_case import edge_kind, node_kind
from prepare_matpower_cyclone_input import build_path_cache
from utils import dist_btw_2node


NON_TRANSMISSION_EDGE_TYPES = {
    "MG-hospital",
    "gridfinder",
    "OSM-gridfinder",
    "connect",
    "connect-isolated",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_edge(first: str, second: str, key: str) -> tuple[str, str, str]:
    return (*sorted((str(first), str(second))), str(key))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_graph(path: Path) -> nx.MultiGraph:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError(f"expected an undirected networkx.MultiGraph: {path}")
    return graph


def real_edge_lengths(
    simplified_graph: nx.MultiGraph,
    reference_graph: nx.MultiGraph,
) -> np.ndarray:
    """Return path-summed lengths for simplified edges in iteration order.

    This reproduces ``add_edge_attribute_for_simplified_network``: each
    simplified edge stands for a shortest path through the unsimplified graph,
    and its physical length is the sum of the geodesic segment lengths along
    that path. Parallel simplified edges with the same endpoints share a path.
    """

    path_length_by_endpoints: dict[tuple[str, str], float] = {}
    lengths: list[float] = []
    for raw_u, raw_v, _ in simplified_graph.edges(keys=True):
        pair = tuple(sorted((str(raw_u), str(raw_v))))
        length = path_length_by_endpoints.get(pair)
        if length is None:
            path = nx.bidirectional_shortest_path(
                reference_graph, raw_u, raw_v
            )
            length = sum(
                dist_btw_2node(reference_graph, first, second)
                for first, second in zip(path[:-1], path[1:])
            )
            path_length_by_endpoints[pair] = length
        lengths.append(length)
    values = np.asarray(lengths, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("path-summed edge lengths must be finite and nonnegative")
    return values


def build_model(args: argparse.Namespace) -> dict[str, Any]:
    graph = load_graph(args.network)
    reference_graph = load_graph(args.reference_network)
    real_lengths_km = real_edge_lengths(graph, reference_graph)
    bus_rows = read_csv(args.bus_mapping)
    branch_rows = read_csv(args.branch_mapping)
    generator_rows = read_csv(args.generator_mapping)
    hospital_rows = read_csv(args.hospital_mapping)

    bus_by_node = {row["node_id"]: row for row in bus_rows}
    active_power_nodes = {
        node for node, row in bus_by_node.items() if int(row["active"]) == 1
    }
    supplied_hospitals = {
        row["hospital_id"]
        for row in hospital_rows
        if int(row["public_grid_supplied"]) == 1
    }
    retained_nodes = active_power_nodes | supplied_hospitals
    missing_nodes = sorted(retained_nodes - {str(node) for node in graph.nodes})
    if missing_nodes:
        raise ValueError(f"retained model nodes are absent from graph: {missing_nodes[:5]}")

    matpower_branch: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in branch_rows:
        key = canonical_edge(row["from_node"], row["to_node"], row["edge_key"])
        if key in matpower_branch:
            raise ValueError(f"duplicate MATPOWER branch mapping for {key}")
        matpower_branch[key] = row

    graph_edges = list(graph.edges(keys=True, data=True))
    edge_mapping_rows: list[dict[str, object]] = []
    retained_edges: list[
        tuple[int, str, str, bool, bool, float, float, float, int]
    ] = []
    all_edge_length_km: list[float] = []
    all_edge_is_transmission: list[bool] = []
    all_edge_is_ablation: list[bool] = []
    matched_power_keys: set[tuple[str, str, str]] = set()
    for uid, (raw_u, raw_v, raw_key, data) in enumerate(graph_edges, start=1):
        u, v, key = str(raw_u), str(raw_v), str(raw_key)
        identity = canonical_edge(u, v, key)
        is_hospital_edge = (
            node_kind(graph.nodes[raw_u]) == "hospital"
            or node_kind(graph.nodes[raw_v]) == "hospital"
        )
        branch = None if is_hospital_edge else matpower_branch.get(identity)
        kind = edge_kind(data)
        length_km = float(real_lengths_km[uid - 1])
        if not np.isfinite(length_km) or length_km < 0.0:
            raise ValueError(f"edge has invalid length: {identity}={length_km}")
        is_transmission = kind not in NON_TRANSMISSION_EDGE_TYPES
        is_ablation = bool(data.get("ablation", False))
        all_edge_length_km.append(length_km)
        all_edge_is_transmission.append(is_transmission)
        all_edge_is_ablation.append(is_ablation)
        if not is_hospital_edge and branch is None:
            raise ValueError(f"power edge lacks MATPOWER mapping: {identity}")
        if branch is not None:
            matched_power_keys.add(identity)
        retained = u in retained_nodes and v in retained_nodes
        edge_mapping_rows.append(
            {
                "branch_id": uid,
                "from_node": u,
                "to_node": v,
                "edge_key": key,
                "edge_type": kind,
                "length_km": length_km,
                "transmission_line": int(is_transmission),
                "hospital_connection": int(is_hospital_edge),
                "ablation_edge": int(is_ablation),
                "retained": int(retained),
                "matpower_branch_id": branch["branch_id"] if branch else "",
            }
        )
        if retained:
            retained_edges.append(
                (
                    uid,
                    u,
                    v,
                    branch is not None,
                    is_transmission,
                    length_km,
                    float(branch["x_pu"]) if branch else np.nan,
                    float(branch["rate_a_mw"]) if branch else np.nan,
                    int(branch["branch_id"]) if branch else -1,
                )
            )
    missing_branches = sorted(set(matpower_branch) - matched_power_keys)
    if missing_branches:
        raise ValueError(
            f"MATPOWER mappings absent from simplified graph: {missing_branches[:5]}"
        )

    node_ids = sorted(retained_nodes)
    node_index = {node: index for index, node in enumerate(node_ids)}
    node_pd = np.asarray(
        [float(bus_by_node[node]["pd_mw"]) if node in bus_by_node else 0.0 for node in node_ids]
    )
    pmax_by_node: Counter[str] = Counter()
    for row in generator_rows:
        pmax_by_node[row["node_id"]] += float(row["pmax_mw"])
    node_pmax = np.asarray([pmax_by_node[node] for node in node_ids], dtype=float)
    node_is_power = np.asarray([node in active_power_nodes for node in node_ids])

    hospital_ids = [row["hospital_id"] for row in hospital_rows]
    if len(set(hospital_ids)) != len(hospital_ids):
        raise ValueError("hospital mapping contains duplicate hospital IDs")
    hospital_modeled = np.asarray(
        [hospital in supplied_hospitals for hospital in hospital_ids], dtype=bool
    )
    hospital_node = np.asarray(
        [node_index.get(hospital, -1) for hospital in hospital_ids], dtype=np.int64
    )

    edge_uid = np.asarray([edge[0] for edge in retained_edges], dtype=np.int64)
    edge_u = np.asarray([node_index[edge[1]] for edge in retained_edges], dtype=np.int64)
    edge_v = np.asarray([node_index[edge[2]] for edge in retained_edges], dtype=np.int64)
    edge_is_power = np.asarray([edge[3] for edge in retained_edges], dtype=bool)
    edge_is_transmission = np.asarray(
        [edge[4] for edge in retained_edges], dtype=bool
    )
    edge_length_km = np.asarray([edge[5] for edge in retained_edges], dtype=float)
    edge_x = np.asarray([edge[6] for edge in retained_edges], dtype=float)
    edge_rate = np.asarray([edge[7] for edge in retained_edges], dtype=float)
    matpower_branch_id = np.asarray([edge[8] for edge in retained_edges], dtype=np.int64)

    # Service connections must be leaves; otherwise a hospital could become a
    # non-physical transmission bridge in the connected-component step.
    retained_graph = graph.subgraph(retained_nodes)
    non_leaf_hospitals = sorted(
        hospital
        for hospital in supplied_hospitals
        if retained_graph.degree[hospital] != 1
    )
    if non_leaf_hospitals:
        raise ValueError(
            "modeled hospital service nodes must have degree one: "
            f"{non_leaf_hospitals[:5]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_mapping_path = args.output_dir / "hybrid_edge_mapping.csv"
    model_path = args.output_dir / "hybrid_power_model.npz"
    cache_path = args.output_dir / "hybrid_path_cache.npz"
    write_csv(edge_mapping_path, edge_mapping_rows)
    atomic_npz(
        model_path,
        format_version=np.asarray([2], dtype=np.int16),
        total_edge_count=np.asarray([len(graph_edges)], dtype=np.int64),
        all_edge_length_km=np.asarray(all_edge_length_km, dtype=float),
        all_edge_is_transmission=np.asarray(
            all_edge_is_transmission, dtype=bool
        ),
        all_edge_is_ablation=np.asarray(all_edge_is_ablation, dtype=bool),
        node_ids=np.asarray(node_ids),
        node_is_power=node_is_power,
        node_pd_mw=node_pd,
        node_pmax_mw=node_pmax,
        edge_uid=edge_uid,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_is_power=edge_is_power,
        edge_is_transmission=edge_is_transmission,
        edge_length_km=edge_length_km,
        edge_x_pu=edge_x,
        edge_rate_mw=edge_rate,
        matpower_branch_id=matpower_branch_id,
        hospital_ids=np.asarray(hospital_ids),
        hospital_node=hospital_node,
        hospital_modeled=hospital_modeled,
    )
    cache_report = build_path_cache(
        args.reference_network, edge_mapping_path, cache_path
    )

    # The intact retained graph must be connected: it represents the single
    # source-backed active MATPOWER component plus its 711 service leaves.
    component_count = nx.number_connected_components(retained_graph)
    report = {
        "schema_version": 1,
        "counts": {
            "source_nodes": graph.number_of_nodes(),
            "source_edges": graph.number_of_edges(),
            "retained_nodes": len(node_ids),
            "retained_edges": len(retained_edges),
            "retained_power_edges": int(np.count_nonzero(edge_is_power)),
            "retained_hospital_connections": int(np.count_nonzero(~edge_is_power)),
            "ablation_connections": int(np.count_nonzero(all_edge_is_ablation)),
            "retained_components": component_count,
            "hospitals": len(hospital_ids),
            "modeled_hospitals": int(np.count_nonzero(hospital_modeled)),
            "unmodeled_hospitals": int(np.count_nonzero(~hospital_modeled)),
            "generators": len(generator_rows),
        },
        "capacity": {
            "load_mw": float(node_pd.sum()),
            "generator_pmax_mw": float(node_pmax.sum()),
        },
        "path_cache": cache_report,
        "outputs": {
            "model": str(model_path),
            "edge_mapping": str(edge_mapping_path),
            "path_cache": str(cache_path),
        },
    }
    if component_count != 1:
        raise ValueError(f"retained hybrid topology has {component_count} components")
    (args.output_dir / "hybrid_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", type=Path, default=Path("network_save/simplified_hospital_MG.pickle")
    )
    parser.add_argument(
        "--reference-network", type=Path, default=Path("network_save/MG_hospital.pickle")
    )
    parser.add_argument("--bus-mapping", type=Path, required=True)
    parser.add_argument("--branch-mapping", type=Path, required=True)
    parser.add_argument("--generator-mapping", type=Path, required=True)
    parser.add_argument("--hospital-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = build_model(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
