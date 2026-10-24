#!/usr/bin/env python3
"""Map saved cyclone node fragilities to MATPOWER branch probabilities.

The saved wind layers are ESRI shapefiles with very wide DBF records. Reading
them through GeoPandas loads dozens of unused columns and geometry. This module
contains a small, read-only DBF reader that extracts only ``id``, tower failure,
and line failure, then applies the same full-network paths used by the existing
cyclone workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


DBF_FIELDS = ("id", "tower_frac", "line_frac_")


@dataclass(frozen=True)
class DbfField:
    name: str
    kind: str
    offset: int
    width: int
    decimals: int


def dbf_schema(path: Path) -> tuple[int, int, int, dict[str, DbfField]]:
    """Return record count, header size, record size, and field definitions."""

    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"truncated DBF header: {path}")
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]
        descriptors = handle.read(header_length - 33)
        terminator = handle.read(1)
    if terminator != b"\r":
        raise ValueError(f"invalid DBF field terminator in {path}")
    if len(descriptors) % 32:
        raise ValueError(f"invalid DBF descriptor length in {path}")

    fields: dict[str, DbfField] = {}
    offset = 1  # deletion marker
    for start in range(0, len(descriptors), 32):
        descriptor = descriptors[start : start + 32]
        name = descriptor[:11].split(b"\0", 1)[0].decode("ascii")
        width = descriptor[16]
        field = DbfField(
            name=name,
            kind=chr(descriptor[11]),
            offset=offset,
            width=width,
            decimals=descriptor[17],
        )
        fields[name] = field
        offset += width
    if offset != record_length:
        raise ValueError(
            f"DBF fields occupy {offset} bytes but record length is "
            f"{record_length}: {path}"
        )
    return record_count, header_length, record_length, fields


def read_dbf_columns(
    path: Path,
    requested: Iterable[str] = DBF_FIELDS,
    chunk_records: int = 2048,
) -> dict[str, list[str | float]]:
    """Read selected character/numeric DBF columns without loading geometry."""

    record_count, header_length, record_length, fields = dbf_schema(path)
    requested = tuple(requested)
    missing = [name for name in requested if name not in fields]
    if missing:
        raise ValueError(f"DBF is missing required fields {missing}: {path}")

    output: dict[str, list[str | float]] = {name: [] for name in requested}
    kept = 0
    with path.open("rb") as handle:
        handle.seek(header_length)
        while kept < record_count:
            want = min(chunk_records, record_count - kept)
            block = handle.read(want * record_length)
            if len(block) != want * record_length:
                raise ValueError(f"truncated DBF record data: {path}")
            view = memoryview(block)
            for row in range(want):
                base = row * record_length
                if view[base] == ord("*"):
                    continue
                for name in requested:
                    field = fields[name]
                    raw = bytes(
                        view[
                            base + field.offset :
                            base + field.offset + field.width
                        ]
                    ).strip()
                    if field.kind in {"N", "F"}:
                        # The ESRI Shapefile DBF driver writes missing numeric
                        # values as a field full of asterisks. Treat that
                        # standard null marker like an empty numeric field,
                        # while continuing to reject other malformed values.
                        if not raw or raw == b"*" * len(raw):
                            value = math.nan
                        else:
                            value = float(raw)
                    elif field.kind == "C":
                        value = raw.decode("utf-8", errors="replace")
                    else:
                        raise ValueError(
                            f"unsupported DBF type {field.kind!r} for {name}"
                        )
                    output[name].append(value)
            kept += want
    return output


def resolve_dbf(wind_track: Path) -> Path:
    if wind_track.is_file() and wind_track.suffix.lower() == ".dbf":
        return wind_track
    if not wind_track.is_dir():
        raise FileNotFoundError(f"wind-track path does not exist: {wind_track}")
    candidates = sorted(wind_track.glob("*.dbf"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one DBF in {wind_track}, found {len(candidates)}"
        )
    return candidates[0]


def load_multigraph(path: Path) -> nx.MultiGraph:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("expected an undirected networkx.MultiGraph")
    return graph


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


def build_path_cache(
    network_path: Path,
    branch_mapping_path: Path,
    output_path: Path,
) -> dict[str, int]:
    graph = load_multigraph(network_path)
    with branch_mapping_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["branch_id"]))
    expected = list(range(1, len(rows) + 1))
    actual = [int(row["branch_id"]) for row in rows]
    if actual != expected:
        raise ValueError("branch mapping IDs must be contiguous and one-based")

    node_index: dict[str, int] = {}
    node_ids: list[str] = []
    flat_paths: list[int] = []
    path_indptr = [0]
    path_by_endpoints: dict[tuple[str, str], list[str]] = {}
    max_path_nodes = 0

    for row in rows:
        first, second = row["from_node"], row["to_node"]
        pair = tuple(sorted((first, second)))
        path = path_by_endpoints.get(pair)
        if path is None:
            path = [str(node) for node in nx.bidirectional_shortest_path(
                graph, first, second
            )]
            path_by_endpoints[pair] = path
        elif path[0] != first:
            path = list(reversed(path))
        max_path_nodes = max(max_path_nodes, len(path))
        for node in path:
            if node not in node_index:
                node_index[node] = len(node_ids)
                node_ids.append(node)
            flat_paths.append(node_index[node])
        path_indptr.append(len(flat_paths))

    atomic_npz(
        output_path,
        format_version=np.asarray([1], dtype=np.int16),
        branch_uid=np.asarray(actual, dtype=np.int32),
        node_ids=np.asarray(node_ids),
        path_nodes=np.asarray(flat_paths, dtype=np.int32),
        path_indptr=np.asarray(path_indptr, dtype=np.int64),
    )
    return {
        "branches": len(rows),
        "unique_full_network_nodes": len(node_ids),
        "unique_endpoint_pairs": len(path_by_endpoints),
        "path_node_references": len(flat_paths),
        "max_path_nodes": max_path_nodes,
    }


def load_path_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    if cache.get("format_version", np.asarray([])).tolist() != [1]:
        raise ValueError(f"unsupported path-cache version: {path}")
    branch_count = len(cache["branch_uid"])
    if len(cache["path_indptr"]) != branch_count + 1:
        raise ValueError(f"invalid path-cache offsets: {path}")
    return cache


def node_failure_probabilities(
    dbf_path: Path,
) -> tuple[dict[str, tuple[float, float]], int]:
    columns = read_dbf_columns(dbf_path)
    probabilities: dict[str, tuple[float, float]] = {}
    duplicate_ids = 0
    for raw_id, raw_tower, raw_line in zip(
        columns["id"], columns["tower_frac"], columns["line_frac_"]
    ):
        node_id = str(raw_id)
        tower = 0.0 if not math.isfinite(float(raw_tower)) else float(raw_tower)
        line = 0.0 if not math.isfinite(float(raw_line)) else float(raw_line)
        value = (min(1.0, max(0.0, tower)), min(1.0, max(0.0, line)))
        if node_id in probabilities:
            duplicate_ids += 1
            old = probabilities[node_id]
            value = max(old[0], value[0]), max(old[1], value[1])
        probabilities[node_id] = value
    return probabilities, duplicate_ids


def calculate_branch_probabilities(
    cache: dict[str, np.ndarray],
    node_probabilities: dict[str, tuple[float, float]],
    failure_mode: str = "combined",
) -> tuple[np.ndarray, dict[str, int]]:
    if failure_mode not in {"combined", "line"}:
        raise ValueError(
            "failure_mode must be 'combined' or 'line', "
            f"not {failure_mode!r}"
        )
    node_ids = [str(value) for value in cache["node_ids"]]
    tower = np.zeros(len(node_ids), dtype=float)
    line = np.zeros(len(node_ids), dtype=float)
    present = np.zeros(len(node_ids), dtype=bool)
    for index, node_id in enumerate(node_ids):
        value = node_probabilities.get(node_id)
        if value is not None:
            tower[index], line[index] = value
            present[index] = True

    flat = cache["path_nodes"]
    indptr = cache["path_indptr"]
    probabilities = np.zeros(len(cache["branch_uid"]), dtype=float)
    missing_segment_endpoints = 0
    paths_without_hazard_nodes = 0
    for branch in range(len(probabilities)):
        path = flat[indptr[branch] : indptr[branch + 1]]
        if len(path) < 2:
            raise ValueError(f"branch {branch + 1} has fewer than two path nodes")
        tower_survival = float(np.prod(1.0 - tower[path[1:-1]]))
        line_survival = 1.0
        for first, second in zip(path[:-1], path[1:]):
            if present[first] and present[second]:
                segment_probability = max(line[first], line[second])
            elif present[first]:
                segment_probability = line[first]
                missing_segment_endpoints += 1
            elif present[second]:
                segment_probability = line[second]
                missing_segment_endpoints += 1
            else:
                segment_probability = 0.0
                missing_segment_endpoints += 2
            line_survival *= 1.0 - segment_probability
        if failure_mode == "line":
            # The original connected-component workflow sampled only its
            # ``line_fail`` edge attribute.  Keep that hazard definition for
            # the hybrid run so the ablation changes supply, demand, and flow
            # constraints rather than the physical damage model.
            probabilities[branch] = 1.0 - line_survival
        else:
            probabilities[branch] = 1.0 - tower_survival * line_survival
        if not np.any(present[path]):
            paths_without_hazard_nodes += 1
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities, {
        "cache_nodes_missing_from_hazard": int(np.count_nonzero(~present)),
        "missing_segment_endpoints": missing_segment_endpoints,
        "paths_without_hazard_nodes": paths_without_hazard_nodes,
    }


def atomic_csv(path: Path, rows: Iterable[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["branch_uid", "failure_probability"])
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def prepare_track(
    path_cache: Path,
    wind_track: Path,
    output_path: Path,
    failure_mode: str = "combined",
) -> dict[str, object]:
    dbf_path = resolve_dbf(wind_track)
    cache = load_path_cache(path_cache)
    node_probabilities, duplicate_ids = node_failure_probabilities(dbf_path)
    probabilities, quality = calculate_branch_probabilities(
        cache, node_probabilities, failure_mode=failure_mode
    )
    branch_ids = cache["branch_uid"].astype(int)
    atomic_csv(
        output_path,
        zip(branch_ids.tolist(), probabilities.tolist()),
    )
    return {
        "wind_dbf": str(dbf_path),
        "output": str(output_path),
        "hazard_nodes": len(node_probabilities),
        "duplicate_hazard_node_ids": duplicate_ids,
        "branches": len(probabilities),
        "branches_with_positive_probability": int(np.count_nonzero(probabilities > 0)),
        "mean_failure_probability": float(np.mean(probabilities)),
        "maximum_failure_probability": float(np.max(probabilities)),
        "failure_mode": failure_mode,
        "quality": quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("build-cache")
    cache_parser.add_argument("--network", type=Path, required=True)
    cache_parser.add_argument("--branch-mapping", type=Path, required=True)
    cache_parser.add_argument("--output", type=Path, required=True)

    track_parser = subparsers.add_parser("track")
    track_parser.add_argument("--path-cache", type=Path, required=True)
    track_parser.add_argument("--wind-track", type=Path, required=True)
    track_parser.add_argument("--output", type=Path, required=True)
    track_parser.add_argument(
        "--failure-mode",
        choices=("combined", "line"),
        default="combined",
        help=(
            "Use combined tower/line failure for MATPOWER (default), or "
            "line-only failure to reproduce the original component sampler."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-cache":
        report = build_path_cache(args.network, args.branch_mapping, args.output)
    else:
        report = prepare_track(
            args.path_cache,
            args.wind_track,
            args.output,
            failure_mode=args.failure_mode,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
