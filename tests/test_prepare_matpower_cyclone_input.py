import csv
import math
import pickle
import struct

import networkx as nx
import numpy as np

from prepare_matpower_cyclone_input import (
    build_path_cache,
    calculate_branch_probabilities,
    load_path_cache,
    read_dbf_columns,
)


def write_test_dbf(path):
    fields = [
        ("id", "C", 10, 0),
        ("tower_frac", "N", 8, 3),
        ("line_frac_", "N", 8, 3),
    ]
    records = [("node/a", 0.2, 0.3), ("node/b", 0.0, 0.4)]
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    header = bytearray(32)
    header[0] = 3
    header[4:8] = struct.pack("<I", len(records))
    header[8:10] = struct.pack("<H", header_length)
    header[10:12] = struct.pack("<H", record_length)
    descriptors = bytearray()
    for name, kind, width, decimals in fields:
        descriptor = bytearray(32)
        descriptor[: len(name)] = name.encode("ascii")
        descriptor[11] = ord(kind)
        descriptor[16] = width
        descriptor[17] = decimals
        descriptors.extend(descriptor)
    body = bytearray()
    for node_id, tower, line in records:
        body.extend(b" ")
        body.extend(node_id.encode("ascii").ljust(10))
        body.extend(f"{tower:8.3f}".encode("ascii"))
        body.extend(f"{line:8.3f}".encode("ascii"))
    path.write_bytes(bytes(header) + bytes(descriptors) + b"\r" + bytes(body))


def test_selected_dbf_reader(tmp_path):
    path = tmp_path / "hazard.dbf"
    write_test_dbf(path)
    values = read_dbf_columns(path, chunk_records=1)
    assert values["id"] == ["node/a", "node/b"]
    assert values["tower_frac"] == [0.2, 0.0]
    assert values["line_frac_"] == [0.3, 0.4]


def test_selected_dbf_reader_treats_asterisk_numeric_as_null(tmp_path):
    path = tmp_path / "hazard.dbf"
    write_test_dbf(path)
    data = path.read_bytes().replace(b"   0.200", b"********", 1)
    path.write_bytes(data)

    values = read_dbf_columns(path, chunk_records=1)

    assert math.isnan(values["tower_frac"][0])
    assert values["line_frac_"][0] == 0.3


def test_path_cache_and_combined_tower_line_probability(tmp_path):
    graph = nx.MultiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph_path = tmp_path / "graph.pickle"
    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)

    mapping_path = tmp_path / "branches.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["branch_id", "from_node", "to_node"]
        )
        writer.writeheader()
        writer.writerow({"branch_id": 1, "from_node": "a", "to_node": "c"})
        writer.writerow({"branch_id": 2, "from_node": "c", "to_node": "a"})

    cache_path = tmp_path / "paths.npz"
    report = build_path_cache(graph_path, mapping_path, cache_path)
    assert report["branches"] == 2
    cache = load_path_cache(cache_path)
    probabilities, quality = calculate_branch_probabilities(
        cache,
        {
            "a": (0.0, 0.1),
            "b": (0.2, 0.3),
            "c": (0.0, 0.4),
        },
    )
    # Tower survival is 0.8 and line survival is (1-0.3)*(1-0.4).
    np.testing.assert_allclose(probabilities, [0.664, 0.664])
    assert quality["cache_nodes_missing_from_hazard"] == 0

    line_only, _ = calculate_branch_probabilities(
        cache,
        {
            "a": (0.0, 0.1),
            "b": (0.2, 0.3),
            "c": (0.0, 0.4),
        },
        failure_mode="line",
    )
    # The original component workflow samples line_fail and does not combine
    # the independently stored tower probability.
    np.testing.assert_allclose(line_only, [0.58, 0.58])
