#!/usr/bin/env python3
"""Build a feasible DC MATPOWER case from the existing Mozambique grid graph.

The graph topology is preserved exactly for all non-hospital nodes and edges.
Hospital service edges are converted to load-to-bus mappings rather than power
branches. Generator capacities come from the committed, source-backed CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, diags, hstack, vstack


DEFAULT_NETWORK = Path("network_save/simplified_hospital_MG.pickle")
DEFAULT_GENERATORS = Path("data/matpower_sources/generator_capacity.csv")
DEFAULT_SOURCES = Path("data/matpower_sources/sources.json")
DEFAULT_OUTPUT = Path("data/matpower_output")
DEFAULT_CASE_NAME = "case_mozambique_simplified"

BASE_MVA = 100.0
DEFAULT_REACTANCE_OHM_PER_KM = 0.4
MIN_LENGTH_KM = 0.05
MIN_X_PU = 1e-4
DEFAULT_VOLTAGE_KV = 110.0

LOAD_NODE_WEIGHTS = {
    "substation": 8.0,
    "transformer": 4.0,
    "portal": 2.0,
    "transition": 2.0,
    "gridfinder": 1.0,
    "plant": 1.0,
    "generator": 1.0,
    "connection": 1.0,
}

GENERATOR_COST = {
    "hydro": 5.0,
    "solar": 6.0,
    "gas": 40.0,
    "hfo": 85.0,
    "diesel": 100.0,
}


@dataclass(frozen=True)
class GeneratorRecord:
    node_id: str
    name: str
    status: str
    fuel: str
    installed_mw: float
    pmax_mw: float
    source_id: str
    source_detail: str
    notes: str


@dataclass
class BranchRecord:
    uid: int
    from_node: str
    to_node: str
    edge_key: str
    edge_type: str
    from_bus: int
    to_bus: int
    length_km: float
    voltage_kv: float
    circuits: int
    x_pu: float
    rate_a_mw: float
    status: int = 1
    flow_mw: float = 0.0


def node_kind(data: dict[str, Any]) -> str:
    properties = data.get("properties") or {}
    return str(data.get("power") or properties.get("power") or "unknown")


def edge_attributes(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("attribute", data)
    return value if isinstance(value, dict) else {}


def edge_kind(data: dict[str, Any]) -> str:
    return str(edge_attributes(data).get("power") or "unknown")


def numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))]


def voltage_kv_from_value(value: Any) -> float | None:
    values = numeric_values(value)
    if not values:
        return None
    voltage = max(values)
    return voltage / 1000.0 if voltage > 1000 else voltage


def node_voltage_kv(data: dict[str, Any]) -> float | None:
    return voltage_kv_from_value((data.get("properties") or {}).get("voltage"))


def iter_coordinate_pairs(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            yield float(value[0]), float(value[1])
        else:
            for child in value:
                yield from iter_coordinate_pairs(child)


def node_lon_lat(data: dict[str, Any]) -> tuple[float, float]:
    geometry = data.get("geometry")
    coordinates = geometry.get("coordinates") if hasattr(geometry, "get") else None
    pairs = list(iter_coordinate_pairs(coordinates))
    if not pairs:
        raise ValueError(f"node has no usable geometry: {data}")
    # A mean coordinate is sufficient for endpoint distance and works for points,
    # line vertices, and small substation/plant polygons.
    lon = sum(p[0] for p in pairs) / len(pairs)
    lat = sum(p[1] for p in pairs) / len(pairs)
    return lon, lat


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def circuit_count(attributes: dict[str, Any]) -> int:
    circuits = numeric_values(attributes.get("circuits"))
    if circuits:
        return max(1, int(round(max(circuits))))
    cables = numeric_values(attributes.get("cables"))
    if cables:
        return max(1, int(round(max(cables) / 3.0)))
    return 1


def assumed_rate_mw(voltage_kv: float) -> float:
    """Return a transparent planning proxy when no branch rating is available.

    These are deliberately not presented as observed thermal ratings. They are
    sized so the source-backed national operating snapshot is feasible without
    changing the mapped topology; sensitivity runs should vary them.
    """
    if voltage_kv <= 66:
        return 125.0
    if voltage_kv <= 132:
        return 250.0
    if voltage_kv <= 275:
        return 625.0
    if voltage_kv <= 330:
        return 1000.0
    return 1750.0


def load_generator_records(path: Path) -> list[GeneratorRecord]:
    records: list[GeneratorRecord] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                GeneratorRecord(
                    node_id=row["node_id"],
                    name=row["official_name"],
                    status=row["status"].strip().lower(),
                    fuel=row["fuel"].strip().lower(),
                    installed_mw=float(row["installed_mw"]),
                    pmax_mw=float(row["pmax_mw"]),
                    source_id=row["source_id"],
                    source_detail=row["source_detail"],
                    notes=row["notes"],
                )
            )
    return records


def load_graph(path: Path) -> nx.MultiGraph:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("expected an undirected networkx.MultiGraph")
    return graph


def power_graph_and_hospital_mapping(
    graph: nx.MultiGraph,
) -> tuple[nx.MultiGraph, dict[str, str], list[str]]:
    hospital_nodes = sorted(
        (str(n) for n, data in graph.nodes(data=True) if node_kind(data) == "hospital")
    )
    hospital_set = set(hospital_nodes)
    hospital_to_grid: dict[str, str] = {}
    unresolved: list[str] = []
    for hospital in hospital_nodes:
        candidates = sorted(str(n) for n in graph.neighbors(hospital) if str(n) not in hospital_set)
        if candidates:
            hospital_to_grid[hospital] = candidates[0]
        else:
            unresolved.append(hospital)

    power_graph = graph.copy()
    power_graph.remove_nodes_from(hospital_nodes)
    return power_graph, hospital_to_grid, unresolved


def stable_components(graph: nx.MultiGraph) -> tuple[dict[str, int], dict[int, set[str]]]:
    components = sorted(
        ({str(n) for n in component} for component in nx.connected_components(graph)),
        key=lambda values: (-len(values), min(values)),
    )
    node_component: dict[str, int] = {}
    component_nodes: dict[int, set[str]] = {}
    for component_id, nodes in enumerate(components, start=1):
        component_nodes[component_id] = nodes
        for node in nodes:
            node_component[node] = component_id
    return node_component, component_nodes


def infer_edge_voltage_kv(
    graph: nx.MultiGraph, from_node: str, to_node: str, data: dict[str, Any]
) -> float:
    direct = voltage_kv_from_value(edge_attributes(data).get("voltage"))
    if direct:
        return direct
    endpoints = [
        value
        for value in (
            node_voltage_kv(graph.nodes[from_node]),
            node_voltage_kv(graph.nodes[to_node]),
        )
        if value
    ]
    return min(endpoints) if endpoints else DEFAULT_VOLTAGE_KV


def build_branches(
    graph: nx.MultiGraph,
    bus_id: dict[str, int],
    active_components: set[int],
    node_component: dict[str, int],
) -> list[BranchRecord]:
    coordinates = {str(n): node_lon_lat(data) for n, data in graph.nodes(data=True)}
    edges = sorted(
        graph.edges(keys=True, data=True),
        key=lambda item: (str(item[0]), str(item[1]), str(item[2])),
    )
    records: list[BranchRecord] = []
    for uid, (raw_u, raw_v, key, data) in enumerate(edges, start=1):
        u, v = str(raw_u), str(raw_v)
        if u == v:
            raise ValueError(f"MATPOWER cannot represent self-loop {u} key={key}")
        attributes = edge_attributes(data)
        voltage_kv = infer_edge_voltage_kv(graph, raw_u, raw_v, data)
        circuits = circuit_count(attributes)
        length_km = max(MIN_LENGTH_KM, haversine_km(coordinates[u], coordinates[v]))
        x_ohm = DEFAULT_REACTANCE_OHM_PER_KM * length_km / circuits
        x_pu = max(MIN_X_PU, x_ohm * BASE_MVA / (voltage_kv**2))
        rate = assumed_rate_mw(voltage_kv) * circuits
        records.append(
            BranchRecord(
                uid=uid,
                from_node=u,
                to_node=v,
                edge_key=str(key),
                edge_type=edge_kind(data),
                from_bus=bus_id[u],
                to_bus=bus_id[v],
                length_km=length_km,
                voltage_kv=voltage_kv,
                circuits=circuits,
                x_pu=x_pu,
                rate_a_mw=rate,
                status=int(node_component[u] in active_components),
            )
        )
    return records


def component_load_targets(
    global_target: float,
    activity: dict[int, float],
    capacity: dict[int, float],
    reserve_fraction: float,
) -> dict[int, float]:
    # A compact water-filling allocation with component capacity caps.
    result = {component: 0.0 for component in activity}
    remaining = global_target
    open_components = set(activity)
    while open_components and remaining > 1e-9:
        denominator = sum(activity[c] for c in open_components) or float(len(open_components))
        capped_any = False
        provisional = {
            c: remaining * (activity[c] / denominator if denominator else 1.0 / len(open_components))
            for c in open_components
        }
        for component in list(open_components):
            cap = reserve_fraction * capacity[component]
            room = cap - result[component]
            if provisional[component] >= room - 1e-9:
                result[component] += max(0.0, room)
                remaining = global_target - sum(result.values())
                open_components.remove(component)
                capped_any = True
        if not capped_any:
            for component in open_components:
                result[component] += provisional[component]
            remaining = 0.0
    return result


def allocate_loads(
    graph: nx.MultiGraph,
    bus_id: dict[str, int],
    node_component: dict[str, int],
    active_components: set[int],
    hospital_to_grid: dict[str, str],
    generator_capacity: dict[int, float],
    official_peak_mw: float,
    reserve_fraction: float,
    hospital_share: float,
) -> tuple[np.ndarray, dict[str, int], dict[int, float], float]:
    hospital_count: Counter[str] = Counter(
        grid_node for grid_node in hospital_to_grid.values() if grid_node in bus_id
    )
    infra_raw: dict[str, float] = {}
    hospital_raw: dict[str, float] = {}
    component_activity: Counter[int] = Counter()
    eligible_by_component: defaultdict[int, list[str]] = defaultdict(list)

    for raw_node, data in graph.nodes(data=True):
        node = str(raw_node)
        component = node_component[node]
        if component not in active_components:
            continue
        kind = node_kind(data)
        if kind not in LOAD_NODE_WEIGHTS and hospital_count[node] == 0:
            continue
        base = LOAD_NODE_WEIGHTS.get(kind, 0.5)
        degree_factor = 1.0 + 0.20 * math.log1p(graph.degree[raw_node])
        infra_raw[node] = base * degree_factor
        hospital_raw[node] = float(hospital_count[node])
        component_activity[component] += infra_raw[node] + 2.0 * hospital_raw[node]
        eligible_by_component[component].append(node)

    for component in active_components:
        if not eligible_by_component[component]:
            generator_nodes = [
                node
                for node in bus_id
                if node_component[node] == component and node_kind(graph.nodes[node]) in {"plant", "generator"}
            ]
            if not generator_nodes:
                raise ValueError(f"active component {component} has no load-eligible or generator node")
            node = sorted(generator_nodes)[0]
            infra_raw[node] = 1.0
            hospital_raw[node] = 0.0
            component_activity[component] = 1.0
            eligible_by_component[component].append(node)

    total_capacity = sum(generator_capacity.values())
    global_target = min(official_peak_mw, reserve_fraction * total_capacity)
    targets = component_load_targets(
        global_target,
        dict(component_activity),
        generator_capacity,
        reserve_fraction,
    )

    loads = np.zeros(len(bus_id), dtype=float)
    bus_weights: dict[int, float] = {}
    for component, nodes in eligible_by_component.items():
        infra_total = sum(infra_raw[n] for n in nodes)
        hospital_total = sum(hospital_raw[n] for n in nodes)
        for node in nodes:
            infrastructure_part = infra_raw[node] / infra_total
            hospital_part = (
                hospital_raw[node] / hospital_total if hospital_total else infrastructure_part
            )
            weight = (1.0 - hospital_share) * infrastructure_part + hospital_share * hospital_part
            index = bus_id[node] - 1
            loads[index] = targets[component] * weight
            bus_weights[bus_id[node]] = weight
    return loads, dict(hospital_count), bus_weights, global_target


def dc_opf(
    loads: np.ndarray,
    branches: list[BranchRecord],
    generators: list[dict[str, Any]],
    bus_types: np.ndarray,
) -> dict[str, Any]:
    active_global = [i for i, bus_type in enumerate(bus_types) if bus_type != 4]
    global_to_local = {global_index: local for local, global_index in enumerate(active_global)}
    active_branches = [
        branch
        for branch in branches
        if branch.status
        and branch.from_bus - 1 in global_to_local
        and branch.to_bus - 1 in global_to_local
    ]
    active_generators = [g for g in generators if g["status"] > 0 and g["bus"] - 1 in global_to_local]
    nb = len(active_global)
    nl = len(active_branches)
    ng = len(active_generators)
    if nb == 0 or ng == 0:
        return {"success": False, "message": "no active buses or generators"}

    edge_rows: list[int] = []
    edge_cols: list[int] = []
    edge_data: list[float] = []
    for row, branch in enumerate(active_branches):
        edge_rows.extend([row, row])
        edge_cols.extend(
            [global_to_local[branch.from_bus - 1], global_to_local[branch.to_bus - 1]]
        )
        edge_data.extend([1.0, -1.0])
    incidence = coo_matrix((edge_data, (edge_rows, edge_cols)), shape=(nl, nb)).tocsr()
    # MATPOWER stores x in per unit and uses radians internally, so the MW
    # branch coefficient is baseMVA / x.
    susceptance = np.array([BASE_MVA / branch.x_pu for branch in active_branches])
    branch_flow = diags(susceptance) @ incidence
    bbus = incidence.T @ branch_flow

    gen_rows = [global_to_local[g["bus"] - 1] for g in active_generators]
    gen_cols = list(range(ng))
    generation_map = coo_matrix(
        (np.ones(ng), (gen_rows, gen_cols)), shape=(nb, ng)
    ).tocsr()
    equality = hstack([bbus, -generation_map], format="csr")
    rhs = -loads[active_global]

    zeros = coo_matrix((nl, ng)).tocsr()
    inequality = vstack(
        [hstack([branch_flow, zeros]), hstack([-branch_flow, zeros])],
        format="csr",
    )
    limits = np.array([branch.rate_a_mw for branch in active_branches])
    upper = np.concatenate([limits, limits])

    objective = np.concatenate(
        [np.zeros(nb), np.array([g["cost"] for g in active_generators])]
    )
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * nb
    for global_index in active_global:
        if bus_types[global_index] == 3:
            bounds[global_to_local[global_index]] = (0.0, 0.0)
    bounds.extend((0.0, g["pmax_mw"]) for g in active_generators)

    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=upper,
        A_eq=equality,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return {"success": False, "message": result.message}

    theta = np.zeros(len(bus_types))
    theta[active_global] = result.x[:nb]
    pg = result.x[nb:]
    flows = np.asarray(branch_flow @ result.x[:nb]).reshape(-1)
    return {
        "success": True,
        "message": result.message,
        "objective": float(result.fun),
        "theta": theta,
        "pg": pg,
        "flows": flows,
        "active_branches": active_branches,
        "active_generators": active_generators,
    }


def find_feasible_scale(
    target_loads: np.ndarray,
    branches: list[BranchRecord],
    generators: list[dict[str, Any]],
    bus_types: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    full = dc_opf(target_loads, branches, generators, bus_types)
    if full["success"]:
        return 1.0, full
    low, high = 0.0, 1.0
    best = dc_opf(target_loads * low, branches, generators, bus_types)
    if not best["success"]:
        raise RuntimeError(f"zero-load DC OPF is infeasible: {best['message']}")
    for _ in range(40):
        middle = (low + high) / 2.0
        candidate = dc_opf(target_loads * middle, branches, generators, bus_types)
        if candidate["success"]:
            low, best = middle, candidate
        else:
            high = middle
    return low, best


def matrix_lines(rows: Iterable[Iterable[float]]) -> str:
    return "\n".join("    " + " ".join(f"{float(value):.10g}" for value in row) + ";" for row in rows)


def matlab_string(value: str) -> str:
    return value.replace("'", "''")


def write_matpower_case(
    path: Path,
    case_name: str,
    bus: list[list[float]],
    gen: list[list[float]],
    branch: list[list[float]],
    gencost: list[list[float]],
    hospital_bus_ids: list[int],
    hospital_count: list[int],
) -> None:
    content = f"""function mpc = {case_name}
%{case_name.upper()} Source-backed simplified Mozambique DC case.
% Generated by build_matpower_case.py. Do not edit by hand.

mpc.version = '2';
mpc.baseMVA = {BASE_MVA:g};

mpc.bus = [
{matrix_lines(bus)}
];

mpc.gen = [
{matrix_lines(gen)}
];

mpc.branch = [
{matrix_lines(branch)}
];

mpc.gencost = [
{matrix_lines(gencost)}
];

mpc.bus_uid = (1:size(mpc.bus, 1))';
mpc.branch_uid = (1:size(mpc.branch, 1))';
mpc.hospital_buses = [{" ".join(str(x) for x in hospital_bus_ids)}]';
mpc.hospital_count = [
{matrix_lines([[count] for count in hospital_count])}
];
mpc.source_note = '{matlab_string("Generator capacity provenance is in data/matpower_sources/generator_capacity.csv")}';
end
"""
    path.write_text(content)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_case(args: argparse.Namespace) -> dict[str, Any]:
    full_graph = load_graph(args.network)
    power_graph, hospital_to_grid, unresolved_hospitals = power_graph_and_hospital_mapping(
        full_graph
    )
    node_ids = sorted(str(node) for node in power_graph.nodes)
    bus_id = {node: index for index, node in enumerate(node_ids, start=1)}
    node_component, component_nodes = stable_components(power_graph)

    source_manifest = json.loads(args.sources.read_text())
    source_ids = {source["id"] for source in source_manifest["sources"]}
    generator_records = load_generator_records(args.generators)
    missing_sources = sorted({g.source_id for g in generator_records} - source_ids)
    if missing_sources:
        raise ValueError(f"generator table references unknown sources: {missing_sources}")

    operational = [g for g in generator_records if g.status == "operational" and g.pmax_mw > 0]
    missing_generator_nodes = [g.node_id for g in operational if g.node_id not in bus_id]
    if missing_generator_nodes:
        raise ValueError(f"operational generator nodes are missing from graph: {missing_generator_nodes}")

    generator_capacity_by_component: Counter[int] = Counter()
    for generator in operational:
        generator_capacity_by_component[node_component[generator.node_id]] += generator.pmax_mw
    active_components = set(generator_capacity_by_component)
    if not active_components:
        raise ValueError("no operational generators are connected to the topology")

    branches = build_branches(
        power_graph, bus_id, active_components, node_component
    )
    official_peak = (
        args.official_peak_mw
        if args.official_peak_mw is not None
        else float(source_manifest["model_controls"]["edm_peak_mw"])
    )
    target_loads, hospital_counts, bus_weights, target_total = allocate_loads(
        power_graph,
        bus_id,
        node_component,
        active_components,
        hospital_to_grid,
        dict(generator_capacity_by_component),
        official_peak,
        args.reserve_fraction,
        args.hospital_share,
    )

    generators: list[dict[str, Any]] = []
    for uid, record in enumerate(operational, start=1):
        generators.append(
            {
                "uid": uid,
                "bus": bus_id[record.node_id],
                "node_id": record.node_id,
                "name": record.name,
                "fuel": record.fuel,
                "installed_mw": record.installed_mw,
                "pmax_mw": record.pmax_mw,
                "cost": GENERATOR_COST.get(record.fuel, 75.0),
                "status": 1,
                "source_id": record.source_id,
            }
        )

    reference_bus_by_component: dict[int, int] = {}
    for component in active_components:
        candidates = [
            generator
            for generator in generators
            if node_component[generator["node_id"]] == component
        ]
        reference_bus_by_component[component] = max(
            candidates, key=lambda generator: generator["pmax_mw"]
        )["bus"]

    generator_buses = {generator["bus"] for generator in generators}
    bus_types = np.full(len(node_ids), 4, dtype=int)
    for node, identifier in bus_id.items():
        component = node_component[node]
        if component not in active_components:
            continue
        bus_types[identifier - 1] = 2 if identifier in generator_buses else 1
        if reference_bus_by_component[component] == identifier:
            bus_types[identifier - 1] = 3

    scale, solution = find_feasible_scale(
        target_loads, branches, generators, bus_types
    )
    final_loads = target_loads * scale
    theta_degrees = np.degrees(solution["theta"])
    for generator, pg in zip(solution["active_generators"], solution["pg"]):
        generator["pg_mw"] = float(pg)
    for branch, flow in zip(solution["active_branches"], solution["flows"]):
        branch.flow_mw = float(flow)

    incident_voltage: defaultdict[str, list[float]] = defaultdict(list)
    for branch in branches:
        incident_voltage[branch.from_node].append(branch.voltage_kv)
        incident_voltage[branch.to_node].append(branch.voltage_kv)

    bus_matrix: list[list[float]] = []
    bus_rows: list[dict[str, Any]] = []
    for node in node_ids:
        identifier = bus_id[node]
        data = power_graph.nodes[node]
        voltage = node_voltage_kv(data) or max(
            incident_voltage.get(node, [DEFAULT_VOLTAGE_KV])
        )
        component = node_component[node]
        pd = float(final_loads[identifier - 1])
        bus_matrix.append(
            [
                identifier,
                bus_types[identifier - 1],
                pd,
                0,
                0,
                0,
                component,
                1,
                theta_degrees[identifier - 1],
                voltage,
                component,
                1.1,
                0.9,
            ]
        )
        lon, lat = node_lon_lat(data)
        bus_rows.append(
            {
                "bus_id": identifier,
                "node_id": node,
                "node_type": node_kind(data),
                "component_id": component,
                "active": int(component in active_components),
                "base_kv": f"{voltage:.6f}",
                "pd_mw": f"{pd:.10f}",
                "load_weight": f"{bus_weights.get(identifier, 0.0):.12f}",
                "hospital_count": hospital_counts.get(node, 0),
                "longitude": f"{lon:.8f}",
                "latitude": f"{lat:.8f}",
            }
        )

    gen_matrix: list[list[float]] = []
    gen_rows: list[dict[str, Any]] = []
    for generator in generators:
        pg = generator.get("pg_mw", 0.0)
        gen_matrix.append(
            [
                generator["bus"], pg, 0, 9999, -9999, 1, BASE_MVA, 1,
                generator["pmax_mw"], 0, 0, 0, 0, 0, 0, 0,
                9999, 9999, 9999, 9999, 0,
            ]
        )
        gen_rows.append(
            {
                "generator_id": generator["uid"],
                "bus_id": generator["bus"],
                "node_id": generator["node_id"],
                "official_name": generator["name"],
                "fuel": generator["fuel"],
                "installed_mw": generator["installed_mw"],
                "pmax_mw": generator["pmax_mw"],
                "pg_mw": f"{pg:.10f}",
                "source_id": generator["source_id"],
            }
        )

    branch_matrix = [
        [
            branch.from_bus,
            branch.to_bus,
            0,
            branch.x_pu,
            0,
            branch.rate_a_mw,
            branch.rate_a_mw * 1.15,
            branch.rate_a_mw * 1.30,
            0,
            0,
            branch.status,
            -360,
            360,
        ]
        for branch in branches
    ]
    gencost_matrix = [
        [2, 0, 0, 3, 0, generator["cost"], 0] for generator in generators
    ]
    branch_rows = [
        {
            "branch_id": branch.uid,
            "from_bus": branch.from_bus,
            "to_bus": branch.to_bus,
            "from_node": branch.from_node,
            "to_node": branch.to_node,
            "edge_key": branch.edge_key,
            "edge_type": branch.edge_type,
            "length_km": f"{branch.length_km:.8f}",
            "voltage_kv": f"{branch.voltage_kv:.6f}",
            "circuits": branch.circuits,
            "x_pu": f"{branch.x_pu:.12g}",
            "rate_a_mw": f"{branch.rate_a_mw:.6f}",
            "status": branch.status,
            "base_flow_mw": f"{branch.flow_mw:.10f}",
            "base_loading": f"{abs(branch.flow_mw) / branch.rate_a_mw:.10f}",
        }
        for branch in branches
    ]

    hospital_rows: list[dict[str, Any]] = []
    for hospital in sorted(
        str(node)
        for node, data in full_graph.nodes(data=True)
        if node_kind(data) == "hospital"
    ):
        grid_node = hospital_to_grid.get(hospital)
        identifier = bus_id.get(grid_node) if grid_node else None
        component = node_component.get(grid_node) if grid_node else None
        hospital_rows.append(
            {
                "hospital_id": hospital,
                "grid_node_id": grid_node or "",
                "bus_id": identifier or "",
                "component_id": component or "",
                "public_grid_supplied": int(component in active_components) if component else 0,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_path = args.output_dir / f"{args.case_name}.m"
    hospital_bus_ids = sorted(
        identifier
        for node, count in hospital_counts.items()
        if count and (identifier := bus_id.get(node)) and node_component[node] in active_components
    )
    write_matpower_case(
        case_path,
        args.case_name,
        bus_matrix,
        gen_matrix,
        branch_matrix,
        gencost_matrix,
        hospital_bus_ids,
        [hospital_counts.get(node, 0) for node in node_ids],
    )
    write_csv(
        args.output_dir / "bus_mapping.csv", list(bus_rows[0]), bus_rows
    )
    write_csv(
        args.output_dir / "branch_mapping.csv", list(branch_rows[0]), branch_rows
    )
    write_csv(
        args.output_dir / "generator_mapping.csv", list(gen_rows[0]), gen_rows
    )
    write_csv(
        args.output_dir / "hospital_bus_mapping.csv", list(hospital_rows[0]), hospital_rows
    )

    graph_generator_nodes = {
        str(node)
        for node, data in power_graph.nodes(data=True)
        if node_kind(data) in {"plant", "generator"}
    }
    capacity_nodes = {record.node_id for record in generator_records}
    active_load = float(final_loads.sum())
    active_flows = [
        abs(branch.flow_mw) / branch.rate_a_mw
        for branch in branches
        if branch.status
    ]
    report = {
        "case_name": args.case_name,
        "network_path": str(args.network.resolve()),
        "source_snapshot_year": source_manifest["snapshot_year"],
        "base_mva": BASE_MVA,
        "counts": {
            "input_nodes": full_graph.number_of_nodes(),
            "input_edges": full_graph.number_of_edges(),
            "power_buses": len(node_ids),
            "power_branches": len(branches),
            "components": len(component_nodes),
            "active_components": len(active_components),
            "operational_generators": len(generators),
            "hospitals": len(hospital_rows),
            "mapped_hospitals": len(hospital_to_grid),
            "supplied_hospitals": sum(int(row["public_grid_supplied"]) for row in hospital_rows),
        },
        "capacity": {
            "official_edm_peak_mw": official_peak,
            "operational_installed_mw": sum(g["installed_mw"] for g in generators),
            "available_pmax_mw": sum(g["pmax_mw"] for g in generators),
            "initial_target_load_mw": target_total,
            "feasible_load_scale": scale,
            "final_load_mw": active_load,
            "reserve_fraction_of_pmax": 1.0 - active_load / sum(g["pmax_mw"] for g in generators),
        },
        "dc_opf": {
            "success": bool(solution["success"]),
            "objective": solution.get("objective"),
            "max_branch_loading": max(active_flows, default=0.0),
            "overloaded_branches": sum(value > 1.0 + 1e-7 for value in active_flows),
            "generation_mw": sum(g.get("pg_mw", 0.0) for g in generators),
            "load_mw": active_load,
            "balance_error_mw": abs(
                sum(g.get("pg_mw", 0.0) for g in generators) - active_load
            ),
        },
        "quality": {
            "unresolved_hospitals": unresolved_hospitals,
            "unmatched_graph_generator_nodes": sorted(graph_generator_nodes - capacity_nodes),
            "non_operational_capacity_nodes": sorted(
                record.node_id for record in generator_records if record.status != "operational"
            ),
            "minimum_branch_x_pu": min(branch.x_pu for branch in branches),
            "default_voltage_branches": sum(
                math.isclose(branch.voltage_kv, DEFAULT_VOLTAGE_KV) for branch in branches
            ),
        },
        "outputs": {
            "case": str(case_path),
            "bus_mapping": str(args.output_dir / "bus_mapping.csv"),
            "branch_mapping": str(args.output_dir / "branch_mapping.csv"),
            "generator_mapping": str(args.output_dir / "generator_mapping.csv"),
            "hospital_mapping": str(args.output_dir / "hospital_bus_mapping.csv"),
        },
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--generators", type=Path, default=DEFAULT_GENERATORS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-name", default=DEFAULT_CASE_NAME)
    parser.add_argument("--official-peak-mw", type=float)
    parser.add_argument("--reserve-fraction", type=float, default=0.85)
    parser.add_argument("--hospital-share", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.reserve_fraction < 1:
        parser.error("--reserve-fraction must be between 0 and 1")
    if not 0 <= args.hospital_share <= 1:
        parser.error("--hospital-share must be between 0 and 1")
    return args


def main() -> int:
    report = build_case(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
