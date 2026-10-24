"""Propagate capacity-v2 public-grid cyclone failures to water and telecom.

The capacity-v2 wind workflow writes one connected-component CSV per Monte
Carlo iteration.  Those component files contain the information needed by the
legacy water and communication dependency models, but their names are
versioned by the power scenario.  This module consumes only the non-ablation
``capacity_v2_public`` component files and writes equally versioned downstream
results without overwriting the legacy outputs.  A water hospital whose
upstream dependency cannot be matched to the power-component table is marked
unevaluated (``-1``), rather than being interpreted as a certain failure.

Water and communication can be evaluated together or independently. Static
graph reachability is precomputed once in a compact model and reused by all
Slurm array workers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from hospital_grid_ablation import hospital_node_id
from dependency_models import (
    CommunicationDependencyModel,
    ComponentState,
    DependencyModels,
    WaterDependencyModel,
)
from synthetic_tracks import TRACK_COUNTS, expected_track_keys


MODEL_VERSION = "capacity-v2-public-dependencies-v3-water-simplified"
POWER_SCENARIO = "capacity_v2_public"
WATER_POWER_TYPES = ("gridfinder", "portal", "substation", "transformer")
COMMUNICATION_POWER_TYPES = (
    "gridfinder",
    "transition",
    "transformer",
    "generator",
    "plant",
    "portal",
    "substation",
)


def _ordered_nodes_by_attribute(
    graph: nx.DiGraph,
    attribute: str,
    values: Sequence[str],
) -> tuple[str, ...]:
    """Match the node ordering used by the original propagation scripts."""

    nodes = tuple(graph.nodes)
    return tuple(
        node
        for value in values
        for node in nodes
        if graph.nodes[node].get(attribute) == value
    )


def _hospital_records(
    hospitals: gpd.GeoDataFrame,
    selector: pd.Series,
) -> tuple[tuple[str, bool], ...]:
    selected = hospitals.loc[selector]
    records: list[tuple[str, bool]] = []
    for index, row in selected.iterrows():
        geometry = row.geometry
        valid_geometry = geometry is not None and not geometry.is_empty
        records.append((hospital_node_id(index), valid_geometry))
    return tuple(records)


def build_dependency_models(
    hospitals: gpd.GeoDataFrame,
    water_graph: nx.DiGraph,
    communication_graph: nx.DiGraph,
) -> DependencyModels:
    """Precompute graph reachability while preserving the legacy semantics."""

    if "water_sour" not in hospitals or "has_tel" not in hospitals:
        raise ValueError("Hospital data must contain water_sour and has_tel")

    water_records = _hospital_records(hospitals, hospitals["water_sour"] == 1)
    water_power_nodes = _ordered_nodes_by_attribute(
        water_graph, "water", WATER_POWER_TYPES
    )
    water_dependencies: list[Optional[tuple[str, ...]]] = []
    for hospital_id, valid_geometry in water_records:
        if hospital_id not in water_graph:
            # Service attributes come from SARA, while the legacy dependency
            # graph omits a small number of facilities.  They are genuinely
            # unevaluated and must retain the -1 sentinel rather than aborting
            # the whole synthetic-track rerun or being assigned zero risk.
            water_dependencies.append(None)
            continue
        if not valid_geometry or water_graph.in_degree(hospital_id) == 0:
            water_dependencies.append(None)
            continue
        ancestors = nx.ancestors(water_graph, hospital_id)
        dependencies = tuple(
            node for node in water_power_nodes if node in ancestors
        )
        water_dependencies.append(dependencies or None)

    communication_records = _hospital_records(
        hospitals, hospitals["has_tel"] == 1
    )
    communication_power_nodes = _ordered_nodes_by_attribute(
        communication_graph,
        "communication",
        COMMUNICATION_POWER_TYPES,
    )
    towers = tuple(
        node
        for node in communication_graph.nodes
        if communication_graph.nodes[node].get("communication") == "ctower"
    )
    tower_power_source: dict[str, Optional[str]] = {}
    for tower in towers:
        ancestors = nx.ancestors(communication_graph, tower)
        # The original nested loop overwrites the tower for every upstream
        # power node, so the last node in its ordered list is authoritative.
        sources = [
            node for node in communication_power_nodes if node in ancestors
        ]
        tower_power_source[tower] = sources[-1] if sources else None

    communication_dependencies: list[Optional[tuple[str, ...]]] = []
    tower_set = set(towers)
    for hospital_id, valid_geometry in communication_records:
        if hospital_id not in communication_graph:
            communication_dependencies.append(None)
            continue
        if not valid_geometry or communication_graph.in_degree(hospital_id) == 0:
            communication_dependencies.append(None)
            continue
        predecessors = tuple(communication_graph.predecessors(hospital_id))
        unexpected = set(predecessors).difference(tower_set)
        if unexpected:
            raise ValueError(
                f"Communication hospital {hospital_id} has non-tower "
                f"predecessors: {sorted(map(str, unexpected))[:5]}"
            )
        communication_dependencies.append(predecessors or None)

    return DependencyModels(
        version=MODEL_VERSION,
        water=WaterDependencyModel(
            hospital_ids=tuple(record[0] for record in water_records),
            upstream_power_nodes=tuple(water_dependencies),
            power_nodes=water_power_nodes,
        ),
        communication=CommunicationDependencyModel(
            hospital_ids=tuple(record[0] for record in communication_records),
            predecessor_towers=tuple(communication_dependencies),
            tower_power_source=tuple(tower_power_source.items()),
            power_nodes=communication_power_nodes,
        ),
    )


def load_component_state(
    path: Path,
    relevant_power_nodes: Iterable[str],
) -> ComponentState:
    """Read a component CSV safely and retain only dependency power nodes."""

    frame = pd.read_csv(path, usecols=["ccs", "fail_prob"])
    relevant = set(relevant_power_nodes)
    probability_by_node: dict[str, float] = {}
    component_by_node: dict[str, int] = {}
    for component_id, row in frame.iterrows():
        try:
            nodes = ast.literal_eval(row["ccs"])
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid ccs literal in {path}: row {component_id}") from error
        if not isinstance(nodes, (list, tuple)):
            raise ValueError(
                f"Component nodes must be a list or tuple in {path}: "
                f"row {component_id}"
            )
        probability = float(row["fail_prob"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                f"Invalid failure probability in {path}: "
                f"row {component_id}={probability}"
            )
        for node in nodes:
            if node not in relevant:
                continue
            if node in component_by_node:
                raise ValueError(f"Power node occurs in multiple components: {node}")
            probability_by_node[node] = probability
            component_by_node[node] = int(component_id)
    return ComponentState(probability_by_node, component_by_node)


def evaluate_water_dependencies(
    model: WaterDependencyModel,
    state: ComponentState,
) -> np.ndarray:
    """Evaluate public-water hospital failure for one Monte Carlo sample."""

    values = np.full(len(model.hospital_ids), -1.0, dtype=float)
    for index, dependencies in enumerate(model.upstream_power_nodes):
        if dependencies is None:
            continue
        if any(node not in state.component_by_node for node in dependencies):
            # Missing cross-network identity is missing information, not
            # evidence that the water service certainly failed.
            continue
        probability_by_component: dict[int, float] = {}
        for node in dependencies:
            component = state.component_by_node[node]
            probability = state.probability_by_node[node]
            previous = probability_by_component.setdefault(component, probability)
            if not math.isclose(previous, probability, abs_tol=1e-12):
                raise ValueError(
                    f"Inconsistent probabilities within component {component}"
                )
        survival = math.prod(
            1.0 - probability
            for probability in probability_by_component.values()
        )
        values[index] = 1.0 - survival
    return values


def evaluate_communication_dependencies(
    model: CommunicationDependencyModel,
    state: ComponentState,
) -> np.ndarray:
    """Evaluate hospital communication failure for one Monte Carlo sample."""

    tower_sources = dict(model.tower_power_source)
    values = np.full(len(model.hospital_ids), -1.0, dtype=float)
    for index, towers in enumerate(model.predecessor_towers):
        if towers is None:
            continue
        failure_probability = 1.0
        used_components: set[int] = set()
        for tower in towers:
            source = tower_sources.get(tower)
            if source is None or source not in state.component_by_node:
                # This is the original model's fail-closed behavior for a
                # tower whose upstream power node is absent from the component
                # table.  A probability of one does not change the product.
                continue
            probability = state.probability_by_node[source]
            component = state.component_by_node[source]
            if probability != 1.0 and component not in used_components:
                failure_probability *= probability
                used_components.add(component)
        values[index] = failure_probability
    return values


def _reference_water(
    graph: nx.DiGraph,
    hospital_ids: Sequence[str],
    power_nodes: Sequence[str],
    state: ComponentState,
) -> np.ndarray:
    """Literal one-sample implementation used to validate optimization."""

    graph = graph.copy()
    for node in power_nodes:
        for descendant in nx.descendants(graph, node):
            if node not in state.component_by_node:
                graph.nodes[descendant]["fail_prob"] = 1.0
                continue
            probability = state.probability_by_node[node]
            component = state.component_by_node[node]
            current = graph.nodes[descendant].get("fail_prob", -1)
            if current >= 0:
                components = graph.nodes[descendant].setdefault("cc", [])
                if component not in components:
                    graph.nodes[descendant]["fail_prob"] = (
                        probability + current - current * probability
                    )
            else:
                graph.nodes[descendant]["fail_prob"] = probability
                graph.nodes[descendant]["cc"] = [component]
    values = []
    for hospital_id in hospital_ids:
        if hospital_id not in graph or graph.in_degree(hospital_id) == 0:
            values.append(-1.0)
        else:
            values.append(float(graph.nodes[hospital_id].get("fail_prob", -1)))
    return np.asarray(values)


def _reference_communication(
    graph: nx.DiGraph,
    hospital_ids: Sequence[str],
    power_nodes: Sequence[str],
    state: ComponentState,
) -> np.ndarray:
    """Literal one-sample implementation used to validate optimization."""

    graph = graph.copy()
    for node in power_nodes:
        for descendant in nx.descendants(graph, node):
            if graph.nodes[descendant].get("communication") != "ctower":
                continue
            if node in state.component_by_node:
                graph.nodes[descendant]["fail_prob"] = state.probability_by_node[node]
                graph.nodes[descendant]["ccs"] = state.component_by_node[node]
            else:
                graph.nodes[descendant]["fail_prob"] = 1.0
    values = []
    for hospital_id in hospital_ids:
        if hospital_id not in graph:
            values.append(-1.0)
            continue
        predecessors = tuple(graph.predecessors(hospital_id))
        if not predecessors:
            values.append(-1.0)
            continue
        probability = 1.0
        used_components: set[int] = set()
        for predecessor in predecessors:
            failure = float(graph.nodes[predecessor].get("fail_prob", 1))
            component = graph.nodes[predecessor].get("ccs")
            if failure != 1.0 and component not in used_components:
                probability *= failure
                used_components.add(component)
        values.append(probability)
    return np.asarray(values)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_models(path: Path) -> DependencyModels:
    value = _load_pickle(path)
    if not isinstance(value, DependencyModels) or value.version != MODEL_VERSION:
        raise ValueError(f"Unsupported dependency model in {path}")
    return value


def prepare_models(args: argparse.Namespace) -> None:
    hospitals = gpd.read_file(args.hospital_data)
    if hospitals.crs is None:
        raise ValueError("SARA_processed must define a CRS")
    if hospitals.crs.to_epsg() != 4326:
        hospitals = hospitals.to_crs(epsg=4326)
    water_graph = _load_pickle(args.water_graph)
    communication_graph = _load_pickle(args.communication_graph)
    if not isinstance(water_graph, nx.DiGraph):
        raise TypeError("Water graph is not a directed NetworkX graph")
    if not isinstance(communication_graph, nx.DiGraph):
        raise TypeError("Communication graph is not a directed NetworkX graph")

    models = build_dependency_models(
        hospitals, water_graph, communication_graph
    )
    validation_path = component_path(
        args.component_root,
        "STORM_DATA_IBTRACS_SI_1000_YEARS_0",
        0,
        1,
        args.threshold,
    )
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Public capacity-v2 validation component is missing: {validation_path}"
        )
    state = load_component_state(validation_path, models.relevant_power_nodes)
    optimized_water = evaluate_water_dependencies(models.water, state)
    reference_water = _reference_water(
        water_graph,
        models.water.hospital_ids,
        models.water.power_nodes,
        state,
    )
    reference_water[
        np.asarray([
            dependencies is None
            or any(
                node not in state.component_by_node
                for node in dependencies
            )
            for dependencies in models.water.upstream_power_nodes
        ])
    ] = -1.0
    optimized_communication = evaluate_communication_dependencies(
        models.communication, state
    )
    reference_communication = _reference_communication(
        communication_graph,
        models.communication.hospital_ids,
        models.communication.power_nodes,
        state,
    )
    reference_communication[
        np.asarray([
            towers is None
            for towers in models.communication.predecessor_towers
        ])
    ] = -1.0
    if not np.allclose(optimized_water, reference_water, atol=1e-12):
        difference = np.abs(optimized_water - reference_water)
        raise ValueError(
            "Optimized water propagation differs from the legacy algorithm: "
            f"max_abs={difference.max()}"
        )
    if not np.allclose(
        optimized_communication, reference_communication, atol=1e-12
    ):
        difference = np.abs(
            optimized_communication - reference_communication
        )
        raise ValueError(
            "Optimized communication propagation differs from the legacy "
            f"algorithm: max_abs={difference.max()}"
        )

    _atomic_pickle(args.model_output, models)
    print(json.dumps({
        "model_version": models.version,
        "model_output": str(args.model_output),
        "water_hospitals": len(models.water.hospital_ids),
        "communication_hospitals": len(models.communication.hospital_ids),
        "water_power_nodes": len(models.water.power_nodes),
        "communication_power_nodes": len(models.communication.power_nodes),
        "water_unevaluated": sum(
            dependency is None
            for dependency in models.water.upstream_power_nodes
        ),
        "communication_unevaluated": sum(
            towers is None
            for towers in models.communication.predecessor_towers
        ),
        "validation_component": str(validation_path),
        "legacy_equivalence": True,
    }, indent=2))


def component_path(
    component_root: Path,
    folder: str,
    track_id: int,
    iteration: int,
    threshold: int,
) -> Path:
    return component_root / (
        f"{folder}__{POWER_SCENARIO}_{track_id}_{iteration}_{threshold}.csv"
    )


def failure_metadata_path(
    failure_root: Path,
    folder: str,
    track_id: int,
    threshold: int,
) -> Path:
    return failure_root / (
        f"{threshold}_{folder}__{POWER_SCENARIO}_{track_id}.pickle"
    )


def track_result_path(
    output_root: Path,
    service: str,
    folder: str,
    track_id: int,
    threshold: int,
) -> Path:
    return output_root / service / "tracks" / (
        f"{service}_cyclone_{folder}_{track_id}_{threshold}.csv"
    )


def _iterations_from_metadata(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Power-run metadata is missing: {path}")
    metadata = _load_pickle(path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Unexpected power metadata in {path}")
    if metadata.get("capacity_model") != "unmet_demand":
        raise ValueError(
            f"Refusing non-capacity-v2 power metadata in {path}: "
            f"capacity_model={metadata.get('capacity_model')!r}"
        )
    if metadata.get("random_seed") != 1234:
        raise ValueError(
            f"Unexpected power pairing seed in {path}: "
            f"random_seed={metadata.get('random_seed')!r}"
        )
    iterations = int(metadata.get("iterations", 0))
    if not 0 < iterations <= 10000:
        raise ValueError(f"Invalid iteration count in {path}: {iterations}")
    return iterations


def _track_result_complete(
    path: Path,
    hospital_ids: Sequence[str],
    iterations: int,
) -> bool:
    if not path.is_file() or path.stat().st_size < 64:
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "id", "mean", "std", "sample_num", "evaluated_sample_num",
        "power_scenario", "model_version",
    }
    if not required.issubset(frame.columns):
        return False
    if tuple(frame["id"].astype(str)) != tuple(hospital_ids):
        return False
    if not (pd.to_numeric(frame["sample_num"], errors="coerce") == iterations).all():
        return False
    if not (frame["power_scenario"] == POWER_SCENARIO).all():
        return False
    if not (frame["model_version"] == MODEL_VERSION).all():
        return False
    evaluated_sample_num = pd.to_numeric(
        frame["evaluated_sample_num"], errors="coerce"
    )
    if not evaluated_sample_num.between(0, iterations).all():
        return False
    mean = pd.to_numeric(frame["mean"], errors="coerce")
    return bool(mean.notna().all() and mean.between(-1, 1).all())


def _result_frame(
    hospital_ids: Sequence[str],
    samples: Sequence[np.ndarray],
) -> pd.DataFrame:
    matrix = np.vstack(samples)
    evaluated = pd.DataFrame(matrix).where(matrix >= 0)
    evaluated_sample_num = evaluated.notna().sum(axis=0).to_numpy()
    mean = evaluated.mean(axis=0).fillna(-1.0).to_numpy()
    standard_deviation = evaluated.std(axis=0, ddof=1).to_numpy()
    return pd.DataFrame({
        "id": hospital_ids,
        "mean": mean,
        "std": standard_deviation,
        "sample_num": len(samples),
        "evaluated_sample_num": evaluated_sample_num,
        "power_scenario": POWER_SCENARIO,
        "model_version": MODEL_VERSION,
    })


def _selected_services(service: str) -> tuple[str, ...]:
    if service == "both":
        return ("water", "comm")
    if service in {"water", "comm"}:
        return (service,)
    raise ValueError(f"Unsupported downstream service: {service}")


def run_chunk(args: argparse.Namespace) -> None:
    models = _load_models(args.model_input)
    services = _selected_services(args.service)
    model_by_service = {
        "water": models.water,
        "comm": models.communication,
    }
    relevant_power_nodes: set[str] = set()
    for service in services:
        relevant_power_nodes.update(model_by_service[service].power_nodes)
    track_keys = expected_track_keys()
    if not 0 <= args.start < args.stop <= len(track_keys):
        raise ValueError(
            f"Invalid track slice [{args.start}, {args.stop}) for "
            f"{len(track_keys)} tracks"
        )

    completed = 0
    skipped = 0
    for folder, track_id in track_keys[args.start:args.stop]:
        metadata_path = failure_metadata_path(
            args.failure_root, folder, track_id, args.threshold
        )
        iterations = _iterations_from_metadata(metadata_path)
        result_paths = {
            service: track_result_path(
                args.output_root, service, folder, track_id, args.threshold
            )
            for service in services
        }
        if all(
            _track_result_complete(
                result_paths[service],
                model_by_service[service].hospital_ids,
                iterations,
            )
            for service in services
        ):
            skipped += 1
            continue

        samples: dict[str, list[np.ndarray]] = {
            service: [] for service in services
        }
        for iteration in range(1, iterations + 1):
            path = component_path(
                args.component_root,
                folder,
                track_id,
                iteration,
                args.threshold,
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Component {iteration}/{iterations} is missing: {path}"
                )
            state = load_component_state(path, relevant_power_nodes)
            if "water" in samples:
                samples["water"].append(
                    evaluate_water_dependencies(models.water, state)
                )
            if "comm" in samples:
                samples["comm"].append(
                    evaluate_communication_dependencies(
                        models.communication, state
                    )
                )

        for service in services:
            model = model_by_service[service]
            _atomic_csv(
                result_paths[service],
                _result_frame(model.hospital_ids, samples[service]),
            )
            if not _track_result_complete(
                result_paths[service], model.hospital_ids, iterations
            ):
                raise RuntimeError(
                    f"Written {service} result failed validation: "
                    f"folder={folder}, track={track_id}"
                )
        completed += 1
        print(
            f"completed folder={folder} track={track_id} "
            f"samples={iterations}",
            flush=True,
        )

    print(json.dumps({
        "track_start": args.start,
        "track_stop": args.stop,
        "computed": completed,
        "already_complete": skipped,
        "power_scenario": POWER_SCENARIO,
        "services": services,
    }))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_service(
    service: str,
    hospital_ids: Sequence[str],
    output_root: Path,
    threshold: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    offset = 0
    mean_name = "water_mean" if service == "water" else "comm_mean"
    for storm_set, track_count in enumerate(TRACK_COUNTS):
        folder = f"STORM_DATA_IBTRACS_SI_1000_YEARS_{storm_set}"
        columns: dict[str, pd.Series] = {}
        for track_id in range(track_count):
            path = track_result_path(
                output_root, service, folder, track_id, threshold
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing {service} track result: {path}")
            frame = pd.read_csv(path)
            if tuple(frame["id"].astype(str)) != tuple(hospital_ids):
                raise ValueError(f"Hospital identity mismatch in {path}")
            if not (frame["power_scenario"] == POWER_SCENARIO).all():
                raise ValueError(f"Scenario mismatch in {path}")
            if not (frame["model_version"] == MODEL_VERSION).all():
                raise ValueError(f"Model-version mismatch in {path}")
            values = pd.to_numeric(frame["mean"], errors="coerce")
            if values.isna().any() or not values.between(-1, 1).all():
                raise ValueError(f"Invalid mean probabilities in {path}")
            columns[f"{storm_set}_{track_id}"] = values.reset_index(drop=True)
        result = pd.concat(
            [pd.Series(hospital_ids, name="id"), pd.DataFrame(columns)],
            axis=1,
        )
        evaluated = result[list(columns)].where(
            result[list(columns)] >= 0
        )
        result[mean_name] = evaluated.mean(axis=1).fillna(-1.0)
        aggregate_path = output_root / service / (
            f"{service}_cyclone_all_{folder}_{threshold}.csv"
        )
        _atomic_csv(aggregate_path, result)
        records.append({
            "service": service,
            "storm_set": storm_set,
            "track_count": track_count,
            "aggregate": str(aggregate_path),
            "sha256": _sha256(aggregate_path),
            "rows": len(result),
            "first_global_track": offset,
            "last_global_track": offset + track_count - 1,
        })
        offset += track_count
    return records


def summarize(args: argparse.Namespace) -> None:
    models = _load_models(args.model_input)
    services = _selected_services(args.service)
    model_by_service = {
        "water": models.water,
        "comm": models.communication,
    }
    records: list[dict[str, object]] = []
    for service in services:
        records.extend(_summarize_service(
            service,
            model_by_service[service].hospital_ids,
            args.output_root,
            args.threshold,
        ))
    manifest = {
        "model_version": models.version,
        "power_scenario": POWER_SCENARIO,
        "threshold": args.threshold,
        "expected_tracks": len(expected_track_keys()),
        "services": services,
        "water_hospitals": len(models.water.hospital_ids),
        "communication_hospitals": len(models.communication.hospital_ids),
        "aggregates": records,
    }
    manifest_path = args.output_root / "downstream_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def _default_scratch_path(relative: str) -> Path:
    scratch = os.environ.get("SCRATCH", "/scratch/users/zywei")
    return Path(scratch) / relative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--hospital-data", type=Path, default=Path("SARA_processed"))
    prepare.add_argument(
        "--water-graph",
        type=Path,
        default=Path("network_save/MG_w_hospital_new2.pickle"),
    )
    prepare.add_argument(
        "--communication-graph",
        type=Path,
        default=Path("network_save/MG_c_0205.pickle"),
    )
    prepare.add_argument(
        "--component-root",
        type=Path,
        default=_default_scratch_path("cascade_result/cc_csv_cyclone"),
    )
    prepare.add_argument(
        "--model-output",
        type=Path,
        default=_default_scratch_path(
            "wind_result/dependency_models/"
            "water_comm_capacity_v2_public_v3_water_new2.pickle"
        ),
    )
    prepare.add_argument("--threshold", type=int, default=25)
    prepare.set_defaults(func=prepare_models)

    run = subparsers.add_parser("run-chunk")
    run.add_argument("--start", type=int, required=True)
    run.add_argument("--stop", type=int, required=True)
    run.add_argument("--threshold", type=int, default=25)
    run.add_argument(
        "--service",
        choices=("water", "comm", "both"),
        default="water",
        help="Downstream service to evaluate (default: water)",
    )
    run.add_argument(
        "--model-input",
        type=Path,
        default=_default_scratch_path(
            "wind_result/dependency_models/"
            "water_comm_capacity_v2_public_v3_water_new2.pickle"
        ),
    )
    run.add_argument(
        "--component-root",
        type=Path,
        default=_default_scratch_path("cascade_result/cc_csv_cyclone"),
    )
    run.add_argument(
        "--failure-root",
        type=Path,
        default=_default_scratch_path("cascade_result/failure"),
    )
    run.add_argument(
        "--output-root",
        type=Path,
        default=_default_scratch_path(
            "wind_result/downstream_capacity_v2_public_v3_water_new2"
        ),
    )
    run.set_defaults(func=run_chunk)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--threshold", type=int, default=25)
    summary.add_argument(
        "--service",
        choices=("water", "comm", "both"),
        default="water",
        help="Downstream service to summarize (default: water)",
    )
    summary.add_argument(
        "--model-input",
        type=Path,
        default=_default_scratch_path(
            "wind_result/dependency_models/"
            "water_comm_capacity_v2_public_v3_water_new2.pickle"
        ),
    )
    summary.add_argument(
        "--output-root",
        type=Path,
        default=_default_scratch_path(
            "wind_result/downstream_capacity_v2_public_v3_water_new2"
        ),
    )
    summary.set_defaults(func=summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.threshold != 25:
        raise ValueError(
            "This versioned rerun is restricted to the completed threshold-25 "
            "capacity-v2 public result"
        )
    args.func(args)


if __name__ == "__main__":
    main()
