#!/usr/bin/env python3
"""Run electricity and downstream services from one hybrid outage sample set.

The synthetic hybrid electricity archive stores hospital averages, not the
grid-node state required by the water and communication dependency graphs.
This runner deterministically recreates the same edge draws (seed 1234), solves
the hybrid DC-service model once per sample, and propagates that exact node
state to both downstream systems.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from dependency_models import ComponentState, DependencyModels
from hybrid_power_model import HybridPowerModel, load_probabilities
from propagate_cyclone_dependencies import (
    evaluate_communication_dependencies,
    evaluate_water_dependencies,
)


MODEL_VERSION = "hybrid-public-multisystem-v1"
POWER_SCENARIO = "hybrid_public"


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


def atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def load_dependency_models(path: Path) -> DependencyModels:
    with path.open("rb") as handle:
        models = pickle.load(handle)
    if not isinstance(models, DependencyModels):
        raise TypeError(f"unexpected dependency-model object in {path}")
    return models


def component_state(
    model: HybridPowerModel,
    component_by_node: np.ndarray,
    service_by_component: np.ndarray,
    relevant_nodes: frozenset[str],
) -> ComponentState:
    """Expose hybrid component service for downstream dependency nodes."""

    node_lookup = {
        node_id: index
        for index, node_id in enumerate(model.node_ids)
        if node_id in relevant_nodes and model.node_is_power[index]
    }
    probability_by_node = {
        node_id: float(1.0 - service_by_component[component_by_node[index]])
        for node_id, index in node_lookup.items()
    }
    component_by_id = {
        node_id: int(component_by_node[index])
        for node_id, index in node_lookup.items()
    }
    return ComponentState(probability_by_node, component_by_id)


def result_frame(
    hospital_ids: tuple[str, ...],
    samples: list[np.ndarray],
) -> pd.DataFrame:
    matrix = np.vstack(samples)
    evaluated = pd.DataFrame(matrix).where(matrix >= 0.0)
    evaluated_count = evaluated.notna().sum(axis=0).to_numpy()
    return pd.DataFrame(
        {
            "id": hospital_ids,
            "mean": evaluated.mean(axis=0).fillna(-1.0).to_numpy(),
            "std": evaluated.std(axis=0, ddof=1).to_numpy(),
            "sample_num": len(samples),
            "evaluated_sample_num": evaluated_count,
            "power_scenario": POWER_SCENARIO,
            "model_version": MODEL_VERSION,
        }
    )


def electricity_frame(
    model: HybridPowerModel,
    failure_sum: np.ndarray,
    samples: int,
) -> pd.DataFrame:
    failure = np.full(len(model.hospital_ids), -1.0, dtype=float)
    failure[model.hospital_modeled] = (
        failure_sum[model.hospital_modeled] / samples
    )
    return pd.DataFrame(
        {
            "hospital_id": model.hospital_ids,
            "modeled": model.hospital_modeled.astype(np.int8),
            "failure_probability": failure,
            "sample_num": samples,
            "power_scenario": POWER_SCENARIO,
            "model_version": MODEL_VERSION,
        }
    )


def simulate(args: argparse.Namespace) -> dict[str, object]:
    if args.samples < 1:
        raise ValueError("samples must be positive")
    model = HybridPowerModel(args.model)
    probabilities = load_probabilities(args.probabilities, model.total_edge_count)
    dependencies = (
        load_dependency_models(args.dependency_model)
        if args.dependency_model is not None
        else None
    )
    if dependencies is None and not args.write_electricity:
        raise ValueError("nothing selected: provide dependencies or write electricity")

    positive_global = np.flatnonzero(probabilities > 0.0)
    retained_lookup = np.full(model.total_edge_count, -1, dtype=np.int64)
    retained_lookup[model.edge_uid - 1] = np.arange(len(model.edge_uid))
    relevant_nodes = (
        dependencies.relevant_power_nodes if dependencies is not None else frozenset()
    )
    retained_relevant = relevant_nodes.intersection(model.node_ids)

    rng = np.random.RandomState(args.seed)
    intact = model.intact_state()
    hospital_failure_sum = np.zeros(len(model.hospital_ids), dtype=float)
    water_samples: list[np.ndarray] = []
    communication_samples: list[np.ndarray] = []
    failed_global_counts: list[int] = []
    failed_retained_counts: list[int] = []
    hospital_service: list[float] = []
    demand_service: list[float] = []
    flow_constrained: list[bool] = []

    for _ in range(args.samples):
        failed_global = positive_global[
            rng.uniform(0.0, 1.0, len(positive_global))
            < probabilities[positive_global]
        ]
        failed = np.zeros(len(model.edge_uid), dtype=bool)
        retained_failed = retained_lookup[failed_global]
        retained_failed = retained_failed[retained_failed >= 0]
        failed[retained_failed] = True
        state = model.solve_sample_state(failed) if np.any(failed) else intact
        result = state.result

        hospital_failure_sum[model.hospital_modeled] += result.hospital_failure[
            model.hospital_modeled
        ]
        failed_global_counts.append(len(failed_global))
        failed_retained_counts.append(result.failed_edges)
        hospital_service.append(result.hospital_served_fraction)
        demand_service.append(
            result.served_mw / result.demand_mw if result.demand_mw else 1.0
        )
        flow_constrained.append(result.flow_constrained_components > 0)

        if dependencies is not None:
            downstream_state = component_state(
                model,
                state.component_by_node,
                state.service_by_component,
                relevant_nodes,
            )
            water_samples.append(
                evaluate_water_dependencies(dependencies.water, downstream_state)
            )
            communication_samples.append(
                evaluate_communication_dependencies(
                    dependencies.communication, downstream_state
                )
            )

    outputs: dict[str, str] = {}
    if args.write_electricity:
        path = args.output_root / "electricity" / "tracks" / (
            f"electricity_cyclone_{args.storm_folder}_{args.track_id}_"
            f"{args.threshold}.csv"
        )
        atomic_dataframe(
            electricity_frame(model, hospital_failure_sum, args.samples), path
        )
        outputs["electricity"] = str(path)
    if dependencies is not None:
        for service, ids, values in (
            ("water", dependencies.water.hospital_ids, water_samples),
            (
                "comm",
                dependencies.communication.hospital_ids,
                communication_samples,
            ),
        ):
            path = args.output_root / service / "tracks" / (
                f"{service}_cyclone_{args.storm_folder}_{args.track_id}_"
                f"{args.threshold}.csv"
            )
            atomic_dataframe(result_frame(ids, values), path)
            outputs[service] = str(path)

    summary: dict[str, object] = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "power_scenario": POWER_SCENARIO,
        "storm_folder": args.storm_folder,
        "track_id": args.track_id,
        "threshold": args.threshold,
        "samples": args.samples,
        "random_seed": args.seed,
        "code_commit": os.environ.get("INFRACASCADE_CODE_COMMIT", ""),
        "inputs": {
            "model": str(args.model),
            "probabilities": str(args.probabilities),
            "dependency_model": (
                str(args.dependency_model)
                if args.dependency_model is not None
                else None
            ),
        },
        "cohorts": {
            "electricity_total": len(model.hospital_ids),
            "electricity_modeled": model.modeled_hospital_count,
            "water": (
                len(dependencies.water.hospital_ids)
                if dependencies is not None
                else 0
            ),
            "communication": (
                len(dependencies.communication.hospital_ids)
                if dependencies is not None
                else 0
            ),
            "relevant_dependency_power_nodes": len(relevant_nodes),
            "retained_dependency_power_nodes": len(retained_relevant),
        },
        "result": {
            "mean_failed_edges": float(np.mean(failed_global_counts)),
            "mean_failed_retained_edges": float(np.mean(failed_retained_counts)),
            "mean_hospital_failure_probability": float(
                1.0 - np.mean(hospital_service)
            ),
            "mean_unserved_demand_fraction": float(1.0 - np.mean(demand_service)),
            "samples_with_flow_constraint": float(np.mean(flow_constrained)),
        },
        "missing_data_policy": {
            "electricity": "-1 for the nine hospitals outside the retained active grid",
            "water": "-1 when any required upstream power node is unavailable",
            "communication": (
                "legacy redundant-tower product; an absent tower source is fail-closed"
            ),
        },
        "outputs": outputs,
    }
    summary_path = args.output_root / "summaries" / (
        f"summary_{args.storm_folder}_{args.track_id}_{args.threshold}.json"
    )
    atomic_json(summary, summary_path)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--dependency-model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--storm-folder", required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--write-electricity", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    summary = simulate(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
