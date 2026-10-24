#!/usr/bin/env python3
"""Connected-component cyclone model with DC-flow-informed service.

The model deliberately keeps the original cascade structure: sampled line
outages split the public grid into connected components and every modeled
hospital in a component receives the same service fraction.  It replaces the
historical 4 MW-per-hospital calculation with the source-backed MATPOWER bus
loads and generator Pmax values.  Within each component, a compact DC linear
program finds the largest uniformly served fraction while respecting generator
and branch limits.  The resulting flow coefficient is

    feasible service / capacity-only adequacy.

Hospitals outside the source-backed active grid are reported as -1 and are not
included in performance aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


BASE_MVA = 100.0
EPSILON = 1e-9


@dataclass(frozen=True)
class ComponentResult:
    demand_mw: float
    supply_mw: float
    capacity_adequacy: float
    flow_coefficient: float
    served_fraction: float
    has_power_bus: bool
    has_generation: bool
    flow_constrained: bool
    capacity_constrained: bool


@dataclass(frozen=True)
class SampleResult:
    hospital_failure: np.ndarray
    failed_edges: int
    components: int
    components_with_demand: int
    no_generation_components: int
    flow_constrained_components: int
    capacity_constrained_components: int
    demand_mw: float
    served_mw: float
    capacity_only_served_mw: float
    hospital_served_fraction: float


@dataclass(frozen=True)
class SampleState:
    """Full hybrid state for consumers that depend on grid nodes.

    ``result`` preserves the public hospital/output contract used by the
    original hybrid cyclone workflow.  The additional arrays expose the
    connected-component identity and DC-feasible service fraction assigned to
    every retained node, allowing water and communication dependencies to use
    the exact same Monte Carlo realization.
    """

    result: SampleResult
    component_by_node: np.ndarray
    service_by_component: np.ndarray

    @property
    def node_failure(self) -> np.ndarray:
        return 1.0 - self.service_by_component[self.component_by_node]


@dataclass(frozen=True)
class LengthFailures:
    """Length-weighted initial and partial-service transmission failures."""

    initial_km: float
    cascade_km: float


def _as_scalar(archive: dict[str, np.ndarray], name: str) -> int:
    values = archive.get(name)
    if values is None or values.size != 1:
        raise ValueError(f"hybrid model lacks scalar {name!r}")
    return int(values.reshape(-1)[0])


class HybridPowerModel:
    """Immutable topology and electrical parameters used by every track."""

    def __init__(self, model_path: Path):
        with np.load(model_path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        self.format_version = _as_scalar(data, "format_version")
        if self.format_version not in (1, 2):
            raise ValueError(f"unsupported hybrid model format: {model_path}")

        self.model_path = Path(model_path)
        self.node_ids = data["node_ids"].astype(str)
        self.node_is_power = data["node_is_power"].astype(bool)
        self.node_pd_mw = data["node_pd_mw"].astype(float)
        self.node_pmax_mw = data["node_pmax_mw"].astype(float)
        self.edge_uid = data["edge_uid"].astype(np.int64)
        self.edge_u = data["edge_u"].astype(np.int64)
        self.edge_v = data["edge_v"].astype(np.int64)
        self.edge_is_power = data["edge_is_power"].astype(bool)
        self.edge_x_pu = data["edge_x_pu"].astype(float)
        self.edge_rate_mw = data["edge_rate_mw"].astype(float)
        self.hospital_ids = data["hospital_ids"].astype(str)
        self.hospital_node = data["hospital_node"].astype(np.int64)
        self.hospital_modeled = data["hospital_modeled"].astype(bool)
        self.total_edge_count = _as_scalar(data, "total_edge_count")
        self.all_edge_length_km = data.get(
            "all_edge_length_km", np.asarray([], dtype=float)
        ).astype(float)
        self.all_edge_is_transmission = data.get(
            "all_edge_is_transmission", np.asarray([], dtype=bool)
        ).astype(bool)
        self.all_edge_is_ablation = data.get(
            "all_edge_is_ablation",
            np.zeros(self.total_edge_count, dtype=bool),
        ).astype(bool)
        self.edge_is_transmission = data.get(
            "edge_is_transmission", np.asarray([], dtype=bool)
        ).astype(bool)
        self.edge_length_km = data.get(
            "edge_length_km", np.asarray([], dtype=float)
        ).astype(float)

        node_count = len(self.node_ids)
        edge_count = len(self.edge_uid)
        for name, values in (
            ("node_is_power", self.node_is_power),
            ("node_pd_mw", self.node_pd_mw),
            ("node_pmax_mw", self.node_pmax_mw),
        ):
            if len(values) != node_count:
                raise ValueError(f"{name} length does not match node_ids")
        for name, values in (
            ("edge_u", self.edge_u),
            ("edge_v", self.edge_v),
            ("edge_is_power", self.edge_is_power),
            ("edge_x_pu", self.edge_x_pu),
            ("edge_rate_mw", self.edge_rate_mw),
        ):
            if len(values) != edge_count:
                raise ValueError(f"{name} length does not match edge_uid")
        if np.any(self.edge_u < 0) or np.any(self.edge_u >= node_count):
            raise ValueError("edge_u contains an invalid node index")
        if np.any(self.edge_v < 0) or np.any(self.edge_v >= node_count):
            raise ValueError("edge_v contains an invalid node index")
        if np.any(self.edge_uid < 1) or np.any(self.edge_uid > self.total_edge_count):
            raise ValueError("edge_uid is outside the probability-table range")
        if len(np.unique(self.edge_uid)) != edge_count:
            raise ValueError("edge_uid values are not unique")
        if self.format_version >= 2:
            if len(self.all_edge_length_km) != self.total_edge_count:
                raise ValueError(
                    "all_edge_length_km length does not match total edges"
                )
            if len(self.all_edge_is_transmission) != self.total_edge_count:
                raise ValueError(
                    "all_edge_is_transmission length does not match total edges"
                )
            if len(self.all_edge_is_ablation) != self.total_edge_count:
                raise ValueError(
                    "all_edge_is_ablation length does not match total edges"
                )
            if len(self.edge_is_transmission) != edge_count:
                raise ValueError("edge_is_transmission length does not match edges")
            if len(self.edge_length_km) != edge_count:
                raise ValueError("edge_length_km length does not match edges")
            if np.any(~np.isfinite(self.all_edge_length_km)) or np.any(
                self.all_edge_length_km < 0.0
            ):
                raise ValueError("all edge lengths must be finite and nonnegative")
            if np.any(~np.isfinite(self.edge_length_km)) or np.any(
                self.edge_length_km < 0.0
            ):
                raise ValueError(
                    "retained edge lengths must be finite and nonnegative"
                )
        if np.any(self.edge_x_pu[self.edge_is_power] <= 0):
            raise ValueError("power branches must have positive reactance")
        if np.any(self.edge_rate_mw[self.edge_is_power] <= 0):
            raise ValueError("power branches must have positive ratings")
        if len(self.hospital_ids) != len(self.hospital_node):
            raise ValueError("hospital ID and node-index lengths differ")
        if len(self.hospital_ids) != len(self.hospital_modeled):
            raise ValueError("hospital modeled flags have the wrong length")
        if np.any(self.hospital_node[self.hospital_modeled] < 0):
            raise ValueError("a modeled hospital lacks a retained node")
        if np.any(self.hospital_node[self.hospital_modeled] >= node_count):
            raise ValueError("a modeled hospital has an invalid node index")

        self.modeled_hospital_count = int(np.count_nonzero(self.hospital_modeled))
        self.total_demand_mw = float(self.node_pd_mw.sum())
        self.total_pmax_mw = float(self.node_pmax_mw.sum())

    def intact_service_fraction(self) -> float:
        """Return the minimum modeled-hospital service in the intact network."""

        result = self.solve_sample(np.zeros(len(self.edge_uid), dtype=bool))
        modeled = result.hospital_failure[self.hospital_modeled]
        return float(1.0 - modeled.max(initial=0.0))

    def intact_sample(self) -> SampleResult:
        """Return the setup-verified intact state without resolving its LP."""

        hospital_failure = np.full(len(self.hospital_ids), -1.0, dtype=float)
        hospital_failure[self.hospital_modeled] = 0.0
        return SampleResult(
            hospital_failure=hospital_failure,
            failed_edges=0,
            components=1,
            components_with_demand=1,
            no_generation_components=0,
            flow_constrained_components=0,
            capacity_constrained_components=0,
            demand_mw=self.total_demand_mw,
            served_mw=self.total_demand_mw,
            capacity_only_served_mw=self.total_demand_mw,
            hospital_served_fraction=1.0,
        )

    def intact_state(self) -> SampleState:
        """Return the verified intact state, including retained-node service."""

        return SampleState(
            result=self.intact_sample(),
            component_by_node=np.zeros(len(self.node_ids), dtype=np.int64),
            service_by_component=np.ones(1, dtype=float),
        )

    def _component_service(
        self,
        nodes: np.ndarray,
        power_edge_indices: np.ndarray,
    ) -> ComponentResult:
        power_nodes = nodes[self.node_is_power[nodes]]
        has_power_bus = bool(len(power_nodes))
        demand = float(self.node_pd_mw[power_nodes].sum())
        generator_nodes = power_nodes[self.node_pmax_mw[power_nodes] > EPSILON]
        pmax = self.node_pmax_mw[generator_nodes]
        supply = float(pmax.sum())
        has_generation = supply > EPSILON

        if not has_power_bus:
            return ComponentResult(
                demand_mw=0.0,
                supply_mw=0.0,
                capacity_adequacy=0.0,
                flow_coefficient=0.0,
                served_fraction=0.0,
                has_power_bus=False,
                has_generation=False,
                flow_constrained=False,
                capacity_constrained=False,
            )
        if demand <= EPSILON:
            service = 1.0 if has_generation else 0.0
            return ComponentResult(
                demand_mw=demand,
                supply_mw=supply,
                capacity_adequacy=service,
                flow_coefficient=1.0 if service else 0.0,
                served_fraction=service,
                has_power_bus=True,
                has_generation=has_generation,
                flow_constrained=False,
                capacity_constrained=False,
            )

        adequacy = min(1.0, max(0.0, supply / demand))
        if not has_generation:
            return ComponentResult(
                demand_mw=demand,
                supply_mw=0.0,
                capacity_adequacy=0.0,
                flow_coefficient=0.0,
                served_fraction=0.0,
                has_power_bus=True,
                has_generation=False,
                flow_constrained=False,
                capacity_constrained=True,
            )

        # A one-bus island has no transmission limit.  Service is determined
        # solely by the source-backed capacity-to-demand ratio.
        if len(power_nodes) == 1 or len(power_edge_indices) == 0:
            return ComponentResult(
                demand_mw=demand,
                supply_mw=supply,
                capacity_adequacy=adequacy,
                flow_coefficient=1.0,
                served_fraction=adequacy,
                has_power_bus=True,
                has_generation=True,
                flow_constrained=False,
                capacity_constrained=adequacy < 1.0 - 1e-8,
            )

        local_index = {int(node): index for index, node in enumerate(power_nodes)}
        local_u = np.asarray(
            [local_index[int(self.edge_u[index])] for index in power_edge_indices],
            dtype=np.int64,
        )
        local_v = np.asarray(
            [local_index[int(self.edge_v[index])] for index in power_edge_indices],
            dtype=np.int64,
        )
        branch_count = len(power_edge_indices)
        rows = np.repeat(np.arange(branch_count, dtype=np.int64), 2)
        columns = np.column_stack((local_u, local_v)).reshape(-1)
        values = np.tile(np.asarray([1.0, -1.0]), branch_count)
        incidence = coo_matrix(
            (values, (rows, columns)),
            shape=(branch_count, len(power_nodes)),
        ).tocsr()
        susceptance = BASE_MVA / self.edge_x_pu[power_edge_indices]
        branch_flow = diags(susceptance) @ incidence
        bbus = (incidence.T @ branch_flow).tocsc()

        # Eliminate one voltage angle.  Transfer sensitivities are then found
        # for each generator and for the full component demand vector.  The
        # remaining LP has at most 14 variables for the Mozambique case.
        reference = 0
        keep = np.arange(1, len(power_nodes), dtype=np.int64)
        reduced = bbus[keep][:, keep]
        rhs = np.zeros((len(power_nodes) - 1, len(generator_nodes) + 1))
        for column, node in enumerate(generator_nodes):
            local = local_index[int(node)]
            if local != reference:
                rhs[local - 1, column] = 1.0
        local_demand = self.node_pd_mw[power_nodes]
        rhs[:, -1] = -local_demand[keep]
        try:
            solved = splu(reduced).solve(rhs)
        except RuntimeError as exc:
            raise RuntimeError(
                f"singular DC component with {len(power_nodes)} buses and "
                f"{branch_count} branches"
            ) from exc
        theta = np.zeros((len(power_nodes), len(generator_nodes) + 1))
        theta[keep, :] = solved
        sensitivities = np.asarray(branch_flow @ theta)

        # Variables are generator dispatches followed by uniform service
        # fraction lambda.  Maximize lambda subject to nodal DC flow, branch
        # ratings, and Pmax.  Total balance supplies the omitted reference-bus
        # nodal equation.
        flow_matrix = sensitivities
        a_ub = np.vstack((flow_matrix, -flow_matrix))
        rates = self.edge_rate_mw[power_edge_indices]
        b_ub = np.concatenate((rates, rates))
        a_eq = np.concatenate((np.ones(len(generator_nodes)), [-demand]))[None, :]
        objective = np.zeros(len(generator_nodes) + 1)
        objective[-1] = -1.0
        bounds = [(0.0, float(value)) for value in pmax]
        bounds.append((0.0, 1.0))
        solution = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=np.asarray([0.0]),
            bounds=bounds,
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(
                "hybrid DC service optimization failed: " + solution.message
            )
        service = min(1.0, max(0.0, float(solution.x[-1])))
        # Numerical tolerance may place lambda a few ulps above the analytical
        # capacity ceiling.
        service = min(service, adequacy)
        coefficient = service / adequacy if adequacy > EPSILON else 0.0
        return ComponentResult(
            demand_mw=demand,
            supply_mw=supply,
            capacity_adequacy=adequacy,
            flow_coefficient=min(1.0, max(0.0, coefficient)),
            served_fraction=service,
            has_power_bus=True,
            has_generation=True,
            flow_constrained=service < adequacy - 1e-7,
            capacity_constrained=adequacy < 1.0 - 1e-8,
        )

    def solve_sample_state(self, failed_edges: np.ndarray) -> SampleState:
        """Solve one outage realization and retain its node-level state."""

        failed_edges = np.asarray(failed_edges, dtype=bool)
        if failed_edges.shape != (len(self.edge_uid),):
            raise ValueError("failed_edges has the wrong shape")
        surviving = ~failed_edges
        rows = np.concatenate((self.edge_u[surviving], self.edge_v[surviving]))
        columns = np.concatenate((self.edge_v[surviving], self.edge_u[surviving]))
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(len(self.node_ids), len(self.node_ids)),
        ).tocsr()
        component_count, labels = connected_components(
            adjacency, directed=False, return_labels=True
        )

        nodes_by_component: list[list[int]] = [[] for _ in range(component_count)]
        for node, label in enumerate(labels):
            nodes_by_component[int(label)].append(node)
        power_edges_by_component: list[list[int]] = [
            [] for _ in range(component_count)
        ]
        for edge in np.flatnonzero(surviving & self.edge_is_power):
            label = int(labels[self.edge_u[edge]])
            if labels[self.edge_v[edge]] != label:
                raise RuntimeError("surviving edge endpoints have different labels")
            power_edges_by_component[label].append(int(edge))

        component_results: list[ComponentResult] = []
        service_by_component = np.zeros(component_count, dtype=float)
        for label in range(component_count):
            result = self._component_service(
                np.asarray(nodes_by_component[label], dtype=np.int64),
                np.asarray(power_edges_by_component[label], dtype=np.int64),
            )
            component_results.append(result)
            service_by_component[label] = result.served_fraction

        hospital_failure = np.full(len(self.hospital_ids), -1.0, dtype=float)
        modeled_nodes = self.hospital_node[self.hospital_modeled]
        hospital_failure[self.hospital_modeled] = (
            1.0 - service_by_component[labels[modeled_nodes]]
        )
        modeled_failure = hospital_failure[self.hospital_modeled]
        hospital_service = 1.0 - float(modeled_failure.mean())
        demand = sum(result.demand_mw for result in component_results)
        served = sum(
            result.demand_mw * result.served_fraction
            for result in component_results
        )
        capacity_only_served = sum(
            result.demand_mw * result.capacity_adequacy
            for result in component_results
        )
        result = SampleResult(
            hospital_failure=hospital_failure,
            failed_edges=int(np.count_nonzero(failed_edges)),
            components=component_count,
            components_with_demand=sum(
                result.demand_mw > EPSILON for result in component_results
            ),
            no_generation_components=sum(
                result.demand_mw > EPSILON and not result.has_generation
                for result in component_results
            ),
            flow_constrained_components=sum(
                result.flow_constrained for result in component_results
            ),
            capacity_constrained_components=sum(
                result.capacity_constrained for result in component_results
            ),
            demand_mw=float(demand),
            served_mw=float(served),
            capacity_only_served_mw=float(capacity_only_served),
            hospital_served_fraction=hospital_service,
        )
        return SampleState(
            result=result,
            component_by_node=labels.astype(np.int64, copy=False),
            service_by_component=service_by_component,
        )

    def solve_sample(self, failed_edges: np.ndarray) -> SampleResult:
        """Solve one outage realization using the historical result contract."""

        return self.solve_sample_state(failed_edges).result

    def length_failures(
        self,
        failed_global: np.ndarray,
        failed_retained: np.ndarray,
        state: SampleState,
    ) -> LengthFailures:
        """Return direct removal and downstream unpowered line lengths.

        Initial failures are sampled removals plus surviving transmission lines
        in completely unpowered components. Cascade failures exclude both of
        those groups. For each line in a partially served component, its
        unserved fraction is multiplied by its length.
        """

        failed_global = np.asarray(failed_global, dtype=np.int64)
        failed_retained = np.asarray(failed_retained, dtype=bool)
        if self.format_version < 2:
            raise ValueError("hybrid model does not contain edge-length metadata")
        if failed_retained.shape != (len(self.edge_uid),):
            raise ValueError("failed_retained has the wrong shape")
        if np.any(failed_global < 0) or np.any(
            failed_global >= self.total_edge_count
        ):
            raise ValueError("failed_global contains an invalid edge index")

        initial = float(
            self.all_edge_length_km[failed_global][
                self.all_edge_is_transmission[failed_global]
            ].sum()
        )
        surviving_transmission = (~failed_retained) & self.edge_is_transmission
        edge_indices = np.flatnonzero(surviving_transmission)
        if not len(edge_indices):
            return LengthFailures(initial_km=initial, cascade_km=0.0)
        labels = state.component_by_node[self.edge_u[edge_indices]]
        if np.any(
            state.component_by_node[self.edge_v[edge_indices]] != labels
        ):
            raise RuntimeError("surviving edge endpoints have different components")
        service = state.service_by_component[labels]
        isolated = service <= EPSILON
        initial += float(self.edge_length_km[edge_indices[isolated]].sum())
        partial = service > EPSILON
        unserved_fraction = 1.0 - service[partial]
        cascade = float(
            np.sum(self.edge_length_km[edge_indices[partial]] * unserved_fraction)
        )
        return LengthFailures(initial_km=initial, cascade_km=cascade)


def load_probabilities(path: Path, expected: int) -> np.ndarray:
    probabilities = np.full(expected, np.nan, dtype=float)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        id_column = "edge_uid" if "edge_uid" in (reader.fieldnames or []) else "branch_uid"
        if id_column not in (reader.fieldnames or []) or "failure_probability" not in (
            reader.fieldnames or []
        ):
            raise ValueError(
                "probability CSV must contain edge_uid/branch_uid and "
                "failure_probability"
            )
        for row in reader:
            uid = int(row[id_column])
            if uid < 1 or uid > expected or math.isfinite(probabilities[uid - 1]):
                raise ValueError(f"invalid or duplicate probability ID {uid}")
            probabilities[uid - 1] = float(row["failure_probability"])
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("probability CSV does not cover every topology edge")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("failure probabilities must be within [0, 1]")
    return probabilities


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, object]) -> None:
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


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def simulate_track(args: argparse.Namespace) -> dict[str, object]:
    if args.samples < 1:
        raise ValueError("samples must be positive")
    model = HybridPowerModel(args.model)
    probabilities = load_probabilities(args.probabilities, model.total_edge_count)
    retained_probabilities = probabilities[model.edge_uid - 1]
    positive_global = np.flatnonzero(probabilities > 0.0)
    shared_positive = positive_global[~model.all_edge_is_ablation[positive_global]]
    ablation_positive = positive_global[model.all_edge_is_ablation[positive_global]]
    retained_lookup = np.full(model.total_edge_count, -1, dtype=np.int64)
    retained_lookup[model.edge_uid - 1] = np.arange(len(model.edge_uid))

    shared_rng = np.random.RandomState(args.seed)
    ablation_rng = np.random.RandomState(args.seed + 1)
    intact_result = model.intact_sample()
    hospital_failure_sum = np.zeros(len(model.hospital_ids), dtype=float)
    sample_rows: list[dict[str, object]] = []
    hospital_service: list[float] = []
    served_fraction: list[float] = []
    flow_coefficients: list[float] = []
    for sample_id in range(1, args.samples + 1):
        failed_shared = shared_positive[
            shared_rng.uniform(0.0, 1.0, len(shared_positive))
            < probabilities[shared_positive]
        ]
        failed_ablation = ablation_positive[
            ablation_rng.uniform(0.0, 1.0, len(ablation_positive))
            < probabilities[ablation_positive]
        ]
        failed_global = np.concatenate((failed_shared, failed_ablation))
        failed = np.zeros(len(model.edge_uid), dtype=bool)
        retained_failed = retained_lookup[failed_global]
        retained_failed = retained_failed[retained_failed >= 0]
        failed[retained_failed] = True
        result = model.solve_sample(failed) if np.any(failed) else intact_result
        hospital_failure_sum[model.hospital_modeled] += result.hospital_failure[
            model.hospital_modeled
        ]
        demand_service = (
            result.served_mw / result.demand_mw if result.demand_mw > EPSILON else 1.0
        )
        capacity_service = (
            result.capacity_only_served_mw / result.demand_mw
            if result.demand_mw > EPSILON
            else 1.0
        )
        flow_coefficient = (
            demand_service / capacity_service
            if capacity_service > EPSILON
            else 0.0
        )
        hospital_service.append(result.hospital_served_fraction)
        served_fraction.append(demand_service)
        flow_coefficients.append(flow_coefficient)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "failed_edges": len(failed_global),
                "failed_retained_edges": result.failed_edges,
                "components": result.components,
                "components_with_demand": result.components_with_demand,
                "no_generation_components": result.no_generation_components,
                "flow_constrained_components": result.flow_constrained_components,
                "capacity_constrained_components": result.capacity_constrained_components,
                "demand_mw": f"{result.demand_mw:.10f}",
                "served_mw": f"{result.served_mw:.10f}",
                "capacity_only_served_mw": f"{result.capacity_only_served_mw:.10f}",
                "served_fraction": f"{demand_service:.12f}",
                "flow_coefficient": f"{flow_coefficient:.12f}",
                "hospital_served_fraction": f"{result.hospital_served_fraction:.12f}",
            }
        )

    final_failure = np.full(len(model.hospital_ids), -1.0, dtype=float)
    final_failure[model.hospital_modeled] = (
        hospital_failure_sum[model.hospital_modeled] / args.samples
    )
    hospital_rows = [
        {
            "hospital_id": hospital_id,
            "modeled": int(modeled),
            "failure_probability": f"{failure:.12f}",
        }
        for hospital_id, modeled, failure in zip(
            model.hospital_ids, model.hospital_modeled, final_failure
        )
    ]
    sample_fields = list(sample_rows[0])
    atomic_csv(args.sample_output, sample_fields, sample_rows)
    atomic_csv(
        args.hospital_output,
        ["hospital_id", "modeled", "failure_probability"],
        hospital_rows,
    )

    hospital_service_array = np.asarray(hospital_service)
    served_array = np.asarray(served_fraction)
    coefficient_array = np.asarray(flow_coefficients)
    summary: dict[str, object] = {
        "schema_version": 1,
        "model": "connected_components_with_uniform_dc_service",
        "storm_folder": args.storm_folder,
        "track_id": args.track_id,
        "samples": args.samples,
        "random_seed": args.seed,
        "ablation_random_seed": args.seed + 1,
        "code_commit": os.environ.get("INFRACASCADE_CODE_COMMIT", ""),
        "inputs": {
            "model": str(args.model),
            "probabilities": str(args.probabilities),
            "hazard": "line failure only; original positive-edge sampling order",
        },
        "base": {
            "modeled_hospitals": model.modeled_hospital_count,
            "unmodeled_hospitals": int(len(model.hospital_ids) - model.modeled_hospital_count),
            "load_mw": model.total_demand_mw,
            "generator_pmax_mw": model.total_pmax_mw,
            # The setup job explicitly solves and rejects an infeasible intact
            # model before any track workers are released.
            "intact_minimum_hospital_service": 1.0,
        },
        "hazard": {
            "edges": model.total_edge_count,
            "retained_edges": len(model.edge_uid),
            "edges_with_positive_probability": int(len(positive_global)),
            "shared_edges_with_positive_probability": int(len(shared_positive)),
            "ablation_edges_with_positive_probability": int(len(ablation_positive)),
            "expected_failed_edges": float(probabilities.sum()),
            "maximum_failure_probability": float(probabilities.max(initial=0.0)),
        },
        "result": {
            "mean_failed_edges": float(
                np.mean([int(row["failed_edges"]) for row in sample_rows])
            ),
            "probability_any_edge_failure": float(
                np.mean([int(row["failed_edges"]) > 0 for row in sample_rows])
            ),
            "mean_served_fraction": float(served_array.mean()),
            "mean_flow_coefficient": float(coefficient_array.mean()),
            "mean_hospital_served_fraction": float(hospital_service_array.mean()),
            "mean_hospital_failure_probability": float(
                1.0 - hospital_service_array.mean()
            ),
            "median_hospital_failure_probability": float(
                1.0 - percentile(hospital_service_array, 0.50)
            ),
            "p95_hospital_failure_probability": percentile(
                1.0 - hospital_service_array, 0.95
            ),
            "probability_any_hospital_disruption": float(
                np.mean(hospital_service_array < 1.0 - 1e-9)
            ),
            "samples_with_flow_constraint": float(
                np.mean(
                    [int(row["flow_constrained_components"]) > 0 for row in sample_rows]
                )
            ),
        },
        "outputs": {
            "samples": str(args.sample_output),
            "hospitals": str(args.hospital_output),
        },
    }
    atomic_json(args.summary_output, summary)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--storm-folder", required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--hospital-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    summary = simulate_track(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
