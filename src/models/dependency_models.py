"""Stable serializable types for cross-infrastructure dependency models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ComponentState:
    """Failure probability and component identity for relevant power nodes."""

    probability_by_node: Mapping[str, float]
    component_by_node: Mapping[str, int]


@dataclass(frozen=True)
class WaterDependencyModel:
    hospital_ids: tuple[str, ...]
    upstream_power_nodes: tuple[Optional[tuple[str, ...]], ...]
    power_nodes: tuple[str, ...]


@dataclass(frozen=True)
class CommunicationDependencyModel:
    hospital_ids: tuple[str, ...]
    predecessor_towers: tuple[Optional[tuple[str, ...]], ...]
    tower_power_source: tuple[tuple[str, Optional[str]], ...]
    power_nodes: tuple[str, ...]


@dataclass(frozen=True)
class DependencyModels:
    version: str
    water: WaterDependencyModel
    communication: CommunicationDependencyModel

    @property
    def relevant_power_nodes(self) -> frozenset[str]:
        return frozenset(self.water.power_nodes).union(
            self.communication.power_nodes
        )
