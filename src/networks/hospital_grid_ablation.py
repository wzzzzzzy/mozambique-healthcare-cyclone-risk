"""Helpers for the counterfactual in which solar hospitals use the public grid."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Hashable, Iterable, List, Tuple

import numpy as np
from shapely.geometry import mapping, shape


EARTH_RADIUS_KM = 6371.0088
MAX_EXISTING_HOSPITAL_OFFSET_KM = 0.1
GRID_CONNECTION_TYPES = frozenset(
    {
        "plant",
        "generator",
        "substation",
        "transformer",
        "transition",
        "portal",
        "gridfinder",
    }
)


@dataclass(frozen=True)
class GridAssignment:
    """A hospital's connection to the public-grid graph."""

    hospital_node_id: Hashable
    grid_node_id: Hashable
    distance_km: float
    source: str


def hospital_node_id(index):
    """Return the graph node identifier used for a SARA dataframe index."""

    return f"node/46774825{str(index).zfill(4)}"


def _as_point(geometry):
    if geometry is None:
        raise ValueError("Cannot assign a hospital without geometry to the grid")

    geom = shape(geometry) if isinstance(geometry, Mapping) else geometry
    if getattr(geom, "is_empty", False):
        raise ValueError("Cannot assign a hospital with empty geometry to the grid")

    point = geom if getattr(geom, "geom_type", None) == "Point" else geom.centroid
    if not (math.isfinite(point.x) and math.isfinite(point.y)):
        raise ValueError("Hospital/grid geometry contains non-finite coordinates")
    return point


def _great_circle_distances_km(lon, lat, candidate_lons, candidate_lats):
    """Vectorized haversine distance from one WGS84 point to candidate points."""

    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    dlon = candidate_lons - lon_rad
    dlat = candidate_lats - lat_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + math.cos(lat_rad) * np.cos(candidate_lats) * np.sin(dlon / 2.0) ** 2
    )
    angular_distance = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * angular_distance


class NearestGridNodeIndex:
    """Exact nearest-node lookup over connectable nodes in a power graph."""

    def __init__(
        self,
        graph,
        allowed_power_types=GRID_CONNECTION_TYPES,
        allowed_node_ids=None,
    ):
        node_ids = []
        lons = []
        lats = []
        allowed_ids = (
            None
            if allowed_node_ids is None
            else {str(node_id) for node_id in allowed_node_ids}
        )

        candidates = sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
        for node_id, data in candidates:
            if allowed_ids is not None and str(node_id) not in allowed_ids:
                continue
            if data.get("power") not in allowed_power_types:
                continue
            try:
                point = _as_point(data.get("geometry"))
            except (AttributeError, TypeError, ValueError):
                continue
            node_ids.append(node_id)
            lons.append(point.x)
            lats.append(point.y)

        if not node_ids:
            raise ValueError("The power graph contains no geocoded connectable nodes")

        self.node_ids = node_ids
        self.lons = np.radians(np.asarray(lons, dtype=float))
        self.lats = np.radians(np.asarray(lats, dtype=float))

    def nearest(self, geometry):
        point = _as_point(geometry)
        distances = _great_circle_distances_km(
            point.x, point.y, self.lons, self.lats
        )
        nearest_index = int(np.argmin(distances))
        return self.node_ids[nearest_index], float(distances[nearest_index])


def _existing_grid_connection(graph, hospital_id, geometry):
    grid_neighbors = [
        node_id
        for node_id in graph.neighbors(hospital_id)
        if graph.nodes[node_id].get("power") != "hospital"
    ]
    if not grid_neighbors:
        raise ValueError(
            f"Existing hospital node {hospital_id!r} has no public-grid neighbor"
        )

    # Existing hospital nodes are expected to have degree one. If an older graph
    # has more than one service edge, use the geographically closest neighbor.
    hospital_point = _as_point(geometry)
    distances = []
    for node_id in grid_neighbors:
        grid_point = _as_point(graph.nodes[node_id].get("geometry"))
        distance = _great_circle_distances_km(
            hospital_point.x,
            hospital_point.y,
            np.asarray([math.radians(grid_point.x)]),
            np.asarray([math.radians(grid_point.y)]),
        )[0]
        distances.append(float(distance))
    nearest_index = int(np.argmin(distances))
    return grid_neighbors[nearest_index], distances[nearest_index]


def _distance_between(geometry_a, geometry_b):
    point_a = _as_point(geometry_a)
    point_b = _as_point(geometry_b)
    return float(
        _great_circle_distances_km(
            point_a.x,
            point_a.y,
            np.asarray([math.radians(point_b.x)]),
            np.asarray([math.radians(point_b.y)]),
        )[0]
    )


def add_assigned_hospitals_to_reference_graph(
    graph,
    hospital_records: Iterable[Tuple[Hashable, object]],
    assignments: Iterable[GridAssignment],
    copy_graph=True,
):
    """Mirror newly assigned hospital service edges into a path graph.

    The wind workflow uses this to put each new hospital edge into the full
    network before calculating failure attributes for the corresponding edge
    in the simplified graph. The initial failure values are placeholders and
    are overwritten by ``add_edge_attribute_for_simplified_network``.
    """

    records_by_id = dict(hospital_records)
    assignment_list = list(assignments)
    if len(records_by_id) != len(assignment_list):
        raise ValueError("Hospital records and assignments must be one-to-one")

    working_graph = graph.copy() if copy_graph else graph
    for assignment in assignment_list:
        if assignment.source != "ablation":
            continue
        hospital_id = assignment.hospital_node_id
        grid_node_id = assignment.grid_node_id
        if hospital_id not in records_by_id:
            raise ValueError(f"Missing geometry for hospital {hospital_id!r}")
        if hospital_id in working_graph:
            raise ValueError(
                f"New hospital id {hospital_id!r} already exists in reference graph"
            )
        if grid_node_id not in working_graph:
            raise ValueError(
                f"Assigned grid node {grid_node_id!r} is missing from reference graph"
            )

        point = _as_point(records_by_id[hospital_id])
        working_graph.add_node(
            hospital_id,
            geometry=mapping(point),
            properties={"power": "hospital"},
            power="hospital",
        )
        working_graph.add_edge(
            hospital_id,
            grid_node_id,
            attribute={"power": "MG-hospital"},
            ablation=True,
            length=assignment.distance_km,
            line_fail=0.0,
            tower_fail=0.0,
        )

    return working_graph


def attach_hospitals_to_nearest_grid(
    graph,
    hospital_records: Iterable[Tuple[Hashable, object]],
    copy_graph=True,
    candidate_grid_node_ids=None,
):
    """Attach hospitals missing from ``graph`` to their closest grid node.

    Existing public-grid hospital nodes keep their original service connection.
    Missing (solar-only) hospitals receive a conceptual service edge. The wind
    workflow subsequently assigns that edge the same event-specific fragility
    calculation used for existing public-hospital service edges. The hospitals
    inherit their connected component's outage probability and count toward its
    hospital demand.
    """

    records = list(hospital_records)
    working_graph = graph.copy() if copy_graph else graph
    grid_index = NearestGridNodeIndex(
        working_graph,
        allowed_node_ids=candidate_grid_node_ids,
    )
    assignments: List[GridAssignment] = []
    seen_hospital_ids = set()

    for hospital_id, geometry in records:
        if hospital_id in seen_hospital_ids:
            raise ValueError(f"Duplicate hospital node id: {hospital_id!r}")
        seen_hospital_ids.add(hospital_id)
        point = _as_point(geometry)

        if hospital_id in working_graph:
            if working_graph.nodes[hospital_id].get("power") != "hospital":
                raise ValueError(
                    f"Hospital id {hospital_id!r} collides with a non-hospital node"
                )
            offset_km = _distance_between(
                point,
                working_graph.nodes[hospital_id].get("geometry"),
            )
            if offset_km > MAX_EXISTING_HOSPITAL_OFFSET_KM:
                raise ValueError(
                    f"SARA geometry for {hospital_id!r} is {offset_km:.3f} km "
                    "from the existing graph node; row identity may have changed"
                )
            grid_node_id, distance_km = _existing_grid_connection(
                working_graph, hospital_id, point
            )
            source = "existing"
        else:
            grid_node_id, distance_km = grid_index.nearest(point)
            working_graph.add_node(
                hospital_id,
                geometry=mapping(point),
                properties={"power": "hospital"},
                power="hospital",
            )
            working_graph.add_edge(
                hospital_id,
                grid_node_id,
                attribute={"power": "MG-hospital"},
                ablation=True,
                length=distance_km,
                line_fail=0.0,
                tower_fail=0.0,
            )
            source = "ablation"

        assignments.append(
            GridAssignment(
                hospital_node_id=hospital_id,
                grid_node_id=grid_node_id,
                distance_km=distance_km,
                source=source,
            )
        )

    return working_graph, assignments
