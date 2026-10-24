#!/usr/bin/env python3
"""Build a public drinking-water dependency graph from OpenStreetMap features.

Replaces the legacy graph, which used ``man_made=wastewater_plant`` features as
upstream suppliers and attached every facility to its nearest water tower
without a distance limit.  Here:

* only drinking-water supply features enter the graph: ``water_works``,
  public or unknown-operator storage (towers, covered reservoirs, water tanks),
  non-sewage pumping stations, utility or powered boreholes, and
  utility-operated standpipes.  Wastewater plants are excluded and audited;
* facilities are grouped into supply systems by mapped water pipelines or by
  proximity, and every link records which evidence created it;
* only pumping elements depend on the grid (treatment works, pumping stations,
  powered boreholes).  Storage gets an implied pump only when its system has
  no mapped pumping element.  Systems without any source or pump get no power
  dependency, so their facilities stay unevaluated downstream;
* a healthcare facility is attached only when a system element lies within
  the service distance; otherwise it stays unevaluated.

The output graph keeps the legacy schema read by
``propagate_cyclone_dependencies.build_dependency_models``: a directed graph
whose nodes carry a ``water`` type, grid nodes typed by their power class, and
healthcare facilities keyed by ``hospital_node_id``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping
from scipy.spatial import cKDTree

from hospital_grid_ablation import hospital_node_id

METRIC_CRS = "EPSG:32736"
GRID_TYPES = ("gridfinder", "portal", "substation", "transformer")
PIPED_UTILITY = re.compile(
    r"fipag|\bfpag\b|[aá]guas? d[aeo]s?\b|adrm|\baias\b|conselho municipal|munic[ií]p|"
    r"abastecimento|\bara[ -]?(centro|sul|norte|zambeze)\b",
    re.IGNORECASE,
)
PUBLIC_OTHER = re.compile(r"\bestado\b|governo|governamental|pronasar|distrit", re.IGNORECASE)
# Operator values that describe the asset rather than name who runs it.
NOT_AN_OPERATOR = re.compile(r"^(n/?a|unknown|tanque|tanque e painel|carta)$", re.IGNORECASE)
WATER_WORDS = re.compile(r"[aá]gua|water|reservat|fipag", re.IGNORECASE)
SEWAGE = re.compile(r"sewage|wastewater|esgoto|residua", re.IGNORECASE)
POWERED_PUMP = {"powered", "electric", "electrical", "motor", "submersible", "diesel", "solar"}
SERVICE_SENSITIVITY_KM = (1, 2, 3, 5, 10)

# Name-based reclassifications of OSM features whose tag contradicts their name.
OVERRIDES = {
    "way/327367822": (
        "water_works",
        "named 'FPAG Mutua': FIPAG's Mutua treatment plant on the Pungwe that supplies "
        "Beira and Dondo; tagged man_made=wastewater_plant in OSM",
    ),
    "way/389993901": (
        "water_works",
        "likely the Chicamba treatment plant of FIPAG's Chimoio-Manica-Gondola system: 2.2 km "
        "from the Chicamba Dam intake, settling basins and a clear-water tank in imagery, no "
        "sewered town nearby; tagged man_made=wastewater_plant in OSM",
    ),
}


def ownership(name: str | None, operator: str | None) -> str:
    text = " ".join(filter(None, (name, operator)))
    if PIPED_UTILITY.search(text):
        return "piped_utility"
    if PUBLIC_OTHER.search(text):
        return "public_other"
    if operator and not NOT_AN_OPERATOR.match(operator.strip()):
        return "other_operator"
    return "unknown"


def classify(row: pd.Series, use_overrides: bool) -> dict[str, object]:
    tags = json.loads(row["tags"])
    man_made, amenity = tags.get("man_made"), tags.get("amenity")
    owner = ownership(tags.get("name"), tags.get("operator"))
    out = {"ownership": owner, "kind": None, "powered": False, "include": False, "reason": ""}

    if use_overrides and row["osm_id"] in OVERRIDES:
        kind, reason = OVERRIDES[row["osm_id"]]
        out.update(kind=kind, powered=True, include=True, reason=f"override: {reason}")
        return out
    if man_made == "wastewater_plant":
        out.update(kind="wastewater_plant", reason="sewage treatment, not drinking-water supply")
    elif man_made == "water_works":
        out.update(kind="water_works", powered=True, include=owner != "other_operator",
                   reason="drinking-water treatment works" if owner != "other_operator"
                   else "water works run by a non-utility operator")
    elif man_made in ("water_tower", "reservoir_covered"):
        include = owner != "other_operator"
        out.update(kind=man_made, include=include,
                   reason="public or unknown-operator storage" if include
                   else f"storage run by '{tags.get('operator')}' (private/NGO)")
    elif man_made in ("storage_tank", "tank"):
        is_water = tags.get("content") == "water" or tags.get("substance") == "water" or bool(
            WATER_WORDS.search(" ".join(filter(None, (tags.get("name"), tags.get("operator"))))))
        include = is_water and owner != "other_operator"
        out.update(kind="water_storage" if is_water else "tank_other", include=include,
                   reason="water storage tank" if include
                   else ("tank content not water or not tagged" if not is_water
                         else f"tank run by '{tags.get('operator')}' (private/NGO)"))
    elif man_made == "pumping_station":
        sewage = bool(SEWAGE.search(" ".join(str(tags.get(k, "")) for k in
                                             ("substance", "pumping_station", "name"))))
        include = not sewage and owner != "other_operator"
        out.update(kind="pumping_station", powered=True, include=include,
                   reason="pumping station (substance not tagged; assumed water)" if include
                   else "sewage or non-utility pumping station")
    elif man_made == "water_well":
        powered = str(tags.get("pump", "")).lower() in POWERED_PUMP
        include = owner == "piped_utility" or powered
        out.update(kind="borehole", powered=powered, include=include,
                   reason="piped-utility or powered borehole" if include
                   else "community or hand-pump well, not part of a piped system")
    elif man_made == "water_tap" or amenity in ("water_point", "drinking_water"):
        include = owner == "piped_utility"
        out.update(kind="standpipe", include=include,
                   reason="piped-utility standpipe (evidence of piped distribution)" if include
                   else "tap or water point without a piped-water utility operator")
    elif tags.get("office") == "water_utility":
        out.update(kind="utility_office", reason="office, not infrastructure")
    else:
        out.update(kind="raw_water_reservoir", reason="raw-water reservoir, not a supply node")
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def pipeline_links(facilities: gpd.GeoDataFrame, pipelines: gpd.GeoDataFrame,
                   snap_km: float, join_km: float) -> list[tuple[int, int, str]]:
    """Facility pairs connected through the same cluster of mapped pipelines."""

    pipes = pipelines.loc[~pipelines["substance"].isin(["gas", "oil", "fuel", "sewage"])].copy()
    if pipes.empty:
        return []
    pipes = pipes.to_crs(METRIC_CRS).reset_index(drop=True)
    pipes["geometry"] = [g.exterior if g.geom_type == "Polygon" else g for g in pipes.geometry]
    uf = UnionFind(len(pipes))
    node_owner: dict[int, int] = {}
    for i, ids in enumerate(pipes["node_ids"]):
        for node in json.loads(ids):
            if node in node_owner:
                uf.union(i, node_owner[node])
            node_owner[node] = i
    ends = [(i, shapely.Point(g.coords[k])) for i, g in enumerate(pipes.geometry) for k in (0, -1)]
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            if ends[a][0] != ends[b][0] and ends[a][1].distance(ends[b][1]) <= join_km * 1000:
                uf.union(ends[a][0], ends[b][0])
    metric = facilities.to_crs(METRIC_CRS)
    touched: dict[int, list[int]] = {}
    for f_index, point in enumerate(metric.geometry):
        for p_index, line in enumerate(pipes.geometry):
            if point.distance(line) <= snap_km * 1000:
                touched.setdefault(uf.find(p_index), []).append(f_index)
    links = []
    for cluster, members in touched.items():
        substances = set(pipes.loc[[i for i in range(len(pipes)) if uf.find(i) == cluster], "substance"])
        evidence = "pipeline" if "water" in substances else "pipeline_substance_untagged"
        members = sorted(set(members))
        links += [(members[0], other, evidence) for other in members[1:]]
    return links


def nearest_place(point, places: gpd.GeoDataFrame, tree: cKDTree) -> tuple[str, float]:
    distance, index = tree.query([point.x, point.y])
    return str(places.iloc[index]["name"]), float(distance) / 1000


def node_geometry(geometry) -> dict:
    return json.loads(json.dumps(mapping(geometry)))


def restrict_features(features: gpd.GeoDataFrame, args: argparse.Namespace) -> None:
    """Apply the optional kind, pumping-substance, and national-border filters in place."""

    def drop(mask, reason):
        mask = mask & features["include"]
        features.loc[mask, "include"] = False
        features.loc[mask, "reason"] = reason

    if args.kinds:
        drop(~features["kind"].isin(args.kinds), "kind not selected for this graph")
    if args.pumping_requires_substance_water:
        substance = features["tags"].map(lambda text: json.loads(text).get("substance"))
        drop((features["kind"] == "pumping_station") & (substance != "water"),
             "pumping station without substance=water")
    if args.country_boundary:
        world = gpd.read_file(args.country_boundary)
        column = next(c for c in world.columns if world[c].astype(str).eq(args.country_name).any())
        world = world.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = world.loc[world[column] == args.country_name].total_bounds
        world = world.cx[minx - 2:maxx + 2, miny - 2:maxy + 2].to_crs(METRIC_CRS)
        world["geometry"] = world.geometry.make_valid()
        country = world.loc[world[column] == args.country_name].union_all()
        others = world.loc[world[column] != args.country_name].union_all()
        points = features.to_crs(METRIC_CRS)
        # The buffer keeps coastal features that a generalized coastline cuts off,
        # but never a feature that lies inside a neighbouring country.
        inside = points.within(country.buffer(args.country_buffer_km * 1000)) & ~points.within(others)
        drop(~inside, f"outside {args.country_name}")


def build(args: argparse.Namespace) -> None:
    for path in (args.graph_output, args.report_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    features = gpd.read_file(args.features, layer="features")
    pipelines = gpd.read_file(args.features, layer="pipelines")
    places = gpd.read_file(args.features, layer="places")
    classes = features.apply(classify, axis=1, result_type="expand", use_overrides=not args.no_overrides)
    features = pd.concat([features, classes], axis=1)
    restrict_features(features, args)
    included = features.loc[features["include"]].reset_index(drop=True)
    metric = included.to_crs(METRIC_CRS)
    xy = np.column_stack([metric.geometry.x, metric.geometry.y])

    # Supply systems: pipeline evidence first, then proximity.
    links = pipeline_links(included, pipelines, args.pipeline_snap_km, args.pipeline_join_km)
    tree = cKDTree(xy)
    pipeline_pairs = {(min(a, b), max(a, b)) for a, b, _ in links}
    for a, b in sorted(tree.query_pairs(args.link_km * 1000)):
        if (a, b) not in pipeline_pairs:
            links.append((a, b, "proximity"))
    uf = UnionFind(len(included))
    for a, b, _ in links:
        uf.union(a, b)
    roots = [uf.find(i) for i in range(len(included))]
    order = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    included["system_id"] = [f"water_system/{order[r] + 1:03d}" for r in roots]

    # Power: only pumping elements. A system with storage but no mapped pump gets
    # one implied pump, at the storage element nearest the system centre. With
    # --all-systems-grid-dependent, a system with neither gets one at the element
    # nearest its centre, so every system depends on the grid.
    included["power_role"] = np.where(included["powered"], "mapped_pump", "none")
    storage = included["kind"].isin(["water_tower", "reservoir_covered", "water_storage"])
    for system, members in included.groupby("system_id").groups.items():
        members = list(members)
        if included.loc[members, "powered"].any():
            continue
        centre = xy[members].mean(axis=0)
        tanks = [i for i in members if storage[i]]
        if tanks:
            main = min(tanks, key=lambda i: float(np.hypot(*(xy[i] - centre))))
            included.loc[main, "power_role"] = "implied_pump_at_storage"
        elif args.all_systems_grid_dependent:
            main = min(members, key=lambda i: float(np.hypot(*(xy[i] - centre))))
            included.loc[main, "power_role"] = "implied_pump_no_storage"

    grid = pickle.load(args.power_graph.open("rb"))
    grid_nodes = [n for n in grid.nodes if grid.nodes[n].get("power") in GRID_TYPES]
    grid_points = gpd.GeoSeries(
        [shapely.geometry.shape(grid.nodes[n]["geometry"]).centroid for n in grid_nodes],
        crs="EPSG:4326").to_crs(METRIC_CRS)
    grid_tree = cKDTree(np.column_stack([grid_points.x, grid_points.y]))

    graph = nx.MultiDiGraph(
        source="OpenStreetMap via Geofabrik", features=str(args.features),
        parameters={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )
    power_rows = []
    for i, row in included.iterrows():
        tags = json.loads(row["tags"])
        graph.add_node(row["osm_id"], geometry=node_geometry(row.geometry), properties=tags,
                       water=row["kind"], system_id=row["system_id"], ownership=row["ownership"],
                       power_role=row["power_role"])
        graph.add_edge(row["osm_id"], row["system_id"],
                       attribute={"water": f"{row['kind']}-water_system"})
        if row["power_role"] == "none":
            continue
        distance, index = grid_tree.query(xy[i])
        grid_id = grid_nodes[index]
        on_grid = args.all_systems_grid_dependent or distance <= args.power_km * 1000
        power_rows.append({"osm_id": row["osm_id"], "kind": row["kind"], "system_id": row["system_id"],
                           "power_role": row["power_role"], "grid_node": grid_id,
                           "grid_type": grid.nodes[grid_id]["power"], "distance_km": distance / 1000,
                           "linked": bool(on_grid)})
        if on_grid:
            node = grid.nodes[grid_id]
            graph.add_node(grid_id, geometry=node["geometry"], properties=node.get("properties", {}),
                           water=node["power"])
            graph.add_edge(grid_id, row["osm_id"],
                           attribute={"water": f"power-{row['kind']}"}, distance_km=distance / 1000)

    place_metric = places.dropna(subset=["name"]).to_crs(METRIC_CRS).reset_index(drop=True)
    place_rank = place_metric["place"].map({"city": 0, "town": 1, "village": 2})
    place_tree = cKDTree(np.column_stack([place_metric.geometry.x, place_metric.geometry.y]))
    system_rows = []
    for system, members in included.groupby("system_id").groups.items():
        sub, sub_xy = included.loc[members], xy[list(members)]
        centre = shapely.Point(sub_xy.mean(axis=0))
        nearby = place_tree.query_ball_point([centre.x, centre.y], 15_000)
        if nearby:
            best = min(nearby, key=lambda k: (place_rank[k], centre.distance(place_metric.geometry[k])))
            label = str(place_metric.loc[best, "name"])
        else:
            label = nearest_place(centre, place_metric, place_tree)[0]
        lon_lat = gpd.GeoSeries([centre], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        implied = [r for r in ("implied_pump_at_storage", "implied_pump_no_storage")
                   if (sub["power_role"] == r).any()]
        source = "mapped_pump" if sub["powered"].any() else (implied[0] if implied else "no_source_mapped")
        graph.add_node(system, geometry=node_geometry(lon_lat), properties={"name": label},
                       water="water_system", label=label, source=source)
        system_rows.append({
            "system_id": system, "label": label, "facilities": len(sub),
            **{f"n_{k}": int((sub["kind"] == k).sum()) for k in
               ("water_works", "pumping_station", "borehole", "water_tower",
                "reservoir_covered", "water_storage", "standpipe")},
            "piped_utility_members": int((sub["ownership"] == "piped_utility").sum()),
            "power_source": source, "lon": lon_lat.x, "lat": lon_lat.y,
        })

    hospitals = gpd.read_file(args.hospitals)
    piped = hospitals.loc[hospitals["water_sour"] == 1]
    hospital_rows = []
    for index, row in piped.iterrows():
        node_id = hospital_node_id(index)
        record = {"hospital_id": node_id, "name": row.get("name"), "urban": row.get("urban"),
                  "has_geometry": row.geometry is not None and not row.geometry.is_empty}
        if record["has_geometry"]:
            point = gpd.GeoSeries([row.geometry], crs=hospitals.crs).to_crs(METRIC_CRS).iloc[0]
            distance, index_near = tree.query([point.x, point.y])
            nearest = included.iloc[index_near]
            record.update(nearest_osm_id=nearest["osm_id"], nearest_kind=nearest["kind"],
                          system_id=nearest["system_id"], distance_km=distance / 1000,
                          attached=bool(distance <= args.service_km * 1000))
            if record["attached"]:
                graph.add_node(node_id, geometry=node_geometry(row.geometry),
                               properties={"water": "hospital"}, water="hospital")
                graph.add_edge(nearest["system_id"], node_id,
                               attribute={"water": "water_system-hospital"},
                               distance_km=distance / 1000, via=nearest["osm_id"])
        else:
            record["attached"] = False
        hospital_rows.append(record)

    hospital_frame = pd.DataFrame(hospital_rows)
    power_frame = pd.DataFrame(power_rows)
    system_frame = pd.DataFrame(system_rows)
    served = hospital_frame.loc[hospital_frame["attached"]].groupby("system_id").size()
    system_frame["hospitals_attached"] = system_frame["system_id"].map(served).fillna(0).astype(int)
    linked = power_frame.loc[power_frame["linked"]].groupby("system_id").size() if len(power_frame) else pd.Series(dtype=int)
    system_frame["grid_links"] = system_frame["system_id"].map(linked).fillna(0).astype(int)
    evaluable = {s for s in system_frame.loc[system_frame["grid_links"] > 0, "system_id"]}
    hospital_frame["evaluable"] = hospital_frame["attached"] & hospital_frame["system_id"].isin(evaluable)

    args.report_dir.mkdir(parents=True)
    keep = ["osm_id", "man_made", "amenity", "office", "name", "operator", "kind", "ownership",
            "include", "reason", "area_ha"]
    features_out = features[keep].copy()
    features_out["lon"], features_out["lat"] = features.geometry.x, features.geometry.y
    features_out.to_csv(args.report_dir / "facilities_classified.csv", index=False)
    included.drop(columns=["geometry", "footprint_wkt", "tags"]).assign(
        lon=included.geometry.x, lat=included.geometry.y
    ).to_csv(args.report_dir / "facilities_included.csv", index=False)
    pd.DataFrame([{"from": included.loc[a, "osm_id"], "to": included.loc[b, "osm_id"],
                   "evidence": e, "distance_km": float(np.hypot(*(xy[a] - xy[b]))) / 1000}
                  for a, b, e in links]).to_csv(args.report_dir / "system_links.csv", index=False)
    system_frame.to_csv(args.report_dir / "systems.csv", index=False)
    power_frame.to_csv(args.report_dir / "power_links.csv", index=False)
    hospital_frame.to_csv(args.report_dir / "hospital_attachment.csv", index=False)
    audit_wastewater(features, included, args).to_csv(
        args.report_dir / "wastewater_plant_audit.csv", index=False)

    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    with args.graph_output.open("xb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    link_counts = pd.Series([e for *_, e in links]).value_counts().to_dict() if links else {}
    report = {
        "schema_version": 1,
        "graph": str(args.graph_output),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "features_extracted": int(len(features)),
        "features_included": int(len(included)),
        "included_by_kind": included["kind"].value_counts().to_dict(),
        "excluded_by_reason": features.loc[~features["include"], "reason"].value_counts().to_dict(),
        "systems": int(len(system_frame)),
        "systems_by_power_source": system_frame["power_source"].value_counts().to_dict(),
        "system_links_by_evidence": link_counts,
        "powered_elements": int(len(power_frame)),
        "powered_elements_linked_to_grid": int(power_frame["linked"].sum()) if len(power_frame) else 0,
        "piped_water_facilities": int(len(hospital_frame)),
        "with_coordinates": int(hospital_frame["has_geometry"].sum()),
        "attached_within_service_km": int(hospital_frame["attached"].sum()),
        "attached_and_grid_dependent": int(hospital_frame["evaluable"].sum()),
        "attachment_distance_quantiles_km": hospital_frame.loc[
            hospital_frame["has_geometry"], "distance_km"].quantile([.25, .5, .75, .9]).round(2).to_dict(),
        "service_km_sensitivity": {
            str(km): {
                "within_km": int((hospital_frame["distance_km"] <= km).sum()),
                "within_km_and_grid_dependent": int(
                    ((hospital_frame["distance_km"] <= km)
                     & hospital_frame["system_id"].isin(evaluable)).sum()),
            }
            for km in SERVICE_SENSITIVITY_KM
        },
    }
    (args.report_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


def audit_wastewater(features: gpd.GeoDataFrame, included: gpd.GeoDataFrame,
                     args: argparse.Namespace) -> pd.DataFrame:
    """Evidence on every OSM wastewater plant, including the legacy suppliers."""

    legacy = set()
    if args.legacy_water_graph and args.legacy_water_graph.exists():
        old = pickle.load(args.legacy_water_graph.open("rb"))
        legacy = {n for n in old.nodes if old.nodes[n].get("water") == "wastewater_plant"}
    plants = features.loc[features["man_made"] == "wastewater_plant"].to_crs(METRIC_CRS)
    works = features.loc[features["man_made"] == "water_works"].to_crs(METRIC_CRS)
    public = included.loc[included["ownership"] == "piped_utility"].to_crs(METRIC_CRS)
    rows = []
    for _, plant in plants.iterrows():
        def nearest(frame):
            if frame.empty:
                return None, None
            d = frame.geometry.distance(plant.geometry)
            k = d.idxmin()
            return frame.loc[k, "name"] or frame.loc[k, "osm_id"], round(float(d[k]) / 1000, 2)
        works_name, works_km = nearest(works)
        public_name, public_km = nearest(public)
        name = plant["name"] if isinstance(plant["name"], str) else ""
        if plant["osm_id"] in OVERRIDES and not args.no_overrides:
            verdict = "reclassified as drinking-water works by name"
        elif re.search(r"\betar\b|residua|esgoto|sewage", name, re.IGNORECASE):
            verdict = "wastewater treatment (named ETAR/sewage)"
        else:
            verdict = "function not verifiable from OSM; excluded"
        lon_lat = gpd.GeoSeries([plant.geometry], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        rows.append({"osm_id": plant["osm_id"], "name": name, "operator": plant["operator"],
                     "area_ha": plant["area_ha"], "lon": round(lon_lat.x, 4), "lat": round(lon_lat.y, 4),
                     "used_in_legacy_graph": plant["osm_id"] in legacy,
                     "nearest_water_works": works_name, "nearest_water_works_km": works_km,
                     "nearest_public_water_feature": public_name, "nearest_public_water_km": public_km,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="GeoPackage from extract_osm_water_features.py")
    parser.add_argument("--power-graph", type=Path, required=True, help="simplified_hospital_MG.pickle")
    parser.add_argument("--hospitals", type=Path, required=True, help="SARA_processed")
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--legacy-water-graph", type=Path)
    parser.add_argument("--link-km", type=float, default=3.0, help="proximity linkage within a system")
    parser.add_argument("--pipeline-snap-km", type=float, default=0.25)
    parser.add_argument("--pipeline-join-km", type=float, default=0.05)
    parser.add_argument("--power-km", type=float, default=10.0, help="max distance to a grid node")
    parser.add_argument("--service-km", type=float, default=3.0, help="max facility-to-system distance")
    parser.add_argument("--no-overrides", action="store_true", help="ignore name-based reclassifications")
    parser.add_argument("--kinds", nargs="+",
                        help="keep only these facility kinds, e.g. water_works water_tower")
    parser.add_argument("--pumping-requires-substance-water", action="store_true",
                        help="keep pumping stations only when tagged substance=water")
    parser.add_argument("--country-boundary", type=Path, help="country polygons for clipping")
    parser.add_argument("--country-name", default="Mozambique")
    parser.add_argument("--country-buffer-km", type=float, default=2.0,
                        help="buffer so coastal features survive a generalized boundary")
    parser.add_argument("--all-systems-grid-dependent", action="store_true",
                        help="give every system a grid link: implied pump where none is mapped, "
                             "and no distance limit to the nearest grid node")
    build(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
