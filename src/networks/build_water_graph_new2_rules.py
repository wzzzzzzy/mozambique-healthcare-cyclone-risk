#!/usr/bin/env python3
"""Rebuild a water dependency graph from verified OSM features with the new2 rules.

Node set: the drinking-water facilities kept by ``build_osm_water_network.py``
(``facilities_included.csv``).  Links follow
``water_network/generate_water_network.ipynb``, which built
``MG_w_hospital_new2.pickle``:

* every water facility -> nearest grid node, searched with ``find_nearest_in_G``
  (3 x 3 block of one-degree cells, no distance limit);
* treatment works -> storage when the nearest works is closer than 15 km;
* each piped-water healthcare facility -> nearest storage, again with
  ``find_nearest_in_G`` and no distance limit unless ``--facility-max-km`` is
  given; a facility beyond the cap stays unattached (unevaluated downstream).

Storage means water towers and covered reservoirs; treatment works take the
place the wastewater plants had in new2.  Grid nodes are restricted to the
types the dependency model reads, which are the only types new2 used.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping

from hospital_grid_ablation import hospital_node_id
from utils import dist_btw_2Point, find_nearest_in_G, generate_indexing_specify, get_node_pos

GRID_TYPES = ("gridfinder", "portal", "substation", "transformer")
PLANT_KINDS = ("water_works",)
STORAGE_KINDS = ("water_tower", "reservoir_covered")
PLANT_TO_STORAGE_KM = 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facilities", type=Path, required=True, help="facilities_included.csv")
    parser.add_argument("--power-graph", type=Path, required=True, help="simplified_hospital_MG.pickle")
    parser.add_argument("--hospitals", type=Path, required=True, help="SARA_processed")
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--facility-max-km", type=float,
                        help="cap on the healthcare facility -> storage link (new2 had none)")
    args = parser.parse_args()
    for path in (args.graph_output, args.report_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    facilities = pd.read_csv(args.facilities)
    facilities = facilities.loc[facilities["kind"].isin(PLANT_KINDS + STORAGE_KINDS)]
    graph = nx.MultiDiGraph(rules="new2 (generate_water_network.ipynb)", facilities=str(args.facilities))
    for row in facilities.itertuples(index=False):
        graph.add_node(row.osm_id, geometry=mapping(Point(row.lon, row.lat)),
                       properties={"man_made": row.kind, "name": row.name if isinstance(row.name, str) else None},
                       water=row.kind)
    plants = list(facilities.loc[facilities["kind"].isin(PLANT_KINDS), "osm_id"])
    storage = list(facilities.loc[facilities["kind"].isin(STORAGE_KINDS), "osm_id"])
    links = []

    power = pickle.load(args.power_graph.open("rb"))
    grid_nodes = [n for n in power.nodes if power.nodes[n].get("power") in GRID_TYPES]
    grid_index, _ = generate_indexing_specify(power, grid_nodes)
    for node in plants + storage:
        distance, grid_id = find_nearest_in_G(get_node_pos(graph, node), grid_index, power)
        if grid_id == -1:
            links.append(("power", node, None, "no grid node in neighbouring cells"))
            continue
        source = power.nodes[grid_id]
        graph.add_node(grid_id, geometry=source["geometry"], properties=source.get("properties", {}),
                       water=source["power"])
        graph.add_edge(grid_id, node, attribute={"water": f"power-{graph.nodes[node]['water']}"})
        links.append(("power", node, distance, grid_id))

    for node in storage:
        point = get_node_pos(graph, node)
        nearest = min(((dist_btw_2Point(point, get_node_pos(graph, p)), p) for p in plants), default=(None, None))
        if nearest[0] is not None and nearest[0] < PLANT_TO_STORAGE_KM:
            graph.add_edge(nearest[1], node, attribute={"water": f"water_works-{graph.nodes[node]['water']}"})
            links.append(("plant", node, nearest[0], nearest[1]))

    hospitals = gpd.read_file(args.hospitals)
    piped = hospitals.loc[hospitals["water_sour"] == 1]
    storage_index, _ = generate_indexing_specify(graph, storage)
    for index, row in piped.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        node_id = hospital_node_id(index)
        graph.add_node(node_id, geometry=json.loads(json.dumps(mapping(row.geometry))),
                       properties={"water": "hospital"}, water="hospital")
        distance, nearest = find_nearest_in_G(row.geometry, storage_index, graph)
        if nearest != -1 and args.facility_max_km is not None and distance > args.facility_max_km:
            links.append(("hospital_beyond_cap", node_id, distance, nearest))
            continue
        if nearest != -1:
            graph.add_edge(nearest, node_id, attribute={"water": "storage-hospital"})
        links.append(("hospital", node_id, distance if nearest != -1 else None, nearest))

    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    with args.graph_output.open("xb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    frame = pd.DataFrame(links, columns=["link", "node", "distance_km", "other"])
    args.report_dir.mkdir(parents=True)
    frame.to_csv(args.report_dir / "links.csv", index=False)
    stats = {}
    for link, group in frame.groupby("link"):
        d = group["distance_km"].dropna().to_numpy(float)
        stats[link] = {"linked": int(len(d)), "unlinked": int(group["distance_km"].isna().sum()),
                       "median_km": float(np.median(d)) if len(d) else None,
                       "p90_km": float(np.quantile(d, .9)) if len(d) else None,
                       "max_km": float(d.max()) if len(d) else None}
    report = {
        "schema_version": 1, "graph": str(args.graph_output), "rules": "new2",
        "facility_max_km": args.facility_max_km,
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "water_works": len(plants), "storage": len(storage),
        "storage_fed_by_works": int((frame["link"] == "plant").sum()),
        "links": stats,
    }
    (args.report_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
