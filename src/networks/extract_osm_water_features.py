#!/usr/bin/env python3
"""Extract water-supply features for Mozambique from an OpenStreetMap PBF.

Every object with a water-related tag is written, including features that the
network builder later excludes (wastewater plants, private tanks, hand-pump
wells), so each inclusion decision can be audited.

Outputs a GeoPackage with three layers:
  features   points (centroid of areas) with tags as JSON
  pipelines  man_made=pipeline ways with their OSM node ids
  places     place=city/town/village nodes, used to name systems
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmium
from shapely.geometry import LineString, Point, Polygon

FEATURE_TAGS = (
    ("man_made", "water_works"),
    ("man_made", "wastewater_plant"),
    ("man_made", "water_tower"),
    ("man_made", "reservoir_covered"),
    ("man_made", "storage_tank"),
    ("man_made", "tank"),
    ("man_made", "pumping_station"),
    ("man_made", "water_well"),
    ("man_made", "water_tap"),
    ("amenity", "water_point"),
    ("amenity", "drinking_water"),
    ("office", "water_utility"),
    ("landuse", "reservoir"),
    ("water", "reservoir"),
)
PIPELINE_TAG = ("man_made", "pipeline")
PLACE_TAGS = (("place", "city"), ("place", "town"), ("place", "village"))


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def way_geometry(way) -> tuple[object, list[int]]:
    coords, node_ids = [], []
    for node in way.nodes:
        if not node.location.valid():
            continue
        coords.append((node.location.lon, node.location.lat))
        node_ids.append(node.ref)
    if len(coords) >= 4 and way.is_closed():
        return Polygon(coords), node_ids
    if len(coords) >= 2:
        return LineString(coords), node_ids
    return None, node_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing file: {args.output}")

    features, pipelines, places, skipped_relations = [], [], [], 0
    tag_filter = osmium.filter.TagFilter(*FEATURE_TAGS, PIPELINE_TAG, *PLACE_TAGS)
    processor = osmium.FileProcessor(args.pbf).with_locations().with_filter(tag_filter)
    for obj in processor:
        tags = dict(obj.tags)
        if obj.is_relation():
            skipped_relations += 1
            continue
        if obj.is_node():
            osm_id, geometry, footprint = f"node/{obj.id}", Point(obj.lon, obj.lat), None
            node_ids = [obj.id]
        else:
            osm_id = f"way/{obj.id}"
            footprint, node_ids = way_geometry(obj)
            if footprint is None:
                continue
            geometry = footprint.centroid if isinstance(footprint, Polygon) else footprint
        if any(tags.get(k) == v for k, v in PLACE_TAGS) and obj.is_node():
            places.append({"osm_id": osm_id, "place": tags["place"], "name": tags.get("name"),
                           "geometry": geometry})
        if tags.get("man_made") == "pipeline" and obj.is_way():
            pipelines.append({
                "osm_id": osm_id,
                "substance": tags.get("substance"),
                "location": tags.get("location"),
                "operator": tags.get("operator"),
                "node_ids": json.dumps(node_ids),
                "tags": json.dumps(tags, ensure_ascii=False),
                "geometry": footprint.exterior if isinstance(footprint, Polygon) else footprint,
            })
        if any(tags.get(k) == v for k, v in FEATURE_TAGS):
            features.append({
                "osm_id": osm_id,
                "man_made": tags.get("man_made"),
                "amenity": tags.get("amenity"),
                "office": tags.get("office"),
                "name": tags.get("name"),
                "operator": tags.get("operator"),
                "area_ha": None,
                "tags": json.dumps(tags, ensure_ascii=False),
                "geometry": Point(geometry.x, geometry.y) if not isinstance(geometry, Point) else geometry,
                "footprint_wkt": footprint.wkt if footprint is not None else None,
            })

    feature_frame = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    polygons = feature_frame["footprint_wkt"].fillna("").str.startswith("POLYGON")
    if polygons.any():
        areas = gpd.GeoSeries.from_wkt(feature_frame.loc[polygons, "footprint_wkt"], crs="EPSG:4326")
        feature_frame.loc[polygons, "area_ha"] = areas.to_crs(epsg=32736).area.to_numpy() / 1e4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_file(args.output, layer="features", driver="GPKG")
    gpd.GeoDataFrame(pipelines, geometry="geometry", crs="EPSG:4326").to_file(
        args.output, layer="pipelines", driver="GPKG")
    gpd.GeoDataFrame(places, geometry="geometry", crs="EPSG:4326").to_file(
        args.output, layer="places", driver="GPKG")

    report = {
        "schema_version": 1,
        "source_pbf": str(args.pbf),
        "source_md5": md5sum(args.pbf),
        "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": len(feature_frame),
        "pipelines": len(pipelines),
        "places": len(places),
        "relations_skipped": skipped_relations,
        "feature_counts": feature_frame.fillna({"man_made": "", "amenity": "", "office": ""})
        .groupby(["man_made", "amenity", "office"]).size().rename("n").reset_index()
        .to_dict("records"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_counts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
