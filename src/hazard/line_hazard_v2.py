#!/usr/bin/env python3
"""Line hazard v2: route-sampled winds and tower + length-normalized conductor fragility.

Main case (adopted): wind threshold 24 m/s, conductor reference length
``L0 = 0.25 km``, mapped towers/poles included, wind sampled along every
physical segment at a maximum spacing of 0.5 km.

For branch ``b`` (a hybrid-model edge; its path through the unsimplified
network comes from the model's path cache)

    P_b = 1 - prod_{i in towers(b)} (1 - p_i^tower) prod_{j in spans(b)} (1 - p_j^conductor),
    p_j^conductor = 1 - prod_k [1 - q(V_k)] ** (dL_k / L0),

where towers are interior path nodes mapped as OSM ``power=tower``/``pole``,
``p^tower`` is the Emanuel-form tower curve (V_thresh = threshold,
V_half = 110 kn) and ``q`` the line curve ``2**((V - V0)/30) - 1`` on
``[V0, V0 + 30]``; both curves are those of the original node wind layers and
are provisional.  Spans are sampled at the midpoints of ``ceil(L/dx)`` equal
intervals.  Facility links (one endpoint outside the hazard network) keep the
single draw with the grid endpoint's ``q``.  ``--rule previous`` reproduces
``prepare_matpower_cyclone_input --failure-mode line`` for validation.

Subcommands
-----------
geometry  nodes, spans, wind sample points and towers of the path caches.
track     winds for one event (synthetic STORM track, historical IBTrACS
          track, or a wind raster) and branch probabilities per path cache.
chunk     synthetic tracks ``global_start..global_stop-1`` with one MATLAB
          engine; writes winds and probabilities per track, skipping tracks
          already complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

KNOT_MS = 1850.0 / 3600.0  # m/s per knot as in the node wind layers
V_HALF_TOWER_KN = 110.0
EARTH_RADIUS_M = 6_371_008.8


# ----------------------------------------------------------------------------
# fragility
def line_curve(wind: np.ndarray, threshold: float, span: float = 30.0) -> np.ndarray:
    clipped = np.clip(wind, threshold, threshold + span)
    return np.clip(np.exp(0.6931 * (clipped - threshold) / span) - 1.0, 0.0, 1.0)


def tower_curve(wind: np.ndarray, threshold: float) -> np.ndarray:
    v = wind / KNOT_MS
    v_thresh = threshold / KNOT_MS
    v_n = np.clip((v - v_thresh) / (V_HALF_TOWER_KN - v_thresh), 0.0, None)
    return v_n**3 / (1.0 + v_n**3)


# ----------------------------------------------------------------------------
# geometry
def build_geometry(args: argparse.Namespace) -> None:
    from pyproj import Geod

    from build_matpower_case import node_kind
    from utils import get_node_pos

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with args.hazard_network.open("rb") as handle:
        hazard = pickle.load(handle)
    caches = {}
    for spec in args.cache:
        name, path = spec.split("=", 1)
        with np.load(path, allow_pickle=False) as archive:
            caches[name] = {k: archive[k] for k in archive.files}

    names = sorted({str(n) for c in caches.values() for n in c["node_ids"]})
    index = {n: i for i, n in enumerate(names)}
    present = np.array([n in hazard for n in names])
    lon = np.full(len(names), np.nan)
    lat = np.full(len(names), np.nan)
    support = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        if present[i]:
            point = get_node_pos(hazard, n)
            lon[i], lat[i] = point.x, point.y
            support[i] = node_kind(hazard.nodes[n]) in ("tower", "pole")

    segment_index: dict[tuple[int, int], int] = {}
    seg_a, seg_b = [], []
    per_cache = {}
    for name, cache in caches.items():
        local = np.array([index[str(n)] for n in cache["node_ids"]])
        flat = local[cache["path_nodes"]]
        indptr = cache["path_indptr"]
        seg_ptr, seg_ids, tower_ptr, tower_ids = [0], [], [0], []
        for b in range(len(cache["branch_uid"])):
            path = flat[indptr[b]:indptr[b + 1]]
            for u, v in zip(path[:-1], path[1:]):
                key = (min(u, v), max(u, v))
                if key not in segment_index:
                    segment_index[key] = len(seg_a)
                    seg_a.append(key[0])
                    seg_b.append(key[1])
                seg_ids.append(segment_index[key])
            seg_ptr.append(len(seg_ids))
            interior = path[1:-1]
            tower_ids.extend(interior[support[interior]].tolist())
            tower_ptr.append(len(tower_ids))
        per_cache[name] = {"branch_uid": cache["branch_uid"].astype(np.int64),
                           "seg_ptr": np.array(seg_ptr), "seg_ids": np.array(seg_ids, dtype=np.int64),
                           "tower_ptr": np.array(tower_ptr), "tower_ids": np.array(tower_ids, dtype=np.int64)}

    seg_a, seg_b = np.array(seg_a), np.array(seg_b)
    physical = present[seg_a] & present[seg_b]
    facility_node = np.where(present[seg_a] & ~present[seg_b], seg_a,
                             np.where(present[seg_b] & ~present[seg_a], seg_b, -1))
    geod = Geod(ellps="WGS84")
    length_km = np.zeros(len(seg_a))
    _, _, metres = geod.inv(lon[seg_a[physical]], lat[seg_a[physical]], lon[seg_b[physical]], lat[seg_b[physical]])
    length_km[physical] = metres / 1000.0
    n_samples = np.where(physical, np.maximum(1, np.ceil(length_km / args.max_spacing_km)), 0).astype(np.int64)
    sample_ptr = np.concatenate([[0], np.cumsum(n_samples)])
    owner = np.repeat(np.arange(len(seg_a)), n_samples)
    fraction = (np.arange(sample_ptr[-1]) - sample_ptr[owner] + 0.5) / n_samples[owner]
    sample_lon = lon[seg_a[owner]] + fraction * (lon[seg_b[owner]] - lon[seg_a[owner]])
    sample_lat = lat[seg_a[owner]] + fraction * (lat[seg_b[owner]] - lat[seg_a[owner]])
    sample_interval_km = (length_km / np.maximum(n_samples, 1))[owner]

    arrays = {"node_ids": np.array(names), "node_present": present, "node_lon": lon, "node_lat": lat,
              "node_support": support, "seg_a": seg_a, "seg_b": seg_b, "seg_physical": physical,
              "seg_facility_node": facility_node, "seg_length_km": length_km, "sample_ptr": sample_ptr,
              "sample_lon": sample_lon, "sample_lat": sample_lat, "sample_interval_km": sample_interval_km,
              "max_spacing_km": np.array([args.max_spacing_km]), "caches": np.array(sorted(per_cache))}
    for name, data in per_cache.items():
        arrays.update({f"{name}__{k}": v for k, v in data.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({"nodes": len(names), "present": int(present.sum()), "supports": int(support.sum()),
                      "segments": len(seg_a), "physical": int(physical.sum()),
                      "facility_links": int((facility_node >= 0).sum()), "samples": int(sample_ptr[-1]),
                      "physical_km": float(length_km.sum()),
                      "branches": {k: int(len(v["branch_uid"])) for k, v in per_cache.items()}}, indent=2))


# ----------------------------------------------------------------------------
# winds
def storm_centres(storm_root: Path, folder: str, track_id: int, boundary: Path):
    """STORM centres inside Mozambique, interpolated as in the node-wind scripts."""

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    moz = gpd.read_file(storm_root / "Moz_cyclone_track" / folder)
    table = pd.read_csv(storm_root / "all_tracks" / f"{folder}.csv")
    year, month, tcid = moz["Year"][track_id], moz["Month"][track_id], moz["TCID"][track_id]
    track = table[(table["Year"] == year) & (table["Month"] == month) & (table["TCID"] == tcid)]
    countries = gpd.read_file(boundary)
    mozambique = countries[countries["COUNTRY"] == "Mozambique"]["geometry"][151]
    centres, last = [], None
    for _, row in track.iterrows():
        current = (row["Lon"], row["Lat"], row["Raius"], row["MaxSpeed"])
        if last is not None and last[2] > 0:
            for i in range(1, 6):
                centres.append(tuple(i / 6.0 * a + (1 - i / 6.0) * b for a, b in zip(last, current)))
        centres.append(current)
        last = current
    return [(lon, lat, radius * 1000.0, speed) for lon, lat, radius, speed in centres
            if mozambique.contains(Point([lon, lat]))]


def historical_centres(points_csv: Path, track_id: int, boundary: Path):
    import pandas as pd
    from shapely.geometry import Point

    from prepare_historical_wind_track import KNOT_TO_MS, NAUTICAL_MILE_TO_M, load_country

    points = pd.read_csv(points_csv)
    points = points[points["track_id"].eq(track_id)]
    country = load_country(boundary)
    centres = []
    for row in points.itertuples(index=False):
        vmax, rmax = float(row.wind_knots) * KNOT_TO_MS, float(row.rmw_nm) * NAUTICAL_MILE_TO_M
        if (country.covers(Point(float(row.longitude), float(row.latitude)))
                and np.isfinite(vmax) and np.isfinite(rmax) and vmax > 0 and rmax > 0):
            centres.append((float(row.longitude), float(row.latitude), rmax, vmax))
    return centres


def start_engine(chavas_dir: Path):
    import matlab.engine

    engine = matlab.engine.start_matlab()
    engine.cd(str(chavas_dir), nargout=0)
    return engine


def profile_winds(engine, centres, lon: np.ndarray, lat: np.ndarray, historical: bool) -> np.ndarray:
    """Peak ER11E04 wind at each point; synthetic: geodesic, 0 beyond r0; historical: haversine."""

    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    peak = np.zeros(len(lon)) if historical else np.full(len(lon), -np.inf)
    valid = np.isfinite(lon)
    for c_lon, c_lat, rmax, vmax in centres:
        radial = engine.ER11E04_nondim_rmaxinput(vmax, rmax, 5e-5, 0, 1.5e-3, 2 / 1000, 0, 1.0, 0, 0.15,
                                                 nargout=11)
        radii = np.asarray(radial[0], dtype=float).reshape(-1)
        profile = np.asarray(radial[1], dtype=float).reshape(-1)
        d = np.full(len(lon), np.inf)
        if historical:
            lo, la = np.radians(lon[valid]), np.radians(lat[valid])
            h = (np.sin((la - math.radians(c_lat)) / 2) ** 2 + math.cos(math.radians(c_lat)) * np.cos(la)
                 * np.sin((lo - math.radians(c_lon)) / 2) ** 2)
            d[valid] = EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
            wind = np.interp(d, radii, profile, left=profile[0], right=0.0)
        else:
            _, _, d[valid] = geod.inv(np.full(valid.sum(), c_lon), np.full(valid.sum(), c_lat),
                                      lon[valid], lat[valid])
            wind = np.where(d > radii[-1], 0.0, np.interp(d, radii, profile))
        peak = np.maximum(peak, wind)
    return np.where(np.isfinite(peak), peak, 0.0)


def raster_winds(raster: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    import rasterio as rs

    out = np.zeros(len(lon))
    valid = np.isfinite(lon)
    with rs.open(raster) as src:
        values = src.read(1).astype(float)
        rows, cols = rs.transform.rowcol(src.transform, xs=lon[valid], ys=lat[valid])
    out[valid] = values[np.asarray(rows), np.asarray(cols)]
    return out


# ----------------------------------------------------------------------------
# probabilities
def branch_probabilities(geo, cache: str, node_wind, sample_wind, threshold, l0_km, towers, rule):
    present = geo["node_present"]
    q_node = np.where(present, line_curve(node_wind, threshold), 0.0)
    seg_a, seg_b = geo["seg_a"], geo["seg_b"]
    with np.errstate(divide="ignore"):
        if rule == "previous":
            p_seg = np.maximum(q_node[seg_a], q_node[seg_b])
            seg_log = np.log1p(-p_seg)
        else:
            ptr = geo["sample_ptr"]
            term = (geo["sample_interval_km"] / l0_km) * np.log1p(-line_curve(sample_wind, threshold))
            physical_log = np.zeros(len(seg_a))
            nonempty = ptr[1:] > ptr[:-1]
            sums = np.add.reduceat(term, ptr[:-1][nonempty]) if term.size else np.array([])
            physical_log[nonempty] = sums
            facility = geo["seg_facility_node"]
            seg_log = np.where(geo["seg_physical"], physical_log,
                               np.where(facility >= 0, np.log1p(-q_node[np.maximum(facility, 0)]), 0.0))
        seg_ptr, seg_ids = geo[f"{cache}__seg_ptr"], geo[f"{cache}__seg_ids"]
        total = np.add.reduceat(np.append(seg_log[seg_ids], 0.0), seg_ptr[:-1])
        total[seg_ptr[1:] == seg_ptr[:-1]] = 0.0
        if towers and rule != "previous":
            t_ptr, t_ids = geo[f"{cache}__tower_ptr"], geo[f"{cache}__tower_ids"]
            tower_log = np.log1p(-tower_curve(node_wind, threshold))
            t_total = np.add.reduceat(np.append(tower_log[t_ids], 0.0), t_ptr[:-1])
            t_total[t_ptr[1:] == t_ptr[:-1]] = 0.0
            total = total + t_total
    return np.clip(-np.expm1(total), 0.0, 1.0), seg_log


def save_winds(path: Path, node_wind, sample_wind, centres) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, node_wind=node_wind.astype(np.float32), sample_wind=sample_wind.astype(np.float32),
                        centres_inside=np.array([-1 if centres is None else centres]))


def write_probabilities(path: Path, uids: np.ndarray, probabilities: np.ndarray) -> None:
    from prepare_matpower_cyclone_input import atomic_csv

    atomic_csv(path, zip(uids.astype(int).tolist(), probabilities.tolist()))


def run_track(args: argparse.Namespace) -> None:
    for path in [*args.output, *( [args.winds_output] if args.winds_output else [])]:
        target = Path(path.split("=", 1)[1]) if isinstance(path, str) else path
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
    with np.load(args.geometry, allow_pickle=False) as archive:
        geo = {k: archive[k] for k in archive.files}
    lon = np.concatenate([geo["node_lon"], geo["sample_lon"]])
    lat = np.concatenate([geo["node_lat"], geo["sample_lat"]])
    n_nodes = len(geo["node_ids"])
    if args.winds_input:
        with np.load(args.winds_input) as archive:
            node_wind, sample_wind = archive["node_wind"].astype(float), archive["sample_wind"].astype(float)
    else:
        if args.raster:
            wind, centres = raster_winds(args.raster, lon, lat), None
        else:
            if args.track_points:
                c = historical_centres(args.track_points, args.track_id, args.boundary)
            else:
                c = storm_centres(args.storm_root, args.folder, args.track_id, args.boundary)
            engine = start_engine(args.chavas_dir)
            try:
                wind = profile_winds(engine, c, lon, lat, historical=bool(args.track_points))
            finally:
                engine.quit()
            centres = len(c)
        node_wind, sample_wind = wind[:n_nodes], wind[n_nodes:]
        if args.winds_output:
            save_winds(args.winds_output, node_wind, sample_wind, centres)
    report = {"rule": args.rule, "threshold_ms": args.threshold, "l0_km": args.l0_km, "towers": args.towers,
              "max_node_wind_ms": float(node_wind[geo["node_present"]].max(initial=0.0)),
              "max_sample_wind_ms": float(sample_wind.max(initial=0.0))}
    for spec in args.output:
        cache, path = spec.split("=", 1)
        p, seg_log = branch_probabilities(geo, cache, node_wind, sample_wind, args.threshold, args.l0_km,
                                          args.towers == "mapped", args.rule)
        write_probabilities(Path(path), geo[f"{cache}__branch_uid"], p)
        report[cache] = {"branches_with_positive_probability": int((p > 0).sum()),
                         "mean_failure_probability": float(p.mean())}
    print(json.dumps(report))


def run_chunk(args: argparse.Namespace) -> None:
    from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS

    with np.load(args.geometry, allow_pickle=False) as archive:
        geo = {k: archive[k] for k in archive.files}
    caches = [str(c) for c in geo["caches"]]
    lon = np.concatenate([geo["node_lon"], geo["sample_lon"]])
    lat = np.concatenate([geo["node_lat"], geo["sample_lat"]])
    n_nodes = len(geo["node_ids"])
    offsets = np.concatenate([[0], np.cumsum(TRACK_COUNTS)])
    engine = None
    try:
        for global_id in range(args.global_start, min(args.global_stop, offsets[-1])):
            storm_set = int(np.searchsorted(offsets, global_id, side="right") - 1)
            track_id = global_id - int(offsets[storm_set])
            name = f"dataset_{storm_set}/track_{track_id:04d}"
            winds_path = args.winds_dir / f"{name}.npz"
            outputs = {c: args.probability_dir / c / f"{name}.csv" for c in caches}
            if winds_path.exists() and all(p.exists() for p in outputs.values()):
                continue
            if winds_path.exists():
                with np.load(winds_path) as archive:
                    node_wind = archive["node_wind"].astype(float)
                    sample_wind = archive["sample_wind"].astype(float)
            else:
                folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
                centres = storm_centres(args.storm_root, folder, track_id, args.boundary)
                if engine is None and centres:
                    engine = start_engine(args.chavas_dir)
                wind = profile_winds(engine, centres, lon, lat, historical=False)
                node_wind, sample_wind = wind[:n_nodes], wind[n_nodes:]
                save_winds(winds_path, node_wind, sample_wind, len(centres))
            for cache, path in outputs.items():
                p, _ = branch_probabilities(geo, cache, node_wind, sample_wind, args.threshold, args.l0_km,
                                            args.towers == "mapped", "length")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_probabilities(path, geo[f"{cache}__branch_uid"], p)
            print(f"{name}: max node wind {node_wind[geo['node_present']].max(initial=0.0):.2f} m/s", flush=True)
    finally:
        if engine is not None:
            engine.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--hazard-network", type=Path, required=True, help="merge_MG_gridfiner.pickle")
    g.add_argument("--cache", action="append", required=True, help="NAME=path to hybrid_path_cache.npz")
    g.add_argument("--max-spacing-km", type=float, default=0.5)
    g.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("track")
    t.add_argument("--geometry", type=Path, required=True)
    source = t.add_mutually_exclusive_group()
    source.add_argument("--folder", help="STORM folder (synthetic)")
    source.add_argument("--track-points", type=Path, help="historical_track_points.csv")
    source.add_argument("--raster", type=Path)
    source.add_argument("--winds-input", type=Path, help="reuse saved winds")
    t.add_argument("--track-id", type=int)
    t.add_argument("--storm-root", type=Path)
    t.add_argument("--boundary", type=Path)
    t.add_argument("--chavas-dir", type=Path)
    t.add_argument("--threshold", type=float, default=24.0)
    t.add_argument("--l0-km", type=float, default=0.25)
    t.add_argument("--towers", choices=("none", "mapped"), default="mapped")
    t.add_argument("--rule", choices=("length", "previous"), default="length")
    t.add_argument("--winds-output", type=Path)
    t.add_argument("--output", action="append", required=True, help="CACHE=probability CSV")
    c = sub.add_parser("chunk")
    c.add_argument("--geometry", type=Path, required=True)
    c.add_argument("--storm-root", type=Path, required=True)
    c.add_argument("--boundary", type=Path, required=True)
    c.add_argument("--chavas-dir", type=Path, required=True)
    c.add_argument("--global-start", type=int, required=True)
    c.add_argument("--global-stop", type=int, required=True)
    c.add_argument("--winds-dir", type=Path, required=True)
    c.add_argument("--probability-dir", type=Path, required=True)
    c.add_argument("--threshold", type=float, default=24.0)
    c.add_argument("--l0-km", type=float, default=0.25)
    c.add_argument("--towers", choices=("none", "mapped"), default="mapped")
    args = parser.parse_args()
    {"geometry": build_geometry, "track": run_track, "chunk": run_chunk}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
