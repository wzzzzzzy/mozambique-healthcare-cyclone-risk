#!/usr/bin/env python3
"""Branch failure probabilities with a length-normalized line fragility.

Follows ``prepare_matpower_cyclone_input.py track`` (same path cache, node
wind layer, facility-link rule and output format) and changes only how a
physical line segment fails.

Fragility
---------
``q(V)`` is the event failure probability of a reference length ``L0`` of
line; transmission and distribution lines share the same curve and ``L0``.
The provisional curve is the manuscript's line curve,
``q(V) = 2**((V - V0) / 30) - 1`` on ``[V0, V0 + 30]`` (0 below, 1 above).
It is a sensitivity case, not a local calibration, and not a tower-failure
probability.

Aggregation
-----------
Wind is sampled at the midpoints of ``n = ceil(L_e / dx)`` equal intervals of
every physical segment ``e`` (interval lengths sum to ``L_e``), and

    p_e = 1 - prod_j [1 - q(V_j)] ** (dL_j / L0),

i.e. the failure rate ``r(V) = -ln[1 - q(V)] / L0`` integrated along the route.
Subdividing a line under uniform wind leaves ``p_e`` unchanged.  Failure
increments are independent conditional on the wind field; a branch fails if
any segment on its path fails.  ``dx`` sets numerical accuracy, ``L0`` the
physical failure rate.

Segments with only one endpoint in the hazard network (facility links) and
segments with none keep the previous rule: one draw with the present
endpoint's ``q``, or no failure.

Towers
------
With ``--towers mapped`` a branch also fails if a support structure on it
fails,

    P = 1 - prod_{i in towers} (1 - p_i^tower) prod_{j in spans} (1 - p_j^conductor),

where towers are the interior path nodes mapped as OSM ``power=tower`` or
``power=pole`` (gridfinder raster vertices are not supports) and
``p_i^tower`` is the tower curve of the node wind layer
(``tower_frac_prop_lost``; Emanuel form, V_half = 110 kn), as in the previous
``--failure-mode combined``.  The tower curve is provisional.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import rasterio as rs
from geopy import distance

from build_matpower_case import node_kind
from prepare_matpower_cyclone_input import (
    atomic_csv,
    load_path_cache,
    node_failure_probabilities,
    resolve_dbf,
)
from utils import get_node_pos


def fragility(wind: np.ndarray, threshold: float, span: float = 30.0) -> np.ndarray:
    """The manuscript line curve (``prepare_idai_threshold_wind.fragility``)."""

    clipped = np.clip(wind, threshold, threshold + span)
    return np.exp(0.6931 * (clipped - threshold) / span) - 1.0


class RasterWind:
    """Nearest-cell sampling, as in ``prepare_idai_threshold_wind.sample_wind``."""

    def __init__(self, path: Path):
        with rs.open(path) as src:
            self.values = src.read(1).astype(float)
            self.transform = src.transform

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        rows, cols = rs.transform.rowcol(self.transform, xs=lon, ys=lat)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < self.values.shape[0]) & (cols >= 0) & (cols < self.values.shape[1])
        if not inside.all():
            raise ValueError(f"{int((~inside).sum())} wind samples fall outside the raster")
        return self.values[rows, cols]


def log_survival(q: np.ndarray, interval_km: float, l0_km: float) -> float:
    with np.errstate(divide="ignore"):
        return float(np.sum((interval_km / l0_km) * np.log1p(-np.clip(q, 0.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path-cache", type=Path, required=True, help="model hybrid_path_cache.npz")
    parser.add_argument("--wind-track", type=Path, required=True, help="node wind layer of the event")
    parser.add_argument("--reference-network", type=Path, required=True, help="MG_hospital.pickle (node positions)")
    parser.add_argument("--raster", type=Path, required=True, help="event wind raster sampled along segments")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--l0-km", type=float, required=True)
    parser.add_argument("--max-spacing-km", type=float, default=0.5)
    parser.add_argument("--towers", choices=("none", "mapped"), default="none",
                        help="add the failure of mapped towers/poles on each branch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    for path in (args.output, report_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    cache = load_path_cache(args.path_cache)
    node_ids = [str(v) for v in cache["node_ids"]]
    node_probabilities, _ = node_failure_probabilities(resolve_dbf(args.wind_track))
    present = np.array([n in node_probabilities for n in node_ids])
    node_line = np.array([node_probabilities.get(n, (0.0, 0.0))[1] for n in node_ids])
    node_tower = np.array([node_probabilities.get(n, (0.0, 0.0))[0] for n in node_ids])
    with args.reference_network.open("rb") as handle:
        graph = pickle.load(handle)
    is_support = np.array([node_kind(graph.nodes[n]) in ("tower", "pole") for n in node_ids]) & present
    points = [get_node_pos(graph, n) for n in node_ids]
    lon = np.array([p.x for p in points])
    lat = np.array([p.y for p in points])
    wind_at = RasterWind(args.raster)

    segment_cache: dict[tuple[int, int], tuple[float, float, str]] = {}
    totals = {"line_km": 0.0, "expected_damaged_line_km": 0.0, "wind_samples": 0,
              "facility_links": 0, "segments_without_hazard": 0}

    def segment(a: int, b: int) -> tuple[float, float, str]:
        key = (min(a, b), max(a, b))
        if key in segment_cache:
            return segment_cache[key]
        if present[a] and present[b]:
            length = distance.distance((lat[a], lon[a]), (lat[b], lon[b])).km
            n = max(1, math.ceil(length / args.max_spacing_km))
            f = (np.arange(n) + 0.5) / n
            q = fragility(wind_at(lon[a] + f * (lon[b] - lon[a]), lat[a] + f * (lat[b] - lat[a])), args.threshold)
            value = (log_survival(q, length / n, args.l0_km), length, "line")
            totals["line_km"] += length
            totals["wind_samples"] += n
            p_e = 1.0 - math.exp(value[0]) if math.isfinite(value[0]) else 1.0
            totals["expected_damaged_line_km"] += p_e * length
        elif present[a] or present[b]:
            p = node_line[a] if present[a] else node_line[b]
            value = (math.log1p(-p) if p < 1.0 else -math.inf, 0.0, "facility")
            totals["facility_links"] += 1
        else:
            value = (0.0, 0.0, "none")
            totals["segments_without_hazard"] += 1
        segment_cache[key] = value
        return value

    flat, indptr = cache["path_nodes"], cache["path_indptr"]
    probabilities = np.zeros(len(cache["branch_uid"]))
    branch_length = np.zeros(len(probabilities))
    towers_on_branches = 0
    for branch in range(len(probabilities)):
        path = flat[indptr[branch]:indptr[branch + 1]]
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            s, length, _ = segment(int(a), int(b))
            total += s
            branch_length[branch] += length
        if args.towers == "mapped":
            interior = path[1:-1]
            supports = interior[is_support[interior]]
            towers_on_branches += len(supports)
            p_tower = np.clip(node_tower[supports], 0.0, 1.0)
            total += float(np.sum(np.log1p(-p_tower))) if np.all(p_tower < 1.0) else -math.inf
        probabilities[branch] = 1.0 - math.exp(total) if math.isfinite(total) else 1.0
    probabilities = np.clip(probabilities, 0.0, 1.0)

    # Self-check: under uniform wind the probability depends only on L / L0.
    q40 = float(fragility(np.array([40.0]), 25.0)[0])
    lengths = np.array([v[1] for v in segment_cache.values() if v[2] == "line"])
    fine = np.array([1.0 - math.exp(log_survival(np.full(max(1, math.ceil(L / (args.max_spacing_km / 2))), q40),
                                                  L / max(1, math.ceil(L / (args.max_spacing_km / 2))), args.l0_km))
                     for L in lengths])
    closed = 1.0 - (1.0 - q40) ** (lengths / args.l0_km)

    branch_ids = cache["branch_uid"].astype(int)
    atomic_csv(args.output, zip(branch_ids.tolist(), probabilities.tolist()))
    report = {
        "rule": "length-normalized", "towers": args.towers, "threshold_ms": args.threshold, "l0_km": args.l0_km,
        "max_spacing_km": args.max_spacing_km, "raster": str(args.raster), "wind_track": str(args.wind_track),
        "branches": int(len(probabilities)), "unique_segments": len(segment_cache), **totals,
        "support_nodes": int(is_support.sum()), "tower_references_on_branches": towers_on_branches,
        "expected_removed_branch_km": float(np.sum(probabilities * branch_length)),
        "branches_with_positive_probability": int(np.count_nonzero(probabilities > 0)),
        "mean_failure_probability": float(probabilities.mean()),
        "uniform_wind_check_max_abs_difference": float(np.abs(fine - closed).max()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
