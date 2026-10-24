#!/usr/bin/env python3
"""Write Cyclone Idai node wind layers for a sweep of line-fragility thresholds.

The hazard reproduces ``add_wind_to_power_network.update_fail_prob_for_gdf``:
the Idai wind raster is read at every node of ``merge_MG_gridfiner`` and the
same tower and line fragility curves are applied for each threshold.  Each
layer has the node schema of the synthetic-track wind folders, so it can be
passed unchanged to ``prepare_matpower_cyclone_input.py track``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import rasterio as rs

from utils import generate_shp_some_nodes


V_HALF_KNOTS = 110.0


def sample_wind(raster: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rs.open(raster) as src:
        values = src.read(1)
        rows, cols = rs.transform.rowcol(src.transform, xs=xs, ys=ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < values.shape[0]) & (cols >= 0) & (cols < values.shape[1])
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} network nodes fall outside {raster}")
    return values[rows, cols].astype(float)


def fragility(wind_ms: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    v_thresh = threshold * 3600 / 1850
    v_knots = wind_ms * 3600 / 1850
    v_n = np.maximum(0.0, v_knots - v_thresh) / (V_HALF_KNOTS - v_thresh)
    tower = v_n**3 / (1 + v_n**3)
    clipped = np.clip(wind_ms, threshold, threshold + 30)
    line = np.exp(0.6931 * (clipped - threshold) / 30) - 1
    return tower, line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(range(18, 31)))
    args = parser.parse_args()

    with args.network.open("rb") as handle:
        graph = pickle.load(handle)
    nodes = generate_shp_some_nodes(graph, list(graph.nodes))
    wind = sample_wind(
        args.raster,
        nodes.geometry.x.to_numpy(dtype=float),
        nodes.geometry.y.to_numpy(dtype=float),
    )

    reports = []
    for threshold in args.thresholds:
        tower, line = fragility(wind, threshold)
        layer = nodes.copy()
        layer["wind_intensity"] = wind
        layer["tower_frac_prop_lost"] = tower
        layer["line_frac_prop_lost"] = line
        output = args.output_root / f"idai_threshold_{threshold:02d}"
        output.parent.mkdir(parents=True, exist_ok=True)
        layer.to_file(output)
        reports.append({
            "threshold_ms": threshold,
            "nodes": int(len(layer)),
            "maximum_node_wind_ms": float(wind.max()),
            "nodes_above_threshold": int(np.count_nonzero(wind > threshold)),
            "nodes_with_positive_line_failure": int(np.count_nonzero(line > 0)),
            "nodes_with_positive_tower_failure": int(np.count_nonzero(tower > 0)),
            "output": str(output),
        })
        print(json.dumps(reports[-1]))
    report = {
        "schema_version": 1,
        "raster": str(args.raster),
        "network": str(args.network),
        "thresholds": reports,
    }
    (args.output_root / "wind_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
