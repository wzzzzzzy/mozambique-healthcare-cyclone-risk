#!/usr/bin/env python3
"""Figure S8: reconstructed multilayer infrastructure representation.

Port of the two panels of ``renzhi.ipynb`` (cells 18 and 20) to the current
model inputs:

(a) national map: electricity infrastructure (yellow), water infrastructure
    (cyan) and healthcare facilities (red); solid lines connect elements within
    a system, dotted lines are dependencies between layers, coloured by the
    supplying layer (electricity -> water / facility in yellow, water ->
    facility in cyan).  The box marks the enlarged view.
(b) enlarged layer view: the same elements inside a window, stacked by layer
    (electricity on top, water in the middle, facilities at the bottom).

Electricity comes from ``simplified_hospital_MG.pickle`` (the hybrid model's
grid and facility connections); water from the rebuilt OSM water graph
(``MG_w_hospital_new6.pickle``: water works, water towers, covered reservoirs
and their electricity and facility links).  Communication is not shown.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils import get_node_pos

ELECTRICITY = "#f2c200"   # yellow
WATER = "#00bcd4"         # cyan
FACILITY = "#d62728"      # red
POWER_TYPES = {"portal", "substation", "gridfinder", "transformer"}
WATER_TYPES = {"water_works", "water_tower", "reservoir_covered"}
LAYER_Z = {"p": 30, "w": 20, "h": 10}
COLOR = {"p": ELECTRICITY, "w": WATER, "h": FACILITY}


def load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def xy(graph, node) -> tuple[float, float]:
    point = get_node_pos(graph, node)
    return point.x, point.y


def collect(power, water):
    """Nodes and edges by layer: within-system (solid) and dependency (dotted) links."""

    nodes = {"p": {}, "w": {}, "h": {}}
    solid = {"p": [], "w": []}
    dotted = {"p": [], "w": []}  # keyed by the supplying layer
    for n, data in power.nodes(data=True):
        nodes["h" if data.get("power") == "hospital" else "p"][n] = xy(power, n)
    for u, v in power.edges():
        a = "h" if power.nodes[u].get("power") == "hospital" else "p"
        b = "h" if power.nodes[v].get("power") == "hospital" else "p"
        segment = (nodes[a][u], nodes[b][v])
        (solid["p"] if a == b == "p" else dotted["p"]).append(segment)

    def layer(data):
        kind = data.get("water")
        return "w" if kind in WATER_TYPES else "h" if kind == "hospital" else "p" if kind in POWER_TYPES else None

    for n, data in water.nodes(data=True):
        k = layer(data)
        if k is not None:
            nodes[k].setdefault(n, xy(water, n))
    for u, v in water.edges():
        a, b = layer(water.nodes[u]), layer(water.nodes[v])
        if a is None or b is None:
            continue
        segment = (xy(water, u), xy(water, v))
        if a == b == "w":
            solid["w"].append(segment)
        elif "p" in (a, b):
            dotted["p"].append(segment)          # electricity supplies the water asset
        else:
            dotted["w"].append(segment)          # water supplies the facility
    return nodes, solid, dotted


def inside(point, window) -> bool:
    x0, x1, y0, y1 = window
    return x0 < point[0] < x1 and y0 < point[1] < y1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power-graph", type=Path, required=True, help="simplified_hospital_MG.pickle")
    parser.add_argument("--water-graph", type=Path, required=True, help="MG_w_hospital_new6.pickle")
    parser.add_argument("--boundary", type=Path, required=True, help="World_Countries layer")
    parser.add_argument("--window", type=float, nargs=4, default=[32.4, 32.8, -26.2, -25.8],
                        metavar=("LON0", "LON1", "LAT0", "LAT1"), help="enlarged view (default Maputo)")
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    nodes, solid, dotted = collect(load(args.power_graph), load(args.water_graph))
    countries = gpd.read_file(args.boundary)
    mozambique = countries[countries["COUNTRY"] == "Mozambique"]

    fig = plt.figure(figsize=(13, 8.6))
    ax = fig.add_axes([0.02, 0.1, 0.4, 0.86])
    mozambique.boundary.plot(ax=ax, color="black", linewidth=1.4)
    ax.add_collection(LineCollection(solid["p"], colors=ELECTRICITY, linewidths=0.9, zorder=2))
    ax.add_collection(LineCollection(solid["w"], colors=WATER, linewidths=1.1, zorder=3))
    ax.add_collection(LineCollection(dotted["p"], colors=ELECTRICITY, linewidths=0.8, linestyles="dotted", zorder=2))
    ax.add_collection(LineCollection(dotted["w"], colors=WATER, linewidths=0.8, linestyles="dotted", zorder=3))
    for k, size, alpha, z in (("p", 3, 0.35, 2), ("w", 7, 1.0, 4), ("h", 5, 1.0, 5)):
        pts = np.array(list(nodes[k].values()))
        ax.scatter(pts[:, 0], pts[:, 1], s=size, color=COLOR[k], alpha=alpha, linewidths=0, zorder=z)
    x0, x1, y0, y1 = args.window
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", linewidth=1.0, zorder=6))
    ax.set(xlim=(30, 41), ylim=(-27, -10), xticks=[], yticks=[], aspect="equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(a)", loc="left", fontsize=11)

    ax3 = fig.add_axes([0.42, 0.08, 0.57, 0.9], projection="3d")
    grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
    for k in ("h", "w", "p"):
        ax3.plot_surface(grid_x, grid_y, np.full_like(grid_x, LAYER_Z[k]), color=COLOR[k], alpha=0.07, linewidth=0)

    def lifted(segments, za, zb):
        return [((a[0], a[1], za), (b[0], b[1], zb)) for a, b in segments if inside(a, args.window) and inside(b, args.window)]

    def z_of(point):
        for k in ("h", "w", "p"):
            if point in nodes_in[k]:
                return LAYER_Z[k]
        return LAYER_Z["p"]

    nodes_in = {k: {p for p in v.values() if inside(p, args.window)} for k, v in nodes.items()}
    ax3.add_collection3d(Line3DCollection(lifted(solid["p"], LAYER_Z["p"], LAYER_Z["p"]), colors=ELECTRICITY, linewidths=1.2))
    ax3.add_collection3d(Line3DCollection(lifted(solid["w"], LAYER_Z["w"], LAYER_Z["w"]), colors=WATER, linewidths=1.2))
    for k in ("p", "w"):
        segs = [((a[0], a[1], z_of(a)), (b[0], b[1], z_of(b))) for a, b in dotted[k]
                if inside(a, args.window) and inside(b, args.window)]
        ax3.add_collection3d(Line3DCollection(segs, colors=COLOR[k], linewidths=0.9, linestyles="dotted"))
    for k in ("p", "w", "h"):
        pts = np.array(list(nodes_in[k])) if nodes_in[k] else np.empty((0, 2))
        ax3.scatter(pts[:, 0], pts[:, 1], np.full(len(pts), LAYER_Z[k]), color=COLOR[k], s=14, depthshade=False)
    ax3.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(5, 35), xticks=[], yticks=[], zticks=[])
    ax3.grid(False)
    ax3.set_title("(b)", loc="left", fontsize=11)

    fig.legend(handles=[
        Line2D([], [], color=ELECTRICITY, lw=2, label="Electricity"),
        Line2D([], [], color=WATER, lw=2, label="Water"),
        Line2D([], [], marker="o", ls="none", color=FACILITY, label="Healthcare facility"),
        Line2D([], [], color="0.4", lw=1.5, label="Connection within a system"),
        Line2D([], [], color="0.4", lw=1.5, ls="dotted", label="Dependency between layers"),
    ], loc="lower center", ncol=5, frameon=False, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    report = {k: len(v) for k, v in nodes.items()} | {
        "electricity_lines": len(solid["p"]), "water_lines": len(solid["w"]),
        "electricity_dependencies": len(dotted["p"]), "water_dependencies": len(dotted["w"]),
        "window": args.window, "window_nodes": {k: len(v) for k, v in nodes_in.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
