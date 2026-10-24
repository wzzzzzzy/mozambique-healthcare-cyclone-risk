#!/usr/bin/env python3
"""Check direct wind-damage functions for healthcare facilities against Cyclone Idai.

Observed (PDNA, 2019): 94 health units damaged (4 destroyed, 90 partially),
about 14% of health infrastructure in the four assessed provinces (Sofala,
Manica, Tete, Zambezia).

Subcommands
-----------
winds     (Sherlock) peak Idai wind at every SARA facility from two hazards:
          * ``raster``: the Idai footprint used in the manuscript case study;
          * ``er11``: the ER11E04 profile along the IBTrACS track, computed as in
            ``prepare_historical_wind_track.py`` (same model as the synthetic set).
evaluate  apply each damage function and compare modelled and observed damage.

Both hazards are treated as 10-min mean winds.  Functions calibrated on
1-min winds (CLIMADA, Eberenz et al. 2021) receive ``V / c`` with the
10-min/1-min ratio ``c`` (0.93 main, 0.88 alternative).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PDNA_PROVINCES = ("Sofala", "Manica", "Tete", "Zambezia")
OBSERVED = {"damaged": 94, "destroyed": 4, "share": 0.14}
KNOT = 1850.0 / 3600.0  # m/s per knot, as in the manuscript code

# name: (V_thresh m/s, V_half m/s, wind basis, label)
FUNCTIONS = {
    "repo_110kn": (25.0, 110 * KNOT, "10min", "Current model (25 m/s, 110 kn)"),
    "emanuel_150kt": (50 * KNOT, 150 * KNOT, "10min", "Emanuel 2011 main (50 kt, 150 kt)"),
    "climada_default": (25.7, 74.7, "1min", "CLIMADA default (25.7, 74.7 m/s)"),
    "si_tdr": (25.7, 52.4, "1min", "Eberenz SI, TDR (25.7, 52.4 m/s)"),
    "si_rmsf": (25.7, 46.8, "1min", "Eberenz SI, RMSF (25.7, 46.8 m/s)"),
}


def emanuel(wind: np.ndarray, v_thresh: float, v_half: float) -> np.ndarray:
    x = np.clip(wind - v_thresh, 0.0, None) / (v_half - v_thresh)
    return x**3 / (1.0 + x**3)


def load_facilities(path: Path):
    import geopandas as gpd

    sara = gpd.read_file(path)
    sara["node_id"] = [f"node/46774825{str(i).zfill(4)}" for i in sara.index]
    return sara


def run_winds(args: argparse.Namespace) -> None:
    import matlab.engine
    import rasterio as rs
    from shapely.geometry import Point

    from prepare_historical_wind_track import KNOT_TO_MS, NAUTICAL_MILE_TO_M, haversine_m, load_country

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sara = load_facilities(args.hospitals)
    sara = sara[sara.geometry.notna()].copy()
    lon = sara.geometry.x.to_numpy(float)
    lat = sara.geometry.y.to_numpy(float)

    with rs.open(args.raster) as src:
        values = src.read(1).astype(float)
        rows, cols = rs.transform.rowcol(src.transform, xs=lon, ys=lat)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < values.shape[0]) & (cols >= 0) & (cols < values.shape[1])
    raster_wind = np.zeros(len(sara))
    raster_wind[inside] = values[rows[inside], cols[inside]]
    raster_wind = np.clip(raster_wind, 0.0, None)  # nodata is -1

    points = pd.read_csv(args.track_points)
    points = points.loc[points["track_id"].eq(args.track_id)]
    country = load_country(args.country_layer)
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    er11_wind = np.zeros(len(sara))
    engine = matlab.engine.start_matlab()
    try:
        engine.cd(str(args.chavas_directory), nargout=0)
        for row in points.itertuples(index=False):
            if not country.covers(Point(float(row.longitude), float(row.latitude))):
                continue
            vmax = float(row.wind_knots) * KNOT_TO_MS
            rmax = float(row.rmw_nm) * NAUTICAL_MILE_TO_M
            if not (np.isfinite(vmax) and np.isfinite(rmax) and vmax > 0 and rmax > 0):
                continue
            radial = engine.ER11E04_nondim_rmaxinput(vmax, rmax, 5e-5, 0, 1.5e-3, 2 / 1000, 0, 1.0, 0, 0.15,
                                                     nargout=11)
            radii = np.asarray(radial[0], dtype=float).reshape(-1)
            wind = np.asarray(radial[1], dtype=float).reshape(-1)
            distance = haversine_m(float(row.longitude), float(row.latitude), lon_rad, lat_rad)
            er11_wind = np.maximum(er11_wind, np.interp(distance, radii, wind, left=wind[0], right=0.0))
    finally:
        engine.quit()

    table = pd.DataFrame({
        "node_id": sara["node_id"], "facility_i": sara["facility_i"], "region": sara["region"],
        "type": sara["type"], "longitude": lon, "latitude": lat,
        "wind_raster_ms": raster_wind, "wind_er11_ms": er11_wind,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table[["wind_raster_ms", "wind_er11_ms"]].describe().to_string())


def run_evaluate(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    winds = pd.read_csv(args.winds)
    assessed = winds["region"].isin(PDNA_PROVINCES)
    rows = []
    for hazard in ("raster", "er11"):
        v10 = winds[f"wind_{hazard}_ms"].to_numpy(float)
        for name, (v_thresh, v_half, basis, label) in FUNCTIONS.items():
            ratios = (0.93, 0.88) if basis == "1min" else (None,)
            for ratio in ratios:
                wind = v10 / ratio if ratio else v10
                damage = emanuel(wind, v_thresh, v_half)
                for scope, mask in (("PDNA provinces", assessed), ("national", np.ones(len(winds), bool))):
                    d = damage[mask]
                    rows.append({
                        "hazard": hazard, "function": name, "label": label,
                        "ten_to_one_min_ratio": ratio, "scope": scope, "facilities": int(mask.sum()),
                        "expected_damaged": float(d.sum()),
                        "expected_share": float(d.mean()),
                        "count_ratio_gt_0.2": int((d > 0.2).sum()),
                        "count_ratio_gt_0.5": int((d > 0.5).sum()),
                        "facilities_wind_gt_threshold": int((wind[mask] > v_thresh).sum()),
                        "max_damage": float(d.max()),
                    })
    table = pd.DataFrame(rows)
    by_province = []
    for hazard in ("raster", "er11"):
        frame = winds.assign(wind=winds[f"wind_{hazard}_ms"])
        for region, group in frame.groupby("region"):
            entry = {"hazard": hazard, "region": region, "facilities": len(group),
                     "max_wind_ms": group["wind"].max(), "facilities_wind_gt_25": int((group["wind"] > 25).sum())}
            for name, (v_thresh, v_half, basis, _) in FUNCTIONS.items():
                wind = group["wind"] / 0.93 if basis == "1min" else group["wind"]
                entry[f"expected_{name}"] = float(emanuel(wind.to_numpy(float), v_thresh, v_half).sum())
            by_province.append(entry)

    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "idai_direct_damage_methods.csv", index=False)
    pd.DataFrame(by_province).to_csv(args.output_dir / "idai_direct_damage_by_province.csv", index=False)
    wind_summary = {
        hazard: {
            "max_ms": float(winds[f"wind_{hazard}_ms"].max()),
            "facilities_gt_25_ms_pdna": int((winds.loc[assessed, f"wind_{hazard}_ms"] > 25).sum()),
            "facilities_gt_33_ms_pdna": int((winds.loc[assessed, f"wind_{hazard}_ms"] > 33).sum()),
        }
        for hazard in ("raster", "er11")
    }
    (args.output_dir / "idai_direct_damage_report.json").write_text(json.dumps(
        {"observed": OBSERVED, "pdna_provinces": PDNA_PROVINCES,
         "facilities_in_pdna_provinces": int(assessed.sum()), "winds": wind_summary}, indent=2) + "\n")

    main = table[(table["scope"] == "PDNA provinces") & (table["ten_to_one_min_ratio"].fillna(0.93) == 0.93)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    labels = [FUNCTIONS[n][3] for n in FUNCTIONS]
    for axis, hazard, title in zip(axes, ("raster", "er11"), ("(a) Idai footprint raster", "(b) ER11E04 on IBTrACS track")):
        sub = main[main["hazard"] == hazard].set_index("function").loc[list(FUNCTIONS)]
        y = np.arange(len(sub))
        axis.barh(y - 0.27, sub["expected_damaged"], 0.27, color="#4c8c7a", label="Expected count (Σ damage ratio)")
        axis.barh(y, sub["count_ratio_gt_0.2"], 0.27, color="#e0a458", label="Facilities with ratio > 0.2")
        axis.barh(y + 0.27, sub["count_ratio_gt_0.5"], 0.27, color="#b5523b", label="Facilities with ratio > 0.5")
        axis.axvline(OBSERVED["damaged"], color="black", lw=1.2, ls="--", label="PDNA damaged (94)")
        axis.axvline(OBSERVED["destroyed"], color="black", lw=0.8, ls=":", label="PDNA destroyed (4)")
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_xlim(0, 105)
        axis.set_xlabel("Damaged healthcare facilities (PDNA provinces)", fontsize=9)
        axis.set_yticks(y, labels)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.27, wspace=0.08)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_dir / f"idai_direct_damage_check{suffix}", dpi=250, bbox_inches="tight")
    print(main[["hazard", "function", "expected_damaged", "expected_share", "count_ratio_gt_0.2",
                "count_ratio_gt_0.5", "facilities_wind_gt_threshold"]].round(3).to_string(index=False))


def run_map(args: argparse.Namespace) -> None:
    """Map Idai wind and facility damage ratios for every geolocated facility."""

    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")
    winds = pd.read_csv(args.winds)
    points = gpd.GeoDataFrame(winds, geometry=gpd.points_from_xy(winds["longitude"], winds["latitude"]),
                              crs="EPSG:4326")
    wind = points[f"wind_{args.hazard}_ms"].to_numpy(float)
    current = emanuel(wind, *FUNCTIONS["repo_110kn"][:2])
    regional = emanuel(wind / 0.93, *FUNCTIONS["si_tdr"][:2])
    countries = gpd.read_file(args.boundary).to_crs(epsg=4326)
    mozambique = countries[countries["COUNTRY"] == "Mozambique"]

    panels = [
        (wind, "viridis", Normalize(0, 50), "(a) Idai peak wind (m s$^{-1}$)"),
        (current, "OrRd", Normalize(0, 0.6), "(b) Current model: damage ratio"),
        (regional, "OrRd", Normalize(0, 0.6), "(c) Eberenz SI (TDR): damage ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2))
    for axis, (values, cmap, norm, title) in zip(axes, panels):
        mozambique.plot(ax=axis, color="gainsboro", edgecolor="black", linewidth=0.6)
        x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
        zero = values <= 1e-6
        axis.scatter(x[zero], y[zero], color="silver", s=3, linewidths=0)
        order = np.flatnonzero(~zero)[np.argsort(values[~zero])]
        axis.scatter(x[order], y[order], c=values[order], cmap=cmap, norm=norm, s=10, linewidths=0)
        if values is not wind:
            flagged = values > 0.2
            axis.scatter(x[flagged], y[flagged], s=26, facecolors="none", edgecolors="black", linewidths=0.6)
            axis.text(0.97, 0.03, f"Σ ratio = {values.sum():.1f}\nratio > 0.2: {int(flagged.sum())}",
                      transform=axis.transAxes, fontsize=8.5, va="bottom", ha="right")
        axis.set(xlim=(30, 41), ylim=(-27, -10), xticks=[], yticks=[], aspect="equal")
        axis.set_title(title, loc="left", fontsize=10)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, orientation="horizontal",
                           fraction=0.04, pad=0.02)
        bar.ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="none", mec="black",
                               label="Damage ratio > 0.2 (non-functional threshold, Espejo et al. 2025)")],
               loc="lower center", frameon=False, fontsize=8.5)
    fig.suptitle(f"Cyclone Idai, {len(points)} geolocated healthcare facilities "
                 f"({'footprint raster' if args.hazard == 'raster' else 'ER11E04 on IBTrACS track'})",
                 fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08, wspace=0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=250, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    winds = sub.add_parser("winds")
    winds.add_argument("--hospitals", type=Path, required=True)
    winds.add_argument("--raster", type=Path, required=True)
    winds.add_argument("--track-points", type=Path, required=True)
    winds.add_argument("--track-id", type=int, required=True)
    winds.add_argument("--country-layer", type=Path, required=True)
    winds.add_argument("--chavas-directory", type=Path, required=True)
    winds.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--winds", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    mapping = sub.add_parser("map")
    mapping.add_argument("--winds", type=Path, required=True)
    mapping.add_argument("--boundary", type=Path, required=True)
    mapping.add_argument("--hazard", choices=("raster", "er11"), default="raster")
    mapping.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    {"winds": run_winds, "evaluate": run_evaluate, "map": run_map}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
