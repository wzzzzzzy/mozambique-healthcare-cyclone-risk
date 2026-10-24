#!/usr/bin/env python3
"""Solar panel damage with exposure-dependent gust factors.

The solar fragility is defined for the 3-s gust (lognormal, median 80 m/s,
beta 0.32).  The modeled site wind V is a 10-min mean, and the gust is
``g * V`` with the WMO 3-s / 10-min gust factor for the site exposure
(Harper, Kepert and Ginger, 2010):

    inland (roughly open terrain)              g = 1.66
    coastal, onshore flow (wind from the sea)  g = 1.38
    coastal, offshore flow (wind from land)    g = 1.52
    (open sea, g = 1.23, does not apply: all facilities are on land)

A facility is coastal within ``--coastal-km`` of the ocean coastline
(Mozambique's boundary minus its land borders).  The flow direction is taken at
the time of the facility's peak wind: tangential to the storm (clockwise in the
southern hemisphere) turned ``--inflow-deg`` toward the centre; the flow is
onshore when it blows from the seaward side, i.e. against the direction from
the facility to its nearest coastline point.  The previous runs used g = 1.64
for every facility.

Subcommands
-----------
winds    peak wind and storm-to-site azimuth at the peak for the solar facilities,
         synthetic tracks [start, stop), same centres and ER11E04 profile as the
         production solar runs (one MATLAB engine).
convert  damage per facility and track for the chosen exposure rule, written in
         the layout of the production solar outputs (irradiance copied unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

GUST = {"inland": 1.66, "onshore": 1.38, "offshore": 1.52}
MEDIAN, BETA, PREVIOUS_GUST = 80.0, 0.32, 1.64
STORM_SETS = [f"STORM_DATA_IBTRACS_SI_1000_YEARS_{i}" for i in range(10)]
TRACK_COUNTS = (290, 274, 263, 276, 273, 279, 251, 264, 242, 291)


def solar_facilities(hospitals: Path):
    import geopandas as gpd

    sara = gpd.read_file(hospitals)
    solar = sara[(sara["ele_main_s"] == 3) | (sara["ele_sec_so"] == 3)]
    solar = solar[~solar["geometry"].isna()].copy()
    solar["node_id"] = [f"node/46774825{str(i).zfill(4)}" for i in solar.index]
    return solar


def damage(gust: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    out = np.zeros_like(gust, dtype=float)
    positive = gust > 0
    out[positive] = norm.cdf(np.log(gust[positive] / MEDIAN) / BETA)
    return out


def run_winds(args: argparse.Namespace) -> None:
    from pyproj import Geod

    from line_hazard_v2 import start_engine, storm_centres

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    solar = solar_facilities(args.hospitals)
    lon, lat = solar.geometry.x.to_numpy(float), solar.geometry.y.to_numpy(float)
    geod = Geod(ellps="WGS84")
    offsets = np.concatenate([[0], np.cumsum(TRACK_COUNTS)])
    ids, peaks, azimuths = [], [], []
    engine = None
    try:
        for global_id in range(args.global_start, min(args.global_stop, offsets[-1])):
            storm_set = int(np.searchsorted(offsets, global_id, side="right") - 1)
            track_id = global_id - int(offsets[storm_set])
            centres = storm_centres(args.storm_root, STORM_SETS[storm_set], track_id, args.boundary)
            peak = np.zeros(len(lon))
            azimuth = np.full(len(lon), np.nan)
            for c_lon, c_lat, rmax, vmax in centres:
                if engine is None:
                    engine = start_engine(args.chavas_dir)
                radial = engine.ER11E04_nondim_rmaxinput(vmax, rmax, 5e-5, 0, 1.5e-3, 2 / 1000, 0, 1.0, 0, 0.15,
                                                         nargout=11)
                radii = np.asarray(radial[0], dtype=float).reshape(-1)
                profile = np.asarray(radial[1], dtype=float).reshape(-1)
                az, _, d = geod.inv(np.full(len(lon), c_lon), np.full(len(lon), c_lat), lon, lat)
                wind = np.maximum(0.0, np.where(d > radii[-1], 0.0, np.interp(d, radii, profile)))
                higher = wind > peak
                peak[higher] = wind[higher]
                azimuth[higher] = az[higher]
            ids.append(global_id)
            peaks.append(peak)
            azimuths.append(azimuth)
            print(f"{STORM_SETS[storm_set]}:{track_id} centres {len(centres)} max wind {peak.max():.2f}", flush=True)
    finally:
        if engine is not None:
            engine.quit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, global_ids=np.array(ids), node_ids=solar["node_id"].to_numpy(),
                        peak_wind=np.array(peaks, dtype=np.float32), azimuth_at_peak=np.array(azimuths, dtype=np.float32))


def coast_geometry(boundary: Path):
    import geopandas as gpd

    countries = gpd.read_file(boundary).to_crs(epsg=4326)
    mozambique = countries[countries["COUNTRY"] == "Mozambique"].geometry.iloc[0]
    neighbours = countries[(countries["COUNTRY"] != "Mozambique") & countries.intersects(mozambique.buffer(0.05))]
    # Remove the land borders one neighbour at a time (unary_union segfaults with
    # the shapely 1.8 build on Sherlock); the result is the same coastline.
    coast = mozambique.boundary
    for geometry in neighbours.geometry:
        coast = coast.difference(geometry.buffer(0.02))
    return coast


def run_convert(args: argparse.Namespace) -> None:
    import geopandas as gpd
    from pyproj import Geod
    from scipy.stats import norm
    from shapely.ops import nearest_points

    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    solar = solar_facilities(args.hospitals)
    coast = coast_geometry(args.boundary)
    coast_m = gpd.GeoSeries([coast], crs=4326).to_crs(32737).iloc[0]
    points_m = solar.to_crs(32737).geometry
    distance_km = (points_m.distance(coast_m) / 1000.0).to_numpy()
    nearest = gpd.GeoSeries([nearest_points(p, coast_m)[1] for p in points_m], crs=32737).to_crs(4326)
    geod = Geod(ellps="WGS84")
    seaward, _, _ = geod.inv(solar.geometry.x.to_numpy(), solar.geometry.y.to_numpy(),
                             nearest.x.to_numpy(), nearest.y.to_numpy())
    coastal = distance_km <= args.coastal_km

    chunks = [np.load(p, allow_pickle=True) for p in sorted(args.winds_dir.glob("solar_winds_*.npz"))]
    ids = np.concatenate([c["global_ids"] for c in chunks])
    order = np.argsort(ids)
    peak = np.concatenate([c["peak_wind"] for c in chunks])[order].astype(float)
    az_peak = np.concatenate([c["azimuth_at_peak"] for c in chunks])[order].astype(float)
    node_ids = chunks[0]["node_ids"].astype(str)
    if not np.array_equal(ids[order], np.arange(sum(TRACK_COUNTS))):
        raise ValueError("solar wind chunks do not cover the catalogue")
    if not np.array_equal(node_ids, solar["node_id"].to_numpy()):
        raise ValueError("facility order differs from the wind chunks")

    wind_to = az_peak + 90.0 + args.inflow_deg
    onshore = np.cos(np.radians(wind_to - seaward[None, :])) < 0
    gust_factor = np.where(coastal[None, :], np.where(onshore, GUST["onshore"], GUST["offshore"]), GUST["inland"])
    new_damage = damage(gust_factor * peak)

    args.output_root.mkdir(parents=True)
    checks, offset = {"max_wind_vs_saved_damage_inversion": 0.0}, 0
    old_sum = new_sum = 0.0
    for folder, count in zip(STORM_SETS, TRACK_COUNTS):
        source = args.source_root / f"{folder}_var_track_solar_damage_func"
        frame = gpd.read_file(source)
        rows = np.array([int(np.flatnonzero(node_ids == n)[0]) for n in frame["node_id"].astype(str)])
        for t in range(count):
            column = str(t)
            old = frame[column].astype(float).to_numpy()
            valid = np.isfinite(old)
            # Saved damage used gust = 1.64 V: invert where it carries enough digits.
            usable = valid & (old > 1e-9) & (old < 1 - 1e-9)
            recovered = MEDIAN * np.exp(BETA * norm.ppf(old[usable])) / PREVIOUS_GUST
            if usable.any():
                diff = np.abs(recovered - peak[offset + t, rows[usable]]).max()
                checks["max_wind_vs_saved_damage_inversion"] = max(checks["max_wind_vs_saved_damage_inversion"], float(diff))
            value = new_damage[offset + t, rows]
            frame[column] = np.where(valid, value, np.nan)
            old_sum += float(old[valid].sum())
            new_sum += float(value[valid].sum())
        frame.to_file(args.output_root / source.name)
        irr = args.source_root / f"{folder}_var_track_solar_irr_func"
        shutil.copytree(irr, args.output_root / irr.name)
        offset += count
        print(f"{folder}: written", flush=True)

    cells = gust_factor.size
    report = {
        "gust_factors": GUST, "coastal_km": args.coastal_km, "inflow_deg": args.inflow_deg,
        "coastline_km": float(coast_m.length / 1000.0),
        "facilities": int(len(solar)), "coastal_facilities": int(coastal.sum()),
        "facility_track_cells_by_class": {
            "inland": int((gust_factor == GUST["inland"]).sum()),
            "coastal_onshore": int((gust_factor == GUST["onshore"]).sum()),
            "coastal_offshore": int((gust_factor == GUST["offshore"]).sum()), "total": int(cells)},
        "coastal_cells_with_wind_ge_24": {
            "onshore": int(((gust_factor == GUST["onshore"]) & (peak >= 24)).sum()),
            "offshore": int(((gust_factor == GUST["offshore"]) & (peak >= 24)).sum())},
        "mean_damage_previous_1.64": old_sum / cells, "mean_damage_exposure": new_sum / cells,
        "checks": checks,
    }
    for band in (2.0, 5.0, 10.0, 20.0):
        c = distance_km <= band
        g = np.where(c[None, :], np.where(onshore, GUST["onshore"], GUST["offshore"]), GUST["inland"])
        report.setdefault("mean_damage_by_coastal_km", {})[f"{band:g}"] = float(damage(g * peak).mean())
    (args.output_root / "solar_exposure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_root / "facility_exposure.npz", node_ids=node_ids, distance_km=distance_km,
                        seaward_azimuth=seaward, coastal=coastal)
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("winds")
    w.add_argument("--hospitals", type=Path, required=True)
    w.add_argument("--storm-root", type=Path, required=True)
    w.add_argument("--boundary", type=Path, required=True)
    w.add_argument("--chavas-dir", type=Path, required=True)
    w.add_argument("--global-start", type=int, required=True)
    w.add_argument("--global-stop", type=int, required=True)
    w.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("convert")
    c.add_argument("--hospitals", type=Path, required=True)
    c.add_argument("--boundary", type=Path, required=True)
    c.add_argument("--winds-dir", type=Path, required=True)
    c.add_argument("--source-root", type=Path, required=True, help="production synthetic_solar")
    c.add_argument("--coastal-km", type=float, default=5.0)
    c.add_argument("--inflow-deg", type=float, default=20.0)
    c.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    {"winds": run_winds, "convert": run_convert}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
