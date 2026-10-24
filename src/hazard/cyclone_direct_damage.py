#!/usr/bin/env python3
"""Direct wind-damage ratio of every healthcare facility for every synthetic cyclone.

Site winds are the saved peak winds of the hospital wind run
(``synthetic_new_hospital``: ``{set}_{track}_w``; ER11E04 profile driven by the
STORM 10-min intensity).  Each damage function is applied per facility and
cyclone; the main function is the Eberenz et al. (2021) South-Indian
calibration (TDR) on 1-min winds (10-min / 0.93), with a facility counted as
non-functional when its damage ratio exceeds 0.2 (Espejo et al., 2025).

Checks
------
* the saved building damage ``_b`` equals the current function of ``_w``;
* the winds reproduce the published solar damage (gust = 1.64 x wind) at the
  solar facilities, i.e. both runs used the same site winds.

Outputs (``--output-dir``)
--------------------------
damage_ratio_<function>.csv.gz   facility x cyclone damage ratio
cyclone_summary.csv              per cyclone: winds, sum of ratios, counts > 0.2 / > 0.5
facility_summary.csv             per facility: mean ratio, annual non-functional rate
direct_damage_report.json        checks and catalogue totals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STORM_SETS = [f"STORM_DATA_IBTRACS_SI_1000_YEARS_{i}" for i in range(10)]
TRACK_COUNTS = (290, 274, 263, 276, 273, 279, 251, 264, 242, 291)
CATALOGUE_YEARS = 10_000.0
KNOT = 1850.0 / 3600.0
TEN_TO_ONE_MIN = 0.93
NONFUNCTIONAL = 0.2

# name: (V_thresh m/s, V_half m/s, wind basis)
FUNCTIONS = {
    "si_tdr": (25.7, 52.4, "1min"),
    "si_rmsf": (25.7, 46.8, "1min"),
    "current": (25.0, 110 * KNOT, "10min"),
    "climada_default": (25.7, 74.7, "1min"),
    "emanuel_150kt": (50 * KNOT, 150 * KNOT, "10min"),
}


def emanuel(wind: np.ndarray, v_thresh: float, v_half: float) -> np.ndarray:
    x = np.clip(wind - v_thresh, 0.0, None) / (v_half - v_thresh)
    return x**3 / (1.0 + x**3)


def damage(wind: np.ndarray, name: str) -> np.ndarray:
    v_thresh, v_half, basis = FUNCTIONS[name]
    return emanuel(wind / TEN_TO_ONE_MIN if basis == "1min" else wind, v_thresh, v_half)


def load_winds(root: Path):
    import geopandas as gpd

    identity, winds, saved_b, columns = None, [], [], []
    for storm_set, (folder, count) in enumerate(zip(STORM_SETS, TRACK_COUNTS)):
        frame = gpd.read_file(root / f"track_{folder}")
        frame = frame.set_index("node_id")
        if identity is None:
            identity = frame[["facility_i", "region", "type", "latitude", "longitude"]].copy()
            identity["geolocated"] = frame.geometry.notna() & ~frame.geometry.is_empty
        elif not frame.index.equals(identity.index):
            raise ValueError(f"facility order differs in {folder}")
        w = frame[[f"{storm_set}_{t}_w" for t in range(count)]].apply(pd.to_numeric, errors="coerce")
        b = frame[[f"{storm_set}_{t}_b" for t in range(count)]].apply(pd.to_numeric, errors="coerce")
        winds.append(w.to_numpy(float))
        saved_b.append(b.to_numpy(float))
        columns += [f"{storm_set}_{t}" for t in range(count)]
    return identity, np.hstack(winds), np.hstack(saved_b), columns


def solar_check(solar_root: Path, identity: pd.DataFrame, wind: np.ndarray, columns: list[str]) -> dict:
    import geopandas as gpd
    from scipy.stats import norm

    position = {node: i for i, node in enumerate(identity.index)}
    column = {c: j for j, c in enumerate(columns)}
    worst, cells = 0.0, 0
    for storm_set, (folder, count) in enumerate(zip(STORM_SETS, TRACK_COUNTS)):
        path = solar_root / f"{folder}_var_track_solar_damage_func"
        saved = gpd.read_file(path).set_index("node_id")
        rows = np.array([position[n] for n in saved.index])
        for t in range(count):
            observed = pd.to_numeric(saved[str(t)], errors="coerce").to_numpy(float)
            v = wind[rows, column[f"{storm_set}_{t}"]]
            gust = 1.64 * v
            predicted = np.where(gust > 0, norm.cdf(np.log(np.where(gust > 0, gust, 1.0) / 80.0) / 0.32), 0.0)
            valid = np.isfinite(observed)
            worst = max(worst, float(np.abs(predicted[valid] - observed[valid]).max(initial=0.0)))
            cells += int(valid.sum())
    return {"solar_cells_compared": cells, "solar_max_abs_difference": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hospital-wind-root", type=Path, required=True)
    parser.add_argument("--solar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    identity, raw_wind, saved_b, columns = load_winds(args.hospital_wind_root)
    located = identity["geolocated"].to_numpy()
    # Tracks with no centre inside Mozambique were saved as all-missing columns;
    # they carry no local wind.
    empty_tracks = [c for j, c in enumerate(columns) if np.isnan(raw_wind[located, j]).all()]
    wind = np.where(located[:, None], np.nan_to_num(np.clip(raw_wind, 0.0, None), nan=0.0), np.nan)

    current_from_wind = damage(wind, "current")
    b = np.where(np.isfinite(saved_b), saved_b, 0.0)
    report = {
        "facilities": int(len(identity)),
        "geolocated": int(located.sum()),
        "cyclones": len(columns),
        "empty_tracks_set_to_zero_wind": len(empty_tracks),
        "check_saved_building_damage_max_abs_difference": float(
            np.nanmax(np.abs(current_from_wind[located] - b[located]))),
        **solar_check(args.solar_root, identity, wind, columns),
        "functions": {k: {"v_thresh_ms": v[0], "v_half_ms": round(v[1], 3), "wind_basis": v[2]}
                      for k, v in FUNCTIONS.items()},
        "ten_to_one_minute_ratio": TEN_TO_ONE_MIN,
        "nonfunctional_threshold": NONFUNCTIONAL,
    }

    args.output_dir.mkdir(parents=True)
    head = identity[["facility_i", "region", "type"]].reset_index()
    storm_set = [int(c.split("_")[0]) for c in columns]
    summary = pd.DataFrame({"cyclone": columns, "storm_set": storm_set,
                            "track_id": [int(c.split("_")[1]) for c in columns],
                            "max_facility_wind_ms": np.nanmax(wind, axis=0),
                            "facilities_wind_gt_25_ms": (wind > 25).sum(axis=0)})
    facility = head.copy()
    facility["geolocated"] = located
    facility["max_wind_ms"] = np.nanmax(wind, axis=1)
    totals = {}
    for name in FUNCTIONS:
        ratio = damage(wind, name)
        summary[f"{name}_sum_ratio"] = np.nansum(ratio, axis=0)
        summary[f"{name}_count_gt_0.2"] = (ratio > NONFUNCTIONAL).sum(axis=0)
        summary[f"{name}_count_gt_0.5"] = (ratio > 0.5).sum(axis=0)
        facility[f"{name}_mean_ratio_per_cyclone"] = np.nanmean(ratio, axis=1)
        facility[f"{name}_annual_mean_ratio"] = np.nansum(ratio, axis=1) / CATALOGUE_YEARS
        facility[f"{name}_annual_nonfunctional_rate"] = (ratio > NONFUNCTIONAL).sum(axis=1) / CATALOGUE_YEARS
        totals[name] = {
            "annual_expected_sum_ratio": float(np.nansum(ratio) / CATALOGUE_YEARS),
            "annual_nonfunctional_facility_events": float((ratio > NONFUNCTIONAL).sum() / CATALOGUE_YEARS),
            "cyclones_with_any_nonfunctional": int((summary[f"{name}_count_gt_0.2"] > 0).sum()),
            "max_nonfunctional_in_one_cyclone": int(summary[f"{name}_count_gt_0.2"].max()),
        }
        if name in ("si_tdr", "current"):
            matrix = pd.concat([head, pd.DataFrame(ratio, columns=columns)], axis=1)
            matrix.to_csv(args.output_dir / f"damage_ratio_{name}.csv.gz", index=False, float_format="%.6g")
    summary.to_csv(args.output_dir / "cyclone_summary.csv", index=False)
    facility.to_csv(args.output_dir / "facility_summary.csv", index=False)
    report["totals"] = totals
    (args.output_dir / "direct_damage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
