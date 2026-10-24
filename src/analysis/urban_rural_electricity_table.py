#!/usr/bin/env python3
"""Urban/rural table of healthcare-facility electricity supply and cyclone disruption.

Groups follow Figure 2 (SARA ``urban``: 1 urban, 2 rural; generator present
when ``has_genera`` is "yes").  Supply configuration uses the SARA 2018
questionnaire: Q410 main source (``ele_main_s``: 1 public grid, 2 generator,
3 solar, 4 gas, 5 private grid, 96 other; missing = no electricity, Q408) and
Q411 back-up source (``ele_sec_so``: 0 none, 1 private grid, 2 generator,
3 solar, 96 other).

Annualized electricity disruption index, s_h = (1/T) sum_z s_h(z), T = 10,000 yr:
* grid-supplied facilities: hybrid power model (``df_final_hybrid_25.csv``);
* solar-supplied facilities: independent solar model (the ``solar_only`` rows
  of the all-grid comparison table, i.e. the same values as Figure 2c-d).
Urban-rural differences are tested with a two-sided Mann-Whitney U test
(normal approximation with tie correction).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ANNUAL = 2703 / 10_000
GROUPS = ("Urban", "Rural")
MAIN = {1: "Public grid", 3: "Solar", 2: "Generator", 4: "Gas", 5: "Other", 96: "Other"}


def mann_whitney(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    values = np.concatenate([a, b])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(values, return_counts=True)
    n = n1 + n2
    variance = n1 * n2 / 12 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1)))
    z = (u - n1 * n2 / 2) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2))


def describe(values: pd.Series) -> str:
    v = values.dropna()
    if v.empty:
        return "--"
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    return f"{v.mean():.4f} ({median:.4f}; {q1:.4f}–{q3:.4f})"


def pct(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.0f}%)" if total else "0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hospitals", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--solar-per-hospital", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    sara = gpd.read_file(args.hospitals)
    sara["node_id"] = [f"node/46774825{str(i).zfill(4)}" for i in sara.index]
    public = pd.read_csv(args.public_summary)
    tracks = [c for c in public.columns if c not in ("node_id", "facility_i")]
    public["grid_index"] = public[tracks].where(public[tracks] >= 0).mean(axis=1) * ANNUAL
    solar = pd.read_csv(args.solar_per_hospital)
    solar = solar.loc[solar["baseline_source"] == "solar_only", ["node_id", "baseline_mean"]]
    solar["solar_index"] = solar["baseline_mean"].where(solar["baseline_mean"] >= 0) * ANNUAL

    f = sara.merge(public[["node_id", "grid_index"]], on="node_id", how="left")
    f = f.merge(solar[["node_id", "solar_index"]], on="node_id", how="left")
    f["group"] = pd.to_numeric(f["urban"], errors="coerce").map({1: "Urban", 2: "Rural"})
    f["generator"] = f["has_genera"].astype(str).str.strip().str.lower().eq("yes")
    main_source = pd.to_numeric(f["ele_main_s"], errors="coerce")
    backup = pd.to_numeric(f["ele_sec_so"], errors="coerce")
    f["main"] = main_source.map(MAIN).fillna("No electricity")
    grid = main_source.eq(1)

    rows: list[tuple[str, dict[str, str]]] = []

    def add(label, func):
        rows.append((label, {**{g: func(f[f["group"] == g]) for g in GROUPS}, "All": func(f)}))

    def grid_backup(d, code):
        return pct(int((grid[d.index] & backup[d.index].eq(code)).sum()), int(grid[d.index].sum()))

    add("Facilities", lambda d: str(len(d)))
    add("Main source: public grid", lambda d: pct(int((d["main"] == "Public grid").sum()), len(d)))
    add("Main source: solar", lambda d: pct(int((d["main"] == "Solar").sum()), len(d)))
    add("Main source: generator, gas or other",
        lambda d: pct(int(d["main"].isin(["Generator", "Gas", "Other"]).sum()), len(d)))
    add("No electricity", lambda d: pct(int((d["main"] == "No electricity").sum()), len(d)))
    add("Grid-supplied with generator back-up", lambda d: grid_backup(d, 2))
    add("Grid-supplied with solar back-up", lambda d: grid_backup(d, 3))
    add("Generator present", lambda d: pct(int(d["generator"].sum()), len(d)))
    add("Grid-supplied, modelled (n)", lambda d: str(int(d["grid_index"].notna().sum())))
    add("  index, mean (median; IQR), yr⁻¹", lambda d: describe(d["grid_index"]))
    add("  index > 0.02 yr⁻¹",
        lambda d: pct(int((d["grid_index"] > 0.02).sum()), int(d["grid_index"].notna().sum())))
    add("Solar-supplied, modelled (n)", lambda d: str(int(d["solar_index"].notna().sum())))
    add("  index, mean (median; IQR), yr⁻¹ ", lambda d: describe(d["solar_index"]))
    add("All modelled facilities, mean index, yr⁻¹",
        lambda d: f"{pd.concat([d['grid_index'], d['solar_index']]).mean():.4f}")

    urban, rural = f[f["group"] == "Urban"], f[f["group"] == "Rural"]
    tests = {
        "grid_index_urban_vs_rural_p": mann_whitney(urban["grid_index"].to_numpy(float),
                                                    rural["grid_index"].to_numpy(float)),
        "solar_index_urban_vs_rural_p": mann_whitney(urban["solar_index"].to_numpy(float),
                                                     rural["solar_index"].to_numpy(float)),
        "grid_index_generator_vs_none_p": mann_whitney(f.loc[f["generator"], "grid_index"].to_numpy(float),
                                                       f.loc[~f["generator"], "grid_index"].to_numpy(float)),
        "modelled_any_index_mean": {g: float(pd.concat([d["grid_index"], d["solar_index"]]).mean())
                                    for g, d in (("Urban", urban), ("Rural", rural), ("All", f))},
    }

    table = pd.DataFrame([{"": label, **values} for label, values in rows])
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "urban_rural_electricity_table.csv", index=False)
    f.drop(columns="geometry").to_csv(args.output_dir / "urban_rural_facility_values.csv", index=False)
    (args.output_dir / "urban_rural_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    (args.output_dir / "urban_rural_electricity_table.md").write_text(table.to_markdown(index=False) + "\n")
    latex = table.rename(columns={"": "Characteristic"}).to_latex(index=False, escape=False, column_format="lrrr")
    latex = latex.replace("%", r"\%").replace("yr⁻¹", r"yr$^{-1}$").replace("–", "--")
    (args.output_dir / "urban_rural_electricity_table.tex").write_text(latex)
    print(table.to_string(index=False))
    print(json.dumps(tests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
