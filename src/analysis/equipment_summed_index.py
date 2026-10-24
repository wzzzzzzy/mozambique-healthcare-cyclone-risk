#!/usr/bin/env python3
"""Summed disruption index for equipment groups, split by reported generator.

Back-of-the-envelope aggregation of the saved facility values written by
``facility_type_electricity_services.py`` (``facility_values.csv``).  For the
evaluated grid-supplied facilities recording an equipment item E,

    B_E = sum_{h in E} s_h = N_E x mean index   (yr^-1),

the frequency-weighted sum of fractional facility electricity shortfalls, in
full-facility-shortfall equivalents per year (ten facilities with a 10%
shortfall each contribute one equivalent).  Each facility has equal weight; B_E
does not measure equipment electricity use, interrupted treatments or the
number of facilities losing all power.

The share of B_E at facilities without a reported generator is given for two
definitions: no generator named as main or alternative source (Q410/Q411), and
no generator reported functioning on the survey day (Q413).  Neither absence
establishes the absence of every alternative supply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GROUPS = {
    "Oxygen concentrators": "oxygen",
    "Neonatal incubators": "incubator",
    "Grid/generator-powered vaccine refrigerators": "fridge_grid",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facility-values", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    f = pd.read_csv(args.facility_values)
    for column in ["generator_reported", "generator_functioning", *GROUPS.values()]:
        f[column] = f[column].astype(str).str.lower().eq("true")
    grid = f[(f["model"] == "grid") & f["index"].notna()]

    rows, numbers = [], {}
    for label, column in GROUPS.items():
        d = grid[grid[column]]
        total = float(d["index"].sum())
        entry = {"facilities": len(d), "mean_index": float(d["index"].mean()), "summed_index": total}
        for key, flag in (("no_generator_reported", "generator_reported"),
                          ("no_functioning_generator_reported", "generator_functioning")):
            without = d[~d[flag]]
            entry[key] = {"facilities": len(without), "summed_index": float(without["index"].sum()),
                          "share_of_summed_index": float(without["index"].sum() / total)}
        numbers[label] = entry
        rows.append({
            "Equipment at evaluated grid facilities": label,
            "Facilities": len(d),
            "Mean index (yr⁻¹)": f"{entry['mean_index']:.4f}",
            "Summed index (yr⁻¹)": f"{total:.2f}",
            "Without reported generator": entry["no_generator_reported"]["facilities"],
            "Their summed index (yr⁻¹)": f"{entry['no_generator_reported']['summed_index']:.2f}",
            "Share": f"{100 * entry['no_generator_reported']['share_of_summed_index']:.0f}%",
            "Without functioning generator": entry["no_functioning_generator_reported"]["facilities"],
            "Their summed index (yr⁻¹) ": f"{entry['no_functioning_generator_reported']['summed_index']:.2f}",
            "Share ": f"{100 * entry['no_functioning_generator_reported']['share_of_summed_index']:.0f}%",
        })
    table = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "equipment_summed_index.csv", index=False)
    (args.output_dir / "equipment_summed_index.md").write_text(table.to_markdown(index=False) + "\n")
    (args.output_dir / "equipment_summed_index.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(table.to_string(index=False))
    print(json.dumps(numbers, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
