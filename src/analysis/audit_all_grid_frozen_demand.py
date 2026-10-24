#!/usr/bin/env python3
"""Check that the frozen-demand all-grid run changes facility dependencies only.

For every track completed in the frozen-demand run this compares

* facility results with the published current-system run (``--baseline-root``)
  for the facilities already on the public grid; with the baseline demand
  vector frozen these must be identical, and
* sample records with the published all-grid run (``--all-grid-root``); the
  failed-edge counts must be identical (same damage draws on existing lines
  and on the added facility links), while demand and served power may differ.

Writes a per-track CSV and a JSON report; exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DRAW_COLUMNS = ["sample_id", "failed_edges", "failed_retained_edges"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--all-grid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    for name in ("frozen_demand_audit_tracks.csv", "frozen_demand_audit_report.json"):
        if (args.output_dir / name).exists():
            raise SystemExit(f"refusing to overwrite {args.output_dir / name}")

    rows = []
    for frozen_path in sorted(args.frozen_root.glob("results/dataset_*/track_*_hospitals.csv")):
        relative = frozen_path.relative_to(args.frozen_root)
        frozen = pd.read_csv(frozen_path).set_index("hospital_id")
        baseline = pd.read_csv(args.baseline_root / relative).set_index("hospital_id")
        existing = baseline.index[baseline["modeled"] == 1]
        delta = (frozen.loc[existing, "failure_probability"]
                 - baseline.loc[existing, "failure_probability"]).abs()

        sample_name = frozen_path.name.replace("_hospitals.csv", "_samples.csv")
        frozen_samples = pd.read_csv(frozen_path.with_name(sample_name))
        published_samples = pd.read_csv((args.all_grid_root / relative).with_name(sample_name))
        draws_equal = frozen_samples[DRAW_COLUMNS].equals(published_samples[DRAW_COLUMNS])

        added = frozen.index.difference(baseline.index)
        rows.append({
            "track": str(relative.parent.name) + "/" + frozen_path.name.split("_hospitals")[0],
            "existing_facilities": len(existing),
            "existing_max_abs_delta": float(delta.max()),
            "existing_cells_differing": int((delta > args.tolerance).sum()),
            "baseline_modeled_in_frozen": bool(frozen.loc[existing, "modeled"].eq(1).all()),
            "added_facilities": len(added),
            "damage_draws_equal_published_all_grid": draws_equal,
            "frozen_demand_mw": float(frozen_samples["demand_mw"].iloc[0]),
            "frozen_mean_hospital_served": float(frozen_samples["hospital_served_fraction"].mean()),
            "published_mean_hospital_served": float(published_samples["hospital_served_fraction"].mean()),
        })

    table = pd.DataFrame(rows)
    passed = bool(len(table)) and bool(
        (table["existing_cells_differing"] == 0).all()
        and table["baseline_modeled_in_frozen"].all()
        and table["damage_draws_equal_published_all_grid"].all()
    )
    report = {
        "tracks_checked": int(len(table)),
        "tolerance": args.tolerance,
        "existing_facilities": int(table["existing_facilities"].max()) if len(table) else 0,
        "max_abs_delta_existing_vs_baseline": float(table["existing_max_abs_delta"].max()) if len(table) else None,
        "tracks_with_existing_differences": int((table["existing_cells_differing"] > 0).sum()) if len(table) else None,
        "tracks_with_different_damage_draws": int((~table["damage_draws_equal_published_all_grid"]).sum()) if len(table) else None,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "frozen_demand_audit_tracks.csv", index=False)
    (args.output_dir / "frozen_demand_audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
