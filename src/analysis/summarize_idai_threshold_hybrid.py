#!/usr/bin/env python3
"""Summarize the Cyclone Idai hybrid threshold sweep and check its hazard.

The hazard check compares each threshold's hybrid line-failure probabilities
with the ``line_fail`` edge attribute stored by the original 2024 Idai sweep
(``network_save/Mozambique_Idai_2019_flip_wind_<t>.pickle``).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def read_probabilities(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        return {int(row["branch_uid"]): float(row["failure_probability"]) for row in csv.DictReader(handle)}


def compare_with_legacy(probabilities: dict[int, float], mapping: pd.DataFrame, legacy: Path) -> dict[str, object]:
    with legacy.open("rb") as handle:
        graph = pickle.load(handle)
    differences = []
    missing = 0
    for row in mapping.itertuples(index=False):
        key = (row.from_node, row.to_node, int(row.edge_key))
        if not graph.has_edge(*key) or "line_fail" not in graph.edges[key]:
            missing += 1
            continue
        differences.append(probabilities[int(row.branch_id)] - float(graph.edges[key]["line_fail"]))
    differences = np.abs(np.asarray(differences))
    return {
        "legacy_network": str(legacy),
        "compared_edges": int(len(differences)),
        "edges_missing_from_legacy": missing,
        "max_abs_difference": float(differences.max(initial=0.0)),
        "edges_differing_above_1e-9": int(np.count_nonzero(differences > 1e-9)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--edge-mapping", type=Path, required=True)
    parser.add_argument("--legacy-network-dir", type=Path)
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(range(18, 31)))
    args = parser.parse_args()

    mapping = pd.read_csv(args.edge_mapping, dtype={"from_node": str, "to_node": str})
    wind_report = json.loads((args.result_root / "wind" / "wind_report.json").read_text())
    wind_by_threshold = {row["threshold_ms"]: row for row in wind_report["thresholds"]}

    rows, wide, checks = [], None, {}
    for threshold in args.thresholds:
        tag = f"threshold_{threshold:02d}"
        summary = json.loads((args.result_root / "results" / f"{tag}_summary.json").read_text())
        hospitals = pd.read_csv(args.result_root / "results" / f"{tag}_hospitals.csv")
        probabilities = read_probabilities(args.result_root / "probabilities" / f"{tag}.csv")
        modeled = hospitals["modeled"].eq(1)
        values = hospitals.loc[modeled, "failure_probability"]
        rows.append({
            "threshold_ms": threshold,
            "nodes_with_positive_line_failure": wind_by_threshold[threshold]["nodes_with_positive_line_failure"],
            "edges_with_positive_probability": summary["hazard"]["edges_with_positive_probability"],
            "expected_failed_edges": summary["hazard"]["expected_failed_edges"],
            "mean_failed_edges": summary["result"]["mean_failed_edges"],
            "modeled_hospitals": int(modeled.sum()),
            "mean_hospital_failure_probability": float(values.mean()),
            "median_hospital_failure_probability": float(values.median()),
            "hospitals_failure_ge_0.5": int((values >= 0.5).sum()),
            "hospitals_failure_gt_0": int((values > 0).sum()),
            "probability_any_hospital_disruption": summary["result"]["probability_any_hospital_disruption"],
            "mean_served_fraction": summary["result"]["mean_served_fraction"],
        })
        column = hospitals.set_index("hospital_id")["failure_probability"].rename(str(threshold))
        wide = column.to_frame() if wide is None else wide.join(column)
        if args.legacy_network_dir is not None:
            legacy = args.legacy_network_dir / f"Mozambique_Idai_2019_flip_wind_{threshold}.pickle"
            checks[str(threshold)] = compare_with_legacy(probabilities, mapping, legacy) if legacy.exists() else {"legacy_network": str(legacy), "missing": True}

    table = pd.DataFrame(rows)
    table.to_csv(args.result_root / "idai_threshold_hybrid_summary.csv", index=False)
    wide.index.name = "hospital_id"
    wide.reset_index().to_csv(args.result_root / "df_idai_threshold_hybrid.csv", index=False)
    if checks:
        (args.result_root / "hazard_check_vs_2024.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(table.to_string(index=False))
    if checks:
        print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
