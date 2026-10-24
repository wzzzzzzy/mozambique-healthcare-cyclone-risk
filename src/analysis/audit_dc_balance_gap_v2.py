#!/usr/bin/env python3
"""Compare power-balance and DC-feasible service for every damaged hybrid island (line-hazard v2).

Identical to ``audit_dc_balance_gap.py`` except that ``tracks`` reads each
track's branch probabilities from ``--probability-dir`` (line-hazard v2
stage-1 output) instead of recomputing the previous hazard.

For each connected component with positive demand in each Monte Carlo sample
that has at least one failed retained edge, record

    lambda_balance = min(1, G_C / D_C)    (``capacity_adequacy``)
    lambda_DC      = DC-optimized uniform served fraction (``served_fraction``)
    delta          = lambda_balance - lambda_DC

The samples reproduce the hybrid electricity run (``hybrid_cyclone_v1``,
commit 1bdf76c): one RandomState(seed) stream over the edges with positive
failure probability.  ``hybrid_power_model`` is used unchanged; a subclass only
observes each island result and the raw LP value before it is clipped to
lambda_balance.

Subcommands: ``tracks`` processes a range of global track ids; ``summarize``
merges the chunk files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import hybrid_power_model as hpm
from hybrid_power_model import EPSILON, HybridPowerModel, load_probabilities
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS

FIELDS = ["global_id", "storm_set", "track_id", "sample", "island_key", "buses",
          "branches", "demand_mw", "supply_mw", "lambda_balance", "lambda_dc",
          "lp_solved", "lp_raw"]

_raw_lp: list[float] = []
_original_linprog = hpm.linprog


def _recording_linprog(*args, **kwargs):
    solution = _original_linprog(*args, **kwargs)
    _raw_lp.append(float(solution.x[-1]) if solution.success else float("nan"))
    return solution


hpm.linprog = _recording_linprog


class RecordingModel(HybridPowerModel):
    """Hybrid model that keeps every island result of the last solved sample."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.islands: list[dict[str, object]] = []

    def _component_service(self, nodes, power_edge_indices):
        _raw_lp.clear()
        result = super()._component_service(nodes, power_edge_indices)
        if result.demand_mw > EPSILON:
            power_nodes = np.sort(nodes[self.node_is_power[nodes]])
            branches = np.sort(np.asarray(power_edge_indices, dtype=np.int64))
            key = hashlib.sha1(power_nodes.tobytes() + b"|" + branches.tobytes()).hexdigest()[:16]
            self.islands.append({
                "island_key": key,
                "buses": int(len(power_nodes)),
                "branches": int(len(branches)),
                "demand_mw": result.demand_mw,
                "supply_mw": result.supply_mw,
                "lambda_balance": result.capacity_adequacy,
                "lambda_dc": result.served_fraction,
                "lp_solved": int(bool(_raw_lp)),
                "lp_raw": _raw_lp[-1] if _raw_lp else "",
            })
        return result


def global_to_track(global_id: int) -> tuple[int, int]:
    offset = 0
    for storm_set, count in enumerate(TRACK_COUNTS):
        if global_id < offset + count:
            return storm_set, global_id - offset
        offset += count
    raise ValueError(f"global id {global_id} is beyond the catalogue")


def run_tracks(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    model = RecordingModel(args.model)
    rows = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with gzip.open(partial, "wt", newline="") as handle, tempfile.TemporaryDirectory() as work:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for global_id in range(args.global_start, args.global_stop):
            storm_set, track_id = global_to_track(global_id)
            folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
            probability_csv = args.probability_dir / f"dataset_{storm_set}" / f"track_{track_id:04d}.csv"
            probabilities = load_probabilities(probability_csv, model.total_edge_count)
            positive = np.flatnonzero(probabilities > 0.0)
            lookup = np.full(model.total_edge_count, -1, dtype=np.int64)
            lookup[model.edge_uid - 1] = np.arange(len(model.edge_uid))
            rng = np.random.RandomState(args.seed)
            service = []
            for sample in range(1, args.samples + 1):
                failed_global = positive[rng.uniform(0.0, 1.0, len(positive)) < probabilities[positive]]
                retained = lookup[failed_global]
                retained = retained[retained >= 0]
                if len(retained) == 0:
                    service.append(1.0)
                    continue
                failed = np.zeros(len(model.edge_uid), dtype=bool)
                failed[retained] = True
                model.islands.clear()
                service.append(model.solve_sample_state(failed).result.hospital_served_fraction)
                for island in model.islands:
                    writer.writerow({"global_id": global_id, "storm_set": storm_set,
                                     "track_id": track_id, "sample": sample, **island})
                    rows += 1
            # Check against the published hybrid run for the same track.
            print(f"track {global_id} ({folder}:{track_id}) mean_hospital_failure="
                  f"{1.0 - float(np.mean(service)):.12f} islands_so_far={rows}", flush=True)
    partial.replace(args.output)


def summarize(args: argparse.Namespace) -> None:
    import pandas as pd

    files = sorted(args.chunk_dir.glob("islands_*.csv.gz"))
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    frame["delta"] = frame["lambda_balance"] - frame["lambda_dc"]
    tol = args.tolerance

    def stats(sub: pd.DataFrame) -> dict[str, object]:
        if sub.empty:
            return {"islands": 0}
        delta = sub["delta"]
        weights = sub["demand_mw"]
        return {
            "islands": int(len(sub)),
            "max_delta": float(delta.max()),
            "min_delta": float(delta.min()),
            "islands_delta_gt_tol": int((delta > tol).sum()),
            "fraction_delta_gt_tol": float((delta > tol).mean()),
            "demand_weighted_fraction_delta_gt_tol": float(weights[delta > tol].sum() / weights.sum()),
            "mean_delta": float(delta.mean()),
            "delta_quantiles_where_gt_tol": (
                delta[delta > tol].quantile([0.5, 0.9, 0.99]).round(6).to_dict()
                if (delta > tol).any() else {}),
        }

    lp = frame["lp_solved"] == 1
    no_generation = frame["supply_mw"] <= EPSILON
    unique = frame.drop_duplicates("island_key")
    raw = pd.to_numeric(frame.loc[lp, "lp_raw"], errors="coerce")
    report = {
        "schema_version": 1,
        "tolerance": tol,
        "chunks": len(files),
        "tracks": int(frame["global_id"].nunique()),
        "damaged_samples": int(frame[["global_id", "sample"]].drop_duplicates().shape[0]),
        "definition": "islands with positive demand in samples with >=1 failed retained edge",
        "all_islands": stats(frame),
        "islands_with_generation": stats(frame.loc[~no_generation]),
        "lp_solved_islands": stats(frame.loc[lp]),
        "no_generation_islands": int(no_generation.sum()),
        "single_bus_or_no_branch_with_generation": int((~lp & ~no_generation).sum()),
        "unique_islands": stats(unique),
        "unique_lp_solved_islands": stats(unique.loc[unique["lp_solved"] == 1]),
        "lp_raw_exceeding_lambda_balance_max": float(
            (raw - frame.loc[lp, "lambda_balance"]).max()) if lp.any() else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dc_balance_gap_report.json").write_text(json.dumps(report, indent=2) + "\n")
    top = unique.sort_values("delta", ascending=False).head(50)
    top.to_csv(args.output_dir / "dc_balance_gap_top_islands.csv", index=False)
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tracks = sub.add_parser("tracks")
    tracks.add_argument("--model", type=Path, required=True)
    tracks.add_argument("--probability-dir", type=Path, required=True)
    tracks.add_argument("--global-start", type=int, required=True)
    tracks.add_argument("--global-stop", type=int, required=True)
    tracks.add_argument("--samples", type=int, default=100)
    tracks.add_argument("--seed", type=int, default=1234)
    tracks.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--chunk-dir", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    run_tracks(args) if args.command == "tracks" else summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
