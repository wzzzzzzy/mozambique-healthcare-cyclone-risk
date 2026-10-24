#!/usr/bin/env python3
"""One-at-a-time sensitivity of the hybrid power model (line-hazard v2 probabilities).

Identical to ``hybrid_sensitivity.py`` except that ``tracks`` reads each
track's branch probabilities from ``--probability-dir`` (the line-hazard v2
stage-1 output, ``dataset_<s>/track_<tttt>.csv``) instead of recomputing the
previous hazard from the node wind layers.

Scenarios change one assumption of the baseline model (``hybrid_cyclone_v1``)
and keep everything else fixed: topology, facility cohort, hazard, and the
sampled line failures (same seed and sampling as the published run).

    alpha       P_D,i = D_tot [(1-alpha) a_i/sum a + alpha n_i/sum n]
    demand      D_tot = 0.80 x available Pmax (1,001.76 MW) or 1,093 MW
    generation  Pmax_g x 0.9 or 1.1, capped by nameplate; demand held fixed
    ratings     branch ratings x 0.5, 0.75, 1.25

The allocation weights a_i/sum a and n_i/sum n come from the undamaged
network (``bus_mapping.csv``) and stay fixed within each scenario.  Demand is
never rescaled to restore feasibility: each scenario's undamaged network is
solved, and any deficit there is reported and used for undamaged samples.

Subcommands: ``build`` (scenario models + undamaged-network check),
``tracks`` (all scenarios on a range of tracks), ``summarize``.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from hybrid_power_model import HybridPowerModel, load_probabilities
from synthetic_tracks import STORM_FOLDER_TEMPLATE, TRACK_COUNTS

BASELINE_ALPHA = 0.25
SCENARIOS = (
    ("baseline", {}),
    ("alpha_0.00", {"alpha": 0.0}),
    ("alpha_0.50", {"alpha": 0.5}),
    ("demand_cap80", {"demand_fraction_of_pmax": 0.80}),
    ("demand_1093", {"demand_mw": 1093.0}),
    ("generation_0.90", {"generation": 0.90}),
    ("generation_1.10", {"generation": 1.10}),
    ("ratings_0.50", {"ratings": 0.50}),
    ("ratings_0.75", {"ratings": 0.75}),
    ("ratings_1.25", {"ratings": 1.25}),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    with np.load(args.model, allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    index = {node: i for i, node in enumerate(base["node_ids"].astype(str))}
    d0 = float(base["node_pd_mw"].sum())
    pmax0 = float(base["node_pmax_mw"].sum())

    # Recover the two normalized weight vectors from the undamaged allocation.
    buses = [row for row in read_csv(args.bus_mapping) if float(row["load_weight"]) > 0]
    weight = np.zeros(len(index))
    count = np.zeros(len(index))
    for row in buses:
        weight[index[row["node_id"]]] = float(row["load_weight"])
        count[index[row["node_id"]]] = float(row["hospital_count"])
    hospital_share = count / count.sum()
    infra_share = (weight - BASELINE_ALPHA * hospital_share) / (1.0 - BASELINE_ALPHA)
    if infra_share.min() < -1e-12 or abs(infra_share.sum() - 1.0) > 1e-9:
        raise ValueError("recovered infrastructure weights are not a distribution")
    if np.abs(d0 * weight - base["node_pd_mw"]).max() > 1e-6:
        raise ValueError("baseline demand is not D_tot times the stored weights")

    generators = read_csv(args.generator_mapping)
    def generator_pmax(factor: float) -> np.ndarray:
        pmax = np.zeros(len(index))
        for g in generators:
            pmax[index[g["node_id"]]] += min(float(g["installed_mw"]), factor * float(g["pmax_mw"]))
        return pmax
    if np.abs(generator_pmax(1.0) - base["node_pmax_mw"]).max() > 1e-9:
        raise ValueError("generator mapping does not reproduce node Pmax")

    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True)
    records = []
    for name, change in SCENARIOS:
        arrays = dict(base)
        alpha = change.get("alpha", BASELINE_ALPHA)
        demand = change.get("demand_mw", d0)
        if "demand_fraction_of_pmax" in change:
            demand = change["demand_fraction_of_pmax"] * pmax0
        if "alpha" in change or demand != d0:
            arrays["node_pd_mw"] = demand * ((1 - alpha) * infra_share + alpha * hospital_share)
        if "generation" in change:
            arrays["node_pmax_mw"] = generator_pmax(change["generation"])
        if "ratings" in change:
            arrays["edge_rate_mw"] = base["edge_rate_mw"] * change["ratings"]
        path = models_dir / f"{name}.npz"
        np.savez(path, **arrays)

        model = HybridPowerModel(path)
        intact = model.solve_sample_state(np.zeros(len(model.edge_uid), dtype=bool)).result
        modeled = intact.hospital_failure[model.hospital_modeled]
        records.append({
            "scenario": name, "change": change, "alpha": alpha,
            "total_demand_mw": float(arrays["node_pd_mw"].sum()),
            "available_pmax_mw": float(arrays["node_pmax_mw"].sum()),
            "rating_factor": change.get("ratings", 1.0),
            "intact_served_fraction": intact.served_mw / intact.demand_mw,
            "intact_mean_facility_failure": float(modeled.mean()),
            "intact_max_facility_failure": float(modeled.max()),
            "intact_flow_constrained_islands": intact.flow_constrained_components,
            "intact_capacity_constrained_islands": intact.capacity_constrained_components,
            "intact_deficit": bool(modeled.max() > 1e-9),
            "model": str(path),
        })
        print(json.dumps(records[-1]))
    (args.output_dir / "scenarios.json").write_text(json.dumps(records, indent=2) + "\n")


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
    scenarios = json.loads((args.scenario_dir / "scenarios.json").read_text())
    models = [HybridPowerModel(Path(s["model"])) for s in scenarios]
    reference = models[0]
    intact = [m.solve_sample_state(np.zeros(len(m.edge_uid), dtype=bool)).result.hospital_failure
              for m in models]
    modeled = reference.hospital_modeled
    lookup = np.full(reference.total_edge_count, -1, dtype=np.int64)
    lookup[reference.edge_uid - 1] = np.arange(len(reference.edge_uid))

    track_ids = list(range(args.global_start, args.global_stop))
    failure = np.full((len(models), len(track_ids), len(reference.hospital_ids)), -1.0)
    flow_samples = np.zeros((len(models), len(track_ids)), dtype=np.int32)
    damaged = np.zeros(len(track_ids), dtype=np.int32)
    with tempfile.TemporaryDirectory() as work:
        for t, global_id in enumerate(track_ids):
            storm_set, track_id = global_to_track(global_id)
            folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
            probability_csv = args.probability_dir / f"dataset_{storm_set}" / f"track_{track_id:04d}.csv"
            probabilities = load_probabilities(probability_csv, reference.total_edge_count)
            positive = np.flatnonzero(probabilities > 0.0)
            rng = np.random.RandomState(args.seed)
            sums = np.zeros((len(models), int(modeled.sum())))
            for _ in range(args.samples):
                failed_global = positive[rng.uniform(0.0, 1.0, len(positive)) < probabilities[positive]]
                retained = lookup[failed_global]
                retained = retained[retained >= 0]
                if len(retained) == 0:
                    for s in range(len(models)):
                        sums[s] += intact[s][modeled]
                    continue
                damaged[t] += 1
                failed = np.zeros(len(reference.edge_uid), dtype=bool)
                failed[retained] = True
                for s, model in enumerate(models):
                    result = model.solve_sample_state(failed).result
                    sums[s] += result.hospital_failure[modeled]
                    flow_samples[s, t] += int(result.flow_constrained_components > 0)
            failure[:, t, modeled] = sums / args.samples
            print(f"track {global_id} done; damaged samples {damaged[t]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".partial.npz")
    np.savez_compressed(partial, global_ids=np.asarray(track_ids), failure=failure,
                        flow_constrained_samples=flow_samples, damaged_samples=damaged,
                        scenarios=np.asarray([s["scenario"] for s in scenarios]),
                        hospital_ids=reference.hospital_ids)
    partial.replace(args.output)


def summarize(args: argparse.Namespace) -> None:
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    scenarios = json.loads((args.scenario_dir / "scenarios.json").read_text())
    chunks = [np.load(p, allow_pickle=False) for p in sorted(args.chunk_dir.glob("tracks_*.npz"))]
    ids = np.concatenate([c["global_ids"] for c in chunks])
    order = np.argsort(ids)
    failure = np.concatenate([c["failure"] for c in chunks], axis=1)[:, order]
    flow = np.concatenate([c["flow_constrained_samples"] for c in chunks], axis=1)[:, order]
    damaged = np.concatenate([c["damaged_samples"] for c in chunks])[order]
    hospital_ids = chunks[0]["hospital_ids"].astype(str)
    if len(ids) != sum(TRACK_COUNTS) or not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError(f"expected all {sum(TRACK_COUNTS)} tracks, found {len(ids)}")
    modeled = failure[0, 0] >= 0
    f = failure[:, :, modeled]                     # scenario x track x facility
    track_mean = f.mean(axis=2)
    facility_mean = f.mean(axis=1)

    check = {}
    if args.published:
        published = pd.read_csv(args.published).set_index("node_id")
        columns = [f"{s}_{t}" for s, n in enumerate(TRACK_COUNTS) for t in range(n)]
        pub = published.loc[hospital_ids[modeled], columns].to_numpy(float).T
        check = {"baseline_vs_published_max_abs_difference": float(np.abs(pub - f[0]).max())}

    rows = []
    for s, scenario in enumerate(scenarios):
        diff_track = track_mean[s] - track_mean[0]
        diff_facility = facility_mean[s] - facility_mean[0]
        rows.append({
            "scenario": scenario["scenario"],
            "total_demand_mw": round(scenario["total_demand_mw"], 2),
            "available_pmax_mw": round(scenario["available_pmax_mw"], 2),
            "rating_factor": scenario["rating_factor"],
            "alpha": scenario["alpha"],
            "intact_mean_facility_failure": scenario["intact_mean_facility_failure"],
            "intact_deficit": scenario["intact_deficit"],
            "mean_facility_failure_per_event": float(f[s].mean()),
            "annual_mean_facility_failure": float(f[s].mean() * len(ids) / 10_000),
            "storm_attributable_mean_failure": float(
                f[s].mean() - scenario["intact_mean_facility_failure"]),
            "change_vs_baseline": float(f[s].mean() - f[0].mean()),
            "relative_change_vs_baseline": float(f[s].mean() / f[0].mean() - 1.0),
            "track_pearson_vs_baseline": float(pearsonr(track_mean[0], track_mean[s])[0]),
            "track_spearman_vs_baseline": float(spearmanr(track_mean[0], track_mean[s])[0]),
            "tracks_changed_gt_1e-6": int((np.abs(diff_track) > 1e-6).sum()),
            "max_track_change": float(np.abs(diff_track).max()),
            "max_facility_change": float(np.abs(diff_facility).max()),
            "damaged_samples_with_flow_constraint": int(flow[s].sum()),
            "damaged_samples": int(damaged.sum()),
        })
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    (args.output_dir / "sensitivity_summary.json").write_text(
        json.dumps({"checks": check, "scenarios": rows}, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "facility_means.npz", scenarios=table["scenario"].to_numpy(),
                        hospital_ids=hospital_ids[modeled], facility_mean=facility_mean,
                        track_mean=track_mean)
    print(json.dumps(check))
    print(table.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--bus-mapping", type=Path, required=True)
    b.add_argument("--generator-mapping", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    t = sub.add_parser("tracks")
    t.add_argument("--scenario-dir", type=Path, required=True)
    t.add_argument("--probability-dir", type=Path, required=True)
    t.add_argument("--global-start", type=int, required=True)
    t.add_argument("--global-stop", type=int, required=True)
    t.add_argument("--samples", type=int, default=100)
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--scenario-dir", type=Path, required=True)
    s.add_argument("--chunk-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--published", type=Path, help="df_final_hybrid_25.csv for the baseline check")
    args = parser.parse_args()
    {"build": build, "tracks": run_tracks, "summarize": summarize}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
