#!/bin/bash
#SBATCH --job-name=line-hazard-v2-check
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Build the line-hazard v2 geometry and check it against earlier results:
#   1. Idai raster, main case (24 m/s, L0 0.25 km, towers) = idai_tower_conductor_v1 output;
#   2. Idai raster, previous rule (25 m/s) = published Idai input;
#   3. synthetic tracks: recomputed node winds = stored node wind layers, and the
#      previous rule = prepare_matpower_cyclone_input on those layers.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
ROOT="${LINE_HAZARD_ROOT:?}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
CODE_DIR="$ROOT/code"
GEOMETRY="$ROOT/geometry.npz"
CHECK="$ROOT/validation"
IDAI="$SCRATCH/idai_threshold_hybrid_v1"
WIND_ROOT="$SCRATCH/wind_result/synthetic_capacity_v2_public_net"
TRACKS=("3 72" "0 76" "5 54" "0 0")
mkdir -p "$CHECK"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
module load matlab/R2022b
module load gcc/9.1.0
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ENV_ROOT/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
H="$PROJECT_DIR/src/hazard/line_hazard_v2.py"

[[ -e "$GEOMETRY" ]] || python3 "$H" geometry \
    --hazard-network network_save/merge_MG_gridfiner.pickle \
    --cache "public=$SCRATCH/hybrid_cyclone_v1/model/hybrid_path_cache.npz" \
    --cache "all_grid=$SCRATCH/hybrid_all_grid_frozen_demand_v1/model/hybrid_path_cache.npz" \
    --max-spacing-km 0.5 --output "$GEOMETRY"

python3 "$H" track --geometry "$GEOMETRY" --raster "$IDAI/input/Mozambique_Idai_2019_flip.tif" \
    --threshold 24 --l0-km 0.25 --towers mapped --output "public=$CHECK/idai_t24_main.csv"
python3 "$H" track --geometry "$GEOMETRY" --raster "$IDAI/input/Mozambique_Idai_2019_flip.tif" \
    --threshold 25 --rule previous --output "public=$CHECK/idai_t25_previous.csv"

for spec in "${TRACKS[@]}"; do
    read -r storm_set track <<< "$spec"
    folder="STORM_DATA_IBTRACS_SI_1000_YEARS_${storm_set}"
    tag="${storm_set}_${track}"
    python3 "$H" track --geometry "$GEOMETRY" --folder "$folder" --track-id "$track" \
        --storm-root "$SCRATCH/cyclone_map/STORM" --boundary "$SCRATCH/World_Countries" \
        --chavas-dir "$PROJECT_DIR/ER11E04" --threshold 25 --rule previous \
        --winds-output "$CHECK/winds_${tag}.npz" --output "public=$CHECK/synthetic_${tag}_previous.csv"
    python3 src/networks/prepare_matpower_cyclone_input.py track --path-cache "$SCRATCH/hybrid_cyclone_v1/model/hybrid_path_cache.npz" \
        --wind-track "$WIND_ROOT/track_${folder}_${track}" --failure-mode line \
        --output "$CHECK/synthetic_${tag}_published.csv" > /dev/null
done

python3 - "$ROOT" "$IDAI" "$WIND_ROOT" "${TRACKS[@]}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from prepare_matpower_cyclone_input import read_dbf_columns, resolve_dbf
root, idai, wind_root = map(Path, sys.argv[1:4])
check = root / "validation"
def diff(a, b):
    x = pd.read_csv(a).set_index("branch_uid")["failure_probability"]
    y = pd.read_csv(b).set_index("branch_uid")["failure_probability"]
    return float((x - y.reindex(x.index)).abs().max())
report = {
    "idai_main_vs_tested": diff(check / "idai_t24_main.csv",
                                Path(f"{idai}/../idai_tower_conductor_v1/threshold_24/probabilities/tower_conductor_l0_0.25km.csv")),
    "idai_previous_vs_published": diff(check / "idai_t25_previous.csv", idai / "probabilities/threshold_25.csv"),
}
geo = np.load(root / "geometry.npz")
ids, present = geo["node_ids"].astype(str), geo["node_present"]
for spec in sys.argv[4:]:
    storm_set, track = spec.split()
    tag = f"{storm_set}_{track}"
    winds = np.load(check / f"winds_{tag}.npz")["node_wind"].astype(float)
    cols = read_dbf_columns(resolve_dbf(wind_root / f"track_STORM_DATA_IBTRACS_SI_1000_YEARS_{storm_set}_{track}"),
                            ("id", "wind_inten"))
    stored = pd.Series([float(v) for v in cols["wind_inten"]], index=[str(v) for v in cols["id"]])
    stored = stored[~stored.index.duplicated()].fillna(0.0)
    mine = pd.Series(winds[present], index=ids[present])
    report[f"synthetic_{tag}"] = {
        "max_stored_wind": float(stored.max()),
        "node_wind_max_abs_difference": float((mine - stored.reindex(mine.index)).abs().max()),
        "previous_rule_max_abs_difference": diff(check / f"synthetic_{tag}_previous.csv",
                                                 check / f"synthetic_{tag}_published.csv"),
    }
(check / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
