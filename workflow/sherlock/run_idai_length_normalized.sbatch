#!/bin/bash
#SBATCH --job-name=idai-length-normalized
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Cyclone Idai with the length-normalized line fragility, on the same
# public-grid hybrid model, node wind layer (25 m/s), seed and sample count as
# run_idai_threshold_hybrid.sbatch.  Sweeps the reference length L0 and checks
# convergence of the wind-sampling spacing (500 m against 250 m).
# IDAI_THRESHOLD selects the line-curve threshold (default 25 m/s) and its node
# wind layer; IDAI_SPACINGS overrides the spacings tested.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
SAMPLES="${1:-100}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${IDAI_LENGTH_ROOT:-$SCRATCH/idai_length_normalized_v1}"
CODE_DIR="$RESULT_ROOT/code"
MODEL_DIR="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_cyclone_v1}/model"
IDAI_ROOT="$SCRATCH/idai_threshold_hybrid_v1"
RASTER="$IDAI_ROOT/input/Mozambique_Idai_2019_flip.tif"
THRESHOLD="${IDAI_THRESHOLD:-25}"
WIND_TRACK="$IDAI_ROOT/wind/idai_threshold_${THRESHOLD}"
PUBLISHED="$IDAI_ROOT/probabilities/threshold_${THRESHOLD}.csv"
L0_VALUES=(0.25 0.5 1 2)
read -r -a SPACINGS <<< "${IDAI_SPACINGS:-0.5 0.25}"

for path in "$PROJECT_DIR/src/hazard/length_normalized_line_hazard.py" "$RASTER" "$WIND_TRACK" \
    "$MODEL_DIR/hybrid_power_model.npz" "$MODEL_DIR/hybrid_path_cache.npz" \
    "$PROJECT_DIR/network_save/MG_hospital.pickle" "$PUBLISHED" \
    "$ENV_ROOT/lib/python3.9/site-packages"; do
    [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done
mkdir -p "$RESULT_ROOT/probabilities" "$RESULT_ROOT/results"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "code commit $(git rev-parse HEAD)"

run_hybrid() {
    local tag="$1" probability="$2"
    python3 src/models/hybrid_power_model.py \
        --model "$MODEL_DIR/hybrid_power_model.npz" \
        --probabilities "$probability" \
        --samples "$SAMPLES" \
        --seed 1234 \
        --storm-folder IDAI_2019_LENGTH_NORMALIZED \
        --track-id "$THRESHOLD" \
        --sample-output "$RESULT_ROOT/results/${tag}_samples.csv" \
        --hospital-output "$RESULT_ROOT/results/${tag}_hospitals.csv" \
        --summary-output "$RESULT_ROOT/results/${tag}_summary.json" > /dev/null
}

# Previous rule, recomputed here: must equal the published input for this threshold.
python3 src/networks/prepare_matpower_cyclone_input.py track \
    --path-cache "$MODEL_DIR/hybrid_path_cache.npz" \
    --wind-track "$WIND_TRACK" \
    --failure-mode line \
    --output "$RESULT_ROOT/probabilities/previous_rule.csv"
cmp "$RESULT_ROOT/probabilities/previous_rule.csv" "$PUBLISHED" \
    && echo "previous rule reproduces the published Idai input"
run_hybrid previous_rule "$RESULT_ROOT/probabilities/previous_rule.csv"

for spacing in "${SPACINGS[@]}"; do
    for l0 in "${L0_VALUES[@]}"; do
        tag="l0_${l0}km_dx_${spacing}km"
        python3 "$PROJECT_DIR/src/hazard/length_normalized_line_hazard.py" \
            --path-cache "$MODEL_DIR/hybrid_path_cache.npz" \
            --wind-track "$WIND_TRACK" \
            --reference-network "$PROJECT_DIR/network_save/MG_hospital.pickle" \
            --raster "$RASTER" \
            --threshold "$THRESHOLD" \
            --l0-km "$l0" \
            --max-spacing-km "$spacing" \
            --output "$RESULT_ROOT/probabilities/${tag}.csv" > /dev/null
        run_hybrid "$tag" "$RESULT_ROOT/probabilities/${tag}.csv"
        echo "done $tag"
    done
done

python3 - "$RESULT_ROOT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
rows = []
for summary in sorted((root / "results").glob("*_summary.json")):
    tag = summary.name[:-len("_summary.json")]
    result = json.loads(summary.read_text())["result"]
    hospitals = pd.read_csv(root / "results" / f"{tag}_hospitals.csv")
    modeled = hospitals[hospitals["modeled"] == 1]["failure_probability"]
    hazard = root / "probabilities" / f"{tag}.json"
    extra = json.loads(hazard.read_text()) if hazard.exists() else {}
    rows.append({"variant": tag, "l0_km": extra.get("l0_km"), "max_spacing_km": extra.get("max_spacing_km"),
                 "expected_damaged_line_km": extra.get("expected_damaged_line_km"),
                 "expected_removed_branch_km": extra.get("expected_removed_branch_km"),
                 "mean_branch_probability": extra.get("mean_failure_probability"),
                 "mean_facility_failure": result["mean_hospital_failure_probability"],
                 "facilities_failure_gt_0.5": int((modeled > 0.5).sum()),
                 "facilities_any_failure": int((modeled > 0).sum()),
                 "mean_failed_edges": result["mean_failed_edges"],
                 "uniform_wind_check": extra.get("uniform_wind_check_max_abs_difference")})
table = pd.DataFrame(rows)
table.to_csv(root / "idai_length_normalized_summary.csv", index=False)
print(table.to_string(index=False))
PY
