#!/bin/bash
#SBATCH --job-name=idai-tower-conductor
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal
#SBATCH --array=18-30

# Cyclone Idai, one wind threshold per array task (task id = threshold, m/s),
# on the public-grid hybrid model with the run_idai_threshold_hybrid.sbatch
# node wind layers, seed and sample count.  Variants:
#   previous_line      prepare_matpower_cyclone_input --failure-mode line (published input)
#   previous_combined  prepare_matpower_cyclone_input --failure-mode combined
#   conductor_l0_*     length-normalized conductor spans (500 m wind sampling)
#   tower_conductor_l0_*  conductor spans plus mapped towers/poles
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SAMPLES="${1:-100}"
THRESHOLD="$SLURM_ARRAY_TASK_ID"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${IDAI_TOWER_ROOT:-$SCRATCH/idai_tower_conductor_v1}"
CODE_DIR="$RESULT_ROOT/code"
MODEL_DIR="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_cyclone_v1}/model"
IDAI_ROOT="$SCRATCH/idai_threshold_hybrid_v1"
RASTER="$IDAI_ROOT/input/Mozambique_Idai_2019_flip.tif"
WIND_TRACK="$IDAI_ROOT/wind/idai_threshold_${THRESHOLD}"
PUBLISHED="$IDAI_ROOT/probabilities/threshold_${THRESHOLD}.csv"
OUT="$RESULT_ROOT/threshold_${THRESHOLD}"
L0_VALUES=(0.25 0.5 1 2)

for path in "$PROJECT_DIR/src/hazard/length_normalized_line_hazard.py" "$RASTER" "$WIND_TRACK" "$PUBLISHED" \
    "$MODEL_DIR/hybrid_power_model.npz" "$MODEL_DIR/hybrid_path_cache.npz" \
    "$PROJECT_DIR/network_save/MG_hospital.pickle" "$ENV_ROOT/lib/python3.9/site-packages"; do
    [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done
[[ -e "$OUT" ]] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/probabilities" "$OUT/results"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "threshold $THRESHOLD; code commit $(git rev-parse HEAD)"

run_hybrid() {
    local tag="$1"
    python3 src/models/hybrid_power_model.py \
        --model "$MODEL_DIR/hybrid_power_model.npz" \
        --probabilities "$OUT/probabilities/${tag}.csv" \
        --samples "$SAMPLES" \
        --seed 1234 \
        --storm-folder IDAI_2019_TOWER_CONDUCTOR \
        --track-id "$THRESHOLD" \
        --sample-output "$OUT/results/${tag}_samples.csv" \
        --hospital-output "$OUT/results/${tag}_hospitals.csv" \
        --summary-output "$OUT/results/${tag}_summary.json" > /dev/null
    echo "done $tag"
}

for mode in line combined; do
    python3 src/networks/prepare_matpower_cyclone_input.py track \
        --path-cache "$MODEL_DIR/hybrid_path_cache.npz" \
        --wind-track "$WIND_TRACK" \
        --failure-mode "$mode" \
        --output "$OUT/probabilities/previous_${mode}.csv" > "$OUT/probabilities/previous_${mode}.json"
    run_hybrid "previous_${mode}"
done
cmp "$OUT/probabilities/previous_line.csv" "$PUBLISHED" && echo "previous line rule reproduces the published input"

for l0 in "${L0_VALUES[@]}"; do
    for towers in none mapped; do
        tag="$([[ $towers == none ]] && echo conductor || echo tower_conductor)_l0_${l0}km"
        python3 "$PROJECT_DIR/src/hazard/length_normalized_line_hazard.py" \
            --path-cache "$MODEL_DIR/hybrid_path_cache.npz" \
            --wind-track "$WIND_TRACK" \
            --reference-network "$PROJECT_DIR/network_save/MG_hospital.pickle" \
            --raster "$RASTER" \
            --threshold "$THRESHOLD" \
            --l0-km "$l0" \
            --max-spacing-km 0.5 \
            --towers "$towers" \
            --output "$OUT/probabilities/${tag}.csv" > /dev/null
        run_hybrid "$tag"
    done
done
