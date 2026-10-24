#!/bin/bash
#SBATCH --job-name=idai-threshold-hybrid
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

# Cyclone Idai line-fragility threshold sweep (18-30 m/s) on the public-grid
# hybrid power model. Same model, seed, and sample count as the synthetic
# hybrid run in run_hybrid_cyclone_chunk.sbatch.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
SAMPLES="${1:-100}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${IDAI_THRESHOLD_ROOT:-$SCRATCH/idai_threshold_hybrid_v1}"
CODE_DIR="${IDAI_THRESHOLD_CODE_DIR:-$RESULT_ROOT/code}"
MODEL_DIR="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_cyclone_v1}/model"
RASTER="${IDAI_RASTER:-$RESULT_ROOT/input/Mozambique_Idai_2019_flip.tif}"
THRESHOLDS=(18 19 20 21 22 23 24 25 26 27 28 29 30)

for path in "$PROJECT_DIR/src/hazard/prepare_idai_threshold_wind.py" "$PROJECT_DIR/src/analysis/summarize_idai_threshold_hybrid.py" \
    "$RASTER" "$MODEL_DIR/hybrid_power_model.npz" "$MODEL_DIR/hybrid_path_cache.npz" \
    "$MODEL_DIR/hybrid_edge_mapping.csv" "$PROJECT_DIR/network_save/merge_MG_gridfiner.pickle" \
    "$ENV_ROOT/lib/python3.9/site-packages"; do
    [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done
mkdir -p "$RESULT_ROOT/wind" "$RESULT_ROOT/probabilities" "$RESULT_ROOT/results"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export INFRACASCADE_CODE_COMMIT="$(git rev-parse HEAD)"
echo "code commit $INFRACASCADE_CODE_COMMIT"

python3 "$PROJECT_DIR/src/hazard/prepare_idai_threshold_wind.py" \
    --raster "$RASTER" \
    --network network_save/merge_MG_gridfiner.pickle \
    --output-root "$RESULT_ROOT/wind" \
    --thresholds "${THRESHOLDS[@]}"

for threshold in "${THRESHOLDS[@]}"; do
    tag="threshold_${threshold}"
    probability="$RESULT_ROOT/probabilities/${tag}.csv"
    python3 src/networks/prepare_matpower_cyclone_input.py track \
        --path-cache "$MODEL_DIR/hybrid_path_cache.npz" \
        --wind-track "$RESULT_ROOT/wind/idai_threshold_${threshold}" \
        --failure-mode line \
        --output "$probability"
    python3 src/models/hybrid_power_model.py \
        --model "$MODEL_DIR/hybrid_power_model.npz" \
        --probabilities "$probability" \
        --samples "$SAMPLES" \
        --seed 1234 \
        --storm-folder IDAI_2019_THRESHOLD \
        --track-id "$threshold" \
        --sample-output "$RESULT_ROOT/results/${tag}_samples.csv" \
        --hospital-output "$RESULT_ROOT/results/${tag}_hospitals.csv" \
        --summary-output "$RESULT_ROOT/results/${tag}_summary.json" > /dev/null
    echo "threshold $threshold done"
done

python3 "$PROJECT_DIR/src/analysis/summarize_idai_threshold_hybrid.py" \
    --result-root "$RESULT_ROOT" \
    --edge-mapping "$MODEL_DIR/hybrid_edge_mapping.csv" \
    --legacy-network-dir "$PROJECT_DIR/network_save" \
    --thresholds "${THRESHOLDS[@]}"
