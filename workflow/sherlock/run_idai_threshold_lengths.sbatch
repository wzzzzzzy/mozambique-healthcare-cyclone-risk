#!/bin/bash
#SBATCH --job-name=idai-threshold-lengths
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

# Destroyed transmission-line length for the Cyclone Idai threshold sweep on
# the hybrid model. Reuses the sweep's line-failure probabilities and the
# Figure 5 length cache built for the same hybrid model.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
SAMPLES="${1:-100}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${IDAI_THRESHOLD_ROOT:-$SCRATCH/idai_threshold_hybrid_v1}"
MODEL="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_cyclone_v1}/model/hybrid_power_model.npz"
LENGTH_CACHE="${INFRACASCADE_LENGTH_CACHE:-$SCRATCH/hybrid_figure5_length_v2/real_edge_lengths.npz}"

for path in "$MODEL" "$LENGTH_CACHE" "$RESULT_ROOT/probabilities" "$ENV_ROOT/lib/python3.9/site-packages"; do
    [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done
mkdir -p "$RESULT_ROOT/lengths"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
(git log -1 --oneline)

for threshold in 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    python3 src/models/hybrid_figure5_metrics.py track \
        --model "$MODEL" \
        --length-cache "$LENGTH_CACHE" \
        --probabilities "$RESULT_ROOT/probabilities/threshold_${threshold}.csv" \
        --storm-set -1 \
        --storm-folder IDAI_2019_THRESHOLD \
        --track-id "$threshold" \
        --samples "$SAMPLES" \
        --seed 1234 \
        --output "$RESULT_ROOT/lengths/threshold_${threshold}.json" > /dev/null
    echo "threshold $threshold done"
done
