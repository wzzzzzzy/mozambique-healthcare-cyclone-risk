#!/bin/bash
#SBATCH --job-name=hybrid-sens
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=6GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

# All sensitivity scenarios on one range of tracks, same sampled failures.
# Usage: sbatch --array=0-N run_hybrid_sensitivity.sbatch TRACKS_PER_TASK SCENARIO_DIR
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TRACKS_PER_TASK="$1"; SCENARIO_DIR="$2"
TOTAL_TRACKS=2703
START=$((SLURM_ARRAY_TASK_ID * TRACKS_PER_TASK))
STOP=$((START + TRACKS_PER_TASK)); (( STOP <= TOTAL_TRACKS )) || STOP=$TOTAL_TRACKS
(( START < TOTAL_TRACKS )) || exit 2
OUTPUT="$SCENARIO_DIR/chunks/tracks_$(printf '%04d' "$START")_$(printf '%04d' "$STOP").npz"
[[ -s "$OUTPUT" ]] && { echo "complete: $OUTPUT"; exit 0; }
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 src/models/hybrid_sensitivity.py tracks \
    --scenario-dir "$SCENARIO_DIR" \
    --path-cache "$SCRATCH/hybrid_cyclone_v1/model/hybrid_path_cache.npz" \
    --wind-root "$SCRATCH/wind_result/synthetic_capacity_v2_public_net" \
    --global-start "$START" --global-stop "$STOP" \
    --samples 100 --seed 1234 --output "$OUTPUT"
