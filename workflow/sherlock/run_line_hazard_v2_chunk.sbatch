#!/bin/bash
#SBATCH --job-name=line-hazard-v2
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Stage 1 of the line-hazard v2 rerun: route-sampled ER11E04 winds and branch
# probabilities (24 m/s, L0 0.25 km, towers + conductor) for synthetic tracks
# [task * TRACKS_PER_TASK, (task + 1) * TRACKS_PER_TASK).  Resumable.
# Usage: sbatch --array=0-N run_line_hazard_v2_chunk.sbatch TRACKS_PER_TASK
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TRACKS_PER_TASK="${1:?TRACKS_PER_TASK}"
ROOT="${LINE_HAZARD_ROOT:?}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
START=$((SLURM_ARRAY_TASK_ID * TRACKS_PER_TASK))

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

python3 "$PROJECT_DIR/src/hazard/line_hazard_v2.py" chunk \
    --geometry "$ROOT/geometry.npz" \
    --storm-root "$SCRATCH/cyclone_map/STORM" \
    --boundary "$SCRATCH/World_Countries" \
    --chavas-dir "$PROJECT_DIR/ER11E04" \
    --global-start "$START" \
    --global-stop "$((START + TRACKS_PER_TASK))" \
    --winds-dir "$ROOT/winds" \
    --probability-dir "$ROOT/probabilities" \
    --threshold 24 --l0-km 0.25 --towers mapped
