#!/bin/bash
#SBATCH --job-name=fig5-prepare
#SBATCH --output=/scratch/users/zywei/infracascade_logs/figures/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/figures/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${INFRACASCADE_FIGURE5_METRIC_ROOT:-$SCRATCH/hybrid_figure5_length_v2}"
mkdir -p "$RESULT_ROOT" "$SCRATCH/infracascade_logs/figures"
cd "$PROJECT_DIR"
module purge
module load viz python/3.9.0 json-glib/1.4.4 py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python3 src/models/hybrid_figure5_metrics.py manifest \
    --admin-boundaries "$SCRATCH/wind_result/notebook_inputs/afr_g2014_2013_1" \
    --storm-root "$SCRATCH/cyclone_map/STORM" \
    --output "$RESULT_ROOT/category_1_2_tracks.csv"
python3 src/models/hybrid_figure5_metrics.py lengths \
    --model "$SCRATCH/hybrid_cyclone_v1/model/hybrid_power_model.npz" \
    --network "$PROJECT_DIR/network_save/simplified_hospital_MG.pickle" \
    --reference-network "$PROJECT_DIR/network_save/MG_hospital.pickle" \
    --output "$RESULT_ROOT/real_edge_lengths.npz"
