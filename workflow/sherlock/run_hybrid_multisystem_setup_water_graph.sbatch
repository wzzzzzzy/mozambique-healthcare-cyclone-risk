#!/bin/bash
#SBATCH --job-name=hybrid-ms-setup-water
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid_multisystem/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid_multisystem/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

# Same as run_hybrid_multisystem_setup.sbatch, but the water graph is an argument.
# Usage: sbatch run_hybrid_multisystem_setup_water_graph.sbatch WATER_GRAPH
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
(( $# == 1 )) || { echo "Usage: sbatch $0 WATER_GRAPH" >&2; exit 2; }
WATER_GRAPH="$1"
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
MODEL_PATH="${INFRACASCADE_DEPENDENCY_MODEL:?INFRACASCADE_DEPENDENCY_MODEL must name a new model path}"
[[ -e "$MODEL_PATH" ]] && { echo "Refusing to overwrite $MODEL_PATH" >&2; exit 1; }

required=(
    "$PROJECT_DIR/SARA_processed"
    "$WATER_GRAPH"
    "$PROJECT_DIR/network_save/MG_c_0205.pickle"
    "$SCRATCH/cascade_result/cc_csv_cyclone/STORM_DATA_IBTRACS_SI_1000_YEARS_0__capacity_v2_public_0_1_25.csv"
    "$ENV_ROOT/lib/python3.9/site-packages"
)
for path in "${required[@]}"; do
    [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done

mkdir -p "$(dirname "$MODEL_PATH")"
cd "$PROJECT_DIR"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
md5sum "$WATER_GRAPH"
exec python3 src/models/propagate_cyclone_dependencies.py prepare \
    --threshold 25 \
    --water-graph "$WATER_GRAPH" \
    --communication-graph "$PROJECT_DIR/network_save/MG_c_0205.pickle" \
    --model-output "$MODEL_PATH"
