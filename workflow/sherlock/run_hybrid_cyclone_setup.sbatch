#!/bin/bash
#SBATCH --job-name=hybrid-setup
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"

PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
RESULT_ROOT="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_cyclone_v1}"
MODEL_DIR="$RESULT_ROOT/model"

required_paths=(
    "$PROJECT_DIR/src/networks/build_matpower_case.py"
    "$PROJECT_DIR/src/networks/build_hybrid_power_model.py"
    "$PROJECT_DIR/network_save/simplified_hospital_MG.pickle"
    "$PROJECT_DIR/network_save/MG_hospital.pickle"
    "$PROJECT_DIR/data/matpower_sources/generator_capacity.csv"
    "$PROJECT_DIR/data/matpower_sources/sources.json"
    "$ENV_ROOT/lib/python3.9/site-packages"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required hybrid setup input is missing: $required_path" >&2
        exit 1
    fi
done

module load python/3.9.0
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p "$MODEL_DIR"
python3 src/networks/build_matpower_case.py \
    --output-dir "$MODEL_DIR"
python3 src/networks/build_hybrid_power_model.py \
    --bus-mapping "$MODEL_DIR/bus_mapping.csv" \
    --branch-mapping "$MODEL_DIR/branch_mapping.csv" \
    --generator-mapping "$MODEL_DIR/generator_mapping.csv" \
    --hospital-mapping "$MODEL_DIR/hospital_bus_mapping.csv" \
    --output-dir "$MODEL_DIR"
python3 - "$MODEL_DIR/hybrid_power_model.npz" <<'PY'
import sys
from pathlib import Path
from hybrid_power_model import HybridPowerModel

model = HybridPowerModel(Path(sys.argv[1]))
service = model.intact_service_fraction()
if service < 1.0 - 1e-7:
    raise SystemExit(f"intact hybrid model is not feasible: service={service}")
print(
    f"hybrid model verified: nodes={len(model.node_ids)} "
    f"edges={len(model.edge_uid)} hospitals={model.modeled_hospital_count} "
    f"intact_service={service:.12f}"
)
PY
