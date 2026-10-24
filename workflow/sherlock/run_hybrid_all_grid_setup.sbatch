#!/bin/bash
#SBATCH --job-name=hybrid-allgrid-setup
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid_all_grid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid_all_grid/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"

PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
PUBLIC_ROOT="${HYBRID_PUBLIC_ROOT:-$SCRATCH/hybrid_cyclone_v1}"
RESULT_ROOT="${HYBRID_CYCLONE_ROOT:-$SCRATCH/hybrid_all_grid_v1}"
MODEL_DIR="$RESULT_ROOT/model"

required_paths=(
    "$PROJECT_DIR/src/networks/prepare_hybrid_hospital_scenario.py"
    "$PROJECT_DIR/src/networks/build_matpower_case.py"
    "$PROJECT_DIR/src/networks/build_hybrid_power_model.py"
    "$PROJECT_DIR/network_save/simplified_hospital_MG.pickle"
    "$PROJECT_DIR/network_save/MG_hospital.pickle"
    "$PROJECT_DIR/SARA_processed"
    "$PROJECT_DIR/data/matpower_sources/generator_capacity.csv"
    "$PROJECT_DIR/data/matpower_sources/sources.json"
    "$PUBLIC_ROOT/model/branch_mapping.csv"
    "$PUBLIC_ROOT/model/bus_mapping.csv"
    "$PUBLIC_ROOT/model/generator_mapping.csv"
    "$ENV_ROOT/lib/python3.9/site-packages"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required all-grid setup input is missing: $required_path" >&2
        exit 1
    fi
done

module load python/3.9.0
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p "$MODEL_DIR" "$SCRATCH/infracascade_logs/hybrid_all_grid"
python3 src/networks/prepare_hybrid_hospital_scenario.py \
    --scenario all_grid \
    --active-bus-mapping "$PUBLIC_ROOT/model/bus_mapping.csv" \
    --output-dir "$MODEL_DIR"
python3 src/networks/build_matpower_case.py \
    --network "$MODEL_DIR/simplified_hospital_scenario.pickle" \
    --output-dir "$MODEL_DIR" \
    --case-name case_mozambique_all_grid
python3 src/networks/build_hybrid_power_model.py \
    --network "$MODEL_DIR/simplified_hospital_scenario.pickle" \
    --reference-network "$MODEL_DIR/hospital_scenario_reference.pickle" \
    --bus-mapping "$MODEL_DIR/bus_mapping.csv" \
    --branch-mapping "$MODEL_DIR/branch_mapping.csv" \
    --generator-mapping "$MODEL_DIR/generator_mapping.csv" \
    --hospital-mapping "$MODEL_DIR/hospital_bus_mapping.csv" \
    --output-dir "$MODEL_DIR"
python3 - "$MODEL_DIR/hybrid_power_model.npz" \
    "$PUBLIC_ROOT/model/branch_mapping.csv" \
    "$MODEL_DIR/branch_mapping.csv" \
    "$PUBLIC_ROOT/model/generator_mapping.csv" \
    "$MODEL_DIR/generator_mapping.csv" <<'PY'
import csv
import sys
from pathlib import Path

from hybrid_power_model import HybridPowerModel


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


model = HybridPowerModel(Path(sys.argv[1]))
if len(model.hospital_ids) != 1380:
    raise SystemExit(f"expected 1380 all-grid hospitals, found {len(model.hospital_ids)}")
service = model.intact_service_fraction()
if service < 1.0 - 1e-7:
    raise SystemExit(f"intact all-grid hybrid model is not feasible: service={service}")

public_branches = rows(sys.argv[2])
all_grid_branches = rows(sys.argv[3])
branch_fields = ("from_node", "to_node", "edge_key", "x_pu", "rate_a_mw", "status")
if [tuple(row[field] for field in branch_fields) for row in public_branches] != [
    tuple(row[field] for field in branch_fields) for row in all_grid_branches
]:
    raise SystemExit("all-grid counterfactual changed the public-grid branches")

public_generators = rows(sys.argv[4])
all_grid_generators = rows(sys.argv[5])
generator_fields = ("node_id", "pmax_mw")
if [tuple(row[field] for field in generator_fields) for row in public_generators] != [
    tuple(row[field] for field in generator_fields) for row in all_grid_generators
]:
    raise SystemExit("all-grid counterfactual changed generator capacities")

print(
    f"all-grid hybrid model verified: nodes={len(model.node_ids)} "
    f"edges={len(model.edge_uid)} hospitals={model.modeled_hospital_count}/1380 "
    f"intact_service={service:.12f}"
)
PY
