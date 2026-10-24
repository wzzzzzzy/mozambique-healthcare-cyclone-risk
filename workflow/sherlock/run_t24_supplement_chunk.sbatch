#!/bin/bash
#SBATCH --job-name=t24-supplement
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Supplementary analyses on the line-hazard v2 probabilities (24 m/s, L0 0.25 km,
# towers + conductor), synthetic tracks [task * N, (task + 1) * N):
#   decomp       original (demand-reallocated) all-grid model, for the decomposition
#   sensitivity  one-at-a-time scenarios (hybrid_sensitivity_v1 scenario models)
#   dc           DC versus power-balance island audit
# Usage: sbatch --array=0-M run_t24_supplement_chunk.sbatch MODE TRACKS_PER_TASK
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
MODE="${1:?MODE}"
N="${2:?TRACKS_PER_TASK}"
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
START=$((SLURM_ARRAY_TASK_ID * N)); STOP=$((START + N)); (( STOP <= 2703 )) || STOP=2703
TAG="$(printf '%04d' "$START")_$(printf '%04d' "$STOP")"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$ENV_ROOT/lib/python3.9/site-packages"
cd "$CODE"

case "$MODE" in
decomp)
    MODEL="$SCRATCH/hybrid_all_grid_final/model/hybrid_power_model.npz"
    TRACK_COUNTS=(290 274 263 276 273 279 251 264 242 291)
    for ((global=START; global<STOP; global++)); do
        offset=0
        for set in "${!TRACK_COUNTS[@]}"; do
            if (( global < offset + TRACK_COUNTS[set] )); then storm_set=$set; track=$((global - offset)); break; fi
            offset=$((offset + TRACK_COUNTS[set]))
        done
        name="dataset_${storm_set}/track_$(printf '%04d' "$track")"
        prefix="$ROOT/all_grid_reallocated/results/$name"
        [[ -s "${prefix}_summary.json" && -s "${prefix}_hospitals.csv" ]] && continue
        mkdir -p "$(dirname "$prefix")"
        python3 src/models/hybrid_power_model.py --model "$MODEL" --probabilities "$ROOT/probabilities/all_grid/${name}.csv" \
            --samples 100 --seed 1234 --storm-folder "STORM_DATA_IBTRACS_SI_1000_YEARS_${storm_set}" --track-id "$track" \
            --sample-output "${prefix}_samples.csv" --hospital-output "${prefix}_hospitals.csv" \
            --summary-output "${prefix}_summary.json" > /dev/null
    done ;;
sensitivity)
    OUT="$ROOT/sensitivity/chunks/tracks_${TAG}.npz"; mkdir -p "$(dirname "$OUT")"
    [[ -s "$OUT" ]] || python3 "$CODE/src/analysis/hybrid_sensitivity_v2.py" tracks --scenario-dir "$SCRATCH/hybrid_sensitivity_v1" \
        --probability-dir "$ROOT/probabilities/public" --global-start "$START" --global-stop "$STOP" \
        --samples 100 --seed 1234 --output "$OUT" ;;
dc)
    OUT="$ROOT/dc_audit/chunks/islands_${TAG}.csv.gz"; mkdir -p "$(dirname "$OUT")"
    [[ -s "$OUT" ]] || python3 "$CODE/src/analysis/audit_dc_balance_gap_v2.py" tracks \
        --model "$SCRATCH/hybrid_cyclone_v1/model/hybrid_power_model.npz" --probability-dir "$ROOT/probabilities/public" \
        --global-start "$START" --global-stop "$STOP" --samples 100 --seed 1234 --output "$OUT" ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "done $MODE $TAG"
