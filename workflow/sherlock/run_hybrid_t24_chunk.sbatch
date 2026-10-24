#!/bin/bash
#SBATCH --job-name=hybrid-t24
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Stage 2 of the line-hazard v2 rerun, per synthetic track, from the stage-1
# probabilities (24 m/s, L0 0.25 km, towers + conductor):
#   public    hybrid electricity (hybrid_cyclone_v1 model)
#   ms        water and communication (new6 dependency model)
#   all_grid  frozen-demand all-grid counterfactual
#   figure5   province metrics for category 1-2 tracks
# Same models, seed and samples as the published runs.  Resumable.
# Usage: sbatch --array=0-N run_hybrid_t24_chunk.sbatch TRACKS_PER_TASK SAMPLES
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TRACKS_PER_TASK="${1:?TRACKS_PER_TASK}"
SAMPLES="${2:-100}"
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"                 # hybrid model, figure-5 metrics
MS_CODE="$CODE"  # paper repo: one code version for all stages
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
PUBLIC_MODEL="$SCRATCH/hybrid_cyclone_v1/model/hybrid_power_model.npz"
ALL_GRID_MODEL="$SCRATCH/hybrid_all_grid_frozen_demand_v1/model/hybrid_power_model.npz"
DEPENDENCY_MODEL="$SCRATCH/hybrid_multisystem_water_new6_v1/model/water_comm_dependencies.pickle"
FIGURE5_ROOT="$SCRATCH/hybrid_figure5_length_v2"
TRACK_COUNTS=(290 274 263 276 273 279 251 264 242 291)
FIRST=$((SLURM_ARRAY_TASK_ID * TRACKS_PER_TASK))
LAST=$((FIRST + TRACKS_PER_TASK - 1)); (( LAST < 2703 )) || LAST=2702

module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SITE="$ENV_ROOT/lib/python3.9/site-packages"

hybrid() {  # model probabilities prefix
    [[ -s "${3}_summary.json" && -s "${3}_hospitals.csv" ]] && return 0
    mkdir -p "$(dirname "$3")"
    (cd "$CODE" && PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$SITE" python3 src/models/hybrid_power_model.py --model "$1" --probabilities "$2" \
        --samples "$SAMPLES" --seed 1234 --storm-folder "$folder" --track-id "$track" \
        --sample-output "${3}_samples.csv" --hospital-output "${3}_hospitals.csv" \
        --summary-output "${3}_summary.json" > /dev/null)
}

for ((global=FIRST; global<=LAST; global++)); do
    offset=0
    for set in "${!TRACK_COUNTS[@]}"; do
        if (( global < offset + TRACK_COUNTS[set] )); then storm_set=$set; track=$((global - offset)); break; fi
        offset=$((offset + TRACK_COUNTS[set]))
    done
    folder="STORM_DATA_IBTRACS_SI_1000_YEARS_${storm_set}"
    name="dataset_${storm_set}/track_$(printf '%04d' "$track")"
    public_p="$ROOT/probabilities/public/${name}.csv"
    all_grid_p="$ROOT/probabilities/all_grid/${name}.csv"
    [[ -s "$public_p" && -s "$all_grid_p" ]] || { echo "missing stage-1 probabilities for $name" >&2; exit 1; }

    hybrid "$PUBLIC_MODEL" "$public_p" "$ROOT/public/results/$name"
    hybrid "$ALL_GRID_MODEL" "$all_grid_p" "$ROOT/all_grid/results/$name"

    if [[ ! -s "$ROOT/multisystem/summaries/summary_${folder}_${track}_24.json" ]]; then
        (cd "$MS_CODE" && PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$SITE" python3 src/models/hybrid_multisystem_model.py --model "$PUBLIC_MODEL" \
            --probabilities "$public_p" --dependency-model "$DEPENDENCY_MODEL" --output-root "$ROOT/multisystem" \
            --storm-folder "$folder" --track-id "$track" --threshold 24 --samples "$SAMPLES" --seed 1234 > /dev/null)
    fi

    if awk -F, -v id="$global" 'NR > 1 && $1 == id {found=1} END {exit !found}' "$FIGURE5_ROOT/category_1_2_tracks.csv"; then
        figure5_out="$ROOT/figure5/summaries/summary_${folder}_${track}_25.json"  # file name expected by the summarizer
        if [[ ! -s "$figure5_out" ]]; then
            mkdir -p "$(dirname "$figure5_out")"
            (cd "$CODE" && PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$SITE" python3 src/models/hybrid_figure5_metrics.py track --model "$PUBLIC_MODEL" \
                --length-cache "$FIGURE5_ROOT/real_edge_lengths.npz" --probabilities "$public_p" \
                --storm-set "$storm_set" --storm-folder "$folder" --track-id "$track" \
                --samples "$SAMPLES" --seed 1234 --output "$figure5_out" > /dev/null)
        fi
    fi
    echo "done $name"
done
