#!/bin/bash
#SBATCH --job-name=hybrid-five-draw-cascade
#SBATCH --output=/scratch/users/zywei/infracascade_logs/figures/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/figures/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${L_SCRATCH:?L_SCRATCH is not set}"

PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
PUBLIC_SUMMARY="${INFRACASCADE_PUBLIC_SUMMARY:-$SCRATCH/hybrid_cyclone_v1/df_final_hybrid_25.csv}"
HOSPITAL_DATA="${INFRACASCADE_HOSPITAL_DATA:-$PROJECT_DIR/SARA_processed}"
BOUNDARY_DATA="${INFRACASCADE_BOUNDARY_DATA:-$SCRATCH/World_Countries}"
STORM_ROOT="${INFRACASCADE_STORM_ROOT:-$SCRATCH/cyclone_map/STORM}"
OUTPUT_DIR="${1:-$SCRATCH/hybrid_cyclone_v1/figures/representative_cyclones_hybrid}"
TRACK_FILE="$OUTPUT_DIR/representative_track_keys.txt"

required_paths=(
    "$PROJECT_DIR/src/figures/draw_cascade.py"
    "$PROJECT_DIR/ER11E04"
    "$PROJECT_DIR/matlab/generate_cyclone_footprints.m"
    "$ENV_ROOT/lib/python3.9/site-packages"
    "$PUBLIC_SUMMARY"
    "$HOSPITAL_DATA"
    "$BOUNDARY_DATA"
    "$STORM_ROOT/all_tracks"
    "$STORM_ROOT/Moz_cyclone_track"
    "$TRACK_FILE"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required draw_cascade-style figure input is missing: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$SCRATCH/infracascade_logs/figures" "$OUTPUT_DIR"

module purge
module load python/3.9.0
module load matlab/R2022b
module load gcc/9.1.0
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ENV_ROOT/lib"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$L_SCRATCH/matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$L_SCRATCH/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

TRACKS="$(<"$TRACK_FILE")"
cd "$PROJECT_DIR"
python3 src/figures/hybrid_representative_cyclones_figure.py \
    --public-summary "$PUBLIC_SUMMARY" \
    --hospital-data "$HOSPITAL_DATA" \
    --boundary "$BOUNDARY_DATA" \
    --wind-style draw-cascade \
    --storm-root "$STORM_ROOT" \
    --er11-dir "$PROJECT_DIR/ER11E04" \
    --matlab-helper-dir "$PROJECT_DIR/matlab" \
    --tracks "$TRACKS" \
    --output-dir "$OUTPUT_DIR"

artifacts=(
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid.pdf"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid.png"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid_data.csv"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid_metadata.json"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid_wind_footprints.csv"
)
for artifact in "${artifacts[@]}"; do
    if [[ ! -s "$artifact" ]]; then
        echo "Expected draw_cascade-style artifact was not generated: $artifact" >&2
        exit 1
    fi
done
ls -lh "${artifacts[@]}"
