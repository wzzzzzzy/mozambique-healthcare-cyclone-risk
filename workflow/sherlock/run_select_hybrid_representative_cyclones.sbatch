#!/bin/bash
#SBATCH --job-name=select-hybrid-cyclones
#SBATCH --output=/scratch/users/zywei/infracascade_logs/figures/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/figures/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
: "${L_SCRATCH_JOB:?L_SCRATCH_JOB is not set}"

PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:-$HOME/code/InfraCascade}"
ENV_ROOT="${INFRACASCADE_ENV_ROOT:-/home/groups/jsuckale/zywei/envs/envs_osm}"
PUBLIC_SUMMARY="${INFRACASCADE_PUBLIC_SUMMARY:-$SCRATCH/hybrid_cyclone_v1/df_final_hybrid_25.csv}"
HOSPITAL_DATA="${INFRACASCADE_HOSPITAL_DATA:-$PROJECT_DIR/SARA_processed}"
BOUNDARY_DATA="${INFRACASCADE_BOUNDARY_DATA:-$SCRATCH/World_Countries}"
WIND_ARCHIVE="${INFRACASCADE_CATALOGUE0_WIND_ARCHIVE:-$SCRATCH/wind_result/catalogue0_wind_track.tar.gz}"
OUTPUT_DIR="${1:-$SCRATCH/hybrid_cyclone_v1/figures/representative_cyclones_hybrid}"
SELECTED_WIND_ROOT="$OUTPUT_DIR/selected_wind_footprints"
LOCAL_INPUT="$L_SCRATCH_JOB/representative_cyclone_input"

required_paths=(
    "$ENV_ROOT/lib/python3.9/site-packages"
    "$PUBLIC_SUMMARY"
    "$HOSPITAL_DATA"
    "$BOUNDARY_DATA"
    "$WIND_ARCHIVE"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required representative-case input is missing: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$SCRATCH/infracascade_logs/figures" "$OUTPUT_DIR" "$LOCAL_INPUT"
tar -xzf "$WIND_ARCHIVE" -C "$LOCAL_INPUT"
WIND_ROOT="$LOCAL_INPUT/wind_track"
if [[ ! -d "$WIND_ROOT" ]]; then
    echo "Wind archive did not contain wind_track/: $WIND_ARCHIVE" >&2
    exit 1
fi

module purge
module load viz
module load python/3.9.0
module load py-scikit-learn/1.0.2_py39
module load json-glib/1.4.4
export PYTHONPATH="$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

cd "$PROJECT_DIR"
python3 src/analysis/select_hybrid_representative_cyclones.py \
    --public-summary "$PUBLIC_SUMMARY" \
    --hospital-data "$HOSPITAL_DATA" \
    --boundary "$BOUNDARY_DATA" \
    --wind-root "$WIND_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --selected-wind-root "$SELECTED_WIND_ROOT"

TRACKS="$(<"$OUTPUT_DIR/representative_track_keys.txt")"
python3 src/figures/hybrid_representative_cyclones_figure.py \
    --public-summary "$PUBLIC_SUMMARY" \
    --hospital-data "$HOSPITAL_DATA" \
    --boundary "$BOUNDARY_DATA" \
    --wind-root "$SELECTED_WIND_ROOT" \
    --tracks "$TRACKS" \
    --output-dir "$OUTPUT_DIR"

artifacts=(
    "$OUTPUT_DIR/representative_case_candidates.csv"
    "$OUTPUT_DIR/representative_case_selection.csv"
    "$OUTPUT_DIR/representative_case_selection.json"
    "$OUTPUT_DIR/representative_track_keys.txt"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid.pdf"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid.png"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid_data.csv"
    "$OUTPUT_DIR/figure_representative_cyclones_hybrid_metadata.json"
)
for artifact in "${artifacts[@]}"; do
    if [[ ! -s "$artifact" ]]; then
        echo "Expected representative-case artifact was not generated: $artifact" >&2
        exit 1
    fi
done
ls -lh "${artifacts[@]}"
