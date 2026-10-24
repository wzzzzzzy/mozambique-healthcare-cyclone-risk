#!/bin/bash
#SBATCH --job-name=hybrid-t24-summary
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Stage 3 of the line-hazard v2 rerun: catalogue summaries in the layouts used
# by the figure code.  The water/communication files carry the hazard label 24;
# compat_25/ links them under the *_25.csv names the figure loaders expect.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
SAMPLES="${1:-100}"
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
MS_CODE="$CODE"  # paper repo: one code version for all stages
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
SITE="$ENV_ROOT/lib/python3.9/site-packages"
DEPENDENCY_MODEL="$SCRATCH/hybrid_multisystem_water_new6_v1/model/water_comm_dependencies.pickle"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39

cd "$CODE"
export PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$SITE"
python3 src/analysis/summarize_hybrid_cyclones.py --result-root "$ROOT/public" --samples "$SAMPLES" --hospitals SARA_processed \
    --summary-output "$ROOT/public/hybrid_cyclone_summary.csv" \
    --hospital-output "$ROOT/public/df_final_hybrid_t24.csv" \
    --report-output "$ROOT/public/validation_report.json"
python3 src/analysis/summarize_hybrid_cyclones.py --result-root "$ROOT/all_grid" --samples "$SAMPLES" --hospitals SARA_processed \
    --hospital-scenario all_grid \
    --summary-output "$ROOT/all_grid/hybrid_all_grid_cyclone_summary.csv" \
    --hospital-output "$ROOT/all_grid/df_final_hybrid_all_grid_t24.csv" \
    --report-output "$ROOT/all_grid/validation_report.json"
python3 src/models/hybrid_figure5_metrics.py summarize --manifest "$SCRATCH/hybrid_figure5_length_v2/category_1_2_tracks.csv" \
    --result-root "$ROOT/figure5" --samples "$SAMPLES" --output "$ROOT/figure5/hybrid_multisystem_track_summary.csv"

cd "$MS_CODE"
PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$SITE" python3 src/analysis/summarize_hybrid_multisystem.py synthetic --result-root "$ROOT/multisystem" \
    --dependency-model "$DEPENDENCY_MODEL" --samples "$SAMPLES" --threshold 24
for system in water comm; do
    mkdir -p "$ROOT/multisystem/compat_25/$system"
    for f in "$ROOT/multisystem/$system"/${system}_cyclone_all_*_24.csv; do
        ln -s "$f" "$ROOT/multisystem/compat_25/$system/$(basename "${f%_24.csv}")_25.csv"
    done
done
echo "summaries complete"
