#!/bin/bash
#SBATCH --job-name=t24-supplement-summary
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Summaries of the line-hazard v2 supplementary analyses (see run_t24_supplement_chunk.sbatch).
# Usage: sbatch run_t24_supplement_summary.sbatch MODE      (decomp | sensitivity | dc)
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
MODE="${1:?MODE}"
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
export PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$ENV_ROOT/lib/python3.9/site-packages"
cd "$CODE"

case "$MODE" in
decomp)
    R="$ROOT/all_grid_reallocated"
    python3 src/analysis/summarize_hybrid_cyclones.py --result-root "$R" --samples 100 --hospitals SARA_processed \
        --hospital-scenario all_grid --summary-output "$R/hybrid_all_grid_cyclone_summary.csv" \
        --hospital-output "$R/df_final_hybrid_all_grid_reallocated_t24.csv" --report-output "$R/validation_report.json"
    # Frozen-demand run: existing facilities equal the baseline; damage draws equal the reallocated run.
    python3 "$CODE/src/analysis/audit_all_grid_frozen_demand.py" --frozen-root "$ROOT/all_grid" --baseline-root "$ROOT/public" \
        --all-grid-root "$R" --output-dir "$ROOT/all_grid/audit_frozen_vs_reallocated"
    module load json-glib/1.4.4
    export MPLBACKEND=Agg
    python3 src/figures/capacity_ablation_figure.py --public-summary "$ROOT/public/df_final_hybrid_t24.csv" \
        --all-grid-summary "$R/df_final_hybrid_all_grid_reallocated_t24.csv" \
        --solar-root "$SCRATCH/wind_result/synthetic_solar" --hospital-data SARA_processed \
        --boundary "$SCRATCH/World_Countries" --output-prefix "$R/figure_06_hybrid_all_grid_reallocated" ;;
sensitivity)
    python3 "$CODE/src/analysis/hybrid_sensitivity_v2.py" summarize --scenario-dir "$SCRATCH/hybrid_sensitivity_v1" \
        --chunk-dir "$ROOT/sensitivity/chunks" --output-dir "$ROOT/sensitivity/summary" \
        --published "$ROOT/public/df_final_hybrid_t24.csv" ;;
dc)
    python3 "$CODE/src/analysis/audit_dc_balance_gap_v2.py" summarize --chunk-dir "$ROOT/dc_audit/chunks" \
        --output-dir "$ROOT/dc_audit/summary" ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "summary $MODE complete"
