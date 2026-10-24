#!/bin/bash
#SBATCH --job-name=figures-solar-exp
#SBATCH --output=/scratch/users/zywei/infracascade_logs/figures/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/figures/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Solar-dependent manuscript figures (Figure 2, Figure 4, Figure S2) with the
# exposure-dependent solar gust factors (solar_exposure_gust.py: 1.66 inland,
# 1.38 / 1.52 for onshore / offshore flow within the coastal band).  Everything
# else is the line-hazard v2 main case, as in run_figures_t24.sbatch.
# Usage: sbatch run_figures_t24_solar_exposure.sbatch SOLAR_ROOT OUTPUT_DIR
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
SOLAR_ROOT="${1:?SOLAR_ROOT}"
OUT="${2:?OUTPUT_DIR}"
[[ -e "$OUT" ]] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
WIND_ROOT="$SCRATCH/wind_result"
COMPAT="$ROOT/compat"
PUBLIC_SUMMARY="$COMPAT/public/df_final_hybrid_25.csv"
ALL_GRID_SUMMARY="$COMPAT/all_grid/df_final_hybrid_all_grid_25.csv"
for f in "$SOLAR_ROOT/solar_exposure_report.json" "$PUBLIC_SUMMARY" "$ALL_GRID_SUMMARY"; do
    [[ -e "$f" ]] || { echo "Missing $f" >&2; exit 1; }
done

mkdir -p "$OUT"
cd "$CODE"
module purge
module load viz
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
module load py-seaborn/0.12.1_py39
module load py-jupyter/1.0.0_py39
export MPLBACKEND=Agg
export PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$ENV_ROOT/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
echo "figure code $(git rev-parse --short HEAD) + local edits; solar root $SOLAR_ROOT"

# Figure 2 (electricity sources).
D="$OUT/figure_02"; mkdir -p "$D/executed_notebooks"
export INFRACASCADE_WIND_ROOT="$WIND_ROOT" INFRACASCADE_PUBLIC_SUMMARY="$PUBLIC_SUMMARY"
export INFRACASCADE_ALL_GRID_SUMMARY="$WIND_ROOT/df_final_all_grid_25.csv" INFRACASCADE_SOLAR_ROOT="$SOLAR_ROOT"
export INFRACASCADE_HOSPITAL_DATA="$CODE/SARA_processed" INFRACASCADE_BOUNDARY_DATA="$SCRATCH/World_Countries"
export MOZ_MANUSCRIPT_FIGURE_DIR="$D" MOZ_FIGURE_DATA_ROOT="$D" INFRACASCADE_PROJECT_DIR="$CODE"
python3 -m jupyter nbconvert --to notebook --execute moz_notebook/figure_02_electricity_sources.ipynb \
    --ExecutePreprocessor.timeout=3600 --output figure_02_electricity_sources_executed.ipynb --output-dir "$D/executed_notebooks"

# Figure 4 (frozen-demand all-grid counterfactual).
mkdir -p "$OUT/figure_04_all_grid"
python3 src/figures/capacity_ablation_figure.py --public-summary "$PUBLIC_SUMMARY" \
    --all-grid-summary "$ALL_GRID_SUMMARY" --solar-root "$SOLAR_ROOT" \
    --hospital-data SARA_processed --boundary "$SCRATCH/World_Countries" \
    --output-prefix "$OUT/figure_04_all_grid/figure_06_hybrid_all_grid_frozen_demand"

# Figure S2 (city electricity).
mkdir -p "$OUT/figure_S2"
python3 src/figures/hybrid_city_electricity_figure.py --public-summary "$PUBLIC_SUMMARY" --solar-root "$SOLAR_ROOT" \
    --hospital-data "$CODE/SARA_processed" \
    --admin3 "$WIND_ROOT/notebook_inputs/moz_admin_boundaries.shp/moz_admin3.shp" --output-dir "$OUT/figure_S2"

ls -R "$OUT" | grep -E "\.(png|pdf)$"
echo "figures complete"
