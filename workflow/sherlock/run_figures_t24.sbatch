#!/bin/bash
#SBATCH --job-name=figures-t24
#SBATCH --output=/scratch/users/zywei/infracascade_logs/figures/%x-%j.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/figures/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Manuscript figures from the line-hazard v2 rerun (24 m/s, L0 0.25 km,
# towers + conductor).  Same figure code and non-hazard inputs as before; only
# the electricity / water / communication / all-grid / Figure 5
# results change.  Figure loaders expect *_25 file names, provided as links
# under $ROOT/compat.
# Usage: sbatch run_figures_t24.sbatch OUTPUT_DIR
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
OUT="${1:?OUTPUT_DIR}"
[[ -e "$OUT" ]] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
WIND_ROOT="$SCRATCH/wind_result"
SOLAR_ROOT="$WIND_ROOT/synthetic_solar"
COMPAT="$ROOT/compat"

# File-name links for the figure loaders (created once).
mkdir -p "$COMPAT/public" "$COMPAT/all_grid"
[[ -e "$COMPAT/public/df_final_hybrid_25.csv" ]] || ln -s "$ROOT/public/df_final_hybrid_t24.csv" "$COMPAT/public/df_final_hybrid_25.csv"
[[ -e "$COMPAT/all_grid/df_final_hybrid_all_grid_25.csv" ]] || \
    ln -s "$ROOT/all_grid/df_final_hybrid_all_grid_t24.csv" "$COMPAT/all_grid/df_final_hybrid_all_grid_25.csv"
PUBLIC_SUMMARY="$COMPAT/public/df_final_hybrid_25.csv"
MULTISYSTEM="$ROOT/multisystem/compat_25"

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
echo "figure code $(git rev-parse --short HEAD) + local edits"; git diff --stat | tail -1

notebook() {  # notebook output_dir
    python3 -m jupyter nbconvert --to notebook --execute "moz_notebook/$1.ipynb" \
        --ExecutePreprocessor.timeout=3600 --output "${1}_executed.ipynb" --output-dir "$2/executed_notebooks"
}

# Figure 2 (electricity sources).  The historical-versus-synthetic figure (7)
# built in the original run belongs to the separate GRL study and is omitted.
D="$OUT/figures_2_7"; mkdir -p "$D/prepared" "$D/executed_notebooks"
export INFRACASCADE_WIND_ROOT="$WIND_ROOT" INFRACASCADE_PUBLIC_SUMMARY="$PUBLIC_SUMMARY"
export INFRACASCADE_ALL_GRID_SUMMARY="$WIND_ROOT/df_final_all_grid_25.csv" INFRACASCADE_SOLAR_ROOT="$SOLAR_ROOT"
export INFRACASCADE_HOSPITAL_DATA="$CODE/SARA_processed" INFRACASCADE_BOUNDARY_DATA="$SCRATCH/World_Countries"
export MOZ_MANUSCRIPT_FIGURE_DIR="$D" MOZ_FIGURE_DATA_ROOT="$D/prepared"
notebook figure_02_electricity_sources "$D"

# Figure 3 (cascading service failures).
D="$OUT/figure_03"; mkdir -p "$D/prepared" "$D/executed_notebooks"
export MOZ_MANUSCRIPT_FIGURE_DIR="$D" MOZ_FIGURE_DATA_ROOT="$D/prepared" INFRACASCADE_PROJECT_DIR="$CODE"
python3 src/figures/prepare_failure_figure_inputs.py --public-summary "$PUBLIC_SUMMARY" --water-root "$MULTISYSTEM/water" \
    --communication-root "$MULTISYSTEM/comm" --direct-root "$WIND_ROOT/notebook_inputs/synthetic_new_hospital" \
    --hospital-data "$CODE/SARA_processed" --storm-root "$SCRATCH/cyclone_map/STORM" --output-dir "$D/prepared"
notebook figure_03_cascading_service_failures "$D"

# Figure S1 (direct versus indirect; regional direct-damage function).
python3 "$CODE/src/figures/figure_S1_regional_direct.py" --public-summary "$PUBLIC_SUMMARY" \
    --water-root "$MULTISYSTEM/water" --communication-root "$MULTISYSTEM/comm" \
    --direct-root "$WIND_ROOT/notebook_inputs/synthetic_new_hospital" \
    --regional-ratio "$SCRATCH/cyclone_direct_damage_v1/results/damage_ratio_si_tdr.csv.gz" \
    --hospital-data "$CODE/SARA_processed" --storm-root "$SCRATCH/cyclone_map/STORM" --output-dir "$OUT/figure_S1"

# Figure 4 (frozen-demand all-grid counterfactual).
mkdir -p "$OUT/figure_04_all_grid"
python3 src/figures/capacity_ablation_figure.py --public-summary "$PUBLIC_SUMMARY" \
    --all-grid-summary "$COMPAT/all_grid/df_final_hybrid_all_grid_25.csv" --solar-root "$SOLAR_ROOT" \
    --hospital-data SARA_processed --boundary "$SCRATCH/World_Countries" \
    --output-prefix "$OUT/figure_04_all_grid/figure_06_hybrid_all_grid_frozen_demand"

# Figure 5 (grid cascade by province; legacy-compatible lengths).
D="$OUT/figure_05"; mkdir -p "$D/prepared" "$D/executed_notebooks"
export MOZ_MANUSCRIPT_FIGURE_DIR="$D" MOZ_FIGURE_DATA_ROOT="$D/prepared"
export INFRACASCADE_CASCADE_ADMIN_SUMMARY="$D/cascade_admin_summary.gpkg"
export INFRACASCADE_POWER_GRAPH="$CODE/network_save/simplified_hospital_MG.pickle"
python3 src/figures/prepare_hybrid_cascade_admin_summary.py --admin-boundaries "$WIND_ROOT/notebook_inputs/afr_g2014_2013_1" \
    --storm-root "$SCRATCH/cyclone_map/STORM" --hybrid-track-summary "$ROOT/figure5/hybrid_multisystem_track_summary.csv" \
    --hospital-data "$CODE/SARA_processed" --public-summary "$PUBLIC_SUMMARY" \
    --length-convention legacy_compatible --output "$INFRACASCADE_CASCADE_ADMIN_SUMMARY"
notebook figure_05_grid_cascade "$D"

# Figure S2 (city electricity).
mkdir -p "$OUT/figure_S2"
python3 src/figures/hybrid_city_electricity_figure.py --public-summary "$PUBLIC_SUMMARY" --solar-root "$SOLAR_ROOT" \
    --hospital-data "$CODE/SARA_processed" \
    --admin3 "$WIND_ROOT/notebook_inputs/moz_admin_boundaries.shp/moz_admin3.shp" --output-dir "$OUT/figure_S2"

ls -R "$OUT" | grep -E "\.(png|pdf)$"
echo "figures complete"
