#!/bin/bash
#SBATCH --job-name=solar-exposure
#SBATCH --output=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/users/zywei/infracascade_logs/hybrid/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc,normal

# Solar damage with exposure-dependent gust factors (solar_exposure_gust.py).
# Usage: sbatch --array=0-27 run_solar_exposure_gust.sbatch winds 100
#        sbatch run_solar_exposure_gust.sbatch convert [COASTAL_KM]
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set}"
MODE="${1:?MODE}"
ROOT="${LINE_HAZARD_ROOT:?}"
CODE="${INFRACASCADE_PROJECT_DIR:?}"
ENV_ROOT="/home/groups/jsuckale/zywei/envs/envs_osm"
OUT="$ROOT/solar_exposure"
module purge
module load python/3.9.0
module load json-glib/1.4.4
module load py-scikit-learn/1.0.2_py39
module load matlab/R2022b
module load gcc/9.1.0
export PYTHONPATH="$CODE/src/common:$CODE/src/networks:$CODE/src/hazard:$CODE/src/models:$CODE/src/analysis:$CODE/src/figures:$ENV_ROOT/lib/python3.9/site-packages"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ENV_ROOT/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
G="$CODE/src/hazard/solar_exposure_gust.py"
case "$MODE" in
winds)
    N="${2:?TRACKS_PER_TASK}"; START=$((SLURM_ARRAY_TASK_ID * N))
    OUTPUT="$OUT/winds/solar_winds_$(printf '%04d' "$START").npz"
    [[ -s "$OUTPUT" ]] || python3 "$G" winds --hospitals SARA_processed --storm-root "$SCRATCH/cyclone_map/STORM" \
        --boundary "$SCRATCH/World_Countries" --chavas-dir "$CODE/ER11E04" \
        --global-start "$START" --global-stop "$((START + N))" --output "$OUTPUT" ;;
convert)
    KM="${2:-5}"
    python3 "$G" convert --hospitals SARA_processed --boundary "$SCRATCH/World_Countries" \
        --winds-dir "$OUT/winds" --source-root "$SCRATCH/wind_result/synthetic_solar" \
        --coastal-km "$KM" --inflow-deg 20 --output-root "$OUT/synthetic_solar_coastal${KM}km" ;;
esac
