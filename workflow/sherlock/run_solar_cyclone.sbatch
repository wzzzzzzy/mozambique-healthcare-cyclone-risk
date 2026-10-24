#!/bin/bash
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=128GB
 
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:?set to the repository root}"
cd "$PROJECT_DIR"
ml python/3.9.0
export PYTHONPATH=$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:$HOME/envs/envs_osm/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/envs/envs_osm/lib/
ml load json-glib/1.4.4
ml load py-scikit-learn/1.0.2_py39
ml load matlab/R2022b
ml load gcc/9.1.0

python3 src/hazard/simulation_on_solar.py   --folder "${1}"

# python3 simulation_on_water_graph.py --flood_or_cyclone cyclone --threshold 25 --folder 
# python3 analyze_compare_flood_map.py --gdf_fn "${1}" 